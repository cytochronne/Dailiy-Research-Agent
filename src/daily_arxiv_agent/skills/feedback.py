"""Feedback recording and explainable recommendation refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ValidationError, model_validator

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    EmbeddingCacheMetadata,
    EmbeddingCacheScope,
    EmbeddingInputRole,
    EmbeddingProviderCacheMetadata,
    EvidenceSource,
    FeedbackEvent,
    FeedbackInfluenceRecord,
    FeedbackRefinementStatus,
    FeedbackValue,
    PaperMetadata,
    RankingScoreBreakdown,
    Recommendation,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
    EmbeddingProviderError,
    normalize_embedding_text,
    normalize_provider_input_text,
)
from daily_arxiv_agent.embeddings.provider import create_embedding_provider
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
    build_paper_preference_text,
    cosine_similarity,
)
from daily_arxiv_agent.storage import SQLitePaperStore


SEMANTIC_FEEDBACK_INPUT_VERSION = "paper-metadata-v1"


class FeedbackInput(BaseModel):
    """User feedback payload accepted by the refinement Skill."""

    paper_id: str
    value: FeedbackValue
    note: str | None = None

    @model_validator(mode="after")
    def require_paper_id(self) -> "FeedbackInput":
        if not self.paper_id.strip():
            raise ValueError("feedback paper_id must not be blank")
        return self


@dataclass(frozen=True)
class FeedbackAdjustment:
    """Score movement caused by active feedback events."""

    score_delta: float
    rationale: str
    matched_event_count: int
    influences: tuple[FeedbackInfluenceRecord, ...] = ()


class FeedbackRefinementSkill:
    """Record like/dislike feedback and return refined recommendations."""

    def __init__(
        self,
        *,
        store: SQLitePaperStore | None = None,
        vectorizer: DeterministicTextVectorizer | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: AppConfig | None = None,
        feedback_weight: float = 6.0,
    ) -> None:
        self.store = store
        self.vectorizer = vectorizer or DeterministicTextVectorizer()
        self.config = config or AppConfig.from_env()
        self._embedding_provider = embedding_provider
        self.feedback_weight = feedback_weight

    def record_feedback(
        self,
        feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]],
        *,
        recommendations: Sequence[Recommendation] = (),
        papers: Sequence[PaperMetadata] = (),
        profile_id: str = "default",
        recommendation_run_id: str | None = None,
    ) -> SkillResult[list[FeedbackEvent]]:
        paper_lookup = _paper_lookup(recommendations, papers)
        normalized, error = _normalize_feedback_inputs(
            feedback,
            paper_lookup=paper_lookup,
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
            store=self.store,
        )
        if error is not None:
            return error

        events = normalized or []
        if self.store is not None:
            self.store.save_feedback_events(events)

        return SkillResult[list[FeedbackEvent]](
            status=SkillStatus.SUCCESS,
            data=events,
            evidence_source=_feedback_evidence_source(events),
            provenance=[
                event.paper.provenance for event in events if event.paper is not None
            ],
            message="Recorded paper feedback.",
            metadata={
                "profile_id": profile_id,
                "recommendation_run_id": recommendation_run_id,
                "feedback_count": len(events),
                "feedback_rule": "latest_wins",
            },
        )

    def refine(
        self,
        recommendations: Sequence[Recommendation],
        *,
        feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]] = (),
        papers: Sequence[PaperMetadata] = (),
        profile_id: str = "default",
        recommendation_run_id: str | None = None,
        semantic_context: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SkillResult[list[Recommendation]]:
        paper_lookup = _paper_lookup(recommendations, papers)
        new_events: list[FeedbackEvent] = []
        if feedback:
            normalized, error = _normalize_feedback_inputs(
                feedback,
                paper_lookup=paper_lookup,
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id,
                store=self.store,
            )
            if error is not None:
                return error
            new_events = normalized or []
            if self.store is not None:
                self.store.save_feedback_events(new_events)

        if not recommendations:
            return SkillResult[list[Recommendation]](
                status=SkillStatus.EMPTY,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                message="No recommendations are available for feedback refinement.",
                metadata={
                    "profile_id": profile_id,
                    "recommendation_run_id": recommendation_run_id,
                    "feedback_count": len(new_events),
                },
            )

        feedback_events = (
            self.store.list_feedback_events(
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id,
            )
            if self.store is not None
            else new_events
        )
        feedback_events = _hydrate_events(
            feedback_events,
            paper_lookup,
            store=self.store,
        )
        active_events = latest_feedback_events(feedback_events)

        semantic_context_payload = _semantic_context_for_refinement(
            semantic_context,
            recommendations,
        )
        if _should_use_semantic_feedback(recommendations, semantic_context_payload):
            return self._refine_semantic(
                recommendations,
                active_events=active_events,
                feedback_event_count=len(feedback_events),
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id,
                semantic_context=semantic_context_payload,
                top_k=top_k,
            )

        return self._refine_deterministic(
            recommendations,
            active_events=active_events,
            feedback_event_count=len(feedback_events),
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
            top_k=top_k,
        )

    def _refine_deterministic(
        self,
        recommendations: Sequence[Recommendation],
        *,
        active_events: Mapping[str, FeedbackEvent],
        feedback_event_count: int,
        profile_id: str,
        recommendation_run_id: str | None,
        top_k: int | None,
    ) -> SkillResult[list[Recommendation]]:
        adjustments = {
            recommendation.paper.paper_id: feedback_adjustment_for_paper(
                recommendation.paper,
                active_events.values(),
                vectorizer=self.vectorizer,
                feedback_weight=self.feedback_weight,
            )
            for recommendation in recommendations
        }
        refined = _refined_recommendations(
            recommendations,
            adjustments=adjustments,
            top_k=top_k,
        )
        return _successful_refinement_result(
            refined,
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
            feedback_event_count=feedback_event_count,
            active_feedback_count=len(active_events),
            refinement_mode="deterministic_feedback",
            refinement_status=(
                FeedbackRefinementStatus.APPLIED
                if any(item.feedback_influences for item in refined)
                else FeedbackRefinementStatus.SKIPPED
            ),
            message="Refined recommendations using latest paper feedback.",
        )

    def _refine_semantic(
        self,
        recommendations: Sequence[Recommendation],
        *,
        active_events: Mapping[str, FeedbackEvent],
        feedback_event_count: int,
        profile_id: str,
        recommendation_run_id: str | None,
        semantic_context: Mapping[str, Any],
        top_k: int | None,
    ) -> SkillResult[list[Recommendation]]:
        source_events = [
            event
            for event in active_events.values()
            if event.paper is not None and _paper_has_embedding_text(event.paper)
        ]
        skipped_feedback_count = len(active_events) - len(source_events)
        context = _resolve_semantic_feedback_context(
            self.config,
            semantic_context,
            store=self.store,
            embedding_provider=self._embedding_provider,
        )

        if not source_events:
            adjustments = {
                recommendation.paper.paper_id: FeedbackAdjustment(
                    score_delta=0.0,
                    rationale="No active semantic feedback had hydrated paper metadata.",
                    matched_event_count=0,
                )
                for recommendation in recommendations
            }
            refined = _refined_recommendations(
                recommendations,
                adjustments=adjustments,
                top_k=top_k,
            )
            return _successful_refinement_result(
                refined,
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id,
                feedback_event_count=feedback_event_count,
                active_feedback_count=len(active_events),
                refinement_mode="semantic_feedback",
                refinement_status=FeedbackRefinementStatus.SKIPPED,
                message=(
                    "Semantic feedback was recorded, but no hydrated feedback "
                    "paper metadata was available for refinement."
                ),
                semantic_context=context,
                cache_metadata=EmbeddingCacheMetadata(enabled=context.cache_enabled),
                skipped_feedback_count=skipped_feedback_count,
            )

        try:
            vectors, cache_metadata = self._semantic_feedback_vectors(
                source_events,
                recommendations,
                context=context,
                profile_id=profile_id,
            )
            adjustments = _semantic_feedback_adjustments(
                recommendations,
                source_events,
                vectors=vectors,
                feedback_weight=self.feedback_weight,
            )
        except EmbeddingConfigurationError as exc:
            return _failed_semantic_refinement_result(
                recommendations,
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id,
                feedback_event_count=feedback_event_count,
                active_feedback_count=len(active_events),
                semantic_context=context,
                code="semantic_feedback_configuration_failed",
                message=f"Semantic feedback configuration failed: {exc}",
                retryable=False,
                skipped_feedback_count=skipped_feedback_count,
            )
        except EmbeddingProviderError as exc:
            return _failed_semantic_refinement_result(
                recommendations,
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id,
                feedback_event_count=feedback_event_count,
                active_feedback_count=len(active_events),
                semantic_context=context,
                code="semantic_feedback_provider_failed",
                message=f"Semantic feedback provider failed: {exc}",
                retryable=True,
                skipped_feedback_count=skipped_feedback_count,
            )
        except Exception as exc:
            return _failed_semantic_refinement_result(
                recommendations,
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id,
                feedback_event_count=feedback_event_count,
                active_feedback_count=len(active_events),
                semantic_context=context,
                code="semantic_feedback_refinement_failed",
                message=f"Semantic feedback refinement failed: {exc}",
                retryable=True,
                skipped_feedback_count=skipped_feedback_count,
            )

        refined = _refined_recommendations(
            recommendations,
            adjustments=adjustments,
            top_k=top_k,
        )
        return _successful_refinement_result(
            refined,
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
            feedback_event_count=feedback_event_count,
            active_feedback_count=len(active_events),
            refinement_mode="semantic_feedback",
            refinement_status=(
                FeedbackRefinementStatus.APPLIED
                if any(item.feedback_influences for item in refined)
                else FeedbackRefinementStatus.SKIPPED
            ),
            message="Refined semantic recommendations using latest paper feedback.",
            semantic_context=context,
            cache_metadata=cache_metadata,
            skipped_feedback_count=skipped_feedback_count,
        )

    def _semantic_feedback_vectors(
        self,
        source_events: Sequence[FeedbackEvent],
        recommendations: Sequence[Recommendation],
        *,
        context: "_SemanticFeedbackContext",
        profile_id: str,
    ) -> tuple[dict[str, list[float]], EmbeddingCacheMetadata]:
        cache_metadata = EmbeddingCacheMetadata(enabled=context.cache_enabled)
        vectors: dict[str, list[float]] = {}
        misses: list[tuple[_FeedbackEmbeddingInput, Any]] = []
        inputs = _feedback_embedding_inputs(source_events, recommendations)

        for item in inputs:
            identity = (
                SQLitePaperStore.embedding_identity(
                    provider=context.provider,
                    model=context.model,
                    dimensions=context.dimensions,
                    input_version=context.input_version,
                    serialized_input=item.serialized_input,
                    cache_scope=item.cache_scope,
                    profile_id=(
                        profile_id
                        if item.cache_scope == EmbeddingCacheScope.PROFILE
                        else None
                    ),
                )
                if self.store is not None
                else None
            )
            cached = (
                self.store.load_embedding(identity, cache_enabled=context.cache_enabled)
                if self.store is not None
                else None
            )
            if cached is not None:
                cache_metadata.hits += 1
                vectors[item.key] = cached.vector
                continue
            if context.cache_enabled:
                cache_metadata.misses += 1
            else:
                cache_metadata.disabled_requests += 1
            misses.append((item, identity))

        if not misses:
            return vectors, cache_metadata

        provider = self._provider_for_semantic_context(context)
        provider_vectors = provider.embed_texts([item.text for item, _identity in misses])
        if len(provider_vectors) != len(misses):
            raise EmbeddingProviderError(
                "embedding provider returned an unexpected vector count."
            )

        for (item, identity), vector in zip(misses, provider_vectors, strict=True):
            dense_vector = _validate_dense_vector(vector, dimensions=context.dimensions)
            vectors[item.key] = dense_vector
            if self.store is None:
                continue
            saved = self.store.save_embedding(
                identity,
                dense_vector,
                input_role=item.role,
                metadata={
                    "input_version": context.input_version,
                    "item_id": item.item_id,
                },
                cache_enabled=context.cache_enabled,
            )
            if saved is not None:
                cache_metadata.writes += 1

        return vectors, cache_metadata

    def _provider_for_semantic_context(
        self,
        context: "_SemanticFeedbackContext",
    ) -> EmbeddingProvider:
        if self._embedding_provider is not None:
            return self._embedding_provider
        config = replace(
            self.config,
            embedding_provider=context.provider,
            embedding_model=context.model,
            embedding_dimensions=context.dimensions,
        )
        return create_embedding_provider(config)


@dataclass(frozen=True)
class _SemanticFeedbackContext:
    provider: str
    provider_mode: str
    provider_label: str
    model: str
    dimensions: int | None
    input_version: str
    cache_enabled: bool


@dataclass(frozen=True)
class _FeedbackEmbeddingInput:
    key: str
    item_id: str
    text: str
    serialized_input: dict[str, Any]
    role: EmbeddingInputRole
    cache_scope: EmbeddingCacheScope


def latest_feedback_events(
    events: Sequence[FeedbackEvent],
) -> dict[str, FeedbackEvent]:
    """Return the latest event per paper; this documents the conflict rule."""

    latest: dict[str, FeedbackEvent] = {}
    for event in sorted(events, key=lambda item: (item.created_at, item.event_id)):
        latest[event.paper_id] = event
    return latest


def feedback_adjustment_for_paper(
    paper: PaperMetadata,
    events: Sequence[FeedbackEvent],
    *,
    vectorizer: DeterministicTextVectorizer,
    feedback_weight: float,
) -> FeedbackAdjustment:
    """Calculate the signed feedback score adjustment for one candidate paper."""

    candidate_vector = vectorizer.vectorize(build_paper_preference_text(paper))
    score_delta = 0.0
    rationale_parts: list[str] = []
    matched_event_count = 0
    influences: list[FeedbackInfluenceRecord] = []

    for event in latest_feedback_events(events).values():
        if event.paper is None:
            continue
        source_vector = vectorizer.vectorize(build_paper_preference_text(event.paper))
        similarity = cosine_similarity(source_vector, candidate_vector)
        if similarity <= 0:
            continue

        sign = 1.0 if event.value == FeedbackValue.LIKE else -1.0
        signed_delta = sign * similarity * feedback_weight
        score_delta += signed_delta
        matched_event_count += 1
        action = "liked" if event.value == FeedbackValue.LIKE else "disliked"
        direction = "up" if signed_delta > 0 else "down"
        rationale_parts.append(
            f"{action} {event.paper_id} moved similar papers {direction} "
            f"({signed_delta:+.3f})"
        )
        influences.append(
            FeedbackInfluenceRecord(
                source_paper_id=event.paper_id,
                source_title=event.paper.title,
                target_paper_id=paper.paper_id,
                target_title=paper.title,
                similarity=round(similarity, 4),
                signed_score_delta=round(signed_delta, 4),
                value=event.value,
                refinement_status=FeedbackRefinementStatus.APPLIED,
                event_id=event.event_id,
            )
        )

    return FeedbackAdjustment(
        score_delta=round(score_delta, 4),
        rationale="; ".join(rationale_parts),
        matched_event_count=matched_event_count,
        influences=tuple(influences),
    )


def _refined_recommendations(
    recommendations: Sequence[Recommendation],
    *,
    adjustments: Mapping[str, FeedbackAdjustment],
    top_k: int | None,
) -> list[Recommendation]:
    rescored: list[tuple[Recommendation, float, FeedbackAdjustment]] = []
    for recommendation in recommendations:
        adjustment = adjustments.get(
            recommendation.paper.paper_id,
            FeedbackAdjustment(score_delta=0.0, rationale="", matched_event_count=0),
        )
        new_score = round(recommendation.score + adjustment.score_delta, 4)
        rescored.append((recommendation, new_score, adjustment))

    rescored.sort(
        key=lambda item: (-item[1], item[0].rank, item[0].paper.title.lower())
    )
    limit = len(rescored) if top_k is None else max(top_k, 0)
    return [
        _refined_recommendation(
            recommendation,
            rank=rank,
            new_score=new_score,
            adjustment=adjustment,
        )
        for rank, (recommendation, new_score, adjustment) in enumerate(
            rescored[:limit],
            start=1,
        )
    ]


def _refined_recommendation(
    recommendation: Recommendation,
    *,
    rank: int,
    new_score: float,
    adjustment: FeedbackAdjustment,
) -> Recommendation:
    score_delta = round(new_score - recommendation.score, 4)
    influences = list(adjustment.influences)
    refinement_status = (
        FeedbackRefinementStatus.APPLIED
        if influences
        else FeedbackRefinementStatus.SKIPPED
    )
    return recommendation.model_copy(
        update={
            "rank": rank,
            "score": new_score,
            "rationale": _refined_rationale(
                recommendation.rationale,
                recommendation.rank,
                score_delta,
                adjustment,
            ),
            "previous_rank": recommendation.rank,
            "previous_score": recommendation.score,
            "score_delta": score_delta,
            "rank_delta": recommendation.rank - rank,
            "score_breakdown": _updated_score_breakdown(
                recommendation.score_breakdown,
                new_score=new_score,
                score_delta=score_delta,
            ),
            "feedback_influences": influences,
            "refinement_status": refinement_status,
            "feedback_error": None,
        }
    )


def _updated_score_breakdown(
    original: RankingScoreBreakdown | None,
    *,
    new_score: float,
    score_delta: float,
) -> RankingScoreBreakdown:
    if original is None:
        return RankingScoreBreakdown(
            feedback=round(score_delta, 4),
            total=round(new_score, 4),
            signals=(["feedback"] if abs(score_delta) > 0.0001 else []),
        )

    feedback = round(original.feedback + score_delta, 4)
    signals = [signal for signal in original.signals if signal != "feedback"]
    if abs(feedback) > 0.0001:
        signals.append("feedback")
    return original.model_copy(
        update={
            "feedback": feedback,
            "total": round(new_score, 4),
            "signals": signals,
        }
    )


def _successful_refinement_result(
    refined: Sequence[Recommendation],
    *,
    profile_id: str,
    recommendation_run_id: str | None,
    feedback_event_count: int,
    active_feedback_count: int,
    refinement_mode: str,
    refinement_status: FeedbackRefinementStatus,
    message: str,
    semantic_context: _SemanticFeedbackContext | None = None,
    cache_metadata: EmbeddingCacheMetadata | None = None,
    skipped_feedback_count: int = 0,
) -> SkillResult[list[Recommendation]]:
    metadata: dict[str, Any] = {
        "profile_id": profile_id,
        "recommendation_run_id": recommendation_run_id,
        "feedback_count": feedback_event_count,
        "active_feedback_count": active_feedback_count,
        "feedback_rule": "latest_wins",
        "refinement_mode": refinement_mode,
        "refinement_status": refinement_status.value,
        "influence_count": sum(len(item.feedback_influences) for item in refined),
        "skipped_feedback_count": skipped_feedback_count,
    }
    if semantic_context is not None:
        provider_metadata = EmbeddingProviderCacheMetadata(
            provider=semantic_context.provider,
            provider_mode=semantic_context.provider_mode,
            provider_label=semantic_context.provider_label,
            model=semantic_context.model,
            dimensions=semantic_context.dimensions,
            cache=cache_metadata or EmbeddingCacheMetadata(
                enabled=semantic_context.cache_enabled
            ),
        ).model_dump(mode="json")
        metadata.update(
            {
                "semantic_provider": {
                    key: value
                    for key, value in provider_metadata.items()
                    if key != "cache"
                },
                "embedding_cache": provider_metadata["cache"],
            }
        )

    return SkillResult[list[Recommendation]](
        status=SkillStatus.SUCCESS,
        data=list(refined),
        evidence_source=_recommendation_evidence_source(refined),
        provenance=[item.paper.provenance for item in refined],
        message=message,
        metadata=metadata,
    )


def _failed_semantic_refinement_result(
    recommendations: Sequence[Recommendation],
    *,
    profile_id: str,
    recommendation_run_id: str | None,
    feedback_event_count: int,
    active_feedback_count: int,
    semantic_context: _SemanticFeedbackContext,
    code: str,
    message: str,
    retryable: bool,
    skipped_feedback_count: int,
) -> SkillResult[list[Recommendation]]:
    error = SkillError(code=code, message=message, retryable=retryable)
    failed = [
        recommendation.model_copy(
            update={
                "refinement_status": FeedbackRefinementStatus.FAILED,
                "feedback_error": error,
                "feedback_influences": [],
            }
        )
        for recommendation in recommendations
    ]
    provider_metadata = EmbeddingProviderCacheMetadata(
        provider=semantic_context.provider,
        provider_mode=semantic_context.provider_mode,
        provider_label=semantic_context.provider_label,
        model=semantic_context.model,
        dimensions=semantic_context.dimensions,
        cache=EmbeddingCacheMetadata(enabled=semantic_context.cache_enabled),
    ).model_dump(mode="json")
    return SkillResult[list[Recommendation]](
        status=SkillStatus.ERROR,
        data=failed,
        evidence_source=_recommendation_evidence_source(failed),
        provenance=[item.paper.provenance for item in failed],
        error=error,
        message="Semantic feedback refinement failed; original recommendations were preserved.",
        metadata={
            "profile_id": profile_id,
            "recommendation_run_id": recommendation_run_id,
            "feedback_count": feedback_event_count,
            "active_feedback_count": active_feedback_count,
            "feedback_rule": "latest_wins",
            "refinement_mode": "semantic_feedback",
            "refinement_status": FeedbackRefinementStatus.FAILED.value,
            "influence_count": 0,
            "skipped_feedback_count": skipped_feedback_count,
            "semantic_provider": {
                key: value
                for key, value in provider_metadata.items()
                if key != "cache"
            },
            "embedding_cache": provider_metadata["cache"],
            "feedback_error": error.model_dump(mode="json"),
        },
    )


def _semantic_feedback_adjustments(
    recommendations: Sequence[Recommendation],
    source_events: Sequence[FeedbackEvent],
    *,
    vectors: Mapping[str, Sequence[float]],
    feedback_weight: float,
) -> dict[str, FeedbackAdjustment]:
    adjustments: dict[str, FeedbackAdjustment] = {}
    for recommendation in recommendations:
        candidate_key = _candidate_input_key(recommendation.paper.paper_id)
        candidate_vector = vectors[candidate_key]
        score_delta = 0.0
        rationale_parts: list[str] = []
        influences: list[FeedbackInfluenceRecord] = []
        for event in latest_feedback_events(source_events).values():
            if event.paper is None:
                continue
            source_key = _feedback_event_input_key(event)
            similarity = _cosine_dense(vectors[source_key], candidate_vector)
            if similarity <= 0:
                continue
            sign = 1.0 if event.value == FeedbackValue.LIKE else -1.0
            signed_delta = sign * similarity * feedback_weight
            score_delta += signed_delta
            action = "liked" if event.value == FeedbackValue.LIKE else "disliked"
            direction = "up" if signed_delta > 0 else "down"
            rationale_parts.append(
                f"{action} {event.paper_id} semantically moved this paper {direction} "
                f"({signed_delta:+.3f}; similarity {similarity:.3f})"
            )
            influences.append(
                FeedbackInfluenceRecord(
                    source_paper_id=event.paper_id,
                    source_title=event.paper.title,
                    target_paper_id=recommendation.paper.paper_id,
                    target_title=recommendation.paper.title,
                    similarity=round(similarity, 4),
                    signed_score_delta=round(signed_delta, 4),
                    value=event.value,
                    refinement_status=FeedbackRefinementStatus.APPLIED,
                    event_id=event.event_id,
                )
            )

        adjustments[recommendation.paper.paper_id] = FeedbackAdjustment(
            score_delta=round(score_delta, 4),
            rationale="; ".join(rationale_parts),
            matched_event_count=len(influences),
            influences=tuple(influences),
        )
    return adjustments


def _semantic_context_for_refinement(
    explicit_context: Mapping[str, Any] | None,
    recommendations: Sequence[Recommendation],
) -> dict[str, Any]:
    if explicit_context:
        return dict(explicit_context)
    for recommendation in recommendations:
        if recommendation.semantic_context:
            return dict(recommendation.semantic_context)
    return {}


def _should_use_semantic_feedback(
    recommendations: Sequence[Recommendation],
    semantic_context: Mapping[str, Any],
) -> bool:
    if semantic_context:
        return True
    for recommendation in recommendations:
        breakdown = recommendation.score_breakdown
        if breakdown is None:
            continue
        if breakdown.semantic_similarities or abs(breakdown.semantic_seed) > 0.0001:
            return True
    return False


def _resolve_semantic_feedback_context(
    config: AppConfig,
    semantic_context: Mapping[str, Any],
    *,
    store: SQLitePaperStore | None,
    embedding_provider: EmbeddingProvider | None,
) -> _SemanticFeedbackContext:
    nested_context = _mapping_value(semantic_context, "semantic_context")
    provider_context = _mapping_value(semantic_context, "semantic_provider")
    cache_context = _mapping_value(semantic_context, "embedding_cache")

    provider = _string_value(
        nested_context.get("provider")
        or provider_context.get("provider")
        or semantic_context.get("provider")
        or _configured_provider_name(config, embedding_provider),
        default=_configured_provider_name(config, embedding_provider),
    )
    model = _string_value(
        nested_context.get("model")
        or provider_context.get("model")
        or semantic_context.get("model")
        or config.embedding_model,
        default=config.embedding_model,
    )
    dimensions = _optional_int_value(
        nested_context.get("dimensions")
        if "dimensions" in nested_context
        else (
            provider_context.get("dimensions")
            if "dimensions" in provider_context
            else semantic_context.get("dimensions")
        )
    )
    if dimensions is None:
        dimensions = (
            config.embedding_dimensions
            if config.embedding_dimensions is not None
            else getattr(embedding_provider, "dimensions", None)
        )
    provider_mode = _string_value(
        provider_context.get("provider_mode")
        or ("fake" if provider == "fake" else "live"),
        default=("fake" if provider == "fake" else "live"),
    )
    provider_label = _string_value(
        provider_context.get("provider_label") or f"{provider}:{model}",
        default=f"{provider}:{model}",
    )
    cache_enabled = bool(
        cache_context.get("enabled", config.embedding_cache_enabled)
    ) and store is not None
    input_version = _string_value(
        nested_context.get("input_version")
        or semantic_context.get("input_version")
        or SEMANTIC_FEEDBACK_INPUT_VERSION,
        default=SEMANTIC_FEEDBACK_INPUT_VERSION,
    )
    return _SemanticFeedbackContext(
        provider=provider,
        provider_mode=provider_mode,
        provider_label=provider_label,
        model=model,
        dimensions=dimensions,
        input_version=input_version,
        cache_enabled=cache_enabled,
    )


def _feedback_embedding_inputs(
    source_events: Sequence[FeedbackEvent],
    recommendations: Sequence[Recommendation],
) -> list[_FeedbackEmbeddingInput]:
    inputs: list[_FeedbackEmbeddingInput] = []
    seen: set[str] = set()
    for event in source_events:
        if event.paper is None:
            continue
        item = _paper_embedding_input(
            event.paper,
            key=_feedback_event_input_key(event),
            item_id=event.paper_id,
            role=EmbeddingInputRole.FEEDBACK,
            cache_scope=EmbeddingCacheScope.PROFILE,
        )
        if item is None or item.key in seen:
            continue
        seen.add(item.key)
        inputs.append(item)

    for recommendation in recommendations:
        item = _paper_embedding_input(
            recommendation.paper,
            key=_candidate_input_key(recommendation.paper.paper_id),
            item_id=recommendation.paper.paper_id,
            role=EmbeddingInputRole.CANDIDATE,
            cache_scope=EmbeddingCacheScope.GLOBAL,
        )
        if item is None or item.key in seen:
            continue
        seen.add(item.key)
        inputs.append(item)
    return inputs


def _paper_embedding_input(
    paper: PaperMetadata,
    *,
    key: str,
    item_id: str,
    role: EmbeddingInputRole,
    cache_scope: EmbeddingCacheScope,
) -> _FeedbackEmbeddingInput | None:
    payload = _metadata_payload(paper)
    if payload is None:
        return None
    return _FeedbackEmbeddingInput(
        key=key,
        item_id=item_id,
        text=_payload_text(payload),
        serialized_input=payload,
        role=role,
        cache_scope=cache_scope,
    )


def _metadata_payload(paper: PaperMetadata) -> dict[str, Any] | None:
    payload = {
        "title": " ".join(paper.title.split()),
        "abstract": " ".join((paper.abstract or "").split()),
        "categories": [
            " ".join(category.split())
            for category in paper.categories
            if category.strip()
        ],
    }
    if not (payload["title"] or payload["abstract"] or payload["categories"]):
        return None
    return payload


def _payload_text(payload: Mapping[str, Any]) -> str:
    parts = [
        str(payload.get("title") or ""),
        str(payload.get("abstract") or ""),
        " ".join(str(category) for category in payload.get("categories") or []),
    ]
    return normalize_provider_input_text(" ".join(part for part in parts if part))


def _paper_has_embedding_text(paper: PaperMetadata) -> bool:
    payload = _metadata_payload(paper)
    return payload is not None and bool(normalize_embedding_text(_payload_text(payload)))


def _feedback_event_input_key(event: FeedbackEvent) -> str:
    return f"feedback:{event.event_id}"


def _candidate_input_key(paper_id: str) -> str:
    return f"candidate:{paper_id}"


def _validate_dense_vector(
    vector: Sequence[float],
    *,
    dimensions: int | None,
) -> list[float]:
    values = [float(value) for value in vector]
    if not values:
        raise EmbeddingProviderError("embedding provider returned an empty vector.")
    if dimensions is not None and len(values) != dimensions:
        raise EmbeddingProviderError(
            "embedding provider returned a vector with invalid dimensions."
        )
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingProviderError(
            "embedding provider returned a non-finite vector value."
        )
    return values


def _cosine_dense(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise EmbeddingProviderError("embedding vector dimensions do not match.")
    dot = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _configured_provider_name(
    config: AppConfig,
    embedding_provider: EmbeddingProvider | None,
) -> str:
    if embedding_provider is not None:
        class_name = embedding_provider.__class__.__name__.lower()
        if "fake" in class_name:
            return "fake"
    return (config.embedding_provider or "openai").strip().lower() or "openai"


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if isinstance(value, Mapping):
        return value
    return {}


def _string_value(value: object, *, default: str) -> str:
    if value is None:
        return default
    normalized = " ".join(str(value).split())
    return normalized or default


def _optional_int_value(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _recommendation_evidence_source(
    recommendations: Sequence[Recommendation],
) -> EvidenceSource:
    return (
        EvidenceSource.ABSTRACT
        if any(item.evidence_source == EvidenceSource.ABSTRACT for item in recommendations)
        else EvidenceSource.METADATA
    )


def _normalize_feedback_inputs(
    feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]],
    *,
    paper_lookup: dict[str, PaperMetadata],
    profile_id: str,
    recommendation_run_id: str | None,
    store: SQLitePaperStore | None = None,
) -> tuple[list[FeedbackEvent] | None, SkillResult[list[FeedbackEvent]] | None]:
    events: list[FeedbackEvent] = []
    for raw_feedback in feedback:
        try:
            if isinstance(raw_feedback, FeedbackEvent):
                event = raw_feedback
            else:
                feedback_input = (
                    raw_feedback
                    if isinstance(raw_feedback, FeedbackInput)
                    else FeedbackInput.model_validate(raw_feedback)
                )
                event = FeedbackEvent(
                    profile_id=profile_id,
                    recommendation_run_id=recommendation_run_id,
                    paper_id=feedback_input.paper_id,
                    value=feedback_input.value,
                    paper=(
                        paper_lookup.get(feedback_input.paper_id)
                        or _paper_from_store(store, feedback_input.paper_id)
                    ),
                    note=feedback_input.note,
                )

            if event.paper is None:
                paper = paper_lookup.get(event.paper_id) or _paper_from_store(
                    store,
                    event.paper_id,
                )
                if paper is not None:
                    event = event.model_copy(update={"paper": paper})
            if event.recommendation_run_id is None and recommendation_run_id is not None:
                event = event.model_copy(
                    update={"recommendation_run_id": recommendation_run_id}
                )
            events.append(event)
        except ValidationError as exc:
            code = (
                "invalid_feedback_value"
                if "value" in {str(part) for error in exc.errors() for part in error["loc"]}
                else "invalid_feedback_input"
            )
            return None, SkillResult[list[FeedbackEvent]](
                status=SkillStatus.ERROR,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code=code,
                    message="Feedback must include a paper_id and value of like or dislike.",
                    retryable=False,
                ),
                metadata={"validation_error": str(exc)},
            )

    return events, None


def _hydrate_events(
    events: Sequence[FeedbackEvent],
    paper_lookup: dict[str, PaperMetadata],
    *,
    store: SQLitePaperStore | None = None,
) -> list[FeedbackEvent]:
    hydrated: list[FeedbackEvent] = []
    for event in events:
        if event.paper is None:
            paper = paper_lookup.get(event.paper_id) or _paper_from_store(
                store,
                event.paper_id,
            )
            if paper is not None:
                hydrated.append(event.model_copy(update={"paper": paper}))
                continue
        hydrated.append(event)
    return hydrated


def _paper_from_store(
    store: SQLitePaperStore | None,
    paper_id: str,
) -> PaperMetadata | None:
    if store is None:
        return None
    return store.get_paper(paper_id)


def _paper_lookup(
    recommendations: Sequence[Recommendation],
    papers: Sequence[PaperMetadata],
) -> dict[str, PaperMetadata]:
    lookup = {
        recommendation.paper.paper_id: recommendation.paper
        for recommendation in recommendations
    }
    lookup.update({paper.paper_id: paper for paper in papers})
    return lookup


def _feedback_evidence_source(events: Sequence[FeedbackEvent]) -> EvidenceSource:
    return (
        EvidenceSource.ABSTRACT
        if any(event.paper and event.paper.abstract for event in events)
        else EvidenceSource.METADATA
    )


def _refined_rationale(
    original_rationale: str,
    previous_rank: int,
    score_delta: float,
    adjustment: FeedbackAdjustment,
) -> str:
    feedback_text = (
        adjustment.rationale
        if adjustment.rationale
        else "No active feedback matched this paper."
    )
    return (
        f"{original_rationale} Feedback adjustment: {feedback_text}. "
        f"Previous rank: {previous_rank}; score delta: {score_delta:+.4f}."
    )
