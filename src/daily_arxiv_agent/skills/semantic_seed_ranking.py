"""Semantic seed-first ranking for retrieved arXiv candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Mapping, Sequence

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    EmbeddingCacheMetadata,
    EmbeddingCacheScope,
    EmbeddingInputRole,
    EmbeddingProviderCacheMetadata,
    EvidenceSource,
    FeedbackEvent,
    PaperMetadata,
    QueryPlan,
    RankingScoreBreakdown,
    Recommendation,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SeedPreference,
    SeedRecord,
    SemanticSimilarityDetail,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
    EmbeddingProviderError,
    SemanticReadiness,
    normalize_embedding_text,
    normalize_provider_input_text,
)
from daily_arxiv_agent.embeddings.provider import (
    check_semantic_readiness,
    create_embedding_provider,
)
from daily_arxiv_agent.skills.feedback import feedback_adjustment_for_paper
from daily_arxiv_agent.skills.ranking import (
    SEMANTIC_SEED_RANKING_MODE,
    SEMANTIC_TOPIC_SEED_RANKING_MODE,
    _category_signal,
    _date_sort_value,
    _lexical_signal,
    _newest_published_date,
    _normalize_source_metadata_by_paper_id,
    _phrase_signal,
    _query_phrases,
    _query_source_signal,
    _query_terms,
    _recency_signal,
)
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
)
from daily_arxiv_agent.storage import SQLitePaperStore


EMBEDDING_INPUT_VERSION = "paper-metadata-v1"


class SemanticSeedRankingSkill:
    """Rank candidates by seed-paper embedding similarity plus bounded signals."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        store: SQLitePaperStore | None = None,
        config: AppConfig | None = None,
        vectorizer: DeterministicTextVectorizer | None = None,
        semantic_weight: float = 100.0,
        lexical_cap: float = 3.0,
        phrase_cap: float = 2.0,
        query_source_cap: float = 1.5,
        recency_cap: float = 1.0,
        category_cap: float = 1.0,
        feedback_cap: float = 2.0,
        minimum_semantic_similarity: float = 0.35,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self._embedding_provider = embedding_provider
        self._provider_injected = embedding_provider is not None
        self.store = store
        self.vectorizer = vectorizer or DeterministicTextVectorizer()
        self.semantic_weight = semantic_weight
        self.lexical_cap = lexical_cap
        self.phrase_cap = phrase_cap
        self.query_source_cap = query_source_cap
        self.recency_cap = recency_cap
        self.category_cap = category_cap
        self.feedback_cap = feedback_cap
        self.minimum_semantic_similarity = minimum_semantic_similarity

        provider_name = _provider_name(self.config, embedding_provider)
        self.provider = provider_name
        self.provider_mode = "fake" if provider_name == "fake" else "live"
        self.model = self.config.embedding_model
        self.dimensions = (
            self.config.embedding_dimensions
            if self.config.embedding_dimensions is not None
            else getattr(embedding_provider, "dimensions", None)
        )
        self.provider_label = f"{self.provider}:{self.model}"
        self.cache_enabled = bool(self.config.embedding_cache_enabled and store is not None)

    def check_readiness(
        self,
        seed_preference: SeedPreference | None,
    ) -> SemanticReadiness:
        """Return trace-safe readiness status using this skill's provider setup."""

        seed_inputs = _seed_embedding_inputs(seed_preference)
        seed_texts = [item.text for item in seed_inputs]
        if self._provider_injected:
            seed_quality = (
                "usable"
                if any(normalize_embedding_text(text) for text in seed_texts)
                else "missing"
            )
            error_code = (
                "semantic_seed_quality_error"
                if seed_quality != "usable"
                else None
            )
            return SemanticReadiness(
                provider=self.provider,
                provider_mode=self.provider_mode,
                provider_label=self.provider_label,
                credential_status="injected",
                model=self.model,
                endpoint=None,
                endpoint_safety="not_applicable",
                cache_enabled=self.cache_enabled,
                seed_quality=seed_quality,
                can_run=error_code is None,
                error_code=error_code,
            )
        return check_semantic_readiness(
            self.config,
            seed_texts=seed_texts,
            cache_enabled=self.cache_enabled,
        )

    def rank(
        self,
        papers: Sequence[PaperMetadata],
        *,
        topic: str | None = None,
        seed_preference: SeedPreference | None = None,
        feedback_events: Sequence[FeedbackEvent] | None = None,
        top_k: int = 5,
        query_plan: QueryPlan | None = None,
        retrieval_query: RetrievalQuery | None = None,
        retrieval_source_metadata_by_paper_id: Mapping[
            str, Sequence[RetrievalSourceMetadata | Mapping[str, object]]
        ]
        | None = None,
        profile_id: str | None = None,
    ) -> SkillResult[list[Recommendation]]:
        """Rank papers in semantic seed mode without deterministic fallback."""

        ranking_mode = _semantic_ranking_mode(topic)
        seed_inputs = _seed_embedding_inputs(seed_preference)
        if seed_preference is None or not seed_inputs:
            return self._error_result(
                code="semantic_seed_quality_error",
                message="Semantic seed ranking requires usable seed title, abstract, or category text.",
                ranking_mode=ranking_mode,
                top_k=top_k,
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                failure_reason="seed_metadata_missing_text",
            )

        seed_profile_id = profile_id or seed_preference.profile_id
        seed_ids = _seed_paper_ids(seed_preference)
        candidates = [paper for paper in papers if paper.paper_id not in seed_ids]
        if not candidates:
            metadata = self._base_metadata(
                ranking_mode=ranking_mode,
                top_k=top_k,
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                cache_metadata=EmbeddingCacheMetadata(enabled=self.cache_enabled),
            )
            metadata.update(
                {
                    "candidate_count": 0,
                    "seed_excluded_count": len(papers) - len(candidates),
                    "qualifying_count": 0,
                    "fallback_count": 0,
                    "score_signals": [],
                }
            )
            return SkillResult[list[Recommendation]](
                status=SkillStatus.EMPTY,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                message="No non-seed candidates are available for semantic ranking.",
                metadata=metadata,
            )

        candidate_inputs = [_candidate_embedding_input(paper) for paper in candidates]
        try:
            vectors, cache_metadata = self._embed_inputs(
                [*seed_inputs, *candidate_inputs],
                seed_profile_id=seed_profile_id,
            )
        except EmbeddingConfigurationError as exc:
            return self._error_result(
                code="semantic_embedding_configuration_failed",
                message=f"Semantic embedding configuration failed: {exc}",
                ranking_mode=ranking_mode,
                top_k=top_k,
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                failure_reason="embedding_configuration_failed",
            )
        except EmbeddingProviderError as exc:
            return self._error_result(
                code="semantic_embedding_provider_failed",
                message=f"Semantic embedding provider failed: {exc}",
                ranking_mode=ranking_mode,
                top_k=top_k,
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                failure_reason="embedding_provider_failed",
            )

        effective_topic = topic or (retrieval_query.topic if retrieval_query else None)
        query_terms = _query_terms(effective_topic, query_plan)
        query_phrases = _query_phrases(effective_topic, query_plan)
        newest_published_date = _newest_published_date(candidates)
        source_metadata = _normalize_source_metadata_by_paper_id(
            retrieval_source_metadata_by_paper_id
        )

        try:
            scored = [
                self._score_paper(
                    paper,
                    seed_inputs=seed_inputs,
                    vectors=vectors,
                    query_terms=query_terms,
                    query_phrases=query_phrases,
                    retrieval_query=retrieval_query,
                    retrieval_source_metadata=source_metadata.get(paper.paper_id, ()),
                    newest_published_date=newest_published_date,
                    feedback_events=feedback_events or (),
                )
                for paper in candidates
            ]
            scored.sort(
                key=lambda item: (
                    -_semantic_bucket_rank(item.semantic_bucket),
                    -item.semantic_similarity,
                    -item.secondary_score,
                    -_date_sort_value(item.paper.published_date),
                    item.paper.title.lower(),
                )
            )
        except EmbeddingProviderError as exc:
            return self._error_result(
                code="semantic_embedding_provider_failed",
                message=f"Semantic embedding provider failed: {exc}",
                ranking_mode=ranking_mode,
                top_k=top_k,
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                failure_reason="embedding_provider_failed",
            )
        except Exception as exc:
            return self._error_result(
                code="semantic_ranking_failed",
                message=f"Semantic seed ranking failed: {exc}",
                ranking_mode=ranking_mode,
                top_k=top_k,
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                failure_reason="ranking_failed",
            )

        selected = _select_semantic_scored(scored, max(top_k, 0))
        recommendations = [
            _recommendation_from_semantic_scored(
                item,
                rank=rank,
                fallback=not item.qualifies,
            )
            for rank, item in enumerate(selected, start=1)
        ]
        result_evidence = (
            EvidenceSource.ABSTRACT
            if any(item.evidence_source == EvidenceSource.ABSTRACT for item in recommendations)
            else EvidenceSource.METADATA
        )
        metadata = self._base_metadata(
            ranking_mode=ranking_mode,
            top_k=top_k,
            seed_preference=seed_preference,
            feedback_events=feedback_events or (),
            cache_metadata=cache_metadata,
        )
        metadata.update(
            {
                "candidate_count": len(candidates),
                "seed_excluded_count": len(papers) - len(candidates),
                "score_signals": _score_signal_names(scored),
                "qualifying_count": sum(1 for item in scored if item.qualifies),
                "fallback_count": sum(
                    1
                    for recommendation in recommendations
                    if recommendation.score_breakdown
                    and recommendation.score_breakdown.fallback
                ),
                "minimum_semantic_similarity": self.minimum_semantic_similarity,
                "semantic_similarity_threshold": self.minimum_semantic_similarity,
            }
        )
        return SkillResult[list[Recommendation]](
            status=SkillStatus.SUCCESS,
            data=recommendations,
            evidence_source=result_evidence,
            provenance=[item.paper.provenance for item in recommendations],
            metadata=metadata,
        )

    def _score_paper(
        self,
        paper: PaperMetadata,
        *,
        seed_inputs: Sequence["_EmbeddingInput"],
        vectors: Mapping[str, list[float]],
        query_terms: Sequence[str],
        query_phrases: Sequence[str],
        retrieval_query: RetrievalQuery | None,
        retrieval_source_metadata: Sequence[RetrievalSourceMetadata],
        newest_published_date: date | None,
        feedback_events: Sequence[FeedbackEvent],
    ) -> "_SemanticScoredPaper":
        candidate_key = _candidate_input_key(paper.paper_id)
        candidate_vector = vectors[candidate_key]
        similarity_details: list[SemanticSimilarityDetail] = []
        for seed_input in seed_inputs:
            similarity = _cosine_dense(vectors[seed_input.key], candidate_vector)
            similarity_details.append(
                SemanticSimilarityDetail(
                    source_id=seed_input.item_id,
                    target_id=paper.paper_id,
                    similarity=round(similarity, 4),
                    source_role=EmbeddingInputRole.SEED,
                    target_role=EmbeddingInputRole.CANDIDATE,
                    source_title=seed_input.title,
                    target_title=paper.title,
                    score=round(max(similarity, 0.0) * self.semantic_weight, 4),
                )
            )
        best_similarity = max(
            (detail.similarity for detail in similarity_details),
            default=0.0,
        )
        semantic_score = max(best_similarity, 0.0) * self.semantic_weight
        semantic_bucket = _semantic_bucket(best_similarity)

        secondary = self._secondary_signals(
            paper,
            query_terms=query_terms,
            query_phrases=query_phrases,
            retrieval_query=retrieval_query,
            retrieval_source_metadata=retrieval_source_metadata,
            newest_published_date=newest_published_date,
            feedback_events=feedback_events,
        )
        score = semantic_score + secondary.score
        qualifies = best_similarity >= self.minimum_semantic_similarity
        evidence_source = EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA
        breakdown = RankingScoreBreakdown(
            lexical=round(secondary.lexical, 4),
            phrase=round(secondary.phrase, 4),
            query_source=round(secondary.query_source, 4),
            recency=round(secondary.recency, 4),
            category=round(secondary.category, 4),
            semantic_seed=round(semantic_score, 4),
            feedback=round(secondary.feedback, 4),
            total=round(score, 4),
            evidence_score=round(max(best_similarity, 0.0), 4),
            matched_terms=secondary.matched_terms,
            matched_phrases=secondary.matched_phrases,
            semantic_similarities=similarity_details,
            signals=_signals_from_scores(
                semantic_seed=semantic_score,
                lexical=secondary.lexical,
                phrase=secondary.phrase,
                query_source=secondary.query_source,
                recency=secondary.recency,
                category=secondary.category,
                feedback=secondary.feedback,
            ),
        )
        return _SemanticScoredPaper(
            paper=paper,
            score=round(score, 4),
            semantic_similarity=best_similarity,
            semantic_bucket=semantic_bucket,
            secondary_score=secondary.score,
            rationale=_semantic_rationale(
                best_similarity=best_similarity,
                semantic_bucket=semantic_bucket,
                similarity_details=similarity_details,
                secondary=secondary,
                evidence_source=evidence_source,
            ),
            evidence_source=evidence_source,
            breakdown=breakdown,
            qualifies=qualifies,
        )

    def _secondary_signals(
        self,
        paper: PaperMetadata,
        *,
        query_terms: Sequence[str],
        query_phrases: Sequence[str],
        retrieval_query: RetrievalQuery | None,
        retrieval_source_metadata: Sequence[RetrievalSourceMetadata],
        newest_published_date: date | None,
        feedback_events: Sequence[FeedbackEvent],
    ) -> "_SecondarySignals":
        lexical_raw, matched_terms = _lexical_signal(paper, query_terms)
        phrase_raw, matched_phrases = _phrase_signal(paper, query_phrases)
        query_source_raw = _query_source_signal(retrieval_source_metadata)
        recency_raw = _recency_signal(paper.published_date, newest_published_date)
        category_raw = _category_signal(paper, retrieval_query)
        feedback_adjustment = feedback_adjustment_for_paper(
            paper,
            feedback_events,
            vectorizer=self.vectorizer,
            feedback_weight=1.0,
        )
        feedback_score = _bounded(
            feedback_adjustment.score_delta,
            lower=-self.feedback_cap,
            upper=self.feedback_cap,
        )
        lexical = min(lexical_raw, self.lexical_cap)
        phrase = min(phrase_raw, self.phrase_cap)
        query_source = min(query_source_raw, self.query_source_cap)
        recency = min(recency_raw, self.recency_cap)
        category = min(category_raw, self.category_cap)
        return _SecondarySignals(
            lexical=lexical,
            phrase=phrase,
            query_source=query_source,
            recency=recency,
            category=category,
            feedback=feedback_score,
            matched_terms=sorted(set(matched_terms)),
            matched_phrases=sorted(set(matched_phrases)),
            feedback_rationale=feedback_adjustment.rationale,
        )

    def _embed_inputs(
        self,
        inputs: Sequence["_EmbeddingInput"],
        *,
        seed_profile_id: str,
    ) -> tuple[dict[str, list[float]], EmbeddingCacheMetadata]:
        cache_metadata = EmbeddingCacheMetadata(enabled=self.cache_enabled)
        vectors: dict[str, list[float]] = {}
        misses: list[tuple[_EmbeddingInput, Any]] = []

        for item in inputs:
            identity = self._identity_for_input(item, seed_profile_id=seed_profile_id)
            cached = (
                self.store.load_embedding(identity, cache_enabled=self.cache_enabled)
                if self.store is not None
                else None
            )
            if cached is not None:
                cache_metadata.hits += 1
                vectors[item.key] = cached.vector
                continue
            if self.cache_enabled:
                cache_metadata.misses += 1
            else:
                cache_metadata.disabled_requests += 1
            misses.append((item, identity))

        if not misses:
            return vectors, cache_metadata

        provider = self._provider()
        provider_vectors = provider.embed_texts([item.text for item, _identity in misses])
        if len(provider_vectors) != len(misses):
            raise EmbeddingProviderError("embedding provider returned an unexpected vector count.")

        for (item, identity), vector in zip(misses, provider_vectors, strict=True):
            dense_vector = _validate_dense_vector(vector, dimensions=self.dimensions)
            vectors[item.key] = dense_vector
            if self.store is None:
                continue
            saved = self.store.save_embedding(
                identity,
                dense_vector,
                input_role=item.role,
                metadata={
                    "input_version": EMBEDDING_INPUT_VERSION,
                    "item_id": item.item_id,
                },
                cache_enabled=self.cache_enabled,
            )
            if saved is not None:
                cache_metadata.writes += 1

        return vectors, cache_metadata

    def _identity_for_input(
        self,
        item: "_EmbeddingInput",
        *,
        seed_profile_id: str,
    ):
        if self.store is None:
            return None
        profile_id = seed_profile_id if item.cache_scope == EmbeddingCacheScope.PROFILE else None
        return SQLitePaperStore.embedding_identity(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            input_version=EMBEDDING_INPUT_VERSION,
            serialized_input=item.serialized_input,
            cache_scope=item.cache_scope,
            profile_id=profile_id,
        )

    def _provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = create_embedding_provider(self.config)
        return self._embedding_provider

    def _base_metadata(
        self,
        *,
        ranking_mode: str,
        top_k: int,
        seed_preference: SeedPreference | None,
        feedback_events: Sequence[FeedbackEvent],
        cache_metadata: EmbeddingCacheMetadata,
    ) -> dict[str, Any]:
        provider_metadata = EmbeddingProviderCacheMetadata(
            provider=self.provider,
            provider_mode=self.provider_mode,
            provider_label=self.provider_label,
            model=self.model,
            dimensions=self.dimensions,
            cache=cache_metadata,
        ).model_dump(mode="json")
        return {
            "topic": None,
            "top_k": top_k,
            "seed_profile_id": seed_preference.profile_id if seed_preference else None,
            "seed_count": len(seed_preference.seeds) if seed_preference else 0,
            "feedback_count": len(feedback_events),
            "ranking_mode": ranking_mode,
            "semantic_provider": {
                key: value
                for key, value in provider_metadata.items()
                if key != "cache"
            },
            "embedding_cache": provider_metadata["cache"],
            "semantic_context": {
                "input_version": EMBEDDING_INPUT_VERSION,
                "similarity_metric": "cosine",
                "aggregation": "max_per_seed",
                "provider": self.provider,
                "model": self.model,
                "dimensions": self.dimensions,
            },
        }

    def _error_result(
        self,
        *,
        code: str,
        message: str,
        ranking_mode: str,
        top_k: int,
        seed_preference: SeedPreference | None,
        feedback_events: Sequence[FeedbackEvent],
        failure_reason: str,
    ) -> SkillResult[list[Recommendation]]:
        metadata = self._base_metadata(
            ranking_mode=ranking_mode,
            top_k=top_k,
            seed_preference=seed_preference,
            feedback_events=feedback_events,
            cache_metadata=EmbeddingCacheMetadata(enabled=self.cache_enabled),
        )
        metadata.update(
            {
                "score_signals": [],
                "qualifying_count": 0,
                "fallback_count": 0,
                "semantic_error": {
                    "failure_reason": failure_reason,
                    "recommendations_withheld": True,
                },
            }
        )
        return SkillResult[list[Recommendation]](
            status=SkillStatus.ERROR,
            data=[],
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code=code,
                message=message,
                retryable=code != "semantic_seed_quality_error",
            ),
            metadata=metadata,
        )


@dataclass(frozen=True)
class _EmbeddingInput:
    key: str
    item_id: str
    text: str
    serialized_input: dict[str, Any]
    role: EmbeddingInputRole
    cache_scope: EmbeddingCacheScope
    title: str | None = None


@dataclass(frozen=True)
class _SecondarySignals:
    lexical: float
    phrase: float
    query_source: float
    recency: float
    category: float
    feedback: float
    matched_terms: list[str]
    matched_phrases: list[str]
    feedback_rationale: str

    @property
    def score(self) -> float:
        return (
            self.lexical
            + self.phrase
            + self.query_source
            + self.recency
            + self.category
            + self.feedback
        )


@dataclass(frozen=True)
class _SemanticScoredPaper:
    paper: PaperMetadata
    score: float
    semantic_similarity: float
    semantic_bucket: str
    secondary_score: float
    rationale: str
    evidence_source: EvidenceSource
    breakdown: RankingScoreBreakdown
    qualifies: bool


def _provider_name(
    config: AppConfig,
    embedding_provider: EmbeddingProvider | None,
) -> str:
    if embedding_provider is not None:
        class_name = embedding_provider.__class__.__name__.lower()
        if "fake" in class_name:
            return "fake"
    return (config.embedding_provider or "openai").strip().lower() or "openai"


def _seed_embedding_inputs(
    seed_preference: SeedPreference | None,
) -> list[_EmbeddingInput]:
    if seed_preference is None:
        return []
    inputs: list[_EmbeddingInput] = []
    seen: set[str] = set()
    for index, seed in enumerate(seed_preference.seeds):
        payload = _seed_payload(seed)
        if payload is None:
            continue
        item_id = seed.paper_id or seed.identity or f"seed-{index}"
        if item_id in seen:
            continue
        seen.add(item_id)
        inputs.append(
            _EmbeddingInput(
                key=_seed_input_key(item_id),
                item_id=item_id,
                text=_payload_text(payload),
                serialized_input=payload,
                role=EmbeddingInputRole.SEED,
                cache_scope=EmbeddingCacheScope.PROFILE,
                title=payload.get("title"),
            )
        )
    return inputs


def _seed_payload(seed: SeedRecord) -> dict[str, Any] | None:
    paper = seed.paper
    title = paper.title if paper is not None else seed.title
    abstract = paper.abstract if paper is not None else seed.abstract
    categories = paper.categories if paper is not None else []
    if (
        paper is None
        and seed.input_type in {"arxiv_id", "arxiv_url"}
        and not abstract
        and not categories
        and (title or "").strip() == (seed.paper_id or seed.input_text or "").strip()
    ):
        return None
    return _metadata_payload(
        title=title,
        abstract=abstract,
        categories=categories,
    )


def _candidate_embedding_input(paper: PaperMetadata) -> _EmbeddingInput:
    payload = _metadata_payload(
        title=paper.title,
        abstract=paper.abstract,
        categories=paper.categories,
    )
    if payload is None:
        payload = {"title": paper.title}
    return _EmbeddingInput(
        key=_candidate_input_key(paper.paper_id),
        item_id=paper.paper_id,
        text=_payload_text(payload),
        serialized_input=payload,
        role=EmbeddingInputRole.CANDIDATE,
        cache_scope=EmbeddingCacheScope.GLOBAL,
        title=paper.title,
    )


def _metadata_payload(
    *,
    title: str | None,
    abstract: str | None,
    categories: Sequence[str],
) -> dict[str, Any] | None:
    payload = {
        "title": " ".join((title or "").split()),
        "abstract": " ".join((abstract or "").split()),
        "categories": [" ".join(category.split()) for category in categories if category.strip()],
    }
    if not (
        payload["title"]
        or payload["abstract"]
        or payload["categories"]
    ):
        return None
    return payload


def _payload_text(payload: Mapping[str, Any]) -> str:
    parts = [
        str(payload.get("title") or ""),
        str(payload.get("abstract") or ""),
        " ".join(str(category) for category in payload.get("categories") or []),
    ]
    return normalize_provider_input_text(" ".join(part for part in parts if part))


def _seed_paper_ids(seed_preference: SeedPreference | None) -> set[str]:
    if seed_preference is None:
        return set()
    ids: set[str] = set()
    for seed in seed_preference.seeds:
        if seed.paper_id:
            ids.add(seed.paper_id)
        if seed.paper is not None:
            ids.add(seed.paper.paper_id)
    return ids


def _seed_input_key(item_id: str) -> str:
    return f"seed:{item_id}"


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
        raise EmbeddingProviderError("embedding provider returned a vector with invalid dimensions.")
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingProviderError("embedding provider returned a non-finite vector value.")
    return values


def _cosine_dense(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise EmbeddingProviderError("embedding vector dimensions do not match.")
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _semantic_bucket(similarity: float) -> str:
    if similarity >= 0.75:
        return "high"
    if similarity >= 0.5:
        return "medium"
    if similarity >= 0.25:
        return "low"
    return "none"


def _semantic_bucket_rank(bucket: str) -> int:
    return {"high": 3, "medium": 2, "low": 1, "none": 0}.get(bucket, 0)


def _select_semantic_scored(
    scored: Sequence[_SemanticScoredPaper],
    limit: int,
) -> list[_SemanticScoredPaper]:
    if limit <= 0:
        return []

    selected: list[_SemanticScoredPaper] = []
    selected_ids: set[str] = set()
    for item in scored:
        if not item.qualifies:
            continue
        selected.append(item)
        selected_ids.add(item.paper.paper_id)
        if len(selected) >= limit:
            return selected

    for item in scored:
        if item.paper.paper_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.paper.paper_id)
        if len(selected) >= limit:
            break
    return selected


def _recommendation_from_semantic_scored(
    item: _SemanticScoredPaper,
    *,
    rank: int,
    fallback: bool,
) -> Recommendation:
    breakdown = item.breakdown.model_copy(update={"fallback": fallback})
    rationale = item.rationale
    if fallback:
        rationale = (
            "Low-evidence semantic inclusion after successful embedding: "
            f"{rationale}"
        )
    return Recommendation(
        paper=item.paper,
        rank=rank,
        score=item.score,
        rationale=rationale,
        evidence_source=item.evidence_source,
        score_breakdown=breakdown,
    )


def _semantic_rationale(
    *,
    best_similarity: float,
    semantic_bucket: str,
    similarity_details: Sequence[SemanticSimilarityDetail],
    secondary: _SecondarySignals,
    evidence_source: EvidenceSource,
) -> str:
    parts = [
        (
            f"Semantic seed similarity: {best_similarity:.3f} "
            f"({semantic_bucket} evidence)."
        )
    ]
    if similarity_details:
        top_detail = max(similarity_details, key=lambda item: item.similarity)
        parts.append(f"Top matching seed: {top_detail.source_id}.")
    if secondary.matched_terms:
        parts.append(f"Secondary lexical terms: {', '.join(secondary.matched_terms)}.")
    if secondary.matched_phrases:
        parts.append(f"Secondary phrases: {', '.join(secondary.matched_phrases)}.")
    if abs(secondary.query_source) > 0.0001:
        parts.append(f"Query-source/order signal: {secondary.query_source:.3f}.")
    if secondary.category > 0:
        parts.append("Category fit matched the retrieval filter.")
    if secondary.recency > 0:
        parts.append(f"Recency signal: {secondary.recency:.3f}.")
    if secondary.feedback_rationale:
        parts.append(f"Feedback adjustment: {secondary.feedback_rationale}.")
    parts.append(f"Evidence: {evidence_source.value}.")
    return " ".join(parts)


def _bounded(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _signals_from_scores(**scores: float) -> list[str]:
    return [name for name, value in scores.items() if abs(value) > 0.0001]


def _score_signal_names(scored: Sequence[_SemanticScoredPaper]) -> list[str]:
    signals: set[str] = set()
    for item in scored:
        signals.update(item.breakdown.signals)
    return sorted(signals)


def _semantic_ranking_mode(topic: str | None) -> str:
    return (
        SEMANTIC_TOPIC_SEED_RANKING_MODE
        if topic and topic.strip()
        else SEMANTIC_SEED_RANKING_MODE
    )
