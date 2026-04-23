"""Feedback recording and explainable recommendation refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, ValidationError, model_validator

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    FeedbackEvent,
    FeedbackValue,
    PaperMetadata,
    Recommendation,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
    build_paper_preference_text,
    cosine_similarity,
)
from daily_arxiv_agent.storage import SQLitePaperStore


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


class FeedbackRefinementSkill:
    """Record like/dislike feedback and return refined recommendations."""

    def __init__(
        self,
        *,
        store: SQLitePaperStore | None = None,
        vectorizer: DeterministicTextVectorizer | None = None,
        feedback_weight: float = 6.0,
    ) -> None:
        self.store = store
        self.vectorizer = vectorizer or DeterministicTextVectorizer()
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
        feedback_events = _hydrate_events(feedback_events, paper_lookup)
        active_events = latest_feedback_events(feedback_events)

        rescored: list[tuple[Recommendation, float, FeedbackAdjustment]] = []
        for recommendation in recommendations:
            adjustment = feedback_adjustment_for_paper(
                recommendation.paper,
                active_events.values(),
                vectorizer=self.vectorizer,
                feedback_weight=self.feedback_weight,
            )
            new_score = round(recommendation.score + adjustment.score_delta, 4)
            rescored.append((recommendation, new_score, adjustment))

        rescored.sort(
            key=lambda item: (-item[1], item[0].rank, item[0].paper.title.lower())
        )
        limit = len(rescored) if top_k is None else max(top_k, 0)
        refined: list[Recommendation] = []
        for rank, (recommendation, new_score, adjustment) in enumerate(
            rescored[:limit],
            start=1,
        ):
            score_delta = round(new_score - recommendation.score, 4)
            refined.append(
                Recommendation(
                    paper=recommendation.paper,
                    rank=rank,
                    score=new_score,
                    rationale=_refined_rationale(
                        recommendation.rationale,
                        recommendation.rank,
                        score_delta,
                        adjustment,
                    ),
                    evidence_source=recommendation.evidence_source,
                    previous_rank=recommendation.rank,
                    previous_score=recommendation.score,
                    score_delta=score_delta,
                    rank_delta=recommendation.rank - rank,
                )
            )

        return SkillResult[list[Recommendation]](
            status=SkillStatus.SUCCESS,
            data=refined,
            evidence_source=(
                EvidenceSource.ABSTRACT
                if any(item.evidence_source == EvidenceSource.ABSTRACT for item in refined)
                else EvidenceSource.METADATA
            ),
            provenance=[item.paper.provenance for item in refined],
            message="Refined recommendations using latest paper feedback.",
            metadata={
                "profile_id": profile_id,
                "recommendation_run_id": recommendation_run_id,
                "feedback_count": len(feedback_events),
                "active_feedback_count": len(active_events),
                "feedback_rule": "latest_wins",
            },
        )


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

    return FeedbackAdjustment(
        score_delta=round(score_delta, 4),
        rationale="; ".join(rationale_parts),
        matched_event_count=matched_event_count,
    )


def _normalize_feedback_inputs(
    feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]],
    *,
    paper_lookup: dict[str, PaperMetadata],
    profile_id: str,
    recommendation_run_id: str | None,
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
                    paper=paper_lookup.get(feedback_input.paper_id),
                    note=feedback_input.note,
                )

            if event.paper is None and event.paper_id in paper_lookup:
                event = event.model_copy(update={"paper": paper_lookup[event.paper_id]})
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
) -> list[FeedbackEvent]:
    hydrated: list[FeedbackEvent] = []
    for event in events:
        if event.paper is None and event.paper_id in paper_lookup:
            hydrated.append(event.model_copy(update={"paper": paper_lookup[event.paper_id]}))
            continue
        hydrated.append(event)
    return hydrated


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
