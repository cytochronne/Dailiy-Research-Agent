"""Deterministic evaluation helpers for course demo artifacts.

These helpers intentionally stay small: they summarize fixed recommendation,
feedback, and explanation outputs without introducing a benchmark framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    PaperDeepExplanation,
    PaperMetadata,
    Recommendation,
    SkillError,
    SkillResult,
    SkillStatus,
)


class RecommendationEvaluation(BaseModel):
    """Overlap metrics for a ranked recommendation list."""

    expected_relevant_ids: list[str] = Field(default_factory=list)
    evaluated_paper_ids: list[str] = Field(default_factory=list)
    matched_paper_ids: list[str] = Field(default_factory=list)
    missing_relevant_ids: list[str] = Field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mean_reciprocal_rank: float = 0.0
    zero_data_reason: str | None = None


class SearchQualityEvaluation(BaseModel):
    """Offline search-quality metrics for candidate retrieval and Top-K ranking."""

    expected_relevant_ids: list[str] = Field(default_factory=list)
    candidate_paper_ids: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    relevant_candidate_ids: list[str] = Field(default_factory=list)
    missing_candidate_relevant_ids: list[str] = Field(default_factory=list)
    relevant_candidate_coverage: float = 0.0
    top_k_paper_ids: list[str] = Field(default_factory=list)
    matched_top_k_ids: list[str] = Field(default_factory=list)
    missing_top_k_relevant_ids: list[str] = Field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mean_reciprocal_rank: float = 0.0
    rationale_covered_ids: list[str] = Field(default_factory=list)
    missing_rationale_ids: list[str] = Field(default_factory=list)
    rationale_coverage: float = 0.0
    budget_exhausted: bool | None = None
    zero_data_reason: str | None = None


class FeedbackRankMovement(BaseModel):
    """Before/after rank and score movement for one paper."""

    paper_id: str
    previous_rank: int | None = None
    current_rank: int | None = None
    previous_score: float | None = None
    current_score: float | None = None
    rank_delta: int | None = None
    score_delta: float | None = None
    movement: Literal["up", "down", "unchanged", "new", "removed"]
    feedback_value: Literal["like", "dislike"] | None = None


class FeedbackMovementEvaluation(BaseModel):
    """Summary of recommendation movement after feedback refinement."""

    movements: list[FeedbackRankMovement] = Field(default_factory=list)
    moved_up_ids: list[str] = Field(default_factory=list)
    moved_down_ids: list[str] = Field(default_factory=list)
    unchanged_ids: list[str] = Field(default_factory=list)
    new_ids: list[str] = Field(default_factory=list)
    removed_ids: list[str] = Field(default_factory=list)
    liked_paper_ids: list[str] = Field(default_factory=list)
    disliked_paper_ids: list[str] = Field(default_factory=list)
    movement_count: int = 0
    zero_data_reason: str | None = None


class ExplanationCompleteness(BaseModel):
    """Required-section coverage for one selected-paper explanation."""

    paper_id: str
    mode: ExplanationMode
    evidence_source: EvidenceSource
    required_sections: list[str] = Field(default_factory=list)
    present_sections: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    completeness_score: float = 0.0
    is_complete: bool = False


class RecommendationEvaluationFixture(BaseModel):
    """Fixture shape used by docs or tests to evaluate ranked outputs."""

    model_config = ConfigDict(extra="forbid")

    recommendations: list[dict[str, Any]]
    expected_relevant_paper_ids: list[str]
    k: int | None = Field(default=None, ge=1)

    @field_validator("expected_relevant_paper_ids")
    @classmethod
    def require_expected_ids(cls, value: list[str]) -> list[str]:
        normalized = _dedupe_nonblank(value, label="expected_relevant_paper_ids")
        if not normalized:
            raise ValueError("expected_relevant_paper_ids must include at least one ID")
        return normalized


@dataclass(frozen=True)
class _RecommendationRef:
    paper_id: str
    rank: int
    score: float | None = None


_DEFAULT_REQUIRED_SECTIONS: dict[ExplanationMode, tuple[str, ...]] = {
    ExplanationMode.METHOD: (
        "summary",
        "evidence_note",
        "method.problem",
        "method.method_overview",
        "method.core_workflow",
        "method.inputs_outputs",
        "method.innovation",
    ),
    ExplanationMode.EXPERIMENT: (
        "summary",
        "evidence_note",
        "experiment.datasets",
        "experiment.baselines",
        "experiment.metrics",
        "experiment.experimental_setup",
        "experiment.conclusions",
    ),
    ExplanationMode.LIMITATIONS: (
        "summary",
        "evidence_note",
        "limitations.stated_limitations",
        "limitations.assumptions",
        "limitations.missing_validation",
        "limitations.risks",
    ),
}


def evaluate_recommendations(
    recommendations: Sequence[Recommendation | Mapping[str, Any]],
    expected_relevant_paper_ids: Sequence[str],
    *,
    k: int | None = None,
) -> SkillResult[RecommendationEvaluation]:
    """Compare ranked recommendations against expected relevant paper IDs."""

    try:
        expected_ids = _normalize_expected_ids(expected_relevant_paper_ids)
        refs = _normalize_recommendations(recommendations)
        limit = _normalize_k(k)
    except ValueError as exc:
        return _validation_error(str(exc))

    evaluated = refs if limit is None else refs[:limit]
    evaluated_ids = [item.paper_id for item in evaluated]
    expected_set = set(expected_ids)
    matched_ids = [
        paper_id
        for paper_id in evaluated_ids
        if paper_id in expected_set
    ]
    missing_ids = [paper_id for paper_id in expected_ids if paper_id not in matched_ids]
    precision = len(matched_ids) / len(evaluated) if evaluated else 0.0
    recall = len(matched_ids) / len(expected_ids) if expected_ids else 0.0
    first_relevant_position = next(
        (
            position
            for position, paper_id in enumerate(evaluated_ids, start=1)
            if paper_id in expected_set
        ),
        None,
    )
    mrr = 1.0 / first_relevant_position if first_relevant_position else 0.0
    zero_data_reason = (
        "No recommendations were supplied for evaluation." if not evaluated else None
    )
    data = RecommendationEvaluation(
        expected_relevant_ids=expected_ids,
        evaluated_paper_ids=evaluated_ids,
        matched_paper_ids=matched_ids,
        missing_relevant_ids=missing_ids,
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        mean_reciprocal_rank=round(mrr, 4),
        zero_data_reason=zero_data_reason,
    )
    return SkillResult[RecommendationEvaluation](
        status=SkillStatus.EMPTY if not evaluated else SkillStatus.SUCCESS,
        data=data,
        evidence_source=EvidenceSource.METADATA,
        message=zero_data_reason or "Evaluated recommendation overlap.",
        metadata={
            "expected_count": len(expected_ids),
            "recommendation_count": len(refs),
            "evaluated_count": len(evaluated),
            "k": limit,
        },
    )


def evaluate_recommendation_fixture(
    fixture: Mapping[str, Any],
) -> SkillResult[RecommendationEvaluation]:
    """Validate and evaluate a dictionary-backed recommendation fixture."""

    try:
        parsed = RecommendationEvaluationFixture.model_validate(fixture)
    except ValidationError as exc:
        return _validation_error(
            _format_validation_error(exc),
            code="evaluation_fixture_invalid",
        )
    result = evaluate_recommendations(
        parsed.recommendations,
        parsed.expected_relevant_paper_ids,
        k=parsed.k,
    )
    if result.status == SkillStatus.ERROR:
        return SkillResult[RecommendationEvaluation](
            status=SkillStatus.ERROR,
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code="evaluation_fixture_invalid",
                message=result.error.message if result.error else "Invalid fixture.",
                retryable=False,
            ),
            metadata={"fixture_keys": sorted(fixture.keys())},
        )
    result.metadata["fixture_keys"] = sorted(fixture.keys())
    return result


def evaluate_search_quality(
    candidates: Sequence[PaperMetadata | Mapping[str, Any] | str],
    recommendations: Sequence[Recommendation | Mapping[str, Any]],
    expected_relevant_paper_ids: Sequence[str],
    *,
    k: int | None = None,
    retrieval_metadata: Mapping[str, Any] | None = None,
) -> SkillResult[SearchQualityEvaluation]:
    """Evaluate offline search recall, Top-K quality, and rationale coverage."""

    try:
        expected_ids = _normalize_expected_ids(expected_relevant_paper_ids)
        candidate_ids = _normalize_candidate_ids(candidates)
        refs = _normalize_recommendations(recommendations)
        limit = _normalize_k(k)
    except ValueError as exc:
        return _validation_error(str(exc))

    expected_set = set(expected_ids)
    candidate_id_set = set(candidate_ids)
    relevant_candidate_ids = [
        paper_id for paper_id in expected_ids if paper_id in candidate_id_set
    ]
    missing_candidate_ids = [
        paper_id for paper_id in expected_ids if paper_id not in relevant_candidate_ids
    ]

    evaluated_refs = refs if limit is None else refs[:limit]
    top_k_ids = [item.paper_id for item in evaluated_refs]
    matched_top_k_ids = [
        paper_id for paper_id in top_k_ids if paper_id in expected_set
    ]
    missing_top_k_ids = [
        paper_id for paper_id in expected_ids if paper_id not in matched_top_k_ids
    ]
    first_relevant_position = next(
        (
            position
            for position, paper_id in enumerate(top_k_ids, start=1)
            if paper_id in expected_set
        ),
        None,
    )
    rationale_ids = _recommendation_ids_with_rationale(recommendations)
    rationale_covered_ids = [
        paper_id for paper_id in top_k_ids if paper_id in rationale_ids
    ]
    missing_rationale_ids = [
        paper_id for paper_id in top_k_ids if paper_id not in rationale_ids
    ]
    zero_data_reason = (
        "No candidates or recommendations were supplied for search-quality evaluation."
        if not candidate_ids and not top_k_ids
        else None
    )

    data = SearchQualityEvaluation(
        expected_relevant_ids=expected_ids,
        candidate_paper_ids=candidate_ids,
        candidate_count=len(candidate_ids),
        relevant_candidate_ids=relevant_candidate_ids,
        missing_candidate_relevant_ids=missing_candidate_ids,
        relevant_candidate_coverage=round(
            len(relevant_candidate_ids) / len(expected_ids),
            4,
        ),
        top_k_paper_ids=top_k_ids,
        matched_top_k_ids=matched_top_k_ids,
        missing_top_k_relevant_ids=missing_top_k_ids,
        precision_at_k=round(
            len(matched_top_k_ids) / len(top_k_ids) if top_k_ids else 0.0,
            4,
        ),
        recall_at_k=round(len(matched_top_k_ids) / len(expected_ids), 4),
        mean_reciprocal_rank=round(
            1.0 / first_relevant_position if first_relevant_position else 0.0,
            4,
        ),
        rationale_covered_ids=rationale_covered_ids,
        missing_rationale_ids=missing_rationale_ids,
        rationale_coverage=round(
            len(rationale_covered_ids) / len(top_k_ids) if top_k_ids else 0.0,
            4,
        ),
        budget_exhausted=_budget_exhausted_from_metadata(retrieval_metadata),
        zero_data_reason=zero_data_reason,
    )
    return SkillResult[SearchQualityEvaluation](
        status=SkillStatus.EMPTY if zero_data_reason else SkillStatus.SUCCESS,
        data=data,
        evidence_source=EvidenceSource.METADATA,
        message=zero_data_reason or "Evaluated offline search quality.",
        metadata={
            "expected_count": len(expected_ids),
            "candidate_count": data.candidate_count,
            "top_k_count": len(top_k_ids),
            "k": limit,
            "budget_exhausted": data.budget_exhausted,
        },
    )


def evaluate_feedback_movement(
    before: Sequence[Recommendation | Mapping[str, Any]],
    after: Sequence[Recommendation | Mapping[str, Any]],
    *,
    liked_paper_ids: Sequence[str] = (),
    disliked_paper_ids: Sequence[str] = (),
) -> SkillResult[FeedbackMovementEvaluation]:
    """Compare recommendation ranks and scores before and after feedback."""

    try:
        before_refs = _normalize_recommendations(before)
        after_refs = _normalize_recommendations(after)
        liked_ids = _dedupe_nonblank(liked_paper_ids, label="liked_paper_ids")
        disliked_ids = _dedupe_nonblank(disliked_paper_ids, label="disliked_paper_ids")
    except ValueError as exc:
        return _validation_error(str(exc))

    before_by_id = {item.paper_id: item for item in before_refs}
    after_by_id = {item.paper_id: item for item in after_refs}
    ordered_ids = [item.paper_id for item in after_refs]
    ordered_ids.extend(
        item.paper_id for item in before_refs if item.paper_id not in after_by_id
    )

    movements: list[FeedbackRankMovement] = []
    for paper_id in ordered_ids:
        previous = before_by_id.get(paper_id)
        current = after_by_id.get(paper_id)
        feedback_value = _feedback_value_for(paper_id, liked_ids, disliked_ids)
        if previous is None and current is not None:
            movements.append(
                FeedbackRankMovement(
                    paper_id=paper_id,
                    current_rank=current.rank,
                    current_score=current.score,
                    movement="new",
                    feedback_value=feedback_value,
                )
            )
            continue
        if current is None and previous is not None:
            movements.append(
                FeedbackRankMovement(
                    paper_id=paper_id,
                    previous_rank=previous.rank,
                    previous_score=previous.score,
                    movement="removed",
                    feedback_value=feedback_value,
                )
            )
            continue
        if previous is None or current is None:
            continue

        rank_delta = previous.rank - current.rank
        score_delta = (
            round(current.score - previous.score, 4)
            if current.score is not None and previous.score is not None
            else None
        )
        movement: Literal["up", "down", "unchanged"]
        if rank_delta > 0:
            movement = "up"
        elif rank_delta < 0:
            movement = "down"
        else:
            movement = "unchanged"
        movements.append(
            FeedbackRankMovement(
                paper_id=paper_id,
                previous_rank=previous.rank,
                current_rank=current.rank,
                previous_score=previous.score,
                current_score=current.score,
                rank_delta=rank_delta,
                score_delta=score_delta,
                movement=movement,
                feedback_value=feedback_value,
            )
        )

    zero_data_reason = (
        "No before or after recommendations were supplied for feedback evaluation."
        if not before_refs and not after_refs
        else None
    )
    data = FeedbackMovementEvaluation(
        movements=movements,
        moved_up_ids=[item.paper_id for item in movements if item.movement == "up"],
        moved_down_ids=[item.paper_id for item in movements if item.movement == "down"],
        unchanged_ids=[
            item.paper_id for item in movements if item.movement == "unchanged"
        ],
        new_ids=[item.paper_id for item in movements if item.movement == "new"],
        removed_ids=[item.paper_id for item in movements if item.movement == "removed"],
        liked_paper_ids=liked_ids,
        disliked_paper_ids=disliked_ids,
        movement_count=sum(
            1 for item in movements if item.movement in {"up", "down", "new", "removed"}
        ),
        zero_data_reason=zero_data_reason,
    )
    return SkillResult[FeedbackMovementEvaluation](
        status=SkillStatus.EMPTY if zero_data_reason else SkillStatus.SUCCESS,
        data=data,
        evidence_source=EvidenceSource.METADATA,
        message=zero_data_reason or "Evaluated feedback rank movement.",
        metadata={
            "before_count": len(before_refs),
            "after_count": len(after_refs),
            "movement_count": data.movement_count,
        },
    )


def check_explanation_completeness(
    explanation: PaperDeepExplanation | Mapping[str, Any],
    *,
    required_sections: Sequence[str] | None = None,
) -> SkillResult[ExplanationCompleteness]:
    """Report which required explanation sections are present or missing."""

    try:
        normalized = (
            explanation
            if isinstance(explanation, PaperDeepExplanation)
            else PaperDeepExplanation.model_validate(explanation)
        )
    except ValidationError as exc:
        return _validation_error(
            _format_validation_error(exc),
            code="evaluation_fixture_invalid",
        )
    if required_sections is None:
        try:
            required = list(_DEFAULT_REQUIRED_SECTIONS[normalized.mode])
        except KeyError:
            return _validation_error(f"Unsupported explanation mode: {normalized.mode!r}")
    else:
        required = list(required_sections)
    if not required:
        return _validation_error("required_sections must include at least one section")

    present = [
        section for section in required if _has_present_value(normalized, section)
    ]
    missing = [section for section in required if section not in present]
    score = len(present) / len(required) if required else 0.0
    data = ExplanationCompleteness(
        paper_id=normalized.paper_id,
        mode=normalized.mode,
        evidence_source=normalized.evidence_source,
        required_sections=required,
        present_sections=present,
        missing_sections=missing,
        completeness_score=round(score, 4),
        is_complete=not missing,
    )
    return SkillResult[ExplanationCompleteness](
        status=SkillStatus.SUCCESS,
        data=data,
        evidence_source=normalized.evidence_source,
        message=(
            "All required explanation sections are present."
            if not missing
            else f"Missing {len(missing)} of {len(required)} required explanation sections."
        ),
        provenance=[normalized.provenance],
        metadata={
            "paper_id": normalized.paper_id,
            "mode": normalized.mode.value,
            "missing_count": len(missing),
        },
    )


def _normalize_expected_ids(values: Sequence[str]) -> list[str]:
    expected = _dedupe_nonblank(values, label="expected_relevant_paper_ids")
    if not expected:
        raise ValueError("expected_relevant_paper_ids must include at least one ID")
    return expected


def _normalize_recommendations(
    recommendations: Sequence[Recommendation | Mapping[str, Any]],
) -> list[_RecommendationRef]:
    refs: list[_RecommendationRef] = []
    seen: set[str] = set()
    for position, recommendation in enumerate(recommendations, start=1):
        paper_id = _paper_id_from_recommendation(recommendation)
        if not paper_id:
            raise ValueError(
                f"recommendations[{position - 1}] must include a non-empty paper_id"
            )
        if paper_id in seen:
            raise ValueError(f"Duplicate recommendation paper_id: {paper_id}")
        seen.add(paper_id)
        refs.append(
            _RecommendationRef(
                paper_id=paper_id,
                rank=_rank_from_recommendation(recommendation, position),
                score=_score_from_recommendation(recommendation),
            )
        )
    return sorted(refs, key=lambda item: item.rank)


def _normalize_candidate_ids(
    candidates: Sequence[PaperMetadata | Mapping[str, Any] | str],
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for position, candidate in enumerate(candidates, start=1):
        paper_id = _paper_id_from_candidate(candidate)
        if not paper_id:
            raise ValueError(
                f"candidates[{position - 1}] must include a non-empty paper_id"
            )
        if paper_id in seen:
            continue
        ids.append(paper_id)
        seen.add(paper_id)
    return ids


def _paper_id_from_recommendation(
    recommendation: Recommendation | Mapping[str, Any],
) -> str:
    if isinstance(recommendation, Recommendation):
        return recommendation.paper.paper_id.strip()
    raw_id = recommendation.get("paper_id")
    if raw_id is None:
        paper = recommendation.get("paper")
        if isinstance(paper, Mapping):
            raw_id = paper.get("paper_id")
        elif hasattr(paper, "paper_id"):
            raw_id = getattr(paper, "paper_id")
    return str(raw_id).strip() if raw_id is not None else ""


def _paper_id_from_candidate(candidate: PaperMetadata | Mapping[str, Any] | str) -> str:
    if isinstance(candidate, str):
        return candidate.strip()
    if isinstance(candidate, PaperMetadata):
        return candidate.paper_id.strip()
    raw_id = candidate.get("paper_id")
    if raw_id is None:
        paper = candidate.get("paper")
        if isinstance(paper, Mapping):
            raw_id = paper.get("paper_id")
        elif hasattr(paper, "paper_id"):
            raw_id = getattr(paper, "paper_id")
    return str(raw_id).strip() if raw_id is not None else ""


def _rank_from_recommendation(
    recommendation: Recommendation | Mapping[str, Any],
    position: int,
) -> int:
    raw_rank = (
        recommendation.rank
        if isinstance(recommendation, Recommendation)
        else recommendation.get("rank")
    )
    if raw_rank is None:
        return position
    try:
        rank = int(raw_rank)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rank must be an integer for recommendation {position}") from exc
    if rank < 1:
        raise ValueError(f"rank must be positive for recommendation {position}")
    return rank


def _score_from_recommendation(
    recommendation: Recommendation | Mapping[str, Any],
) -> float | None:
    raw_score = (
        recommendation.score
        if isinstance(recommendation, Recommendation)
        else recommendation.get("score")
    )
    if raw_score is None:
        return None
    try:
        return float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("score must be numeric when supplied") from exc


def _recommendation_ids_with_rationale(
    recommendations: Sequence[Recommendation | Mapping[str, Any]],
) -> set[str]:
    ids: set[str] = set()
    for recommendation in recommendations:
        paper_id = _paper_id_from_recommendation(recommendation)
        rationale = _rationale_from_recommendation(recommendation)
        if paper_id and rationale:
            ids.add(paper_id)
    return ids


def _rationale_from_recommendation(
    recommendation: Recommendation | Mapping[str, Any],
) -> str:
    raw_rationale = (
        recommendation.rationale
        if isinstance(recommendation, Recommendation)
        else recommendation.get("rationale")
    )
    return " ".join(str(raw_rationale).split()) if raw_rationale is not None else ""


def _budget_exhausted_from_metadata(
    retrieval_metadata: Mapping[str, Any] | None,
) -> bool | None:
    if retrieval_metadata is None or "budget_exhausted" not in retrieval_metadata:
        return None
    value = retrieval_metadata["budget_exhausted"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return bool(value)


def _normalize_k(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError("k must be at least 1")
    return value


def _dedupe_nonblank(values: Sequence[str], *, label: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{label} must contain string IDs")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label} must not contain blank IDs")
        if stripped not in seen:
            normalized.append(stripped)
            seen.add(stripped)
    return normalized


def _feedback_value_for(
    paper_id: str,
    liked_paper_ids: Sequence[str],
    disliked_paper_ids: Sequence[str],
) -> Literal["like", "dislike"] | None:
    if paper_id in liked_paper_ids:
        return "like"
    if paper_id in disliked_paper_ids:
        return "dislike"
    return None


def _has_present_value(root: BaseModel | Mapping[str, Any], path: str) -> bool:
    current: Any = root
    for part in path.split("."):
        if isinstance(current, BaseModel):
            current = getattr(current, part, None)
        elif isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = None
        if current is None:
            return False
    return _is_present(current)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        return not _is_missing_evidence_placeholder(stripped)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_is_present(item) for item in value)
    if isinstance(value, Mapping):
        return any(_is_present(item) for item in value.values())
    return True


def _is_missing_evidence_placeholder(value: str) -> bool:
    normalized = " ".join(value.lower().split())
    return (
        "was not found in the available" in normalized
        and normalized.endswith("source.")
    )


def _validation_error(
    message: str,
    *,
    code: str = "evaluation_input_invalid",
) -> SkillResult[Any]:
    return SkillResult[Any](
        status=SkillStatus.ERROR,
        evidence_source=EvidenceSource.METADATA,
        error=SkillError(code=code, message=message, retryable=False),
        metadata={"validation_error": True},
    )


def _format_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "fixture"
    return f"{location}: {first.get('msg', 'Invalid evaluation fixture.')}"
