"""Agent orchestrator that wires independent Skills into inspectable workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    DailyBriefing,
    EvidenceSource,
    FeedbackEvent,
    PaperBriefingItem,
    PaperMetadata,
    Provenance,
    Recommendation,
    RetrievalQuery,
    SeedPreference,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.skills.briefing import DailyBriefingSkill
from daily_arxiv_agent.skills.extraction import PaperExtractionSkill
from daily_arxiv_agent.skills.feedback import FeedbackInput, FeedbackRefinementSkill
from daily_arxiv_agent.skills.followup import FollowupQuery, FollowupSkill
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


ResultT = TypeVar("ResultT")


class WorkflowTraceStep(BaseModel):
    """One visible Skill call in an Agent workflow."""

    step: int
    skill: str
    status: SkillStatus
    input_summary: str
    output_summary: str
    evidence_source: EvidenceSource | None = None
    fallback: bool = False
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationWorkflow(BaseModel):
    """End-to-end recommendation workflow output."""

    run_id: str
    topic: str
    query: RetrievalQuery
    papers: list[PaperMetadata] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    briefing: DailyBriefing | None = None
    trace: list[WorkflowTraceStep] = Field(default_factory=list)


class FeedbackWorkflow(BaseModel):
    """Feedback refinement workflow output."""

    run_id: str
    profile_id: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    trace: list[WorkflowTraceStep] = Field(default_factory=list)


class FollowupWorkflow(BaseModel):
    """Follow-up query workflow output."""

    run_id: str
    query: FollowupQuery
    papers: list[PaperMetadata] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    trace: list[WorkflowTraceStep] = Field(default_factory=list)


class DailyArxivAgentOrchestrator:
    """Coordinate retrieval, ranking, extraction, briefing, feedback, and follow-up."""

    def __init__(
        self,
        *,
        store: SQLitePaperStore | None = None,
        retrieval_skill: ArxivRetrievalSkill | None = None,
        ranking_skill: TopicRankingSkill | None = None,
        extraction_skill: PaperExtractionSkill | None = None,
        briefing_skill: DailyBriefingSkill | None = None,
        feedback_skill: FeedbackRefinementSkill | None = None,
        followup_skill: FollowupSkill | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.store = store or SQLitePaperStore(AppConfig.from_env().db_path)
        self.retrieval_skill = retrieval_skill or ArxivRetrievalSkill(store=self.store)
        self.ranking_skill = ranking_skill or TopicRankingSkill()
        self.extraction_skill = extraction_skill or PaperExtractionSkill(provider=provider)
        self.briefing_skill = briefing_skill or DailyBriefingSkill(provider=provider)
        self.feedback_skill = feedback_skill or FeedbackRefinementSkill(store=self.store)
        self.followup_skill = followup_skill or FollowupSkill(
            store=self.store,
            retrieval_skill=self.retrieval_skill,
        )

    def run_recommendation(
        self,
        query: RetrievalQuery,
        *,
        topic: str | None = None,
        seed_preference: SeedPreference | None = None,
        profile_id: str = "default",
        top_k: int = 5,
        use_cache: bool = True,
        run_id: str | None = None,
        include_profile_feedback: bool = True,
    ) -> SkillResult[RecommendationWorkflow]:
        """Run retrieval, ranking, extraction, and briefing with trace output."""

        workflow_run_id = run_id or uuid4().hex
        ranking_topic = topic or query.topic
        workflow_topic = ranking_topic or "personalized research"
        trace: list[WorkflowTraceStep] = []

        retrieval_result = _safe_skill_call(
            lambda: self.retrieval_skill.retrieve(query, use_cache=use_cache),
            data_default=[],
            error_code="retrieval_skill_failed",
        )
        papers = retrieval_result.data or []
        _append_trace(
            trace,
            skill="arxiv_retrieval",
            input_summary=_query_summary(query),
            result=retrieval_result,
            output_summary=f"{len(papers)} paper(s) retrieved",
        )

        active_seed = seed_preference or self.store.load_seed_preference(profile_id)
        feedback_events = (
            self.store.list_feedback_events(profile_id=profile_id)
            if include_profile_feedback
            else []
        )
        ranking_result = _safe_skill_call(
            lambda: self.ranking_skill.rank(
                papers,
                topic=ranking_topic,
                seed_preference=active_seed,
                feedback_events=feedback_events,
                top_k=top_k,
            ),
            data_default=[],
            error_code="ranking_skill_failed",
        )
        recommendations = ranking_result.data or []
        _append_trace(
            trace,
            skill="ranking",
            input_summary=(
                f"topic={ranking_topic!r}; seed={active_seed is not None}; "
                f"feedback_events={len(feedback_events)}; top_k={top_k}"
            ),
            result=ranking_result,
            output_summary=f"{len(recommendations)} recommendation(s) ranked",
        )

        extraction_result, item_results = self._extract_recommendations(
            recommendations,
            topic=workflow_topic,
        )
        _append_trace(
            trace,
            skill="extraction",
            input_summary=f"{len(recommendations)} recommendation(s)",
            result=extraction_result,
            output_summary=f"{len(extraction_result.data or [])} briefing item(s) extracted",
        )

        briefing_result = _safe_skill_call(
            lambda: self.briefing_skill.generate(
                topic=workflow_topic,
                recommendations=recommendations,
                extraction_results=item_results,
            ),
            data_default=None,
            error_code="briefing_skill_failed",
        )
        _append_trace(
            trace,
            skill="briefing",
            input_summary=f"topic={workflow_topic!r}; items={len(extraction_result.data or [])}",
            result=briefing_result,
            output_summary=(
                "briefing generated"
                if briefing_result.data is not None
                else "no briefing generated"
            ),
        )

        workflow = RecommendationWorkflow(
            run_id=workflow_run_id,
            topic=workflow_topic,
            query=query,
            papers=papers,
            recommendations=recommendations,
            briefing=briefing_result.data,
            trace=trace,
        )
        results: list[SkillResult[Any]] = [
            retrieval_result,
            ranking_result,
            extraction_result,
            briefing_result,
        ]
        return _workflow_result(
            workflow,
            results=results,
            has_user_data=bool(recommendations or papers),
            evidence_source=briefing_result.evidence_source
            or ranking_result.evidence_source
            or retrieval_result.evidence_source,
        )

    def recommend(self, query: RetrievalQuery, **kwargs: Any) -> SkillResult[RecommendationWorkflow]:
        """Alias for run_recommendation."""

        return self.run_recommendation(query, **kwargs)

    def run_feedback_refinement(
        self,
        recommendations: Sequence[Recommendation],
        *,
        feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]],
        papers: Sequence[PaperMetadata] = (),
        profile_id: str = "default",
        recommendation_run_id: str | None = None,
        top_k: int | None = None,
        run_id: str | None = None,
    ) -> SkillResult[FeedbackWorkflow]:
        """Record feedback and return updated recommendations with trace output."""

        workflow_run_id = run_id or recommendation_run_id or uuid4().hex
        trace: list[WorkflowTraceStep] = []
        feedback_result = _safe_skill_call(
            lambda: self.feedback_skill.refine(
                recommendations,
                feedback=feedback,
                papers=papers,
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id or workflow_run_id,
                top_k=top_k,
            ),
            data_default=[],
            error_code="feedback_refinement_failed",
        )
        refined = feedback_result.data or []
        _append_trace(
            trace,
            skill="feedback_refinement",
            input_summary=(
                f"recommendations={len(recommendations)}; "
                f"feedback_items={len(feedback)}; profile_id={profile_id!r}"
            ),
            result=feedback_result,
            output_summary=f"{len(refined)} refined recommendation(s)",
        )
        workflow = FeedbackWorkflow(
            run_id=workflow_run_id,
            profile_id=profile_id,
            recommendations=refined,
            trace=trace,
        )
        return _workflow_result(
            workflow,
            results=[feedback_result],
            has_user_data=bool(refined),
            evidence_source=feedback_result.evidence_source,
        )

    def refine_feedback(
        self,
        recommendations: Sequence[Recommendation],
        **kwargs: Any,
    ) -> SkillResult[FeedbackWorkflow]:
        """Alias for run_feedback_refinement."""

        return self.run_feedback_refinement(recommendations, **kwargs)

    def run_followup_query(
        self,
        query: FollowupQuery,
        *,
        top_k: int = 5,
        run_id: str | None = None,
    ) -> SkillResult[FollowupWorkflow]:
        """Run a local-first follow-up query and rank matching papers when possible."""

        workflow_run_id = run_id or uuid4().hex
        trace: list[WorkflowTraceStep] = []
        followup_result = _safe_skill_call(
            lambda: self.followup_skill.query(query),
            data_default=[],
            error_code="followup_skill_failed",
        )
        papers = followup_result.data or []
        _append_trace(
            trace,
            skill="followup_filter",
            input_summary=_followup_summary(query),
            result=followup_result,
            output_summary=f"{len(papers)} paper(s) matched",
        )

        results: list[SkillResult[Any]] = [followup_result]
        recommendations: list[Recommendation] = []
        if papers and query.topic:
            ranking_result = _safe_skill_call(
                lambda: self.ranking_skill.rank(
                    papers,
                    topic=query.topic,
                    top_k=top_k,
                ),
                data_default=[],
                error_code="followup_ranking_failed",
            )
            recommendations = ranking_result.data or []
            results.append(ranking_result)
            _append_trace(
                trace,
                skill="ranking",
                input_summary=f"topic={query.topic!r}; top_k={top_k}",
                result=ranking_result,
                output_summary=f"{len(recommendations)} follow-up recommendation(s)",
            )

        workflow = FollowupWorkflow(
            run_id=workflow_run_id,
            query=query,
            papers=papers,
            recommendations=recommendations,
            trace=trace,
        )
        return _workflow_result(
            workflow,
            results=results,
            has_user_data=bool(papers),
            evidence_source=(
                results[-1].evidence_source
                if results[-1].evidence_source is not None
                else followup_result.evidence_source
            ),
        )

    def follow_up(self, query: FollowupQuery, **kwargs: Any) -> SkillResult[FollowupWorkflow]:
        """Alias for run_followup_query."""

        return self.run_followup_query(query, **kwargs)

    def _extract_recommendations(
        self,
        recommendations: Sequence[Recommendation],
        *,
        topic: str,
    ) -> tuple[SkillResult[list[PaperBriefingItem]], list[SkillResult[PaperBriefingItem]]]:
        if not recommendations:
            return (
                SkillResult[list[PaperBriefingItem]](
                    status=SkillStatus.EMPTY,
                    data=[],
                    evidence_source=EvidenceSource.METADATA,
                    message="No recommendations are available for extraction.",
                    metadata={"topic": topic},
                ),
                [],
            )

        item_results = [
            _safe_skill_call(
                lambda recommendation=recommendation: self.extraction_skill.extract(
                    recommendation,
                    topic=topic,
                ),
                data_default=None,
                error_code="extraction_skill_failed",
            )
            for recommendation in recommendations
        ]
        items = [result.data for result in item_results if result.data is not None]
        status = _combined_child_status(item_results, has_user_data=bool(items))
        return (
            SkillResult[list[PaperBriefingItem]](
                status=status,
                data=items,
                evidence_source=_combined_evidence_from_results(item_results),
                provenance=[item.provenance for item in items],
                error=_first_error(item_results) if status == SkillStatus.FALLBACK else None,
                message=(
                    "Extracted briefing items."
                    if status == SkillStatus.SUCCESS
                    else "One or more extraction calls used fallback output."
                    if status == SkillStatus.FALLBACK
                    else "No briefing items were extracted."
                ),
                metadata={"topic": topic, "item_count": len(items)},
            ),
            item_results,
        )


def _safe_skill_call(
    call: Callable[[], SkillResult[ResultT]],
    *,
    data_default: ResultT | None,
    error_code: str,
) -> SkillResult[ResultT]:
    try:
        return call()
    except Exception as exc:  # pragma: no cover - exercised by orchestrator tests.
        return SkillResult[ResultT](
            status=SkillStatus.ERROR,
            data=data_default,
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code=error_code,
                message=f"Skill call failed: {exc}",
                retryable=True,
            ),
        )


def _append_trace(
    trace: list[WorkflowTraceStep],
    *,
    skill: str,
    input_summary: str,
    result: SkillResult[Any],
    output_summary: str,
) -> None:
    trace.append(
        WorkflowTraceStep(
            step=len(trace) + 1,
            skill=skill,
            status=result.status,
            input_summary=input_summary,
            output_summary=output_summary,
            evidence_source=result.evidence_source,
            fallback=result.status in {SkillStatus.FALLBACK, SkillStatus.ERROR},
            error_code=result.error.code if result.error else None,
            error_message=result.error.message if result.error else None,
            metadata=result.metadata,
        )
    )


def _workflow_result(
    workflow: ResultT,
    *,
    results: Sequence[SkillResult[Any]],
    has_user_data: bool,
    evidence_source: EvidenceSource | None,
) -> SkillResult[ResultT]:
    status = _combined_child_status(results, has_user_data=has_user_data)
    error = _first_error(results) if status == SkillStatus.FALLBACK else None
    return SkillResult[ResultT](
        status=status,
        data=workflow,
        evidence_source=evidence_source,
        provenance=_combined_provenance(results),
        error=error,
        message=_workflow_message(status),
    )


def _combined_child_status(
    results: Sequence[SkillResult[Any]],
    *,
    has_user_data: bool,
) -> SkillStatus:
    if any(result.status in {SkillStatus.ERROR, SkillStatus.FALLBACK} for result in results):
        return SkillStatus.FALLBACK
    if not has_user_data:
        return SkillStatus.EMPTY
    return SkillStatus.SUCCESS


def _first_error(results: Sequence[SkillResult[Any]]) -> SkillError:
    for result in results:
        if result.error is not None:
            return result.error
    return SkillError(
        code="workflow_fallback",
        message="Workflow used fallback output.",
        retryable=False,
    )


def _combined_provenance(results: Sequence[SkillResult[Any]]) -> list[Provenance]:
    merged: list[Provenance] = []
    seen: set[str] = set()
    for result in results:
        for provenance in result.provenance:
            key = provenance.model_dump_json()
            if key in seen:
                continue
            seen.add(key)
            merged.append(provenance)
    return merged


def _combined_evidence_from_results(results: Sequence[SkillResult[Any]]) -> EvidenceSource:
    if any(result.evidence_source == EvidenceSource.ABSTRACT for result in results):
        return EvidenceSource.ABSTRACT
    return EvidenceSource.METADATA


def _workflow_message(status: SkillStatus) -> str:
    if status == SkillStatus.SUCCESS:
        return "Workflow completed successfully."
    if status == SkillStatus.EMPTY:
        return "Workflow completed without user-facing results."
    return "Workflow completed with fallback output; inspect trace for details."


def _query_summary(query: RetrievalQuery) -> str:
    return (
        f"topic={query.topic!r}; category={query.category!r}; "
        f"start_date={query.start_date}; end_date={query.end_date}; "
        f"max_results={query.max_results}"
    )


def _followup_summary(query: FollowupQuery) -> str:
    return (
        f"topic={query.topic!r}; category={query.category!r}; "
        f"start_date={query.start_date}; end_date={query.end_date}; "
        f"max_results={query.max_results}; fetch_if_empty={query.fetch_if_empty}"
    )
