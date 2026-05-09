"""Agent orchestrator that wires independent Skills into inspectable workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from inspect import Parameter, signature
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    DailyBriefing,
    EvidenceSource,
    ExplanationMode,
    FeedbackEvent,
    PaperBriefingItem,
    PaperDeepExplanation,
    PaperMetadata,
    Provenance,
    QueryPlan,
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
from daily_arxiv_agent.skills.deep_explanation import PaperDeepExplanationSkill
from daily_arxiv_agent.skills.discovery_recommendation import DiscoveryRecommendationSkill
from daily_arxiv_agent.skills.extraction import PaperExtractionSkill
from daily_arxiv_agent.skills.feedback import FeedbackInput, FeedbackRefinementSkill
from daily_arxiv_agent.skills.followup import FollowupQuery, FollowupSkill
from daily_arxiv_agent.skills.query_planning import (
    SEED_DERIVED_PLANNER_SOURCE,
    QueryPlanningSkill,
    build_deterministic_query_plan,
    build_seed_derived_query_plan,
)
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.research_synthesis import ResearchSynthesisSkill
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


ResultT = TypeVar("ResultT")
RECOMMENDATION_MODE_AUTO = "auto"
RECOMMENDATION_MODE_DETERMINISTIC = "deterministic"
RECOMMENDATION_MODE_SEMANTIC_SEED = "semantic_seed"
RECOMMENDATION_MODES = {
    RECOMMENDATION_MODE_AUTO,
    RECOMMENDATION_MODE_DETERMINISTIC,
    RECOMMENDATION_MODE_SEMANTIC_SEED,
}


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


class PaperExplanationWorkflow(BaseModel):
    """Selected-paper explanation workflow output."""

    run_id: str
    paper_id: str
    mode: ExplanationMode
    paper: PaperMetadata | None = None
    explanation: PaperDeepExplanation | None = None
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
        query_planning_skill: QueryPlanningSkill | None = None,
        semantic_ranking_skill: SemanticSeedRankingSkill | None = None,
        deep_explanation_skill: PaperDeepExplanationSkill | None = None,
        discovery_skill: DiscoveryRecommendationSkill | None = None,
        synthesis_skill: ResearchSynthesisSkill | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        config = AppConfig.from_env()
        self.store = (
            store
            or (discovery_skill.store if discovery_skill is not None else None)
            or SQLitePaperStore(config.db_path)
        )

        self.discovery_skill = discovery_skill or DiscoveryRecommendationSkill(
            store=self.store,
            provider=provider,
            config=config,
            query_planning_skill=query_planning_skill,
            retrieval_skill=retrieval_skill,
            ranking_skill=ranking_skill,
            semantic_ranking_skill=semantic_ranking_skill,
            feedback_skill=feedback_skill,
            followup_skill=followup_skill,
        )
        self.synthesis_skill = synthesis_skill or ResearchSynthesisSkill(
            provider=provider,
            store=self.store,
            extraction_skill=extraction_skill,
            briefing_skill=briefing_skill,
            deep_explanation_skill=deep_explanation_skill,
        )

        # Backwards-compatible aliases for tests and external callers that inject
        # or inspect the original single-purpose Skill instances.
        self.query_planning_skill = self.discovery_skill.query_planning_skill
        self.retrieval_skill = self.discovery_skill.retrieval_skill
        self.ranking_skill = self.discovery_skill.ranking_skill
        self.semantic_ranking_skill = self.discovery_skill.semantic_ranking_skill
        self.feedback_skill = self.discovery_skill.feedback_skill
        self.followup_skill = self.discovery_skill.followup_skill
        self.extraction_skill = self.synthesis_skill.extraction_skill
        self.briefing_skill = self.synthesis_skill.briefing_skill
        self.deep_explanation_skill = self.synthesis_skill.deep_explanation_skill

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
        include_debug_trace: bool = False,
        recommendation_mode: str = RECOMMENDATION_MODE_AUTO,
    ) -> SkillResult[RecommendationWorkflow]:
        """Run retrieval, ranking, extraction, and briefing with trace output."""

        workflow_run_id = run_id or uuid4().hex
        ranking_topic = topic or query.topic
        workflow_topic = ranking_topic or "personalized research"
        trace: list[WorkflowTraceStep] = []
        active_seed = seed_preference or self.store.load_seed_preference(profile_id)
        resolved_recommendation_mode = _normalize_recommendation_mode(
            recommendation_mode
        )
        use_seed_derived_plan = _should_use_seed_derived_query_plan(
            query,
            ranking_topic,
            active_seed,
            recommendation_mode=resolved_recommendation_mode,
        )
        use_semantic_ranking = _should_use_semantic_ranking(
            query,
            ranking_topic,
            active_seed,
            recommendation_mode=resolved_recommendation_mode,
        )

        raw_planning_result = _safe_skill_call(
            lambda: _plan_query(
                self.query_planning_skill,
                query,
                seed_preference=active_seed,
                use_seed_derived_plan=use_seed_derived_plan,
            ),
            data_default=None,
            error_code="query_planning_skill_failed",
        )
        planning_result = (
            raw_planning_result
            if use_seed_derived_plan
            else _query_plan_or_fallback(query, raw_planning_result)
        )
        _append_trace(
            trace,
            skill="query_planning",
            input_summary=_query_summary(query),
            result=planning_result,
            output_summary=_query_plan_output_summary(planning_result),
            metadata=_trace_metadata(
                "query_planning",
                _metadata_with_recommendation_mode(
                    planning_result.metadata,
                    resolved_recommendation_mode,
                ),
                include_debug=include_debug_trace,
            ),
        )
        query_plan = planning_result.data
        if query_plan is None:
            workflow = RecommendationWorkflow(
                run_id=workflow_run_id,
                topic=workflow_topic,
                query=query,
                trace=trace,
            )
            return _workflow_result(
                workflow,
                results=[planning_result],
                has_user_data=False,
                evidence_source=planning_result.evidence_source,
            )

        semantic_readiness_result: SkillResult[dict[str, Any]] | None = None
        if use_semantic_ranking:
            semantic_readiness_result = _semantic_readiness_result(
                self.semantic_ranking_skill.check_readiness(active_seed)
            )
            _append_trace(
                trace,
                skill="semantic_readiness",
                input_summary=(
                    f"seed={active_seed is not None}; "
                    f"seed_count={len(active_seed.seeds) if active_seed else 0}"
                ),
                result=semantic_readiness_result,
                output_summary=(
                    "semantic recommendation ready"
                    if semantic_readiness_result.status == SkillStatus.SUCCESS
                    else "semantic recommendation not ready"
                ),
                metadata=_trace_metadata(
                    "semantic_readiness",
                    _metadata_with_recommendation_mode(
                        semantic_readiness_result.metadata,
                        resolved_recommendation_mode,
                    ),
                    include_debug=include_debug_trace,
                ),
            )
            if semantic_readiness_result.status == SkillStatus.ERROR:
                workflow = RecommendationWorkflow(
                    run_id=workflow_run_id,
                    topic=workflow_topic,
                    query=query,
                    trace=trace,
                )
                return _workflow_result(
                    workflow,
                    results=[planning_result, semantic_readiness_result],
                    has_user_data=False,
                    evidence_source=semantic_readiness_result.evidence_source,
                )

        retrieval_result = _safe_skill_call(
            lambda: _retrieve_with_optional_query_plan(
                self.retrieval_skill,
                query,
                use_cache=use_cache,
                query_plan=query_plan,
            ),
            data_default=[],
            error_code="retrieval_skill_failed",
        )
        retrieval_result = _prepare_seed_candidates(
            retrieval_result,
            query=query,
            seed_preference=active_seed,
            top_k=top_k,
            seed_derived=use_seed_derived_plan,
        )
        papers = retrieval_result.data or []
        _append_trace(
            trace,
            skill="arxiv_retrieval",
            input_summary=_query_summary(query),
            result=retrieval_result,
            output_summary=f"{len(papers)} paper(s) retrieved",
            metadata=_trace_metadata(
                "arxiv_retrieval",
                _metadata_with_recommendation_mode(
                    retrieval_result.metadata,
                    resolved_recommendation_mode,
                ),
                include_debug=include_debug_trace,
            ),
        )

        feedback_events = (
            self.store.list_feedback_events(profile_id=profile_id)
            if include_profile_feedback
            else []
        )
        ranking_error_code = (
            "semantic_ranking_skill_failed"
            if use_semantic_ranking
            else "ranking_skill_failed"
        )
        if use_semantic_ranking:
            rank_call = lambda: self.semantic_ranking_skill.rank(
                papers,
                topic=ranking_topic,
                seed_preference=active_seed,
                feedback_events=(),
                top_k=top_k,
                query_plan=query_plan,
                retrieval_query=query,
                retrieval_source_metadata_by_paper_id=retrieval_result.metadata.get(
                    "source_metadata_by_paper_id"
                ),
                profile_id=profile_id,
            )
        else:
            rank_call = lambda: self.ranking_skill.rank(
                papers,
                topic=ranking_topic,
                seed_preference=active_seed,
                feedback_events=feedback_events,
                top_k=top_k,
                query_plan=query_plan,
                retrieval_query=query,
                retrieval_source_metadata_by_paper_id=retrieval_result.metadata.get(
                    "source_metadata_by_paper_id"
                ),
            )
        ranking_result = _safe_skill_call(
            rank_call,
            data_default=[],
            error_code=ranking_error_code,
        )
        recommendations = ranking_result.data or []
        if use_semantic_ranking:
            recommendations = _attach_semantic_context(
                recommendations,
                ranking_result.metadata,
            )
        _append_trace(
            trace,
            skill="ranking",
            input_summary=(
                f"topic={ranking_topic!r}; seed={active_seed is not None}; "
                f"feedback_events={len(feedback_events)}; top_k={top_k}"
            ),
            result=ranking_result,
            output_summary=f"{len(recommendations)} recommendation(s) ranked",
            metadata=_trace_metadata(
                "ranking",
                _metadata_with_recommendation_mode(
                    ranking_result.metadata,
                    resolved_recommendation_mode,
                ),
                include_debug=include_debug_trace,
            ),
        )
        if use_semantic_ranking and ranking_result.status == SkillStatus.ERROR:
            workflow = RecommendationWorkflow(
                run_id=workflow_run_id,
                topic=workflow_topic,
                query=query,
                papers=papers,
                recommendations=[],
                trace=trace,
            )
            results: list[SkillResult[Any]] = [
                planning_result,
                retrieval_result,
                ranking_result,
            ]
            if semantic_readiness_result is not None:
                results.insert(1, semantic_readiness_result)
            return _workflow_result(
                workflow,
                results=results,
                has_user_data=bool(papers),
                evidence_source=ranking_result.evidence_source
                or retrieval_result.evidence_source,
            )

        semantic_feedback_result: SkillResult[list[Recommendation]] | None = None
        if use_semantic_ranking and feedback_events and recommendations:
            semantic_feedback_result = _safe_skill_call(
                lambda: _refine_feedback_with_optional_context(
                    self.feedback_skill,
                    recommendations,
                    feedback=feedback_events,
                    papers=papers,
                    profile_id=profile_id,
                    recommendation_run_id=None,
                    semantic_context=ranking_result.metadata,
                    top_k=top_k,
                ),
                data_default=recommendations,
                error_code="semantic_feedback_refinement_failed",
            )
            if semantic_feedback_result.data is not None:
                recommendations = semantic_feedback_result.data
            _append_trace(
                trace,
                skill="feedback_refinement",
                input_summary=(
                    f"recommendations={len(ranking_result.data or [])}; "
                    f"feedback_events={len(feedback_events)}; profile_id={profile_id!r}"
                ),
                result=semantic_feedback_result,
                output_summary=f"{len(recommendations)} refined recommendation(s)",
                metadata=_trace_metadata(
                    "feedback_refinement",
                    _metadata_with_recommendation_mode(
                        semantic_feedback_result.metadata,
                        resolved_recommendation_mode,
                    ),
                    include_debug=include_debug_trace,
                ),
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

        retrieval_source_metadata_by_paper_id = _source_metadata_by_paper_id(
            retrieval_result.metadata
        )
        briefing_result = _safe_skill_call(
            lambda: _generate_briefing_with_optional_context(
                self.briefing_skill,
                topic=workflow_topic,
                recommendations=recommendations,
                extraction_results=item_results,
                candidate_papers=papers,
                query_plan=query_plan,
                retrieval_query=query,
                retrieval_source_metadata_by_paper_id=(
                    retrieval_source_metadata_by_paper_id
                ),
                ranking_metadata=ranking_result.metadata,
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
            metadata=_trace_metadata(
                "briefing",
                _briefing_trace_metadata(
                    briefing_result,
                    item_count=len(extraction_result.data or []),
                    candidate_count=len(papers),
                ),
                include_debug=include_debug_trace,
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
            planning_result,
            retrieval_result,
            ranking_result,
            extraction_result,
            briefing_result,
        ]
        if semantic_readiness_result is not None:
            results.insert(1, semantic_readiness_result)
        if semantic_feedback_result is not None:
            feedback_result_index = 4 if semantic_readiness_result is not None else 3
            results.insert(feedback_result_index, semantic_feedback_result)
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
        semantic_context: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        run_id: str | None = None,
        include_debug_trace: bool = False,
    ) -> SkillResult[FeedbackWorkflow]:
        """Record feedback and return updated recommendations with trace output."""

        workflow_run_id = run_id or recommendation_run_id or uuid4().hex
        trace: list[WorkflowTraceStep] = []
        feedback_result = _safe_skill_call(
            lambda: _refine_feedback_with_optional_context(
                self.feedback_skill,
                recommendations,
                feedback=feedback,
                papers=papers,
                profile_id=profile_id,
                recommendation_run_id=recommendation_run_id or workflow_run_id,
                semantic_context=semantic_context,
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
            metadata=_trace_metadata(
                "feedback_refinement",
                feedback_result.metadata,
                include_debug=include_debug_trace,
            ),
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
        include_debug_trace: bool = False,
    ) -> SkillResult[FollowupWorkflow]:
        """Run a local-first follow-up query and rank matching papers when possible."""

        workflow_run_id = run_id or uuid4().hex
        trace: list[WorkflowTraceStep] = []
        retrieval_query = _retrieval_query_from_followup(query)
        query_plan = build_deterministic_query_plan(retrieval_query)
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
            metadata=_trace_metadata(
                "followup_filter",
                followup_result.metadata,
                include_debug=include_debug_trace,
            ),
        )

        results: list[SkillResult[Any]] = [followup_result]
        recommendations: list[Recommendation] = []
        if papers and _followup_should_rank(query):
            ranking_result = _safe_skill_call(
                lambda: self.ranking_skill.rank(
                    papers,
                    topic=query.topic,
                    top_k=top_k,
                    query_plan=query_plan,
                    retrieval_query=retrieval_query,
                    retrieval_source_metadata_by_paper_id=followup_result.metadata.get(
                        "source_metadata_by_paper_id"
                    ),
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
                metadata=_trace_metadata(
                    "ranking",
                    ranking_result.metadata,
                    include_debug=include_debug_trace,
                ),
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

    def run_paper_explanation(
        self,
        paper_id: str,
        *,
        mode: ExplanationMode,
        recommendations: Sequence[Recommendation] = (),
        full_text: str | None = None,
        run_id: str | None = None,
    ) -> SkillResult[PaperExplanationWorkflow]:
        """Explain a selected paper in one of the supported deep-explanation modes."""

        workflow_run_id = run_id or uuid4().hex
        trace: list[WorkflowTraceStep] = []
        selected_paper, selection_source = _resolve_selected_paper(
            paper_id,
            recommendations=recommendations,
            store=self.store,
        )
        if selected_paper is None:
            result = SkillResult[PaperDeepExplanation](
                status=SkillStatus.ERROR,
                data=None,
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code="paper_not_found",
                    message=f"Selected paper {paper_id!r} was not found in recommendations or local storage.",
                    retryable=False,
                ),
                metadata={"paper_id": paper_id, "mode": mode.value},
            )
            _append_trace(
                trace,
                skill="deep_explanation",
                input_summary=f"paper_id={paper_id!r}; mode={mode.value!r}",
                result=result,
                output_summary="selected paper not found",
            )
            workflow = PaperExplanationWorkflow(
                run_id=workflow_run_id,
                paper_id=paper_id,
                mode=mode,
                trace=trace,
            )
            return _workflow_result(
                workflow,
                results=[result],
                has_user_data=False,
                evidence_source=result.evidence_source,
            )

        explanation_result = _safe_skill_call(
            lambda: self.deep_explanation_skill.explain(
                selected_paper,
                mode=mode,
                full_text=full_text,
            ),
            data_default=None,
            error_code="deep_explanation_skill_failed",
        )
        _append_trace(
            trace,
            skill="deep_explanation",
            input_summary=(
                f"paper_id={paper_id!r}; mode={mode.value!r}; "
                f"selection_source={selection_source!r}"
            ),
            result=explanation_result,
            output_summary=(
                "deep explanation generated"
                if explanation_result.data is not None
                else "no deep explanation generated"
            ),
        )
        workflow = PaperExplanationWorkflow(
            run_id=workflow_run_id,
            paper_id=paper_id,
            mode=mode,
            paper=selected_paper,
            explanation=explanation_result.data,
            trace=trace,
        )
        return _workflow_result(
            workflow,
            results=[explanation_result],
            has_user_data=explanation_result.data is not None,
            evidence_source=explanation_result.evidence_source,
        )

    def explain_paper(self, paper_id: str, **kwargs: Any) -> SkillResult[PaperExplanationWorkflow]:
        """Alias for run_paper_explanation."""

        return self.run_paper_explanation(paper_id, **kwargs)

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
                error=(
                    _first_error(item_results)
                    if status in {SkillStatus.FALLBACK, SkillStatus.ERROR}
                    else None
                ),
                message=(
                    "Extracted briefing items."
                    if status == SkillStatus.SUCCESS
                    else "One or more extraction calls used fallback output."
                    if status == SkillStatus.FALLBACK
                    else "One or more extraction calls failed."
                    if status == SkillStatus.ERROR
                    else "No briefing items were extracted."
                ),
                metadata={"topic": topic, "item_count": len(items)},
            ),
            item_results,
        )


def _query_plan_or_fallback(
    query: RetrievalQuery,
    result: SkillResult[QueryPlan | None],
) -> SkillResult[QueryPlan]:
    if result.data is not None:
        return SkillResult[QueryPlan](
            status=result.status,
            data=result.data,
            evidence_source=result.evidence_source,
            provenance=result.provenance,
            error=result.error,
            message=result.message,
            metadata=result.metadata,
        )

    fallback_plan = build_deterministic_query_plan(query)
    error = result.error or SkillError(
        code="query_planning_failed",
        message="Query planning failed before producing a plan.",
        retryable=True,
    )
    return SkillResult[QueryPlan](
        status=SkillStatus.FALLBACK,
        data=fallback_plan,
        evidence_source=EvidenceSource.METADATA,
        error=error,
        message="Using deterministic query planning fallback.",
        metadata={
            "requested_mode": fallback_plan.planner.requested_mode.value,
            "source": fallback_plan.planner.source,
            "fallback": True,
            "fallback_reason": error.message,
            "query_variant_count": fallback_plan.variant_count,
            "safe_to_persist": [
                "requested_mode",
                "source",
                "query_variant_count",
            ],
            "debug_only": ["query_variants", "planner_rationale"],
            "query_variants": [
                variant.model_dump(mode="json") for variant in fallback_plan.variants
            ],
        },
    )


def _plan_query(
    query_planning_skill: Any,
    query: RetrievalQuery,
    *,
    seed_preference: SeedPreference | None,
    use_seed_derived_plan: bool,
) -> SkillResult[QueryPlan]:
    if not use_seed_derived_plan:
        return query_planning_skill.plan(query)

    if seed_preference is None:
        return SkillResult[QueryPlan](
            status=SkillStatus.ERROR,
            data=None,
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code="semantic_seed_quality_error",
                message="Seed-derived retrieval requires a seed preference.",
                retryable=False,
            ),
            metadata={
                "requested_mode": query.query_planner_mode.value,
                "source": SEED_DERIVED_PLANNER_SOURCE,
                "fallback": False,
                "query_variant_count": 0,
                "candidate_target": query.effective_candidate_pool_size,
                "raw_terms_debug_only": True,
                "quality_error_reason": "seed_preference_missing",
            },
        )

    plan_from_seed = getattr(query_planning_skill, "plan_from_seed", None)
    if callable(plan_from_seed):
        return plan_from_seed(query, seed_preference)
    return build_seed_derived_query_plan(query, seed_preference)


def _should_use_seed_derived_query_plan(
    query: RetrievalQuery,
    ranking_topic: str | None,
    seed_preference: SeedPreference | None,
    *,
    recommendation_mode: str = RECOMMENDATION_MODE_AUTO,
) -> bool:
    if recommendation_mode == RECOMMENDATION_MODE_DETERMINISTIC:
        return False
    return (
        seed_preference is not None
        and not _has_text(ranking_topic or query.topic)
    )


def _should_use_semantic_ranking(
    query: RetrievalQuery,
    ranking_topic: str | None,
    seed_preference: SeedPreference | None,
    *,
    recommendation_mode: str = RECOMMENDATION_MODE_AUTO,
) -> bool:
    if recommendation_mode == RECOMMENDATION_MODE_DETERMINISTIC:
        return False
    if recommendation_mode == RECOMMENDATION_MODE_SEMANTIC_SEED:
        return True
    return (
        seed_preference is not None
        and not _has_text(ranking_topic or query.topic)
    )


def _normalize_recommendation_mode(value: str | None) -> str:
    normalized = (value or RECOMMENDATION_MODE_AUTO).strip().lower().replace("-", "_")
    if normalized == "semantic":
        normalized = RECOMMENDATION_MODE_SEMANTIC_SEED
    if normalized not in RECOMMENDATION_MODES:
        return RECOMMENDATION_MODE_AUTO
    return normalized


def _metadata_with_recommendation_mode(
    metadata: Mapping[str, Any],
    recommendation_mode: str,
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched.setdefault("recommendation_mode", recommendation_mode)
    return enriched


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _query_plan_output_summary(result: SkillResult[QueryPlan]) -> str:
    plan = result.data
    variant_count = plan.variant_count if plan is not None else 0
    source = result.metadata.get("source")
    if source is None and plan is not None:
        source = plan.planner.source
    summary = f"{variant_count} query variant(s) planned via {source or 'unknown'}"
    if result.status == SkillStatus.FALLBACK:
        return f"{summary}; fallback visible"
    if result.status == SkillStatus.ERROR:
        return f"{summary}; error visible"
    return summary


def _semantic_readiness_result(readiness: Any) -> SkillResult[dict[str, Any]]:
    metadata = readiness.model_dump(mode="json")
    if readiness.can_run:
        return SkillResult[dict[str, Any]](
            status=SkillStatus.SUCCESS,
            data=metadata,
            evidence_source=EvidenceSource.METADATA,
            message="Semantic seed recommendation readiness check passed.",
            metadata=metadata,
        )

    error_code = readiness.error_code or "semantic_readiness_failed"
    return SkillResult[dict[str, Any]](
        status=SkillStatus.ERROR,
        data=metadata,
        evidence_source=EvidenceSource.METADATA,
        error=SkillError(
            code=error_code,
            message=_semantic_readiness_error_message(error_code),
            retryable=error_code != "semantic_seed_quality_error",
        ),
        metadata=metadata,
    )


def _semantic_readiness_error_message(error_code: str) -> str:
    if error_code == "semantic_embedding_credentials_missing":
        return (
            "Semantic seed recommendation requires embedding provider credentials "
            "before retrieval can run."
        )
    if error_code == "semantic_embedding_endpoint_unsafe":
        return (
            "Semantic seed recommendation requires a safe embedding provider endpoint "
            "before retrieval can run."
        )
    if error_code == "semantic_seed_quality_error":
        return (
            "Semantic seed recommendation requires usable seed title, abstract, "
            "or category text."
        )
    return "Semantic seed recommendation is not ready to run."


def _trace_metadata(
    skill: str,
    metadata: dict[str, Any],
    *,
    include_debug: bool,
) -> dict[str, Any]:
    if include_debug:
        return metadata

    if skill == "query_planning":
        if metadata.get("source") == SEED_DERIVED_PLANNER_SOURCE:
            return _pick_metadata(
                metadata,
                (
                    "recommendation_mode",
                    "requested_mode",
                    "source",
                    "fallback",
                    "fallback_reason",
                    "query_variant_count",
                    "candidate_target",
                    "raw_terms_debug_only",
                    "seed_count",
                    "seed_paper_count",
                    "quality_error_reason",
                    "safe_to_persist",
                    "debug_only",
                ),
            )
        return _pick_metadata(
            metadata,
            (
                "recommendation_mode",
                "requested_mode",
                "source",
                "fallback",
                "fallback_reason",
                "query_variant_count",
                "required_terms",
                "optional_terms",
                "phrases",
                "exclusions",
                "safe_to_persist",
                "debug_only",
            ),
        )

    if skill == "arxiv_retrieval":
        redacted = _pick_metadata(
            metadata,
            (
                "recommendation_mode",
                "cache_hit",
                "query_variant_count",
                "request_count",
                "candidate_count",
                "candidate_target",
                "cache_status",
                "budget_exhausted",
                "seed_derived_retrieval",
                "retrieved_candidate_count",
                "seed_excluded_count",
                "candidate_pool_sufficient",
                "candidate_pool_diagnostic",
                "candidate_pool_reason",
                "candidate_pool_minimum",
                "missing_relevant_candidate_count",
            ),
        )
        partial_failures = metadata.get("partial_failures")
        if isinstance(partial_failures, list):
            redacted["partial_failures"] = [
                {
                    key: value
                    for key, value in item.items()
                    if key != "query"
                }
                for item in partial_failures
                if isinstance(item, dict)
            ]
        query_plan = metadata.get("query_plan")
        if isinstance(query_plan, dict):
            planner = query_plan.get("planner")
            if isinstance(planner, dict):
                redacted["planner_source"] = planner.get("source")
                redacted["planner_fallback"] = bool(planner.get("fallback_reason"))
        return redacted

    if skill == "semantic_readiness":
        return _pick_metadata(
            metadata,
            (
                "recommendation_mode",
                "provider",
                "provider_mode",
                "provider_label",
                "credential_status",
                "model",
                "endpoint_safety",
                "cache_enabled",
                "seed_quality",
                "can_run",
                "error_code",
                "warnings",
            ),
        )

    if skill == "ranking":
        return _pick_metadata(
            metadata,
            (
                "recommendation_mode",
                "topic",
                "top_k",
                "seed_profile_id",
                "feedback_count",
                "ranking_mode",
                "score_signals",
                "qualifying_count",
                "fallback_count",
                "minimum_evidence_score",
                "minimum_semantic_similarity",
                "semantic_similarity_threshold",
                "candidate_count",
                "seed_count",
                "seed_excluded_count",
                "semantic_provider",
                "embedding_cache",
                "semantic_context",
                "semantic_error",
            ),
        )

    if skill == "feedback_refinement":
        return _pick_metadata(
            metadata,
            (
                "recommendation_mode",
                "profile_id",
                "recommendation_run_id",
                "feedback_count",
                "active_feedback_count",
                "feedback_rule",
                "refinement_mode",
                "refinement_status",
                "influence_count",
                "skipped_feedback_count",
                "semantic_provider",
                "embedding_cache",
                "feedback_error",
            ),
        )

    if skill == "briefing":
        return _pick_metadata(
            metadata,
            (
                "topic",
                "item_count",
                "candidate_count",
                "trend_status",
                "trend_signal_count",
                "query_echo_count",
                "representative_signal_count",
                "evidence_boundary",
                "fallback_section_availability",
            ),
        )

    if skill == "followup_filter":
        return _pick_metadata(
            metadata,
            (
                "source",
                "local_hit",
                "fetch_attempted",
                "matched_count",
                "query_variant_count",
                "planner_source",
                "cache_hit",
                "cache_status",
                "candidate_count",
            ),
        )

    return metadata


def _attach_semantic_context(
    recommendations: Sequence[Recommendation],
    ranking_metadata: Mapping[str, Any],
) -> list[Recommendation]:
    semantic_context = _semantic_recommendation_context(ranking_metadata)
    if not semantic_context:
        return list(recommendations)
    return [
        recommendation.model_copy(update={"semantic_context": semantic_context})
        for recommendation in recommendations
    ]


def _semantic_recommendation_context(
    ranking_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    context = ranking_metadata.get("semantic_context")
    provider = ranking_metadata.get("semantic_provider")
    cache = ranking_metadata.get("embedding_cache")
    if not any(isinstance(item, Mapping) for item in (context, provider, cache)):
        return {}

    payload: dict[str, Any] = {}
    if isinstance(context, Mapping):
        payload["semantic_context"] = dict(context)
    if isinstance(provider, Mapping):
        payload["semantic_provider"] = dict(provider)
    if isinstance(cache, Mapping):
        payload["embedding_cache"] = dict(cache)
    return payload


def _source_metadata_by_paper_id(
    metadata: Mapping[str, Any],
) -> Mapping[str, object] | None:
    source_metadata = metadata.get("source_metadata_by_paper_id")
    if isinstance(source_metadata, Mapping):
        return source_metadata
    return None


def _refine_feedback_with_optional_context(
    feedback_skill: Any,
    recommendations: Sequence[Recommendation],
    *,
    feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]],
    papers: Sequence[PaperMetadata],
    profile_id: str,
    recommendation_run_id: str | None,
    semantic_context: Mapping[str, Any] | None,
    top_k: int | None,
) -> SkillResult[list[Recommendation]]:
    refine = feedback_skill.refine
    kwargs: dict[str, Any] = {
        "feedback": feedback,
        "papers": papers,
        "profile_id": profile_id,
        "recommendation_run_id": recommendation_run_id,
        "top_k": top_k,
    }
    if semantic_context is not None and _call_accepts_keyword(refine, "semantic_context"):
        kwargs["semantic_context"] = semantic_context
    return refine(recommendations, **kwargs)


def _generate_briefing_with_optional_context(
    briefing_skill: Any,
    *,
    topic: str,
    recommendations: Sequence[Recommendation],
    extraction_results: Sequence[SkillResult[PaperBriefingItem]],
    candidate_papers: Sequence[PaperMetadata],
    query_plan: QueryPlan,
    retrieval_query: RetrievalQuery,
    retrieval_source_metadata_by_paper_id: Mapping[str, object] | None,
    ranking_metadata: Mapping[str, object],
) -> SkillResult[DailyBriefing]:
    generate = briefing_skill.generate
    kwargs: dict[str, Any] = {
        "topic": topic,
        "recommendations": recommendations,
    }
    optional_kwargs: dict[str, Any] = {
        "extraction_results": extraction_results,
        "candidate_papers": candidate_papers,
        "query_plan": query_plan,
        "retrieval_query": retrieval_query,
        "retrieval_source_metadata_by_paper_id": (
            retrieval_source_metadata_by_paper_id
        ),
        "ranking_metadata": ranking_metadata,
    }
    for key, value in optional_kwargs.items():
        if _call_accepts_keyword(generate, key):
            kwargs[key] = value
    return generate(**kwargs)


def _briefing_trace_metadata(
    result: SkillResult[DailyBriefing],
    *,
    item_count: int,
    candidate_count: int,
) -> dict[str, Any]:
    metadata = _pick_metadata(result.metadata, ("topic",))
    metadata["item_count"] = item_count
    metadata["candidate_count"] = candidate_count

    trend_metadata = result.metadata.get("trend_analysis")
    if isinstance(trend_metadata, Mapping):
        metadata["candidate_count"] = trend_metadata.get(
            "candidate_count", candidate_count
        )
        metadata["trend_status"] = trend_metadata.get("status")
        metadata["trend_signal_count"] = trend_metadata.get("signal_count", 0)
        metadata["query_echo_count"] = trend_metadata.get(
            "query_echo_signal_count", 0
        )
        metadata["representative_signal_count"] = trend_metadata.get(
            "representative_signal_count", 0
        )

    briefing = result.data
    if briefing is None:
        metadata.setdefault("trend_signal_count", 0)
        metadata.setdefault("query_echo_count", 0)
        metadata.setdefault("representative_signal_count", 0)
        metadata["fallback_section_availability"] = {
            "trend_overview": False,
            "top_k_comparisons": False,
            "reading_priorities": False,
            "evidence_boundary": False,
        }
        return metadata

    overview = briefing.trend_overview
    metadata.update(
        {
            "item_count": len(briefing.items),
            "candidate_count": overview.candidate_count,
            "trend_status": overview.status.value,
            "trend_signal_count": len(overview.signals),
            "query_echo_count": sum(
                1 for signal in overview.signals if signal.query_echo
            ),
            "representative_signal_count": sum(
                1
                for signal in overview.signals
                if signal.signal_type.value == "hotspot"
                or (
                    signal.signal_type.value == "topic"
                    and signal.strength.value != "weak"
                )
            ),
            "evidence_boundary": _briefing_evidence_boundary_trace(
                briefing.evidence_boundary
            ),
            "fallback_section_availability": _briefing_section_availability(
                briefing
            ),
        }
    )
    return metadata


def _briefing_evidence_boundary_trace(boundary: Any) -> dict[str, Any]:
    return {
        "evidence_sources": [source.value for source in boundary.evidence_sources],
        "unavailable_sources": [
            source.value for source in boundary.unavailable_sources
        ],
        "full_text_used": boundary.full_text_used,
        "note_count": len(boundary.notes),
        "abstention_count": len(boundary.abstentions),
    }


def _briefing_section_availability(briefing: DailyBriefing) -> dict[str, bool]:
    trend_overview = briefing.trend_overview
    evidence_boundary = briefing.evidence_boundary
    return {
        "trend_overview": bool(
            trend_overview.summary
            or trend_overview.signals
            or trend_overview.limitations
            or trend_overview.candidate_count
        ),
        "top_k_comparisons": bool(briefing.top_k_comparisons),
        "reading_priorities": bool(briefing.reading_priorities),
        "evidence_boundary": bool(
            evidence_boundary.evidence_sources
            or evidence_boundary.unavailable_sources
            or evidence_boundary.notes
            or evidence_boundary.abstentions
        ),
    }


def _pick_metadata(metadata: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: metadata[key] for key in keys if key in metadata}


def _retrieval_query_from_followup(query: FollowupQuery) -> RetrievalQuery:
    return RetrievalQuery(
        topic=query.topic,
        category=query.category,
        start_date=query.start_date,
        end_date=query.end_date,
        max_results=query.max_results,
    )


def _followup_should_rank(query: FollowupQuery) -> bool:
    return bool(query.topic or query.category or query.start_date or query.end_date)


def _retrieve_with_optional_query_plan(
    retrieval_skill: Any,
    query: RetrievalQuery,
    *,
    use_cache: bool,
    query_plan: QueryPlan,
) -> SkillResult[list[PaperMetadata]]:
    retrieve = retrieval_skill.retrieve
    if _call_accepts_keyword(retrieve, "query_plan"):
        return retrieve(query, use_cache=use_cache, query_plan=query_plan)
    return retrieve(query, use_cache=use_cache)


def _prepare_seed_candidates(
    result: SkillResult[list[PaperMetadata]],
    *,
    query: RetrievalQuery,
    seed_preference: SeedPreference | None,
    top_k: int,
    seed_derived: bool,
) -> SkillResult[list[PaperMetadata]]:
    if seed_preference is None and not seed_derived:
        return result

    raw_papers = result.data or []
    seed_ids = _seed_paper_ids(seed_preference)
    papers = [paper for paper in raw_papers if paper.paper_id not in seed_ids]
    metadata = dict(result.metadata)

    seed_excluded_count = len(raw_papers) - len(papers)
    if seed_preference is not None:
        metadata.update(
            {
                "seed_paper_ids_excluded": sorted(seed_ids),
                "seed_excluded_count": seed_excluded_count,
                "retrieved_candidate_count": len(raw_papers),
                "candidate_count": len(papers),
            }
        )

    if seed_derived:
        metadata.update(
            _candidate_pool_quality_metadata(
                papers,
                query=query,
                top_k=top_k,
                seed_ids=seed_ids,
                retrieval_metadata=result.metadata,
            )
        )

    status = result.status
    message = result.message
    if status == SkillStatus.SUCCESS and raw_papers and not papers:
        status = SkillStatus.EMPTY
        message = "Only seed papers were retrieved; no non-seed candidates remain."

    return SkillResult[list[PaperMetadata]](
        status=status,
        data=papers,
        evidence_source=result.evidence_source,
        provenance=[paper.provenance for paper in papers],
        error=result.error,
        message=message,
        metadata=metadata,
    )


def _candidate_pool_quality_metadata(
    papers: Sequence[PaperMetadata],
    *,
    query: RetrievalQuery,
    top_k: int,
    seed_ids: set[str],
    retrieval_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_ids = {paper.paper_id for paper in papers}
    expected_relevant_ids = [
        paper_id
        for paper_id in _expected_relevant_paper_ids(retrieval_metadata)
        if paper_id not in seed_ids
    ]
    missing_relevant_ids = [
        paper_id for paper_id in expected_relevant_ids if paper_id not in candidate_ids
    ]
    minimum_count = min(max(top_k, 0), query.effective_candidate_pool_size)
    too_few_candidates = len(papers) < minimum_count
    sufficient = not too_few_candidates and not missing_relevant_ids
    reason = None
    if too_few_candidates:
        reason = "too_few_candidates"
    elif missing_relevant_ids:
        reason = "expected_relevant_candidates_missing"

    return {
        "seed_derived_retrieval": True,
        "candidate_pool_sufficient": sufficient,
        "candidate_pool_diagnostic": (
            "candidate_pool_sufficient"
            if sufficient
            else "candidate_pool_insufficient"
        ),
        "candidate_pool_reason": reason,
        "candidate_pool_minimum": minimum_count,
        "missing_relevant_candidate_count": len(missing_relevant_ids),
        "missing_relevant_candidate_ids": missing_relevant_ids,
    }


def _expected_relevant_paper_ids(metadata: Mapping[str, Any]) -> list[str]:
    raw_ids = (
        metadata.get("expected_relevant_paper_ids")
        or metadata.get("expected_relevant_ids")
        or []
    )
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, str):
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, str):
            continue
        paper_id = raw_id.strip()
        if not paper_id or paper_id in seen:
            continue
        ids.append(paper_id)
        seen.add(paper_id)
    return ids


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


def _call_accepts_keyword(call: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = signature(call).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == keyword
        or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
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
    metadata: dict[str, Any] | None = None,
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
            metadata=result.metadata if metadata is None else metadata,
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
    error = (
        _first_error(results)
        if status in {SkillStatus.FALLBACK, SkillStatus.ERROR}
        else None
    )
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
    if any(result.status == SkillStatus.ERROR for result in results):
        return SkillStatus.ERROR
    if any(result.status == SkillStatus.FALLBACK for result in results):
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
    if status == SkillStatus.ERROR:
        return "Workflow failed; inspect trace for details."
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


def _resolve_selected_paper(
    paper_id: str,
    *,
    recommendations: Sequence[Recommendation],
    store: SQLitePaperStore,
) -> tuple[PaperMetadata | None, str]:
    for recommendation in recommendations:
        if recommendation.paper.paper_id == paper_id:
            return recommendation.paper, "recommendations"
    stored_paper = store.get_paper(paper_id)
    if stored_paper is not None:
        return stored_paper, "store"
    return None, "missing"
