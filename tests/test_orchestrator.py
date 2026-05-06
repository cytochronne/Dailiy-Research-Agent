from datetime import date
from pathlib import Path
import json

import daily_arxiv_agent.cli as cli_module
from daily_arxiv_agent.cli import main
from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    FeedbackEvent,
    FeedbackRefinementStatus,
    FeedbackValue,
    PaperMetadata,
    Provenance,
    RetrievalSourceMetadata,
    QueryPlannerMode,
    Recommendation,
    RetrievalQuery,
    SearchMode,
    SeedPreference,
    SeedRecord,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.embeddings.base import EmbeddingProviderError
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.orchestrator import DailyArxivAgentOrchestrator
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.skills.briefing import DailyBriefingSkill
from daily_arxiv_agent.skills.feedback import FeedbackRefinementSkill
from daily_arxiv_agent.skills.followup import FollowupQuery
from daily_arxiv_agent.skills.query_planning import QueryPlanningSkill
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
    build_paper_preference_text,
)
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_atom_response.xml"
SEARCH_FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_search_quality_response.xml"
TEXT_FIXTURE = Path(__file__).parent / "fixtures" / "sample_paper_text.txt"


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.text)


class RaisingRetrievalSkill:
    def retrieve(self, query, use_cache=True):  # noqa: ANN001, ANN201
        raise RuntimeError("retrieval unavailable")


class FallbackRetrievalSkill:
    def retrieve(self, query, use_cache=True):  # noqa: ANN001, ANN201
        return SkillResult[list[PaperMetadata]](
            status=SkillStatus.FALLBACK,
            data=[],
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code="cached_results_used",
                message="retrieval used cached results",
                retryable=True,
            ),
        )


class StaticRetrievalSkill:
    def __init__(
        self,
        papers: list[PaperMetadata],
        *,
        expected_relevant_paper_ids: list[str] | None = None,
    ) -> None:
        self.papers = papers
        self.expected_relevant_paper_ids = expected_relevant_paper_ids or []
        self.calls = 0
        self.query_plans = []

    def retrieve(self, query, use_cache=True, query_plan=None):  # noqa: ANN001, ANN201
        self.calls += 1
        self.query_plans.append(query_plan)
        source_metadata_by_paper_id = {
            paper.paper_id: [
                RetrievalSourceMetadata(
                    variant_label=(
                        "broad_all_terms" if index % 2 == 0 else "broad_related_terms"
                    ),
                    sort_by="relevance",
                    variant_index=index % 2,
                    position=index,
                    first_seen_order=index,
                    query=(
                        "RAW_EXPANDED_QUERY_SHOULD_BE_REDACTED "
                        f"{query.topic or ''}"
                    ),
                ).model_dump(mode="json")
            ]
            for index, paper in enumerate(self.papers)
        }
        return SkillResult[list[PaperMetadata]](
            status=SkillStatus.SUCCESS if self.papers else SkillStatus.EMPTY,
            data=self.papers,
            evidence_source=(
                EvidenceSource.ABSTRACT
                if any(paper.abstract for paper in self.papers)
                else EvidenceSource.METADATA
            ),
            provenance=[paper.provenance for paper in self.papers],
            metadata={
                "cache_hit": False,
                "query_variant_count": query_plan.variant_count if query_plan else 0,
                "request_count": 1,
                "candidate_count": len(self.papers),
                "query_plan": (
                    query_plan.model_dump(mode="json") if query_plan is not None else None
                ),
                "request_params": {
                    "search_query": "RAW_EXPANDED_QUERY_SHOULD_BE_REDACTED"
                },
                "source_metadata_by_paper_id": source_metadata_by_paper_id,
                "expected_relevant_paper_ids": self.expected_relevant_paper_ids,
            },
        )


class SpyRetrievalSkill:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query, use_cache=True, query_plan=None):  # noqa: ANN001, ANN201
        self.calls += 1
        raise AssertionError("follow-up should use stored papers before fetching")


class RaisingEmbeddingProvider:
    def embed_texts(self, texts):  # noqa: ANN001, ANN201
        raise EmbeddingProviderError("embedding provider unavailable")


class RaisingPlannerProvider:
    def plan_queries(self, **kwargs):  # noqa: ANN003, ANN201
        raise RuntimeError("planner service unavailable")


class FailingSummaryProvider(FakeLLMProvider):
    def summarize_briefing(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("summary unavailable")


class CapturingBriefingSkill(DailyBriefingSkill):
    def __init__(self) -> None:
        super().__init__(provider=FakeLLMProvider())
        self.calls = []

    def generate(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return super().generate(**kwargs)


class LegacyBriefingSkill:
    def __init__(self) -> None:
        self.delegate = DailyBriefingSkill(provider=FakeLLMProvider())

    def generate(self, *, topic, recommendations):  # noqa: ANN001, ANN201
        return self.delegate.generate(topic=topic, recommendations=recommendations)


class LegacyFeedbackSkill:
    def __init__(self) -> None:
        self.delegate = FeedbackRefinementSkill()
        self.calls = []

    def refine(  # noqa: ANN201
        self,
        recommendations,
        *,
        feedback,
        papers=(),
        profile_id="default",
        recommendation_run_id=None,
        top_k=None,
    ):
        self.calls.append(
            {
                "recommendations": recommendations,
                "feedback": feedback,
                "papers": papers,
                "profile_id": profile_id,
                "recommendation_run_id": recommendation_run_id,
                "top_k": top_k,
            }
        )
        return self.delegate.refine(
            recommendations,
            feedback=feedback,
            papers=papers,
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
            top_k=top_k,
        )


class SpyRankingSkill:
    def __init__(self) -> None:
        self.delegate = TopicRankingSkill()
        self.calls = 0
        self.kwargs = []

    def rank(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.calls += 1
        self.kwargs.append(kwargs)
        return self.delegate.rank(*args, **kwargs)


def make_paper(
    paper_id: str,
    title: str,
    abstract: str | None,
    *,
    category: str = "cs.LG",
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=[category],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query="agent briefing",
        ),
    )


def make_trend_papers() -> list[PaperMetadata]:
    return [
        make_paper(
            "2604.91001",
            "Robotic Manipulation for Household Assistance",
            "Robotic manipulation systems coordinate perception and control.",
            category="cs.RO",
        ),
        make_paper(
            "2604.91002",
            "Robotic Manipulation with Policy Learning",
            "Robotic manipulation policies improve dexterous control.",
            category="cs.LG",
        ),
        make_paper(
            "2604.91003",
            "Robotic Manipulation Benchmarks",
            "Robotic manipulation benchmarks compare embodied agents.",
            category="cs.RO",
        ),
        make_paper(
            "2604.91004",
            "Robotic Manipulation from Demonstrations",
            "Robotic manipulation from demonstrations supports robot learning.",
            category="cs.AI",
        ),
    ]


def make_recommendation(paper: PaperMetadata, rank: int, score: float) -> Recommendation:
    return Recommendation(
        paper=paper,
        rank=rank,
        score=score,
        rationale="Initial deterministic ranking.",
        evidence_source=EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA,
    )


def semantic_embedding_text(paper: PaperMetadata) -> str:
    return " ".join(
        part
        for part in [
            paper.title,
            paper.abstract or "",
            " ".join(paper.categories),
        ]
        if part
    )


def make_seed_preference(
    papers: list[PaperMetadata],
    *,
    profile_id: str = "default",
) -> SeedPreference:
    records = [
        SeedRecord(
            identity=f"arxiv:{paper.paper_id}",
            input_text=paper.paper_id,
            input_type="arxiv_id",
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            paper=paper,
            preference_text=build_paper_preference_text(paper),
        )
        for paper in papers
    ]
    preference_text = "\n\n".join(record.preference_text for record in records)
    return SeedPreference(
        profile_id=profile_id,
        seeds=records,
        preference_text=preference_text,
        vector=DeterministicTextVectorizer().vectorize(preference_text),
    )


def make_fallback_seed_preference(
    paper_id: str = "2604.99999",
    *,
    profile_id: str = "default",
) -> SeedPreference:
    record = SeedRecord(
        identity=f"arxiv:{paper_id}",
        input_text=paper_id,
        input_type="arxiv_id",
        paper_id=paper_id,
        title=paper_id,
        preference_text=paper_id,
    )
    return SeedPreference(
        profile_id=profile_id,
        seeds=[record],
        preference_text=record.preference_text,
        vector=DeterministicTextVectorizer().vectorize(record.preference_text),
    )


def test_recommendation_workflow_returns_ordered_trace_and_briefing(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    client = FakeClient(FIXTURE.read_text())
    retrieval = ArxivRetrievalSkill(
        store=store,
        client=client,
        request_delay_seconds=0,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(topic="agents", category="cs.LG", max_results=5),
        top_k=2,
        use_cache=False,
        run_id="run-1",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert workflow.run_id == "run-1"
    assert [step.skill for step in workflow.trace] == [
        "query_planning",
        "arxiv_retrieval",
        "ranking",
        "extraction",
        "briefing",
    ]
    assert [step.status for step in workflow.trace] == [
        SkillStatus.SUCCESS,
        SkillStatus.SUCCESS,
        SkillStatus.SUCCESS,
        SkillStatus.SUCCESS,
        SkillStatus.SUCCESS,
    ]
    planning_metadata = workflow.trace[0].metadata
    assert planning_metadata["source"] == "deterministic"
    assert "query_variants" not in planning_metadata
    assert "planner_rationale" not in planning_metadata
    retrieval_metadata = workflow.trace[1].metadata
    assert retrieval_metadata["candidate_count"] == 2
    assert retrieval_metadata["cache_hit"] is False
    assert retrieval_metadata["query_variant_count"] == 1
    assert retrieval_metadata["planner_source"] == "deterministic"
    assert "query_plan" not in retrieval_metadata
    assert "request_params" not in retrieval_metadata
    assert "source_metadata_by_paper_id" not in retrieval_metadata
    assert "effective_query_key" not in retrieval_metadata
    assert workflow.trace[2].metadata["ranking_mode"] == "query_plan"
    assert "query_source" in workflow.trace[2].metadata["score_signals"]
    assert len(workflow.recommendations) == 2
    assert workflow.briefing is not None
    assert workflow.briefing.highlighted_paper is not None
    briefing_metadata = workflow.trace[4].metadata
    assert briefing_metadata["item_count"] == 2
    assert briefing_metadata["candidate_count"] == 2
    assert briefing_metadata["trend_status"] == "insufficient_candidate_data"
    assert briefing_metadata["trend_signal_count"] == 0
    assert briefing_metadata["query_echo_count"] == 0
    assert briefing_metadata["evidence_boundary"]["full_text_used"] is False
    assert "full_text" in briefing_metadata["evidence_boundary"]["unavailable_sources"]
    assert client.calls == 1
    assert result.provenance is not None
    assert len(result.provenance) == 2


def test_recommendation_workflow_passes_candidate_context_into_briefing_trace(
    tmp_path,
) -> None:
    papers = make_trend_papers()
    retrieval = StaticRetrievalSkill(papers)
    ranking = SpyRankingSkill()
    briefing = CapturingBriefingSkill()
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        ranking_skill=ranking,
        briefing_skill=briefing,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic="robotic manipulation",
            category="cs.RO",
            max_results=4,
            search_mode=SearchMode.BROAD,
        ),
        top_k=2,
        use_cache=False,
        run_id="run-briefing-context",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert workflow.briefing is not None
    assert retrieval.calls == 1
    assert ranking.calls == 1
    assert len(briefing.calls) == 1

    briefing_kwargs = briefing.calls[0]
    assert briefing_kwargs["candidate_papers"] == papers
    assert briefing_kwargs["recommendations"] == workflow.recommendations
    assert len(briefing_kwargs["extraction_results"]) == len(workflow.recommendations)
    assert briefing_kwargs["query_plan"].required_terms == [
        "robotic",
        "manipulation",
    ]
    assert briefing_kwargs["retrieval_query"].topic == "robotic manipulation"
    assert briefing_kwargs["retrieval_source_metadata_by_paper_id"]
    assert briefing_kwargs["ranking_metadata"]["ranking_mode"] == "query_plan"

    overview = workflow.briefing.trend_overview
    assert overview.candidate_count == len(papers)
    assert overview.top_k_count == 2
    assert any(signal.query_echo for signal in overview.signals)

    briefing_step = workflow.trace[4]
    assert briefing_step.skill == "briefing"
    assert briefing_step.metadata["item_count"] == 2
    assert briefing_step.metadata["candidate_count"] == len(papers)
    assert briefing_step.metadata["trend_status"] == overview.status.value
    assert briefing_step.metadata["trend_signal_count"] == len(overview.signals)
    assert briefing_step.metadata["query_echo_count"] >= 1
    assert briefing_step.metadata["evidence_boundary"]["full_text_used"] is False
    assert briefing_step.metadata["fallback_section_availability"] == {
        "trend_overview": True,
        "top_k_comparisons": True,
        "reading_priorities": True,
        "evidence_boundary": True,
    }

    trace_metadata = json.dumps(
        [step.metadata for step in workflow.trace],
        sort_keys=True,
        default=str,
    )
    assert "RAW_EXPANDED_QUERY_SHOULD_BE_REDACTED" not in trace_metadata
    assert "search_query" not in trace_metadata
    assert "source_metadata_by_paper_id" not in workflow.trace[1].metadata


def test_briefing_llm_failure_marks_workflow_fallback_with_enhanced_sections(
    tmp_path,
) -> None:
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=StaticRetrievalSkill(make_trend_papers()),
        provider=FailingSummaryProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(topic="robotic manipulation", category="cs.RO", max_results=4),
        top_k=2,
        use_cache=False,
        run_id="run-briefing-fallback",
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "llm_briefing_failed"
    workflow = result.data
    assert workflow is not None
    assert workflow.briefing is not None
    assert workflow.briefing.top_k_comparisons
    assert workflow.briefing.reading_priorities
    assert workflow.briefing.evidence_boundary.full_text_used is False

    briefing_step = workflow.trace[4]
    assert briefing_step.status == SkillStatus.FALLBACK
    assert briefing_step.error_code == "llm_briefing_failed"
    assert briefing_step.metadata["fallback_section_availability"] == {
        "trend_overview": True,
        "top_k_comparisons": True,
        "reading_priorities": True,
        "evidence_boundary": True,
    }
    assert briefing_step.metadata["evidence_boundary"]["full_text_used"] is False


def test_orchestrator_supports_direct_like_briefing_skill_without_context_kwargs(
    tmp_path,
) -> None:
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=StaticRetrievalSkill(make_trend_papers()[:1]),
        briefing_skill=LegacyBriefingSkill(),
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(topic="robotic manipulation", category="cs.RO", max_results=1),
        top_k=1,
        use_cache=False,
        run_id="run-legacy-briefing",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert workflow.briefing is not None
    assert workflow.briefing.trend_overview.status.value == "not_assessed"
    assert workflow.trace[4].metadata["trend_status"] == "not_assessed"


def test_recommendation_workflow_records_planner_fallback_and_continues(
    tmp_path,
) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    retrieval = ArxivRetrievalSkill(
        store=store,
        client=FakeClient(FIXTURE.read_text()),
        request_delay_seconds=0,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval,
        query_planning_skill=QueryPlanningSkill(provider=RaisingPlannerProvider()),
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic="agents",
            category="cs.LG",
            max_results=5,
            query_planner_mode=QueryPlannerMode.LLM,
        ),
        top_k=2,
        use_cache=False,
        run_id="run-planner-fallback",
    )

    assert result.status == SkillStatus.FALLBACK
    workflow = result.data
    assert workflow is not None
    planning_step = workflow.trace[0]
    assert planning_step.skill == "query_planning"
    assert planning_step.status == SkillStatus.FALLBACK
    assert planning_step.fallback is True
    assert planning_step.error_code == "query_planner_llm_failed"
    assert planning_step.metadata["source"] == "deterministic"
    assert planning_step.metadata["fallback"] is True
    assert len(workflow.recommendations) == 2


def test_topicless_stored_seed_preference_drives_seed_derived_retrieval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    seed_paper = make_paper(
        "2604.20001",
        "Multimodal LLM Agents for Robotic Manipulation",
        (
            "We present multimodal LLM agents for robotic manipulation with "
            "vision-language planning and closed-loop control."
        ),
        category="cs.RO",
    )
    store.save_seed_preference(make_seed_preference([seed_paper], profile_id="demo"))
    client = FakeClient(SEARCH_FIXTURE.read_text())
    retrieval = ArxivRetrievalSkill(
        store=store,
        client=client,
        request_delay_seconds=0,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.RO",
            search_mode=SearchMode.BROAD,
            candidate_pool_size=3,
            page_size=3,
            max_requests=4,
        ),
        profile_id="demo",
        top_k=2,
        use_cache=False,
        run_id="run-seed-derived",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    planning_step = workflow.trace[0]
    assert planning_step.metadata["source"] == "seed_derived"
    assert planning_step.metadata["query_variant_count"] >= 1
    assert planning_step.metadata["candidate_target"] == 3
    assert planning_step.metadata["raw_terms_debug_only"] is True
    assert "required_terms" not in planning_step.metadata

    readiness_step = workflow.trace[1]
    assert readiness_step.skill == "semantic_readiness"
    assert readiness_step.status == SkillStatus.SUCCESS
    assert readiness_step.metadata["provider_mode"] == "fake"
    assert readiness_step.metadata["seed_quality"] == "usable"

    retrieval_step = workflow.trace[2]
    assert retrieval_step.metadata["planner_source"] == "seed_derived"
    assert retrieval_step.metadata["retrieved_candidate_count"] == 3
    assert retrieval_step.metadata["candidate_count"] == 2
    assert retrieval_step.metadata["seed_excluded_count"] == 1
    assert retrieval_step.metadata["candidate_pool_sufficient"] is True
    assert retrieval_step.metadata["candidate_pool_diagnostic"] == (
        "candidate_pool_sufficient"
    )
    candidate_ids = [paper.paper_id for paper in workflow.papers]
    assert candidate_ids == ["2604.20002", "2604.20003"]
    assert {
        item.paper.paper_id for item in workflow.recommendations
    } == set(candidate_ids)
    ranking_step = workflow.trace[3]
    assert ranking_step.metadata["ranking_mode"] == "semantic_seed"
    assert ranking_step.metadata["semantic_provider"]["provider_mode"] == "fake"

    trace_metadata = json.dumps(
        [step.metadata for step in workflow.trace],
        sort_keys=True,
        default=str,
    )
    assert "Multimodal LLM Agents" not in trace_metadata
    assert "robotic" not in trace_metadata
    assert "vision-language" not in trace_metadata
    assert "search_query" not in trace_metadata


def test_explicit_topic_with_seed_keeps_topic_query_planning(tmp_path) -> None:
    seed = make_paper(
        "2604.92001",
        "Compiler Register Allocation",
        "Graph coloring for register pressure in optimizing compilers.",
        category="cs.PL",
    )
    retrieval = StaticRetrievalSkill(make_trend_papers())
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic="robotic manipulation",
            category="cs.RO",
            search_mode=SearchMode.BROAD,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=2,
        use_cache=False,
        run_id="run-topic-seed",
    )

    assert result.status == SkillStatus.SUCCESS
    assert retrieval.query_plans
    plan = retrieval.query_plans[0]
    assert plan.planner.source == "deterministic"
    assert plan.required_terms == ["robotic", "manipulation"]
    assert "compiler" not in plan.required_terms


def test_deterministic_mode_disables_topicless_seed_semantic_path(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    seed = make_paper(
        "2604.92101",
        "Robotic Manipulation for Household Assistance",
        "Robotic manipulation systems coordinate perception and control.",
        category="cs.RO",
    )
    retrieval = StaticRetrievalSkill(make_trend_papers())
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.RO",
            search_mode=SearchMode.BROAD,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=2,
        use_cache=False,
        run_id="run-deterministic-seed",
        recommendation_mode="deterministic",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert "semantic_readiness" not in [step.skill for step in workflow.trace]
    assert retrieval.query_plans
    assert retrieval.query_plans[0].planner.source == "deterministic"
    ranking_step = next(step for step in workflow.trace if step.skill == "ranking")
    assert ranking_step.metadata["recommendation_mode"] == "deterministic"
    assert ranking_step.metadata["ranking_mode"] != "semantic_seed"


def test_semantic_seed_mode_with_topic_uses_topic_retrieval_and_semantic_ranking(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    seed = make_paper(
        "2604.92201",
        "Robotic Manipulation for Household Assistance",
        "Robotic manipulation systems coordinate perception and control.",
        category="cs.RO",
    )
    candidate = make_paper(
        "2604.92202",
        "Embodied Task Planning",
        "Robots infer manipulation steps from scene goals.",
        category="cs.RO",
    )
    retrieval = StaticRetrievalSkill([candidate])
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic="robotic manipulation",
            category="cs.RO",
            search_mode=SearchMode.BROAD,
            candidate_pool_size=1,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=1,
        use_cache=False,
        run_id="run-semantic-topic-seed",
        recommendation_mode="semantic_seed",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert [step.skill for step in workflow.trace][:4] == [
        "query_planning",
        "semantic_readiness",
        "arxiv_retrieval",
        "ranking",
    ]
    plan = retrieval.query_plans[0]
    assert plan.planner.source == "deterministic"
    assert plan.required_terms == ["robotic", "manipulation"]
    ranking_step = workflow.trace[3]
    assert ranking_step.metadata["recommendation_mode"] == "semantic_seed"
    assert ranking_step.metadata["ranking_mode"] == "semantic_topic_seed"
    assert ranking_step.metadata["semantic_provider"]["provider_mode"] == "fake"


def test_seed_paper_ids_are_excluded_and_insufficient_pool_is_labeled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    seed = make_paper(
        "2604.91001",
        "Robotic Manipulation for Household Assistance",
        "Robotic manipulation systems coordinate perception and control.",
        category="cs.RO",
    )
    retrieval = StaticRetrievalSkill([seed])
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.RO",
            search_mode=SearchMode.BROAD,
            candidate_pool_size=1,
            max_requests=4,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=1,
        use_cache=False,
        run_id="run-seed-excluded",
    )

    assert result.status == SkillStatus.EMPTY
    workflow = result.data
    assert workflow is not None
    assert workflow.papers == []
    assert workflow.recommendations == []
    retrieval_step = next(
        step for step in workflow.trace if step.skill == "arxiv_retrieval"
    )
    assert retrieval_step.metadata["retrieved_candidate_count"] == 1
    assert retrieval_step.metadata["candidate_count"] == 0
    assert retrieval_step.metadata["seed_excluded_count"] == 1
    assert retrieval_step.metadata["candidate_pool_sufficient"] is False
    assert retrieval_step.metadata["candidate_pool_diagnostic"] == (
        "candidate_pool_insufficient"
    )
    assert retrieval_step.metadata["candidate_pool_reason"] == "too_few_candidates"


def test_seed_derived_retrieval_labels_missing_expected_relevant_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    seed = make_paper(
        "2604.92001",
        "Robotic Manipulation with Vision Language Agents",
        "Vision language agents plan dexterous robotic manipulation tasks.",
        category="cs.RO",
    )
    candidates = [
        make_paper(
            "2604.92002",
            "Robot Manipulation Planning",
            "Planning policies coordinate manipulation skills.",
            category="cs.RO",
        ),
        make_paper(
            "2604.92003",
            "Embodied Agent Control",
            "Embodied control systems use feedback for robot actions.",
            category="cs.RO",
        ),
    ]
    retrieval = StaticRetrievalSkill(
        candidates,
        expected_relevant_paper_ids=["2604.92002", "2604.92999"],
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.RO",
            search_mode=SearchMode.BROAD,
            candidate_pool_size=2,
            max_requests=4,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=2,
        use_cache=False,
        run_id="run-seed-missing-relevant",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    retrieval_step = next(
        step for step in workflow.trace if step.skill == "arxiv_retrieval"
    )
    assert retrieval_step.metadata["candidate_pool_sufficient"] is False
    assert retrieval_step.metadata["candidate_pool_diagnostic"] == (
        "candidate_pool_insufficient"
    )
    assert retrieval_step.metadata["candidate_pool_reason"] == (
        "expected_relevant_candidates_missing"
    )
    assert retrieval_step.metadata["missing_relevant_candidate_count"] == 1
    assert "2604.92999" not in json.dumps(retrieval_step.metadata, sort_keys=True)


def test_seed_metadata_quality_error_skips_retrieval(tmp_path) -> None:
    retrieval = SpyRetrievalSkill()
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.RO",
            search_mode=SearchMode.BROAD,
            max_requests=4,
        ),
        seed_preference=make_fallback_seed_preference(),
        top_k=1,
        use_cache=False,
        run_id="run-seed-quality-error",
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "semantic_seed_quality_error"
    workflow = result.data
    assert workflow is not None
    assert workflow.papers == []
    assert workflow.recommendations == []
    assert retrieval.calls == 0
    assert [step.skill for step in workflow.trace] == ["query_planning"]
    planning_step = workflow.trace[0]
    assert planning_step.status == SkillStatus.ERROR
    assert planning_step.metadata["source"] == "seed_derived"
    assert planning_step.metadata["quality_error_reason"] == "seed_metadata_missing_text"
    assert planning_step.metadata["query_variant_count"] == 0


def test_semantic_readiness_failure_short_circuits_before_retrieval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_REUSE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    retrieval = SpyRetrievalSkill()
    seed = make_paper(
        "2604.93001",
        "Vision Language Robot Planning",
        "Embodied agents plan manipulation tasks from visual context.",
        category="cs.RO",
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.RO",
            search_mode=SearchMode.BROAD,
            max_requests=4,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=1,
        use_cache=False,
        run_id="run-semantic-readiness-error",
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "semantic_embedding_credentials_missing"
    assert retrieval.calls == 0
    workflow = result.data
    assert workflow is not None
    assert workflow.papers == []
    assert workflow.recommendations == []
    assert [step.skill for step in workflow.trace] == [
        "query_planning",
        "semantic_readiness",
    ]
    readiness_step = workflow.trace[1]
    assert readiness_step.status == SkillStatus.ERROR
    assert readiness_step.metadata["provider_mode"] == "live"
    assert readiness_step.metadata["credential_status"] == "missing"


def test_semantic_ranking_failure_short_circuits_extraction_and_briefing(
    tmp_path,
) -> None:
    seed = make_paper(
        "2604.94001",
        "Vision Language Robot Planning",
        "Embodied agents plan manipulation tasks from visual context.",
        category="cs.RO",
    )
    candidate = make_paper(
        "2604.94002",
        "Embodied Task Planning",
        "Robots infer manipulation steps from scene goals.",
        category="cs.RO",
    )
    briefing = CapturingBriefingSkill()
    semantic_ranking = SemanticSeedRankingSkill(
        embedding_provider=RaisingEmbeddingProvider(),
        store=SQLitePaperStore(tmp_path / "semantic.sqlite3"),
        config=AppConfig(embedding_provider="fake"),
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        retrieval_skill=StaticRetrievalSkill([candidate]),
        semantic_ranking_skill=semantic_ranking,
        briefing_skill=briefing,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.RO",
            search_mode=SearchMode.BROAD,
            max_requests=4,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=1,
        use_cache=False,
        run_id="run-semantic-ranking-error",
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "semantic_embedding_provider_failed"
    workflow = result.data
    assert workflow is not None
    assert workflow.recommendations == []
    assert [step.skill for step in workflow.trace] == [
        "query_planning",
        "semantic_readiness",
        "arxiv_retrieval",
        "ranking",
    ]
    ranking_step = workflow.trace[3]
    assert ranking_step.status == SkillStatus.ERROR
    assert ranking_step.metadata["ranking_mode"] == "semantic_seed"
    assert briefing.calls == []


def test_category_date_only_recommendation_ranks_by_category_recency(
    tmp_path,
) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    retrieval = ArxivRetrievalSkill(
        store=store,
        client=FakeClient(FIXTURE.read_text()),
        request_delay_seconds=0,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            category="cs.LG",
            start_date=date(2026, 4, 19),
            end_date=date(2026, 4, 21),
            max_results=5,
        ),
        top_k=2,
        use_cache=False,
        run_id="run-category-date",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert len(workflow.recommendations) == 2
    ranking_step = workflow.trace[2]
    assert ranking_step.skill == "ranking"
    assert ranking_step.metadata["ranking_mode"] == "category_recency"


def test_empty_retrieval_result_produces_empty_inspectable_workflow(tmp_path) -> None:
    empty_feed = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    retrieval = ArxivRetrievalSkill(
        store=store,
        client=FakeClient(empty_feed),
        request_delay_seconds=0,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(topic="unlikely topic", category="cs.LG", max_results=5),
        top_k=2,
        use_cache=False,
        run_id="run-empty",
    )

    assert result.status == SkillStatus.EMPTY
    workflow = result.data
    assert workflow is not None
    assert workflow.papers == []
    assert workflow.recommendations == []
    assert [step.skill for step in workflow.trace] == [
        "query_planning",
        "arxiv_retrieval",
        "ranking",
        "extraction",
        "briefing",
    ]
    assert workflow.trace[1].status == SkillStatus.EMPTY


def test_feedback_refinement_workflow_records_feedback_and_returns_updates(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    anchor = make_paper(
        "2604.00001",
        "Agent Workflows for Research Recommendation",
        "Daily briefing agents rank research papers from preference signals.",
    )
    similar = make_paper(
        "2604.00002",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    unrelated = make_paper(
        "2604.00003",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    recommendations = [
        make_recommendation(unrelated, rank=1, score=2.0),
        make_recommendation(similar, rank=2, score=1.0),
    ]
    orchestrator = DailyArxivAgentOrchestrator(store=store, provider=FakeLLMProvider())

    result = orchestrator.run_feedback_refinement(
        recommendations,
        feedback=[{"paper_id": anchor.paper_id, "value": "like"}],
        papers=[anchor],
        recommendation_run_id="run-1",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert [step.skill for step in workflow.trace] == ["feedback_refinement"]
    assert [item.paper.paper_id for item in workflow.recommendations] == [
        "2604.00002",
        "2604.00003",
    ]
    assert workflow.recommendations[0].score_delta is not None
    assert workflow.recommendations[0].score_delta > 0
    assert len(store.list_feedback_events(recommendation_run_id="run-1")) == 1


def test_feedback_refinement_supports_legacy_skill_without_semantic_context(
    tmp_path,
) -> None:
    paper = make_paper(
        "2604.00004",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    legacy_feedback = LegacyFeedbackSkill()
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        feedback_skill=legacy_feedback,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_feedback_refinement(
        [make_recommendation(paper, rank=1, score=2.0)],
        feedback=[],
        recommendation_run_id="run-legacy-feedback",
        semantic_context={"provider": "fake"},
    )

    assert result.status == SkillStatus.SUCCESS
    assert legacy_feedback.calls
    assert "semantic_context" not in legacy_feedback.calls[0]


def test_semantic_feedback_refinement_uses_originating_recommendation_context(
    tmp_path,
) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    seed = make_paper(
        "2604.30001",
        "Autonomous Lab Optimization",
        "Closed-loop experiments improve molecular discovery.",
    )
    feedback_source = make_paper(
        "2604.30002",
        "Bayesian Experimental Design",
        "Adaptive experiment selection improves chemical discovery.",
    )
    similar = make_paper(
        "2604.30003",
        "Active Learning for Molecular Experiments",
        "Bayesian experiment selection accelerates chemical discovery.",
    )
    unrelated = make_paper(
        "2604.30004",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    vector_map = {
        semantic_embedding_text(seed): [1.0, 0.0],
        semantic_embedding_text(feedback_source): [1.0, 0.0],
        semantic_embedding_text(similar): [1.0, 0.0],
        semantic_embedding_text(unrelated): [0.0, 1.0],
    }
    config = AppConfig(
        embedding_provider="fake",
        embedding_model="fake-feedback",
        embedding_dimensions=2,
    )
    ranking_provider = FakeEmbeddingProvider(dimensions=2, vector_map=vector_map)
    feedback_provider = FakeEmbeddingProvider(dimensions=2, vector_map=vector_map)
    semantic_ranking = SemanticSeedRankingSkill(
        embedding_provider=ranking_provider,
        store=store,
        config=config,
        minimum_semantic_similarity=0.0,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=StaticRetrievalSkill([similar, unrelated]),
        semantic_ranking_skill=semantic_ranking,
        feedback_skill=FeedbackRefinementSkill(
            store=store,
            embedding_provider=feedback_provider,
            config=config,
        ),
        provider=FakeLLMProvider(),
    )

    recommendation_result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.LG",
            search_mode=SearchMode.BROAD,
            candidate_pool_size=2,
            max_requests=4,
        ),
        seed_preference=make_seed_preference([seed]),
        top_k=2,
        use_cache=False,
        run_id="run-semantic-origin",
    )

    workflow = recommendation_result.data
    assert workflow is not None
    assert recommendation_result.status == SkillStatus.SUCCESS
    assert workflow.recommendations
    assert workflow.recommendations[0].semantic_context["semantic_context"][
        "provider"
    ] == "fake"

    feedback_result = orchestrator.run_feedback_refinement(
        workflow.recommendations,
        feedback=[{"paper_id": feedback_source.paper_id, "value": "like"}],
        papers=[feedback_source],
        recommendation_run_id="run-semantic-origin",
    )

    assert feedback_result.status == SkillStatus.SUCCESS
    feedback_workflow = feedback_result.data
    assert feedback_workflow is not None
    assert [step.skill for step in feedback_workflow.trace] == ["feedback_refinement"]
    feedback_step = feedback_workflow.trace[0]
    assert feedback_step.metadata["refinement_mode"] == "semantic_feedback"
    assert feedback_step.metadata["semantic_provider"]["provider"] == "fake"
    assert feedback_step.metadata["embedding_cache"]["hits"] >= 1
    refined = feedback_workflow.recommendations
    assert refined[0].feedback_influences
    assert refined[0].refinement_status == FeedbackRefinementStatus.APPLIED
    assert refined[0].score_breakdown is not None
    assert refined[0].score_breakdown.feedback > 0


def test_semantic_recommendation_applies_profile_feedback_semantically(
    tmp_path,
) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    seed = make_paper(
        "2604.31001",
        "Autonomous Lab Optimization",
        "Closed-loop experiments improve molecular discovery.",
    )
    feedback_source = make_paper(
        "2604.31002",
        "Bayesian Experimental Design",
        "Adaptive experiment selection improves chemical discovery.",
    )
    similar = make_paper(
        "2604.31003",
        "Active Learning for Molecular Experiments",
        "Bayesian experiment selection accelerates chemical discovery.",
    )
    unrelated = make_paper(
        "2604.31004",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    store.save_feedback_event(
        FeedbackEvent(
            profile_id="demo",
            paper_id=feedback_source.paper_id,
            value=FeedbackValue.LIKE,
            paper=feedback_source,
        )
    )
    vector_map = {
        semantic_embedding_text(seed): [1.0, 0.0],
        semantic_embedding_text(feedback_source): [1.0, 0.0],
        semantic_embedding_text(similar): [1.0, 0.0],
        semantic_embedding_text(unrelated): [0.0, 1.0],
    }
    config = AppConfig(
        embedding_provider="fake",
        embedding_model="fake-feedback",
        embedding_dimensions=2,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=StaticRetrievalSkill([similar, unrelated]),
        semantic_ranking_skill=SemanticSeedRankingSkill(
            embedding_provider=FakeEmbeddingProvider(
                dimensions=2,
                vector_map=vector_map,
            ),
            store=store,
            config=config,
            minimum_semantic_similarity=0.0,
        ),
        feedback_skill=FeedbackRefinementSkill(
            store=store,
            embedding_provider=FakeEmbeddingProvider(
                dimensions=2,
                vector_map=vector_map,
            ),
            config=config,
        ),
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(
            topic=None,
            category="cs.LG",
            search_mode=SearchMode.BROAD,
            candidate_pool_size=2,
            max_requests=4,
        ),
        seed_preference=make_seed_preference([seed], profile_id="demo"),
        profile_id="demo",
        top_k=2,
        use_cache=False,
        run_id="run-semantic-profile-feedback",
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert [step.skill for step in workflow.trace] == [
        "query_planning",
        "semantic_readiness",
        "arxiv_retrieval",
        "ranking",
        "feedback_refinement",
        "extraction",
        "briefing",
    ]
    ranking_step = workflow.trace[3]
    assert ranking_step.metadata["feedback_count"] == 0
    feedback_step = workflow.trace[4]
    assert feedback_step.metadata["refinement_mode"] == "semantic_feedback"
    assert feedback_step.metadata["influence_count"] >= 1
    assert workflow.recommendations[0].feedback_influences
    assert workflow.recommendations[0].score_breakdown is not None
    assert workflow.recommendations[0].score_breakdown.feedback > 0


def test_followup_workflow_filters_stored_papers_without_fetching(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    paper = make_paper(
        "2604.00001",
        "Explainable Agents for Daily Research Briefings",
        "Agent workflows for research-paper recommendation.",
    )
    store.save_papers([paper])
    retrieval = SpyRetrievalSkill()
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_followup_query(
        FollowupQuery(
            topic="agent workflow",
            category="cs.LG",
            start_date=date(2026, 4, 19),
            end_date=date(2026, 4, 21),
        )
    )

    assert result.status == SkillStatus.SUCCESS
    workflow = result.data
    assert workflow is not None
    assert [paper.paper_id for paper in workflow.papers] == ["2604.00001"]
    assert workflow.trace[0].skill == "followup_filter"
    assert workflow.trace[0].metadata["fetch_attempted"] is False
    assert retrieval.calls == 0


def test_skill_failure_is_visible_in_trace_and_returns_workflow_error(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=RaisingRetrievalSkill(),
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(topic="agents", category="cs.LG", max_results=5),
        run_id="run-failure",
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "retrieval_skill_failed"
    workflow = result.data
    assert workflow is not None
    assert workflow.trace[0].skill == "query_planning"
    first_step = workflow.trace[1]
    assert first_step.skill == "arxiv_retrieval"
    assert first_step.status == SkillStatus.ERROR
    assert first_step.fallback is True
    assert first_step.error_code == "retrieval_skill_failed"


def test_skill_fallback_is_visible_in_trace_and_returns_workflow_fallback(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=FallbackRetrievalSkill(),
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_recommendation(
        RetrievalQuery(topic="agents", category="cs.LG", max_results=5),
        run_id="run-fallback",
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "cached_results_used"
    workflow = result.data
    assert workflow is not None
    assert workflow.trace[1].status == SkillStatus.FALLBACK


def test_paper_explanation_workflow_runs_after_recommendation_workflow(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    client = FakeClient(FIXTURE.read_text())
    retrieval = ArxivRetrievalSkill(
        store=store,
        client=client,
        request_delay_seconds=0,
    )
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval,
        provider=FakeLLMProvider(),
    )
    recommendation_result = orchestrator.run_recommendation(
        RetrievalQuery(topic="agents", category="cs.LG", max_results=5),
        top_k=1,
        use_cache=False,
        run_id="run-6-recommend",
    )

    recommendation_workflow = recommendation_result.data
    assert recommendation_workflow is not None
    selected = recommendation_workflow.recommendations[0]

    explanation_result = orchestrator.run_paper_explanation(
        selected.paper.paper_id,
        mode=ExplanationMode.METHOD,
        recommendations=recommendation_workflow.recommendations,
        full_text=TEXT_FIXTURE.read_text(),
        run_id="run-6-explain",
    )

    assert explanation_result.status == SkillStatus.SUCCESS
    workflow = explanation_result.data
    assert workflow is not None
    assert workflow.run_id == "run-6-explain"
    assert workflow.trace[0].skill == "deep_explanation"
    assert workflow.explanation is not None
    assert workflow.explanation.method is not None
    assert workflow.explanation.evidence_source == EvidenceSource.FULL_TEXT


def test_missing_selected_paper_returns_structured_not_found_error(tmp_path) -> None:
    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        provider=FakeLLMProvider(),
    )

    result = orchestrator.run_paper_explanation(
        "missing-paper",
        mode=ExplanationMode.LIMITATIONS,
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "paper_not_found"
    workflow = result.data
    assert workflow is not None
    assert workflow.trace[0].skill == "deep_explanation"
    assert workflow.trace[0].status == SkillStatus.ERROR


def test_cli_demo_runs_fixture_backed_workflow_end_to_end(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")

    exit_code = main(
        [
            "demo",
            "--fixture",
            str(FIXTURE),
            "--db-path",
            str(tmp_path / "cli.sqlite3"),
            "--topic",
            "agents",
            "--category",
            "cs.LG",
            "--max-results",
            "5",
            "--search-mode",
            "broad",
            "--query-planner-mode",
            "deterministic",
            "--candidate-pool-size",
            "20",
            "--page-size",
            "10",
            "--max-requests",
            "2",
            "--top-k",
            "2",
            "--no-cache",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "success"
    assert [step["skill"] for step in payload["data"]["trace"]] == [
        "query_planning",
        "arxiv_retrieval",
        "ranking",
        "extraction",
        "briefing",
    ]
    assert payload["data"]["query"]["search_mode"] == SearchMode.BROAD.value
    assert payload["data"]["query"]["query_planner_mode"] == QueryPlannerMode.DETERMINISTIC.value
    assert payload["data"]["query"]["candidate_pool_size"] == 20
    assert payload["data"]["query"]["page_size"] == 10
    assert payload["data"]["query"]["max_requests"] == 2
    assert len(payload["data"]["recommendations"]) == 2
    briefing = payload["data"]["briefing"]
    assert briefing["trend_overview"]["status"] in {
        "available",
        "limited",
        "insufficient_candidate_data",
    }
    assert "top_k_comparisons" in briefing
    assert "reading_priorities" in briefing
    assert briefing["evidence_boundary"]["full_text_used"] is False


def test_default_orchestrator_uses_arxiv_delay_from_env(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ARXIV_REQUEST_DELAY_SECONDS", "0")
    monkeypatch.setenv("LLM_PROVIDER", "fake")

    orchestrator = DailyArxivAgentOrchestrator(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3")
    )

    assert orchestrator.retrieval_skill.request_delay_seconds == 0


def test_cli_returns_nonzero_exit_code_for_fallback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "_run_demo",
        lambda args: SkillResult[dict[str, str]](
            status=SkillStatus.FALLBACK,
            data={"run": "demo"},
            error=SkillError(
                code="fallback_for_test",
                message="forced fallback",
                retryable=False,
            ),
        ),
    )

    exit_code = main(["demo"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fallback"
    assert exit_code == 1


def test_cli_returns_nonzero_exit_code_for_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "_run_followup",
        lambda args: SkillResult[dict[str, str]](
            status=SkillStatus.ERROR,
            data={"run": "followup"},
            error=SkillError(
                code="error_for_test",
                message="forced error",
                retryable=False,
            ),
        ),
    )

    exit_code = main(["followup"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert exit_code == 1
