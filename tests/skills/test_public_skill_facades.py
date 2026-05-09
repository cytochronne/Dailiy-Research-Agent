from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    PaperMetadata,
    Provenance,
    QueryPlannerMode,
    Recommendation,
    RetrievalQuery,
    SearchMode,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.orchestrator import DailyArxivAgentOrchestrator
from daily_arxiv_agent.skills.discovery_recommendation import (
    DiscoveryRecommendationSkill,
)
from daily_arxiv_agent.skills.query_planning import QueryPlanningSkill
from daily_arxiv_agent.skills.research_synthesis import ResearchSynthesisSkill
from daily_arxiv_agent.storage import SQLitePaperStore


def make_recommendation() -> Recommendation:
    paper = PaperMetadata(
        paper_id="2604.00001",
        title="Explainable Agents for Daily Research Briefings",
        authors=["Ada Lovelace"],
        abstract=(
            "We propose an agent workflow that retrieves, ranks, and synthesizes "
            "daily research briefings with evidence-bounded reading guidance."
        ),
        categories=["cs.LG"],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url="https://arxiv.org/abs/2604.00001",
        pdf_url="https://arxiv.org/pdf/2604.00001",
        provenance=Provenance(
            source="arxiv",
            source_url="https://arxiv.org/abs/2604.00001",
            query="agent briefing",
        ),
    )
    return Recommendation(
        paper=paper,
        rank=1,
        score=7.5,
        rationale="Matched explicit terms: agent, briefing.",
        evidence_source=EvidenceSource.ABSTRACT,
    )


def test_discovery_facade_delegates_planning_and_seed_parsing(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "discovery.sqlite3")
    planner = QueryPlanningSkill()
    skill = DiscoveryRecommendationSkill(
        store=store,
        query_planning_skill=planner,
    )

    assert skill.store is store
    assert skill.query_planning_skill is planner
    assert skill.retrieval_skill.store is store
    assert skill.followup_skill.retrieval_skill is skill.retrieval_skill

    query = RetrievalQuery(
        topic="agent briefing",
        search_mode=SearchMode.STRICT,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )
    plan_result = skill.plan_query(query)

    assert plan_result.status == SkillStatus.SUCCESS
    assert plan_result.data is not None
    assert "agent" in plan_result.data.required_terms

    seed_result = skill.build_seed_preference(
        ["Explainable agents for daily research briefings"],
        profile_id="facade-test",
    )

    assert seed_result.status == SkillStatus.SUCCESS
    assert seed_result.data is not None
    assert seed_result.data.profile_id == "facade-test"


def test_synthesis_facade_delegates_extraction_briefing_and_explanation(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "synthesis.sqlite3")
    recommendation = make_recommendation()
    skill = ResearchSynthesisSkill(
        provider=FakeLLMProvider(),
        store=store,
    )

    extraction_result = skill.extract_paper(
        recommendation,
        topic="agent briefing",
    )
    assert extraction_result.status == SkillStatus.SUCCESS
    assert extraction_result.data is not None

    briefing_result = skill.generate_briefing(
        topic="agent briefing",
        recommendations=[recommendation],
        extraction_results=[extraction_result],
        candidate_papers=[recommendation.paper],
    )
    assert briefing_result.status == SkillStatus.SUCCESS
    assert briefing_result.data is not None
    assert briefing_result.data.highlighted_paper.paper_id == recommendation.paper.paper_id

    explanation_result = skill.explain_paper(
        recommendation.paper,
        mode=ExplanationMode.METHOD,
        full_text=(
            "summary: The paper studies agent workflows for research briefings.\n"
            "method: Retrieval, ranking, and synthesis are chained together."
        ),
    )
    assert explanation_result.status == SkillStatus.SUCCESS
    assert explanation_result.data is not None


def test_orchestrator_uses_public_facades_with_legacy_aliases(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "orchestrator.sqlite3")
    orchestrator = DailyArxivAgentOrchestrator(
        store=store,
        provider=FakeLLMProvider(),
    )

    assert orchestrator.discovery_skill.store is store
    assert (
        orchestrator.query_planning_skill
        is orchestrator.discovery_skill.query_planning_skill
    )
    assert orchestrator.retrieval_skill is orchestrator.discovery_skill.retrieval_skill
    assert orchestrator.ranking_skill is orchestrator.discovery_skill.ranking_skill
    assert orchestrator.feedback_skill is orchestrator.discovery_skill.feedback_skill
    assert orchestrator.followup_skill is orchestrator.discovery_skill.followup_skill
    assert orchestrator.extraction_skill is orchestrator.synthesis_skill.extraction_skill
    assert orchestrator.briefing_skill is orchestrator.synthesis_skill.briefing_skill
    assert (
        orchestrator.deep_explanation_skill
        is orchestrator.synthesis_skill.deep_explanation_skill
    )
