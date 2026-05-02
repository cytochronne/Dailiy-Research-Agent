from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    MethodExplanation,
    PaperDeepExplanation,
    PaperMetadata,
    Provenance,
    QueryPlannerMode,
    Recommendation,
    RetrievalQuery,
    SearchMode,
    SkillStatus,
)
from daily_arxiv_agent.evaluation.metrics import (
    check_explanation_completeness,
    evaluate_feedback_movement,
    evaluate_recommendation_fixture,
    evaluate_recommendations,
    evaluate_search_quality,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.arxiv_retrieval import (
    ArxivRetrievalSkill,
    parse_atom_response,
)
from daily_arxiv_agent.skills.query_planning import QueryPlanningSkill
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


SEARCH_QUALITY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "arxiv_search_quality_response.xml"
)
SEARCH_TOPIC = "multimodal llm agents for robotic manipulation"
SEARCH_EXPECTED_RELEVANT_IDS = ["2604.20001", "2604.20002", "2604.20003"]


def make_paper(paper_id: str, title: str | None = None) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title or f"Paper {paper_id}",
        authors=["Ada Lovelace"],
        abstract="Agent workflows for research-paper recommendation.",
        categories=["cs.LG"],
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


def make_recommendation(paper_id: str, rank: int, score: float) -> Recommendation:
    return Recommendation(
        paper=make_paper(paper_id),
        rank=rank,
        score=score,
        rationale="Deterministic ranking.",
        evidence_source=EvidenceSource.ABSTRACT,
    )


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class SearchQualityClient:
    """Route planned query variants to deterministic quality-fixture subsets."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        search_query = str(params["search_query"])
        return FakeResponse(_search_quality_feed(_ids_for_search_query(search_query)))


class DivergentQualityPlannerProvider:
    def plan_queries(self, **kwargs):  # noqa: ANN003, ANN201
        return {
            "required_terms": ["compiler", "register", "allocation"],
            "phrases": ["compiler register allocation"],
            "related_terms": ["graph coloring"],
            "rationale": "Valid JSON but unrelated to the user's robotics topic.",
        }


def _ids_for_search_query(search_query: str) -> list[str]:
    if 'all:"multimodal llm agents for robotic manipulation"' in search_query:
        return ["2604.20001"]
    if "language" in search_query or "embodied" in search_query:
        return ["2604.20002", "2604.20003", "2604.20005"]
    if "ti:" in search_query or "abs:" in search_query:
        return [
            "2604.20001",
            "2604.20002",
            "2604.20003",
            "2604.20004",
            "2604.20005",
        ]
    if " OR " in search_query:
        return ["2604.20003", "2604.20004", "2604.20006"]
    return ["2604.20001"]


def _search_quality_feed(paper_ids: list[str]) -> str:
    root = ET.fromstring(SEARCH_QUALITY_FIXTURE.read_text())
    entries = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        entry_id = entry.findtext("{http://www.w3.org/2005/Atom}id") or ""
        if any(paper_id in entry_id for paper_id in paper_ids):
            entries.append(ET.tostring(entry, encoding="unicode"))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        + "\n".join(entries)
        + "\n</feed>\n"
    )


def _search_quality_papers() -> list[PaperMetadata]:
    return parse_atom_response(
        SEARCH_QUALITY_FIXTURE.read_text(),
        RetrievalQuery(topic=SEARCH_TOPIC),
    )


def make_method_explanation() -> PaperDeepExplanation:
    paper = make_paper("2604.00001", "Explainable Agents")
    return PaperDeepExplanation(
        paper_id=paper.paper_id,
        title=paper.title,
        mode=ExplanationMode.METHOD,
        summary="The paper studies explainable daily research agents.",
        evidence_source=EvidenceSource.FULL_TEXT,
        evidence_note="Generated from full text.",
        method=MethodExplanation(
            problem="Explaining ranked research-paper recommendations.",
            method_overview="A workflow retrieves, ranks, and explains papers.",
            core_workflow=["retrieve", "rank", "explain"],
            inputs_outputs=[],
            innovation="Evidence labels are attached to generated claims.",
        ),
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )


def test_recommendation_evaluation_compares_ranked_results_with_expected_ids() -> None:
    recommendations = [
        make_recommendation("2604.00001", rank=1, score=9.0),
        make_recommendation("2604.00002", rank=2, score=4.0),
        make_recommendation("2604.00003", rank=3, score=1.0),
    ]

    result = evaluate_recommendations(
        recommendations,
        ["2604.00002", "2604.99999"],
        k=2,
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.evaluated_paper_ids == ["2604.00001", "2604.00002"]
    assert data.matched_paper_ids == ["2604.00002"]
    assert data.missing_relevant_ids == ["2604.99999"]
    assert data.precision_at_k == 0.5
    assert data.recall_at_k == 0.5
    assert data.mean_reciprocal_rank == 0.5


def test_recommendation_fixture_validates_and_evaluates_simple_rows() -> None:
    result = evaluate_recommendation_fixture(
        {
            "recommendations": [
                {"paper_id": "2604.00001", "rank": 1, "score": 9.0},
                {"paper_id": "2604.00002", "rank": 2, "score": 4.0},
            ],
            "expected_relevant_paper_ids": ["2604.00001"],
            "k": 1,
        }
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data is not None
    assert result.data.matched_paper_ids == ["2604.00001"]
    assert result.metadata["fixture_keys"] == [
        "expected_relevant_paper_ids",
        "k",
        "recommendations",
    ]


def test_recommendation_evaluation_uses_rank_order_before_applying_k() -> None:
    result = evaluate_recommendation_fixture(
        {
            "recommendations": [
                {"paper_id": "2604.99999", "rank": 2, "score": 0.0},
                {"paper_id": "2604.00001", "rank": 1, "score": 9.0},
            ],
            "expected_relevant_paper_ids": ["2604.00001"],
            "k": 1,
        }
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.evaluated_paper_ids == ["2604.00001"]
    assert data.precision_at_k == 1.0
    assert data.recall_at_k == 1.0


def test_feedback_evaluation_detects_rank_movement_after_likes_and_dislikes() -> None:
    before = [
        make_recommendation("2604.00001", rank=1, score=4.0),
        make_recommendation("2604.00002", rank=2, score=3.0),
        make_recommendation("2604.00003", rank=3, score=1.0),
    ]
    after = [
        make_recommendation("2604.00002", rank=1, score=5.0),
        make_recommendation("2604.00001", rank=2, score=3.5),
        make_recommendation("2604.00003", rank=3, score=1.0),
    ]

    result = evaluate_feedback_movement(
        before,
        after,
        liked_paper_ids=["2604.00002"],
        disliked_paper_ids=["2604.00001"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.moved_up_ids == ["2604.00002"]
    assert data.moved_down_ids == ["2604.00001"]
    assert data.unchanged_ids == ["2604.00003"]
    moved_up = data.movements[0]
    assert moved_up.paper_id == "2604.00002"
    assert moved_up.rank_delta == 1
    assert moved_up.score_delta == 2.0
    assert moved_up.feedback_value == "like"


def test_explanation_completeness_reports_present_and_missing_sections() -> None:
    result = check_explanation_completeness(make_method_explanation())

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.paper_id == "2604.00001"
    assert "method.problem" in data.present_sections
    assert "method.inputs_outputs" in data.missing_sections
    assert data.completeness_score < 1.0
    assert not data.is_complete


def test_explanation_completeness_treats_missing_evidence_placeholders_as_absent() -> None:
    paper = make_paper("2604.00001", "Explainable Agents")
    explanation = FakeLLMProvider().explain_paper(
        paper,
        mode=ExplanationMode.EXPERIMENT,
        content=paper.abstract or "",
        evidence_source=EvidenceSource.ABSTRACT,
    )

    result = check_explanation_completeness(explanation)

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert "experiment.datasets" in data.missing_sections
    assert "experiment.metrics" in data.missing_sections
    assert data.completeness_score < 1.0
    assert not data.is_complete


def test_explanation_completeness_rejects_empty_required_sections() -> None:
    result = check_explanation_completeness(
        make_method_explanation(),
        required_sections=[],
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "evaluation_input_invalid"
    assert "required_sections" in result.error.message


def test_empty_recommendations_return_meaningful_zero_data_result() -> None:
    result = evaluate_recommendations([], ["2604.00001"])

    assert result.status == SkillStatus.EMPTY
    data = result.data
    assert data is not None
    assert data.zero_data_reason == "No recommendations were supplied for evaluation."
    assert data.precision_at_k == 0.0
    assert data.recall_at_k == 0.0


def test_malformed_evaluation_fixture_returns_structured_validation_error() -> None:
    result = evaluate_recommendation_fixture(
        {
            "recommendations": [{"rank": 1, "score": 9.0}],
            "expected_relevant_paper_ids": ["2604.00001"],
        }
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "evaluation_fixture_invalid"
    assert "paper_id" in result.error.message


def test_search_quality_fixture_shows_broad_retrieval_improves_candidate_coverage(
    tmp_path,
) -> None:
    strict_query = RetrievalQuery(
        topic=SEARCH_TOPIC,
        search_mode=SearchMode.STRICT,
        candidate_pool_size=20,
        page_size=10,
        max_requests=1,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )
    broad_query = strict_query.model_copy(
        update={
            "search_mode": SearchMode.BROAD,
            "max_requests": 4,
            "query_planner_mode": QueryPlannerMode.LLM,
        }
    )
    strict_plan = QueryPlanningSkill().plan(strict_query).data
    broad_plan = QueryPlanningSkill(provider=FakeLLMProvider()).plan(broad_query).data
    assert strict_plan is not None
    assert broad_plan is not None

    strict_result = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "strict.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(strict_query, query_plan=strict_plan)
    broad_result = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "broad.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(broad_query, query_plan=broad_plan)

    strict_eval = evaluate_search_quality(
        strict_result.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=strict_result.metadata,
    )
    broad_eval = evaluate_search_quality(
        broad_result.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=broad_result.metadata,
    )

    assert strict_eval.status == SkillStatus.SUCCESS
    assert broad_eval.status == SkillStatus.SUCCESS
    assert strict_eval.data is not None
    assert broad_eval.data is not None
    assert strict_eval.data.relevant_candidate_ids == ["2604.20001"]
    assert broad_eval.data.relevant_candidate_ids == SEARCH_EXPECTED_RELEVANT_IDS
    assert (
        broad_eval.data.relevant_candidate_coverage
        > strict_eval.data.relevant_candidate_coverage
    )


def test_search_quality_evaluation_covers_candidate_count_top_k_and_rationales(
    tmp_path,
) -> None:
    query = RetrievalQuery(
        topic=SEARCH_TOPIC,
        search_mode=SearchMode.BROAD,
        candidate_pool_size=100,
        page_size=50,
        max_requests=4,
        query_planner_mode=QueryPlannerMode.LLM,
    )
    plan = QueryPlanningSkill(provider=FakeLLMProvider()).plan(query).data
    assert plan is not None
    retrieval = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "quality.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(query, query_plan=plan)
    ranking = TopicRankingSkill().rank(
        retrieval.data or [],
        topic=SEARCH_TOPIC,
        query_plan=plan,
        retrieval_query=query,
        retrieval_source_metadata_by_paper_id=retrieval.metadata[
            "source_metadata_by_paper_id"
        ],
        top_k=3,
    )

    assert ranking.status == SkillStatus.SUCCESS
    recommendations = ranking.data or []
    top_ids = [recommendation.paper.paper_id for recommendation in recommendations]
    assert top_ids == SEARCH_EXPECTED_RELEVANT_IDS
    assert "2604.20006" not in top_ids
    assert all(
        recommendation.score_breakdown is not None
        and not recommendation.score_breakdown.fallback
        for recommendation in recommendations
    )

    evaluation = evaluate_search_quality(
        retrieval.data or [],
        recommendations,
        SEARCH_EXPECTED_RELEVANT_IDS,
        k=3,
        retrieval_metadata=retrieval.metadata,
    )

    assert evaluation.status == SkillStatus.SUCCESS
    data = evaluation.data
    assert data is not None
    assert data.candidate_count >= 5
    assert data.budget_exhausted is True
    assert data.top_k_paper_ids == SEARCH_EXPECTED_RELEVANT_IDS
    assert data.precision_at_k == 1.0
    assert data.recall_at_k == 1.0
    assert data.rationale_coverage == 1.0
    assert data.missing_rationale_ids == []


def test_bad_llm_query_plan_falls_back_before_search_quality_can_degrade(
    tmp_path,
) -> None:
    deterministic_query = RetrievalQuery(
        topic=SEARCH_TOPIC,
        search_mode=SearchMode.BROAD,
        candidate_pool_size=20,
        page_size=10,
        max_requests=4,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )
    llm_query = deterministic_query.model_copy(
        update={"query_planner_mode": QueryPlannerMode.LLM}
    )
    deterministic_plan = QueryPlanningSkill().plan(deterministic_query).data
    bad_llm_result = QueryPlanningSkill(
        provider=DivergentQualityPlannerProvider()
    ).plan(llm_query)
    assert deterministic_plan is not None
    assert bad_llm_result.status == SkillStatus.FALLBACK
    assert bad_llm_result.data is not None
    assert bad_llm_result.error is not None
    assert bad_llm_result.error.code == "query_planner_semantic_guard_failed"

    deterministic_retrieval = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "deterministic.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(deterministic_query, query_plan=deterministic_plan)
    fallback_retrieval = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "fallback.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(llm_query, query_plan=bad_llm_result.data)

    deterministic_eval = evaluate_search_quality(
        deterministic_retrieval.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=deterministic_retrieval.metadata,
    )
    fallback_eval = evaluate_search_quality(
        fallback_retrieval.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=fallback_retrieval.metadata,
    )

    assert deterministic_eval.data is not None
    assert fallback_eval.data is not None
    assert fallback_eval.data.relevant_candidate_coverage == (
        deterministic_eval.data.relevant_candidate_coverage
    )
    assert fallback_eval.data.candidate_count == deterministic_eval.data.candidate_count


def test_search_quality_rejects_empty_expected_relevance_set() -> None:
    result = evaluate_search_quality(_search_quality_papers(), [], [])

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "evaluation_input_invalid"
    assert "expected_relevant_paper_ids" in result.error.message
