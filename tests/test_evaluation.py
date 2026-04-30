from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    MethodExplanation,
    PaperDeepExplanation,
    PaperMetadata,
    Provenance,
    Recommendation,
    SkillStatus,
)
from daily_arxiv_agent.evaluation.metrics import (
    check_explanation_completeness,
    evaluate_feedback_movement,
    evaluate_recommendation_fixture,
    evaluate_recommendations,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider


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
