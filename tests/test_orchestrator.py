from datetime import date
from pathlib import Path
import json

import daily_arxiv_agent.cli as cli_module
from daily_arxiv_agent.cli import main
from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    PaperMetadata,
    Provenance,
    Recommendation,
    RetrievalQuery,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.orchestrator import DailyArxivAgentOrchestrator
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.skills.followup import FollowupQuery
from daily_arxiv_agent.storage import SQLitePaperStore


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_atom_response.xml"
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


class SpyRetrievalSkill:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query, use_cache=True):  # noqa: ANN001, ANN201
        self.calls += 1
        raise AssertionError("follow-up should use stored papers before fetching")


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


def make_recommendation(paper: PaperMetadata, rank: int, score: float) -> Recommendation:
    return Recommendation(
        paper=paper,
        rank=rank,
        score=score,
        rationale="Initial deterministic ranking.",
        evidence_source=EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA,
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
    ]
    assert len(workflow.recommendations) == 2
    assert workflow.briefing is not None
    assert workflow.briefing.highlighted_paper is not None
    assert client.calls == 1
    assert result.provenance is not None
    assert len(result.provenance) == 2


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
    first_step = workflow.trace[0]
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
    assert workflow.trace[0].status == SkillStatus.FALLBACK


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
        "arxiv_retrieval",
        "ranking",
        "extraction",
        "briefing",
    ]
    assert len(payload["data"]["recommendations"]) == 2


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
