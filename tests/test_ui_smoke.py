from __future__ import annotations

from datetime import date
import importlib
import sys

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    QueryPlannerMode,
    Recommendation,
    RetrievalQuery,
    SearchMode,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.orchestrator import (
    FeedbackWorkflow,
    RecommendationWorkflow,
    WorkflowTraceStep,
)
from daily_arxiv_agent.storage import SQLitePaperStore
import daily_arxiv_agent.ui.streamlit_app as ui_module
from daily_arxiv_agent.ui.streamlit_app import (
    recommendation_empty_state_message,
    recommendation_rows,
    recommendation_summary_metrics,
    result_notice,
    workflow_trace_rows,
)


def make_paper(paper_id: str, title: str, abstract: str | None) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
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


def make_recommendation(paper_id: str, title: str, score: float) -> Recommendation:
    return Recommendation(
        paper=make_paper(
            paper_id,
            title,
            "Daily research agents rank and summarize new papers.",
        ),
        rank=1,
        score=score,
        rationale="Keyword and abstract alignment with the topic.",
        evidence_source=EvidenceSource.ABSTRACT,
    )


def test_streamlit_app_import_has_no_live_provider_side_effect(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("daily_arxiv_agent.ui.streamlit_app", None)

    module = importlib.import_module("daily_arxiv_agent.ui.streamlit_app")

    assert hasattr(module, "main")


def test_recommendation_rows_render_structured_objects() -> None:
    recommendation = make_recommendation(
        "2604.00001",
        "Agent Workflows for Research Recommendation",
        8.75,
    )

    rows = recommendation_rows([recommendation])

    assert rows == [
        {
            "rank": 1,
            "paper_id": "2604.00001",
            "title": "Agent Workflows for Research Recommendation",
            "score": 8.75,
            "evidence": "abstract",
            "categories": "cs.LG",
            "rationale": "Keyword and abstract alignment with the topic.",
            "arxiv_url": "https://arxiv.org/abs/2604.00001",
        }
    ]


def test_workflow_trace_rows_render_evidence_and_fallback_details() -> None:
    trace = [
        WorkflowTraceStep(
            step=1,
            skill="ranking",
            status=SkillStatus.FALLBACK,
            input_summary="topic='agents'",
            output_summary="2 recommendation(s) ranked",
            evidence_source=EvidenceSource.ABSTRACT,
            fallback=True,
            error_code="ranking_fallback",
            error_message="Fallback ranking was used.",
            metadata={"ranking_mode": "query_plan"},
        )
    ]

    rows = workflow_trace_rows(trace)

    assert rows == [
        {
            "step": 1,
            "skill": "ranking",
            "status": "fallback",
            "evidence": "abstract",
            "fallback": "yes",
            "input": "topic='agents'",
            "output": "2 recommendation(s) ranked",
            "error": "ranking_fallback: Fallback ranking was used.",
            "planner_source": "",
            "planner_fallback": "",
            "query_variants": "",
            "candidates": "",
            "cache": "",
            "ranking_mode": "query_plan",
        }
    ]


def test_workflow_trace_rows_surface_search_metadata_without_raw_queries() -> None:
    trace = [
        WorkflowTraceStep(
            step=1,
            skill="query_planning",
            status=SkillStatus.SUCCESS,
            input_summary="topic='agents'",
            output_summary="2 query variant(s) planned",
            evidence_source=EvidenceSource.METADATA,
            metadata={
                "source": "deterministic",
                "fallback": False,
                "query_variant_count": 2,
            },
        ),
        WorkflowTraceStep(
            step=2,
            skill="arxiv_retrieval",
            status=SkillStatus.SUCCESS,
            input_summary="topic='agents'",
            output_summary="12 paper(s) retrieved",
            evidence_source=EvidenceSource.METADATA,
            metadata={
                "planner_source": "deterministic",
                "planner_fallback": False,
                "query_variant_count": 2,
                "candidate_count": 12,
                "cache_hit": False,
                "cache_status": "complete",
            },
        ),
    ]

    rows = workflow_trace_rows(trace)

    assert rows[0]["planner_source"] == "deterministic"
    assert rows[0]["planner_fallback"] == "no"
    assert rows[0]["query_variants"] == "2"
    assert rows[1]["candidates"] == "12"
    assert rows[1]["cache"] == "miss/complete"
    assert "planner_rationale" not in rows[0]


def test_empty_recommendation_state_is_clear() -> None:
    result = SkillResult(
        status=SkillStatus.EMPTY,
        data=None,
        evidence_source=EvidenceSource.METADATA,
        message="No papers are available for ranking.",
    )

    message = recommendation_empty_state_message(result)

    assert message == "No papers matched the current retrieval and ranking filters."


def test_recommendation_summary_metrics_show_search_surface() -> None:
    workflow = RecommendationWorkflow(
        run_id="run-search-surface-123",
        topic="agent briefing",
        query=RetrievalQuery(
            topic="agent briefing",
            search_mode=SearchMode.BROAD,
            candidate_pool_size=50,
        ),
        papers=[make_paper("2604.00001", "Agent Briefings", "abstract")],
        recommendations=[
            make_recommendation("2604.00001", "Agent Briefings", 8.5),
        ],
        trace=[
            WorkflowTraceStep(
                step=1,
                skill="query_planning",
                status=SkillStatus.FALLBACK,
                input_summary="topic='agent briefing'",
                output_summary="2 query variant(s) planned",
                evidence_source=EvidenceSource.METADATA,
                metadata={"source": "deterministic", "fallback": True},
            ),
            WorkflowTraceStep(
                step=2,
                skill="arxiv_retrieval",
                status=SkillStatus.SUCCESS,
                input_summary="topic='agent briefing'",
                output_summary="8 paper(s) retrieved",
                evidence_source=EvidenceSource.METADATA,
                metadata={
                    "candidate_count": 8,
                    "cache_hit": True,
                    "cache_status": "complete",
                },
            ),
        ],
    )

    metrics = recommendation_summary_metrics(workflow)

    assert metrics["run_id"] == "run-search-s"
    assert metrics["candidate_pool_size"] == 50
    assert metrics["candidates_retrieved"] == 8
    assert metrics["recommendations_shown"] == 1
    assert metrics["planner_source"] == "deterministic"
    assert metrics["planner_fallback"] == "yes"
    assert metrics["cache_hit"] == "yes"
    assert metrics["cache_status"] == "complete"


def test_build_retrieval_query_uses_ui_search_controls() -> None:
    state = {
        "topic": "agent briefing",
        "category": "cs.LG",
        "start_date": date(2026, 4, 20),
        "end_date": date(2026, 4, 21),
        "max_results": 25,
        "search_mode": SearchMode.BROAD.value,
        "candidate_pool_size": 75,
        "arxiv_page_size": 25,
        "arxiv_max_requests": 3,
        "query_planner_mode": QueryPlannerMode.DETERMINISTIC.value,
    }

    query = ui_module._build_retrieval_query(state)

    assert query.search_mode == SearchMode.BROAD
    assert query.candidate_pool_size == 75
    assert query.effective_candidate_pool_size == 75
    assert query.page_size == 25
    assert query.max_requests == 3
    assert query.query_planner_mode == QueryPlannerMode.DETERMINISTIC


def test_option_index_falls_back_for_stale_session_values() -> None:
    options = [SearchMode.STRICT.value, SearchMode.BROAD.value]

    assert (
        ui_module._option_index(
            options,
            "invalid",
            default_value=SearchMode.BROAD.value,
        )
        == 1
    )
    assert ui_module._option_index(options, SearchMode.STRICT.value) == 0


def test_fallback_notice_includes_error_message() -> None:
    notice = result_notice(
        SkillResult(
            status=SkillStatus.FALLBACK,
            data=[],
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code="cached_results_used",
                message="Using cached results because the API request failed.",
                retryable=True,
            ),
            message="Workflow completed with fallback output.",
        ),
        empty_message="No results yet.",
    )

    assert notice["kind"] == "warning"
    assert "cached_results_used" in notice["message"]
    assert "Using cached results because the API request failed." in notice["message"]


def test_main_renders_runtime_error_created_during_action(monkeypatch) -> None:
    class FakeStreamlit:
        def __init__(self) -> None:
            self.session_state = {}
            self.errors: list[str] = []

        def set_page_config(self, **kwargs):  # noqa: ANN001, ANN201
            return None

        def markdown(self, *args, **kwargs):  # noqa: ANN001, ANN201
            return None

        def error(self, message):  # noqa: ANN001, ANN201
            self.errors.append(message)

    fake_st = FakeStreamlit()

    def record_error(st, state):  # noqa: ANN001, ANN201
        state["ui_runtime_error"] = "UI action failed: missing API key"

    monkeypatch.setattr(ui_module, "_import_streamlit", lambda: fake_st)
    monkeypatch.setattr(ui_module, "_render_sidebar", lambda st, state: None)
    monkeypatch.setattr(ui_module, "_render_hero", lambda st, state: None)
    monkeypatch.setattr(ui_module, "_render_recommendation_workspace", record_error)
    monkeypatch.setattr(ui_module, "_render_feedback_and_explanation", lambda st, state: None)
    monkeypatch.setattr(ui_module, "_render_followup_workspace", lambda st, state: None)

    ui_module.main()

    assert fake_st.errors == ["UI action failed: missing API key"]


def test_hero_pills_escape_user_controlled_values() -> None:
    class FakeMetricColumn:
        def metric(self, *args, **kwargs):  # noqa: ANN001, ANN201
            return None

    class FakeStreamlit:
        def __init__(self) -> None:
            self.markdown_calls: list[str] = []

        def markdown(self, body, **kwargs):  # noqa: ANN001, ANN201
            self.markdown_calls.append(body)

        def columns(self, count):  # noqa: ANN001, ANN201
            return [FakeMetricColumn() for _ in range(count)]

    fake_st = FakeStreamlit()
    state = {
        "provider_mode": "fake",
        "profile_id": "<script>alert(1)</script>",
        "db_path": "<b>data.sqlite3</b>",
        "recommendation_result": None,
    }

    ui_module._render_hero(fake_st, state)

    pill_markup = fake_st.markdown_calls[1]
    assert "<script>alert(1)</script>" not in pill_markup
    assert "<b>data.sqlite3</b>" not in pill_markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in pill_markup
    assert "&lt;b&gt;data.sqlite3&lt;/b&gt;" in pill_markup


def test_feedback_refinement_uses_current_displayed_recommendations(monkeypatch) -> None:
    initial = RecommendationWorkflow(
        run_id="run-1",
        topic="agent briefing",
        query=RetrievalQuery(topic="agent briefing", category="cs.LG", max_results=5),
        papers=[make_paper("2604.00001", "Initial Paper", "Initial abstract.")],
        recommendations=[
            make_recommendation("2604.00001", "Initial Paper", 8.0),
        ],
    )
    refined_recommendation = make_recommendation(
        "2604.00002",
        "Refined Paper",
        9.2,
    ).model_copy(
        update={
            "previous_rank": 2,
            "previous_score": 7.1,
            "score_delta": 2.1,
            "rank_delta": 1,
        }
    )
    refined = SkillResult(
        status=SkillStatus.SUCCESS,
        data=FeedbackWorkflow(
            run_id="run-1",
            profile_id="default",
            recommendations=[refined_recommendation],
        ),
    )
    captured: dict[str, object] = {}

    class SpyOrchestrator:
        def run_feedback_refinement(self, recommendations, **kwargs):  # noqa: ANN001, ANN201
            captured["paper_ids"] = [item.paper.paper_id for item in recommendations]
            captured["feedback"] = kwargs["feedback"]
            return refined

    runtime = ui_module.RuntimeContext(
        store=SQLitePaperStore(":memory:"),
        orchestrator=SpyOrchestrator(),  # type: ignore[arg-type]
        provider_label="fake",
        db_path=":memory:",
    )
    monkeypatch.setattr(ui_module, "_build_runtime", lambda **kwargs: runtime)

    state = {
        "provider_mode": "fake",
        "db_path": ":memory:",
        "profile_id": "default",
        "recommendation_result": SkillResult(status=SkillStatus.SUCCESS, data=initial),
        "feedback_result": refined,
        "feedback_choice_2604.00002": "like",
        "ui_runtime_error": None,
    }

    ui_module._run_feedback_refinement(state, initial)

    assert captured["paper_ids"] == ["2604.00002"]
    assert captured["feedback"] == [{"paper_id": "2604.00002", "value": "like"}]
    assert state["feedback_result"] == refined
