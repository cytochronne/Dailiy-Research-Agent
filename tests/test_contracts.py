from datetime import date

import pytest
from pydantic import ValidationError

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    MethodExplanation,
    PaperDeepExplanation,
    PaperMetadata,
    QueryPlan,
    QueryPlanVariant,
    QueryPlannerMode,
    QueryPlannerProvenance,
    RankingScoreBreakdown,
    Recommendation,
    RetrievalBudget,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SearchMode,
    Provenance,
    SkillError,
    SkillResult,
    SkillStatus,
)


def make_paper() -> PaperMetadata:
    provenance = Provenance(
        source="arxiv",
        source_url="https://arxiv.org/abs/2501.00001",
        query="cat:cs.LG",
    )
    return PaperMetadata(
        paper_id="2501.00001",
        title="A Test Paper",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="This paper studies a testable agent architecture.",
        categories=["cs.LG"],
        published_date=date(2025, 1, 1),
        updated_date=date(2025, 1, 2),
        arxiv_url="https://arxiv.org/abs/2501.00001",
        pdf_url="https://arxiv.org/pdf/2501.00001",
        provenance=provenance,
    )


def test_paper_metadata_supports_required_arxiv_fields() -> None:
    paper = make_paper()

    assert paper.paper_id == "2501.00001"
    assert paper.title == "A Test Paper"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.abstract is not None
    assert paper.categories == ["cs.LG"]
    assert str(paper.arxiv_url) == "https://arxiv.org/abs/2501.00001"
    assert str(paper.pdf_url) == "https://arxiv.org/pdf/2501.00001"
    assert paper.provenance.source == "arxiv"


def test_skill_result_success_includes_data_and_provenance_without_error() -> None:
    paper = make_paper()
    result = SkillResult[list[PaperMetadata]](
        status=SkillStatus.SUCCESS,
        data=[paper],
        evidence_source=EvidenceSource.ABSTRACT,
        provenance=[paper.provenance],
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data == [paper]
    assert result.error is None
    assert result.evidence_source == EvidenceSource.ABSTRACT


def test_fallback_skill_result_requires_structured_error() -> None:
    result = SkillResult[None](
        status=SkillStatus.FALLBACK,
        data=None,
        evidence_source=EvidenceSource.METADATA,
        error=SkillError(
            code="arxiv_unavailable",
            message="arXiv request failed; using cached results.",
            retryable=True,
        ),
        message="Using cached results.",
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.retryable is True


def test_error_or_fallback_without_error_is_invalid() -> None:
    with pytest.raises(ValidationError):
        SkillResult[None](status=SkillStatus.ERROR)


def test_empty_optional_fields_serialize_consistently() -> None:
    paper = PaperMetadata(
        paper_id="2501.00002",
        title="Minimal Paper",
        arxiv_url="https://arxiv.org/abs/2501.00002",
        provenance=Provenance(source="arxiv"),
    )

    payload = paper.model_dump(mode="json")

    assert payload["authors"] == []
    assert payload["categories"] == []
    assert payload["abstract"] is None
    assert payload["pdf_url"] is None


def test_blank_paper_identity_is_invalid() -> None:
    with pytest.raises(ValidationError):
        PaperMetadata(
            paper_id=" ",
            title="Title",
            arxiv_url="https://arxiv.org/abs/2501.00003",
            provenance=Provenance(source="arxiv"),
        )


def test_broad_retrieval_query_supports_candidate_pool_and_default_planner_mode() -> None:
    query = RetrievalQuery(
        topic="multimodal llm agents for robotic manipulation",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=100,
    )

    assert query.search_mode == SearchMode.BROAD
    assert query.candidate_pool_size == 100
    assert query.query_planner_mode == QueryPlannerMode.AUTO
    assert query.effective_candidate_pool_size == 100


def test_retrieval_query_rejects_invalid_date_range() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(
            topic="agents",
            start_date=date(2026, 4, 21),
            end_date=date(2026, 4, 20),
        )


def test_retrieval_budget_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        RetrievalBudget(candidate_pool_size=0)

    with pytest.raises(ValidationError):
        RetrievalBudget(page_size=101)


def test_query_plan_requires_valid_variants_and_preserves_planner_provenance() -> None:
    plan = QueryPlan(
        search_mode=SearchMode.BROAD,
        planner=QueryPlannerProvenance(
            requested_mode=QueryPlannerMode.AUTO,
            source="deterministic",
        ),
        variants=[
            QueryPlanVariant(
                label="title_terms",
                search_query='ti:"robotic manipulation" OR abs:"robotic manipulation"',
                sort_by="relevance",
            )
        ],
    )

    assert plan.variant_count == 1
    assert plan.planner.requested_mode == QueryPlannerMode.AUTO
    assert plan.variants[0].label == "title_terms"


def test_query_plan_rejects_empty_variant_query() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            search_mode=SearchMode.BROAD,
            planner=QueryPlannerProvenance(
                requested_mode=QueryPlannerMode.DETERMINISTIC,
                source="deterministic",
            ),
            variants=[QueryPlanVariant(label="empty", search_query=" ")],
        )


def test_retrieval_source_metadata_is_run_scoped() -> None:
    metadata = RetrievalSourceMetadata(
        variant_label="broad_terms",
        sort_by="relevance",
        variant_index=0,
        position=3,
        first_seen_order=1,
        query='all:"agent"',
    )

    assert metadata.variant_label == "broad_terms"
    assert metadata.first_seen_order == 1
    assert metadata.query == 'all:"agent"'


def test_recommendation_can_expose_score_breakdown() -> None:
    paper = make_paper()
    breakdown = RankingScoreBreakdown(
        lexical=3.0,
        phrase=2.0,
        total=5.0,
        evidence_score=5.0,
        matched_terms=["agent"],
        matched_phrases=["agent architecture"],
        signals=["lexical", "phrase"],
    )

    recommendation = Recommendation(
        paper=paper,
        rank=1,
        score=5.0,
        rationale="Matched explicit terms: agent.",
        score_breakdown=breakdown,
    )

    assert recommendation.score_breakdown is not None
    assert recommendation.score_breakdown.total == 5.0
    assert recommendation.score_breakdown.signals == ["lexical", "phrase"]


def test_deep_explanation_contract_supports_mode_specific_sections() -> None:
    paper = make_paper()
    explanation = PaperDeepExplanation(
        paper_id=paper.paper_id,
        title=paper.title,
        mode=ExplanationMode.METHOD,
        summary="A concise method explanation.",
        evidence_source=EvidenceSource.FULL_TEXT,
        evidence_note="This explanation is based on the available full-text source.",
        method=MethodExplanation(
            problem="The paper addresses explainable paper recommendation.",
            method_overview="It uses a staged agent workflow.",
            core_workflow=["Retrieve papers", "Rank them", "Explain one paper"],
            inputs_outputs=["Input: topic query", "Output: explanation"],
            innovation="It keeps evidence labels attached to generated claims.",
        ),
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )

    assert explanation.mode == ExplanationMode.METHOD
    assert explanation.method is not None
    assert explanation.experiment is None


def test_config_reads_environment_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAILY_ARXIV_DB_PATH", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_CHAT_COMPLETIONS_PATH", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    monkeypatch.delenv("LLM_RETRY_BACKOFF_SECONDS", raising=False)
    monkeypatch.delenv("LLM_OUTPUT_RETRIES", raising=False)
    monkeypatch.delenv("LLM_BRIEFING_MAX_RETRIES", raising=False)
    monkeypatch.delenv("LLM_BRIEFING_RETRY_BACKOFF_SECONDS", raising=False)
    monkeypatch.delenv("LLM_BRIEFING_OUTPUT_RETRIES", raising=False)
    monkeypatch.delenv("ARXIV_REQUEST_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("DAILY_ARXIV_SEARCH_MODE", raising=False)
    monkeypatch.delenv("DAILY_ARXIV_CANDIDATE_POOL_SIZE", raising=False)
    monkeypatch.delenv("ARXIV_PAGE_SIZE", raising=False)
    monkeypatch.delenv("ARXIV_MAX_REQUESTS_PER_SEARCH", raising=False)
    monkeypatch.delenv("QUERY_PLANNER_MODE", raising=False)

    config = AppConfig.from_env()

    assert config.db_path == "data/daily_arxiv.sqlite3"
    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-5-mini"
    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.llm_chat_completions_path == "/chat/completions"
    assert config.llm_timeout_seconds == 30.0
    assert config.llm_max_retries == 2
    assert config.llm_retry_backoff_seconds == 1.0
    assert config.llm_output_retries == 1
    assert config.llm_briefing_max_retries == 4
    assert config.llm_briefing_retry_backoff_seconds == 1.5
    assert config.llm_briefing_output_retries == 2
    assert config.arxiv_request_delay_seconds == 3.0
    assert config.search_mode == SearchMode.BROAD
    assert config.candidate_pool_size == 100
    assert config.arxiv_page_size == 50
    assert config.arxiv_max_requests_per_search == 4
    assert config.query_planner_mode == QueryPlannerMode.AUTO


def test_config_ignores_invalid_float_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARXIV_REQUEST_DELAY_SECONDS", "not-a-number")

    config = AppConfig.from_env()

    assert config.arxiv_request_delay_seconds == 3.0


def test_config_ignores_invalid_int_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "not-an-int")

    config = AppConfig.from_env()

    assert config.llm_max_retries == 2


def test_config_reads_custom_chat_completion_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_PATH", "v1/chat/completions")

    config = AppConfig.from_env()

    assert config.llm_chat_completions_path == "v1/chat/completions"


def test_config_accepts_openai_api_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    config = AppConfig.from_env()

    assert config.llm_api_key == "sk-test"


def test_config_reads_custom_llm_retry_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "2.5")
    monkeypatch.setenv("LLM_OUTPUT_RETRIES", "2")
    monkeypatch.setenv("LLM_BRIEFING_MAX_RETRIES", "7")
    monkeypatch.setenv("LLM_BRIEFING_RETRY_BACKOFF_SECONDS", "3.5")
    monkeypatch.setenv("LLM_BRIEFING_OUTPUT_RETRIES", "4")

    config = AppConfig.from_env()

    assert config.llm_max_retries == 5
    assert config.llm_retry_backoff_seconds == 2.5
    assert config.llm_output_retries == 2
    assert config.llm_briefing_max_retries == 7
    assert config.llm_briefing_retry_backoff_seconds == 3.5
    assert config.llm_briefing_output_retries == 4


def test_config_reads_custom_search_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_ARXIV_SEARCH_MODE", "strict")
    monkeypatch.setenv("DAILY_ARXIV_CANDIDATE_POOL_SIZE", "75")
    monkeypatch.setenv("ARXIV_PAGE_SIZE", "25")
    monkeypatch.setenv("ARXIV_MAX_REQUESTS_PER_SEARCH", "3")
    monkeypatch.setenv("QUERY_PLANNER_MODE", "deterministic")

    config = AppConfig.from_env()

    assert config.search_mode == SearchMode.STRICT
    assert config.candidate_pool_size == 75
    assert config.arxiv_page_size == 25
    assert config.arxiv_max_requests_per_search == 3
    assert config.query_planner_mode == QueryPlannerMode.DETERMINISTIC


def test_config_ignores_out_of_bounds_search_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAILY_ARXIV_CANDIDATE_POOL_SIZE", "0")
    monkeypatch.setenv("ARXIV_PAGE_SIZE", "101")
    monkeypatch.setenv("ARXIV_MAX_REQUESTS_PER_SEARCH", "0")

    config = AppConfig.from_env()

    assert config.candidate_pool_size == 100
    assert config.arxiv_page_size == 50
    assert config.arxiv_max_requests_per_search == 4
