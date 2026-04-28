from datetime import date

import pytest
from pydantic import ValidationError

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
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
