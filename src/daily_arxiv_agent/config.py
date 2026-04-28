"""Configuration helpers for the local Daily arXiv agent."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration read from environment variables."""

    db_path: str = "data/daily_arxiv.sqlite3"
    llm_provider: str = "openai"
    llm_api_key: str | None = None
    llm_model: str | None = "gpt-5-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_chat_completions_path: str = "/chat/completions"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 1.0
    llm_output_retries: int = 1
    llm_briefing_max_retries: int = 4
    llm_briefing_retry_backoff_seconds: float = 1.5
    llm_briefing_output_retries: int = 2
    arxiv_request_delay_seconds: float = 3.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            db_path=os.getenv("DAILY_ARXIV_DB_PATH", cls.db_path),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            llm_api_key=(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None),
            llm_model=os.getenv("LLM_MODEL") or cls.llm_model,
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_chat_completions_path=os.getenv(
                "LLM_CHAT_COMPLETIONS_PATH",
                cls.llm_chat_completions_path,
            ),
            llm_timeout_seconds=_float_from_env(
                "LLM_TIMEOUT_SECONDS",
                cls.llm_timeout_seconds,
            ),
            llm_max_retries=_int_from_env(
                "LLM_MAX_RETRIES",
                cls.llm_max_retries,
            ),
            llm_retry_backoff_seconds=_float_from_env(
                "LLM_RETRY_BACKOFF_SECONDS",
                cls.llm_retry_backoff_seconds,
            ),
            llm_output_retries=_int_from_env(
                "LLM_OUTPUT_RETRIES",
                cls.llm_output_retries,
            ),
            llm_briefing_max_retries=_int_from_env(
                "LLM_BRIEFING_MAX_RETRIES",
                cls.llm_briefing_max_retries,
            ),
            llm_briefing_retry_backoff_seconds=_float_from_env(
                "LLM_BRIEFING_RETRY_BACKOFF_SECONDS",
                cls.llm_briefing_retry_backoff_seconds,
            ),
            llm_briefing_output_retries=_int_from_env(
                "LLM_BRIEFING_OUTPUT_RETRIES",
                cls.llm_briefing_output_retries,
            ),
            arxiv_request_delay_seconds=_float_from_env(
                "ARXIV_REQUEST_DELAY_SECONDS",
                cls.arxiv_request_delay_seconds,
            ),
        )


def _float_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def _int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default
