"""Configuration helpers for the local Daily arXiv agent."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration read from environment variables."""

    db_path: str = "data/daily_arxiv.sqlite3"
    llm_provider: str = "fake"
    llm_api_key: str | None = None
    llm_model: str | None = None
    arxiv_request_delay_seconds: float = 3.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            db_path=os.getenv("DAILY_ARXIV_DB_PATH", cls.db_path),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_model=os.getenv("LLM_MODEL") or None,
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

