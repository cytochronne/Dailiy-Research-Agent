"""Configuration helpers for the local Daily arXiv agent."""

from __future__ import annotations

from dataclasses import dataclass
import os

from .contracts import QueryPlannerMode, SearchMode


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
    embedding_provider: str = "openai"
    embedding_api_key: str | None = None
    embedding_reuse_openai_api_key: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_path: str = "/embeddings"
    embedding_timeout_seconds: float = 30.0
    embedding_max_retries: int = 2
    embedding_retry_backoff_seconds: float = 1.0
    embedding_dimensions: int | None = None
    embedding_cache_enabled: bool = True
    arxiv_request_delay_seconds: float = 3.0
    search_mode: SearchMode = SearchMode.BROAD
    candidate_pool_size: int = 100
    arxiv_page_size: int = 50
    arxiv_max_requests_per_search: int = 4
    query_planner_mode: QueryPlannerMode = QueryPlannerMode.AUTO

    @classmethod
    def from_env(cls) -> "AppConfig":
        embedding_reuse_openai_api_key = _bool_from_env(
            "EMBEDDING_REUSE_OPENAI_API_KEY",
            cls.embedding_reuse_openai_api_key,
        )
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
            embedding_provider=os.getenv(
                "EMBEDDING_PROVIDER",
                cls.embedding_provider,
            ),
            embedding_api_key=_embedding_api_key_from_env(
                reuse_openai_api_key=embedding_reuse_openai_api_key,
            ),
            embedding_reuse_openai_api_key=embedding_reuse_openai_api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL") or cls.embedding_model,
            embedding_base_url=os.getenv(
                "EMBEDDING_BASE_URL",
                cls.embedding_base_url,
            ),
            embedding_path=os.getenv("EMBEDDING_PATH", cls.embedding_path),
            embedding_timeout_seconds=_float_from_env(
                "EMBEDDING_TIMEOUT_SECONDS",
                cls.embedding_timeout_seconds,
            ),
            embedding_max_retries=_int_from_env(
                "EMBEDDING_MAX_RETRIES",
                cls.embedding_max_retries,
            ),
            embedding_retry_backoff_seconds=_float_from_env(
                "EMBEDDING_RETRY_BACKOFF_SECONDS",
                cls.embedding_retry_backoff_seconds,
            ),
            embedding_dimensions=_optional_bounded_int_from_env(
                "EMBEDDING_DIMENSIONS",
                cls.embedding_dimensions,
                min_value=1,
                max_value=20000,
            ),
            embedding_cache_enabled=_bool_from_env(
                "EMBEDDING_CACHE_ENABLED",
                cls.embedding_cache_enabled,
            ),
            arxiv_request_delay_seconds=_float_from_env(
                "ARXIV_REQUEST_DELAY_SECONDS",
                cls.arxiv_request_delay_seconds,
            ),
            search_mode=_search_mode_from_env(
                "DAILY_ARXIV_SEARCH_MODE",
                cls.search_mode,
            ),
            candidate_pool_size=_bounded_int_from_env(
                "DAILY_ARXIV_CANDIDATE_POOL_SIZE",
                cls.candidate_pool_size,
                min_value=1,
                max_value=500,
            ),
            arxiv_page_size=_bounded_int_from_env(
                "ARXIV_PAGE_SIZE",
                cls.arxiv_page_size,
                min_value=1,
                max_value=100,
            ),
            arxiv_max_requests_per_search=_bounded_int_from_env(
                "ARXIV_MAX_REQUESTS_PER_SEARCH",
                cls.arxiv_max_requests_per_search,
                min_value=1,
                max_value=20,
            ),
            query_planner_mode=_query_planner_mode_from_env(
                "QUERY_PLANNER_MODE",
                cls.query_planner_mode,
            ),
        )


def _embedding_api_key_from_env(*, reuse_openai_api_key: bool) -> str | None:
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    if embedding_api_key:
        return embedding_api_key
    if reuse_openai_api_key:
        return os.getenv("OPENAI_API_KEY") or None
    return None


def _bool_from_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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


def _bounded_int_from_env(
    name: str,
    default: int,
    *,
    min_value: int,
    max_value: int,
) -> int:
    value = _int_from_env(name, default)
    if value < min_value or value > max_value:
        return default
    return value


def _optional_bounded_int_from_env(
    name: str,
    default: int | None,
    *,
    min_value: int,
    max_value: int,
) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = int(raw_value)
    except ValueError:
        return default

    if value < min_value or value > max_value:
        return default
    return value


def _search_mode_from_env(name: str, default: SearchMode) -> SearchMode:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return SearchMode(raw_value.lower())
    except ValueError:
        return default


def _query_planner_mode_from_env(
    name: str,
    default: QueryPlannerMode,
) -> QueryPlannerMode:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return QueryPlannerMode(raw_value.lower())
    except ValueError:
        return default
