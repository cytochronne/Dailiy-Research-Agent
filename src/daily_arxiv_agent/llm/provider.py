"""Factory for configured LLM providers."""

from __future__ import annotations

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.llm.openai_provider import OpenAILLMProvider

LIVE_API_PROVIDER_NAMES = frozenset({"live", "openai"})


def create_llm_provider(config: AppConfig | None = None) -> LLMProvider:
    """Create the configured LLM provider."""

    resolved = config or AppConfig.from_env()
    provider = resolved.llm_provider.strip().lower() or "openai"
    if provider == "fake":
        return FakeLLMProvider()
    if provider in LIVE_API_PROVIDER_NAMES and not (resolved.llm_api_key or "").strip():
        raise ValueError(
            "LLM_API_KEY or OPENAI_API_KEY is required when "
            f"LLM_PROVIDER is {provider!r}."
        )

    # Any non-fake provider name is treated as a custom OpenAI-compatible API.
    return OpenAILLMProvider(
        api_key=resolved.llm_api_key,
        model=resolved.llm_model or "gpt-5-mini",
        base_url=resolved.llm_base_url,
        chat_completions_path=resolved.llm_chat_completions_path,
        timeout_seconds=resolved.llm_timeout_seconds,
        max_retries=resolved.llm_max_retries,
        retry_backoff_seconds=resolved.llm_retry_backoff_seconds,
        output_retries=resolved.llm_output_retries,
        briefing_max_retries=resolved.llm_briefing_max_retries,
        briefing_retry_backoff_seconds=resolved.llm_briefing_retry_backoff_seconds,
        briefing_output_retries=resolved.llm_briefing_output_retries,
    )
