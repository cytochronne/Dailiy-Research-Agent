"""Factory for configured LLM providers."""

from __future__ import annotations

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.llm.fake import FakeLLMProvider


def create_llm_provider(config: AppConfig | None = None) -> LLMProvider:
    """Create the configured LLM provider.

    Unit 2 intentionally ships only the deterministic fake provider. The boundary
    keeps later live providers from leaking into extraction and briefing code.
    """

    resolved = config or AppConfig.from_env()
    if resolved.llm_provider == "fake":
        return FakeLLMProvider()
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{resolved.llm_provider}'. "
        "Unit 2 supports only 'fake'."
    )

