"""Provider protocol for LLM-dependent Skills."""

from __future__ import annotations

from typing import Protocol, Sequence

from daily_arxiv_agent.contracts import PaperBriefingItem, PaperMetadata, Recommendation


class LLMProvider(Protocol):
    """Minimal adapter boundary for structured extraction and briefing text."""

    def extract_paper(
        self,
        paper: PaperMetadata,
        *,
        topic: str,
        recommendation: Recommendation | None = None,
    ) -> PaperBriefingItem:
        """Return structured briefing fields for one paper."""

    def summarize_briefing(
        self,
        *,
        topic: str,
        items: Sequence[PaperBriefingItem],
    ) -> str:
        """Return the executive summary for a daily briefing."""

