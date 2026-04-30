"""Provider protocol for LLM-dependent Skills."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    PaperBriefingItem,
    PaperDeepExplanation,
    PaperMetadata,
    Recommendation,
    RetrievalQuery,
)


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

    def explain_paper(
        self,
        paper: PaperMetadata,
        *,
        mode: ExplanationMode,
        content: str,
        evidence_source: EvidenceSource,
    ) -> PaperDeepExplanation:
        """Return a mode-specific deep explanation for one selected paper."""

    def plan_queries(
        self,
        *,
        query: RetrievalQuery,
        deterministic_terms: Sequence[str],
    ) -> dict[str, Any]:
        """Return structured query-planning terms for a retrieval query."""
