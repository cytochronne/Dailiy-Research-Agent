"""Deterministic fake LLM provider for tests and local demos."""

from __future__ import annotations

import re
from typing import Sequence

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperBriefingItem,
    PaperMetadata,
    Recommendation,
)


class FakeLLMProvider:
    """Produce stable structured output without external credentials."""

    def extract_paper(
        self,
        paper: PaperMetadata,
        *,
        topic: str,
        recommendation: Recommendation | None = None,
    ) -> PaperBriefingItem:
        evidence_source = (
            EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA
        )
        rank = recommendation.rank if recommendation else 1
        score = recommendation.score if recommendation else 0.0
        rationale = (
            recommendation.rationale
            if recommendation
            else f"Paper metadata is being reviewed for topic '{topic}'."
        )

        if paper.abstract:
            summary = _first_sentence(paper.abstract)
            contributions = [_contribution_from_text(paper.abstract, topic)]
            methods = _methods_from_text(paper.abstract)
        else:
            summary = (
                f"Metadata only: '{paper.title}' has no abstract available, so the "
                "briefing is limited to title, category, and provenance fields."
            )
            contributions = [
                "Metadata indicates a potentially relevant paper, but abstract-level "
                "claims are unavailable."
            ]
            methods = []

        return PaperBriefingItem(
            paper_id=paper.paper_id,
            title=paper.title,
            rank=rank,
            score=score,
            summary=summary,
            contributions=contributions,
            methods=methods,
            relevance_rationale=f"{rationale} Evidence: {evidence_source.value}.",
            evidence_source=evidence_source,
            provenance=paper.provenance,
            arxiv_url=paper.arxiv_url,
        )

    def summarize_briefing(
        self,
        *,
        topic: str,
        items: Sequence[PaperBriefingItem],
    ) -> str:
        if not items:
            return f"No ranked papers were available for '{topic}'."
        leader = items[0]
        return (
            f"{len(items)} ranked paper(s) were reviewed for '{topic}'. "
            f"The top paper is '{leader.title}' with evidence from "
            f"{leader.evidence_source.value}."
        )


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.split())
    match = re.search(r"(.+?[.!?])(?:\s|$)", stripped)
    return match.group(1) if match else stripped


def _contribution_from_text(text: str, topic: str) -> str:
    if topic:
        return f"Connects the paper's abstract evidence to the requested topic: {topic}."
    return f"Summarizes abstract evidence: {_first_sentence(text)}"


def _methods_from_text(text: str) -> list[str]:
    method_terms = [
        "agent",
        "workflow",
        "ranking",
        "recommendation",
        "retrieval",
        "evaluation",
        "model",
    ]
    lower_text = text.lower()
    matched = [term for term in method_terms if term in lower_text]
    if not matched:
        return ["Method details are only available at abstract level."]
    return [f"Abstract mentions {term}." for term in matched[:3]]

