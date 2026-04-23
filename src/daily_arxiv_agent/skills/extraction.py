"""Structured extraction Skill for ranked papers."""

from __future__ import annotations

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperBriefingItem,
    Recommendation,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.llm.provider import create_llm_provider


class PaperExtractionSkill:
    """Extract briefing fields for one recommendation through an LLM adapter."""

    def __init__(self, *, provider: LLMProvider | None = None) -> None:
        self.provider = provider or create_llm_provider()

    def extract(
        self,
        recommendation: Recommendation,
        *,
        topic: str,
    ) -> SkillResult[PaperBriefingItem]:
        try:
            item = self.provider.extract_paper(
                recommendation.paper,
                topic=topic,
                recommendation=recommendation,
            )
        except Exception as exc:
            item = _fallback_item(recommendation, topic=topic)
            return SkillResult[PaperBriefingItem](
                status=SkillStatus.FALLBACK,
                data=item,
                evidence_source=item.evidence_source,
                provenance=[recommendation.paper.provenance],
                error=SkillError(
                    code="llm_extraction_failed",
                    message=f"LLM extraction failed: {exc}",
                    retryable=True,
                ),
                message="Using metadata-only fallback extraction.",
                metadata={"topic": topic, "paper_id": recommendation.paper.paper_id},
            )

        return SkillResult[PaperBriefingItem](
            status=SkillStatus.SUCCESS,
            data=item,
            evidence_source=item.evidence_source,
            provenance=[recommendation.paper.provenance],
            metadata={"topic": topic, "paper_id": recommendation.paper.paper_id},
        )


def _fallback_item(recommendation: Recommendation, *, topic: str) -> PaperBriefingItem:
    paper = recommendation.paper
    return PaperBriefingItem(
        paper_id=paper.paper_id,
        title=paper.title,
        rank=recommendation.rank,
        score=recommendation.score,
        summary=(
            f"Metadata-only fallback for '{paper.title}' while extracting topic "
            f"'{topic}'."
        ),
        contributions=[
            "Structured extraction was unavailable, so no abstract-level claims are made."
        ],
        methods=[],
        relevance_rationale=recommendation.rationale,
        evidence_source=EvidenceSource.METADATA,
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )

