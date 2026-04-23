"""Daily briefing generation from ranked papers."""

from __future__ import annotations

from collections.abc import Sequence

from daily_arxiv_agent.contracts import (
    BriefingTableRow,
    DailyBriefing,
    EvidenceSource,
    PaperBriefingItem,
    Provenance,
    Recommendation,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.llm.provider import create_llm_provider
from daily_arxiv_agent.skills.extraction import PaperExtractionSkill


class DailyBriefingSkill:
    """Generate a first daily briefing from ranked metadata and abstracts."""

    def __init__(self, *, provider: LLMProvider | None = None) -> None:
        self.provider = provider or create_llm_provider()

    def generate(
        self,
        *,
        topic: str,
        recommendations: Sequence[Recommendation],
        extraction_results: Sequence[SkillResult[PaperBriefingItem]] | None = None,
    ) -> SkillResult[DailyBriefing]:
        if not recommendations:
            briefing = DailyBriefing(
                topic=topic,
                executive_summary=f"No ranked papers were available for '{topic}'.",
                evidence_source=EvidenceSource.METADATA,
            )
            return SkillResult[DailyBriefing](
                status=SkillStatus.EMPTY,
                data=briefing,
                evidence_source=EvidenceSource.METADATA,
                message="No ranked papers are available for a daily briefing.",
                metadata={"topic": topic},
            )

        if extraction_results is None:
            extraction_skill = PaperExtractionSkill(provider=self.provider)
            extraction_results = [
                extraction_skill.extract(recommendation, topic=topic)
                for recommendation in recommendations
            ]
        extracted_items = [
            result.data for result in extraction_results if result.data is not None
        ]
        table = [_table_row(recommendation) for recommendation in recommendations]
        evidence_source = _combined_evidence(extracted_items)
        provenance = [recommendation.paper.provenance for recommendation in recommendations]
        extraction_errors = [
            result.error
            for result in extraction_results
            if result.status in {SkillStatus.FALLBACK, SkillStatus.ERROR}
            and result.error is not None
        ]

        try:
            executive_summary = self.provider.summarize_briefing(
                topic=topic,
                items=extracted_items,
            )
        except Exception as exc:
            briefing = _briefing(
                topic=topic,
                executive_summary=_fallback_summary(topic, extracted_items),
                table=table,
                items=extracted_items,
                evidence_source=evidence_source,
                provenance=provenance,
            )
            return SkillResult[DailyBriefing](
                status=SkillStatus.FALLBACK,
                data=briefing,
                evidence_source=evidence_source,
                provenance=provenance,
                error=SkillError(
                    code="llm_briefing_failed",
                    message=f"LLM briefing generation failed: {exc}",
                    retryable=True,
                ),
                message="Using deterministic fallback briefing.",
                metadata={"topic": topic},
            )

        briefing = _briefing(
            topic=topic,
            executive_summary=executive_summary,
            table=table,
            items=extracted_items,
            evidence_source=evidence_source,
            provenance=provenance,
        )
        if extraction_errors:
            codes = ", ".join(sorted({error.code for error in extraction_errors}))
            messages = "; ".join(error.message for error in extraction_errors)
            return SkillResult[DailyBriefing](
                status=SkillStatus.FALLBACK,
                data=briefing,
                evidence_source=evidence_source,
                provenance=provenance,
                error=SkillError(
                    code=codes,
                    message=messages,
                    retryable=any(error.retryable for error in extraction_errors),
                ),
                message="Using fallback extraction for one or more briefing items.",
                metadata={"topic": topic},
            )
        return SkillResult[DailyBriefing](
            status=SkillStatus.SUCCESS,
            data=briefing,
            evidence_source=evidence_source,
            provenance=provenance,
            metadata={"topic": topic},
        )


def _briefing(
    *,
    topic: str,
    executive_summary: str,
    table: list[BriefingTableRow],
    items: list[PaperBriefingItem],
    evidence_source: EvidenceSource,
    provenance: list[Provenance],
) -> DailyBriefing:
    return DailyBriefing(
        topic=topic,
        executive_summary=executive_summary,
        summary_table=table,
        highlighted_paper=items[0] if items else None,
        items=items,
        evidence_source=evidence_source,
        provenance=provenance,
    )


def _table_row(recommendation: Recommendation) -> BriefingTableRow:
    return BriefingTableRow(
        rank=recommendation.rank,
        paper_id=recommendation.paper.paper_id,
        title=recommendation.paper.title,
        score=recommendation.score,
        key_reason=recommendation.rationale,
        evidence_source=recommendation.evidence_source,
        arxiv_url=recommendation.paper.arxiv_url,
    )


def _combined_evidence(items: Sequence[PaperBriefingItem]) -> EvidenceSource:
    if any(item.evidence_source == EvidenceSource.ABSTRACT for item in items):
        return EvidenceSource.ABSTRACT
    return EvidenceSource.METADATA


def _fallback_summary(topic: str, items: Sequence[PaperBriefingItem]) -> str:
    return (
        f"Deterministic fallback briefing for '{topic}' includes {len(items)} "
        "ranked paper(s)."
    )
