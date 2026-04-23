from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    Recommendation,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.briefing import DailyBriefingSkill


def make_recommendation(
    paper_id: str,
    rank: int,
    title: str = "Explainable Agents for Daily Research Briefings",
) -> Recommendation:
    paper = PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract="We propose an agent workflow for daily research briefings.",
        categories=["cs.LG"],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query="agent briefing",
        ),
    )
    return Recommendation(
        paper=paper,
        rank=rank,
        score=8.0 - rank,
        rationale="Matched explicit terms: agent, briefing.",
        evidence_source=EvidenceSource.ABSTRACT,
    )


class FailingSummaryProvider(FakeLLMProvider):
    def summarize_briefing(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("summary unavailable")


class FailingExtractionProvider(FakeLLMProvider):
    def extract_paper(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("extraction unavailable")


def test_briefing_generation_includes_summary_table_highlight_and_all_references() -> None:
    recommendations = [
        make_recommendation("2604.00001", 1),
        make_recommendation("2604.00002", 2, "Daily Research Recommendation Workflows"),
    ]

    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="agent briefing",
        recommendations=recommendations,
    )

    assert result.status == SkillStatus.SUCCESS
    briefing = result.data
    assert briefing is not None
    assert briefing.topic == "agent briefing"
    assert briefing.executive_summary
    assert briefing.highlighted_paper is not None
    assert briefing.highlighted_paper.paper_id == "2604.00001"
    assert [row.paper_id for row in briefing.summary_table] == ["2604.00001", "2604.00002"]
    assert [item.paper_id for item in briefing.items] == ["2604.00001", "2604.00002"]
    assert all(row.evidence_source == EvidenceSource.ABSTRACT for row in briefing.summary_table)


def test_briefing_generation_handles_empty_recommendations() -> None:
    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="agent briefing",
        recommendations=[],
    )

    assert result.status == SkillStatus.EMPTY
    assert result.data is not None
    assert result.data.summary_table == []
    assert result.message == "No ranked papers are available for a daily briefing."


def test_llm_adapter_failure_returns_fallback_briefing() -> None:
    result = DailyBriefingSkill(provider=FailingSummaryProvider()).generate(
        topic="agent briefing",
        recommendations=[make_recommendation("2604.00001", 1)],
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "llm_briefing_failed"
    assert result.data is not None
    assert result.data.highlighted_paper is not None


def test_extraction_failure_propagates_to_fallback_briefing_status() -> None:
    result = DailyBriefingSkill(provider=FailingExtractionProvider()).generate(
        topic="agent briefing",
        recommendations=[make_recommendation("2604.00001", 1)],
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "llm_extraction_failed"
    assert result.message == "Using fallback extraction for one or more briefing items."
    assert result.data is not None
    assert result.data.highlighted_paper is not None
