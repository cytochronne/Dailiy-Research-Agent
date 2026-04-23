from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    Recommendation,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.extraction import PaperExtractionSkill


def make_recommendation(abstract: str | None = "We propose an agent workflow for daily research briefings.") -> Recommendation:
    paper = PaperMetadata(
        paper_id="2604.00001",
        title="Explainable Agents for Daily Research Briefings",
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=["cs.LG"],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url="https://arxiv.org/abs/2604.00001",
        pdf_url="https://arxiv.org/pdf/2604.00001",
        provenance=Provenance(
            source="arxiv",
            source_url="https://arxiv.org/abs/2604.00001",
            query="agent briefing",
        ),
    )
    return Recommendation(
        paper=paper,
        rank=1,
        score=7.5,
        rationale="Matched explicit terms: agent, briefing.",
        evidence_source=EvidenceSource.ABSTRACT if abstract else EvidenceSource.METADATA,
    )


class FailingProvider:
    def extract_paper(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("provider unavailable")

    def summarize_briefing(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("provider unavailable")


def test_fake_llm_extraction_returns_expected_structure() -> None:
    result = PaperExtractionSkill(provider=FakeLLMProvider()).extract(
        make_recommendation(),
        topic="agent briefing",
    )

    assert result.status == SkillStatus.SUCCESS
    item = result.data
    assert item is not None
    assert item.paper_id == "2604.00001"
    assert item.summary
    assert item.contributions
    assert item.methods
    assert "agent" in item.relevance_rationale.lower()
    assert item.evidence_source == EvidenceSource.ABSTRACT
    assert str(item.arxiv_url) == "https://arxiv.org/abs/2604.00001"


def test_missing_abstract_labels_metadata_and_avoids_fabricated_methods() -> None:
    result = PaperExtractionSkill(provider=FakeLLMProvider()).extract(
        make_recommendation(abstract=None),
        topic="agent briefing",
    )

    item = result.data
    assert result.status == SkillStatus.SUCCESS
    assert item is not None
    assert item.evidence_source == EvidenceSource.METADATA
    assert item.methods == []
    assert "metadata only" in item.summary.lower()


def test_llm_adapter_failure_returns_fallback_extraction() -> None:
    result = PaperExtractionSkill(provider=FailingProvider()).extract(
        make_recommendation(),
        topic="agent briefing",
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "llm_extraction_failed"
    assert result.data is not None
    assert result.data.evidence_source == EvidenceSource.METADATA
