from datetime import date

from daily_arxiv_agent.contracts import EvidenceSource, PaperMetadata, Provenance, SkillStatus
from daily_arxiv_agent.skills.ranking import TopicRankingSkill


def make_paper(
    paper_id: str,
    title: str,
    abstract: str | None,
    category: str = "cs.LG",
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=[category],
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


def test_keyword_query_ranks_matching_papers_above_unrelated_papers() -> None:
    matching = make_paper(
        "2604.00001",
        "Explainable Agents for Daily Research Briefings",
        "We study agent workflows for research-paper recommendation.",
    )
    unrelated = make_paper(
        "2604.00002",
        "A Survey of Compiler Register Allocation",
        "This work studies low-level optimization in compilers.",
    )

    result = TopicRankingSkill().rank([unrelated, matching], topic="agent briefing", top_k=2)

    assert result.status == SkillStatus.SUCCESS
    recommendations = result.data or []
    assert [item.paper.paper_id for item in recommendations] == ["2604.00001", "2604.00002"]
    assert recommendations[0].score > recommendations[1].score


def test_top_k_output_includes_rank_score_rationale_and_provenance() -> None:
    paper = make_paper(
        "2604.00001",
        "Explainable Agents for Daily Research Briefings",
        "Agent workflows for research-paper recommendation.",
    )

    result = TopicRankingSkill().rank([paper], topic="agent", top_k=5)

    assert result.status == SkillStatus.SUCCESS
    recommendations = result.data or []
    assert len(recommendations) == 1
    assert recommendations[0].rank == 1
    assert recommendations[0].score > 0
    assert "agent" in recommendations[0].rationale.lower()
    assert recommendations[0].paper.provenance.source == "arxiv"
    assert recommendations[0].evidence_source == EvidenceSource.ABSTRACT


def test_fewer_papers_than_top_k_returns_all_available_papers() -> None:
    papers = [
        make_paper("2604.00001", "Agent Briefings", "Agent ranking."),
        make_paper("2604.00002", "Research Workflows", "Daily research workflow."),
    ]

    result = TopicRankingSkill().rank(papers, topic="agent briefing", top_k=10)

    assert result.status == SkillStatus.SUCCESS
    assert len(result.data or []) == 2


def test_missing_abstract_uses_metadata_evidence_label() -> None:
    paper = make_paper("2604.00001", "Agent Briefings", None)

    result = TopicRankingSkill().rank([paper], topic="agent", top_k=1)

    recommendation = (result.data or [])[0]
    assert recommendation.evidence_source == EvidenceSource.METADATA
    assert result.evidence_source == EvidenceSource.METADATA
