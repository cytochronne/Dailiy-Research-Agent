from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    FeedbackEvent,
    FeedbackValue,
    PaperMetadata,
    Provenance,
    SkillStatus,
)
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill


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


def test_seed_preference_ranks_similar_papers_without_explicit_topic() -> None:
    similar = make_paper(
        "2604.00001",
        "Agent Workflows for Research Paper Recommendation",
        "Daily briefing systems can rank papers using agent preference signals.",
    )
    unrelated = make_paper(
        "2604.00002",
        "A Survey of Compiler Register Allocation",
        "This work studies low-level optimization in compilers.",
    )
    preference_result = SeedParsingSkill(metadata_client=None).build_preference(
        ["Agent workflows for research paper recommendation"]
    )
    preference = preference_result.data

    result = TopicRankingSkill().rank(
        [unrelated, similar],
        seed_preference=preference,
        top_k=2,
    )

    assert result.status == SkillStatus.SUCCESS
    recommendations = result.data or []
    assert [item.paper.paper_id for item in recommendations] == ["2604.00001", "2604.00002"]
    assert recommendations[0].score > recommendations[1].score
    assert "seed-paper similarity" in recommendations[0].rationale.lower()
    assert result.metadata["ranking_mode"] == "seed"


def test_hybrid_topic_and_seed_ranking_combines_both_rationales() -> None:
    paper = make_paper(
        "2604.00001",
        "Agent Workflows for Research Paper Recommendation",
        "Daily briefing systems can rank papers using agent preference signals.",
    )
    preference = SeedParsingSkill(metadata_client=None).build_preference(
        ["research paper recommendation"]
    ).data

    result = TopicRankingSkill().rank(
        [paper],
        topic="agent briefing",
        seed_preference=preference,
        top_k=1,
    )

    assert result.status == SkillStatus.SUCCESS
    recommendation = (result.data or [])[0]
    assert "matched explicit terms" in recommendation.rationale.lower()
    assert "seed-paper similarity" in recommendation.rationale.lower()
    assert result.metadata["ranking_mode"] == "hybrid_topic_seed"


def test_feedback_events_can_refine_a_later_ranking_call() -> None:
    liked = make_paper(
        "2604.00001",
        "Agent Workflows for Research Paper Recommendation",
        "Daily briefing systems can rank papers using agent preference signals.",
    )
    similar = make_paper(
        "2604.00002",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    unrelated = make_paper(
        "2604.00003",
        "A Survey of Compiler Register Allocation",
        "This work studies low-level optimization in compilers.",
    )
    feedback = FeedbackEvent(
        profile_id="demo",
        recommendation_run_id="run-1",
        paper_id=liked.paper_id,
        value=FeedbackValue.LIKE,
        paper=liked,
    )

    result = TopicRankingSkill().rank(
        [unrelated, similar],
        feedback_events=[feedback],
        top_k=2,
    )

    assert result.status == SkillStatus.SUCCESS
    recommendations = result.data or []
    assert [item.paper.paper_id for item in recommendations] == ["2604.00002", "2604.00003"]
    assert "feedback adjustment" in recommendations[0].rationale.lower()
    assert result.metadata["ranking_mode"] == "feedback"
