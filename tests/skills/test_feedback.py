from datetime import date, datetime, timezone

from daily_arxiv_agent.contracts import (
    FeedbackEvent,
    FeedbackValue,
    PaperMetadata,
    Provenance,
    Recommendation,
    SkillStatus,
)
from daily_arxiv_agent.skills.feedback import FeedbackRefinementSkill
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


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


def make_recommendation(paper: PaperMetadata, rank: int, score: float) -> Recommendation:
    return Recommendation(
        paper=paper,
        rank=rank,
        score=score,
        rationale="Initial deterministic ranking.",
    )


def test_liking_a_paper_moves_similar_papers_up() -> None:
    anchor = make_paper(
        "2604.00001",
        "Agent Workflows for Research Recommendation",
        "Daily briefing agents rank research papers from preference signals.",
    )
    similar = make_paper(
        "2604.00002",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    unrelated = make_paper(
        "2604.00003",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    recommendations = [
        make_recommendation(unrelated, rank=1, score=2.0),
        make_recommendation(similar, rank=2, score=1.0),
    ]

    result = FeedbackRefinementSkill().refine(
        recommendations,
        feedback=[{"paper_id": anchor.paper_id, "value": "like"}],
        papers=[anchor],
        recommendation_run_id="run-1",
    )

    assert result.status == SkillStatus.SUCCESS
    refined = result.data or []
    assert [item.paper.paper_id for item in refined] == ["2604.00002", "2604.00003"]
    assert refined[0].previous_rank == 2
    assert refined[0].rank_delta == 1
    assert refined[0].score_delta is not None
    assert refined[0].score_delta > 0
    assert "liked 2604.00001" in refined[0].rationale


def test_disliking_a_paper_moves_similar_papers_down() -> None:
    anchor = make_paper(
        "2604.00001",
        "Agent Workflows for Research Recommendation",
        "Daily briefing agents rank research papers from preference signals.",
    )
    similar = make_paper(
        "2604.00002",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    unrelated = make_paper(
        "2604.00003",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    recommendations = [
        make_recommendation(similar, rank=1, score=4.0),
        make_recommendation(unrelated, rank=2, score=2.0),
    ]

    result = FeedbackRefinementSkill().refine(
        recommendations,
        feedback=[{"paper_id": anchor.paper_id, "value": "dislike"}],
        papers=[anchor],
        recommendation_run_id="run-1",
    )

    assert result.status == SkillStatus.SUCCESS
    refined = result.data or []
    assert [item.paper.paper_id for item in refined] == ["2604.00003", "2604.00002"]
    assert refined[1].previous_rank == 1
    assert refined[1].rank_delta == -1
    assert refined[1].score_delta is not None
    assert refined[1].score_delta < 0
    assert "disliked 2604.00001" in refined[1].rationale


def test_refined_recommendations_include_before_after_fields() -> None:
    paper = make_paper(
        "2604.00001",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    recommendations = [make_recommendation(paper, rank=1, score=2.0)]

    result = FeedbackRefinementSkill().refine(
        recommendations,
        feedback=[{"paper_id": paper.paper_id, "value": "like"}],
        recommendation_run_id="run-1",
    )

    refined = (result.data or [])[0]
    assert refined.previous_rank == 1
    assert refined.previous_score == 2.0
    assert refined.score_delta is not None
    assert refined.rank == 1
    assert "Previous rank: 1" in refined.rationale
    assert "score delta" in refined.rationale


def test_feedback_on_missing_paper_is_recorded_without_breaking_refinement() -> None:
    paper = make_paper(
        "2604.00002",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    recommendations = [make_recommendation(paper, rank=1, score=2.0)]
    skill = FeedbackRefinementSkill()

    record_result = skill.record_feedback(
        [{"paper_id": "2604.99999", "value": "like"}],
        recommendations=recommendations,
        recommendation_run_id="run-1",
    )
    refine_result = skill.refine(
        recommendations,
        feedback=[{"paper_id": "2604.99999", "value": "like"}],
        recommendation_run_id="run-1",
    )

    assert record_result.status == SkillStatus.SUCCESS
    assert (record_result.data or [])[0].paper is None
    assert refine_result.status == SkillStatus.SUCCESS
    assert (refine_result.data or [])[0].paper.paper_id == "2604.00002"


def test_conflicting_feedback_uses_latest_event_for_same_paper() -> None:
    paper = make_paper(
        "2604.00001",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    recommendations = [make_recommendation(paper, rank=1, score=2.0)]
    like = FeedbackEvent(
        event_id="event-1",
        profile_id="default",
        recommendation_run_id="run-1",
        paper_id=paper.paper_id,
        value=FeedbackValue.LIKE,
        created_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        paper=paper,
    )
    dislike = FeedbackEvent(
        event_id="event-2",
        profile_id="default",
        recommendation_run_id="run-1",
        paper_id=paper.paper_id,
        value=FeedbackValue.DISLIKE,
        created_at=datetime(2026, 4, 20, 12, 1, tzinfo=timezone.utc),
        paper=paper,
    )

    result = FeedbackRefinementSkill().refine(
        recommendations,
        feedback=[like, dislike],
        recommendation_run_id="run-1",
    )

    refined = (result.data or [])[0]
    assert refined.score_delta is not None
    assert refined.score_delta < 0
    assert "disliked 2604.00001" in refined.rationale
    assert "liked 2604.00001 moved similar papers up" not in refined.rationale


def test_conflicting_feedback_with_mixed_timezone_datetimes_does_not_crash() -> None:
    paper = make_paper(
        "2604.00001",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    recommendations = [make_recommendation(paper, rank=1, score=2.0)]
    like_aware = FeedbackEvent(
        event_id="event-1",
        profile_id="default",
        recommendation_run_id="run-1",
        paper_id=paper.paper_id,
        value=FeedbackValue.LIKE,
        created_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        paper=paper,
    )
    dislike_naive = FeedbackEvent(
        event_id="event-2",
        profile_id="default",
        recommendation_run_id="run-1",
        paper_id=paper.paper_id,
        value=FeedbackValue.DISLIKE,
        created_at=datetime(2026, 4, 20, 12, 1),
        paper=paper,
    )

    result = FeedbackRefinementSkill().refine(
        recommendations,
        feedback=[like_aware, dislike_naive],
        recommendation_run_id="run-1",
    )

    refined = (result.data or [])[0]
    assert result.status == SkillStatus.SUCCESS
    assert refined.score_delta is not None
    assert refined.score_delta < 0
    assert "disliked 2604.00001" in refined.rationale


def test_invalid_feedback_value_returns_structured_error() -> None:
    paper = make_paper(
        "2604.00001",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )

    result = FeedbackRefinementSkill().refine(
        [make_recommendation(paper, rank=1, score=2.0)],
        feedback=[{"paper_id": paper.paper_id, "value": "bookmark"}],
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "invalid_feedback_value"


def test_persisted_feedback_influences_later_recommendation_call(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
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
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    feedback_event = FeedbackEvent(
        profile_id="demo",
        recommendation_run_id="run-1",
        paper_id=liked.paper_id,
        value=FeedbackValue.LIKE,
        paper=liked,
    )

    store.save_feedback_event(feedback_event)
    result = TopicRankingSkill().rank(
        [unrelated, similar],
        feedback_events=store.list_feedback_events(
            profile_id="demo",
            recommendation_run_id="run-1",
        ),
        top_k=2,
    )

    assert result.status == SkillStatus.SUCCESS
    assert [item.paper.paper_id for item in result.data or []] == [
        "2604.00002",
        "2604.00003",
    ]
