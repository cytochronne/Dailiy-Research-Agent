from datetime import date, datetime, timezone

from daily_arxiv_agent.contracts import (
    EmbeddingInputRole,
    FeedbackEvent,
    FeedbackRefinementStatus,
    FeedbackValue,
    PaperMetadata,
    Provenance,
    RankingScoreBreakdown,
    Recommendation,
    SemanticSimilarityDetail,
    SkillStatus,
)
from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.embeddings.base import EmbeddingProviderError
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
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


def make_semantic_recommendation(
    paper: PaperMetadata,
    rank: int,
    score: float,
) -> Recommendation:
    return Recommendation(
        paper=paper,
        rank=rank,
        score=score,
        rationale="Initial semantic ranking.",
        score_breakdown=RankingScoreBreakdown(
            semantic_seed=score,
            total=score,
            evidence_score=0.8,
            semantic_similarities=[
                SemanticSimilarityDetail(
                    source_id="seed:demo",
                    target_id=paper.paper_id,
                    similarity=0.8,
                    source_role=EmbeddingInputRole.SEED,
                    target_role=EmbeddingInputRole.CANDIDATE,
                    score=score,
                )
            ],
            signals=["semantic_seed"],
        ),
        semantic_context=semantic_context(),
    )


def semantic_context() -> dict[str, object]:
    return {
        "semantic_context": {
            "input_version": "paper-metadata-v1",
            "similarity_metric": "cosine",
            "aggregation": "max_per_feedback",
            "provider": "fake",
            "model": "fake-feedback",
            "dimensions": 2,
        },
        "semantic_provider": {
            "provider": "fake",
            "provider_mode": "fake",
            "provider_label": "fake:fake-feedback",
            "model": "fake-feedback",
            "dimensions": 2,
        },
        "embedding_cache": {"enabled": False},
    }


def embedding_text(paper: PaperMetadata) -> str:
    return " ".join(
        part
        for part in [
            paper.title,
            paper.abstract or "",
            " ".join(paper.categories),
        ]
        if part
    )


def semantic_feedback_skill(
    vector_map: dict[str, list[float]],
) -> FeedbackRefinementSkill:
    return FeedbackRefinementSkill(
        embedding_provider=FakeEmbeddingProvider(dimensions=2, vector_map=vector_map),
        config=AppConfig(
            embedding_provider="fake",
            embedding_model="fake-feedback",
            embedding_dimensions=2,
            embedding_cache_enabled=False,
        ),
    )


class RaisingEmbeddingProvider:
    def embed_texts(self, texts):  # noqa: ANN001, ANN201
        raise EmbeddingProviderError("semantic feedback provider unavailable")


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


def test_semantic_like_uses_embedding_similarity_and_records_influence() -> None:
    anchor = make_paper(
        "2604.10001",
        "Autonomous Lab Optimization",
        "Closed-loop experiments improve molecular discovery.",
    )
    similar = make_paper(
        "2604.10002",
        "Active Learning for Molecular Experiments",
        "Bayesian experiment selection accelerates chemical discovery.",
    )
    unrelated = make_paper(
        "2604.10003",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    skill = semantic_feedback_skill(
        {
            embedding_text(anchor): [1.0, 0.0],
            embedding_text(similar): [1.0, 0.0],
            embedding_text(unrelated): [0.0, 1.0],
        }
    )

    result = skill.refine(
        [
            make_semantic_recommendation(unrelated, rank=1, score=2.0),
            make_semantic_recommendation(similar, rank=2, score=1.0),
        ],
        feedback=[{"paper_id": anchor.paper_id, "value": "like"}],
        papers=[anchor],
        recommendation_run_id="run-semantic-feedback",
    )

    assert result.status == SkillStatus.SUCCESS
    refined = result.data or []
    assert [item.paper.paper_id for item in refined] == ["2604.10002", "2604.10003"]
    top = refined[0]
    assert top.previous_rank == 2
    assert top.rank_delta == 1
    assert top.score_delta == 6.0
    assert top.refinement_status == FeedbackRefinementStatus.APPLIED
    assert top.score_breakdown is not None
    assert top.score_breakdown.semantic_seed == 1.0
    assert top.score_breakdown.feedback == 6.0
    assert top.score_breakdown.total == 7.0
    influence = top.feedback_influences[0]
    assert influence.source_paper_id == anchor.paper_id
    assert influence.target_paper_id == similar.paper_id
    assert influence.similarity == 1.0
    assert influence.signed_score_delta == 6.0
    assert influence.value == FeedbackValue.LIKE
    assert influence.refinement_status == FeedbackRefinementStatus.APPLIED
    assert result.metadata["refinement_mode"] == "semantic_feedback"
    assert result.metadata["influence_count"] == 1


def test_semantic_feedback_accepts_flat_originating_context() -> None:
    anchor = make_paper(
        "2604.10101",
        "Autonomous Lab Optimization",
        "Closed-loop experiments improve molecular discovery.",
    )
    candidate = make_paper(
        "2604.10102",
        "Active Learning for Molecular Experiments",
        "Bayesian experiment selection accelerates chemical discovery.",
    )
    skill = semantic_feedback_skill(
        {
            embedding_text(anchor): [1.0, 0.0],
            embedding_text(candidate): [1.0, 0.0],
        }
    )
    flat_context = {
        "input_version": "paper-metadata-v1",
        "provider": "fake",
        "model": "flat-origin-model",
        "dimensions": 2,
    }

    result = skill.refine(
        [make_semantic_recommendation(candidate, rank=1, score=1.0)],
        feedback=[{"paper_id": anchor.paper_id, "value": "like"}],
        papers=[anchor],
        recommendation_run_id="run-flat-context",
        semantic_context=flat_context,
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.metadata["semantic_provider"]["model"] == "flat-origin-model"
    assert (result.data or [])[0].feedback_influences


def test_semantic_dislike_uses_embedding_similarity_and_records_negative_delta() -> None:
    anchor = make_paper(
        "2604.11001",
        "Autonomous Lab Optimization",
        "Closed-loop experiments improve molecular discovery.",
    )
    similar = make_paper(
        "2604.11002",
        "Active Learning for Molecular Experiments",
        "Bayesian experiment selection accelerates chemical discovery.",
    )
    unrelated = make_paper(
        "2604.11003",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )
    skill = semantic_feedback_skill(
        {
            embedding_text(anchor): [1.0, 0.0],
            embedding_text(similar): [1.0, 0.0],
            embedding_text(unrelated): [0.0, 1.0],
        }
    )

    result = skill.refine(
        [
            make_semantic_recommendation(similar, rank=1, score=4.0),
            make_semantic_recommendation(unrelated, rank=2, score=2.0),
        ],
        feedback=[{"paper_id": anchor.paper_id, "value": "dislike"}],
        papers=[anchor],
        recommendation_run_id="run-semantic-feedback",
    )

    assert result.status == SkillStatus.SUCCESS
    refined = result.data or []
    assert [item.paper.paper_id for item in refined] == ["2604.11003", "2604.11002"]
    moved_down = refined[1]
    assert moved_down.previous_rank == 1
    assert moved_down.rank_delta == -1
    assert moved_down.score_delta == -6.0
    influence = moved_down.feedback_influences[0]
    assert influence.signed_score_delta == -6.0
    assert influence.value == FeedbackValue.DISLIKE


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


def test_semantic_conflicting_feedback_uses_latest_event_for_same_paper() -> None:
    anchor = make_paper(
        "2604.12001",
        "Autonomous Lab Optimization",
        "Closed-loop experiments improve molecular discovery.",
    )
    candidate = make_paper(
        "2604.12002",
        "Active Learning for Molecular Experiments",
        "Bayesian experiment selection accelerates chemical discovery.",
    )
    like = FeedbackEvent(
        event_id="event-1",
        profile_id="default",
        recommendation_run_id="run-1",
        paper_id=anchor.paper_id,
        value=FeedbackValue.LIKE,
        created_at=datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc),
        paper=anchor,
    )
    dislike = FeedbackEvent(
        event_id="event-2",
        profile_id="default",
        recommendation_run_id="run-1",
        paper_id=anchor.paper_id,
        value=FeedbackValue.DISLIKE,
        created_at=datetime(2026, 4, 20, 12, 1, tzinfo=timezone.utc),
        paper=anchor,
    )
    skill = semantic_feedback_skill(
        {
            embedding_text(anchor): [1.0, 0.0],
            embedding_text(candidate): [1.0, 0.0],
        }
    )

    result = skill.refine(
        [make_semantic_recommendation(candidate, rank=1, score=4.0)],
        feedback=[like, dislike],
        recommendation_run_id="run-1",
    )

    refined = (result.data or [])[0]
    assert result.status == SkillStatus.SUCCESS
    assert refined.score_delta == -6.0
    assert len(refined.feedback_influences) == 1
    assert refined.feedback_influences[0].event_id == "event-2"
    assert refined.feedback_influences[0].value == FeedbackValue.DISLIKE


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


def test_semantic_feedback_missing_metadata_is_recorded_without_score_delta() -> None:
    candidate = make_paper(
        "2604.13002",
        "Active Learning for Molecular Experiments",
        "Bayesian experiment selection accelerates chemical discovery.",
    )
    provider = FakeEmbeddingProvider(dimensions=2)
    skill = FeedbackRefinementSkill(
        embedding_provider=provider,
        config=AppConfig(
            embedding_provider="fake",
            embedding_model="fake-feedback",
            embedding_dimensions=2,
            embedding_cache_enabled=False,
        ),
    )

    result = skill.refine(
        [make_semantic_recommendation(candidate, rank=1, score=4.0)],
        feedback=[{"paper_id": "2604.13999", "value": "like"}],
        recommendation_run_id="run-semantic-missing-metadata",
    )

    refined = (result.data or [])[0]
    assert result.status == SkillStatus.SUCCESS
    assert refined.rank == 1
    assert refined.score == 4.0
    assert refined.score_delta == 0.0
    assert refined.refinement_status == FeedbackRefinementStatus.SKIPPED
    assert refined.feedback_influences == []
    assert provider.calls == []
    assert result.metadata["refinement_mode"] == "semantic_feedback"
    assert result.metadata["skipped_feedback_count"] == 1


def test_semantic_feedback_failure_preserves_original_recommendations() -> None:
    anchor = make_paper(
        "2604.14001",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    candidate = make_paper(
        "2604.14002",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    skill = FeedbackRefinementSkill(
        embedding_provider=RaisingEmbeddingProvider(),
        config=AppConfig(
            embedding_provider="fake",
            embedding_model="fake-feedback",
            embedding_dimensions=2,
            embedding_cache_enabled=False,
        ),
    )

    result = skill.refine(
        [make_semantic_recommendation(candidate, rank=1, score=4.0)],
        feedback=[{"paper_id": anchor.paper_id, "value": "like"}],
        papers=[anchor],
        recommendation_run_id="run-semantic-provider-failed",
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "semantic_feedback_provider_failed"
    refined = result.data or []
    assert len(refined) == 1
    assert refined[0].rank == 1
    assert refined[0].score == 4.0
    assert refined[0].previous_rank is None
    assert refined[0].score_delta is None
    assert refined[0].refinement_status == FeedbackRefinementStatus.FAILED
    assert refined[0].feedback_error is not None
    assert refined[0].feedback_error.code == "semantic_feedback_provider_failed"
    assert refined[0].feedback_influences == []
    assert result.metadata["feedback_error"]["code"] == (
        "semantic_feedback_provider_failed"
    )


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
