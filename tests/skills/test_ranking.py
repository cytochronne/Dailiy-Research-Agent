from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    FeedbackEvent,
    FeedbackValue,
    PaperMetadata,
    Provenance,
    QueryPlan,
    QueryPlannerMode,
    QueryPlannerProvenance,
    QueryPlanVariant,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SearchMode,
    SkillStatus,
)
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill


def make_paper(
    paper_id: str,
    title: str,
    abstract: str | None,
    category: str = "cs.LG",
    published_date: date = date(2026, 4, 20),
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=[category],
        published_date=published_date,
        updated_date=published_date,
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


def test_phrase_and_title_term_match_outranks_single_abstract_term() -> None:
    exact = make_paper(
        "2604.10001",
        "Multimodal LLM Agents for Robotic Manipulation",
        "A full system for planning and control.",
    )
    weak = make_paper(
        "2604.10002",
        "A Benchmark for General AI Systems",
        "The benchmark mentions robotic agents in one evaluation.",
    )

    result = TopicRankingSkill().rank(
        [weak, exact],
        topic="multimodal llm agents for robotic manipulation",
        top_k=2,
    )

    recommendations = result.data or []
    assert [item.paper.paper_id for item in recommendations] == [
        "2604.10001",
        "2604.10002",
    ]
    assert recommendations[0].score_breakdown is not None
    assert recommendations[1].score_breakdown is not None
    assert recommendations[0].score_breakdown.phrase > 0
    assert (
        recommendations[0].score_breakdown.lexical
        > recommendations[1].score_breakdown.lexical
    )


def test_relevance_sorted_retrieval_source_gets_small_order_boost() -> None:
    date_first = make_paper(
        "2604.10001",
        "Agent Systems for Research",
        "Agent systems help rank research papers.",
    )
    relevance_first = make_paper(
        "2604.10002",
        "Agent Systems for Research",
        "Agent systems help rank research papers.",
    )
    source_metadata = {
        date_first.paper_id: [
            RetrievalSourceMetadata(
                variant_label="recent",
                sort_by="submittedDate",
                variant_index=1,
                position=0,
                first_seen_order=0,
            )
        ],
        relevance_first.paper_id: [
            RetrievalSourceMetadata(
                variant_label="broad_terms",
                sort_by="relevance",
                variant_index=0,
                position=0,
                first_seen_order=1,
            )
        ],
    }

    result = TopicRankingSkill().rank(
        [date_first, relevance_first],
        topic="agent systems",
        retrieval_source_metadata_by_paper_id=source_metadata,
        top_k=2,
    )

    recommendations = result.data or []
    assert [item.paper.paper_id for item in recommendations] == [
        "2604.10002",
        "2604.10001",
    ]
    assert recommendations[0].score_breakdown is not None
    assert recommendations[1].score_breakdown is not None
    assert (
        recommendations[0].score_breakdown.query_source
        > recommendations[1].score_breakdown.query_source
    )


def test_retrieval_source_metadata_accepts_json_dict_shape() -> None:
    date_first = make_paper(
        "2604.10001",
        "Agent Systems for Research",
        "Agent systems help rank research papers.",
    )
    relevance_first = make_paper(
        "2604.10002",
        "Agent Systems for Research",
        "Agent systems help rank research papers.",
    )
    source_metadata = {
        date_first.paper_id: [
            RetrievalSourceMetadata(
                variant_label="recent",
                sort_by="submittedDate",
                variant_index=1,
                position=0,
                first_seen_order=0,
            ).model_dump(mode="json")
        ],
        relevance_first.paper_id: [
            RetrievalSourceMetadata(
                variant_label="broad_terms",
                sort_by="relevance",
                variant_index=0,
                position=0,
                first_seen_order=1,
            ).model_dump(mode="json")
        ],
    }

    result = TopicRankingSkill().rank(
        [date_first, relevance_first],
        topic="agent systems",
        retrieval_source_metadata_by_paper_id=source_metadata,
        top_k=2,
    )

    recommendations = result.data or []
    assert result.status == SkillStatus.SUCCESS
    assert [item.paper.paper_id for item in recommendations] == [
        "2604.10002",
        "2604.10001",
    ]
    assert recommendations[0].score_breakdown is not None
    assert recommendations[0].score_breakdown.query_source > 0


def test_seed_preference_and_feedback_remain_score_signals() -> None:
    liked = make_paper(
        "2604.10001",
        "Agent Workflows for Research Paper Recommendation",
        "Daily briefing systems can rank papers using agent preference signals.",
    )
    candidate = make_paper(
        "2604.10002",
        "Feedback Agents for Paper Recommendation",
        "Research briefing agents use feedback signals to rank papers.",
    )
    preference = SeedParsingSkill(metadata_client=None).build_preference(
        ["research paper recommendation"]
    ).data
    feedback = FeedbackEvent(
        profile_id="demo",
        recommendation_run_id="run-1",
        paper_id=liked.paper_id,
        value=FeedbackValue.LIKE,
        paper=liked,
    )

    result = TopicRankingSkill().rank(
        [candidate],
        topic="agent briefing",
        seed_preference=preference,
        feedback_events=[feedback],
        top_k=1,
    )

    recommendation = (result.data or [])[0]
    assert recommendation.score_breakdown is not None
    assert recommendation.score_breakdown.seed_similarity > 0
    assert recommendation.score_breakdown.feedback > 0
    assert set(result.metadata["score_signals"]) >= {
        "lexical",
        "seed_similarity",
        "feedback",
    }


def test_recency_boost_is_bounded_below_clear_topic_relevance() -> None:
    clear_match = make_paper(
        "2604.10001",
        "Robotic Manipulation with Multimodal LLM Agents",
        "This paper studies robotic manipulation with multimodal LLM agents.",
        published_date=date(2025, 4, 20),
    )
    recent_weak_match = make_paper(
        "2604.10002",
        "Recent Notes on Robotic Benchmarks",
        "A short benchmark note.",
        published_date=date(2026, 4, 20),
    )

    result = TopicRankingSkill().rank(
        [recent_weak_match, clear_match],
        topic="robotic manipulation multimodal llm agents",
        top_k=2,
    )

    recommendations = result.data or []
    assert [item.paper.paper_id for item in recommendations] == [
        "2604.10001",
        "2604.10002",
    ]
    assert recommendations[0].score_breakdown is not None
    assert recommendations[1].score_breakdown is not None
    assert (
        recommendations[1].score_breakdown.recency
        > recommendations[0].score_breakdown.recency
    )


def test_category_fit_does_not_make_topicless_match_qualify_as_relevant() -> None:
    category_only = make_paper(
        "2604.10001",
        "A Survey of Compiler Register Allocation",
        "This work studies low-level optimization in compilers.",
        category="cs.LG",
    )

    result = TopicRankingSkill().rank(
        [category_only],
        topic="robotic manipulation agents",
        retrieval_query=RetrievalQuery(
            topic="robotic manipulation agents",
            category="cs.LG",
            search_mode=SearchMode.BROAD,
        ),
        top_k=1,
    )

    recommendation = (result.data or [])[0]
    assert recommendation.score_breakdown is not None
    assert recommendation.score_breakdown.category > 0
    assert recommendation.score_breakdown.evidence_score == 0
    assert recommendation.score_breakdown.fallback is True
    assert "fallback" in recommendation.rationale.lower()


def test_category_date_only_retrieval_uses_category_recency_mode() -> None:
    old = make_paper(
        "2604.10001",
        "Older Learning Systems",
        "A category-only retrieval result.",
        category="cs.LG",
        published_date=date(2026, 4, 18),
    )
    recent = make_paper(
        "2604.10002",
        "Recent Learning Systems",
        "A category-only retrieval result.",
        category="cs.LG",
        published_date=date(2026, 4, 20),
    )

    result = TopicRankingSkill().rank(
        [old, recent],
        retrieval_query=RetrievalQuery(
            category="cs.LG",
            start_date=date(2026, 4, 18),
            end_date=date(2026, 4, 20),
        ),
        query_plan=QueryPlan(
            search_mode=SearchMode.BROAD,
            planner=QueryPlannerProvenance(
                requested_mode=QueryPlannerMode.DETERMINISTIC,
                source="deterministic",
            ),
            variants=[
                QueryPlanVariant(
                    label="filters",
                    search_query="cat:cs.LG",
                    sort_by="submittedDate",
                )
            ],
        ),
        top_k=2,
    )

    recommendations = result.data or []
    assert result.status == SkillStatus.SUCCESS
    assert result.metadata["ranking_mode"] == "category_recency"
    assert [item.paper.paper_id for item in recommendations] == [
        "2604.10002",
        "2604.10001",
    ]
    assert "category/date" in recommendations[0].rationale.lower()


def test_nonmatching_top_k_fill_is_labeled_as_fallback() -> None:
    matching = make_paper(
        "2604.10001",
        "Agent Briefings",
        "Agent ranking for research briefings.",
    )
    unrelated = make_paper(
        "2604.10002",
        "Compiler Register Allocation",
        "Low-level compiler optimization for register pressure.",
    )

    result = TopicRankingSkill().rank(
        [unrelated, matching],
        topic="agent briefing",
        top_k=2,
    )

    recommendations = result.data or []
    assert recommendations[0].score_breakdown is not None
    assert recommendations[1].score_breakdown is not None
    assert recommendations[0].score_breakdown.fallback is False
    assert recommendations[1].score_breakdown.fallback is True
    assert "fallback" in recommendations[1].rationale.lower()


def test_missing_ranking_inputs_without_retrieval_context_still_errors() -> None:
    paper = make_paper(
        "2604.10001",
        "A Stored Paper",
        "No ranking context is available.",
    )

    result = TopicRankingSkill().rank([paper], top_k=1)

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "ranking_input_missing"
