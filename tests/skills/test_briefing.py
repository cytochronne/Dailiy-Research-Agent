from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    QueryPlan,
    QueryPlannerMode,
    QueryPlannerProvenance,
    QueryPlanVariant,
    RankingScoreBreakdown,
    Recommendation,
    RetrievalSourceMetadata,
    SearchMode,
    SkillStatus,
    TrendAssessmentStatus,
    TrendSignalStrength,
    TrendSignalType,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.briefing import DailyBriefingSkill


def make_recommendation(
    paper_id: str,
    rank: int,
    title: str = "Explainable Agents for Daily Research Briefings",
    *,
    abstract: str | None = "We propose an agent workflow for daily research briefings.",
    categories: list[str] | None = None,
    published_date: date = date(2026, 4, 20),
    score_breakdown: RankingScoreBreakdown | None = None,
) -> Recommendation:
    paper = PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=categories or ["cs.LG"],
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
    return Recommendation(
        paper=paper,
        rank=rank,
        score=8.0 - rank,
        rationale="Matched explicit terms: agent, briefing.",
        evidence_source=EvidenceSource.ABSTRACT,
        score_breakdown=score_breakdown,
    )


def make_candidate(
    paper_id: str,
    title: str,
    abstract: str | None,
    *,
    categories: list[str] | None = None,
    published_date: date = date(2026, 4, 20),
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=categories or ["cs.LG"],
        published_date=published_date,
        updated_date=published_date,
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query="robotics",
        ),
    )


def make_query_plan(
    *,
    required_terms: list[str] | None = None,
    optional_terms: list[str] | None = None,
    phrases: list[str] | None = None,
) -> QueryPlan:
    return QueryPlan(
        search_mode=SearchMode.BROAD,
        planner=QueryPlannerProvenance(
            requested_mode=QueryPlannerMode.DETERMINISTIC,
            source="deterministic",
        ),
        variants=[
            QueryPlanVariant(
                label="broad_all_terms",
                search_query="all:robotics",
                sort_by="relevance",
            )
        ],
        required_terms=required_terms or [],
        optional_terms=optional_terms or [],
        phrases=phrases or [],
    )


def make_source_metadata(
    variant_label: str,
    *,
    variant_index: int = 0,
    position: int = 0,
    first_seen_order: int = 0,
) -> list[RetrievalSourceMetadata]:
    return [
        RetrievalSourceMetadata(
            variant_label=variant_label,
            sort_by="relevance",
            variant_index=variant_index,
            position=position,
            first_seen_order=first_seen_order,
            query=f"query {variant_label}",
        )
    ]


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


def test_candidate_pool_trends_report_repeated_topics_categories_and_coverage() -> None:
    candidates = [
        make_candidate(
            "2604.10001",
            "Robotic Manipulation with Embodied Control Policies",
            "Robotic manipulation benefits from embodied control and vision policies.",
            categories=["cs.RO", "cs.LG"],
        ),
        make_candidate(
            "2604.10002",
            "Embodied Control for Robotic Manipulation",
            "Embodied control improves robotic manipulation for household tasks.",
            categories=["cs.RO"],
        ),
        make_candidate(
            "2604.10003",
            "Robotic Manipulation via Multimodal Policy Learning",
            "Robotic manipulation systems combine embodied control and imitation.",
            categories=["cs.RO", "cs.AI"],
        ),
        make_candidate(
            "2604.10004",
            "Scene Understanding for Household Robots",
            "Embodied control appears in robot planning and perception pipelines.",
            categories=["cs.RO"],
        ),
        make_candidate(
            "2604.10005",
            "Generalist Agents from Web Data",
            "Foundation model alignment for generalist agents.",
            categories=["cs.LG"],
        ),
        make_candidate(
            "2604.10006",
            "Embodied Control Benchmarks for Manipulators",
            None,
            categories=["cs.RO"],
        ),
    ]
    recommendations = [
        Recommendation(
            paper=candidates[0],
            rank=1,
            score=9.0,
            rationale="Matched robotic manipulation and embodied control.",
            evidence_source=EvidenceSource.ABSTRACT,
            score_breakdown=RankingScoreBreakdown(
                matched_terms=["robotic", "manipulation"],
                matched_phrases=["robotic manipulation"],
            ),
        ),
        Recommendation(
            paper=candidates[1],
            rank=2,
            score=8.2,
            rationale="Matched embodied control.",
            evidence_source=EvidenceSource.ABSTRACT,
            score_breakdown=RankingScoreBreakdown(
                matched_terms=["embodied", "control"],
                matched_phrases=["embodied control"],
            ),
        ),
    ]

    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="robotics agents",
        recommendations=recommendations,
        candidate_papers=candidates,
        query_plan=make_query_plan(required_terms=["robotics", "agents"]),
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data is not None
    overview = result.data.trend_overview
    assert overview.status == TrendAssessmentStatus.AVAILABLE
    assert overview.candidate_count == 6
    assert overview.abstract_count == 5
    assert overview.metadata_only_count == 1
    assert overview.top_k_count == 2
    assert EvidenceSource.CANDIDATE_POOL in overview.evidence_sources
    assert len(overview.signals) <= 6

    labels = {signal.label: signal for signal in overview.signals}
    assert "embodied control" in labels
    assert "robotic manipulation" in labels
    assert labels["embodied control"].support_count >= 4
    assert labels["robotic manipulation"].top_k_count == 2
    assert labels["robotic manipulation"].signal_type == TrendSignalType.HOTSPOT
    assert labels["robotic manipulation"].query_echo is False
    assert "cs.RO" in labels
    assert labels["cs.RO"].signal_type == TrendSignalType.CATEGORY


def test_candidate_pool_trends_are_not_assessed_without_candidate_pool() -> None:
    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="agent briefing",
        recommendations=[make_recommendation("2604.00001", 1)],
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data is not None
    overview = result.data.trend_overview
    assert overview.status == TrendAssessmentStatus.NOT_ASSESSED
    assert overview.candidate_count == 0
    assert overview.signals == []
    assert overview.limitations == [
        "Candidate-pool trend analysis was not requested for this briefing."
    ]


def test_candidate_pool_trends_require_enough_candidates_for_hotspots() -> None:
    candidates = [
        make_candidate(
            "2604.20001",
            "Embodied Control for Robot Arms",
            "Embodied control for robot arms.",
            categories=["cs.RO"],
        ),
        make_candidate(
            "2604.20002",
            "Embodied Control for Mobile Robots",
            "Embodied control for mobile robots.",
            categories=["cs.RO"],
        ),
    ]

    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="robotics",
        recommendations=[
            Recommendation(
                paper=candidates[0],
                rank=1,
                score=9.0,
                rationale="Matched embodied control.",
                evidence_source=EvidenceSource.ABSTRACT,
            )
        ],
        candidate_papers=candidates,
    )

    assert result.data is not None
    overview = result.data.trend_overview
    assert overview.status == TrendAssessmentStatus.INSUFFICIENT_DATA
    assert overview.candidate_count == 2
    assert overview.signals == []
    assert "too small" in overview.limitations[0]


def test_query_echo_terms_are_downgraded_in_candidate_pool_trends() -> None:
    candidates = [
        make_candidate(
            f"2604.3000{index}",
            f"Neural Retrieval Agent System {index}",
            "Neural retrieval agent architecture for search.",
            categories=["cs.AI"],
        )
        for index in range(1, 6)
    ]
    recommendations = [
        Recommendation(
            paper=candidates[0],
            rank=1,
            score=9.0,
            rationale="Matched neural retrieval.",
            evidence_source=EvidenceSource.ABSTRACT,
            score_breakdown=RankingScoreBreakdown(
                matched_terms=["neural", "retrieval", "agent"],
                matched_phrases=["neural retrieval"],
            ),
        )
    ]

    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="neural retrieval agent",
        recommendations=recommendations,
        candidate_papers=candidates,
        query_plan=make_query_plan(
            required_terms=["agent"],
            optional_terms=["neural", "retrieval"],
            phrases=["neural retrieval"],
        ),
    )

    assert result.data is not None
    labels = {signal.label: signal for signal in result.data.trend_overview.signals}
    assert "neural retrieval" in labels
    assert labels["neural retrieval"].query_echo is True
    assert labels["neural retrieval"].strength == TrendSignalStrength.WEAK
    assert labels["neural retrieval"].signal_type != TrendSignalType.HOTSPOT
    assert any("search strategy" in note for note in labels["neural retrieval"].limitations)


def test_query_echo_can_be_kept_when_supported_across_independent_sources() -> None:
    candidates = [
        make_candidate(
            "2604.40001",
            "Embodied Control for Robot Hands",
            "Embodied control for robot hands.",
            categories=["cs.RO"],
        ),
        make_candidate(
            "2604.40002",
            "Embodied Control for Language Agents",
            "Embodied control with language-conditioned agents.",
            categories=["cs.AI"],
        ),
        make_candidate(
            "2604.40003",
            "Embodied Control Benchmarks",
            "Embodied control benchmark suites.",
            categories=["cs.LG"],
        ),
        make_candidate(
            "2604.40004",
            "Embodied Control from Demonstrations",
            "Embodied control from demonstrations.",
            categories=["cs.RO"],
        ),
    ]

    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="embodied control",
        recommendations=[
            Recommendation(
                paper=candidates[0],
                rank=1,
                score=9.0,
                rationale="Matched embodied control.",
                evidence_source=EvidenceSource.ABSTRACT,
                score_breakdown=RankingScoreBreakdown(
                    matched_terms=["embodied", "control"],
                    matched_phrases=["embodied control"],
                ),
            )
        ],
        candidate_papers=candidates,
        query_plan=make_query_plan(phrases=["embodied control"]),
        retrieval_source_metadata_by_paper_id={
            "2604.40001": make_source_metadata("broad_all_terms", variant_index=0),
            "2604.40002": make_source_metadata("broad_phrases", variant_index=1),
            "2604.40003": make_source_metadata("broad_related_terms", variant_index=2),
            "2604.40004": make_source_metadata("broad_all_terms", variant_index=0),
        },
    )

    assert result.data is not None
    labels = {signal.label: signal for signal in result.data.trend_overview.signals}
    assert "embodied control" in labels
    assert labels["embodied control"].query_echo is True
    assert labels["embodied control"].strength == TrendSignalStrength.MODERATE
    assert labels["embodied control"].support_count == 4
    assert any("independent" in note for note in labels["embodied control"].limitations)


def test_metadata_only_candidates_contribute_category_but_not_abstract_signals() -> None:
    candidates = [
        make_candidate(
            "2604.50001",
            "Robot Learning Dataset One",
            None,
            categories=["cs.RO"],
        ),
        make_candidate(
            "2604.50002",
            "Robot Learning Dataset Two",
            None,
            categories=["cs.RO"],
        ),
        make_candidate(
            "2604.50003",
            "Robot Learning Dataset Three",
            None,
            categories=["cs.RO"],
        ),
        make_candidate(
            "2604.50004",
            "Robot Learning Dataset Four",
            None,
            categories=["cs.LG"],
        ),
    ]

    result = DailyBriefingSkill(provider=FakeLLMProvider()).generate(
        topic="robot learning",
        recommendations=[
            Recommendation(
                paper=candidates[0],
                rank=1,
                score=6.0,
                rationale="Metadata-only recommendation.",
                evidence_source=EvidenceSource.METADATA,
            )
        ],
        candidate_papers=candidates,
    )

    assert result.data is not None
    overview = result.data.trend_overview
    assert overview.abstract_count == 0
    assert overview.metadata_only_count == 4
    category_signal = next(
        signal for signal in overview.signals if signal.label == "cs.RO"
    )
    assert category_signal.signal_type == TrendSignalType.CATEGORY
    assert category_signal.evidence_sources == [
        EvidenceSource.CANDIDATE_POOL,
        EvidenceSource.METADATA,
    ]
    assert all(
        EvidenceSource.ABSTRACT not in signal.evidence_sources
        for signal in overview.signals
    )
