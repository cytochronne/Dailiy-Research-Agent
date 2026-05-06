from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    BriefingEvidenceBoundary,
    BriefingTableRow,
    CandidatePoolTrendOverview,
    DailyBriefing,
    EvidenceBoundClaim,
    EvidenceSource,
    EvidenceSupportStatus,
    ExplanationMode,
    FieldEvidenceStatus,
    MethodExplanation,
    PaperDeepExplanation,
    PaperBriefingItem,
    PaperMetadata,
    Provenance,
    QueryPlannerMode,
    ReadingPriority,
    Recommendation,
    RetrievalQuery,
    SeedPreference,
    SeedRecord,
    SearchMode,
    SkillStatus,
    TopKComparisonNote,
    TrendAssessmentStatus,
    TrendSignal,
    TrendSignalStrength,
    TrendSignalType,
)
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
from daily_arxiv_agent.evaluation.metrics import (
    check_explanation_completeness,
    evaluate_briefing_quality,
    evaluate_feedback_movement,
    evaluate_recommendation_fixture,
    evaluate_recommendations,
    evaluate_search_quality,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.arxiv_retrieval import (
    ArxivRetrievalSkill,
    parse_atom_response,
)
from daily_arxiv_agent.skills.query_planning import QueryPlanningSkill
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
    build_paper_preference_text,
)
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


SEARCH_QUALITY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "arxiv_search_quality_response.xml"
)
SEARCH_TOPIC = "multimodal llm agents for robotic manipulation"
SEARCH_EXPECTED_RELEVANT_IDS = ["2604.20001", "2604.20002", "2604.20003"]
SEED_SEARCH_EXPECTED_RELEVANT_IDS = ["2604.20001", "2604.20002", "2604.20003"]


def make_paper(paper_id: str, title: str | None = None) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title or f"Paper {paper_id}",
        authors=["Ada Lovelace"],
        abstract="Agent workflows for research-paper recommendation.",
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


def make_recommendation(paper_id: str, rank: int, score: float) -> Recommendation:
    return Recommendation(
        paper=make_paper(paper_id),
        rank=rank,
        score=score,
        rationale="Deterministic ranking.",
        evidence_source=EvidenceSource.ABSTRACT,
    )


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class SearchQualityClient:
    """Route planned query variants to deterministic quality-fixture subsets."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        search_query = str(params["search_query"])
        return FakeResponse(_search_quality_feed(_ids_for_search_query(search_query)))


class SeedDerivedQualityClient:
    """Route seed-derived query variants to known relevant quality-fixture rows."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        search_query = str(params["search_query"])
        return FakeResponse(
            _search_quality_feed(_ids_for_seed_derived_query(search_query))
        )


class DivergentQualityPlannerProvider:
    def plan_queries(self, **kwargs):  # noqa: ANN003, ANN201
        return {
            "required_terms": ["compiler", "register", "allocation"],
            "phrases": ["compiler register allocation"],
            "related_terms": ["graph coloring"],
            "rationale": "Valid JSON but unrelated to the user's robotics topic.",
        }


def _ids_for_search_query(search_query: str) -> list[str]:
    if 'all:"multimodal llm agents for robotic manipulation"' in search_query:
        return ["2604.20001"]
    if "language" in search_query or "embodied" in search_query:
        return ["2604.20002", "2604.20003", "2604.20005"]
    if "ti:" in search_query or "abs:" in search_query:
        return [
            "2604.20001",
            "2604.20002",
            "2604.20003",
            "2604.20004",
            "2604.20005",
        ]
    if " OR " in search_query:
        return ["2604.20003", "2604.20004", "2604.20006"]
    return ["2604.20001"]


def _ids_for_seed_derived_query(search_query: str) -> list[str]:
    normalized = search_query.lower()
    if any(
        term in normalized
        for term in (
            "multimodal",
            "robotic",
            "manipulation",
            "embodied",
            "planning",
        )
    ):
        return list(SEED_SEARCH_EXPECTED_RELEVANT_IDS)
    return ["2604.20004"]


def _search_quality_feed(paper_ids: list[str]) -> str:
    root = ET.fromstring(SEARCH_QUALITY_FIXTURE.read_text())
    entries = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        entry_id = entry.findtext("{http://www.w3.org/2005/Atom}id") or ""
        if any(paper_id in entry_id for paper_id in paper_ids):
            entries.append(ET.tostring(entry, encoding="unicode"))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        + "\n".join(entries)
        + "\n</feed>\n"
    )


def _search_quality_papers() -> list[PaperMetadata]:
    return parse_atom_response(
        SEARCH_QUALITY_FIXTURE.read_text(),
        RetrievalQuery(topic=SEARCH_TOPIC),
    )


def make_seed_preference_from_papers(
    papers: list[PaperMetadata],
    *,
    profile_id: str = "default",
) -> SeedPreference:
    records = [
        SeedRecord(
            identity=f"arxiv:{paper.paper_id}",
            input_text=paper.paper_id,
            input_type="arxiv_id",
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            paper=paper,
            preference_text=build_paper_preference_text(paper),
        )
        for paper in papers
    ]
    preference_text = "\n\n".join(record.preference_text for record in records)
    return SeedPreference(
        profile_id=profile_id,
        seeds=records,
        preference_text=preference_text,
        vector=DeterministicTextVectorizer().vectorize(preference_text),
    )


def semantic_evaluation_config(*, dimensions: int = 3) -> AppConfig:
    return AppConfig(
        embedding_provider="fake",
        embedding_model="fake-semantic",
        embedding_dimensions=dimensions,
        embedding_cache_enabled=False,
    )


def make_method_explanation() -> PaperDeepExplanation:
    paper = make_paper("2604.00001", "Explainable Agents")
    return PaperDeepExplanation(
        paper_id=paper.paper_id,
        title=paper.title,
        mode=ExplanationMode.METHOD,
        summary="The paper studies explainable daily research agents.",
        evidence_source=EvidenceSource.FULL_TEXT,
        evidence_note="Generated from full text.",
        method=MethodExplanation(
            problem="Explaining ranked research-paper recommendations.",
            method_overview="A workflow retrieves, ranks, and explains papers.",
            core_workflow=["retrieve", "rank", "explain"],
            inputs_outputs=[],
            innovation="Evidence labels are attached to generated claims.",
        ),
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )


def supported_evidence(*sources: EvidenceSource) -> FieldEvidenceStatus:
    return FieldEvidenceStatus(
        status=EvidenceSupportStatus.SUPPORTED,
        sources=list(sources),
    )


def partial_evidence(*sources: EvidenceSource, note: str) -> FieldEvidenceStatus:
    return FieldEvidenceStatus(
        status=EvidenceSupportStatus.PARTIAL,
        sources=list(sources),
        note=note,
    )


def unavailable_evidence(reason: str) -> FieldEvidenceStatus:
    return FieldEvidenceStatus(
        status=EvidenceSupportStatus.UNAVAILABLE,
        abstention_reason=reason,
    )


def make_briefing_paper(
    paper_id: str,
    title: str,
    abstract: str | None,
    *,
    categories: list[str] | None = None,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=categories or ["cs.LG"],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query="briefing quality",
        ),
    )


def make_briefing_item(
    paper: PaperMetadata,
    *,
    rank: int,
    score: float,
    metadata_only: bool = False,
) -> PaperBriefingItem:
    if metadata_only:
        limited = unavailable_evidence(
            "No abstract is available to support this field."
        )
        return PaperBriefingItem(
            paper_id=paper.paper_id,
            title=paper.title,
            rank=rank,
            score=score,
            summary="Only metadata was available for this ranked paper.",
            relevance_rationale="Ranking matched title, category, and retrieval metadata.",
            evidence_source=EvidenceSource.METADATA,
            provenance=paper.provenance,
            arxiv_url=paper.arxiv_url,
            problem=EvidenceBoundClaim(claim=None, evidence=limited),
            approach=EvidenceBoundClaim(claim=None, evidence=limited),
            reading_guide=EvidenceBoundClaim(
                claim=(
                    "Treat this metadata-only result as a follow-up lead before "
                    "making technical claims."
                ),
                evidence=partial_evidence(
                    EvidenceSource.METADATA,
                    EvidenceSource.RANKING,
                    note="Reading guidance uses metadata and ranking context.",
                ),
            ),
        )

    supported = supported_evidence(EvidenceSource.ABSTRACT)
    return PaperBriefingItem(
        paper_id=paper.paper_id,
        title=paper.title,
        rank=rank,
        score=score,
        summary=(
            "The paper studies multimodal agent workflows for robotic manipulation "
            "and daily research triage."
        ),
        contributions=[
            "Connects embodied planning signals to ranked research briefing guidance."
        ],
        methods=["Vision-language planning with closed-loop manipulation control."],
        relevance_rationale=(
            "Matched robotic manipulation, embodied control, and agent planning terms."
        ),
        evidence_source=EvidenceSource.ABSTRACT,
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
        problem=EvidenceBoundClaim(
            claim=(
                "Robotic manipulation papers need triage that separates embodied "
                "planning work from weak category-only matches."
            ),
            evidence=supported,
        ),
        approach=EvidenceBoundClaim(
            claim=(
                "The abstract frames vision-language planning and closed-loop "
                "control as the core agent workflow."
            ),
            evidence=supported,
        ),
        reading_guide=EvidenceBoundClaim(
            claim=(
                "Read first for embodied-control agent design, then compare the "
                "planning assumptions against the other Top-K papers."
            ),
            evidence=partial_evidence(
                EvidenceSource.ABSTRACT,
                EvidenceSource.RANKING,
                note="Reading guidance combines abstract evidence with ranking context.",
            ),
        ),
        contribution_claims=[
            EvidenceBoundClaim(
                claim=(
                    "The contribution is specific to embodied manipulation and "
                    "ranked agent-planning evidence."
                ),
                evidence=supported,
            )
        ],
        method_claims=[
            EvidenceBoundClaim(
                claim=(
                    "The method combines vision-language planning, robotic "
                    "manipulation control, and agent feedback loops."
                ),
                evidence=supported,
            )
        ],
        relevance_evidence=partial_evidence(
            EvidenceSource.ABSTRACT,
            EvidenceSource.RANKING,
            note="Relevance combines abstract terms with ranking context.",
        ),
    )


def make_quality_briefing(
    *,
    include_priorities: bool = True,
    include_boundary: bool = True,
    generic: bool = False,
    full_text_claim: bool = False,
    trend_status: TrendAssessmentStatus = TrendAssessmentStatus.AVAILABLE,
    metadata_only: bool = False,
) -> DailyBriefing:
    paper = make_briefing_paper(
        "2604.20001",
        "Multimodal LLM Agents for Robotic Manipulation",
        "We present multimodal LLM agents for robotic manipulation.",
        categories=["cs.RO", "cs.AI"],
    )
    second = make_briefing_paper(
        "2604.20002",
        "Embodied Language Models for Dexterous Manipulation",
        None if metadata_only else "Language model policies coordinate embodied control.",
        categories=["cs.RO", "cs.LG"],
    )
    items = [
        make_briefing_item(paper, rank=1, score=8.8),
        make_briefing_item(
            second,
            rank=2,
            score=7.4,
            metadata_only=metadata_only,
        ),
    ]
    if generic:
        generic_status = supported_evidence(EvidenceSource.ABSTRACT)
        for item in items:
            item.summary = "This paper is important and relevant to the topic."
            item.contributions = ["It has a useful contribution."]
            item.methods = ["It uses a method."]
            item.relevance_rationale = "This is a good match."
            item.problem = EvidenceBoundClaim(
                claim="This paper addresses an important problem.",
                evidence=generic_status,
            )
            item.approach = EvidenceBoundClaim(
                claim="This paper proposes a useful approach.",
                evidence=generic_status,
            )
            item.reading_guide = EvidenceBoundClaim(
                claim="Read this paper first because it is useful.",
                evidence=generic_status,
            )
    trend_signals: list[TrendSignal] = []
    trend_summary = None
    trend_sources: list[EvidenceSource] = []
    candidate_count = 0
    abstract_count = 0
    metadata_only_count = 0
    if trend_status == TrendAssessmentStatus.AVAILABLE:
        candidate_count = 6
        abstract_count = 5
        metadata_only_count = 1
        trend_summary = "Robotic manipulation and embodied control recur in candidates."
        trend_sources = [EvidenceSource.CANDIDATE_POOL, EvidenceSource.ABSTRACT]
        trend_signals = [
            TrendSignal(
                label="robotic manipulation",
                signal_type=TrendSignalType.HOTSPOT,
                strength=TrendSignalStrength.MODERATE,
                support_count=4,
                candidate_count=6,
                top_k_count=2,
                evidence_sources=trend_sources,
                summary="Repeated across exact and related candidate abstracts.",
            )
        ]

    comparisons = [
        TopKComparisonNote(
            dimension="paper difference",
            note=(
                "Rank 1 emphasizes multimodal robotic manipulation, while rank 2 "
                "focuses on embodied language-model control."
            )
            if not generic
            else "Rank 1 and rank 2 are both useful papers.",
            paper_ids=[item.paper_id for item in items],
            ranks=[item.rank for item in items],
            evidence=supported_evidence(EvidenceSource.ABSTRACT, EvidenceSource.RANKING),
        )
    ]
    priorities = []
    if include_priorities:
        priorities = [
            ReadingPriority(
                priority=1,
                reading_intent=(
                    "start with embodied-control agent design"
                    if not generic
                    else "read this first"
                ),
                paper_id=items[0].paper_id,
                rank=1,
                reason=(
                    "It has the strongest score and explicit abstract support for "
                    "robotic manipulation agents."
                    if not generic
                    else "It is a useful paper."
                ),
                evidence=supported_evidence(EvidenceSource.ABSTRACT, EvidenceSource.RANKING),
            )
        ]
    boundary = BriefingEvidenceBoundary()
    if include_boundary:
        boundary = BriefingEvidenceBoundary(
            evidence_sources=[
                EvidenceSource.METADATA,
                EvidenceSource.ABSTRACT,
                EvidenceSource.RANKING,
                EvidenceSource.RETRIEVAL_METADATA,
                EvidenceSource.CANDIDATE_POOL,
            ],
            unavailable_sources=[EvidenceSource.FULL_TEXT],
            full_text_used=full_text_claim,
            notes=[
                "No PDF or full-text evidence was used."
                if not full_text_claim
                else "Full-text evidence was used to verify the method."
            ],
            abstentions=[
                EvidenceBoundClaim(
                    claim=None,
                    evidence=unavailable_evidence(
                        "PDF and full-text evidence were not used in the default briefing."
                    ),
                )
            ],
        )
    return DailyBriefing(
        topic="multimodal llm agents for robotic manipulation",
        executive_summary=(
            "Full-text evidence shows strong experimental support."
            if full_text_claim
            else "Top papers separate robotic manipulation agents from weak matches."
        )
        if not generic
        else "These papers are useful and relevant.",
        summary_table=[
            BriefingTableRow(
                rank=item.rank,
                paper_id=item.paper_id,
                title=item.title,
                score=item.score,
                key_reason=item.relevance_rationale,
                evidence_source=item.evidence_source,
                arxiv_url=item.arxiv_url,
            )
            for item in items
        ],
        highlighted_paper=items[0],
        items=items,
        evidence_source=EvidenceSource.MIXED,
        trend_overview=CandidatePoolTrendOverview(
            status=trend_status,
            summary=trend_summary,
            candidate_count=candidate_count,
            abstract_count=abstract_count,
            metadata_only_count=metadata_only_count,
            top_k_count=len(items) if trend_status == TrendAssessmentStatus.AVAILABLE else 0,
            signals=trend_signals,
            limitations=[]
            if trend_status == TrendAssessmentStatus.AVAILABLE
            else ["Candidate-pool trend analysis was not assessed."],
            evidence_sources=trend_sources,
        ),
        top_k_comparisons=comparisons,
        reading_priorities=priorities,
        evidence_boundary=boundary,
    )


def test_recommendation_evaluation_compares_ranked_results_with_expected_ids() -> None:
    recommendations = [
        make_recommendation("2604.00001", rank=1, score=9.0),
        make_recommendation("2604.00002", rank=2, score=4.0),
        make_recommendation("2604.00003", rank=3, score=1.0),
    ]

    result = evaluate_recommendations(
        recommendations,
        ["2604.00002", "2604.99999"],
        k=2,
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.evaluated_paper_ids == ["2604.00001", "2604.00002"]
    assert data.matched_paper_ids == ["2604.00002"]
    assert data.missing_relevant_ids == ["2604.99999"]
    assert data.precision_at_k == 0.5
    assert data.recall_at_k == 0.5
    assert data.mean_reciprocal_rank == 0.5


def test_recommendation_fixture_validates_and_evaluates_simple_rows() -> None:
    result = evaluate_recommendation_fixture(
        {
            "recommendations": [
                {"paper_id": "2604.00001", "rank": 1, "score": 9.0},
                {"paper_id": "2604.00002", "rank": 2, "score": 4.0},
            ],
            "expected_relevant_paper_ids": ["2604.00001"],
            "k": 1,
        }
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data is not None
    assert result.data.matched_paper_ids == ["2604.00001"]
    assert result.metadata["fixture_keys"] == [
        "expected_relevant_paper_ids",
        "k",
        "recommendations",
    ]


def test_recommendation_evaluation_uses_rank_order_before_applying_k() -> None:
    result = evaluate_recommendation_fixture(
        {
            "recommendations": [
                {"paper_id": "2604.99999", "rank": 2, "score": 0.0},
                {"paper_id": "2604.00001", "rank": 1, "score": 9.0},
            ],
            "expected_relevant_paper_ids": ["2604.00001"],
            "k": 1,
        }
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.evaluated_paper_ids == ["2604.00001"]
    assert data.precision_at_k == 1.0
    assert data.recall_at_k == 1.0


def test_feedback_evaluation_detects_rank_movement_after_likes_and_dislikes() -> None:
    before = [
        make_recommendation("2604.00001", rank=1, score=4.0),
        make_recommendation("2604.00002", rank=2, score=3.0),
        make_recommendation("2604.00003", rank=3, score=1.0),
    ]
    after = [
        make_recommendation("2604.00002", rank=1, score=5.0),
        make_recommendation("2604.00001", rank=2, score=3.5),
        make_recommendation("2604.00003", rank=3, score=1.0),
    ]

    result = evaluate_feedback_movement(
        before,
        after,
        liked_paper_ids=["2604.00002"],
        disliked_paper_ids=["2604.00001"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.moved_up_ids == ["2604.00002"]
    assert data.moved_down_ids == ["2604.00001"]
    assert data.unchanged_ids == ["2604.00003"]
    moved_up = data.movements[0]
    assert moved_up.paper_id == "2604.00002"
    assert moved_up.rank_delta == 1
    assert moved_up.score_delta == 2.0
    assert moved_up.feedback_value == "like"


def test_explanation_completeness_reports_present_and_missing_sections() -> None:
    result = check_explanation_completeness(make_method_explanation())

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.paper_id == "2604.00001"
    assert "method.problem" in data.present_sections
    assert "method.inputs_outputs" in data.missing_sections
    assert data.completeness_score < 1.0
    assert not data.is_complete


def test_explanation_completeness_treats_missing_evidence_placeholders_as_absent() -> None:
    paper = make_paper("2604.00001", "Explainable Agents")
    explanation = FakeLLMProvider().explain_paper(
        paper,
        mode=ExplanationMode.EXPERIMENT,
        content=paper.abstract or "",
        evidence_source=EvidenceSource.ABSTRACT,
    )

    result = check_explanation_completeness(explanation)

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert "experiment.datasets" in data.missing_sections
    assert "experiment.metrics" in data.missing_sections
    assert data.completeness_score < 1.0
    assert not data.is_complete


def test_explanation_completeness_rejects_empty_required_sections() -> None:
    result = check_explanation_completeness(
        make_method_explanation(),
        required_sections=[],
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "evaluation_input_invalid"
    assert "required_sections" in result.error.message


def test_empty_recommendations_return_meaningful_zero_data_result() -> None:
    result = evaluate_recommendations([], ["2604.00001"])

    assert result.status == SkillStatus.EMPTY
    data = result.data
    assert data is not None
    assert data.zero_data_reason == "No recommendations were supplied for evaluation."
    assert data.precision_at_k == 0.0
    assert data.recall_at_k == 0.0


def test_malformed_evaluation_fixture_returns_structured_validation_error() -> None:
    result = evaluate_recommendation_fixture(
        {
            "recommendations": [{"rank": 1, "score": 9.0}],
            "expected_relevant_paper_ids": ["2604.00001"],
        }
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "evaluation_fixture_invalid"
    assert "paper_id" in result.error.message


def test_search_quality_fixture_shows_broad_retrieval_improves_candidate_coverage(
    tmp_path,
) -> None:
    strict_query = RetrievalQuery(
        topic=SEARCH_TOPIC,
        search_mode=SearchMode.STRICT,
        candidate_pool_size=20,
        page_size=10,
        max_requests=1,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )
    broad_query = strict_query.model_copy(
        update={
            "search_mode": SearchMode.BROAD,
            "max_requests": 4,
            "query_planner_mode": QueryPlannerMode.LLM,
        }
    )
    strict_plan = QueryPlanningSkill().plan(strict_query).data
    broad_plan = QueryPlanningSkill(provider=FakeLLMProvider()).plan(broad_query).data
    assert strict_plan is not None
    assert broad_plan is not None

    strict_result = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "strict.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(strict_query, query_plan=strict_plan)
    broad_result = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "broad.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(broad_query, query_plan=broad_plan)

    strict_eval = evaluate_search_quality(
        strict_result.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=strict_result.metadata,
    )
    broad_eval = evaluate_search_quality(
        broad_result.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=broad_result.metadata,
    )

    assert strict_eval.status == SkillStatus.SUCCESS
    assert broad_eval.status == SkillStatus.SUCCESS
    assert strict_eval.data is not None
    assert broad_eval.data is not None
    assert strict_eval.data.relevant_candidate_ids == ["2604.20001"]
    assert broad_eval.data.relevant_candidate_ids == SEARCH_EXPECTED_RELEVANT_IDS
    assert (
        broad_eval.data.relevant_candidate_coverage
        > strict_eval.data.relevant_candidate_coverage
    )


def test_search_quality_evaluation_covers_candidate_count_top_k_and_rationales(
    tmp_path,
) -> None:
    query = RetrievalQuery(
        topic=SEARCH_TOPIC,
        search_mode=SearchMode.BROAD,
        candidate_pool_size=100,
        page_size=50,
        max_requests=4,
        query_planner_mode=QueryPlannerMode.LLM,
    )
    plan = QueryPlanningSkill(provider=FakeLLMProvider()).plan(query).data
    assert plan is not None
    retrieval = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "quality.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(query, query_plan=plan)
    ranking = TopicRankingSkill().rank(
        retrieval.data or [],
        topic=SEARCH_TOPIC,
        query_plan=plan,
        retrieval_query=query,
        retrieval_source_metadata_by_paper_id=retrieval.metadata[
            "source_metadata_by_paper_id"
        ],
        top_k=3,
    )

    assert ranking.status == SkillStatus.SUCCESS
    recommendations = ranking.data or []
    top_ids = [recommendation.paper.paper_id for recommendation in recommendations]
    assert top_ids == SEARCH_EXPECTED_RELEVANT_IDS
    assert "2604.20006" not in top_ids
    assert all(
        recommendation.score_breakdown is not None
        and not recommendation.score_breakdown.fallback
        for recommendation in recommendations
    )

    evaluation = evaluate_search_quality(
        retrieval.data or [],
        recommendations,
        SEARCH_EXPECTED_RELEVANT_IDS,
        k=3,
        retrieval_metadata=retrieval.metadata,
    )

    assert evaluation.status == SkillStatus.SUCCESS
    data = evaluation.data
    assert data is not None
    assert data.candidate_count >= 5
    assert data.budget_exhausted is True
    assert data.top_k_paper_ids == SEARCH_EXPECTED_RELEVANT_IDS
    assert data.precision_at_k == 1.0
    assert data.recall_at_k == 1.0
    assert data.rationale_coverage == 1.0
    assert data.missing_rationale_ids == []


def test_seed_derived_retrieval_fixture_reports_known_relevant_candidate_recall(
    tmp_path,
) -> None:
    seed = make_briefing_paper(
        "2604.19999",
        "Multimodal LLM Agents for Robotic Manipulation",
        (
            "Vision-language agents plan embodied manipulation tasks with "
            "closed-loop robot control."
        ),
        categories=["cs.RO", "cs.AI"],
    )
    query = RetrievalQuery(
        topic=None,
        category="cs.RO",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=20,
        page_size=10,
        max_requests=3,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )
    seed_preference = make_seed_preference_from_papers([seed], profile_id="eval")
    plan_result = QueryPlanningSkill().plan_from_seed(query, seed_preference)
    assert plan_result.status == SkillStatus.SUCCESS
    assert plan_result.data is not None
    assert plan_result.metadata["source"] == "seed_derived"

    client = SeedDerivedQualityClient()
    retrieval = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "seed-recall.sqlite3"),
        client=client,
        request_delay_seconds=0,
    ).retrieve(query, query_plan=plan_result.data)

    evaluation = evaluate_search_quality(
        retrieval.data or [],
        [],
        SEED_SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=retrieval.metadata,
    )

    assert evaluation.status == SkillStatus.SUCCESS
    assert evaluation.data is not None
    assert evaluation.data.relevant_candidate_ids == SEED_SEARCH_EXPECTED_RELEVANT_IDS
    assert evaluation.data.relevant_candidate_coverage == 1.0
    assert client.calls
    assert all(
        "all:*" not in str(call["params"]["search_query"]) for call in client.calls
    )


def test_semantic_seed_fixture_beats_deterministic_lexical_only_baseline(
    tmp_path,
) -> None:
    seed = make_briefing_paper(
        "2604.30001",
        "Graph Neural Program Repair",
        "Neural models repair code defects from failing tests.",
        categories=["cs.SE"],
    )
    related = make_briefing_paper(
        "2604.30002",
        "Learning Patches from Execution Traces",
        "Models synthesize bug fixes from runtime traces and test failures.",
        categories=["cs.SE"],
    )
    lexical_distractor = make_briefing_paper(
        "2604.30003",
        "Graph Neural Program Repair Bibliography",
        (
            "Graph neural program repair terms appear in citation metadata, "
            "but the paper only indexes references."
        ),
        categories=["cs.SE"],
    )
    unrelated = make_briefing_paper(
        "2604.30004",
        "Register Allocation in Optimizing Compilers",
        "A systems survey of register pressure and compiler allocation heuristics.",
        categories=["cs.PL"],
    )
    seed_preference = make_seed_preference_from_papers([seed], profile_id="eval")
    candidates = [related, lexical_distractor, unrelated]

    deterministic_result = TopicRankingSkill().rank(
        candidates,
        seed_preference=seed_preference,
        top_k=3,
    )
    vectors = {
        build_paper_preference_text(seed): [1.0, 0.0, 0.0],
        build_paper_preference_text(related): [0.98, 0.02, 0.0],
        build_paper_preference_text(lexical_distractor): [0.0, 1.0, 0.0],
        build_paper_preference_text(unrelated): [-1.0, 0.0, 0.0],
    }
    semantic_result = SemanticSeedRankingSkill(
        embedding_provider=FakeEmbeddingProvider(dimensions=3, vector_map=vectors),
        store=SQLitePaperStore(tmp_path / "semantic-ranking.sqlite3"),
        config=semantic_evaluation_config(),
        minimum_semantic_similarity=0.4,
    ).rank(
        candidates,
        seed_preference=seed_preference,
        retrieval_query=RetrievalQuery(topic=None, category="cs.SE"),
        top_k=3,
    )

    assert deterministic_result.status == SkillStatus.SUCCESS
    assert semantic_result.status == SkillStatus.SUCCESS
    deterministic_recommendations = deterministic_result.data or []
    semantic_recommendations = semantic_result.data or []
    assert deterministic_recommendations[0].paper.paper_id == lexical_distractor.paper_id
    assert semantic_recommendations[0].paper.paper_id == related.paper_id

    baseline_eval = evaluate_recommendations(
        deterministic_recommendations,
        [related.paper_id],
        k=1,
    )
    semantic_eval = evaluate_recommendations(
        semantic_recommendations,
        [related.paper_id],
        k=1,
    )

    assert baseline_eval.data is not None
    assert semantic_eval.data is not None
    assert baseline_eval.data.precision_at_k == 0.0
    assert semantic_eval.data.precision_at_k == 1.0
    assert (
        semantic_eval.data.mean_reciprocal_rank
        > baseline_eval.data.mean_reciprocal_rank
    )


def test_bad_llm_query_plan_falls_back_before_search_quality_can_degrade(
    tmp_path,
) -> None:
    deterministic_query = RetrievalQuery(
        topic=SEARCH_TOPIC,
        search_mode=SearchMode.BROAD,
        candidate_pool_size=20,
        page_size=10,
        max_requests=4,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )
    llm_query = deterministic_query.model_copy(
        update={"query_planner_mode": QueryPlannerMode.LLM}
    )
    deterministic_plan = QueryPlanningSkill().plan(deterministic_query).data
    bad_llm_result = QueryPlanningSkill(
        provider=DivergentQualityPlannerProvider()
    ).plan(llm_query)
    assert deterministic_plan is not None
    assert bad_llm_result.status == SkillStatus.FALLBACK
    assert bad_llm_result.data is not None
    assert bad_llm_result.error is not None
    assert bad_llm_result.error.code == "query_planner_semantic_guard_failed"

    deterministic_retrieval = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "deterministic.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(deterministic_query, query_plan=deterministic_plan)
    fallback_retrieval = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "fallback.sqlite3"),
        client=SearchQualityClient(),
        request_delay_seconds=0,
    ).retrieve(llm_query, query_plan=bad_llm_result.data)

    deterministic_eval = evaluate_search_quality(
        deterministic_retrieval.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=deterministic_retrieval.metadata,
    )
    fallback_eval = evaluate_search_quality(
        fallback_retrieval.data or [],
        [],
        SEARCH_EXPECTED_RELEVANT_IDS,
        retrieval_metadata=fallback_retrieval.metadata,
    )

    assert deterministic_eval.data is not None
    assert fallback_eval.data is not None
    assert fallback_eval.data.relevant_candidate_coverage == (
        deterministic_eval.data.relevant_candidate_coverage
    )
    assert fallback_eval.data.candidate_count == deterministic_eval.data.candidate_count


def test_search_quality_rejects_empty_expected_relevance_set() -> None:
    result = evaluate_search_quality(_search_quality_papers(), [], [])

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "evaluation_input_invalid"
    assert "expected_relevant_paper_ids" in result.error.message


def test_enhanced_briefing_quality_passes_specific_supported_sections() -> None:
    briefing = make_quality_briefing()

    result = evaluate_briefing_quality(
        briefing,
        expected_top_k_paper_ids=["2604.20001", "2604.20002"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.quality_passed
    assert data.missing_sections == []
    assert data.top_k_coverage == 1.0
    assert data.trend_status == TrendAssessmentStatus.AVAILABLE
    assert data.trend_signal_coverage == 1.0
    assert data.reading_priority_present
    assert data.evidence_boundary_present
    assert data.claim_support_coverage == 1.0
    assert data.claim_specificity_score >= 0.6
    assert data.generic_claim_locations == []
    assert data.forbidden_evidence_claims == []


def test_briefing_quality_accepts_top_k_when_trends_are_not_assessed() -> None:
    briefing = make_quality_briefing(
        trend_status=TrendAssessmentStatus.NOT_ASSESSED,
    )

    result = evaluate_briefing_quality(
        briefing,
        expected_top_k_paper_ids=["2604.20001", "2604.20002"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.quality_passed
    assert data.top_k_coverage == 1.0
    assert data.trend_status == TrendAssessmentStatus.NOT_ASSESSED
    assert data.trend_signal_coverage is None
    assert "trend_overview" in data.present_sections


def test_briefing_quality_identifies_missing_priorities_and_boundary() -> None:
    briefing = make_quality_briefing(
        include_priorities=False,
        include_boundary=False,
    )

    result = evaluate_briefing_quality(
        briefing,
        expected_top_k_paper_ids=["2604.20001", "2604.20002"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert not data.quality_passed
    assert "reading_priorities" in data.missing_sections
    assert "evidence_boundary" in data.missing_sections
    assert "reading_priorities_missing" in data.failure_reasons
    assert "evidence_boundary_missing" in data.failure_reasons


def test_briefing_quality_fails_structurally_complete_generic_content() -> None:
    briefing = make_quality_briefing(generic=True)

    result = evaluate_briefing_quality(
        briefing,
        expected_top_k_paper_ids=["2604.20001", "2604.20002"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert not data.quality_passed
    assert data.missing_sections == []
    assert data.claim_support_coverage == 1.0
    assert data.claim_specificity_score < 0.6
    assert data.generic_claim_locations
    assert "claim_specificity_low" in data.failure_reasons


def test_briefing_quality_flags_default_mode_full_text_claims() -> None:
    briefing = make_quality_briefing(full_text_claim=True)
    briefing.items[0].contributions = ["PDF analysis shows stronger evidence."]

    result = evaluate_briefing_quality(
        briefing,
        expected_top_k_paper_ids=["2604.20001", "2604.20002"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert not data.quality_passed
    assert "default_mode_full_text_used" in data.failure_reasons
    assert "evidence_boundary.full_text_used" in data.forbidden_evidence_claims
    assert any(
        location.startswith("executive_summary")
        for location in data.forbidden_evidence_claims
    )
    assert "items[0].contributions[0]" in data.forbidden_evidence_claims


def test_briefing_quality_accepts_metadata_only_boundary_when_abstracts_absent() -> None:
    briefing = make_quality_briefing(
        trend_status=TrendAssessmentStatus.NOT_ASSESSED,
        metadata_only=True,
    )
    metadata_sources = [EvidenceSource.METADATA, EvidenceSource.RANKING]
    briefing.evidence_boundary = BriefingEvidenceBoundary(
        evidence_sources=metadata_sources,
        unavailable_sources=[EvidenceSource.ABSTRACT, EvidenceSource.FULL_TEXT],
        full_text_used=False,
        notes=["Only metadata and ranking evidence were available."],
        abstentions=[
            EvidenceBoundClaim(
                claim=None,
                evidence=unavailable_evidence(
                    "Abstracts and PDF full text were unavailable for default briefing."
                ),
            )
        ],
    )
    for item in briefing.items:
        item.evidence_source = EvidenceSource.METADATA
        unavailable = unavailable_evidence("No abstract is available.")
        item.problem = EvidenceBoundClaim(claim=None, evidence=unavailable)
        item.approach = EvidenceBoundClaim(claim=None, evidence=unavailable)
    briefing.summary_table = [
        BriefingTableRow(
            rank=item.rank,
            paper_id=item.paper_id,
            title=item.title,
            score=item.score,
            key_reason=item.relevance_rationale,
            evidence_source=EvidenceSource.METADATA,
            arxiv_url=item.arxiv_url,
        )
        for item in briefing.items
    ]
    briefing.top_k_comparisons = [
        TopKComparisonNote(
            dimension="metadata title difference",
            note=(
                "Rank 1 title emphasizes multimodal robotic manipulation, while "
                "rank 2 title emphasizes embodied language models."
            ),
            paper_ids=[item.paper_id for item in briefing.items],
            ranks=[item.rank for item in briefing.items],
            evidence=partial_evidence(
                EvidenceSource.METADATA,
                EvidenceSource.RANKING,
                note="Comparison is limited to titles and ranking context.",
            ),
        )
    ]
    briefing.reading_priorities = [
        ReadingPriority(
            priority=1,
            reading_intent="start with metadata-backed robotic manipulation lead",
            paper_id=briefing.items[0].paper_id,
            rank=1,
            reason=(
                "The title has the closest robotic manipulation match, but "
                "technical claims require abstract or full-text follow-up."
            ),
            evidence=partial_evidence(
                EvidenceSource.METADATA,
                EvidenceSource.RANKING,
                note="Priority is limited to title metadata and rank.",
            ),
        )
    ]

    result = evaluate_briefing_quality(
        briefing,
        expected_top_k_paper_ids=["2604.20001", "2604.20002"],
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.quality_passed
    assert data.evidence_boundary_present
    assert data.forbidden_evidence_claims == []
    assert data.top_k_coverage == 1.0


def test_fixture_backed_enhanced_briefing_quality_runs_offline() -> None:
    papers = _search_quality_papers()
    papers_by_id = {paper.paper_id: paper for paper in papers}
    top_k_ids = SEARCH_EXPECTED_RELEVANT_IDS
    items = [
        make_briefing_item(
            papers_by_id[paper_id],
            rank=rank,
            score=9.0 - rank,
        )
        for rank, paper_id in enumerate(top_k_ids, start=1)
    ]
    briefing = make_quality_briefing()
    briefing.items = items
    briefing.highlighted_paper = items[0]
    briefing.summary_table = [
        BriefingTableRow(
            rank=item.rank,
            paper_id=item.paper_id,
            title=item.title,
            score=item.score,
            key_reason=item.relevance_rationale,
            evidence_source=item.evidence_source,
            arxiv_url=item.arxiv_url,
        )
        for item in items
    ]
    briefing.trend_overview = CandidatePoolTrendOverview(
        status=TrendAssessmentStatus.AVAILABLE,
        summary=(
            "Robotic manipulation, embodied control, and planning agents recur "
            "across exact and related candidate papers."
        ),
        candidate_count=len(papers),
        abstract_count=sum(1 for paper in papers if paper.abstract),
        metadata_only_count=sum(1 for paper in papers if not paper.abstract),
        top_k_count=len(items),
        signals=[
            TrendSignal(
                label="robotic manipulation",
                signal_type=TrendSignalType.HOTSPOT,
                strength=TrendSignalStrength.MODERATE,
                support_count=3,
                candidate_count=len(papers),
                top_k_count=3,
                evidence_sources=[
                    EvidenceSource.CANDIDATE_POOL,
                    EvidenceSource.ABSTRACT,
                ],
                summary="Appears in exact and related fixture candidates.",
            ),
            TrendSignal(
                label="cs.RO",
                signal_type=TrendSignalType.CATEGORY,
                strength=TrendSignalStrength.MODERATE,
                support_count=5,
                candidate_count=len(papers),
                top_k_count=3,
                evidence_sources=[EvidenceSource.CANDIDATE_POOL],
                summary="Shared robotics category among relevant and weak candidates.",
            ),
        ],
        limitations=["One compiler paper remains an unrelated noisy candidate."],
        evidence_sources=[EvidenceSource.CANDIDATE_POOL, EvidenceSource.ABSTRACT],
    )

    result = evaluate_briefing_quality(
        briefing,
        expected_top_k_paper_ids=SEARCH_EXPECTED_RELEVANT_IDS,
    )

    assert result.status == SkillStatus.SUCCESS
    data = result.data
    assert data is not None
    assert data.quality_passed
    assert data.candidate_count == 6
    assert data.top_k_coverage == 1.0
    assert data.trend_signal_coverage == 1.0
    assert data.claim_support_coverage == 1.0
