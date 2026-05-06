from datetime import date

import pytest
from pydantic import ValidationError

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    BriefingEvidenceBoundary,
    BriefingTableRow,
    CandidatePoolTrendOverview,
    DailyBriefing,
    EmbeddingCacheMetadata,
    EmbeddingCacheScope,
    EmbeddingIdentity,
    EmbeddingInputRole,
    EmbeddingProviderCacheMetadata,
    EmbeddingVector,
    EvidenceSource,
    EvidenceBoundClaim,
    EvidenceSupportStatus,
    ExplanationMode,
    FieldEvidenceStatus,
    MethodExplanation,
    PaperDeepExplanation,
    PaperBriefingItem,
    PaperMetadata,
    QueryPlan,
    QueryPlanVariant,
    QueryPlannerMode,
    QueryPlannerProvenance,
    RankingScoreBreakdown,
    ReadingPriority,
    Recommendation,
    RetrievalBudget,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SearchMode,
    Provenance,
    SkillError,
    SkillResult,
    SkillStatus,
    TopKComparisonNote,
    TrendAssessmentStatus,
    TrendSignal,
    TrendSignalStrength,
    TrendSignalType,
)


def make_paper() -> PaperMetadata:
    provenance = Provenance(
        source="arxiv",
        source_url="https://arxiv.org/abs/2501.00001",
        query="cat:cs.LG",
    )
    return PaperMetadata(
        paper_id="2501.00001",
        title="A Test Paper",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="This paper studies a testable agent architecture.",
        categories=["cs.LG"],
        published_date=date(2025, 1, 1),
        updated_date=date(2025, 1, 2),
        arxiv_url="https://arxiv.org/abs/2501.00001",
        pdf_url="https://arxiv.org/pdf/2501.00001",
        provenance=provenance,
    )


def make_briefing_item(
    *,
    evidence_source: EvidenceSource = EvidenceSource.ABSTRACT,
) -> PaperBriefingItem:
    paper = make_paper()
    return PaperBriefingItem(
        paper_id=paper.paper_id,
        title=paper.title,
        rank=1,
        score=5.0,
        summary="A concise briefing summary.",
        contributions=["Introduces a testable agent architecture."],
        methods=["Staged retrieval, ranking, and synthesis workflow."],
        relevance_rationale="Matched explicit agent architecture terms.",
        evidence_source=evidence_source,
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )


def make_briefing_table_row() -> BriefingTableRow:
    paper = make_paper()
    return BriefingTableRow(
        rank=1,
        paper_id=paper.paper_id,
        title=paper.title,
        score=5.0,
        key_reason="Matched explicit agent architecture terms.",
        evidence_source=EvidenceSource.ABSTRACT,
        arxiv_url=paper.arxiv_url,
    )


def test_paper_metadata_supports_required_arxiv_fields() -> None:
    paper = make_paper()

    assert paper.paper_id == "2501.00001"
    assert paper.title == "A Test Paper"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.abstract is not None
    assert paper.categories == ["cs.LG"]
    assert str(paper.arxiv_url) == "https://arxiv.org/abs/2501.00001"
    assert str(paper.pdf_url) == "https://arxiv.org/pdf/2501.00001"
    assert paper.provenance.source == "arxiv"


def test_skill_result_success_includes_data_and_provenance_without_error() -> None:
    paper = make_paper()
    result = SkillResult[list[PaperMetadata]](
        status=SkillStatus.SUCCESS,
        data=[paper],
        evidence_source=EvidenceSource.ABSTRACT,
        provenance=[paper.provenance],
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data == [paper]
    assert result.error is None
    assert result.evidence_source == EvidenceSource.ABSTRACT


def test_fallback_skill_result_requires_structured_error() -> None:
    result = SkillResult[None](
        status=SkillStatus.FALLBACK,
        data=None,
        evidence_source=EvidenceSource.METADATA,
        error=SkillError(
            code="arxiv_unavailable",
            message="arXiv request failed; using cached results.",
            retryable=True,
        ),
        message="Using cached results.",
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.retryable is True


def test_error_or_fallback_without_error_is_invalid() -> None:
    with pytest.raises(ValidationError):
        SkillResult[None](status=SkillStatus.ERROR)


def test_empty_optional_fields_serialize_consistently() -> None:
    paper = PaperMetadata(
        paper_id="2501.00002",
        title="Minimal Paper",
        arxiv_url="https://arxiv.org/abs/2501.00002",
        provenance=Provenance(source="arxiv"),
    )

    payload = paper.model_dump(mode="json")

    assert payload["authors"] == []
    assert payload["categories"] == []
    assert payload["abstract"] is None
    assert payload["pdf_url"] is None


def test_blank_paper_identity_is_invalid() -> None:
    with pytest.raises(ValidationError):
        PaperMetadata(
            paper_id=" ",
            title="Title",
            arxiv_url="https://arxiv.org/abs/2501.00003",
            provenance=Provenance(source="arxiv"),
        )


def test_broad_retrieval_query_supports_candidate_pool_and_default_planner_mode() -> None:
    query = RetrievalQuery(
        topic="multimodal llm agents for robotic manipulation",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=100,
    )

    assert query.search_mode == SearchMode.BROAD
    assert query.candidate_pool_size == 100
    assert query.query_planner_mode == QueryPlannerMode.AUTO
    assert query.effective_candidate_pool_size == 100


def test_retrieval_query_rejects_invalid_date_range() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(
            topic="agents",
            start_date=date(2026, 4, 21),
            end_date=date(2026, 4, 20),
        )


def test_retrieval_budget_rejects_invalid_bounds() -> None:
    with pytest.raises(ValidationError):
        RetrievalBudget(candidate_pool_size=0)

    with pytest.raises(ValidationError):
        RetrievalBudget(page_size=101)


def test_query_plan_requires_valid_variants_and_preserves_planner_provenance() -> None:
    plan = QueryPlan(
        search_mode=SearchMode.BROAD,
        planner=QueryPlannerProvenance(
            requested_mode=QueryPlannerMode.AUTO,
            source="deterministic",
        ),
        variants=[
            QueryPlanVariant(
                label="title_terms",
                search_query='ti:"robotic manipulation" OR abs:"robotic manipulation"',
                sort_by="relevance",
            )
        ],
    )

    assert plan.variant_count == 1
    assert plan.planner.requested_mode == QueryPlannerMode.AUTO
    assert plan.variants[0].label == "title_terms"


def test_query_plan_rejects_empty_variant_query() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            search_mode=SearchMode.BROAD,
            planner=QueryPlannerProvenance(
                requested_mode=QueryPlannerMode.DETERMINISTIC,
                source="deterministic",
            ),
            variants=[QueryPlanVariant(label="empty", search_query=" ")],
        )


def test_retrieval_source_metadata_is_run_scoped() -> None:
    metadata = RetrievalSourceMetadata(
        variant_label="broad_terms",
        sort_by="relevance",
        variant_index=0,
        position=3,
        first_seen_order=1,
        query='all:"agent"',
    )

    assert metadata.variant_label == "broad_terms"
    assert metadata.first_seen_order == 1
    assert metadata.query == 'all:"agent"'


def test_recommendation_can_expose_score_breakdown() -> None:
    paper = make_paper()
    breakdown = RankingScoreBreakdown(
        lexical=3.0,
        phrase=2.0,
        total=5.0,
        evidence_score=5.0,
        matched_terms=["agent"],
        matched_phrases=["agent architecture"],
        signals=["lexical", "phrase"],
    )

    recommendation = Recommendation(
        paper=paper,
        rank=1,
        score=5.0,
        rationale="Matched explicit terms: agent.",
        score_breakdown=breakdown,
    )

    assert recommendation.score_breakdown is not None
    assert recommendation.score_breakdown.total == 5.0
    assert recommendation.score_breakdown.signals == ["lexical", "phrase"]
    assert recommendation.score_breakdown.semantic_seed == 0.0
    assert recommendation.score_breakdown.semantic_similarities == []


def test_embedding_contracts_capture_identity_vector_and_trace_safe_cache_metadata() -> None:
    identity = EmbeddingIdentity(
        provider="fake",
        model="semantic-test",
        dimensions=3,
        input_version="paper-metadata-v1",
        input_hash="abc123",
        cache_scope=EmbeddingCacheScope.GLOBAL,
    )
    vector = EmbeddingVector(
        identity=identity,
        vector=[0.1, 0.2, 0.3],
        input_role=EmbeddingInputRole.CANDIDATE,
    )
    cache_metadata = EmbeddingCacheMetadata(
        enabled=True,
        scope=EmbeddingCacheScope.GLOBAL,
        hits=2,
        misses=1,
        writes=1,
        corrupt_entries=0,
    )
    provider_metadata = EmbeddingProviderCacheMetadata(
        provider="fake",
        provider_mode="fake",
        provider_label="fake:semantic-test",
        model="semantic-test",
        dimensions=3,
        cache=cache_metadata,
    )

    payload_json = provider_metadata.model_dump_json()

    assert vector.identity == identity
    assert vector.vector == [0.1, 0.2, 0.3]
    assert cache_metadata.requests == 3
    assert "abc123" not in payload_json
    assert "input_hash" not in payload_json


def test_profile_scoped_embedding_identity_requires_profile_id() -> None:
    with pytest.raises(ValidationError):
        EmbeddingIdentity(
            provider="fake",
            model="semantic-test",
            input_version="paper-metadata-v1",
            input_hash="abc123",
            cache_scope=EmbeddingCacheScope.PROFILE,
        )

    identity = EmbeddingIdentity(
        provider="fake",
        model="semantic-test",
        input_version="paper-metadata-v1",
        input_hash="abc123",
        cache_scope=EmbeddingCacheScope.PROFILE,
        profile_id="demo",
    )

    assert identity.profile_id == "demo"


def test_seed_and_feedback_embedding_vectors_must_be_profile_scoped() -> None:
    identity = EmbeddingIdentity(
        provider="fake",
        model="semantic-test",
        dimensions=3,
        input_version="paper-metadata-v1",
        input_hash="abc123",
        cache_scope=EmbeddingCacheScope.GLOBAL,
    )

    for role in (EmbeddingInputRole.SEED, EmbeddingInputRole.FEEDBACK):
        with pytest.raises(ValidationError):
            EmbeddingVector(
                identity=identity,
                vector=[0.1, 0.2, 0.3],
                input_role=role,
            )


def test_minimal_daily_briefing_payloads_remain_valid() -> None:
    paper = make_paper()
    item = PaperBriefingItem(
        paper_id=paper.paper_id,
        title=paper.title,
        rank=1,
        score=5.0,
        summary="A concise briefing summary.",
        relevance_rationale="Matched explicit agent architecture terms.",
        evidence_source=EvidenceSource.ABSTRACT,
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )
    briefing = DailyBriefing(
        topic="agent architectures",
        executive_summary="One top-ranked paper is available.",
        summary_table=[make_briefing_table_row()],
        highlighted_paper=item,
        items=[item],
        evidence_source=EvidenceSource.ABSTRACT,
        provenance=[paper.provenance],
    )

    payload = briefing.model_dump(mode="json")

    assert item.contributions == []
    assert item.methods == []
    assert item.problem is None
    assert payload["executive_summary"] == "One top-ranked paper is available."
    assert payload["trend_overview"]["status"] == TrendAssessmentStatus.NOT_ASSESSED
    assert payload["top_k_comparisons"] == []
    assert payload["reading_priorities"] == []
    assert payload["evidence_boundary"]["full_text_used"] is False


def test_enhanced_daily_briefing_sections_preserve_legacy_fields() -> None:
    paper = make_paper()
    item = make_briefing_item()
    trend = CandidatePoolTrendOverview(
        status=TrendAssessmentStatus.AVAILABLE,
        summary="Robust agent workflows appear repeatedly in the candidate pool.",
        candidate_count=12,
        abstract_count=10,
        metadata_only_count=2,
        top_k_count=3,
        signals=[
            TrendSignal(
                label="agent workflow",
                signal_type=TrendSignalType.HOTSPOT,
                strength=TrendSignalStrength.MODERATE,
                support_count=5,
                candidate_count=12,
                top_k_count=2,
                evidence_sources=[
                    EvidenceSource.CANDIDATE_POOL,
                    EvidenceSource.ABSTRACT,
                ],
                summary="Repeated across abstracts and titles.",
            )
        ],
        evidence_sources=[EvidenceSource.CANDIDATE_POOL, EvidenceSource.ABSTRACT],
    )
    comparison = TopKComparisonNote(
        dimension="workflow focus",
        note="The top paper focuses on architecture, while rank 2 emphasizes evaluation.",
        paper_ids=[paper.paper_id, "2501.00004"],
        ranks=[1, 2],
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.SUPPORTED,
            sources=[EvidenceSource.ABSTRACT, EvidenceSource.RANKING],
        ),
    )
    priority = ReadingPriority(
        priority=1,
        reading_intent="start with the implementation pattern",
        paper_id=paper.paper_id,
        rank=1,
        reason="It has the strongest rank and directly addresses agent architecture.",
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.SUPPORTED,
            sources=[EvidenceSource.ABSTRACT, EvidenceSource.RANKING],
        ),
    )
    boundary = BriefingEvidenceBoundary(
        evidence_sources=[
            EvidenceSource.METADATA,
            EvidenceSource.ABSTRACT,
            EvidenceSource.RANKING,
            EvidenceSource.CANDIDATE_POOL,
        ],
        unavailable_sources=[EvidenceSource.FULL_TEXT],
        notes=["No PDF or full-text evidence was used."],
    )

    briefing = DailyBriefing(
        topic="agent architectures",
        executive_summary="Top papers emphasize reusable agent workflows.",
        summary_table=[make_briefing_table_row()],
        highlighted_paper=item,
        items=[item],
        evidence_source=EvidenceSource.MIXED,
        provenance=[paper.provenance],
        trend_overview=trend,
        top_k_comparisons=[comparison],
        reading_priorities=[priority],
        evidence_boundary=boundary,
    )

    assert briefing.executive_summary == "Top papers emphasize reusable agent workflows."
    assert briefing.summary_table[0].paper_id == paper.paper_id
    assert briefing.items[0].paper_id == paper.paper_id
    assert briefing.provenance == [paper.provenance]
    assert briefing.trend_overview.signals[0].signal_type == TrendSignalType.HOTSPOT
    assert briefing.top_k_comparisons[0].evidence.sources == [
        EvidenceSource.ABSTRACT,
        EvidenceSource.RANKING,
    ]
    assert briefing.reading_priorities[0].reading_intent == (
        "start with the implementation pattern"
    )
    assert briefing.evidence_boundary.unavailable_sources == [EvidenceSource.FULL_TEXT]


def test_enhanced_paper_briefing_fields_serialize_evidence_and_abstentions() -> None:
    item = make_briefing_item()
    item.problem = EvidenceBoundClaim(
        claim="Daily research workflows need transparent ranking evidence.",
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.SUPPORTED,
            sources=[EvidenceSource.ABSTRACT],
        ),
    )
    item.approach = EvidenceBoundClaim(
        claim="The paper frames retrieval, ranking, and synthesis as staged skills.",
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.SUPPORTED,
            sources=[EvidenceSource.ABSTRACT, EvidenceSource.METADATA],
        ),
    )
    item.reading_guide = EvidenceBoundClaim(
        claim="Read first for the agent boundary design, then inspect ranking details.",
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.PARTIAL,
            sources=[EvidenceSource.ABSTRACT, EvidenceSource.RANKING],
            note="Ranking context supports the reading order.",
        ),
    )
    item.method_claims = [
        EvidenceBoundClaim(
            claim=None,
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.UNAVAILABLE,
                abstention_reason="The abstract does not expose enough method detail.",
            ),
        )
    ]
    item.contribution_claims = [
        EvidenceBoundClaim(
            claim=None,
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.UNAVAILABLE,
                abstention_reason="Contribution claims require stronger abstract support.",
            ),
        )
    ]

    payload = item.model_dump(mode="json")

    assert payload["problem"]["claim"] == (
        "Daily research workflows need transparent ranking evidence."
    )
    assert payload["approach"]["evidence"]["sources"] == ["abstract", "metadata"]
    assert payload["reading_guide"]["evidence"]["status"] == "partial"
    assert payload["method_claims"][0]["claim"] is None
    assert payload["method_claims"][0]["evidence"]["abstention_reason"] == (
        "The abstract does not expose enough method detail."
    )
    assert payload["contribution_claims"][0]["evidence"]["status"] == "unavailable"


def test_field_evidence_requires_sources_or_explicit_abstention() -> None:
    with pytest.raises(ValidationError):
        FieldEvidenceStatus(status=EvidenceSupportStatus.SUPPORTED)

    with pytest.raises(ValidationError):
        FieldEvidenceStatus(status=EvidenceSupportStatus.UNAVAILABLE)

    status = FieldEvidenceStatus(
        status=EvidenceSupportStatus.INSUFFICIENT,
        abstention_reason="Only metadata is available.",
    )

    assert status.abstention_reason == "Only metadata is available."


def test_top_k_comparison_rejects_invalid_ranks() -> None:
    with pytest.raises(ValidationError):
        TopKComparisonNote(
            dimension="workflow focus",
            note="Invalid rank example.",
            ranks=[0],
        )


def test_trend_overview_can_represent_insufficient_candidate_data() -> None:
    briefing = DailyBriefing(
        topic="agent architectures",
        executive_summary="Trend analysis was not available for this run.",
        trend_overview=CandidatePoolTrendOverview(
            status=TrendAssessmentStatus.INSUFFICIENT_DATA,
            candidate_count=2,
            abstract_count=1,
            metadata_only_count=1,
            top_k_count=2,
            limitations=["Candidate pool is too small for broader trend claims."],
            evidence_sources=[EvidenceSource.METADATA, EvidenceSource.CANDIDATE_POOL],
        ),
    )

    assert briefing.trend_overview.status == TrendAssessmentStatus.INSUFFICIENT_DATA
    assert briefing.trend_overview.signals == []
    assert briefing.trend_overview.limitations == [
        "Candidate pool is too small for broader trend claims."
    ]


def test_metadata_only_paper_brief_can_omit_richer_claims() -> None:
    item = make_briefing_item(evidence_source=EvidenceSource.METADATA)

    assert item.problem is None
    assert item.approach is None
    assert item.reading_guide is None
    assert item.relevance_evidence is None


def test_deep_explanation_contract_supports_mode_specific_sections() -> None:
    paper = make_paper()
    explanation = PaperDeepExplanation(
        paper_id=paper.paper_id,
        title=paper.title,
        mode=ExplanationMode.METHOD,
        summary="A concise method explanation.",
        evidence_source=EvidenceSource.FULL_TEXT,
        evidence_note="This explanation is based on the available full-text source.",
        method=MethodExplanation(
            problem="The paper addresses explainable paper recommendation.",
            method_overview="It uses a staged agent workflow.",
            core_workflow=["Retrieve papers", "Rank them", "Explain one paper"],
            inputs_outputs=["Input: topic query", "Output: explanation"],
            innovation="It keeps evidence labels attached to generated claims.",
        ),
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )

    assert explanation.mode == ExplanationMode.METHOD
    assert explanation.method is not None
    assert explanation.experiment is None


def test_config_reads_environment_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAILY_ARXIV_DB_PATH", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_CHAT_COMPLETIONS_PATH", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    monkeypatch.delenv("LLM_RETRY_BACKOFF_SECONDS", raising=False)
    monkeypatch.delenv("LLM_OUTPUT_RETRIES", raising=False)
    monkeypatch.delenv("LLM_BRIEFING_MAX_RETRIES", raising=False)
    monkeypatch.delenv("LLM_BRIEFING_RETRY_BACKOFF_SECONDS", raising=False)
    monkeypatch.delenv("LLM_BRIEFING_OUTPUT_RETRIES", raising=False)
    monkeypatch.delenv("ARXIV_REQUEST_DELAY_SECONDS", raising=False)
    monkeypatch.delenv("DAILY_ARXIV_SEARCH_MODE", raising=False)
    monkeypatch.delenv("DAILY_ARXIV_CANDIDATE_POOL_SIZE", raising=False)
    monkeypatch.delenv("ARXIV_PAGE_SIZE", raising=False)
    monkeypatch.delenv("ARXIV_MAX_REQUESTS_PER_SEARCH", raising=False)
    monkeypatch.delenv("QUERY_PLANNER_MODE", raising=False)

    config = AppConfig.from_env()

    assert config.db_path == "data/daily_arxiv.sqlite3"
    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-5-mini"
    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.llm_chat_completions_path == "/chat/completions"
    assert config.llm_timeout_seconds == 30.0
    assert config.llm_max_retries == 2
    assert config.llm_retry_backoff_seconds == 1.0
    assert config.llm_output_retries == 1
    assert config.llm_briefing_max_retries == 4
    assert config.llm_briefing_retry_backoff_seconds == 1.5
    assert config.llm_briefing_output_retries == 2
    assert config.arxiv_request_delay_seconds == 3.0
    assert config.search_mode == SearchMode.BROAD
    assert config.candidate_pool_size == 100
    assert config.arxiv_page_size == 50
    assert config.arxiv_max_requests_per_search == 4
    assert config.query_planner_mode == QueryPlannerMode.AUTO


def test_config_ignores_invalid_float_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARXIV_REQUEST_DELAY_SECONDS", "not-a-number")

    config = AppConfig.from_env()

    assert config.arxiv_request_delay_seconds == 3.0


def test_config_ignores_invalid_int_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "not-an-int")

    config = AppConfig.from_env()

    assert config.llm_max_retries == 2


def test_config_reads_custom_chat_completion_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_PATH", "v1/chat/completions")

    config = AppConfig.from_env()

    assert config.llm_chat_completions_path == "v1/chat/completions"


def test_config_accepts_openai_api_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    config = AppConfig.from_env()

    assert config.llm_api_key == "sk-test"


def test_config_reads_custom_llm_retry_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "2.5")
    monkeypatch.setenv("LLM_OUTPUT_RETRIES", "2")
    monkeypatch.setenv("LLM_BRIEFING_MAX_RETRIES", "7")
    monkeypatch.setenv("LLM_BRIEFING_RETRY_BACKOFF_SECONDS", "3.5")
    monkeypatch.setenv("LLM_BRIEFING_OUTPUT_RETRIES", "4")

    config = AppConfig.from_env()

    assert config.llm_max_retries == 5
    assert config.llm_retry_backoff_seconds == 2.5
    assert config.llm_output_retries == 2
    assert config.llm_briefing_max_retries == 7
    assert config.llm_briefing_retry_backoff_seconds == 3.5
    assert config.llm_briefing_output_retries == 4


def test_config_reads_custom_search_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_ARXIV_SEARCH_MODE", "strict")
    monkeypatch.setenv("DAILY_ARXIV_CANDIDATE_POOL_SIZE", "75")
    monkeypatch.setenv("ARXIV_PAGE_SIZE", "25")
    monkeypatch.setenv("ARXIV_MAX_REQUESTS_PER_SEARCH", "3")
    monkeypatch.setenv("QUERY_PLANNER_MODE", "deterministic")

    config = AppConfig.from_env()

    assert config.search_mode == SearchMode.STRICT
    assert config.candidate_pool_size == 75
    assert config.arxiv_page_size == 25
    assert config.arxiv_max_requests_per_search == 3
    assert config.query_planner_mode == QueryPlannerMode.DETERMINISTIC


def test_config_ignores_out_of_bounds_search_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAILY_ARXIV_CANDIDATE_POOL_SIZE", "0")
    monkeypatch.setenv("ARXIV_PAGE_SIZE", "101")
    monkeypatch.setenv("ARXIV_MAX_REQUESTS_PER_SEARCH", "0")

    config = AppConfig.from_env()

    assert config.candidate_pool_size == 100
    assert config.arxiv_page_size == 50
    assert config.arxiv_max_requests_per_search == 4
