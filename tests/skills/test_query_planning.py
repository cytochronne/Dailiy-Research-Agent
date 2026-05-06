from datetime import date

from daily_arxiv_agent.contracts import (
    PaperMetadata,
    Provenance,
    QueryPlannerMode,
    RetrievalQuery,
    SearchMode,
    SeedPreference,
    SeedRecord,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
    build_paper_preference_text,
)
from daily_arxiv_agent.skills.query_planning import QueryPlanningSkill


def make_seed_paper(
    paper_id: str = "2604.20001",
    *,
    title: str = "Multimodal LLM Agents for Robotic Manipulation",
    abstract: str | None = (
        "We present multimodal LLM agents for robotic manipulation with "
        "vision-language planning and closed-loop control."
    ),
    category: str = "cs.RO",
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=[category, "cs.LG"],
        published_date=date(2026, 4, 24),
        updated_date=date(2026, 4, 24),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query=f"id_list:{paper_id}",
        ),
    )


def make_seed_preference(
    records: list[SeedRecord],
    *,
    profile_id: str = "seed-test",
) -> SeedPreference:
    preference_text = "\n\n".join(record.preference_text for record in records)
    return SeedPreference(
        profile_id=profile_id,
        seeds=records,
        preference_text=preference_text,
        vector=DeterministicTextVectorizer().vectorize(preference_text),
    )


def seed_record_from_paper(paper: PaperMetadata) -> SeedRecord:
    return SeedRecord(
        identity=f"arxiv:{paper.paper_id}",
        input_text=paper.paper_id,
        input_type="arxiv_id",
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        paper=paper,
        preference_text=build_paper_preference_text(paper),
    )


def test_broad_topic_planning_produces_multiple_fielded_variants() -> None:
    query = RetrievalQuery(
        topic="multimodal llm agents for robotic manipulation",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )

    result = QueryPlanningSkill().plan(query)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert plan.search_mode == SearchMode.BROAD
    assert plan.planner.source == "deterministic"
    assert plan.required_terms == [
        "multimodal",
        "llm",
        "agent",
        "robotic",
        "manipulation",
    ]
    assert plan.variant_count >= 3
    queries = [variant.search_query for variant in plan.variants]
    assert any("ti:" in query or "abs:" in query for query in queries)
    assert any("all:" in query for query in queries)
    assert not all(
        query == 'all:"multimodal llm agents for robotic manipulation"'
        for query in queries
    )
    assert result.metadata["query_variant_count"] == plan.variant_count
    assert result.metadata["fallback"] is False


def test_strict_topic_planning_includes_phrase_oriented_variant() -> None:
    query = RetrievalQuery(
        topic="graph neural networks",
        search_mode=SearchMode.STRICT,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )

    result = QueryPlanningSkill().plan(query)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    strict_variants = [variant for variant in plan.variants if "phrase" in variant.label]
    assert strict_variants
    assert any('all:"graph neural networks"' in variant.search_query for variant in strict_variants)


def test_empty_topic_with_category_and_date_filters_still_plans_query() -> None:
    query = RetrievalQuery(
        topic=" ",
        category="cs.LG",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )

    result = QueryPlanningSkill().plan(query)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert plan.required_terms == []
    assert plan.variant_count == 1
    search_query = plan.variants[0].search_query
    assert "cat:cs.LG" in search_query
    assert "submittedDate:[202604010000 TO 202604302359]" in search_query


def test_seed_derived_planning_uses_seed_title_abstract_and_filters() -> None:
    paper = make_seed_paper()
    preference = make_seed_preference([seed_record_from_paper(paper)])
    query = RetrievalQuery(
        topic=None,
        category="cs.RO",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
        search_mode=SearchMode.BROAD,
        candidate_pool_size=25,
        max_requests=4,
    )

    result = QueryPlanningSkill().plan_from_seed(query, preference)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert plan.planner.source == "seed_derived"
    assert plan.required_terms == [
        "multimodal",
        "llm",
        "agent",
        "robotic",
        "manipulation",
    ]
    assert "language" in plan.optional_terms
    queries = [variant.search_query for variant in plan.variants]
    assert len(queries) <= 4
    assert all("cat:cs.RO" in search_query for search_query in queries)
    assert all(
        "submittedDate:[202604010000 TO 202604302359]" in search_query
        for search_query in queries
    )
    assert any("ti:" in search_query and "abs:" in search_query for search_query in queries)
    assert any("language" in search_query for search_query in queries)
    assert all("all:*" not in search_query for search_query in queries)
    assert result.metadata["source"] == "seed_derived"
    assert result.metadata["candidate_target"] == 25
    assert result.metadata["raw_terms_debug_only"] is True
    assert "required_terms" in result.metadata["debug_only"]


def test_seed_derived_planning_adds_seed_category_variant_when_unfiltered() -> None:
    paper = make_seed_paper()
    preference = make_seed_preference([seed_record_from_paper(paper)])
    query = RetrievalQuery(
        topic=None,
        search_mode=SearchMode.BROAD,
        max_requests=4,
    )

    result = QueryPlanningSkill().plan_from_seed(query, preference)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert any("cat:cs.RO" in variant.search_query for variant in plan.variants)


def test_duplicate_seed_papers_do_not_duplicate_query_variants() -> None:
    paper = make_seed_paper()
    record = seed_record_from_paper(paper)
    single = make_seed_preference([record])
    duplicate = make_seed_preference([record, record])
    query = RetrievalQuery(
        topic=None,
        search_mode=SearchMode.BROAD,
        max_requests=4,
    )

    single_result = QueryPlanningSkill().plan_from_seed(query, single)
    duplicate_result = QueryPlanningSkill().plan_from_seed(query, duplicate)

    assert single_result.status == SkillStatus.SUCCESS
    assert duplicate_result.status == SkillStatus.SUCCESS
    assert single_result.data is not None
    assert duplicate_result.data is not None
    assert [variant.search_query for variant in duplicate_result.data.variants] == [
        variant.search_query for variant in single_result.data.variants
    ]


def test_seed_metadata_without_usable_text_returns_quality_error() -> None:
    record = SeedRecord(
        identity="arxiv:2604.99999",
        input_text="2604.99999",
        input_type="arxiv_id",
        paper_id="2604.99999",
        title="2604.99999",
        preference_text="2604.99999",
    )
    preference = make_seed_preference([record])
    query = RetrievalQuery(topic=None, search_mode=SearchMode.BROAD)

    result = QueryPlanningSkill().plan_from_seed(query, preference)

    assert result.status == SkillStatus.ERROR
    assert result.data is None
    assert result.error is not None
    assert result.error.code == "semantic_seed_quality_error"
    assert result.metadata["quality_error_reason"] == "seed_metadata_missing_text"
    assert result.metadata["query_variant_count"] == 0


def test_duplicate_and_near_duplicate_terms_are_deduped() -> None:
    query = RetrievalQuery(
        topic="agents agent studies study LLM llms",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.DETERMINISTIC,
    )

    result = QueryPlanningSkill().plan(query)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert plan.required_terms == ["agent", "study", "llm"]


def test_fake_provider_query_planning_returns_structured_llm_output() -> None:
    query = RetrievalQuery(
        topic="multimodal llm agents",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.LLM,
    )

    result = QueryPlanningSkill(provider=FakeLLMProvider()).plan(query)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert plan.planner.requested_mode == QueryPlannerMode.LLM
    assert plan.planner.source == "fake_llm"
    assert "planner_rationale" in result.metadata
    assert result.metadata["fallback"] is False


def test_auto_mode_with_fake_provider_uses_deterministic_plan() -> None:
    query = RetrievalQuery(
        topic="multimodal llm agents",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.AUTO,
    )

    result = QueryPlanningSkill(provider=FakeLLMProvider()).plan(query)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert plan.planner.source == "deterministic"
    assert result.metadata["source"] == "deterministic"


class RaisingPlannerProvider:
    def plan_queries(self, **kwargs):  # noqa: ANN003, ANN201
        raise RuntimeError("LLM query planning did not return valid JSON.")


class DivergentPlannerProvider:
    def plan_queries(self, **kwargs):  # noqa: ANN003, ANN201
        return {
            "required_terms": ["compiler", "register", "allocation"],
            "phrases": ["compiler register allocation"],
            "related_terms": ["static analysis"],
            "rationale": "A divergent plan.",
        }


class UnsafePlannerProvider:
    def plan_queries(self, **kwargs):  # noqa: ANN003, ANN201
        return {
            "required_terms": ['cat:cs.LG OR all:*'],
            "phrases": [],
            "related_terms": [],
            "rationale": "Unsafe fragment.",
        }


class CategoryAndExclusionPlannerProvider:
    def plan_queries(self, **kwargs):  # noqa: ANN003, ANN201
        return {
            "source": "llm",
            "required_terms": ["robotic", "manipulation"],
            "phrases": ["robotic manipulation"],
            "related_terms": ["embodied control"],
            "suggested_categories": ["q-bio.NC", "math-ph"],
            "exclusions": ["survey"],
            "rationale": "Valid category and exclusion output.",
        }


def test_malformed_llm_output_falls_back_to_deterministic_plan() -> None:
    query = RetrievalQuery(
        topic="robotic manipulation",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.LLM,
    )

    result = QueryPlanningSkill(provider=RaisingPlannerProvider()).plan(query)

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "query_planner_llm_failed"
    plan = result.data
    assert plan is not None
    assert plan.planner.source == "deterministic"
    assert plan.planner.fallback_reason == "LLM query planning did not return valid JSON."
    assert result.metadata["fallback"] is True


def test_semantically_divergent_llm_output_falls_back() -> None:
    query = RetrievalQuery(
        topic="robotic manipulation",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.LLM,
    )

    result = QueryPlanningSkill(provider=DivergentPlannerProvider()).plan(query)

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "query_planner_semantic_guard_failed"
    plan = result.data
    assert plan is not None
    assert plan.required_terms == ["robotic", "manipulation"]
    assert plan.planner.source == "deterministic"


def test_llm_output_accepts_hyphenated_categories_and_preserves_exclusions() -> None:
    query = RetrievalQuery(
        topic="robotic manipulation",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.LLM,
    )

    result = QueryPlanningSkill(provider=CategoryAndExclusionPlannerProvider()).plan(query)

    assert result.status == SkillStatus.SUCCESS
    plan = result.data
    assert plan is not None
    assert plan.planner.source == "llm"
    assert plan.exclusions == ["survey"]
    assert result.metadata["exclusions"] == ["survey"]


def test_unsafe_llm_terms_fall_back_to_deterministic_plan() -> None:
    query = RetrievalQuery(
        topic="robotic manipulation",
        search_mode=SearchMode.BROAD,
        query_planner_mode=QueryPlannerMode.LLM,
    )

    result = QueryPlanningSkill(provider=UnsafePlannerProvider()).plan(query)

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "query_planner_invalid_output"
    plan = result.data
    assert plan is not None
    assert plan.required_terms == ["robotic", "manipulation"]
