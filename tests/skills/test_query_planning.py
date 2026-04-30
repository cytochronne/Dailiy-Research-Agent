from datetime import date

from daily_arxiv_agent.contracts import (
    QueryPlannerMode,
    RetrievalQuery,
    SearchMode,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.query_planning import QueryPlanningSkill


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
