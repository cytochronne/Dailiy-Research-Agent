"""Query planning for broader, inspectable arXiv retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
import math
import re
from typing import Any

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    QueryPlan,
    QueryPlannerMode,
    QueryPlannerProvenance,
    QueryPlanVariant,
    RetrievalQuery,
    SearchMode,
    SeedPreference,
    SeedRecord,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.base import LLMProvider


MAX_REQUIRED_TERMS = 8
MAX_OPTIONAL_TERMS = 8
MAX_PHRASES = 4
MAX_VARIANTS = 4
SEED_DERIVED_PLANNER_SOURCE = "seed_derived"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "using",
    "via",
    "we",
    "with",
}

UNSAFE_TERM_PATTERN = re.compile(r'[:()[\]{}"*]')
CATEGORY_PATTERN = re.compile(r"^[A-Za-z0-9-]+(?:\.[A-Za-z0-9_-]+)?$")


class QueryPlanningSkill:
    """Create deterministic or provider-assisted arXiv query plans."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        max_variants: int = MAX_VARIANTS,
    ) -> None:
        self.provider = provider
        self.max_variants = max(max_variants, 1)

    def plan(self, query: RetrievalQuery) -> SkillResult[QueryPlan]:
        """Return a bounded query plan, falling back to deterministic output."""

        max_variants = _variant_limit(query, self.max_variants)
        deterministic_plan = build_deterministic_query_plan(
            query,
            requested_mode=query.query_planner_mode,
            max_variants=max_variants,
        )
        requested_mode = query.query_planner_mode
        if requested_mode == QueryPlannerMode.DETERMINISTIC:
            return _success_result(deterministic_plan)

        if self.provider is None:
            if requested_mode == QueryPlannerMode.AUTO:
                return _success_result(deterministic_plan)
            return _fallback_result(
                deterministic_plan,
                code="query_planner_provider_missing",
                message="LLM query planning was requested but no LLM provider is configured.",
            )

        if requested_mode == QueryPlannerMode.AUTO and _is_fake_provider(self.provider):
            return _success_result(deterministic_plan)

        try:
            raw_plan = self.provider.plan_queries(
                query=query,
                deterministic_terms=deterministic_plan.required_terms,
            )
        except Exception as exc:
            return _fallback_result(
                _with_fallback_reason(deterministic_plan, str(exc)),
                code="query_planner_llm_failed",
                message=str(exc),
                retryable=True,
            )

        try:
            llm_plan = _plan_from_provider_output(
                query,
                raw_plan,
                deterministic_plan=deterministic_plan,
                requested_mode=requested_mode,
                max_variants=max_variants,
            )
        except ValueError as exc:
            return _fallback_result(
                _with_fallback_reason(deterministic_plan, str(exc)),
                code="query_planner_invalid_output",
                message=str(exc),
            )

        if not _preserves_required_terms(llm_plan, deterministic_plan.required_terms):
            message = "LLM query plan did not preserve enough deterministic required terms."
            return _fallback_result(
                _with_fallback_reason(deterministic_plan, message),
                code="query_planner_semantic_guard_failed",
                message=message,
            )

        return _success_result(llm_plan)

    def plan_from_seed(
        self,
        query: RetrievalQuery,
        seed_preference: SeedPreference,
    ) -> SkillResult[QueryPlan]:
        """Return a bounded query plan derived from seed-paper metadata."""

        return build_seed_derived_query_plan(
            query,
            seed_preference,
            max_variants=self.max_variants,
        )


def build_deterministic_query_plan(
    query: RetrievalQuery,
    *,
    requested_mode: QueryPlannerMode | None = None,
    max_variants: int = MAX_VARIANTS,
) -> QueryPlan:
    """Build a stable local query plan from user retrieval inputs."""

    max_variants = _variant_limit(query, max_variants)
    required_terms = _topic_terms(query.topic)
    phrases = _topic_phrases(query.topic, required_terms)
    variants = _build_variants(
        query,
        required_terms=required_terms,
        optional_terms=[],
        phrases=phrases,
        suggested_categories=[],
        max_variants=max_variants,
    )
    return QueryPlan(
        search_mode=query.search_mode,
        planner=QueryPlannerProvenance(
            requested_mode=requested_mode or query.query_planner_mode,
            source="deterministic",
        ),
        variants=variants,
        required_terms=required_terms,
        optional_terms=[],
        phrases=phrases,
        rationale="Deterministic plan from normalized topic terms and filters.",
    )


def build_seed_derived_query_plan(
    query: RetrievalQuery,
    seed_preference: SeedPreference,
    *,
    max_variants: int = MAX_VARIANTS,
) -> SkillResult[QueryPlan]:
    """Build a trace-safe query plan from usable seed title/abstract metadata."""

    max_variants = _variant_limit(query, max_variants)
    if not _has_usable_seed_text(seed_preference):
        return _seed_quality_error_result(query, seed_preference)

    title_terms = _seed_title_terms(seed_preference)
    abstract_terms = _seed_abstract_terms(seed_preference)
    required_terms = title_terms[:MAX_REQUIRED_TERMS]
    if not required_terms:
        required_terms = abstract_terms[:MAX_REQUIRED_TERMS]
    required_term_set = set(required_terms)
    optional_terms = [
        term
        for term in abstract_terms
        if term not in required_term_set
    ][:MAX_OPTIONAL_TERMS]
    phrases = _seed_phrases(seed_preference, required_terms)
    suggested_categories = _seed_categories(seed_preference, query)

    variants = _build_seed_variants(
        query,
        required_terms=required_terms,
        optional_terms=optional_terms,
        phrases=phrases,
        suggested_categories=suggested_categories,
        max_variants=max_variants,
    )
    plan = QueryPlan(
        search_mode=query.search_mode,
        planner=QueryPlannerProvenance(
            requested_mode=query.query_planner_mode,
            source=SEED_DERIVED_PLANNER_SOURCE,
        ),
        variants=variants,
        required_terms=required_terms,
        optional_terms=optional_terms,
        phrases=phrases,
        rationale="Seed-derived plan from normalized seed title, abstract, and category metadata.",
    )
    return _seed_success_result(plan, query, seed_preference)


def _variant_limit(query: RetrievalQuery, configured_limit: int) -> int:
    budget_limit = max(query.max_requests, 1)
    if query.search_mode == SearchMode.STRICT:
        return 1
    return max(min(configured_limit, budget_limit), 1)


def _plan_from_provider_output(
    query: RetrievalQuery,
    raw_plan: Mapping[str, Any],
    *,
    deterministic_plan: QueryPlan,
    requested_mode: QueryPlannerMode,
    max_variants: int,
) -> QueryPlan:
    if not isinstance(raw_plan, Mapping):
        raise ValueError("LLM query plan output must be an object.")

    required_terms = _normalize_provider_terms(
        raw_plan.get("required_terms"),
        field_name="required_terms",
        max_items=MAX_REQUIRED_TERMS,
    )
    optional_terms = _normalize_provider_terms(
        raw_plan.get("related_terms", raw_plan.get("optional_terms")),
        field_name="related_terms",
        max_items=MAX_OPTIONAL_TERMS,
        required=False,
    )
    exclusions = _normalize_provider_terms(
        raw_plan.get("exclusions"),
        field_name="exclusions",
        max_items=MAX_OPTIONAL_TERMS,
        required=False,
    )
    phrases = _normalize_provider_phrases(
        raw_plan.get("phrases"),
        max_items=MAX_PHRASES,
    )
    categories = _normalize_provider_categories(raw_plan.get("suggested_categories"))
    rationale = _clean_text(raw_plan.get("rationale"))
    source = _clean_text(raw_plan.get("source")) or "llm"
    model = _clean_text(raw_plan.get("model")) or None

    if not required_terms and deterministic_plan.required_terms:
        raise ValueError("LLM query plan must include required terms.")
    if not required_terms and not _filter_clauses(query):
        raise ValueError("LLM query plan produced no usable terms or filters.")

    variants = _build_variants(
        query,
        required_terms=required_terms,
        optional_terms=optional_terms,
        phrases=phrases or deterministic_plan.phrases,
        suggested_categories=categories,
        max_variants=max_variants,
    )
    return QueryPlan(
        search_mode=query.search_mode,
        planner=QueryPlannerProvenance(
            requested_mode=requested_mode,
            source=source,
            model=model,
        ),
        variants=variants,
        required_terms=required_terms,
        optional_terms=optional_terms,
        exclusions=exclusions,
        phrases=phrases or deterministic_plan.phrases,
        rationale=rationale or "LLM-assisted query plan.",
    )


def _build_variants(
    query: RetrievalQuery,
    *,
    required_terms: Sequence[str],
    optional_terms: Sequence[str],
    phrases: Sequence[str],
    suggested_categories: Sequence[str],
    max_variants: int,
) -> list[QueryPlanVariant]:
    if query.search_mode == SearchMode.STRICT:
        variants = _strict_variants(query, required_terms=required_terms, phrases=phrases)
    else:
        variants = _broad_variants(
            query,
            required_terms=required_terms,
            optional_terms=optional_terms,
            phrases=phrases,
            suggested_categories=suggested_categories,
        )

    deduped: list[QueryPlanVariant] = []
    seen_queries: set[str] = set()
    for variant in variants:
        if variant.search_query in seen_queries:
            continue
        deduped.append(variant)
        seen_queries.add(variant.search_query)
        if len(deduped) >= max_variants:
            break

    if not deduped:
        deduped.append(
            QueryPlanVariant(
                label="filters",
                search_query=_with_filters("all:*", query),
                sort_by="submittedDate",
            )
        )
    return deduped


def _build_seed_variants(
    query: RetrievalQuery,
    *,
    required_terms: Sequence[str],
    optional_terms: Sequence[str],
    phrases: Sequence[str],
    suggested_categories: Sequence[str],
    max_variants: int,
) -> list[QueryPlanVariant]:
    expanded_limit = max(
        max_variants,
        MAX_VARIANTS + len(suggested_categories) + 2,
    )
    variants = _build_variants(
        query,
        required_terms=required_terms,
        optional_terms=optional_terms,
        phrases=phrases,
        suggested_categories=suggested_categories,
        max_variants=expanded_limit,
    )
    if not variants:
        variants = [
            QueryPlanVariant(
                label="seed_filters",
                search_query=_with_filters("", query),
                sort_by="submittedDate",
            )
        ]
    if query.search_mode == SearchMode.STRICT:
        return variants[:max_variants]
    return _prioritize_seed_variants(variants, max_variants=max_variants)


def _prioritize_seed_variants(
    variants: Sequence[QueryPlanVariant],
    *,
    max_variants: int,
) -> list[QueryPlanVariant]:
    if len(variants) <= max_variants:
        return list(variants)

    category_variants = [
        variant for variant in variants if variant.label.startswith("broad_category_")
    ]
    non_category_variants = [
        variant for variant in variants if not variant.label.startswith("broad_category_")
    ]
    if not category_variants or max_variants <= 1:
        return non_category_variants[:max_variants]

    selected = non_category_variants[: max_variants - 1]
    selected.append(category_variants[0])
    return selected


def _strict_variants(
    query: RetrievalQuery,
    *,
    required_terms: Sequence[str],
    phrases: Sequence[str],
) -> list[QueryPlanVariant]:
    variants: list[QueryPlanVariant] = []
    if phrases:
        variants.append(
            QueryPlanVariant(
                label="strict_phrase",
                search_query=_with_filters(f'all:"{_escape_query_text(phrases[0])}"', query),
                sort_by="relevance",
            )
        )
    if required_terms:
        variants.append(
            QueryPlanVariant(
                label="strict_all_terms",
                search_query=_with_filters(_all_terms_query(required_terms, operator="AND"), query),
                sort_by="relevance",
            )
        )
    if _filter_clauses(query):
        variants.append(
            QueryPlanVariant(
                label="strict_filters",
                search_query=_with_filters("", query),
                sort_by="submittedDate",
            )
        )
    return variants


def _broad_variants(
    query: RetrievalQuery,
    *,
    required_terms: Sequence[str],
    optional_terms: Sequence[str],
    phrases: Sequence[str],
    suggested_categories: Sequence[str],
) -> list[QueryPlanVariant]:
    variants: list[QueryPlanVariant] = []
    if required_terms:
        variants.append(
            QueryPlanVariant(
                label="broad_all_terms",
                search_query=_with_filters(_all_terms_query(required_terms, operator="AND"), query),
                sort_by="relevance",
            )
        )
        variants.append(
            QueryPlanVariant(
                label="broad_title_abstract",
                search_query=_with_filters(
                    _title_abstract_query([*required_terms, *optional_terms[:2]]),
                    query,
                ),
                sort_by="relevance",
            )
        )
    if phrases:
        variants.append(
            QueryPlanVariant(
                label="broad_phrases",
                search_query=_with_filters(_phrase_query(phrases[:3]), query),
                sort_by="relevance",
            )
        )
    if optional_terms:
        variants.append(
            QueryPlanVariant(
                label="broad_related_terms",
                search_query=_with_filters(
                    _all_terms_query([*required_terms[:3], *optional_terms[:3]], operator="OR"),
                    query,
                ),
                sort_by="relevance",
            )
        )
    if required_terms:
        variants.append(
            QueryPlanVariant(
                label="broad_recent",
                search_query=_with_filters(_all_terms_query(required_terms, operator="OR"), query),
                sort_by="submittedDate",
            )
        )
    for category in suggested_categories:
        if not query.category:
            variants.append(
                QueryPlanVariant(
                    label=f"broad_category_{category}",
                    search_query=_with_filters(
                        _all_terms_query(required_terms, operator="OR"),
                        query,
                        extra_category=category,
                    ),
                    sort_by="relevance",
                )
            )
    if _filter_clauses(query) and not variants:
        variants.append(
            QueryPlanVariant(
                label="broad_filters",
                search_query=_with_filters("", query),
                sort_by="submittedDate",
            )
        )
    return variants


def _all_terms_query(terms: Sequence[str], *, operator: str) -> str:
    joined = f" {operator} ".join(_field_query("all", term) for term in terms)
    return joined or "all:*"


def _title_abstract_query(terms: Sequence[str]) -> str:
    clauses: list[str] = []
    for term in terms:
        clauses.append(_field_query("ti", term))
        clauses.append(_field_query("abs", term))
    return "(" + " OR ".join(clauses) + ")" if clauses else "all:*"


def _phrase_query(phrases: Sequence[str]) -> str:
    clauses = [f'all:"{_escape_query_text(phrase)}"' for phrase in phrases]
    return "(" + " OR ".join(clauses) + ")" if clauses else "all:*"


def _with_filters(
    core_query: str,
    query: RetrievalQuery,
    *,
    extra_category: str | None = None,
) -> str:
    parts: list[str] = []
    normalized_core = core_query.strip()
    if normalized_core:
        parts.append(normalized_core)
    parts.extend(_filter_clauses(query, extra_category=extra_category))
    return " AND ".join(parts) if parts else "all:*"


def _filter_clauses(
    query: RetrievalQuery,
    *,
    extra_category: str | None = None,
) -> list[str]:
    clauses: list[str] = []
    category = extra_category or query.category
    if category:
        clauses.append(f"cat:{category}")
    if query.start_date or query.end_date:
        clauses.append(_date_clause(query.start_date, query.end_date))
    return clauses


def _date_clause(start_date: date | None, end_date: date | None) -> str:
    start = _date_bound(start_date, end=False)
    end = _date_bound(end_date, end=True)
    return f"submittedDate:[{start} TO {end}]"


def _date_bound(value: date | None, *, end: bool) -> str:
    if value is None:
        return "999912312359" if end else "000101010000"
    suffix = "2359" if end else "0000"
    return f"{value:%Y%m%d}{suffix}"


def _topic_terms(topic: str | None) -> list[str]:
    if not topic:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", topic.lower()):
        normalized = _normalize_token(token)
        if not normalized or normalized in STOPWORDS or len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        terms.append(normalized)
        seen.add(normalized)
        if len(terms) >= MAX_REQUIRED_TERMS:
            break
    return terms


def _topic_phrases(topic: str | None, terms: Sequence[str]) -> list[str]:
    phrases: list[str] = []
    cleaned_topic = _clean_phrase(topic)
    if cleaned_topic and len(cleaned_topic.split()) > 1:
        phrases.append(cleaned_topic)

    for index in range(len(terms) - 1):
        phrase = f"{terms[index]} {terms[index + 1]}"
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= MAX_PHRASES:
            break
    return phrases[:MAX_PHRASES]


def _has_usable_seed_text(seed_preference: SeedPreference) -> bool:
    return any(
        (_is_usable_seed_title(seed) and bool(_topic_terms(seed.title)))
        or bool(_topic_terms(_seed_abstract(seed)))
        for seed in seed_preference.seeds
    )


def _seed_title_terms(seed_preference: SeedPreference) -> list[str]:
    return _dedupe_terms(
        term
        for seed in seed_preference.seeds
        if _is_usable_seed_title(seed)
        for term in _topic_terms(seed.title)
    )


def _seed_abstract_terms(seed_preference: SeedPreference) -> list[str]:
    return _dedupe_terms(
        term
        for seed in seed_preference.seeds
        for term in _topic_terms(_seed_abstract(seed))
    )


def _seed_phrases(
    seed_preference: SeedPreference,
    required_terms: Sequence[str],
) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for seed in seed_preference.seeds:
        if not _is_usable_seed_title(seed):
            continue
        for phrase in _topic_phrases(seed.title, _topic_terms(seed.title)):
            if phrase in seen:
                continue
            phrases.append(phrase)
            seen.add(phrase)
            if len(phrases) >= MAX_PHRASES:
                return phrases

    for index in range(len(required_terms) - 1):
        phrase = f"{required_terms[index]} {required_terms[index + 1]}"
        if phrase in seen:
            continue
        phrases.append(phrase)
        seen.add(phrase)
        if len(phrases) >= MAX_PHRASES:
            break
    return phrases


def _seed_categories(
    seed_preference: SeedPreference,
    query: RetrievalQuery,
) -> list[str]:
    if query.category:
        return []
    categories: list[str] = []
    seen: set[str] = set()
    for seed in seed_preference.seeds:
        paper = seed.paper
        if paper is None:
            continue
        for category in paper.categories:
            if not CATEGORY_PATTERN.fullmatch(category) or category in seen:
                continue
            categories.append(category)
            seen.add(category)
            if len(categories) >= 6:
                return categories
    return categories


def _seed_abstract(seed: SeedRecord) -> str | None:
    if seed.abstract:
        return seed.abstract
    if seed.paper is not None:
        return seed.paper.abstract
    return None


def _is_usable_seed_title(seed: SeedRecord) -> bool:
    title = _clean_phrase(seed.title)
    if len(title.split()) < 2:
        return False
    paper_id = seed.paper_id or (seed.paper.paper_id if seed.paper else None)
    return not paper_id or title != _clean_phrase(paper_id)


def _dedupe_terms(values: Iterable[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        terms.append(value)
        seen.add(value)
        if len(terms) >= max(MAX_REQUIRED_TERMS, MAX_OPTIONAL_TERMS):
            break
    return terms


def _normalize_provider_terms(
    value: Any,
    *,
    field_name: str,
    max_items: int,
    required: bool = True,
) -> list[str]:
    if value is None:
        if required:
            raise ValueError(f"LLM query plan missing {field_name}.")
        return []
    if not isinstance(value, list):
        raise ValueError(f"LLM query plan {field_name} must be a list.")
    if len(value) > max_items:
        raise ValueError(f"LLM query plan {field_name} has too many items.")

    terms: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"LLM query plan {field_name} must contain strings.")
        if UNSAFE_TERM_PATTERN.search(item):
            raise ValueError(f"LLM query plan {field_name} contains unsafe query syntax.")
        normalized_items = _topic_terms(item)
        if not normalized_items:
            raise ValueError(f"LLM query plan {field_name} contains an empty term.")
        normalized = " ".join(normalized_items)
        if normalized in {"and", "or", "not"} or normalized in seen:
            continue
        terms.append(normalized)
        seen.add(normalized)
    return terms


def _normalize_provider_phrases(value: Any, *, max_items: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("LLM query plan phrases must be a list.")
    if len(value) > max_items:
        raise ValueError("LLM query plan phrases has too many items.")
    phrases: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("LLM query plan phrases must contain strings.")
        if UNSAFE_TERM_PATTERN.search(item):
            raise ValueError("LLM query plan phrases contains unsafe query syntax.")
        phrase = _clean_phrase(item)
        if not phrase:
            continue
        if phrase not in seen:
            phrases.append(phrase)
            seen.add(phrase)
    return phrases


def _normalize_provider_categories(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("LLM query plan suggested_categories must be a list.")
    if len(value) > 6:
        raise ValueError("LLM query plan suggested_categories has too many items.")
    categories: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("LLM query plan suggested_categories must contain strings.")
        category = item.strip()
        if not CATEGORY_PATTERN.fullmatch(category):
            raise ValueError("LLM query plan suggested_categories contains invalid category.")
        if category not in categories:
            categories.append(category)
    return categories


def _preserves_required_terms(plan: QueryPlan, deterministic_terms: Sequence[str]) -> bool:
    if not deterministic_terms:
        return True
    candidate_terms: set[str] = set()
    for term in [*plan.required_terms, *plan.optional_terms]:
        candidate_terms.update(_topic_terms(term))
    for phrase in plan.phrases:
        candidate_terms.update(_topic_terms(phrase))
    overlap = len(set(deterministic_terms) & candidate_terms)
    required_overlap = max(1, math.ceil(len(set(deterministic_terms)) / 2))
    return overlap >= required_overlap


def _with_fallback_reason(plan: QueryPlan, fallback_reason: str) -> QueryPlan:
    return plan.model_copy(
        update={
            "planner": plan.planner.model_copy(
                update={"fallback_reason": _clean_text(fallback_reason) or "Unknown fallback."}
            )
        }
    )


def _success_result(plan: QueryPlan) -> SkillResult[QueryPlan]:
    return SkillResult[QueryPlan](
        status=SkillStatus.SUCCESS,
        data=plan,
        evidence_source=EvidenceSource.METADATA,
        metadata=_metadata(plan, fallback=False),
    )


def _fallback_result(
    plan: QueryPlan,
    *,
    code: str,
    message: str,
    retryable: bool = False,
) -> SkillResult[QueryPlan]:
    return SkillResult[QueryPlan](
        status=SkillStatus.FALLBACK,
        data=plan,
        evidence_source=EvidenceSource.METADATA,
        error=SkillError(code=code, message=message, retryable=retryable),
        message="Using deterministic query planning fallback.",
        metadata=_metadata(plan, fallback=True),
    )


def _seed_success_result(
    plan: QueryPlan,
    query: RetrievalQuery,
    seed_preference: SeedPreference,
) -> SkillResult[QueryPlan]:
    return SkillResult[QueryPlan](
        status=SkillStatus.SUCCESS,
        data=plan,
        evidence_source=EvidenceSource.METADATA,
        metadata=_seed_metadata(
            plan,
            query,
            seed_preference,
            fallback=False,
        ),
    )


def _seed_quality_error_result(
    query: RetrievalQuery,
    seed_preference: SeedPreference,
) -> SkillResult[QueryPlan]:
    return SkillResult[QueryPlan](
        status=SkillStatus.ERROR,
        data=None,
        evidence_source=EvidenceSource.METADATA,
        error=SkillError(
            code="semantic_seed_quality_error",
            message="Seed-derived retrieval requires at least one usable seed title or abstract.",
            retryable=False,
        ),
        message="Seed metadata is insufficient for seed-derived retrieval.",
        metadata={
            "requested_mode": query.query_planner_mode.value,
            "source": SEED_DERIVED_PLANNER_SOURCE,
            "fallback": False,
            "fallback_reason": None,
            "query_variant_count": 0,
            "candidate_target": query.effective_candidate_pool_size,
            "raw_terms_debug_only": True,
            "seed_count": len(seed_preference.seeds),
            "seed_paper_count": _seed_paper_count(seed_preference),
            "quality_error_reason": "seed_metadata_missing_text",
            "safe_to_persist": [
                "requested_mode",
                "source",
                "query_variant_count",
                "candidate_target",
                "raw_terms_debug_only",
                "seed_count",
                "seed_paper_count",
                "quality_error_reason",
            ],
            "debug_only": [
                "required_terms",
                "optional_terms",
                "phrases",
                "query_variants",
                "planner_rationale",
            ],
        },
    )


def _seed_metadata(
    plan: QueryPlan,
    query: RetrievalQuery,
    seed_preference: SeedPreference,
    *,
    fallback: bool,
) -> dict[str, Any]:
    metadata = _metadata(plan, fallback=fallback)
    metadata.update(
        {
            "candidate_target": query.effective_candidate_pool_size,
            "raw_terms_debug_only": True,
            "seed_count": len(seed_preference.seeds),
            "seed_paper_count": _seed_paper_count(seed_preference),
            "safe_to_persist": [
                "requested_mode",
                "source",
                "query_variant_count",
                "candidate_target",
                "raw_terms_debug_only",
                "seed_count",
                "seed_paper_count",
            ],
            "debug_only": [
                "required_terms",
                "optional_terms",
                "phrases",
                "exclusions",
                "query_variants",
                "planner_rationale",
            ],
        }
    )
    return metadata


def _seed_paper_count(seed_preference: SeedPreference) -> int:
    return sum(1 for seed in seed_preference.seeds if seed.paper_id or seed.paper)


def _metadata(plan: QueryPlan, *, fallback: bool) -> dict[str, Any]:
    return {
        "requested_mode": plan.planner.requested_mode.value,
        "source": plan.planner.source,
        "fallback": fallback,
        "fallback_reason": plan.planner.fallback_reason,
        "required_terms": plan.required_terms,
        "optional_terms": plan.optional_terms,
        "phrases": plan.phrases,
        "exclusions": plan.exclusions,
        "query_variant_count": plan.variant_count,
        "planner_rationale": plan.rationale,
        "safe_to_persist": [
            "requested_mode",
            "source",
            "required_terms",
            "optional_terms",
            "phrases",
            "exclusions",
            "query_variant_count",
        ],
        "debug_only": ["query_variants", "planner_rationale"],
        "query_variants": [variant.model_dump(mode="json") for variant in plan.variants],
    }


def _clean_phrase(value: str | None) -> str:
    if not value:
        return ""
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return " ".join(token for token in tokens if token)


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _is_fake_provider(provider: LLMProvider) -> bool:
    provider_type = type(provider)
    return (
        provider_type.__name__ == "FakeLLMProvider"
        and provider_type.__module__.endswith(".fake")
    )


def _escape_query_text(value: str) -> str:
    return " ".join(value.replace('"', "").split())


def _field_query(field: str, value: str) -> str:
    escaped = _escape_query_text(value)
    if " " in escaped:
        return f'{field}:"{escaped}"'
    return f"{field}:{escaped}"


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
