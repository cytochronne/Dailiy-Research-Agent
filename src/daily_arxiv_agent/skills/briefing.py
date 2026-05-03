"""Daily briefing generation from ranked papers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any

from daily_arxiv_agent.contracts import (
    BriefingEvidenceBoundary,
    BriefingTableRow,
    CandidatePoolTrendOverview,
    DailyBriefing,
    EvidenceSource,
    PaperBriefingItem,
    PaperMetadata,
    Provenance,
    QueryPlan,
    RetrievalQuery,
    RetrievalSourceMetadata,
    Recommendation,
    SkillError,
    SkillResult,
    SkillStatus,
    TrendAssessmentStatus,
    TrendSignal,
    TrendSignalStrength,
    TrendSignalType,
)
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.llm.provider import create_llm_provider
from daily_arxiv_agent.skills.extraction import PaperExtractionSkill


MIN_TREND_CANDIDATES = 3
MIN_SIGNAL_SUPPORT = 2
MAX_TREND_SIGNALS = 6

TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]+")

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
    "with",
}

GENERIC_SIGNAL_TERMS = STOPWORDS | {
    "about",
    "across",
    "analysis",
    "approach",
    "based",
    "benchmark",
    "benchmarks",
    "data",
    "dataset",
    "datasets",
    "demonstrate",
    "evaluation",
    "experiments",
    "framework",
    "method",
    "methods",
    "model",
    "models",
    "new",
    "note",
    "paper",
    "papers",
    "pipeline",
    "pipelines",
    "propose",
    "result",
    "results",
    "study",
    "system",
    "systems",
    "task",
    "tasks",
    "this",
    "toward",
    "towards",
    "use",
    "used",
    "uses",
    "we",
}


@dataclass(frozen=True)
class _CandidateRecord:
    paper: PaperMetadata
    title_text: str
    abstract_text: str
    title_tokens: tuple[str, ...]
    abstract_tokens: tuple[str, ...]
    categories: frozenset[str]
    date_bucket: str | None
    variants: frozenset[str]

    @property
    def has_abstract(self) -> bool:
        return bool(self.paper.abstract)


class DailyBriefingSkill:
    """Generate a first daily briefing from ranked metadata and abstracts."""

    def __init__(self, *, provider: LLMProvider | None = None) -> None:
        self.provider = provider or create_llm_provider()

    def generate(
        self,
        *,
        topic: str,
        recommendations: Sequence[Recommendation],
        extraction_results: Sequence[SkillResult[PaperBriefingItem]] | None = None,
        candidate_papers: Sequence[PaperMetadata] | None = None,
        query_plan: QueryPlan | None = None,
        retrieval_query: RetrievalQuery | None = None,
        retrieval_source_metadata_by_paper_id: Mapping[
            str, Sequence[RetrievalSourceMetadata | Mapping[str, object]]
        ]
        | None = None,
        ranking_metadata: Mapping[str, object] | None = None,
    ) -> SkillResult[DailyBriefing]:
        trend_overview = _candidate_pool_trend_overview(
            topic=topic,
            candidate_papers=candidate_papers,
            recommendations=recommendations,
            query_plan=query_plan,
            retrieval_query=retrieval_query,
            retrieval_source_metadata_by_paper_id=(
                retrieval_source_metadata_by_paper_id
            ),
            ranking_metadata=ranking_metadata,
        )
        if not recommendations:
            evidence_source = _briefing_evidence_source(
                EvidenceSource.METADATA,
                trend_overview=trend_overview,
            )
            briefing = DailyBriefing(
                topic=topic,
                executive_summary=f"No ranked papers were available for '{topic}'.",
                evidence_source=evidence_source,
                trend_overview=trend_overview,
                evidence_boundary=_evidence_boundary(
                    evidence_source=evidence_source,
                    trend_overview=trend_overview,
                ),
            )
            return SkillResult[DailyBriefing](
                status=SkillStatus.EMPTY,
                data=briefing,
                evidence_source=evidence_source,
                message="No ranked papers are available for a daily briefing.",
                metadata={
                    "topic": topic,
                    "trend_analysis": _trend_metadata(trend_overview),
                },
            )

        if extraction_results is None:
            extraction_skill = PaperExtractionSkill(provider=self.provider)
            extraction_results = [
                extraction_skill.extract(recommendation, topic=topic)
                for recommendation in recommendations
            ]
        extracted_items = [
            result.data for result in extraction_results if result.data is not None
        ]
        table = [_table_row(recommendation) for recommendation in recommendations]
        evidence_source = _combined_evidence(extracted_items)
        provenance = [recommendation.paper.provenance for recommendation in recommendations]
        extraction_errors = [
            result.error
            for result in extraction_results
            if result.status in {SkillStatus.FALLBACK, SkillStatus.ERROR}
            and result.error is not None
        ]

        try:
            executive_summary = self.provider.summarize_briefing(
                topic=topic,
                items=extracted_items,
            )
        except Exception as exc:
            briefing_evidence_source = _briefing_evidence_source(
                evidence_source,
                trend_overview=trend_overview,
            )
            briefing = _briefing(
                topic=topic,
                executive_summary=_fallback_summary(topic, extracted_items),
                table=table,
                items=extracted_items,
                evidence_source=briefing_evidence_source,
                provenance=provenance,
                trend_overview=trend_overview,
            )
            return SkillResult[DailyBriefing](
                status=SkillStatus.FALLBACK,
                data=briefing,
                evidence_source=briefing_evidence_source,
                provenance=provenance,
                error=SkillError(
                    code="llm_briefing_failed",
                    message=f"LLM briefing generation failed: {exc}",
                    retryable=True,
                ),
                message="Using deterministic fallback briefing.",
                metadata={
                    "topic": topic,
                    "trend_analysis": _trend_metadata(trend_overview),
                },
            )

        briefing_evidence_source = _briefing_evidence_source(
            evidence_source,
            trend_overview=trend_overview,
        )
        briefing = _briefing(
            topic=topic,
            executive_summary=executive_summary,
            table=table,
            items=extracted_items,
            evidence_source=briefing_evidence_source,
            provenance=provenance,
            trend_overview=trend_overview,
        )
        if extraction_errors:
            codes = ", ".join(sorted({error.code for error in extraction_errors}))
            messages = "; ".join(error.message for error in extraction_errors)
            return SkillResult[DailyBriefing](
                status=SkillStatus.FALLBACK,
                data=briefing,
                evidence_source=briefing_evidence_source,
                provenance=provenance,
                error=SkillError(
                    code=codes,
                    message=messages,
                    retryable=any(error.retryable for error in extraction_errors),
                ),
                message="Using fallback extraction for one or more briefing items.",
                metadata={
                    "topic": topic,
                    "trend_analysis": _trend_metadata(trend_overview),
                },
            )
        return SkillResult[DailyBriefing](
            status=SkillStatus.SUCCESS,
            data=briefing,
            evidence_source=briefing_evidence_source,
            provenance=provenance,
            metadata={
                "topic": topic,
                "trend_analysis": _trend_metadata(trend_overview),
            },
        )


def _briefing(
    *,
    topic: str,
    executive_summary: str,
    table: list[BriefingTableRow],
    items: list[PaperBriefingItem],
    evidence_source: EvidenceSource,
    provenance: list[Provenance],
    trend_overview: CandidatePoolTrendOverview,
) -> DailyBriefing:
    return DailyBriefing(
        topic=topic,
        executive_summary=executive_summary,
        summary_table=table,
        highlighted_paper=items[0] if items else None,
        items=items,
        evidence_source=evidence_source,
        provenance=provenance,
        trend_overview=trend_overview,
        evidence_boundary=_evidence_boundary(
            evidence_source=evidence_source,
            trend_overview=trend_overview,
        ),
    )


def _table_row(recommendation: Recommendation) -> BriefingTableRow:
    return BriefingTableRow(
        rank=recommendation.rank,
        paper_id=recommendation.paper.paper_id,
        title=recommendation.paper.title,
        score=recommendation.score,
        key_reason=recommendation.rationale,
        evidence_source=recommendation.evidence_source,
        arxiv_url=recommendation.paper.arxiv_url,
    )


def _combined_evidence(items: Sequence[PaperBriefingItem]) -> EvidenceSource:
    if any(item.evidence_source == EvidenceSource.ABSTRACT for item in items):
        return EvidenceSource.ABSTRACT
    return EvidenceSource.METADATA


def _fallback_summary(topic: str, items: Sequence[PaperBriefingItem]) -> str:
    return (
        f"Deterministic fallback briefing for '{topic}' includes {len(items)} "
        "ranked paper(s)."
    )


def _candidate_pool_trend_overview(
    *,
    topic: str,
    candidate_papers: Sequence[PaperMetadata] | None,
    recommendations: Sequence[Recommendation],
    query_plan: QueryPlan | None,
    retrieval_query: RetrievalQuery | None,
    retrieval_source_metadata_by_paper_id: Mapping[
        str, Sequence[RetrievalSourceMetadata | Mapping[str, object]]
    ]
    | None,
    ranking_metadata: Mapping[str, object] | None,
) -> CandidatePoolTrendOverview:
    if candidate_papers is None:
        return CandidatePoolTrendOverview(
            status=TrendAssessmentStatus.NOT_ASSESSED,
            summary="Broader candidate-pool trends were not assessed for this briefing.",
            limitations=[
                "Candidate-pool trend analysis was not requested for this briefing."
            ],
        )

    candidates = _unique_candidate_papers(candidate_papers)
    candidate_count = len(candidates)
    abstract_count = sum(1 for paper in candidates if paper.abstract)
    metadata_only_count = candidate_count - abstract_count
    top_k_count = len(recommendations)
    evidence_sources = _trend_evidence_sources(
        abstract_count=abstract_count,
        candidate_count=candidate_count,
    )

    if candidate_count == 0:
        return CandidatePoolTrendOverview(
            status=TrendAssessmentStatus.NOT_ASSESSED,
            summary="Broader candidate-pool trends were not assessed for this briefing.",
            candidate_count=0,
            abstract_count=0,
            metadata_only_count=0,
            top_k_count=top_k_count,
            limitations=["No candidate papers were provided for trend analysis."],
            evidence_sources=evidence_sources,
        )

    if candidate_count < MIN_TREND_CANDIDATES:
        return CandidatePoolTrendOverview(
            status=TrendAssessmentStatus.INSUFFICIENT_DATA,
            summary=(
                "Candidate-pool trend analysis has insufficient data for broader "
                "hotspot claims."
            ),
            candidate_count=candidate_count,
            abstract_count=abstract_count,
            metadata_only_count=metadata_only_count,
            top_k_count=top_k_count,
            limitations=[
                "Candidate pool is too small for broader trend claims.",
            ],
            evidence_sources=evidence_sources,
        )

    source_metadata = _normalize_source_metadata_by_paper_id(
        retrieval_source_metadata_by_paper_id
    )
    records = [
        _candidate_record(
            paper,
            source_metadata=source_metadata.get(paper.paper_id, ()),
        )
        for paper in candidates
    ]
    top_k_ids = {recommendation.paper.paper_id for recommendation in recommendations}
    matched_ranking_labels = _matched_ranking_labels(recommendations)
    query_echo_labels = _query_echo_labels(
        topic=topic,
        query_plan=query_plan,
        retrieval_query=retrieval_query,
        ranking_metadata=ranking_metadata,
    )
    representative_support = _representative_support_count(candidate_count)

    signals = _merge_signals(
        [
            *_text_trend_signals(
                records,
                top_k_ids=top_k_ids,
                matched_ranking_labels=matched_ranking_labels,
                query_echo_labels=query_echo_labels,
                candidate_count=candidate_count,
                representative_support=representative_support,
            ),
            *_category_trend_signals(
                records,
                top_k_ids=top_k_ids,
                candidate_count=candidate_count,
            ),
            *_ranking_only_signals(
                matched_ranking_labels=matched_ranking_labels,
                existing_labels=set(),
                query_echo_labels=query_echo_labels,
                candidate_count=candidate_count,
            ),
        ]
    )
    signals = _sort_and_cap_signals(signals)
    signal_labels = {signal.label for signal in signals}
    ranking_only_signals = _ranking_only_signals(
        matched_ranking_labels=matched_ranking_labels,
        existing_labels=signal_labels,
        query_echo_labels=query_echo_labels,
        candidate_count=candidate_count,
    )
    if ranking_only_signals and len(signals) < MAX_TREND_SIGNALS:
        signals = _sort_and_cap_signals([*signals, *ranking_only_signals])

    query_echo_count = sum(1 for signal in signals if signal.query_echo)
    representative_signal_count = sum(
        1
        for signal in signals
        if signal.signal_type == TrendSignalType.HOTSPOT
        or (
            signal.signal_type == TrendSignalType.TOPIC
            and signal.strength != TrendSignalStrength.WEAK
        )
    )
    limitations = _overview_limitations(
        signals=signals,
        query_echo_count=query_echo_count,
        representative_signal_count=representative_signal_count,
    )
    status = _trend_status(
        signals=signals,
        representative_signal_count=representative_signal_count,
    )

    return CandidatePoolTrendOverview(
        status=status,
        summary=_trend_summary(
            candidate_count=candidate_count,
            abstract_count=abstract_count,
            metadata_only_count=metadata_only_count,
            representative_signal_count=representative_signal_count,
        ),
        candidate_count=candidate_count,
        abstract_count=abstract_count,
        metadata_only_count=metadata_only_count,
        top_k_count=top_k_count,
        signals=signals,
        limitations=limitations,
        evidence_sources=evidence_sources,
    )


def _unique_candidate_papers(
    candidate_papers: Sequence[PaperMetadata],
) -> list[PaperMetadata]:
    candidates_by_id: dict[str, PaperMetadata] = {}
    for paper in candidate_papers:
        candidates_by_id.setdefault(paper.paper_id, paper)
    return list(candidates_by_id.values())


def _candidate_record(
    paper: PaperMetadata,
    *,
    source_metadata: Sequence[RetrievalSourceMetadata],
) -> _CandidateRecord:
    return _CandidateRecord(
        paper=paper,
        title_text=_normalized_text(paper.title),
        abstract_text=_normalized_text(paper.abstract or ""),
        title_tokens=tuple(_tokenize(paper.title)),
        abstract_tokens=tuple(_tokenize(paper.abstract or "")),
        categories=frozenset(paper.categories),
        date_bucket=_date_bucket(paper),
        variants=frozenset(metadata.variant_label for metadata in source_metadata),
    )


def _text_trend_signals(
    records: Sequence[_CandidateRecord],
    *,
    top_k_ids: set[str],
    matched_ranking_labels: Mapping[str, int],
    query_echo_labels: set[str],
    candidate_count: int,
    representative_support: int,
) -> list[TrendSignal]:
    label_support: dict[str, set[str]] = defaultdict(set)
    phrase_labels: set[str] = set()
    term_labels: set[str] = set()

    for record in records:
        labels = _labels_for_record(record)
        for label, is_phrase in labels.items():
            label_support[label].add(record.paper.paper_id)
            if is_phrase:
                phrase_labels.add(label)
            else:
                term_labels.add(label)

    term_labels = _remove_terms_covered_by_phrases(
        term_labels=term_labels,
        phrase_labels=phrase_labels,
        label_support=label_support,
    )
    allowed_labels = phrase_labels | term_labels

    signals: list[TrendSignal] = []
    records_by_id = {record.paper.paper_id: record for record in records}
    for label in allowed_labels:
        support_ids = label_support[label]
        if len(support_ids) < MIN_SIGNAL_SUPPORT:
            continue
        supporting_records = [
            records_by_id[paper_id] for paper_id in support_ids if paper_id in records_by_id
        ]
        signals.append(
            _trend_signal_from_support(
                label=label,
                supporting_records=supporting_records,
                top_k_ids=top_k_ids,
                matched_ranking_count=matched_ranking_labels.get(label, 0),
                query_echo=label in query_echo_labels,
                candidate_count=candidate_count,
                representative_support=representative_support,
            )
        )
    return signals


def _labels_for_record(record: _CandidateRecord) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for tokens in (record.title_tokens, record.abstract_tokens):
        for token in tokens:
            if _valid_signal_term(token):
                labels[token] = False
        for phrase in _phrases_from_tokens(tokens):
            labels[phrase] = True
    return labels


def _remove_terms_covered_by_phrases(
    *,
    term_labels: set[str],
    phrase_labels: set[str],
    label_support: Mapping[str, set[str]],
) -> set[str]:
    filtered: set[str] = set()
    for term in term_labels:
        term_support = len(label_support[term])
        covered = False
        for phrase in phrase_labels:
            phrase_terms = set(phrase.split())
            if term not in phrase_terms:
                continue
            if len(label_support[phrase]) >= max(1, math.ceil(term_support * 0.6)):
                covered = True
                break
        if not covered:
            filtered.add(term)
    return filtered


def _trend_signal_from_support(
    *,
    label: str,
    supporting_records: Sequence[_CandidateRecord],
    top_k_ids: set[str],
    matched_ranking_count: int,
    query_echo: bool,
    candidate_count: int,
    representative_support: int,
) -> TrendSignal:
    support_ids = {record.paper.paper_id for record in supporting_records}
    support_count = len(support_ids)
    top_k_count = max(len(support_ids & top_k_ids), matched_ranking_count)
    independent_count = _independent_dimension_count(supporting_records)
    representative = support_count >= representative_support and independent_count > 0

    if query_echo:
        signal_type = TrendSignalType.TOPIC
        strength = (
            TrendSignalStrength.MODERATE
            if representative
            else TrendSignalStrength.WEAK
        )
    elif representative:
        signal_type = TrendSignalType.HOTSPOT
        strength = _representative_strength(
            support_count=support_count,
            candidate_count=candidate_count,
            independent_count=independent_count,
        )
    else:
        signal_type = TrendSignalType.TOPIC
        strength = TrendSignalStrength.WEAK

    limitations = _signal_limitations(
        query_echo=query_echo,
        representative=representative,
        independent_count=independent_count,
    )
    return TrendSignal(
        label=label,
        signal_type=signal_type,
        strength=strength,
        support_count=support_count,
        candidate_count=candidate_count,
        top_k_count=top_k_count,
        evidence_sources=_support_evidence_sources(
            label=label,
            supporting_records=supporting_records,
            matched_ranking_count=matched_ranking_count,
        ),
        summary=_signal_summary(label, support_count, candidate_count, top_k_count),
        limitations=limitations,
        query_echo=query_echo,
    )


def _category_trend_signals(
    records: Sequence[_CandidateRecord],
    *,
    top_k_ids: set[str],
    candidate_count: int,
) -> list[TrendSignal]:
    category_support: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for category in record.categories:
            category_support[category].add(record.paper.paper_id)

    signals: list[TrendSignal] = []
    for category, support_ids in category_support.items():
        support_count = len(support_ids)
        if support_count < MIN_SIGNAL_SUPPORT:
            continue
        signals.append(
            TrendSignal(
                label=category,
                signal_type=TrendSignalType.CATEGORY,
                strength=_category_strength(
                    support_count=support_count,
                    candidate_count=candidate_count,
                ),
                support_count=support_count,
                candidate_count=candidate_count,
                top_k_count=len(support_ids & top_k_ids),
                evidence_sources=[
                    EvidenceSource.CANDIDATE_POOL,
                    EvidenceSource.METADATA,
                ],
                summary=(
                    f"{support_count} of {candidate_count} candidate paper(s) "
                    f"include category {category}."
                ),
                limitations=[
                    "Category concentration is metadata evidence, not method evidence."
                ],
            )
        )
    return signals


def _ranking_only_signals(
    *,
    matched_ranking_labels: Mapping[str, int],
    existing_labels: set[str],
    query_echo_labels: set[str],
    candidate_count: int,
) -> list[TrendSignal]:
    signals: list[TrendSignal] = []
    for label, support_count in matched_ranking_labels.items():
        if label in existing_labels or support_count < MIN_SIGNAL_SUPPORT:
            continue
        query_echo = label in query_echo_labels
        signals.append(
            TrendSignal(
                label=label,
                signal_type=TrendSignalType.TOPIC,
                strength=TrendSignalStrength.WEAK,
                support_count=support_count,
                candidate_count=candidate_count,
                top_k_count=support_count,
                evidence_sources=[EvidenceSource.RANKING],
                summary=(
                    f"{support_count} Top-K paper(s) matched this ranking term "
                    "or phrase."
                ),
                limitations=[
                    "Ranking context alone is not enough for a candidate-pool hotspot.",
                    *(
                        ["Echoes the search strategy; not treated as a hotspot."]
                        if query_echo
                        else []
                    ),
                ],
                query_echo=query_echo,
            )
        )
    return signals


def _merge_signals(signals: Sequence[TrendSignal]) -> list[TrendSignal]:
    merged: dict[str, TrendSignal] = {}
    for signal in signals:
        existing = merged.get(signal.label)
        if existing is None:
            merged[signal.label] = signal
            continue
        merged[signal.label] = TrendSignal(
            label=signal.label,
            signal_type=_stronger_signal_type(existing.signal_type, signal.signal_type),
            strength=_stronger_strength(existing.strength, signal.strength),
            support_count=max(existing.support_count, signal.support_count),
            candidate_count=existing.candidate_count or signal.candidate_count,
            top_k_count=max(existing.top_k_count or 0, signal.top_k_count or 0),
            evidence_sources=_ordered_sources(
                [*existing.evidence_sources, *signal.evidence_sources]
            ),
            summary=existing.summary or signal.summary,
            limitations=_dedupe_text([*existing.limitations, *signal.limitations]),
            query_echo=existing.query_echo or signal.query_echo,
        )
    return list(merged.values())


def _sort_and_cap_signals(signals: Sequence[TrendSignal]) -> list[TrendSignal]:
    return sorted(signals, key=_signal_sort_key)[:MAX_TREND_SIGNALS]


def _signal_sort_key(signal: TrendSignal) -> tuple[bool, int, int, int, int, str]:
    return (
        signal.query_echo,
        {
            TrendSignalType.HOTSPOT: 0,
            TrendSignalType.TOPIC: 1,
            TrendSignalType.CATEGORY: 2,
            TrendSignalType.EVIDENCE_COVERAGE: 3,
        }[signal.signal_type],
        -_strength_rank(signal.strength),
        -signal.support_count,
        -(signal.top_k_count or 0),
        signal.label,
    )


def _matched_ranking_labels(
    recommendations: Sequence[Recommendation],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for recommendation in recommendations:
        if recommendation.score_breakdown is None:
            continue
        for phrase in recommendation.score_breakdown.matched_phrases:
            normalized = _normalized_phrase(phrase)
            if normalized:
                counts[normalized] += 1
        for term in recommendation.score_breakdown.matched_terms:
            normalized = _normalized_term(term)
            if normalized and _valid_signal_term(normalized):
                counts[normalized] += 1
    return dict(counts)


def _query_echo_labels(
    *,
    topic: str,
    query_plan: QueryPlan | None,
    retrieval_query: RetrievalQuery | None,
    ranking_metadata: Mapping[str, object] | None,
) -> set[str]:
    labels: set[str] = set()
    _add_query_text(labels, topic)
    if retrieval_query is not None and retrieval_query.topic:
        _add_query_text(labels, retrieval_query.topic)
    if query_plan is not None:
        for term in [*query_plan.required_terms, *query_plan.optional_terms]:
            normalized = _normalized_term(term)
            if normalized:
                labels.add(normalized)
        for phrase in query_plan.phrases:
            normalized = _normalized_phrase(phrase)
            if normalized:
                labels.add(normalized)
                labels.update(normalized.split())
    if ranking_metadata is not None:
        for key in (
            "query_terms",
            "required_terms",
            "optional_terms",
            "expanded_terms",
            "matched_terms",
        ):
            _add_metadata_terms(labels, ranking_metadata.get(key))
        for key in ("query_phrases", "phrases", "matched_phrases"):
            _add_metadata_phrases(labels, ranking_metadata.get(key))
    return labels


def _add_query_text(labels: set[str], text: str) -> None:
    tokens = [token for token in _tokenize(text) if _valid_signal_term(token)]
    labels.update(tokens)
    if len(tokens) > 1:
        labels.add(" ".join(tokens))
    for phrase in _phrases_from_tokens(tokens):
        labels.add(phrase)


def _add_metadata_terms(labels: set[str], value: object) -> None:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = _normalized_term(item)
        if normalized:
            labels.add(normalized)


def _add_metadata_phrases(labels: set[str], value: object) -> None:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = _normalized_phrase(item)
        if normalized:
            labels.add(normalized)
            labels.update(normalized.split())


def _phrases_from_tokens(tokens: Sequence[str]) -> set[str]:
    phrases: set[str] = set()
    for size in (2, 3):
        for index in range(0, len(tokens) - size + 1):
            phrase_tokens = tokens[index : index + size]
            if any(not _valid_signal_term(token) for token in phrase_tokens):
                continue
            if len(set(phrase_tokens)) < len(phrase_tokens):
                continue
            phrases.add(" ".join(phrase_tokens))
    return phrases


def _valid_signal_term(token: str) -> bool:
    return len(token) >= 3 and token not in GENERIC_SIGNAL_TERMS


def _support_evidence_sources(
    *,
    label: str,
    supporting_records: Sequence[_CandidateRecord],
    matched_ranking_count: int,
) -> list[EvidenceSource]:
    sources = [EvidenceSource.CANDIDATE_POOL]
    if any(_label_in_text(label, record.abstract_text) for record in supporting_records):
        sources.append(EvidenceSource.ABSTRACT)
    if any(
        not record.has_abstract or _label_in_text(label, record.title_text)
        for record in supporting_records
    ):
        sources.append(EvidenceSource.METADATA)
    if matched_ranking_count:
        sources.append(EvidenceSource.RANKING)
    return _ordered_sources(sources)


def _label_in_text(label: str, text: str) -> bool:
    if " " in label:
        return label in text
    return label in text.split()


def _independent_dimension_count(records: Sequence[_CandidateRecord]) -> int:
    variant_count = len({variant for record in records for variant in record.variants})
    category_count = len({category for record in records for category in record.categories})
    date_count = len({record.date_bucket for record in records if record.date_bucket})
    return sum(count >= 2 for count in (variant_count, category_count, date_count))


def _representative_support_count(candidate_count: int) -> int:
    return min(candidate_count, max(3, math.ceil(candidate_count * 0.3)))


def _representative_strength(
    *,
    support_count: int,
    candidate_count: int,
    independent_count: int,
) -> TrendSignalStrength:
    if support_count >= math.ceil(candidate_count * 0.5) and independent_count >= 2:
        return TrendSignalStrength.STRONG
    return TrendSignalStrength.MODERATE


def _category_strength(
    *,
    support_count: int,
    candidate_count: int,
) -> TrendSignalStrength:
    if support_count >= math.ceil(candidate_count * 0.6):
        return TrendSignalStrength.STRONG
    if support_count >= _representative_support_count(candidate_count):
        return TrendSignalStrength.MODERATE
    return TrendSignalStrength.WEAK


def _signal_limitations(
    *,
    query_echo: bool,
    representative: bool,
    independent_count: int,
) -> list[str]:
    limitations: list[str] = []
    if query_echo and representative:
        limitations.append(
            "Echoes the search strategy but is repeated across independent "
            "candidate evidence."
        )
    elif query_echo:
        limitations.append("Echoes the search strategy; not treated as a hotspot.")
    if not representative:
        limitations.append(
            "Support is below the representativeness gate for hotspot wording."
        )
    if independent_count == 0:
        limitations.append(
            "No independent category, date, or retrieval-source spread was visible."
        )
    return limitations


def _signal_summary(
    label: str,
    support_count: int,
    candidate_count: int,
    top_k_count: int,
) -> str:
    return (
        f"'{label}' appears in {support_count} of {candidate_count} candidate "
        f"paper(s), including {top_k_count} Top-K paper(s)."
    )


def _overview_limitations(
    *,
    signals: Sequence[TrendSignal],
    query_echo_count: int,
    representative_signal_count: int,
) -> list[str]:
    limitations: list[str] = []
    if not signals:
        limitations.append(
            "No repeated candidate-pool signals passed the support threshold."
        )
    if representative_signal_count == 0:
        limitations.append(
            "No topic signal passed both support and representativeness gates."
        )
    if query_echo_count:
        limitations.append(
            f"{query_echo_count} signal(s) were downgraded because they echo the "
            "retrieval or ranking strategy."
        )
    return limitations


def _trend_status(
    *,
    signals: Sequence[TrendSignal],
    representative_signal_count: int,
) -> TrendAssessmentStatus:
    if representative_signal_count:
        return TrendAssessmentStatus.AVAILABLE
    if signals:
        return TrendAssessmentStatus.LIMITED
    return TrendAssessmentStatus.INSUFFICIENT_DATA


def _trend_summary(
    *,
    candidate_count: int,
    abstract_count: int,
    metadata_only_count: int,
    representative_signal_count: int,
) -> str:
    if representative_signal_count:
        return (
            f"Candidate-pool analysis found {representative_signal_count} "
            f"representative topic signal(s) across {candidate_count} candidates "
            f"({abstract_count} with abstracts, {metadata_only_count} metadata-only)."
        )
    return (
        f"Candidate-pool analysis covered {candidate_count} candidates "
        f"({abstract_count} with abstracts, {metadata_only_count} metadata-only), "
        "but broader hotspot claims are limited by the available evidence."
    )


def _trend_evidence_sources(
    *,
    abstract_count: int,
    candidate_count: int,
) -> list[EvidenceSource]:
    if candidate_count == 0:
        return []
    sources = [EvidenceSource.CANDIDATE_POOL, EvidenceSource.METADATA]
    if abstract_count:
        sources.append(EvidenceSource.ABSTRACT)
    return sources


def _normalize_source_metadata_by_paper_id(
    source_metadata_by_paper_id: Mapping[
        str, Sequence[RetrievalSourceMetadata | Mapping[str, object]]
    ]
    | None,
) -> dict[str, list[RetrievalSourceMetadata]]:
    if not source_metadata_by_paper_id:
        return {}

    normalized: dict[str, list[RetrievalSourceMetadata]] = {}
    for paper_id, source_metadata_list in source_metadata_by_paper_id.items():
        items: list[RetrievalSourceMetadata] = []
        for source_metadata in source_metadata_list:
            if isinstance(source_metadata, RetrievalSourceMetadata):
                items.append(source_metadata)
                continue
            try:
                items.append(RetrievalSourceMetadata.model_validate(source_metadata))
            except ValueError:
                continue
        normalized[paper_id] = items
    return normalized


def _briefing_evidence_source(
    item_evidence_source: EvidenceSource,
    *,
    trend_overview: CandidatePoolTrendOverview,
) -> EvidenceSource:
    if not trend_overview.evidence_sources:
        return item_evidence_source
    if trend_overview.status == TrendAssessmentStatus.NOT_ASSESSED:
        return item_evidence_source
    trend_sources = set(trend_overview.evidence_sources)
    if trend_sources == {item_evidence_source}:
        return item_evidence_source
    return EvidenceSource.MIXED


def _evidence_boundary(
    *,
    evidence_source: EvidenceSource,
    trend_overview: CandidatePoolTrendOverview,
) -> BriefingEvidenceBoundary:
    evidence_sources = _ordered_sources(
        [evidence_source, *trend_overview.evidence_sources]
    )
    return BriefingEvidenceBoundary(
        evidence_sources=evidence_sources,
        unavailable_sources=[EvidenceSource.FULL_TEXT],
        full_text_used=False,
        notes=[
            "No PDF or full-text evidence was used for this briefing.",
            *(
                ["Candidate-pool trend analysis was not assessed."]
                if trend_overview.status == TrendAssessmentStatus.NOT_ASSESSED
                else []
            ),
        ],
    )


def _trend_metadata(overview: CandidatePoolTrendOverview) -> dict[str, Any]:
    return {
        "status": overview.status.value,
        "candidate_count": overview.candidate_count,
        "abstract_count": overview.abstract_count,
        "metadata_only_count": overview.metadata_only_count,
        "top_k_count": overview.top_k_count,
        "signal_count": len(overview.signals),
        "query_echo_signal_count": sum(
            1 for signal in overview.signals if signal.query_echo
        ),
        "representative_signal_count": sum(
            1
            for signal in overview.signals
            if signal.signal_type == TrendSignalType.HOTSPOT
            or (
                signal.signal_type == TrendSignalType.TOPIC
                and signal.strength != TrendSignalStrength.WEAK
            )
        ),
    }


def _normalized_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _normalized_term(text: str) -> str:
    tokens = _tokenize(text)
    return tokens[0] if len(tokens) == 1 else ""


def _normalized_phrase(text: str) -> str:
    return " ".join(token for token in _tokenize(text) if _valid_signal_term(token))


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _date_bucket(paper: PaperMetadata) -> str | None:
    published_date = paper.published_date or paper.updated_date
    if published_date is None:
        return None
    return f"{published_date.year:04d}-{published_date.month:02d}"


def _ordered_sources(sources: Sequence[EvidenceSource]) -> list[EvidenceSource]:
    order = [
        EvidenceSource.CANDIDATE_POOL,
        EvidenceSource.ABSTRACT,
        EvidenceSource.METADATA,
        EvidenceSource.RANKING,
        EvidenceSource.RETRIEVAL_METADATA,
        EvidenceSource.FULL_TEXT,
        EvidenceSource.MIXED,
    ]
    unique = set(sources)
    return [source for source in order if source in unique]


def _stronger_signal_type(
    first: TrendSignalType,
    second: TrendSignalType,
) -> TrendSignalType:
    order = {
        TrendSignalType.HOTSPOT: 3,
        TrendSignalType.TOPIC: 2,
        TrendSignalType.CATEGORY: 1,
        TrendSignalType.EVIDENCE_COVERAGE: 0,
    }
    return first if order[first] >= order[second] else second


def _stronger_strength(
    first: TrendSignalStrength,
    second: TrendSignalStrength,
) -> TrendSignalStrength:
    return first if _strength_rank(first) >= _strength_rank(second) else second


def _strength_rank(strength: TrendSignalStrength) -> int:
    return {
        TrendSignalStrength.WEAK: 0,
        TrendSignalStrength.MODERATE: 1,
        TrendSignalStrength.STRONG: 2,
    }[strength]


def _dedupe_text(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
