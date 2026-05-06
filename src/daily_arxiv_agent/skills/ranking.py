"""Deterministic topic ranking for retrieved arXiv papers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import date
import re
from typing import Sequence

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    FeedbackEvent,
    PaperMetadata,
    QueryPlan,
    RankingScoreBreakdown,
    Recommendation,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SeedPreference,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.skills.feedback import feedback_adjustment_for_paper
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
    build_paper_preference_text,
    cosine_similarity,
)


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

SEMANTIC_SEED_RANKING_MODE = "semantic_seed"
SEMANTIC_TOPIC_SEED_RANKING_MODE = "semantic_topic_seed"


class TopicRankingSkill:
    """Rank papers by explainable topic, retrieval, seed, and feedback signals."""

    def __init__(
        self,
        *,
        vectorizer: DeterministicTextVectorizer | None = None,
        seed_similarity_weight: float = 10.0,
        feedback_weight: float = 6.0,
        query_source_weight: float = 1.0,
        recency_weight: float = 1.0,
        category_weight: float = 1.5,
        minimum_evidence_score: float = 0.5,
    ) -> None:
        self.vectorizer = vectorizer or DeterministicTextVectorizer()
        self.seed_similarity_weight = seed_similarity_weight
        self.feedback_weight = feedback_weight
        self.query_source_weight = query_source_weight
        self.recency_weight = recency_weight
        self.category_weight = category_weight
        self.minimum_evidence_score = minimum_evidence_score

    def rank(
        self,
        papers: Sequence[PaperMetadata],
        *,
        topic: str | None = None,
        seed_preference: SeedPreference | None = None,
        feedback_events: Sequence[FeedbackEvent] | None = None,
        top_k: int = 5,
        query_plan: QueryPlan | None = None,
        retrieval_query: RetrievalQuery | None = None,
        retrieval_source_metadata_by_paper_id: Mapping[
            str, Sequence[RetrievalSourceMetadata | Mapping[str, object]]
        ]
        | None = None,
    ) -> SkillResult[list[Recommendation]]:
        effective_topic = _effective_topic(topic, retrieval_query)
        query_terms = _query_terms(effective_topic, query_plan)
        query_phrases = _query_phrases(effective_topic, query_plan)
        category_recency_mode = _category_recency_mode(
            effective_topic,
            seed_preference,
            feedback_events,
            retrieval_query,
            query_plan,
        )
        if (
            not query_terms
            and not query_phrases
            and seed_preference is None
            and not feedback_events
            and not category_recency_mode
        ):
            return SkillResult[list[Recommendation]](
                status=SkillStatus.ERROR,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code="ranking_input_missing",
                    message=(
                        "Ranking requires a topic, a seed preference, feedback, "
                        "or category/date retrieval context."
                    ),
                    retryable=False,
                ),
                metadata={"topic": effective_topic, "top_k": top_k},
            )

        ranking_mode = _ranking_mode(
            effective_topic,
            seed_preference,
            feedback_events,
            category_recency_mode=category_recency_mode,
            query_plan=query_plan,
        )
        if not papers:
            return SkillResult[list[Recommendation]](
                status=SkillStatus.EMPTY,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                message="No papers are available for ranking.",
                metadata={
                    "topic": effective_topic,
                    "top_k": top_k,
                    "seed_profile_id": seed_preference.profile_id
                    if seed_preference
                    else None,
                    "feedback_count": len(feedback_events or []),
                    "ranking_mode": ranking_mode,
                },
            )

        newest_published_date = _newest_published_date(papers)
        source_metadata = _normalize_source_metadata_by_paper_id(
            retrieval_source_metadata_by_paper_id
        )
        scored = [
            _score_paper(
                paper,
                query_terms=query_terms,
                query_phrases=query_phrases,
                category_recency_mode=category_recency_mode,
                retrieval_query=retrieval_query,
                retrieval_source_metadata=source_metadata.get(paper.paper_id, ()),
                newest_published_date=newest_published_date,
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                vectorizer=self.vectorizer,
                seed_similarity_weight=self.seed_similarity_weight,
                feedback_weight=self.feedback_weight,
                query_source_weight=self.query_source_weight,
                recency_weight=self.recency_weight,
                category_weight=self.category_weight,
                minimum_evidence_score=self.minimum_evidence_score,
            )
            for paper in papers
        ]
        scored.sort(
            key=lambda item: (
                -item.score,
                -item.breakdown.evidence_score,
                -_date_sort_value(item.paper.published_date),
                item.paper.title.lower(),
            )
        )

        selected = _select_scored(scored, max(top_k, 0))
        recommendations = [
            _recommendation_from_scored(
                item,
                rank=rank,
                fallback=not item.qualifies,
            )
            for rank, item in enumerate(selected, start=1)
        ]
        result_evidence = (
            EvidenceSource.ABSTRACT
            if any(
                item.evidence_source == EvidenceSource.ABSTRACT
                for item in recommendations
            )
            else EvidenceSource.METADATA
        )
        return SkillResult[list[Recommendation]](
            status=SkillStatus.SUCCESS,
            data=recommendations,
            evidence_source=result_evidence,
            provenance=[item.paper.provenance for item in recommendations],
            metadata={
                "topic": effective_topic,
                "top_k": top_k,
                "seed_profile_id": seed_preference.profile_id
                if seed_preference
                else None,
                "feedback_count": len(feedback_events or []),
                "ranking_mode": ranking_mode,
                "score_signals": _score_signal_names(scored),
                "qualifying_count": sum(1 for item in scored if item.qualifies),
                "fallback_count": sum(
                    1
                    for recommendation in recommendations
                    if recommendation.score_breakdown
                    and recommendation.score_breakdown.fallback
                ),
                "minimum_evidence_score": self.minimum_evidence_score,
            },
        )


class _ScoredPaper:
    def __init__(
        self,
        *,
        paper: PaperMetadata,
        score: float,
        rationale: str,
        evidence_source: EvidenceSource,
        breakdown: RankingScoreBreakdown,
        qualifies: bool,
    ) -> None:
        self.paper = paper
        self.score = score
        self.rationale = rationale
        self.evidence_source = evidence_source
        self.breakdown = breakdown
        self.qualifies = qualifies


def _score_paper(
    paper: PaperMetadata,
    *,
    query_terms: list[str],
    query_phrases: list[str],
    category_recency_mode: bool,
    retrieval_query: RetrievalQuery | None,
    retrieval_source_metadata: Sequence[RetrievalSourceMetadata],
    newest_published_date: date | None,
    seed_preference: SeedPreference | None,
    feedback_events: Sequence[FeedbackEvent],
    vectorizer: DeterministicTextVectorizer,
    seed_similarity_weight: float,
    feedback_weight: float,
    query_source_weight: float,
    recency_weight: float,
    category_weight: float,
    minimum_evidence_score: float,
) -> _ScoredPaper:
    lexical_score, matched_terms = _lexical_signal(paper, query_terms)
    phrase_score, matched_phrases = _phrase_signal(paper, query_phrases)
    query_source_score = (
        _query_source_signal(retrieval_source_metadata) * query_source_weight
    )
    recency_score = (
        _recency_signal(paper.published_date, newest_published_date) * recency_weight
    )
    category_score = _category_signal(paper, retrieval_query) * category_weight

    seed_similarity = 0.0
    seed_score = 0.0
    if seed_preference is not None:
        paper_vector = vectorizer.vectorize(build_paper_preference_text(paper))
        seed_similarity = cosine_similarity(seed_preference.vector, paper_vector)
        seed_score = seed_similarity * seed_similarity_weight

    feedback_adjustment = feedback_adjustment_for_paper(
        paper,
        feedback_events,
        vectorizer=vectorizer,
        feedback_weight=feedback_weight,
    )
    feedback_score = feedback_adjustment.score_delta

    score = (
        lexical_score
        + phrase_score
        + query_source_score
        + recency_score
        + category_score
        + seed_score
        + feedback_score
    )
    evidence_score = (
        lexical_score
        + phrase_score
        + max(seed_score, 0.0)
        + (max(feedback_score, 0.0) if feedback_adjustment.matched_event_count else 0.0)
    )
    if category_recency_mode:
        evidence_score = category_score + recency_score + query_source_score

    breakdown = RankingScoreBreakdown(
        lexical=round(lexical_score, 4),
        phrase=round(phrase_score, 4),
        query_source=round(query_source_score, 4),
        recency=round(recency_score, 4),
        category=round(category_score, 4),
        seed_similarity=round(seed_score, 4),
        feedback=round(feedback_score, 4),
        total=round(score, 4),
        evidence_score=round(evidence_score, 4),
        matched_terms=sorted(set(matched_terms)),
        matched_phrases=sorted(set(matched_phrases)),
        signals=_signals_from_scores(
            lexical=lexical_score,
            phrase=phrase_score,
            query_source=query_source_score,
            recency=recency_score,
            category=category_score,
            seed_similarity=seed_score,
            feedback=feedback_score,
        ),
    )
    qualifies = (
        evidence_score > 0
        if category_recency_mode
        else evidence_score >= minimum_evidence_score
    )

    evidence_source = EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA
    rationale = _rationale(
        matched_terms,
        matched_phrases,
        evidence_source,
        seed_similarity,
        feedback_adjustment.rationale,
        breakdown,
        category_recency_mode=category_recency_mode,
    )
    return _ScoredPaper(
        paper=paper,
        score=round(score, 4),
        rationale=rationale,
        evidence_source=evidence_source,
        breakdown=breakdown,
        qualifies=qualifies,
    )


def _rationale(
    matched_terms: list[str],
    matched_phrases: list[str],
    evidence_source: EvidenceSource,
    seed_similarity: float,
    feedback_rationale: str,
    breakdown: RankingScoreBreakdown,
    *,
    category_recency_mode: bool,
) -> str:
    parts: list[str] = []
    if category_recency_mode:
        parts.append("Ranked by category/date retrieval signals.")

    if matched_terms:
        terms = ", ".join(sorted(set(matched_terms)))
        parts.append(f"Matched explicit terms: {terms}.")

    if matched_phrases:
        phrases = ", ".join(sorted(set(matched_phrases)))
        parts.append(f"Matched phrases: {phrases}.")

    if breakdown.query_source > 0:
        parts.append(f"Query-source/order signal: {breakdown.query_source:.3f}.")

    if breakdown.category > 0:
        parts.append("Category fit matched the retrieval filter.")

    if breakdown.recency > 0:
        parts.append(f"Recency signal: {breakdown.recency:.3f}.")

    if seed_similarity > 0:
        parts.append(f"Seed-paper similarity: {seed_similarity:.3f}.")

    if feedback_rationale:
        parts.append(f"Feedback adjustment: {feedback_rationale}.")

    parts.append(f"Evidence: {evidence_source.value}.")
    return " ".join(parts)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        normalized = _normalize_token(token)
        if not normalized or normalized in STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _date_sort_value(value: date | None) -> int:
    return value.toordinal() if value else 0


def _effective_topic(topic: str | None, retrieval_query: RetrievalQuery | None) -> str | None:
    if topic is not None:
        return topic
    if retrieval_query is not None:
        return retrieval_query.topic
    return None


def _query_terms(topic: str | None, query_plan: QueryPlan | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in [topic or "", *(_plan_terms(query_plan) if query_plan else [])]:
        for term in _tokenize(raw):
            if term in seen:
                continue
            terms.append(term)
            seen.add(term)
    return terms


def _plan_terms(query_plan: QueryPlan | None) -> list[str]:
    if query_plan is None:
        return []
    return [*query_plan.required_terms, *query_plan.optional_terms]


def _query_phrases(topic: str | None, query_plan: QueryPlan | None) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    topic_phrase = _normalized_phrase(topic or "")
    if len(topic_phrase.split()) > 1:
        phrases.append(topic_phrase)
        seen.add(topic_phrase)
    if query_plan is not None:
        for phrase in query_plan.phrases:
            normalized = _normalized_phrase(phrase)
            if len(normalized.split()) <= 1 or normalized in seen:
                continue
            phrases.append(normalized)
            seen.add(normalized)
    return phrases


def _lexical_signal(
    paper: PaperMetadata,
    query_terms: Sequence[str],
) -> tuple[float, list[str]]:
    if not query_terms:
        return 0.0, []

    title_terms = Counter(_tokenize(paper.title))
    abstract_terms = Counter(_tokenize(paper.abstract or ""))
    score = 0.0
    matched_terms: list[str] = []
    for term in query_terms:
        term_score = title_terms[term] * 3.0 + abstract_terms[term] * 1.0
        if term_score <= 0:
            continue
        matched_terms.append(term)
        score += min(term_score, 6.0)
    return score, matched_terms


def _phrase_signal(
    paper: PaperMetadata,
    phrases: Sequence[str],
) -> tuple[float, list[str]]:
    if not phrases:
        return 0.0, []

    title_text = _normalized_text(paper.title)
    abstract_text = _normalized_text(paper.abstract or "")
    score = 0.0
    matched_phrases: list[str] = []
    for phrase in phrases:
        normalized = _normalized_phrase(phrase)
        if not normalized:
            continue
        phrase_score = 0.0
        if normalized in title_text:
            phrase_score += 6.0
        if normalized in abstract_text:
            phrase_score += 3.0
        if phrase_score <= 0:
            continue
        matched_phrases.append(normalized)
        score += phrase_score
    return score, matched_phrases


def _query_source_signal(
    source_metadata: Sequence[RetrievalSourceMetadata],
) -> float:
    best = 0.0
    for metadata in source_metadata:
        sort_boost = 1.2 if metadata.sort_by == "relevance" else 0.4
        position_boost = 1.0 / (metadata.position + 1)
        variant_penalty = max(0.5, 1.0 - metadata.variant_index * 0.15)
        first_seen_penalty = max(0.5, 1.0 - metadata.first_seen_order * 0.02)
        best = max(best, sort_boost * position_boost * variant_penalty * first_seen_penalty)
    return best


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
            except Exception:
                continue
        if items:
            normalized[paper_id] = items
    return normalized


def _recency_signal(
    published_date: date | None,
    newest_published_date: date | None,
) -> float:
    if published_date is None or newest_published_date is None:
        return 0.0
    age_days = max((newest_published_date - published_date).days, 0)
    if age_days >= 365:
        return 0.0
    return 1.0 - (age_days / 365.0)


def _category_signal(
    paper: PaperMetadata,
    retrieval_query: RetrievalQuery | None,
) -> float:
    if retrieval_query is None or not retrieval_query.category:
        return 0.0
    category = retrieval_query.category.lower()
    paper_categories = {paper_category.lower() for paper_category in paper.categories}
    return 1.0 if category in paper_categories else 0.0


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _normalized_phrase(text: str) -> str:
    return _normalized_text(text)


def _newest_published_date(papers: Sequence[PaperMetadata]) -> date | None:
    dates = [paper.published_date for paper in papers if paper.published_date is not None]
    return max(dates) if dates else None


def _category_recency_mode(
    topic: str | None,
    seed_preference: SeedPreference | None,
    feedback_events: Sequence[FeedbackEvent] | None,
    retrieval_query: RetrievalQuery | None,
    query_plan: QueryPlan | None,
) -> bool:
    if topic and topic.strip():
        return False
    if seed_preference is not None or feedback_events:
        return False
    if query_plan is not None and _plan_terms(query_plan):
        return False
    if retrieval_query is None:
        return False
    return bool(
        retrieval_query.category
        or retrieval_query.start_date
        or retrieval_query.end_date
    )


def _select_scored(scored: Sequence[_ScoredPaper], limit: int) -> list[_ScoredPaper]:
    if limit <= 0:
        return []

    selected: list[_ScoredPaper] = []
    selected_ids: set[str] = set()
    for item in scored:
        if not item.qualifies:
            continue
        selected.append(item)
        selected_ids.add(item.paper.paper_id)
        if len(selected) >= limit:
            return selected

    for item in scored:
        if item.paper.paper_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.paper.paper_id)
        if len(selected) >= limit:
            break
    return selected


def _recommendation_from_scored(
    item: _ScoredPaper,
    *,
    rank: int,
    fallback: bool,
) -> Recommendation:
    breakdown = item.breakdown.model_copy(update={"fallback": fallback})
    rationale = item.rationale
    if fallback:
        rationale = _fallback_rationale(rationale)
    return Recommendation(
        paper=item.paper,
        rank=rank,
        score=item.score,
        rationale=rationale,
        evidence_source=item.evidence_source,
        score_breakdown=breakdown,
    )


def _fallback_rationale(rationale: str) -> str:
    return (
        "Fallback inclusion: no direct topic, seed, or positive feedback evidence "
        f"met the normal ranking threshold. {rationale}"
    )


def _signals_from_scores(**scores: float) -> list[str]:
    return [name for name, value in scores.items() if abs(value) > 0.0001]


def _score_signal_names(scored: Sequence[_ScoredPaper]) -> list[str]:
    signals: set[str] = set()
    for item in scored:
        signals.update(item.breakdown.signals)
    return sorted(signals)


def _ranking_mode(
    topic: str | None,
    seed_preference: SeedPreference | None,
    feedback_events: Sequence[FeedbackEvent] | None,
    *,
    category_recency_mode: bool,
    query_plan: QueryPlan | None,
) -> str:
    has_topic = bool(topic and topic.strip())
    has_plan_terms = bool(_plan_terms(query_plan))
    has_seed = seed_preference is not None
    has_feedback = bool(feedback_events)
    if category_recency_mode:
        return "category_recency"
    if has_topic and has_seed and has_feedback:
        return "hybrid_topic_seed_feedback"
    if has_topic and has_seed:
        return "hybrid_topic_seed"
    if has_topic and has_feedback:
        return "hybrid_topic_feedback"
    if has_seed and has_feedback:
        return "seed_feedback"
    if has_feedback:
        return "feedback"
    if has_seed:
        return "seed"
    if has_plan_terms:
        return "query_plan"
    return "topic"
