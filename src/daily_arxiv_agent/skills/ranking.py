"""Deterministic topic ranking for retrieved arXiv papers."""

from __future__ import annotations

from collections import Counter
from datetime import date
import re
from typing import Sequence

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    FeedbackEvent,
    PaperMetadata,
    Recommendation,
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


class TopicRankingSkill:
    """Rank papers by explicit topic, seed-paper similarity, or both."""

    def __init__(
        self,
        *,
        vectorizer: DeterministicTextVectorizer | None = None,
        seed_similarity_weight: float = 10.0,
        feedback_weight: float = 6.0,
    ) -> None:
        self.vectorizer = vectorizer or DeterministicTextVectorizer()
        self.seed_similarity_weight = seed_similarity_weight
        self.feedback_weight = feedback_weight

    def rank(
        self,
        papers: Sequence[PaperMetadata],
        *,
        topic: str | None = None,
        seed_preference: SeedPreference | None = None,
        feedback_events: Sequence[FeedbackEvent] | None = None,
        top_k: int = 5,
    ) -> SkillResult[list[Recommendation]]:
        if (
            not (topic and topic.strip())
            and seed_preference is None
            and not feedback_events
        ):
            return SkillResult[list[Recommendation]](
                status=SkillStatus.ERROR,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code="ranking_input_missing",
                    message="Ranking requires a topic, a seed preference, or both.",
                    retryable=False,
                ),
                metadata={"topic": topic, "top_k": top_k},
            )

        if not papers:
            return SkillResult[list[Recommendation]](
                status=SkillStatus.EMPTY,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                message="No papers are available for ranking.",
                metadata={
                    "topic": topic,
                    "top_k": top_k,
                    "seed_profile_id": seed_preference.profile_id
                    if seed_preference
                    else None,
                    "feedback_count": len(feedback_events or []),
                },
            )

        query_terms = _tokenize(topic or "")
        scored = [
            _score_paper(
                paper,
                query_terms=query_terms,
                topic=topic or "",
                seed_preference=seed_preference,
                feedback_events=feedback_events or (),
                vectorizer=self.vectorizer,
                seed_similarity_weight=self.seed_similarity_weight,
                feedback_weight=self.feedback_weight,
            )
            for paper in papers
        ]
        scored.sort(
            key=lambda item: (
                -item.score,
                -_date_sort_value(item.paper.published_date),
                item.paper.title.lower(),
            )
        )

        recommendations = [
            Recommendation(
                paper=item.paper,
                rank=rank,
                score=item.score,
                rationale=item.rationale,
                evidence_source=item.evidence_source,
            )
            for rank, item in enumerate(scored[: max(top_k, 0)], start=1)
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
                "topic": topic,
                "top_k": top_k,
                "seed_profile_id": seed_preference.profile_id
                if seed_preference
                else None,
                "feedback_count": len(feedback_events or []),
                "ranking_mode": _ranking_mode(topic, seed_preference, feedback_events),
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
    ) -> None:
        self.paper = paper
        self.score = score
        self.rationale = rationale
        self.evidence_source = evidence_source


def _score_paper(
    paper: PaperMetadata,
    *,
    query_terms: list[str],
    topic: str,
    seed_preference: SeedPreference | None,
    feedback_events: Sequence[FeedbackEvent],
    vectorizer: DeterministicTextVectorizer,
    seed_similarity_weight: float,
    feedback_weight: float,
) -> _ScoredPaper:
    title_terms = Counter(_tokenize(paper.title))
    abstract_terms = Counter(_tokenize(paper.abstract or ""))
    category_text = " ".join(paper.categories).lower()
    exact_topic = topic.strip().lower()

    score = 0.0
    matched_terms: list[str] = []
    seed_similarity = 0.0
    for term in query_terms:
        term_score = title_terms[term] * 3.0 + abstract_terms[term] * 1.0
        if term in category_text:
            term_score += 0.5
        if term_score:
            matched_terms.append(term)
            score += term_score

    if exact_topic:
        title_lower = paper.title.lower()
        abstract_lower = (paper.abstract or "").lower()
        if exact_topic in title_lower:
            score += 5.0
        if exact_topic in abstract_lower:
            score += 2.0

    if seed_preference is not None:
        paper_vector = vectorizer.vectorize(build_paper_preference_text(paper))
        seed_similarity = cosine_similarity(seed_preference.vector, paper_vector)
        score += seed_similarity * seed_similarity_weight

    feedback_adjustment = feedback_adjustment_for_paper(
        paper,
        feedback_events,
        vectorizer=vectorizer,
        feedback_weight=feedback_weight,
    )
    score += feedback_adjustment.score_delta

    evidence_source = EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA
    rationale = _rationale(
        matched_terms,
        evidence_source,
        seed_similarity,
        feedback_adjustment.rationale,
    )
    return _ScoredPaper(
        paper=paper,
        score=round(score, 4),
        rationale=rationale,
        evidence_source=evidence_source,
    )


def _rationale(
    matched_terms: list[str],
    evidence_source: EvidenceSource,
    seed_similarity: float,
    feedback_rationale: str,
) -> str:
    parts: list[str] = []
    if matched_terms:
        terms = ", ".join(sorted(set(matched_terms)))
        parts.append(f"Matched explicit terms: {terms}.")
    elif seed_similarity <= 0:
        parts.append("No explicit keyword overlap; included to fill the requested top-k.")

    if seed_similarity > 0:
        parts.append(f"Seed-paper similarity: {seed_similarity:.3f}.")

    if feedback_rationale:
        parts.append(f"Feedback adjustment: {feedback_rationale}.")

    parts.append(f"Evidence: {evidence_source.value}.")
    return " ".join(parts)


def _tokenize(text: str) -> list[str]:
    return [_normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _date_sort_value(value: date | None) -> int:
    return value.toordinal() if value else 0


def _ranking_mode(
    topic: str | None,
    seed_preference: SeedPreference | None,
    feedback_events: Sequence[FeedbackEvent] | None,
) -> str:
    has_topic = bool(topic and topic.strip())
    has_seed = seed_preference is not None
    has_feedback = bool(feedback_events)
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
    return "topic"
