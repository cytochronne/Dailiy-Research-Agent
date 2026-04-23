"""Deterministic topic ranking for retrieved arXiv papers."""

from __future__ import annotations

from collections import Counter
from datetime import date
import re
from typing import Sequence

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Recommendation,
    SkillResult,
    SkillStatus,
)


class TopicRankingSkill:
    """Rank papers by explicit topic and keyword overlap."""

    def rank(
        self,
        papers: Sequence[PaperMetadata],
        *,
        topic: str,
        top_k: int = 5,
    ) -> SkillResult[list[Recommendation]]:
        if not papers:
            return SkillResult[list[Recommendation]](
                status=SkillStatus.EMPTY,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                message="No papers are available for ranking.",
                metadata={"topic": topic, "top_k": top_k},
            )

        query_terms = _tokenize(topic)
        scored = [
            _score_paper(paper, query_terms=query_terms, topic=topic)
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
            metadata={"topic": topic, "top_k": top_k},
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
) -> _ScoredPaper:
    title_terms = Counter(_tokenize(paper.title))
    abstract_terms = Counter(_tokenize(paper.abstract or ""))
    category_text = " ".join(paper.categories).lower()
    exact_topic = topic.strip().lower()

    score = 0.0
    matched_terms: list[str] = []
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

    evidence_source = EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA
    rationale = _rationale(matched_terms, evidence_source)
    return _ScoredPaper(
        paper=paper,
        score=round(score, 4),
        rationale=rationale,
        evidence_source=evidence_source,
    )


def _rationale(matched_terms: list[str], evidence_source: EvidenceSource) -> str:
    if matched_terms:
        terms = ", ".join(sorted(set(matched_terms)))
        return f"Matched explicit terms: {terms}. Evidence: {evidence_source.value}."
    return (
        "No explicit keyword overlap; included to fill the requested top-k. "
        f"Evidence: {evidence_source.value}."
    )


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
