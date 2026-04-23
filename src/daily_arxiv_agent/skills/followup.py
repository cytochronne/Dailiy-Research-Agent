"""Follow-up query Skill for local-first filtering and optional retrieval."""

from __future__ import annotations

from datetime import date
import re
from typing import Sequence

from pydantic import BaseModel, Field, model_validator

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    RetrievalQuery,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.storage import SQLitePaperStore


class FollowupQuery(BaseModel):
    """Normalized follow-up filters against stored or newly retrieved papers."""

    topic: str | None = None
    category: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    max_results: int = Field(default=20, ge=1, le=100)
    fetch_if_empty: bool = True

    @model_validator(mode="after")
    def validate_date_range(self) -> "FollowupQuery":
        if self.start_date is not None and self.end_date is not None:
            if self.start_date > self.end_date:
                raise ValueError("start_date must be before or equal to end_date")
        return self


class FollowupSkill:
    """Answer follow-up topic/date questions by reusing local papers first."""

    def __init__(
        self,
        *,
        store: SQLitePaperStore,
        retrieval_skill: ArxivRetrievalSkill | None = None,
    ) -> None:
        self.store = store
        self.retrieval_skill = retrieval_skill

    def query(self, query: FollowupQuery) -> SkillResult[list[PaperMetadata]]:
        local_matches = _filter_papers(self.store.list_papers(), query)
        if local_matches:
            papers = local_matches[: query.max_results]
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.SUCCESS,
                data=papers,
                evidence_source=_combined_evidence(papers),
                provenance=[paper.provenance for paper in papers],
                message="Answered follow-up query from stored papers.",
                metadata={
                    "source": "local_store",
                    "local_hit": True,
                    "fetch_attempted": False,
                    "query": query.model_dump(mode="json"),
                    "matched_count": len(local_matches),
                },
            )

        if not query.fetch_if_empty:
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.EMPTY,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                message="No stored papers matched the follow-up query.",
                metadata={
                    "source": "local_store",
                    "local_hit": False,
                    "fetch_attempted": False,
                    "query": query.model_dump(mode="json"),
                    "matched_count": 0,
                },
            )

        if self.retrieval_skill is None:
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.FALLBACK,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code="followup_no_retrieval_skill",
                    message=(
                        "No stored papers matched the follow-up query and no "
                        "retrieval Skill is configured."
                    ),
                    retryable=False,
                ),
                message="No local follow-up results are available.",
                metadata={
                    "source": "local_store",
                    "local_hit": False,
                    "fetch_attempted": False,
                    "query": query.model_dump(mode="json"),
                    "matched_count": 0,
                },
            )

        retrieval_query = RetrievalQuery(
            topic=query.topic,
            category=query.category,
            start_date=query.start_date,
            end_date=query.end_date,
            max_results=query.max_results,
        )
        result = self.retrieval_skill.retrieve(retrieval_query, use_cache=True)
        papers = _filter_papers(result.data or [], query)[: query.max_results]
        status = (
            SkillStatus.EMPTY
            if result.status == SkillStatus.SUCCESS and not papers
            else result.status
        )
        metadata = dict(result.metadata)
        metadata.update(
            {
                "source": "retrieval_skill",
                "local_hit": False,
                "fetch_attempted": True,
                "query": query.model_dump(mode="json"),
                "matched_count": len(papers),
            }
        )
        return SkillResult[list[PaperMetadata]](
            status=status,
            data=papers,
            evidence_source=_combined_evidence(papers),
            provenance=[paper.provenance for paper in papers],
            error=result.error,
            message=(
                "No fetched papers matched the follow-up query."
                if status == SkillStatus.EMPTY
                else result.message or "Fetched papers for follow-up query."
            ),
            metadata=metadata,
        )


def _filter_papers(
    papers: Sequence[PaperMetadata],
    query: FollowupQuery,
) -> list[PaperMetadata]:
    matches = [paper for paper in papers if _matches(paper, query)]
    return sorted(
        matches,
        key=lambda paper: (
            -(paper.published_date.toordinal() if paper.published_date else 0),
            paper.title.lower(),
            paper.paper_id,
        ),
    )


def _matches(paper: PaperMetadata, query: FollowupQuery) -> bool:
    if query.category and query.category not in paper.categories:
        return False
    if (query.start_date or query.end_date) and paper.published_date is None:
        return False
    if query.start_date and paper.published_date and paper.published_date < query.start_date:
        return False
    if query.end_date and paper.published_date and paper.published_date > query.end_date:
        return False
    if query.topic and not _topic_matches(paper, query.topic):
        return False
    return True


def _topic_matches(paper: PaperMetadata, topic: str) -> bool:
    query_terms = _tokens(topic)
    if not query_terms:
        return True
    haystack = _tokens(f"{paper.title} {paper.abstract or ''} {' '.join(paper.categories)}")
    return all(term in haystack for term in query_terms)


def _tokens(text: str) -> set[str]:
    return {_normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())}


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _combined_evidence(papers: Sequence[PaperMetadata]) -> EvidenceSource:
    if any(paper.abstract for paper in papers):
        return EvidenceSource.ABSTRACT
    return EvidenceSource.METADATA
