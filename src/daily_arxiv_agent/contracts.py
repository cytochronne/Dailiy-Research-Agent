"""Shared data contracts for Agent and Skill boundaries."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, HttpUrl, model_validator


DataT = TypeVar("DataT")


class EvidenceSource(StrEnum):
    """Source material used for generated or ranked output."""

    METADATA = "metadata"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"


class SkillStatus(StrEnum):
    """Execution status for a Skill result."""

    SUCCESS = "success"
    EMPTY = "empty"
    FALLBACK = "fallback"
    ERROR = "error"


class Provenance(BaseModel):
    """Where a paper or generated output came from."""

    source: str
    source_url: HttpUrl | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query: str | None = None


class RetrievalQuery(BaseModel):
    """Normalized arXiv retrieval inputs."""

    topic: str | None = None
    category: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    start_index: int = Field(default=0, ge=0)
    max_results: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_date_range(self) -> "RetrievalQuery":
        if self.start_date is not None and self.end_date is not None:
            if self.start_date > self.end_date:
                raise ValueError("start_date must be before or equal to end_date")
        return self


class SkillError(BaseModel):
    """Structured error information that can be shown or logged."""

    code: str
    message: str
    retryable: bool = False


class PaperMetadata(BaseModel):
    """Normalized metadata for an arXiv paper."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    categories: list[str] = Field(default_factory=list)
    published_date: date | None = None
    updated_date: date | None = None
    arxiv_url: HttpUrl
    pdf_url: HttpUrl | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def require_meaningful_identity(self) -> "PaperMetadata":
        if not self.paper_id.strip():
            raise ValueError("paper_id must not be blank")
        if not self.title.strip():
            raise ValueError("title must not be blank")
        return self


class Recommendation(BaseModel):
    """Ranked recommendation for one paper."""

    paper: PaperMetadata
    rank: int = Field(ge=1)
    score: float
    rationale: str
    evidence_source: EvidenceSource = EvidenceSource.ABSTRACT


class SkillResult(BaseModel, Generic[DataT]):
    """Standard result envelope returned by every Skill."""

    status: SkillStatus
    data: DataT | None = None
    evidence_source: EvidenceSource | None = None
    provenance: list[Provenance] = Field(default_factory=list)
    error: SkillError | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_shape(self) -> "SkillResult[DataT]":
        if self.status == SkillStatus.SUCCESS and self.error is not None:
            raise ValueError("successful SkillResult must not include an error")
        if self.status in {SkillStatus.ERROR, SkillStatus.FALLBACK} and self.error is None:
            raise ValueError("error or fallback SkillResult must include an error")
        return self
