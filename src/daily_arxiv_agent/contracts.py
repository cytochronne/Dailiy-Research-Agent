"""Shared data contracts for Agent and Skill boundaries."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


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


class FeedbackValue(StrEnum):
    """Allowed paper-level feedback values."""

    LIKE = "like"
    DISLIKE = "dislike"


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


class SeedRecord(BaseModel):
    """One normalized seed contribution for personalization."""

    identity: str
    input_text: str
    input_type: str
    paper_id: str | None = None
    title: str | None = None
    abstract: str | None = None
    paper: PaperMetadata | None = None
    preference_text: str

    @model_validator(mode="after")
    def require_preference_text(self) -> "SeedRecord":
        if not self.identity.strip():
            raise ValueError("seed identity must not be blank")
        if not self.preference_text.strip():
            raise ValueError("seed preference_text must not be blank")
        return self


class SeedPreference(BaseModel):
    """User interest representation built from normalized seed papers."""

    profile_id: str = "default"
    seeds: list[SeedRecord] = Field(default_factory=list)
    preference_text: str
    vector: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def require_seed_signal(self) -> "SeedPreference":
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be blank")
        if not self.seeds:
            raise ValueError("seed preference must include at least one seed")
        if not self.preference_text.strip():
            raise ValueError("seed preference_text must not be blank")
        if not self.vector:
            raise ValueError("seed preference vector must not be empty")
        return self


class Recommendation(BaseModel):
    """Ranked recommendation for one paper."""

    paper: PaperMetadata
    rank: int = Field(ge=1)
    score: float
    rationale: str
    evidence_source: EvidenceSource = EvidenceSource.ABSTRACT
    previous_rank: int | None = Field(default=None, ge=1)
    previous_score: float | None = None
    score_delta: float | None = None
    rank_delta: int | None = None


class FeedbackEvent(BaseModel):
    """One like/dislike event tied to a recommendation context."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    profile_id: str = "default"
    recommendation_run_id: str | None = None
    paper_id: str
    value: FeedbackValue
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    paper: PaperMetadata | None = None
    note: str | None = None

    @model_validator(mode="after")
    def require_feedback_identity(self) -> "FeedbackEvent":
        if not self.event_id.strip():
            raise ValueError("feedback event_id must not be blank")
        if not self.profile_id.strip():
            raise ValueError("feedback profile_id must not be blank")
        if not self.paper_id.strip():
            raise ValueError("feedback paper_id must not be blank")
        return self

    @field_validator("created_at", mode="after")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        """Normalize feedback timestamps to UTC-aware datetimes."""

        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class PaperBriefingItem(BaseModel):
    """Structured briefing fields extracted for one ranked paper."""

    paper_id: str
    title: str
    rank: int = Field(ge=1)
    score: float
    summary: str
    contributions: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    relevance_rationale: str
    evidence_source: EvidenceSource
    provenance: Provenance
    arxiv_url: HttpUrl


class BriefingTableRow(BaseModel):
    """Compact row used by the daily briefing summary table."""

    rank: int = Field(ge=1)
    paper_id: str
    title: str
    score: float
    key_reason: str
    evidence_source: EvidenceSource
    arxiv_url: HttpUrl


class DailyBriefing(BaseModel):
    """Generated daily briefing from ranked papers and extracted metadata."""

    topic: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    executive_summary: str
    summary_table: list[BriefingTableRow] = Field(default_factory=list)
    highlighted_paper: PaperBriefingItem | None = None
    items: list[PaperBriefingItem] = Field(default_factory=list)
    evidence_source: EvidenceSource | None = None
    provenance: list[Provenance] = Field(default_factory=list)


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
