"""Shared data contracts for Agent and Skill boundaries."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
import math
from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


DataT = TypeVar("DataT")


class EvidenceSource(StrEnum):
    """Source material used for generated or ranked output."""

    METADATA = "metadata"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    RANKING = "ranking"
    RETRIEVAL_METADATA = "retrieval_metadata"
    CANDIDATE_POOL = "candidate_pool"
    MIXED = "mixed"


class EvidenceSupportStatus(StrEnum):
    """How strongly available evidence supports a briefing claim."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    NOT_ASSESSED = "not_assessed"
    INSUFFICIENT = "insufficient_evidence"


class TrendAssessmentStatus(StrEnum):
    """Candidate-pool trend assessment availability."""

    AVAILABLE = "available"
    LIMITED = "limited"
    INSUFFICIENT_DATA = "insufficient_candidate_data"
    NOT_ASSESSED = "not_assessed"


class TrendSignalType(StrEnum):
    """Kinds of candidate-pool signals used in a briefing overview."""

    TOPIC = "topic"
    HOTSPOT = "hotspot"
    CATEGORY = "category"
    EVIDENCE_COVERAGE = "evidence_coverage"


class TrendSignalStrength(StrEnum):
    """Conservative strength labels for candidate-pool signals."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


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


class ExplanationMode(StrEnum):
    """Supported selected-paper explanation modes."""

    METHOD = "method"
    EXPERIMENT = "experiment"
    LIMITATIONS = "limitations"


class SearchMode(StrEnum):
    """Search breadth requested for arXiv retrieval."""

    STRICT = "strict"
    BROAD = "broad"


class QueryPlannerMode(StrEnum):
    """Requested query-planning strategy."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    AUTO = "auto"


class RetrievalCacheStatus(StrEnum):
    """Completeness status for cached retrieval result sets."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    PLANNER_CACHE = "planner_cache"


class EmbeddingInputRole(StrEnum):
    """Trace-safe role labels for embedding inputs."""

    CANDIDATE = "candidate"
    SEED = "seed"
    FEEDBACK = "feedback"
    QUERY = "query"


class EmbeddingCacheScope(StrEnum):
    """SQLite embedding cache visibility scope."""

    GLOBAL = "global"
    PROFILE = "profile"


class EmbeddingIdentity(BaseModel):
    """Stable identity for one serialized embedding input."""

    provider: str
    model: str
    dimensions: int | None = Field(default=None, ge=1)
    input_version: str
    input_hash: str
    cache_scope: EmbeddingCacheScope = EmbeddingCacheScope.GLOBAL
    profile_id: str | None = None

    @field_validator("provider", "model", "input_version", "input_hash")
    @classmethod
    def require_non_blank_identity_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("embedding identity fields must not be blank")
        return normalized

    @field_validator("profile_id")
    @classmethod
    def normalize_optional_profile_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def validate_scope_shape(self) -> "EmbeddingIdentity":
        if self.cache_scope == EmbeddingCacheScope.PROFILE and self.profile_id is None:
            raise ValueError("profile-scoped embedding cache entries require profile_id")
        if self.cache_scope == EmbeddingCacheScope.GLOBAL and self.profile_id is not None:
            raise ValueError("global embedding cache entries must not include profile_id")
        return self


class SemanticSimilarityDetail(BaseModel):
    """Compact semantic relatedness detail for one source-target pair."""

    source_id: str
    target_id: str
    similarity: float = Field(ge=-1.0, le=1.0)
    source_role: EmbeddingInputRole = EmbeddingInputRole.SEED
    target_role: EmbeddingInputRole = EmbeddingInputRole.CANDIDATE
    source_title: str | None = None
    target_title: str | None = None
    score: float = 0.0

    @field_validator("source_id", "target_id")
    @classmethod
    def require_similarity_identity(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("semantic similarity identities must not be blank")
        return normalized


class EmbeddingVector(BaseModel):
    """Cached embedding vector plus lifecycle metadata."""

    identity: EmbeddingIdentity
    vector: list[float] = Field(min_length=1)
    input_role: EmbeddingInputRole = EmbeddingInputRole.CANDIDATE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: list[float]) -> list[float]:
        vector = [float(item) for item in value]
        if not all(math.isfinite(item) for item in vector):
            raise ValueError("embedding vector values must be finite")
        return vector

    @model_validator(mode="after")
    def validate_vector_dimensions(self) -> "EmbeddingVector":
        expected_dimensions = self.identity.dimensions
        if expected_dimensions is not None and len(self.vector) != expected_dimensions:
            raise ValueError("embedding vector dimensions do not match identity")
        if (
            self.input_role in {EmbeddingInputRole.SEED, EmbeddingInputRole.FEEDBACK}
            and self.identity.cache_scope != EmbeddingCacheScope.PROFILE
        ):
            raise ValueError("seed and feedback embedding cache entries must be profile-scoped")
        return self


class EmbeddingCacheMetadata(BaseModel):
    """Aggregate cache status safe for normal trace/UI output."""

    enabled: bool = True
    scope: EmbeddingCacheScope = EmbeddingCacheScope.GLOBAL
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    writes: int = Field(default=0, ge=0)
    disabled_requests: int = Field(default=0, ge=0)
    corrupt_entries: int = Field(default=0, ge=0)

    @property
    def requests(self) -> int:
        return self.hits + self.misses + self.disabled_requests


class EmbeddingProviderCacheMetadata(BaseModel):
    """Provider and aggregate cache metadata without raw inputs or cache keys."""

    provider: str
    provider_mode: str
    provider_label: str
    model: str
    dimensions: int | None = Field(default=None, ge=1)
    cache: EmbeddingCacheMetadata = Field(default_factory=EmbeddingCacheMetadata)

    @field_validator("provider", "provider_mode", "provider_label", "model")
    @classmethod
    def require_provider_metadata_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("embedding provider metadata fields must not be blank")
        return normalized


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
    search_mode: SearchMode = SearchMode.STRICT
    candidate_pool_size: int | None = Field(default=None, ge=1, le=500)
    page_size: int = Field(default=50, ge=1, le=100)
    max_requests: int = Field(default=4, ge=1, le=20)
    query_planner_mode: QueryPlannerMode = QueryPlannerMode.AUTO

    @model_validator(mode="after")
    def validate_date_range(self) -> "RetrievalQuery":
        if self.start_date is not None and self.end_date is not None:
            if self.start_date > self.end_date:
                raise ValueError("start_date must be before or equal to end_date")
        return self

    @property
    def effective_candidate_pool_size(self) -> int:
        """Candidate pool size requested for ranking after retrieval."""

        return self.candidate_pool_size or self.max_results


class RetrievalBudget(BaseModel):
    """Bounded retrieval request budget for a planned search."""

    candidate_pool_size: int = Field(default=100, ge=1, le=500)
    page_size: int = Field(default=50, ge=1, le=100)
    max_requests: int = Field(default=4, ge=1, le=20)


class QueryPlanVariant(BaseModel):
    """One arXiv query variant generated by query planning."""

    label: str
    search_query: str
    sort_by: str = "relevance"
    sort_order: str = "descending"

    @field_validator("label", "search_query", "sort_by", "sort_order")
    @classmethod
    def require_non_blank_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query plan variant fields must not be blank")
        return normalized


class QueryPlannerProvenance(BaseModel):
    """How a query plan was produced."""

    requested_mode: QueryPlannerMode
    source: str
    fallback_reason: str | None = None
    model: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source")
    @classmethod
    def require_source(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("planner source must not be blank")
        return normalized


class QueryPlan(BaseModel):
    """Bounded, inspectable set of query variants for retrieval."""

    search_mode: SearchMode
    planner: QueryPlannerProvenance
    variants: list[QueryPlanVariant] = Field(min_length=1)
    required_terms: list[str] = Field(default_factory=list)
    optional_terms: list[str] = Field(default_factory=list)
    phrases: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    rationale: str | None = None

    @property
    def variant_count(self) -> int:
        return len(self.variants)


class RetrievalSourceMetadata(BaseModel):
    """Run-scoped source details for one paper in one retrieval result set."""

    variant_label: str
    sort_by: str
    variant_index: int = Field(ge=0)
    position: int = Field(ge=0)
    first_seen_order: int = Field(ge=0)
    query: str | None = None

    @field_validator("variant_label", "sort_by")
    @classmethod
    def require_source_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("retrieval source metadata fields must not be blank")
        return normalized


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


class RetrievalResultSet(BaseModel):
    """Cached papers and run-scoped metadata for one effective retrieval plan."""

    query: RetrievalQuery
    papers: list[PaperMetadata] = Field(default_factory=list)
    cache_status: RetrievalCacheStatus = RetrievalCacheStatus.COMPLETE
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_metadata_by_paper_id: dict[str, list[RetrievalSourceMetadata]] = Field(
        default_factory=dict
    )
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_query_key: str


class RankingScoreBreakdown(BaseModel):
    """Explainable score components for one ranked recommendation."""

    lexical: float = 0.0
    phrase: float = 0.0
    query_source: float = 0.0
    recency: float = 0.0
    category: float = 0.0
    seed_similarity: float = 0.0
    semantic_seed: float = 0.0
    feedback: float = 0.0
    total: float = 0.0
    evidence_score: float = 0.0
    fallback: bool = False
    matched_terms: list[str] = Field(default_factory=list)
    matched_phrases: list[str] = Field(default_factory=list)
    semantic_similarities: list[SemanticSimilarityDetail] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


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
    score_breakdown: RankingScoreBreakdown | None = None


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


class FieldEvidenceStatus(BaseModel):
    """Evidence state for one briefing claim or abstention."""

    status: EvidenceSupportStatus = EvidenceSupportStatus.NOT_ASSESSED
    sources: list[EvidenceSource] = Field(default_factory=list)
    note: str | None = None
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_support_or_abstention(self) -> "FieldEvidenceStatus":
        if self.status in {
            EvidenceSupportStatus.SUPPORTED,
            EvidenceSupportStatus.PARTIAL,
        } and not self.sources:
            raise ValueError("supported briefing claims must include evidence sources")
        if self.status in {
            EvidenceSupportStatus.UNAVAILABLE,
            EvidenceSupportStatus.INSUFFICIENT,
        } and not (self.note or self.abstention_reason):
            raise ValueError("abstentions must include a note or abstention reason")
        return self


class EvidenceBoundClaim(BaseModel):
    """A paper-level claim with its support or abstention state."""

    claim: str | None = None
    evidence: FieldEvidenceStatus = Field(default_factory=FieldEvidenceStatus)


class TrendSignal(BaseModel):
    """One bounded trend or hotspot signal from the candidate pool."""

    label: str
    signal_type: TrendSignalType = TrendSignalType.TOPIC
    strength: TrendSignalStrength = TrendSignalStrength.WEAK
    support_count: int = Field(default=0, ge=0)
    candidate_count: int | None = Field(default=None, ge=0)
    top_k_count: int | None = Field(default=None, ge=0)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    summary: str | None = None
    limitations: list[str] = Field(default_factory=list)
    query_echo: bool = False


class CandidatePoolTrendOverview(BaseModel):
    """Trend and hotspot overview derived from broader candidate metadata."""

    status: TrendAssessmentStatus = TrendAssessmentStatus.NOT_ASSESSED
    summary: str | None = None
    candidate_count: int = Field(default=0, ge=0)
    abstract_count: int = Field(default=0, ge=0)
    metadata_only_count: int = Field(default=0, ge=0)
    top_k_count: int = Field(default=0, ge=0)
    signals: list[TrendSignal] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)


class TopKComparisonNote(BaseModel):
    """Evidence-bounded comparison across one or more Top-K papers."""

    dimension: str
    note: str
    paper_ids: list[str] = Field(default_factory=list)
    ranks: list[int] = Field(default_factory=list)
    evidence: FieldEvidenceStatus = Field(default_factory=FieldEvidenceStatus)

    @field_validator("ranks")
    @classmethod
    def validate_ranks(cls, value: list[int]) -> list[int]:
        if any(rank < 1 for rank in value):
            raise ValueError("comparison ranks must be positive")
        return value


class ReadingPriority(BaseModel):
    """Goal-aware reading recommendation for one ranked paper."""

    priority: int = Field(ge=1)
    reading_intent: str
    paper_id: str
    rank: int = Field(ge=1)
    reason: str
    evidence: FieldEvidenceStatus = Field(default_factory=FieldEvidenceStatus)


class BriefingEvidenceBoundary(BaseModel):
    """Evidence sources used, unavailable sources, and explicit abstentions."""

    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    unavailable_sources: list[EvidenceSource] = Field(default_factory=list)
    full_text_used: bool = False
    notes: list[str] = Field(default_factory=list)
    abstentions: list[EvidenceBoundClaim] = Field(default_factory=list)


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
    problem: EvidenceBoundClaim | None = None
    approach: EvidenceBoundClaim | None = None
    reading_guide: EvidenceBoundClaim | None = None
    contribution_claims: list[EvidenceBoundClaim] = Field(default_factory=list)
    method_claims: list[EvidenceBoundClaim] = Field(default_factory=list)
    relevance_evidence: FieldEvidenceStatus | None = None


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
    trend_overview: CandidatePoolTrendOverview = Field(
        default_factory=CandidatePoolTrendOverview
    )
    top_k_comparisons: list[TopKComparisonNote] = Field(default_factory=list)
    reading_priorities: list[ReadingPriority] = Field(default_factory=list)
    evidence_boundary: BriefingEvidenceBoundary = Field(
        default_factory=BriefingEvidenceBoundary
    )


class MethodExplanation(BaseModel):
    """Method or framework explanation fields."""

    problem: str
    method_overview: str
    core_workflow: list[str] = Field(default_factory=list)
    inputs_outputs: list[str] = Field(default_factory=list)
    innovation: str


class ExperimentExplanation(BaseModel):
    """Experiment and results explanation fields."""

    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    experimental_setup: str
    conclusions: list[str] = Field(default_factory=list)


class LimitationsExplanation(BaseModel):
    """Limitations explanation fields."""

    stated_limitations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_validation: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class PaperDeepExplanation(BaseModel):
    """Mode-specific explanation for one selected paper."""

    paper_id: str
    title: str
    mode: ExplanationMode
    summary: str
    evidence_source: EvidenceSource
    evidence_note: str
    method: MethodExplanation | None = None
    experiment: ExperimentExplanation | None = None
    limitations: LimitationsExplanation | None = None
    provenance: Provenance
    arxiv_url: HttpUrl

    @model_validator(mode="after")
    def validate_mode_shape(self) -> "PaperDeepExplanation":
        expected = {
            ExplanationMode.METHOD: "method",
            ExplanationMode.EXPERIMENT: "experiment",
            ExplanationMode.LIMITATIONS: "limitations",
        }
        active_name = expected[self.mode]
        active_value = getattr(self, active_name)
        if active_value is None:
            raise ValueError(f"{active_name} explanation is required for mode {self.mode!r}")
        for field_name in expected.values():
            if field_name == active_name:
                continue
            if getattr(self, field_name) is not None:
                raise ValueError(
                    f"{field_name} must be omitted for mode {self.mode!r}"
                )
        return self


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
