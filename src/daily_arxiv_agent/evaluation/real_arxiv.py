"""Frozen real-arXiv evaluation utilities.

The default path is fully offline: read a frozen candidate snapshot and human
labels, rank the candidates with small baselines, and compute overlap metrics.
Live arXiv fetching is intentionally explicit so tests and reports stay stable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    QueryPlan,
    QueryPlanVariant,
    QueryPlannerMode,
    QueryPlannerProvenance,
    Recommendation,
    RetrievalQuery,
    SearchMode,
    SeedPreference,
    SeedRecord,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
from daily_arxiv_agent.embeddings.provider import create_embedding_provider
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.seed_parsing import DeterministicTextVectorizer
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


DEFAULT_REAL_ARXIV_CANDIDATES_PATH = Path(
    "data/evaluation/real_arxiv_candidates.jsonl"
)
DEFAULT_REAL_ARXIV_LABELS_PATH = Path("data/evaluation/real_arxiv_labels.jsonl")


def _dedupe_terms(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


class RealArxivTopic(BaseModel):
    """One fixed evaluation topic."""

    topic_id: str
    topic: str
    category: str
    query: str
    required_terms: list[str]
    optional_terms: list[str] = Field(default_factory=list)

    @field_validator("topic_id", "topic", "category", "query")
    @classmethod
    def require_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("topic fields must not be blank")
        return normalized

    @field_validator("required_terms")
    @classmethod
    def require_terms(cls, value: list[str]) -> list[str]:
        terms = _dedupe_terms(value)
        if not terms:
            raise ValueError("required_terms must not be empty")
        return terms

    @field_validator("optional_terms")
    @classmethod
    def normalize_optional_terms(cls, value: list[str]) -> list[str]:
        return _dedupe_terms(value)


DEFAULT_REAL_ARXIV_TOPICS: tuple[RealArxivTopic, ...] = (
    RealArxivTopic(
        topic_id="llm_tool_agents",
        topic="LLM agents tool use",
        category="cs.AI",
        query='cat:cs.AI AND (all:LLM OR all:"large language model") AND (all:agent OR all:"tool use")',
        required_terms=["llm", "agent", "tool"],
        optional_terms=["large language model", "tool use", "function calling"],
    ),
    RealArxivTopic(
        topic_id="retrieval_augmented_generation",
        topic="retrieval augmented generation",
        category="cs.CL",
        query='cat:cs.CL AND (all:"retrieval augmented generation" OR all:RAG OR all:retrieval)',
        required_terms=["retrieval", "augmented", "generation"],
        optional_terms=["rag", "grounded generation", "knowledge-intensive"],
    ),
    RealArxivTopic(
        topic_id="vision_language_robotics",
        topic="vision-language robotic manipulation",
        category="cs.RO",
        query='cat:cs.RO AND (all:"vision language" OR all:multimodal) AND (all:robot OR all:manipulation)',
        required_terms=["vision", "language", "robot"],
        optional_terms=["manipulation", "vision-language-action", "multimodal"],
    ),
)


class RealArxivCandidate(BaseModel):
    """One frozen candidate paper from arXiv."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str
    query: str
    fetched_at: datetime
    paper_id: str
    title: str
    abstract: str | None = None
    category: str
    categories: list[str] = Field(default_factory=list)
    submitted_date: date | None = None
    arxiv_url: str
    pdf_url: str | None = None
    candidate_rank: int = Field(ge=1)

    @field_validator("topic_id", "query", "paper_id", "title", "category", "arxiv_url")
    @classmethod
    def require_candidate_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("candidate text fields must not be blank")
        return normalized

    @field_validator("categories")
    @classmethod
    def normalize_categories(cls, value: list[str]) -> list[str]:
        return _dedupe_terms(value)


class RealArxivLabel(BaseModel):
    """Human relevance label for one candidate."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str
    paper_id: str
    relevant: bool
    label_note: str | None = None

    @field_validator("topic_id", "paper_id")
    @classmethod
    def require_label_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("label fields must not be blank")
        return normalized


class RealArxivMethodMetrics(BaseModel):
    """Metrics for one method on one topic or macro-average row."""

    topic_id: str
    method: Literal["agent", "semantic_agent", "strict_keyword", "bm25"]
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0
    mean_reciprocal_rank: float = 0.0
    relevant_count: int = 0
    candidate_count: int = 0
    top_5_paper_ids: list[str] = Field(default_factory=list)
    top_10_paper_ids: list[str] = Field(default_factory=list)


class RealArxivEvaluationReport(BaseModel):
    """Complete real-arXiv evaluation report."""

    topics: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    rows: list[RealArxivMethodMetrics] = Field(default_factory=list)
    macro_average_rows: list[RealArxivMethodMetrics] = Field(default_factory=list)
    candidate_count: int = 0
    label_count: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class _RankedItem:
    paper_id: str
    score: float
    original_rank: int


def evaluate_frozen_real_arxiv(
    *,
    candidates_path: Path = DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
    labels_path: Path = DEFAULT_REAL_ARXIV_LABELS_PATH,
    topics: Sequence[RealArxivTopic] = DEFAULT_REAL_ARXIV_TOPICS,
    semantic_provider: Literal["fake", "live", "none"] = "none",
) -> SkillResult[RealArxivEvaluationReport]:
    """Evaluate frozen candidates and labels with agent, keyword, and BM25 rankers."""

    candidate_result = load_real_arxiv_candidates(candidates_path)
    if candidate_result.status == SkillStatus.ERROR or candidate_result.data is None:
        return _report_error_from(candidate_result, stage="candidate_loading")
    label_result = load_real_arxiv_labels(labels_path)
    if label_result.status == SkillStatus.ERROR or label_result.data is None:
        return _report_error_from(label_result, stage="label_loading")

    candidates = candidate_result.data
    labels = label_result.data
    validation_error = _validate_real_arxiv_fixture(candidates, labels, topics)
    if validation_error is not None:
        return validation_error

    topics_by_id = {topic.topic_id: topic for topic in topics}
    candidates_by_topic = _candidates_by_topic(candidates)
    relevant_by_topic = _relevant_ids_by_topic(labels)
    rows: list[RealArxivMethodMetrics] = []
    for topic_id in topics_by_id:
        topic = topics_by_id[topic_id]
        topic_candidates = candidates_by_topic[topic_id]
        relevant_ids = relevant_by_topic[topic_id]
        rows.extend(
            [
                _metrics_for_ranked_ids(
                    topic_id=topic_id,
                    method="agent",
                    ranked_ids=_agent_ranked_ids(topic, topic_candidates),
                    relevant_ids=relevant_ids,
                    candidate_count=len(topic_candidates),
                ),
                *(
                    [
                        _metrics_for_ranked_ids(
                            topic_id=topic_id,
                            method="semantic_agent",
                            ranked_ids=_semantic_agent_ranked_ids(
                                topic,
                                topic_candidates,
                                semantic_provider=semantic_provider,
                            ),
                            relevant_ids=relevant_ids,
                            candidate_count=len(topic_candidates),
                        )
                    ]
                    if semantic_provider != "none"
                    else []
                ),
                _metrics_for_ranked_ids(
                    topic_id=topic_id,
                    method="strict_keyword",
                    ranked_ids=_strict_keyword_ranked_ids(topic, topic_candidates),
                    relevant_ids=relevant_ids,
                    candidate_count=len(topic_candidates),
                ),
                _metrics_for_ranked_ids(
                    topic_id=topic_id,
                    method="bm25",
                    ranked_ids=_bm25_ranked_ids(topic, topic_candidates),
                    relevant_ids=relevant_ids,
                    candidate_count=len(topic_candidates),
                ),
            ]
        )

    macro_rows = _macro_average_rows(rows)
    data = RealArxivEvaluationReport(
        topics=list(topics_by_id),
        methods=[
            "agent",
            *(["semantic_agent"] if semantic_provider != "none" else []),
            "strict_keyword",
            "bm25",
        ],
        rows=rows,
        macro_average_rows=macro_rows,
        candidate_count=len(candidates),
        label_count=len(labels),
    )
    return SkillResult[RealArxivEvaluationReport](
        status=SkillStatus.SUCCESS,
        data=data,
        evidence_source=EvidenceSource.METADATA,
        message="Evaluated frozen real-arXiv recommendation quality.",
        metadata={
            "candidate_path": str(candidates_path),
            "label_path": str(labels_path),
            "topic_count": len(topics),
            "candidate_count": len(candidates),
            "label_count": len(labels),
            "semantic_provider": semantic_provider,
        },
    )


def load_real_arxiv_candidates(
    path: Path,
) -> SkillResult[list[RealArxivCandidate]]:
    """Load candidate JSONL with structured errors."""

    return _load_jsonl_model(path, RealArxivCandidate, label="candidate")


def load_real_arxiv_labels(path: Path) -> SkillResult[list[RealArxivLabel]]:
    """Load label JSONL with structured errors."""

    return _load_jsonl_model(path, RealArxivLabel, label="label")


def write_label_template(
    *,
    candidates_path: Path = DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
    output_path: Path = DEFAULT_REAL_ARXIV_LABELS_PATH,
) -> SkillResult[dict[str, int | str]]:
    """Write a JSONL label template from candidates, preserving any existing labels."""

    candidate_result = load_real_arxiv_candidates(candidates_path)
    if candidate_result.status == SkillStatus.ERROR or candidate_result.data is None:
        return SkillResult[dict[str, int | str]](
            status=SkillStatus.ERROR,
            data={},
            evidence_source=EvidenceSource.METADATA,
            error=candidate_result.error,
            metadata={"stage": "candidate_loading"},
        )
    existing_by_key: dict[tuple[str, str], RealArxivLabel] = {}
    if output_path.exists():
        existing_result = load_real_arxiv_labels(output_path)
        if existing_result.status == SkillStatus.ERROR or existing_result.data is None:
            return SkillResult[dict[str, int | str]](
                status=SkillStatus.ERROR,
                data={},
                evidence_source=EvidenceSource.METADATA,
                error=existing_result.error,
                metadata={"stage": "existing_label_loading"},
            )
        existing_by_key = {
            (label.topic_id, label.paper_id): label for label in existing_result.data
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for candidate in candidate_result.data:
        label = existing_by_key.get((candidate.topic_id, candidate.paper_id))
        payload = {
            "topic_id": candidate.topic_id,
            "paper_id": candidate.paper_id,
            "relevant": bool(label.relevant) if label is not None else False,
            "label_note": label.label_note if label is not None else "",
        }
        rows.append(json.dumps(payload, sort_keys=True))
    output_path.write_text("\n".join(rows) + "\n")
    return SkillResult[dict[str, int | str]](
        status=SkillStatus.SUCCESS,
        data={"label_count": len(rows), "output_path": str(output_path)},
        evidence_source=EvidenceSource.METADATA,
        message="Wrote real-arXiv label template.",
    )


def fetch_real_arxiv_candidates(
    *,
    output_path: Path = DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
    topics: Sequence[RealArxivTopic] = DEFAULT_REAL_ARXIV_TOPICS,
    store_path: Path = Path("data/daily_arxiv_eval.sqlite3"),
    request_delay_seconds: float = 3.0,
    max_results_per_topic: int = 50,
) -> SkillResult[dict[str, int | str]]:
    """Fetch and freeze live arXiv candidates for manual evaluation refreshes."""

    store = SQLitePaperStore(str(store_path))
    retrieval = ArxivRetrievalSkill(
        store=store,
        request_delay_seconds=request_delay_seconds,
    )
    fetched_at = datetime.now(timezone.utc)
    candidates: list[RealArxivCandidate] = []
    for topic in topics:
        query = RetrievalQuery(
            topic=topic.topic,
            category=topic.category,
            max_results=max_results_per_topic,
            candidate_pool_size=max_results_per_topic,
            page_size=min(max_results_per_topic, 50),
            max_requests=1,
            search_mode=SearchMode.STRICT,
            query_planner_mode=QueryPlannerMode.DETERMINISTIC,
        )
        query_plan = QueryPlan(
            search_mode=SearchMode.STRICT,
            planner=QueryPlannerProvenance(
                requested_mode=QueryPlannerMode.DETERMINISTIC,
                source="real_arxiv_evaluation",
            ),
            variants=[
                QueryPlanVariant(
                    label="real_arxiv_topic",
                    search_query=topic.query,
                    sort_by="submittedDate",
                    sort_order="descending",
                )
            ],
            required_terms=topic.required_terms,
            optional_terms=topic.optional_terms,
            phrases=[topic.topic],
            rationale="Frozen real-arXiv evaluation candidate query.",
        )
        result = retrieval.retrieve(query, use_cache=False, query_plan=query_plan)
        if result.status not in {SkillStatus.SUCCESS, SkillStatus.FALLBACK}:
            return SkillResult[dict[str, int | str]](
                status=SkillStatus.ERROR,
                data={},
                evidence_source=EvidenceSource.METADATA,
                error=result.error
                or SkillError(
                    code="real_arxiv_fetch_failed",
                    message=f"Failed to fetch candidates for {topic.topic_id}.",
                    retryable=True,
                ),
                metadata={"topic_id": topic.topic_id, "status": result.status.value},
            )
        papers = list(result.data or [])
        if len(papers) != max_results_per_topic:
            return SkillResult[dict[str, int | str]](
                status=SkillStatus.ERROR,
                data={},
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code="real_arxiv_candidate_count_invalid",
                    message=(
                        f"Expected {max_results_per_topic} candidates for "
                        f"{topic.topic_id}, got {len(papers)}."
                    ),
                    retryable=True,
                ),
                metadata={"topic_id": topic.topic_id, "candidate_count": len(papers)},
            )
        candidates.extend(
            _candidate_from_paper(
                topic=topic,
                paper=paper,
                fetched_at=fetched_at,
                candidate_rank=index,
            )
            for index, paper in enumerate(papers, start=1)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            json.dumps(candidate.model_dump(mode="json"), sort_keys=True)
            for candidate in candidates
        )
        + "\n"
    )
    return SkillResult[dict[str, int | str]](
        status=SkillStatus.SUCCESS,
        data={"candidate_count": len(candidates), "output_path": str(output_path)},
        evidence_source=EvidenceSource.METADATA,
        message="Fetched and wrote frozen real-arXiv candidates.",
    )


def format_real_arxiv_report_markdown(report: RealArxivEvaluationReport) -> str:
    """Render a compact Markdown table for docs or CLI output."""

    lines = [
        "| Topic | Method | Precision@5 | Recall@10 | MRR | Relevant |",
        "|-------|--------|-------------|-----------|-----|----------|",
    ]
    for row in [*report.rows, *report.macro_average_rows]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.topic_id,
                    row.method,
                    f"{row.precision_at_5:.3f}",
                    f"{row.recall_at_10:.3f}",
                    f"{row.mean_reciprocal_rank:.3f}",
                    str(row.relevant_count),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _agent_ranked_ids(
    topic: RealArxivTopic,
    candidates: Sequence[RealArxivCandidate],
) -> list[str]:
    papers = [_paper_from_candidate(candidate) for candidate in candidates]
    result = TopicRankingSkill().rank(
        papers,
        topic=topic.topic,
        top_k=len(papers),
        query_plan=QueryPlan(
            search_mode=SearchMode.STRICT,
            planner=QueryPlannerProvenance(
                requested_mode=QueryPlannerMode.DETERMINISTIC,
                source="real_arxiv_evaluation",
            ),
            variants=[
                QueryPlanVariant(
                    label="frozen_candidates",
                    search_query=topic.query,
                    sort_by="submittedDate",
                    sort_order="descending",
                )
            ],
            required_terms=topic.required_terms,
            optional_terms=topic.optional_terms,
            phrases=[topic.topic],
        ),
        retrieval_query=RetrievalQuery(
            topic=topic.topic,
            category=topic.category,
            max_results=len(papers),
            candidate_pool_size=len(papers),
        ),
    )
    recommendations = result.data or []
    ranked_ids = [recommendation.paper.paper_id for recommendation in recommendations]
    if len(ranked_ids) == len(candidates):
        return ranked_ids
    seen = set(ranked_ids)
    ranked_ids.extend(
        candidate.paper_id for candidate in candidates if candidate.paper_id not in seen
    )
    return ranked_ids


def _semantic_agent_ranked_ids(
    topic: RealArxivTopic,
    candidates: Sequence[RealArxivCandidate],
    *,
    semantic_provider: Literal["fake", "live", "none"],
) -> list[str]:
    papers = [_paper_from_candidate(candidate) for candidate in candidates]
    seed_preference = _topic_seed_preference(topic)
    provider = (
        _fake_semantic_provider()
        if semantic_provider == "fake"
        else create_embedding_provider(AppConfig.from_env())
    )
    skill = SemanticSeedRankingSkill(
        embedding_provider=provider,
        config=_semantic_config(semantic_provider),
        semantic_weight=100.0,
        lexical_cap=3.0,
        phrase_cap=2.0,
        query_source_cap=1.5,
        recency_cap=1.0,
        category_cap=1.0,
        minimum_semantic_similarity=-1.0,
    )
    result = skill.rank(
        papers,
        topic=topic.topic,
        seed_preference=seed_preference,
        top_k=len(papers),
        query_plan=QueryPlan(
            search_mode=SearchMode.STRICT,
            planner=QueryPlannerProvenance(
                requested_mode=QueryPlannerMode.DETERMINISTIC,
                source="real_arxiv_semantic_evaluation",
            ),
            variants=[
                QueryPlanVariant(
                    label="frozen_candidates",
                    search_query=topic.query,
                    sort_by="submittedDate",
                    sort_order="descending",
                )
            ],
            required_terms=topic.required_terms,
            optional_terms=topic.optional_terms,
            phrases=[topic.topic],
        ),
        retrieval_query=RetrievalQuery(
            topic=topic.topic,
            category=topic.category,
            max_results=len(papers),
            candidate_pool_size=len(papers),
        ),
    )
    ranked_ids = [recommendation.paper.paper_id for recommendation in result.data or []]
    if len(ranked_ids) == len(candidates):
        return ranked_ids
    seen = set(ranked_ids)
    ranked_ids.extend(
        candidate.paper_id for candidate in candidates if candidate.paper_id not in seen
    )
    return ranked_ids


def _strict_keyword_ranked_ids(
    topic: RealArxivTopic,
    candidates: Sequence[RealArxivCandidate],
) -> list[str]:
    terms = _topic_tokens(topic)
    ranked = [
        _RankedItem(
            paper_id=candidate.paper_id,
            score=_strict_keyword_score(terms, _candidate_tokens(candidate)),
            original_rank=candidate.candidate_rank,
        )
        for candidate in candidates
    ]
    ranked.sort(key=lambda item: (-item.score, item.original_rank, item.paper_id))
    return [item.paper_id for item in ranked]


def _bm25_ranked_ids(
    topic: RealArxivTopic,
    candidates: Sequence[RealArxivCandidate],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[str]:
    query_terms = _topic_tokens(topic)
    documents = [_candidate_tokens(candidate) for candidate in candidates]
    document_count = len(documents)
    avg_doc_len = (
        sum(len(document) for document in documents) / document_count
        if document_count
        else 0.0
    )
    document_frequencies = Counter(
        term for document in documents for term in set(document)
    )
    ranked: list[_RankedItem] = []
    for candidate, document in zip(candidates, documents, strict=True):
        counts = Counter(document)
        score = 0.0
        doc_len = len(document) or 1
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequencies.get(term, 0)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * doc_len / (avg_doc_len or 1.0))
            score += idf * (frequency * (k1 + 1.0)) / denominator
        ranked.append(
            _RankedItem(
                paper_id=candidate.paper_id,
                score=score,
                original_rank=candidate.candidate_rank,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.original_rank, item.paper_id))
    return [item.paper_id for item in ranked]


def _metrics_for_ranked_ids(
    *,
    topic_id: str,
    method: Literal["agent", "semantic_agent", "strict_keyword", "bm25"],
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    candidate_count: int,
) -> RealArxivMethodMetrics:
    top_5 = list(ranked_ids[:5])
    top_10 = list(ranked_ids[:10])
    first_relevant = next(
        (
            position
            for position, paper_id in enumerate(ranked_ids, start=1)
            if paper_id in relevant_ids
        ),
        None,
    )
    return RealArxivMethodMetrics(
        topic_id=topic_id,
        method=method,
        precision_at_5=round(
            sum(1 for paper_id in top_5 if paper_id in relevant_ids) / 5,
            4,
        ),
        recall_at_10=round(
            sum(1 for paper_id in top_10 if paper_id in relevant_ids)
            / len(relevant_ids)
            if relevant_ids
            else 0.0,
            4,
        ),
        mean_reciprocal_rank=round(1.0 / first_relevant if first_relevant else 0.0, 4),
        relevant_count=len(relevant_ids),
        candidate_count=candidate_count,
        top_5_paper_ids=top_5,
        top_10_paper_ids=top_10,
    )


def _macro_average_rows(
    rows: Sequence[RealArxivMethodMetrics],
) -> list[RealArxivMethodMetrics]:
    rows_by_method: dict[str, list[RealArxivMethodMetrics]] = defaultdict(list)
    for row in rows:
        rows_by_method[row.method].append(row)
    macro_rows: list[RealArxivMethodMetrics] = []
    for method, method_rows in rows_by_method.items():
        macro_rows.append(
            RealArxivMethodMetrics(
                topic_id="macro_average",
                method=method,
                precision_at_5=round(_mean(row.precision_at_5 for row in method_rows), 4),
                recall_at_10=round(_mean(row.recall_at_10 for row in method_rows), 4),
                mean_reciprocal_rank=round(
                    _mean(row.mean_reciprocal_rank for row in method_rows),
                    4,
                ),
                relevant_count=sum(row.relevant_count for row in method_rows),
                candidate_count=sum(row.candidate_count for row in method_rows),
            )
        )
    return macro_rows


def _validate_real_arxiv_fixture(
    candidates: Sequence[RealArxivCandidate],
    labels: Sequence[RealArxivLabel],
    topics: Sequence[RealArxivTopic],
) -> SkillResult[RealArxivEvaluationReport] | None:
    topic_ids = [topic.topic_id for topic in topics]
    candidates_by_topic = _candidates_by_topic(candidates)
    labels_by_key = {(label.topic_id, label.paper_id): label for label in labels}
    if len(labels_by_key) != len(labels):
        return _validation_error("real_arxiv_duplicate_labels", "Duplicate labels found.")
    candidate_keys = {
        (candidate.topic_id, candidate.paper_id) for candidate in candidates
    }
    missing_labels = sorted(candidate_keys - set(labels_by_key))
    unknown_labels = sorted(set(labels_by_key) - candidate_keys)
    if missing_labels:
        return _validation_error(
            "real_arxiv_missing_labels",
            f"Missing labels for {len(missing_labels)} candidate(s).",
            {"examples": missing_labels[:5]},
        )
    if unknown_labels:
        return _validation_error(
            "real_arxiv_unknown_label_ids",
            f"Labels reference {len(unknown_labels)} unknown candidate(s).",
            {"examples": unknown_labels[:5]},
        )
    for topic_id in topic_ids:
        topic_candidates = candidates_by_topic.get(topic_id, [])
        if len(topic_candidates) != 50:
            return _validation_error(
                "real_arxiv_topic_candidate_count_invalid",
                f"Topic {topic_id} must have exactly 50 candidates.",
                {"topic_id": topic_id, "candidate_count": len(topic_candidates)},
            )
        ranks = sorted(candidate.candidate_rank for candidate in topic_candidates)
        if ranks != list(range(1, 51)):
            return _validation_error(
                "real_arxiv_candidate_ranks_invalid",
                f"Topic {topic_id} candidate ranks must be 1..50.",
                {"topic_id": topic_id},
            )
        relevant_count = sum(
            1 for label in labels if label.topic_id == topic_id and label.relevant
        )
        if relevant_count == 0:
            return _validation_error(
                "real_arxiv_topic_labels_empty",
                f"Topic {topic_id} must include at least one relevant label.",
                {"topic_id": topic_id},
            )
    unknown_topic_ids = sorted(set(candidates_by_topic) - set(topic_ids))
    if unknown_topic_ids:
        return _validation_error(
            "real_arxiv_unknown_topic_ids",
            "Candidate file contains unknown topic IDs.",
            {"topic_ids": unknown_topic_ids},
        )
    return None


def _validation_error(
    code: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> SkillResult[RealArxivEvaluationReport]:
    return SkillResult[RealArxivEvaluationReport](
        status=SkillStatus.ERROR,
        data=None,
        evidence_source=EvidenceSource.METADATA,
        error=SkillError(code=code, message=message, retryable=False),
        metadata=dict(metadata or {}),
    )


def _report_error_from(
    result: SkillResult[Any],
    *,
    stage: str,
) -> SkillResult[RealArxivEvaluationReport]:
    return SkillResult[RealArxivEvaluationReport](
        status=SkillStatus.ERROR,
        data=None,
        evidence_source=EvidenceSource.METADATA,
        error=result.error
        or SkillError(
            code="real_arxiv_evaluation_failed",
            message=result.message or "Real arXiv evaluation failed.",
            retryable=False,
        ),
        metadata={"stage": stage, **result.metadata},
    )


def _load_jsonl_model(path: Path, model: type[Any], *, label: str) -> SkillResult[Any]:
    if not path.exists():
        return SkillResult[Any](
            status=SkillStatus.ERROR,
            data=[],
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code=f"real_arxiv_{label}_file_missing",
                message=f"Real arXiv {label} file not found: {path}",
                retryable=False,
            ),
        )
    rows: list[Any] = []
    try:
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                rows.append(model.model_validate(payload))
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                return SkillResult[Any](
                    status=SkillStatus.ERROR,
                    data=[],
                    evidence_source=EvidenceSource.METADATA,
                    error=SkillError(
                        code=f"real_arxiv_{label}_file_invalid",
                        message=(
                            f"Invalid real arXiv {label} row at "
                            f"{path}:{line_number}: {exc}"
                        ),
                        retryable=False,
                    ),
                    metadata={"path": str(path), "line_number": line_number},
                )
    except OSError as exc:
        return SkillResult[Any](
            status=SkillStatus.ERROR,
            data=[],
            evidence_source=EvidenceSource.METADATA,
            error=SkillError(
                code=f"real_arxiv_{label}_file_unreadable",
                message=f"Could not read real arXiv {label} file {path}: {exc}",
                retryable=False,
            ),
        )
    return SkillResult[Any](
        status=SkillStatus.SUCCESS if rows else SkillStatus.EMPTY,
        data=rows,
        evidence_source=EvidenceSource.METADATA,
        metadata={"path": str(path), "row_count": len(rows)},
    )


def _candidate_from_paper(
    *,
    topic: RealArxivTopic,
    paper: PaperMetadata,
    fetched_at: datetime,
    candidate_rank: int,
) -> RealArxivCandidate:
    return RealArxivCandidate(
        topic_id=topic.topic_id,
        query=topic.query,
        fetched_at=fetched_at,
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        category=topic.category,
        categories=paper.categories,
        submitted_date=paper.published_date,
        arxiv_url=str(paper.arxiv_url),
        pdf_url=str(paper.pdf_url) if paper.pdf_url is not None else None,
        candidate_rank=candidate_rank,
    )


def _paper_from_candidate(candidate: RealArxivCandidate) -> PaperMetadata:
    return PaperMetadata(
        paper_id=candidate.paper_id,
        title=candidate.title,
        authors=[],
        abstract=candidate.abstract,
        categories=candidate.categories or [candidate.category],
        published_date=candidate.submitted_date,
        updated_date=candidate.submitted_date,
        arxiv_url=candidate.arxiv_url,
        pdf_url=candidate.pdf_url,
        provenance=Provenance(
            source="frozen_arxiv_evaluation",
            source_url=candidate.arxiv_url,
            query=candidate.query,
        ),
    )


def _topic_seed_preference(topic: RealArxivTopic) -> SeedPreference:
    preference_text = " ".join(
        [
            topic.topic,
            *topic.required_terms,
            *topic.optional_terms,
            topic.category,
        ]
    )
    return SeedPreference(
        profile_id=f"real-arxiv-{topic.topic_id}",
        seeds=[
            SeedRecord(
                identity=f"topic:{topic.topic_id}",
                input_text=topic.topic,
                input_type="topic",
                title=topic.topic,
                abstract=preference_text,
                preference_text=preference_text,
            )
        ],
        preference_text=preference_text,
        vector=DeterministicTextVectorizer().vectorize(preference_text),
    )


def _fake_semantic_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        dimensions=16,
        synonym_groups={
            "llm_tool_agents": (
                "llm",
                "large language model",
                "agent",
                "tool",
                "function calling",
            ),
            "retrieval_augmented_generation": (
                "retrieval",
                "rag",
                "augmented generation",
                "grounded generation",
            ),
            "vision_language_robotics": (
                "vision-language",
                "vision language",
                "multimodal",
                "robot",
                "manipulation",
            ),
        },
    )


def _semantic_config(
    semantic_provider: Literal["fake", "live", "none"],
) -> AppConfig:
    if semantic_provider == "fake":
        return AppConfig(
            embedding_provider="fake",
            embedding_model="fake-real-arxiv-semantic",
            embedding_dimensions=16,
            embedding_cache_enabled=False,
        )
    return AppConfig.from_env()


def _candidates_by_topic(
    candidates: Sequence[RealArxivCandidate],
) -> dict[str, list[RealArxivCandidate]]:
    grouped: dict[str, list[RealArxivCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.topic_id].append(candidate)
    for topic_candidates in grouped.values():
        topic_candidates.sort(key=lambda candidate: candidate.candidate_rank)
    return dict(grouped)


def _relevant_ids_by_topic(labels: Sequence[RealArxivLabel]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        if label.relevant:
            grouped[label.topic_id].add(label.paper_id)
        else:
            grouped.setdefault(label.topic_id, set())
    return dict(grouped)


def _topic_tokens(topic: RealArxivTopic) -> list[str]:
    return _dedupe_terms(
        [
            token
            for text in [*topic.required_terms, *topic.optional_terms, topic.topic]
            for token in _tokenize(text)
        ]
    )


def _candidate_tokens(candidate: RealArxivCandidate) -> list[str]:
    return _tokenize(f"{candidate.title} {candidate.abstract or ''}")


def _strict_keyword_score(terms: Sequence[str], document_tokens: Sequence[str]) -> float:
    counts = Counter(document_tokens)
    return float(sum(counts.get(term, 0) for term in terms))


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


_STOPWORDS = {
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
