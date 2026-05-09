"""Streamlit demo UI for the local Daily arXiv workflow."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape
from typing import Any

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    DailyBriefing,
    EvidenceSource,
    ExplanationMode,
    PaperDeepExplanation,
    QueryPlannerMode,
    Recommendation,
    RetrievalQuery,
    SeedPreference,
    SearchMode,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.llm.provider import create_llm_provider
from daily_arxiv_agent.orchestrator import (
    DailyArxivAgentOrchestrator,
    FeedbackWorkflow,
    PaperExplanationWorkflow,
    RecommendationWorkflow,
    RECOMMENDATION_MODE_AUTO,
    RECOMMENDATION_MODE_DETERMINISTIC,
    RECOMMENDATION_MODE_SEMANTIC_SEED,
    WorkflowTraceStep,
)
from daily_arxiv_agent.skills.feedback import FeedbackValue
from daily_arxiv_agent.skills.followup import FollowupQuery
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


DEFAULT_PROFILE_ID = "default"
DEFAULT_PROVIDER_MODE = "fake"
DEFAULT_TOP_K = 5
DEFAULT_MAX_RESULTS = 10
NEUTRAL_FEEDBACK = "neutral"
PROVIDER_MODES = {
    DEFAULT_PROVIDER_MODE: "Fake LLM only (arXiv may still be live)",
    "environment": "Configured live LLM provider",
}
SEARCH_MODE_LABELS = {
    SearchMode.BROAD.value: "Broad search",
    SearchMode.STRICT.value: "Strict search",
}
QUERY_PLANNER_LABELS = {
    QueryPlannerMode.AUTO.value: "Auto",
    QueryPlannerMode.DETERMINISTIC.value: "Deterministic",
    QueryPlannerMode.LLM.value: "LLM",
}
RECOMMENDATION_MODE_LABELS = {
    RECOMMENDATION_MODE_AUTO: "Auto",
    RECOMMENDATION_MODE_DETERMINISTIC: "Deterministic",
    RECOMMENDATION_MODE_SEMANTIC_SEED: "Semantic seed",
}
APP_CSS = """
<style>
:root {
  --canvas: #f3f7f4;
  --canvas-deep: #e6eee8;
  --surface: #fffffd;
  --surface-muted: #edf4f0;
  --ink: #1c2733;
  --ink-strong: #0f172a;
  --muted: #586879;
  --accent: #0f766e;
  --accent-strong: #0c5f59;
  --accent-pressed: #083f3b;
  --accent-text: #ffffff;
  --accent-soft: rgba(15, 118, 110, 0.11);
  --blue: #315f84;
  --blue-soft: rgba(49, 95, 132, 0.1);
  --amber: #9a6616;
  --border: rgba(15, 23, 42, 0.13);
  --border-strong: rgba(15, 23, 42, 0.22);
  --shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
}
[data-testid="stAppViewContainer"] {
  background:
    linear-gradient(180deg, #f8fbf8 0%, var(--canvas) 48%, var(--canvas-deep) 100%);
  color: var(--ink);
}
[data-testid="stAppViewContainer"] :where(
  p,
  span,
  label,
  li,
  small,
  strong,
  h1,
  h2,
  h3,
  h4,
  h5,
  h6
) {
  color: var(--ink) !important;
}
[data-testid="stSidebar"] :where(
  p,
  span,
  label,
  li,
  small,
  strong,
  h1,
  h2,
  h3,
  h4,
  h5,
  h6
) {
  color: var(--ink) !important;
}
[data-testid="stHeader"] {
  background: rgba(248, 251, 248, 0.88);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(15, 23, 42, 0.06);
}
[data-testid="stSidebar"] {
  background: rgba(255, 255, 253, 0.94);
  border-right: 1px solid var(--border);
}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
  color: var(--ink) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] {
  color: var(--ink) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  color: var(--ink-strong) !important;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 1.28rem;
  letter-spacing: 0;
}
[data-testid="stTextInputRootElement"] input,
[data-testid="stNumberInputRootElement"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div,
[data-baseweb="select"] input {
  color: var(--ink) !important;
  background: rgba(255, 255, 253, 0.96) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputRootElement"],
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-baseweb="select"] > div {
  background: rgba(255, 255, 253, 0.96) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stTextInputRootElement"] input::placeholder,
[data-testid="stNumberInputRootElement"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-baseweb="select"] input::placeholder {
  color: #6b7280 !important;
}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stNumberInputRootElement"]:focus-within,
[data-testid="stTextArea"]:focus-within,
[data-baseweb="select"]:focus-within,
[data-testid="stDateInputField"]:focus-within {
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16) !important;
  border-radius: 8px !important;
}
[data-baseweb="select"] svg,
[data-testid="stDateInputField"] svg {
  fill: var(--ink) !important;
}
[data-testid="stDateInputField"],
[data-testid="stDateInputField"] input {
  color: var(--ink) !important;
  background: rgba(255, 255, 253, 0.96) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stSlider"] [role="slider"] {
  background: var(--accent) !important;
  box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] * {
  color: var(--accent) !important;
  font-weight: 800 !important;
}
[data-testid="stAlertContent"] p,
[data-testid="stAlertContent"] span,
[data-testid="stAlertContent"] div {
  color: inherit !important;
}
[data-testid="stAlert"] {
  border-radius: 8px !important;
  border: 1px solid rgba(15, 23, 42, 0.08) !important;
}
[data-testid="stFormSubmitButton"] button,
.stButton > button {
  color: var(--accent-text) !important;
  background: var(--accent) !important;
  border: 1px solid rgba(8, 65, 61, 0.3) !important;
  border-radius: 8px !important;
  box-shadow: 0 8px 18px rgba(15, 118, 110, 0.18) !important;
  font-weight: 700 !important;
  transition:
    background 160ms ease,
    transform 160ms ease,
    box-shadow 160ms ease !important;
}
[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {
  color: var(--accent-text) !important;
  background: var(--accent-strong) !important;
  border-color: rgba(8, 65, 61, 0.36) !important;
  box-shadow: 0 12px 22px rgba(15, 118, 110, 0.22) !important;
  transform: translateY(-1px);
}
[data-testid="stFormSubmitButton"] button:active,
.stButton > button:active {
  color: var(--accent-text) !important;
  background: var(--accent-pressed) !important;
  transform: translateY(0);
  box-shadow: 0 6px 14px rgba(15, 118, 110, 0.18) !important;
}
[data-testid="stFormSubmitButton"] button:focus,
[data-testid="stFormSubmitButton"] button:focus-visible,
.stButton > button:focus,
.stButton > button:focus-visible {
  color: var(--accent-text) !important;
  outline: 3px solid rgba(15, 118, 110, 0.24) !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 2px rgba(255, 255, 253, 0.9), 0 0 0 6px rgba(15, 118, 110, 0.18) !important;
}
[data-testid="stFormSubmitButton"] button:disabled,
.stButton > button:disabled {
  color: rgba(255, 255, 253, 0.82) !important;
  background: #7fa6a1 !important;
  border-color: transparent !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}
.block-container {
  max-width: 1260px;
  padding-top: 1.75rem;
  padding-bottom: 3rem;
}
.workspace-header,
.section-shell {
  animation: rise 0.42s ease both;
}
.workspace-header {
  padding: 1.6rem 0 1.15rem;
  border-bottom: 1px solid var(--border);
}
.workspace-header__grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 16rem;
  gap: 1.25rem;
  align-items: end;
}
.workspace-header__note {
  border-left: 3px solid var(--accent);
  padding: 0.75rem 0 0.75rem 0.95rem;
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.55;
  background: linear-gradient(90deg, rgba(15, 118, 110, 0.08), transparent);
}
.section-shell {
  border-top: 1px solid var(--border);
  padding: 1.2rem 0 0.2rem;
  margin: 1.65rem 0 0.9rem;
}
.hero-kicker {
  margin: 0;
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.hero-title {
  margin: 0;
  color: var(--ink-strong);
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 2.2rem;
  line-height: 1.08;
  max-width: 50rem;
}
.hero-copy {
  margin: 0.7rem 0 0;
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.55;
  max-width: 48rem;
}
.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.85rem 0 0.85rem;
}
.pill {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(49, 95, 132, 0.18);
  border-radius: 999px;
  padding: 0.34rem 0.72rem;
  color: #244b68;
  background: rgba(255, 255, 253, 0.78);
  font-size: 0.86rem;
  font-weight: 650;
}
.pill strong {
  color: var(--ink-strong) !important;
  font-weight: 750;
}
.section-title {
  margin: 0 0 0.25rem;
  color: var(--ink-strong);
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 1.42rem;
}
.section-copy {
  margin: 0 0 0.55rem;
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.5;
  max-width: 58rem;
}
.section-kicker {
  margin: 0 0 0.25rem;
  color: var(--blue);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}
.run-summary-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.55rem;
  margin: 0.7rem 0 0.45rem;
}
.summary-tile {
  min-height: 4.65rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: rgba(255, 255, 253, 0.82);
  padding: 0.72rem 0.78rem;
}
.summary-tile__label {
  display: block;
  margin-bottom: 0.32rem;
  color: var(--muted) !important;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.summary-tile__value {
  display: block;
  color: var(--ink-strong) !important;
  font-size: 1.08rem;
  font-weight: 800;
  line-height: 1.18;
  overflow-wrap: anywhere;
}
.briefing-callout,
.evidence-panel,
.explanation-panel {
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  background: rgba(255, 255, 253, 0.86);
  box-shadow: var(--shadow);
  margin: 0.7rem 0 1rem;
  padding: 1rem 1.05rem;
}
.briefing-callout h3,
.evidence-panel h3,
.explanation-panel h3 {
  margin: 0 0 0.42rem;
  color: var(--ink-strong) !important;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  font-size: 1.25rem;
}
.briefing-callout p,
.evidence-panel p,
.explanation-panel p {
  margin: 0;
  color: var(--ink) !important;
  line-height: 1.58;
}
.metadata-line {
  margin-top: 0.55rem !important;
  color: var(--muted) !important;
  font-size: 0.86rem;
  font-weight: 650;
}
[data-testid="stMetric"] {
  min-height: 4.7rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: rgba(255, 255, 253, 0.76);
  padding: 0.62rem 0.74rem;
  box-shadow: none;
}
[data-testid="stMetricLabel"] p {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
  color: var(--ink-strong) !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  background: var(--surface);
}
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: rgba(255, 255, 253, 0.78) !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] summary {
  font-weight: 750 !important;
}
[data-baseweb="tab-list"] {
  gap: 0.25rem;
}
[data-baseweb="tab"] {
  border-radius: 8px 8px 0 0 !important;
  color: var(--muted) !important;
  font-weight: 750 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  background: rgba(15, 118, 110, 0.08) !important;
}
@media (max-width: 900px) {
  .workspace-header__grid,
  .run-summary-grid {
    grid-template-columns: 1fr;
  }
  .hero-title {
    font-size: 1.86rem;
  }
  .workspace-header__note {
    padding-top: 0.62rem;
    padding-bottom: 0.62rem;
  }
}
@media (prefers-reduced-motion: reduce) {
  .workspace-header,
  .section-shell {
    animation: none;
  }
  [data-testid="stFormSubmitButton"] button,
  .stButton > button {
    transition: none !important;
  }
}
@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
"""


@dataclass(frozen=True)
class RuntimeContext:
    """Configured runtime dependencies for one UI action."""

    store: SQLitePaperStore
    orchestrator: DailyArxivAgentOrchestrator
    provider_label: str
    db_path: str


def recommendation_rows(
    recommendations: Sequence[Recommendation],
) -> list[dict[str, object]]:
    """Convert structured recommendations into dataframe-friendly rows."""

    rows: list[dict[str, object]] = []
    for recommendation in recommendations:
        row: dict[str, object] = {
            "rank": recommendation.rank,
            "paper_id": recommendation.paper.paper_id,
            "title": recommendation.paper.title,
            "score": round(recommendation.score, 4),
            "evidence": recommendation.evidence_source.value,
            "categories": ", ".join(recommendation.paper.categories),
            "rationale": recommendation.rationale,
            "arxiv_url": str(recommendation.paper.arxiv_url),
        }
        row.update(_semantic_recommendation_row_fields(recommendation))
        if recommendation.previous_rank is not None:
            row["previous_rank"] = recommendation.previous_rank
        if recommendation.rank_delta is not None:
            row["rank_delta"] = recommendation.rank_delta
        if recommendation.score_delta is not None:
            row["score_delta"] = round(recommendation.score_delta, 4)
        rows.append(row)
    return rows


def workflow_trace_rows(trace: Sequence[WorkflowTraceStep]) -> list[dict[str, object]]:
    """Render workflow trace steps with visible evidence and fallback details."""

    rows: list[dict[str, object]] = []
    for step in trace:
        metadata = step.metadata
        rows.append(
            {
                "step": step.step,
                "skill": step.skill,
                "status": step.status.value,
                "evidence": (
                    step.evidence_source.value
                    if step.evidence_source is not None
                    else "n/a"
                ),
                "fallback": "yes" if step.fallback else "no",
                "input": step.input_summary,
                "output": step.output_summary,
                "error": _format_trace_error(step),
                "planner_source": _planner_source_for_step(step),
                "planner_fallback": _planner_fallback_for_step(step),
                "query_variants": _metadata_text(metadata.get("query_variant_count")),
                "candidates": _metadata_text(metadata.get("candidate_count")),
                "cache": _cache_summary_for_step(step),
                "ranking_mode": _metadata_text(metadata.get("ranking_mode")),
                "recommendation_mode": _metadata_text(
                    metadata.get("recommendation_mode")
                ),
                "semantic_provider": _semantic_provider_summary_for_step(step),
                "embedding_cache": _embedding_cache_summary_for_step(step),
                "semantic_state": _semantic_state_for_step(step),
            }
        )
    return rows


def semantic_readiness_panel_rows(
    readiness: Mapping[str, Any] | None,
    *,
    cache_path: str | None,
    recommendation_mode: str,
    seed_count: int,
) -> list[dict[str, object]]:
    """Build trace-safe semantic readiness rows for the pre-run UI panel."""

    metadata = dict(readiness or {})
    cache_enabled = bool(metadata.get("cache_enabled"))
    warnings = metadata.get("warnings") or []
    if not isinstance(warnings, Sequence) or isinstance(warnings, str):
        warnings = []
    return [
        {
            "item": "recommendation_mode",
            "status": _recommendation_mode_label(recommendation_mode),
            "details": _recommendation_mode_help(recommendation_mode),
        },
        {
            "item": "readiness",
            "status": "ready" if metadata.get("can_run") else "blocked",
            "details": _metadata_text(metadata.get("error_code")),
        },
        {
            "item": "provider",
            "status": _metadata_text(metadata.get("provider_label")),
            "details": _metadata_text(metadata.get("provider_mode")),
        },
        {
            "item": "model",
            "status": _metadata_text(metadata.get("model")),
            "details": _metadata_text(metadata.get("provider")),
        },
        {
            "item": "credentials",
            "status": _metadata_text(metadata.get("credential_status")),
            "details": "checked before retrieval",
        },
        {
            "item": "endpoint",
            "status": _metadata_text(metadata.get("endpoint_safety")),
            "details": _metadata_text(metadata.get("endpoint")),
        },
        {
            "item": "cache",
            "status": "enabled" if cache_enabled else "disabled",
            "details": cache_path or "",
        },
        {
            "item": "seed_quality",
            "status": _metadata_text(metadata.get("seed_quality")),
            "details": f"{seed_count} seed input(s)",
        },
        {
            "item": "warnings",
            "status": str(len(warnings)),
            "details": "; ".join(str(item) for item in warnings),
        },
    ]


def semantic_error_notice(
    result: SkillResult[RecommendationWorkflow] | None,
) -> dict[str, str] | None:
    """Return a semantic-specific banner when recommendations need explanation."""

    if result is None:
        return None
    workflow = result.data
    code = (
        result.error.code
        if result.error is not None
        and result.error.code.startswith("semantic_")
        else None
    )
    if code is None and workflow is not None:
        code = _first_semantic_trace_error_code(workflow.trace)
    if code is not None:
        if (
            code == "semantic_feedback_refinement_failed"
            and workflow is not None
            and workflow.recommendations
        ):
            return {
                "kind": "warning",
                "message": (
                    "Semantic feedback refinement failed, so the original "
                    "semantic recommendations were preserved. Inspect the "
                    "feedback diagnostics before trusting rank movement."
                ),
            }
        return _semantic_error_notice_for_code(code)
    if workflow is None:
        return None

    retrieval_state = _semantic_retrieval_state(workflow.trace)
    if retrieval_state == "no_seed_candidates":
        return {
            "kind": "warning",
            "message": (
                "Semantic recommendations were withheld because seed-derived "
                "retrieval returned no non-seed candidates. Broaden the date, "
                "category, or seed set."
            ),
        }
    if _all_semantic_recommendations_low_evidence(workflow.recommendations):
        return {
            "kind": "warning",
            "message": (
                "Semantic embedding succeeded, but every visible recommendation "
                "is below the configured evidence threshold and is labeled as "
                "low evidence."
            ),
        }
    return None


def semantic_provider_cache_rows(
    trace: Sequence[WorkflowTraceStep],
) -> list[dict[str, object]]:
    """Summarize provider, cache, retrieval, and error states without raw payloads."""

    rows: list[dict[str, object]] = []
    for step in trace:
        metadata = step.metadata
        provider = _semantic_provider_summary_for_step(step)
        embedding_cache = _embedding_cache_summary_for_step(step)
        retrieval_cache = _cache_summary_for_step(step)
        semantic_state = _semantic_state_for_step(step)
        if not any(
            (
                provider,
                embedding_cache,
                retrieval_cache,
                semantic_state,
                step.skill in {"semantic_readiness", "arxiv_retrieval", "ranking"},
            )
        ):
            continue
        rows.append(
            {
                "step": step.step,
                "skill": step.skill,
                "status": step.status.value,
                "recommendation_mode": _metadata_text(
                    metadata.get("recommendation_mode")
                ),
                "provider": provider,
                "model": _semantic_model_for_step(step),
                "credential": _metadata_text(metadata.get("credential_status")),
                "endpoint": _metadata_text(metadata.get("endpoint_safety")),
                "embedding_cache": embedding_cache,
                "retrieval_cache": retrieval_cache,
                "candidates": _metadata_text(metadata.get("candidate_count")),
                "semantic_state": semantic_state,
            }
        )
    return rows


def semantic_similarity_rows(
    recommendations: Sequence[Recommendation],
) -> list[dict[str, object]]:
    """Render per-seed similarity diagnostics from structured score data."""

    rows: list[dict[str, object]] = []
    for recommendation in sorted(recommendations, key=lambda item: item.rank):
        breakdown = recommendation.score_breakdown
        if breakdown is None:
            continue
        top = _top_semantic_similarity(breakdown.semantic_similarities)
        for detail in sorted(
            breakdown.semantic_similarities,
            key=lambda item: item.similarity,
            reverse=True,
        ):
            rows.append(
                {
                    "rank": recommendation.rank,
                    "paper_id": recommendation.paper.paper_id,
                    "title": recommendation.paper.title,
                    "seed": _similarity_source_label(detail),
                    "similarity": round(detail.similarity, 4),
                    "weighted_score": round(detail.score, 4),
                    "top_match": "yes" if detail is top else "no",
                }
            )
    return rows


def score_composition_rows(
    recommendations: Sequence[Recommendation],
) -> list[dict[str, object]]:
    """Render score components from structured breakdown fields."""

    rows: list[dict[str, object]] = []
    for recommendation in sorted(recommendations, key=lambda item: item.rank):
        breakdown = recommendation.score_breakdown
        if breakdown is None:
            continue
        rows.append(
            {
                "rank": recommendation.rank,
                "paper_id": recommendation.paper.paper_id,
                "title": recommendation.paper.title,
                "semantic": round(breakdown.semantic_seed, 4),
                "lexical": round(breakdown.lexical, 4),
                "phrase": round(breakdown.phrase, 4),
                "category": round(breakdown.category, 4),
                "recency": round(breakdown.recency, 4),
                "query_source": round(breakdown.query_source, 4),
                "feedback": round(breakdown.feedback, 4),
                "total": round(breakdown.total, 4),
                "evidence_score": round(breakdown.evidence_score, 4),
                "evidence_status": _semantic_evidence_status(breakdown),
            }
        )
    return rows


def recommendation_why_rows(
    recommendations: Sequence[Recommendation],
) -> list[dict[str, object]]:
    """Build row-level why details without parsing rationale text."""

    rows: list[dict[str, object]] = []
    for recommendation in sorted(recommendations, key=lambda item: item.rank):
        breakdown = recommendation.score_breakdown
        if breakdown is None:
            continue
        top = _top_semantic_similarity(breakdown.semantic_similarities)
        rows.append(
            {
                "rank": recommendation.rank,
                "paper_id": recommendation.paper.paper_id,
                "title": recommendation.paper.title,
                "total": round(breakdown.total, 4),
                "semantic": round(breakdown.semantic_seed, 4),
                "top_matching_seed": (
                    _similarity_source_label(top) if top is not None else ""
                ),
                "top_similarity": (
                    round(top.similarity, 4) if top is not None else ""
                ),
                "matched_terms": ", ".join(breakdown.matched_terms),
                "matched_phrases": ", ".join(breakdown.matched_phrases),
                "signals": ", ".join(breakdown.signals),
                "evidence_status": _semantic_evidence_status(breakdown),
            }
        )
    return rows


def feedback_impact_rows(
    recommendations: Sequence[Recommendation],
) -> list[dict[str, object]]:
    """Render semantic feedback movement and influence rows."""

    rows: list[dict[str, object]] = []
    for recommendation in sorted(recommendations, key=lambda item: item.rank):
        influences = recommendation.feedback_influences
        if not influences and not _has_feedback_movement(recommendation):
            continue
        if not influences:
            rows.append(_feedback_impact_row(recommendation, None))
            continue
        for influence in influences:
            rows.append(_feedback_impact_row(recommendation, influence))
    return rows


def recommendation_summary_metrics(
    workflow: RecommendationWorkflow,
) -> dict[str, object]:
    """Build concise recommendation-run metrics for CLI/UI parity tests."""

    planning_metadata = _metadata_for_skill(workflow.trace, "query_planning")
    retrieval_metadata = _metadata_for_skill(workflow.trace, "arxiv_retrieval")
    planner_fallback = (
        planning_metadata.get("fallback")
        or retrieval_metadata.get("planner_fallback")
        or False
    )
    return {
        "run_id": workflow.run_id[:12],
        "candidate_pool_size": workflow.query.effective_candidate_pool_size,
        "candidates_retrieved": _metadata_int(
            retrieval_metadata.get("candidate_count"),
            default=len(workflow.papers),
        ),
        "recommendations_shown": len(workflow.recommendations),
        "planner_source": (
            planning_metadata.get("source")
            or retrieval_metadata.get("planner_source")
            or "unknown"
        ),
        "planner_fallback": "yes" if bool(planner_fallback) else "no",
        "cache_status": retrieval_metadata.get("cache_status") or "unknown",
        "cache_hit": "yes" if retrieval_metadata.get("cache_hit") is True else "no",
    }


def briefing_rows(briefing: DailyBriefing | None) -> list[dict[str, object]]:
    """Render briefing summary table rows."""

    if briefing is None:
        return []
    return [
        {
            "rank": row.rank,
            "paper_id": row.paper_id,
            "title": row.title,
            "score": round(row.score, 4),
            "evidence": row.evidence_source.value,
            "key_reason": row.key_reason,
            "arxiv_url": str(row.arxiv_url),
        }
        for row in briefing.summary_table
    ]


def enhanced_briefing_sections(
    briefing: DailyBriefing | None,
) -> list[dict[str, object]]:
    """Convert enhanced briefing fields into renderable section dictionaries."""

    if briefing is None:
        return []

    boundary = briefing.evidence_boundary
    return [
        {
            "key": "executive_summary",
            "title": "Executive Summary",
            "body": briefing.executive_summary,
        },
        {
            "key": "top_k_reading_guide",
            "title": "Top-K Reading Guide",
            "summary_rows": briefing_rows(briefing),
            "paper_briefs": briefing_paper_brief_rows(briefing),
        },
        {
            "key": "evidence_boundary",
            "title": "Evidence Boundary",
            "full_text_used": "yes" if boundary.full_text_used else "no",
            "evidence_sources": _evidence_sources_text(boundary.evidence_sources),
            "unavailable_sources": _evidence_sources_text(
                boundary.unavailable_sources
            ),
            "notes": list(boundary.notes),
            "abstentions": [
                _claim_text(abstention) for abstention in boundary.abstentions
            ],
        },
    ]


def briefing_paper_brief_rows(
    briefing: DailyBriefing | None,
) -> list[dict[str, object]]:
    """Render detailed Top-K briefing items as row dictionaries."""

    if briefing is None:
        return []
    rows: list[dict[str, object]] = []
    for item in sorted(briefing.items, key=lambda value: value.rank):
        rows.append(
            {
                "rank": item.rank,
                "paper_id": item.paper_id,
                "title": item.title,
                "score": round(item.score, 4),
                "evidence": item.evidence_source.value,
                "summary": item.summary,
                "problem": _claim_text(item.problem),
                "approach": _claim_text(item.approach),
                "reading_guide": _claim_text(item.reading_guide),
                "contributions": _text_list(
                    [
                        *item.contributions,
                        *[
                            _claim_text(claim)
                            for claim in item.contribution_claims
                        ],
                    ]
                ),
                "methods": _text_list(
                    [
                        *item.methods,
                        *[_claim_text(claim) for claim in item.method_claims],
                    ]
                ),
                "relevance_rationale": item.relevance_rationale,
                "relevance_evidence": _evidence_status_text(
                    item.relevance_evidence
                ),
                "arxiv_url": str(item.arxiv_url),
            }
        )
    return rows


def briefing_trend_signal_rows(
    briefing: DailyBriefing | None,
) -> list[dict[str, object]]:
    """Render candidate-pool trend and hotspot signals."""

    if briefing is None:
        return []
    return [
        {
            "label": signal.label,
            "type": signal.signal_type.value,
            "strength": signal.strength.value,
            "support_count": signal.support_count,
            "candidate_count": (
                signal.candidate_count if signal.candidate_count is not None else ""
            ),
            "top_k_count": (
                signal.top_k_count if signal.top_k_count is not None else ""
            ),
            "query_echo": "yes" if signal.query_echo else "no",
            "evidence": _evidence_sources_text(signal.evidence_sources),
            "summary": signal.summary or "",
            "limitations": "; ".join(signal.limitations),
        }
        for signal in briefing.trend_overview.signals
    ]


def briefing_comparison_rows(
    briefing: DailyBriefing | None,
) -> list[dict[str, object]]:
    """Render Top-K comparison notes."""

    if briefing is None:
        return []
    return [
        {
            "dimension": comparison.dimension,
            "note": comparison.note,
            "paper_ids": ", ".join(comparison.paper_ids),
            "ranks": ", ".join(str(rank) for rank in comparison.ranks),
            "evidence": _evidence_status_text(comparison.evidence),
        }
        for comparison in briefing.top_k_comparisons
    ]


def briefing_reading_priority_rows(
    briefing: DailyBriefing | None,
) -> list[dict[str, object]]:
    """Render goal-aware reading priorities."""

    if briefing is None:
        return []
    return [
        {
            "priority": priority.priority,
            "reading_intent": priority.reading_intent,
            "paper_id": priority.paper_id,
            "rank": priority.rank,
            "reason": priority.reason,
            "evidence": _evidence_status_text(priority.evidence),
        }
        for priority in briefing.reading_priorities
    ]


def result_notice(
    result: SkillResult[Any] | None,
    *,
    empty_message: str,
) -> dict[str, str]:
    """Build a user-facing status banner without assuming Streamlit exists."""

    if result is None:
        return {"kind": "info", "message": empty_message}

    if result.status == SkillStatus.SUCCESS:
        return {
            "kind": "success",
            "message": result.message or "Workflow completed successfully.",
        }
    if result.status == SkillStatus.EMPTY:
        return {
            "kind": "info",
            "message": result.message or empty_message,
        }
    if result.status == SkillStatus.FALLBACK:
        return {
            "kind": "warning",
            "message": _format_result_message(
                result,
                default="Workflow completed with fallback output.",
            ),
        }
    return {
        "kind": "error",
        "message": _format_result_message(
            result,
            default="Workflow failed before producing a complete result.",
        ),
    }


def recommendation_empty_state_message(
    result: SkillResult[RecommendationWorkflow] | None,
) -> str:
    """Explain why the recommendation panel has no visible rows."""

    if result is None:
        return "Run the recommendation workflow to populate ranked papers, briefing output, and a visible trace."

    workflow = result.data
    if workflow is not None and workflow.recommendations:
        return ""
    if result.status == SkillStatus.EMPTY:
        return "No papers matched the current retrieval and ranking filters."
    return "The workflow ran, but it did not produce ranked recommendations."


def main() -> None:
    """Launch the Streamlit demo UI."""

    st = _import_streamlit()
    st.set_page_config(
        page_title="Daily arXiv Demo UI",
        page_icon="📚",
        layout="wide",
    )
    _render_app_css(st)
    _ensure_session_state(st.session_state)
    initial_runtime_error = st.session_state.get("ui_runtime_error")
    _render_sidebar(st, st.session_state)
    _render_hero(st, st.session_state)
    _render_runtime_error(st, st.session_state)
    _render_recommendation_workspace(st, st.session_state)
    _render_feedback_and_explanation(st, st.session_state)
    _render_followup_workspace(st, st.session_state)
    if st.session_state.get("ui_runtime_error") != initial_runtime_error:
        _render_runtime_error(st, st.session_state)


def _render_sidebar(st: Any, state: MutableMapping[str, Any]) -> None:
    config = AppConfig.from_env()
    provider_options = list(PROVIDER_MODES)
    planner_options = [mode.value for mode in QueryPlannerMode]
    with st.sidebar:
        st.markdown("### Runtime")
        state["provider_mode"] = st.selectbox(
            "LLM mode",
            options=provider_options,
            index=_option_index(provider_options, state["provider_mode"]),
            format_func=lambda value: PROVIDER_MODES[value],
            help=(
                "Use fake mode to disable live LLM calls. arXiv retrieval and "
                "seed-paper metadata lookup may still use the network."
            ),
        )
        state["query_planner_mode"] = st.selectbox(
            "Query planner",
            options=planner_options,
            index=_option_index(
                planner_options,
                state["query_planner_mode"],
                default_value=config.query_planner_mode.value,
            ),
            format_func=lambda value: QUERY_PLANNER_LABELS[value],
            help=(
                "Auto uses deterministic planning in fake mode and can use the configured "
                "live provider when available."
            ),
        )
        state["include_debug_trace"] = st.toggle(
            "Show debug trace details",
            value=bool(state["include_debug_trace"]),
            help="When enabled, trace metadata may include raw query variants and planner rationale.",
        )
        state["profile_id"] = st.text_input(
            "Profile ID",
            value=state["profile_id"],
            help="Feedback and seed preferences are stored under this local profile.",
        )
        state["db_path"] = st.text_input(
            "SQLite path",
            value=state["db_path"],
            help="Local storage for papers, seed preferences, feedback, and cached full text.",
        )
        state["top_k"] = st.slider(
            "Top K recommendations",
            min_value=1,
            max_value=10,
            value=int(state["top_k"]),
        )
        state["use_cache"] = st.toggle(
            "Reuse cached retrievals",
            value=bool(state["use_cache"]),
        )
        st.caption(
            "Current env default: "
            f"`{config.llm_provider}` / `{config.llm_model or 'default-model'}`"
        )


def _render_app_css(st: Any) -> None:
    html = getattr(st, "html", None)
    if callable(html):
        html(APP_CSS)
        return
    st.markdown(APP_CSS, unsafe_allow_html=True)


def _render_hero(st: Any, state: Mapping[str, Any]) -> None:
    workflow = _workflow_from_result(state.get("recommendation_result"))
    metrics = {
        "papers": len(workflow.papers) if workflow is not None else 0,
        "recommended": len(workflow.recommendations) if workflow is not None else 0,
        "trace": len(workflow.trace) if workflow is not None else 0,
    }
    st.markdown(
        """
<section class="workspace-header" aria-label="Daily arXiv workspace">
  <div class="workspace-header__grid">
    <div>
      <h1 class="hero-title">Daily arXiv Research Briefing Agent</h1>
      <p class="hero-copy">
        Configure a run, inspect ranked papers, and keep evidence boundaries visible while feedback
        changes the recommendation surface.
      </p>
    </div>
    <p class="workspace-header__note">
      Current session state is local. Live providers are used only when selected in runtime settings.
    </p>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )
    pills = [
        f"Runtime: {PROVIDER_MODES[state['provider_mode']]}",
        f"Profile: {state['profile_id']}",
        f"Store: {state['db_path']}",
        f"Loaded: {metrics['recommended']} recommendation(s)",
    ]
    st.markdown(
        "<div class='pill-row'>"
        + "".join(f"<span class='pill'>{escape(pill)}</span>" for pill in pills)
        + "</div>",
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Retrieved Papers", metrics["papers"])
    metric_columns[1].metric("Recommended Papers", metrics["recommended"])
    metric_columns[2].metric("Trace Steps", metrics["trace"])


def _render_recommendation_workspace(st: Any, state: MutableMapping[str, Any]) -> None:
    _render_section_header(
        st,
        eyebrow="Run setup",
        title="Recommendation Workspace",
        copy=(
            "Set retrieval scope, optional seed papers, and ranking mode before "
            "running the briefing workflow."
        ),
    )
    today = date.today()
    search_options = [mode.value for mode in SearchMode]
    with st.form("recommendation_workflow_form"):
        left, right = st.columns([1.1, 0.9])
        with left:
            topic = st.text_input("Research topic / keywords", value=str(state["topic"]))
            category = st.text_input("arXiv category", value=str(state["category"]))
            seed_input = st.text_area(
                "Seed papers (one per line)",
                value=str(state["seed_input"]),
                height=150,
            )
        with right:
            start_date = st.date_input(
                "Start date",
                value=state["start_date"] or (today - timedelta(days=7)),
            )
            end_date = st.date_input(
                "End date",
                value=state["end_date"] or today,
            )
            search_mode = st.selectbox(
                "Search mode",
                options=search_options,
                index=_option_index(
                    search_options,
                    state["search_mode"],
                    default_value=SearchMode.BROAD.value,
                ),
                format_func=lambda value: SEARCH_MODE_LABELS[value],
                help="Broad expands into multiple planned queries; strict keeps narrower exact-query behavior.",
            )
            recommendation_mode = st.selectbox(
                "Recommendation mode",
                options=list(RECOMMENDATION_MODE_LABELS),
                index=_option_index(
                    list(RECOMMENDATION_MODE_LABELS),
                    state["recommendation_mode"],
                    default_value=RECOMMENDATION_MODE_AUTO,
                ),
                format_func=lambda value: RECOMMENDATION_MODE_LABELS[value],
                help=(
                    "Auto uses semantic ranking for seed-only runs and keeps "
                    "topic plus seed runs deterministic. Semantic seed requests "
                    "fail closed when embeddings are not ready."
                ),
            )
            candidate_pool_size = st.number_input(
                "Candidate pool size",
                min_value=1,
                max_value=500,
                value=int(state["candidate_pool_size"]),
                help="Papers to gather before ranking. Top K controls the final recommendation count.",
            )
            with st.expander("Retrieval budget", expanded=False):
                page_size = st.number_input(
                    "arXiv page size",
                    min_value=1,
                    max_value=100,
                    value=int(state["arxiv_page_size"]),
                )
                max_requests = st.number_input(
                    "Max arXiv requests",
                    min_value=1,
                    max_value=20,
                    value=int(state["arxiv_max_requests"]),
                )
        submitted = st.form_submit_button("Run recommendation workflow", use_container_width=True)

    if submitted:
        state["topic"] = topic
        state["category"] = category
        state["seed_input"] = seed_input
        state["start_date"] = start_date
        state["end_date"] = end_date
        state["search_mode"] = search_mode
        state["recommendation_mode"] = recommendation_mode
        state["candidate_pool_size"] = int(candidate_pool_size)
        state["max_results"] = int(candidate_pool_size)
        state["arxiv_page_size"] = int(page_size)
        state["arxiv_max_requests"] = int(max_requests)
        _run_recommendation_workflow(state)

    _render_semantic_readiness_panel(st, state)

    recommendation_result = state.get("recommendation_result")
    _render_notice(
        st,
        recommendation_result,
        empty_message="No recommendation workflow has been run in this session.",
    )
    _render_semantic_banner(st, recommendation_result)
    seed_result = state.get("seed_result")
    if seed_result is not None:
        _render_notice(
            st,
            seed_result,
            empty_message="No seed papers have been entered yet.",
        )

    workflow = _workflow_from_result(recommendation_result)
    if workflow is not None:
        summary = recommendation_summary_metrics(workflow)
        _render_run_summary(st, summary)
        st.caption(
            f"Candidate pool target: {summary['candidate_pool_size']} | "
            f"Cache hit: {summary['cache_hit']} | "
            f"Seed entries: {_seed_count_from_result(seed_result)}"
        )

    empty_message = recommendation_empty_state_message(recommendation_result)
    if workflow is None:
        if empty_message:
            st.info(empty_message)
        return

    if empty_message:
        st.info(empty_message)
        _render_recommendation_diagnostics(st, workflow)
        if workflow.trace:
            st.subheader("Workflow Trace")
            st.dataframe(
                workflow_trace_rows(workflow.trace),
                use_container_width=True,
                hide_index=True,
            )
        return

    if workflow.briefing is not None:
        _render_daily_briefing(st, workflow.briefing)

    st.subheader("Ranked Recommendations")
    st.dataframe(
        recommendation_rows(workflow.recommendations),
        use_container_width=True,
        hide_index=True,
    )
    _render_recommendation_diagnostics(st, workflow)

    if workflow.trace:
        st.subheader("Workflow Trace")
        st.dataframe(
            workflow_trace_rows(workflow.trace),
            use_container_width=True,
            hide_index=True,
        )


def _render_feedback_and_explanation(st: Any, state: MutableMapping[str, Any]) -> None:
    _render_section_header(
        st,
        eyebrow="Refinement",
        title="Feedback and Paper Explanation",
        copy=(
            "Apply like/dislike signals to the current ranking, then open a focused "
            "explanation for one selected paper."
        ),
    )
    recommendation_result = state.get("recommendation_result")
    workflow = _workflow_from_result(recommendation_result)
    if workflow is None or not workflow.recommendations:
        st.info("Run the recommendation workflow first to unlock feedback refinement and paper explanations.")
        return

    active_recommendations = _active_recommendations(state)
    if not active_recommendations:
        st.info("No current recommendations are available for feedback or explanation.")
        return

    with st.form("feedback_form"):
        st.caption("Leave papers as neutral when they should not change the next ranking.")
        for recommendation in active_recommendations:
            key = f"feedback_choice_{recommendation.paper.paper_id}"
            current_value = state.get(key, NEUTRAL_FEEDBACK)
            state[key] = st.selectbox(
                recommendation.paper.title,
                options=[NEUTRAL_FEEDBACK, FeedbackValue.LIKE.value, FeedbackValue.DISLIKE.value],
                index=[NEUTRAL_FEEDBACK, FeedbackValue.LIKE.value, FeedbackValue.DISLIKE.value].index(
                    current_value
                ),
                key=f"widget_{key}",
            )
        apply_feedback = st.form_submit_button("Apply feedback refinement", use_container_width=True)

    if apply_feedback:
        _run_feedback_refinement(state, workflow)

    feedback_result = state.get("feedback_result")
    _render_notice(
        st,
        feedback_result,
        empty_message="No feedback refinement has been run yet.",
    )
    feedback_workflow = _feedback_workflow_from_result(feedback_result)
    if feedback_workflow is not None and feedback_workflow.recommendations:
        st.dataframe(
            recommendation_rows(feedback_workflow.recommendations),
            use_container_width=True,
            hide_index=True,
        )
        impact_rows = feedback_impact_rows(feedback_workflow.recommendations)
        if impact_rows:
            with st.expander("Feedback impact", expanded=True):
                st.dataframe(
                    impact_rows,
                    use_container_width=True,
                    hide_index=True,
                )
        if feedback_workflow.trace:
            with st.expander("Feedback trace", expanded=False):
                st.dataframe(
                    workflow_trace_rows(feedback_workflow.trace),
                    use_container_width=True,
                    hide_index=True,
                )

    paper_options = {
        f"{recommendation.rank}. {recommendation.paper.title}": recommendation.paper.paper_id
        for recommendation in active_recommendations
    }
    selected_label = st.selectbox(
        "Paper for deep explanation",
        options=list(paper_options),
    )
    selected_mode = st.selectbox(
        "Explanation mode",
        options=[mode.value for mode in ExplanationMode],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    if st.button("Generate paper explanation", use_container_width=True):
        _run_paper_explanation(
            state,
            paper_id=paper_options[selected_label],
            mode=ExplanationMode(selected_mode),
            recommendations=active_recommendations,
        )

    explanation_result = state.get("explanation_result")
    _render_notice(
        st,
        explanation_result,
        empty_message="No paper explanation has been generated yet.",
    )
    explanation_workflow = _paper_explanation_workflow_from_result(explanation_result)
    if explanation_workflow is not None and explanation_workflow.explanation is not None:
        _render_explanation(st, explanation_workflow.explanation)
        if explanation_workflow.trace:
            with st.expander("Explanation trace", expanded=False):
                st.dataframe(
                    workflow_trace_rows(explanation_workflow.trace),
                    use_container_width=True,
                    hide_index=True,
                )


def _render_followup_workspace(st: Any, state: MutableMapping[str, Any]) -> None:
    _render_section_header(
        st,
        eyebrow="Local search",
        title="Follow-up Queries",
        copy=(
            "Search stored papers by topic, category, and date range, with optional "
            "retrieval when the local store is empty."
        ),
    )
    today = date.today()
    with st.form("followup_form"):
        left, right = st.columns(2)
        with left:
            followup_topic = st.text_input(
                "Follow-up topic",
                value=str(state["followup_topic"] or state["topic"]),
            )
            followup_category = st.text_input(
                "Follow-up category",
                value=str(state["followup_category"] or state["category"]),
            )
        with right:
            followup_start = st.date_input(
                "Follow-up start date",
                value=state["followup_start_date"] or (today - timedelta(days=7)),
            )
            followup_end = st.date_input(
                "Follow-up end date",
                value=state["followup_end_date"] or today,
            )
            local_only = st.toggle(
                "Local results only",
                value=bool(state["followup_local_only"]),
            )
        submitted = st.form_submit_button("Run follow-up query", use_container_width=True)

    if submitted:
        state["followup_topic"] = followup_topic
        state["followup_category"] = followup_category
        state["followup_start_date"] = followup_start
        state["followup_end_date"] = followup_end
        state["followup_local_only"] = local_only
        _run_followup_query(state)

    followup_result = state.get("followup_result")
    _render_notice(
        st,
        followup_result,
        empty_message="No follow-up query has been run yet.",
    )
    followup_workflow = _followup_workflow_from_result(followup_result)
    if followup_workflow is None:
        return

    if followup_workflow.recommendations:
        st.dataframe(
            recommendation_rows(followup_workflow.recommendations),
            use_container_width=True,
            hide_index=True,
        )
    elif followup_workflow.papers:
        st.dataframe(
            [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "categories": ", ".join(paper.categories),
                    "published_date": paper.published_date.isoformat()
                    if paper.published_date is not None
                    else "",
                    "arxiv_url": str(paper.arxiv_url),
                }
                for paper in followup_workflow.papers
            ],
            use_container_width=True,
            hide_index=True,
        )

    if followup_workflow.trace:
        with st.expander("Follow-up trace", expanded=False):
            st.dataframe(
                workflow_trace_rows(followup_workflow.trace),
                use_container_width=True,
                hide_index=True,
            )


def _render_section_header(
    st: Any,
    *,
    eyebrow: str,
    title: str,
    copy: str,
) -> None:
    st.markdown(
        f"""
<section class="section-shell" aria-label="{escape(title)}">
  <p class="section-kicker">{escape(eyebrow)}</p>
  <h2 class="section-title">{escape(title)}</h2>
  <p class="section-copy">{escape(copy)}</p>
</section>
        """,
        unsafe_allow_html=True,
    )


def _render_run_summary(st: Any, summary: Mapping[str, object]) -> None:
    items = [
        ("Run ID", summary["run_id"]),
        ("Candidates", summary["candidates_retrieved"]),
        ("Recommendations", summary["recommendations_shown"]),
        ("Planner", summary["planner_source"]),
        ("Fallback", summary["planner_fallback"]),
        ("Cache", summary["cache_status"]),
    ]
    tiles = []
    for label, value in items:
        tiles.append(
            "<div class='summary-tile'>"
            f"<span class='summary-tile__label'>{escape(str(label))}</span>"
            f"<span class='summary-tile__value'>{escape(str(value))}</span>"
            "</div>"
        )
    st.markdown(
        "<div class='run-summary-grid'>" + "".join(tiles) + "</div>",
        unsafe_allow_html=True,
    )


def _render_content_heading(st: Any, *, title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="section-shell" aria-label="{escape(title)}">
  <h2 class="section-title">{escape(title)}</h2>
  <p class="section-copy">{escape(copy)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def _html_text(value: object) -> str:
    return escape(str(value or "")).replace("\n", "<br>")


def _run_recommendation_workflow(state: MutableMapping[str, Any]) -> None:
    try:
        runtime = _build_runtime(
            provider_mode=str(state["provider_mode"]),
            db_path=str(state["db_path"]),
        )
        topic = _normalized_text(state["topic"])
        seed_lines = _seed_lines(str(state["seed_input"]))
        seed_result = None
        seed_preference = None
        if seed_lines:
            seed_result = SeedParsingSkill().build_preference(
                seed_lines,
                profile_id=str(state["profile_id"]),
            )
            if seed_result.data is not None:
                runtime.store.save_seed_preference(seed_result.data)
                seed_preference = seed_result.data

        recommendation_result = runtime.orchestrator.run_recommendation(
            query=_build_retrieval_query(state),
            topic=topic or None,
            seed_preference=seed_preference,
            profile_id=str(state["profile_id"]),
            top_k=int(state["top_k"]),
            use_cache=bool(state["use_cache"]),
            include_debug_trace=bool(state["include_debug_trace"]),
            recommendation_mode=str(state["recommendation_mode"]),
        )
    except Exception as exc:
        _record_action_error(state, exc)
        return

    state["ui_runtime_error"] = None
    state["runtime_label"] = runtime.provider_label
    state["seed_result"] = seed_result
    state["recommendation_result"] = recommendation_result
    state["feedback_result"] = None
    state["explanation_result"] = None


def _run_feedback_refinement(
    state: MutableMapping[str, Any],
    workflow: RecommendationWorkflow,
) -> None:
    try:
        runtime = _build_runtime(
            provider_mode=str(state["provider_mode"]),
            db_path=str(state["db_path"]),
        )
        displayed_recommendations = _active_recommendations(state)
        if not displayed_recommendations:
            state["feedback_result"] = None
            state["ui_runtime_error"] = None
            return
        feedback = []
        for recommendation in displayed_recommendations:
            choice = state.get(
                f"feedback_choice_{recommendation.paper.paper_id}",
                NEUTRAL_FEEDBACK,
            )
            if choice == NEUTRAL_FEEDBACK:
                continue
            feedback.append(
                {
                    "paper_id": recommendation.paper.paper_id,
                    "value": choice,
                }
            )
        if not feedback:
            state["feedback_result"] = None
            state["ui_runtime_error"] = None
            return

        state["feedback_result"] = runtime.orchestrator.run_feedback_refinement(
            displayed_recommendations,
            feedback=feedback,
            papers=workflow.papers,
            profile_id=str(state["profile_id"]),
            recommendation_run_id=workflow.run_id,
        )
    except Exception as exc:
        _record_action_error(state, exc)
        return

    state["ui_runtime_error"] = None


def _run_paper_explanation(
    state: MutableMapping[str, Any],
    *,
    paper_id: str,
    mode: ExplanationMode,
    recommendations: Sequence[Recommendation],
) -> None:
    try:
        runtime = _build_runtime(
            provider_mode=str(state["provider_mode"]),
            db_path=str(state["db_path"]),
        )
        state["explanation_result"] = runtime.orchestrator.run_paper_explanation(
            paper_id,
            mode=mode,
            recommendations=recommendations,
        )
    except Exception as exc:
        _record_action_error(state, exc)
        return

    state["ui_runtime_error"] = None


def _run_followup_query(state: MutableMapping[str, Any]) -> None:
    try:
        runtime = _build_runtime(
            provider_mode=str(state["provider_mode"]),
            db_path=str(state["db_path"]),
        )
        query = FollowupQuery(
            topic=_normalized_text(str(state["followup_topic"])) or None,
            category=_normalized_text(str(state["followup_category"])) or None,
            start_date=state["followup_start_date"],
            end_date=state["followup_end_date"],
            max_results=int(state["max_results"]),
            fetch_if_empty=not bool(state["followup_local_only"]),
        )
        state["followup_result"] = runtime.orchestrator.run_followup_query(
            query,
            top_k=int(state["top_k"]),
        )
    except Exception as exc:
        _record_action_error(state, exc)
        return

    state["ui_runtime_error"] = None


def _build_runtime(*, provider_mode: str, db_path: str) -> RuntimeContext:
    config = AppConfig.from_env()
    resolved_db_path = _normalized_text(db_path) or config.db_path
    store = SQLitePaperStore(resolved_db_path)
    if provider_mode == DEFAULT_PROVIDER_MODE:
        provider = FakeLLMProvider()
        provider_label = DEFAULT_PROVIDER_MODE
    else:
        provider = create_llm_provider(config)
        provider_label = f"{config.llm_provider}:{config.llm_model or 'default-model'}"
    orchestrator = DailyArxivAgentOrchestrator(store=store, provider=provider)
    return RuntimeContext(
        store=store,
        orchestrator=orchestrator,
        provider_label=provider_label,
        db_path=resolved_db_path,
    )


def _build_retrieval_query(state: Mapping[str, Any]) -> RetrievalQuery:
    return RetrievalQuery(
        topic=_normalized_text(str(state["topic"])) or None,
        category=_normalized_text(str(state["category"])) or None,
        start_date=state["start_date"],
        end_date=state["end_date"],
        max_results=int(state["max_results"]),
        search_mode=SearchMode(str(state["search_mode"])),
        candidate_pool_size=int(state["candidate_pool_size"]),
        page_size=int(state["arxiv_page_size"]),
        max_requests=int(state["arxiv_max_requests"]),
        query_planner_mode=QueryPlannerMode(str(state["query_planner_mode"])),
    )


def _render_semantic_readiness_panel(
    st: Any,
    state: Mapping[str, Any],
) -> None:
    seed_preference = _seed_preference_from_result(state.get("seed_result"))
    readiness = _semantic_readiness_preview(state, seed_preference)
    rows = semantic_readiness_panel_rows(
        readiness,
        cache_path=str(state.get("db_path") or ""),
        recommendation_mode=str(state.get("recommendation_mode", RECOMMENDATION_MODE_AUTO)),
        seed_count=len(_seed_lines(str(state.get("seed_input") or ""))),
    )
    with st.expander("Semantic readiness", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_semantic_banner(
    st: Any,
    result: SkillResult[RecommendationWorkflow] | None,
) -> None:
    notice = semantic_error_notice(result)
    if notice is None:
        return
    renderer = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }.get(notice["kind"], st.info)
    renderer(notice["message"])


def _render_recommendation_diagnostics(
    st: Any,
    workflow: RecommendationWorkflow,
) -> None:
    _render_content_heading(
        st,
        title="Recommendation Diagnostics",
        copy="Provider state, semantic signals, score composition, and feedback movement.",
    )
    provider_tab, similarity_tab, score_tab, feedback_tab = st.tabs(
        [
            "Provider / Cache / Retrieval",
            "Seed Similarity",
            "Score Composition",
            "Feedback Impact",
        ]
    )
    with provider_tab:
        rows = semantic_provider_cache_rows(workflow.trace)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No semantic provider, cache, or retrieval diagnostics are available.")
    with similarity_tab:
        rows = semantic_similarity_rows(workflow.recommendations)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No seed-candidate similarity diagnostics are available for this run.")
    with score_tab:
        rows = score_composition_rows(workflow.recommendations)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            why_rows = recommendation_why_rows(workflow.recommendations)
            if why_rows:
                st.dataframe(why_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No structured score composition is available for this run.")
    with feedback_tab:
        rows = feedback_impact_rows(workflow.recommendations)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No feedback movement has been applied to the current recommendations.")


def _render_daily_briefing(st: Any, briefing: DailyBriefing) -> None:
    sections = {section["key"]: section for section in enhanced_briefing_sections(briefing)}

    executive = sections["executive_summary"]
    st.markdown(
        f"""
<section class="briefing-callout" aria-label="{escape(str(executive['title']))}">
  <h3>{escape(str(executive["title"]))}</h3>
  <p>{_html_text(executive["body"])}</p>
</section>
        """,
        unsafe_allow_html=True,
    )

    guide = sections["top_k_reading_guide"]
    _render_content_heading(
        st,
        title=str(guide["title"]),
        copy="Ranked papers with evidence labels and reading guidance.",
    )
    summary_rows = guide["summary_rows"]
    if summary_rows:
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
    for brief in guide["paper_briefs"]:
        with st.expander(
            f"Rank {brief['rank']}: {brief['title']}",
            expanded=brief["rank"] == 1,
        ):
            st.caption(
                f"Paper ID: {brief['paper_id']} | Evidence: {brief['evidence']} | "
                f"Score: {brief['score']}"
            )
            st.write(brief["summary"])
            _render_labeled_text(st, "Problem", brief["problem"])
            _render_labeled_text(st, "Approach", brief["approach"])
            _render_labeled_text(st, "Reading Guide", brief["reading_guide"])
            _render_labeled_text(st, "Contributions", brief["contributions"])
            _render_labeled_text(st, "Methods", brief["methods"])
            _render_labeled_text(
                st,
                "Relevance",
                f"{brief['relevance_rationale']} {brief['relevance_evidence']}",
            )

    boundary = sections["evidence_boundary"]
    st.markdown(
        f"""
<section class="evidence-panel" aria-label="{escape(str(boundary['title']))}">
  <h3>{escape(str(boundary["title"]))}</h3>
  <p class="metadata-line">
    Full text used: {escape(str(boundary["full_text_used"]))} | Evidence: {escape(str(boundary["evidence_sources"]))} | Unavailable: {escape(str(boundary["unavailable_sources"]))}
  </p>
</section>
        """,
        unsafe_allow_html=True,
    )
    _render_list_section(st, "Boundary Notes", boundary["notes"])
    _render_list_section(st, "Explicit Abstentions", boundary["abstentions"])


def _render_explanation(st: Any, explanation: PaperDeepExplanation) -> None:
    st.markdown(
        f"""
<section class="explanation-panel" aria-label="{escape(explanation.title)}">
  <h3>{escape(explanation.title)}</h3>
  <p>{_html_text(explanation.summary)}</p>
  <p class="metadata-line">Evidence: {escape(explanation.evidence_source.value)} | Note: {escape(explanation.evidence_note)}</p>
</section>
        """,
        unsafe_allow_html=True,
    )
    if explanation.method is not None:
        st.markdown("**Problem**")
        st.write(explanation.method.problem)
        st.markdown("**Method Overview**")
        st.write(explanation.method.method_overview)
        st.markdown("**Core Workflow**")
        for item in explanation.method.core_workflow:
            st.markdown(f"- {item}")
        st.markdown("**Inputs / Outputs**")
        for item in explanation.method.inputs_outputs:
            st.markdown(f"- {item}")
        st.markdown("**Innovation**")
        st.write(explanation.method.innovation)
    if explanation.experiment is not None:
        _render_list_section(st, "Datasets", explanation.experiment.datasets)
        _render_list_section(st, "Baselines", explanation.experiment.baselines)
        _render_list_section(st, "Metrics", explanation.experiment.metrics)
        st.markdown("**Experimental Setup**")
        st.write(explanation.experiment.experimental_setup)
        _render_list_section(st, "Conclusions", explanation.experiment.conclusions)
    if explanation.limitations is not None:
        _render_list_section(st, "Stated Limitations", explanation.limitations.stated_limitations)
        _render_list_section(st, "Assumptions", explanation.limitations.assumptions)
        _render_list_section(st, "Missing Validation", explanation.limitations.missing_validation)
        _render_list_section(st, "Risks", explanation.limitations.risks)


def _render_list_section(st: Any, title: str, items: Sequence[str]) -> None:
    st.markdown(f"**{title}**")
    if not items:
        st.write("No supported evidence was available for this section.")
        return
    for item in items:
        st.markdown(f"- {item}")


def _render_labeled_text(st: Any, title: str, value: object) -> None:
    text = str(value or "").strip()
    if not text:
        return
    st.markdown(f"**{title}**")
    st.write(text)


def _render_notice(
    st: Any,
    result: SkillResult[Any] | None,
    *,
    empty_message: str,
) -> None:
    notice = result_notice(result, empty_message=empty_message)
    renderer = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }[notice["kind"]]
    renderer(notice["message"])


def _render_runtime_error(st: Any, state: Mapping[str, Any]) -> None:
    error_message = state.get("ui_runtime_error")
    if error_message:
        st.error(error_message)


def _ensure_session_state(state: MutableMapping[str, Any]) -> None:
    today = date.today()
    config = AppConfig.from_env()
    defaults = {
        "provider_mode": DEFAULT_PROVIDER_MODE,
        "profile_id": DEFAULT_PROFILE_ID,
        "db_path": config.db_path,
        "runtime_label": DEFAULT_PROVIDER_MODE,
        "topic": "agent briefing",
        "category": "cs.LG",
        "start_date": today - timedelta(days=7),
        "end_date": today,
        "max_results": DEFAULT_MAX_RESULTS,
        "search_mode": config.search_mode.value,
        "candidate_pool_size": config.candidate_pool_size,
        "arxiv_page_size": config.arxiv_page_size,
        "arxiv_max_requests": config.arxiv_max_requests_per_search,
        "query_planner_mode": config.query_planner_mode.value,
        "recommendation_mode": RECOMMENDATION_MODE_AUTO,
        "include_debug_trace": False,
        "top_k": DEFAULT_TOP_K,
        "use_cache": True,
        "seed_input": "",
        "recommendation_result": None,
        "seed_result": None,
        "feedback_result": None,
        "explanation_result": None,
        "followup_topic": "",
        "followup_category": "",
        "followup_start_date": today - timedelta(days=7),
        "followup_end_date": today,
        "followup_local_only": False,
        "followup_result": None,
        "ui_runtime_error": None,
    }
    for key, value in defaults.items():
        state.setdefault(key, value)


def _active_recommendations(state: Mapping[str, Any]) -> list[Recommendation]:
    feedback_workflow = _feedback_workflow_from_result(state.get("feedback_result"))
    if feedback_workflow is not None and feedback_workflow.recommendations:
        return feedback_workflow.recommendations
    workflow = _workflow_from_result(state.get("recommendation_result"))
    if workflow is None:
        return []
    return workflow.recommendations


def _workflow_from_result(
    result: SkillResult[RecommendationWorkflow] | None,
) -> RecommendationWorkflow | None:
    return result.data if result is not None else None


def _feedback_workflow_from_result(
    result: SkillResult[FeedbackWorkflow] | None,
) -> FeedbackWorkflow | None:
    return result.data if result is not None else None


def _followup_workflow_from_result(result: SkillResult[Any] | None):
    return result.data if result is not None else None


def _paper_explanation_workflow_from_result(
    result: SkillResult[PaperExplanationWorkflow] | None,
) -> PaperExplanationWorkflow | None:
    return result.data if result is not None else None


def _seed_count_from_result(result: SkillResult[Any] | None) -> int:
    if result is None or result.data is None:
        return 0
    seeds = getattr(result.data, "seeds", None)
    if not seeds:
        return 0
    return len(seeds)


def _seed_preference_from_result(
    result: SkillResult[Any] | None,
) -> SeedPreference | None:
    if result is None:
        return None
    data = result.data
    if isinstance(data, SeedPreference):
        return data
    return None


def _semantic_readiness_preview(
    state: Mapping[str, Any],
    seed_preference: SeedPreference | None,
) -> dict[str, Any]:
    config = AppConfig.from_env()
    db_path = _normalized_text(str(state.get("db_path") or "")) or config.db_path
    try:
        store = SQLitePaperStore(db_path)
        readiness = SemanticSeedRankingSkill(
            store=store,
            config=config,
        ).check_readiness(seed_preference)
    except Exception as exc:  # pragma: no cover - defensive UI guard.
        return {
            "provider": config.embedding_provider,
            "provider_mode": (
                "fake"
                if config.embedding_provider.strip().lower() == "fake"
                else "live"
            ),
            "provider_label": f"{config.embedding_provider}:{config.embedding_model}",
            "credential_status": "unknown",
            "model": config.embedding_model,
            "endpoint_safety": "unknown",
            "cache_enabled": config.embedding_cache_enabled,
            "seed_quality": "unknown",
            "can_run": False,
            "error_code": "semantic_readiness_preview_failed",
            "warnings": [str(exc)],
        }
    return readiness.model_dump(mode="json")


def _option_index(
    options: Sequence[str],
    value: Any,
    *,
    default_value: str | None = None,
) -> int:
    selected = str(value)
    if selected in options:
        return options.index(selected)
    if default_value is not None and default_value in options:
        return options.index(default_value)
    return 0


def _claim_text(claim: Any) -> str:
    if claim is None:
        return ""
    claim_text = getattr(claim, "claim", None)
    evidence = getattr(claim, "evidence", None)
    if claim_text:
        evidence_text = _evidence_status_text(evidence)
        if evidence_text:
            return f"{claim_text} ({evidence_text})"
        return str(claim_text)
    return _evidence_status_text(evidence)


def _evidence_status_text(evidence: Any) -> str:
    if evidence is None:
        return ""
    status = getattr(getattr(evidence, "status", None), "value", None)
    parts = [str(status or getattr(evidence, "status", ""))]
    sources = _evidence_sources_text(getattr(evidence, "sources", []))
    if sources:
        parts.append(f"sources: {sources}")
    note = getattr(evidence, "note", None)
    if note:
        parts.append(str(note))
    abstention = getattr(evidence, "abstention_reason", None)
    if abstention:
        parts.append(str(abstention))
    return "; ".join(part for part in parts if part)


def _evidence_sources_text(sources: Sequence[Any]) -> str:
    return ", ".join(
        str(getattr(source, "value", source)) for source in sources if source
    )


def _text_list(values: Sequence[str]) -> str:
    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split())
        if not normalized or normalized in seen:
            continue
        texts.append(normalized)
        seen.add(normalized)
    return "; ".join(texts)


def _trend_status_note(status: str, limitations: Sequence[str]) -> str:
    if limitations:
        return " ".join(limitations)
    if status == "not_assessed":
        return "Candidate-pool trend analysis was not assessed for this briefing."
    if status == "insufficient_candidate_data":
        return "Candidate pool was too small for broader trend claims."
    return f"Candidate-pool trend analysis status: {status}."


def _metadata_for_skill(
    trace: Sequence[WorkflowTraceStep],
    skill: str,
) -> dict[str, Any]:
    for step in trace:
        if step.skill == skill:
            return step.metadata
    return {}


def _semantic_recommendation_row_fields(
    recommendation: Recommendation,
) -> dict[str, object]:
    breakdown = recommendation.score_breakdown
    if breakdown is None:
        return {}
    has_semantic = bool(
        breakdown.semantic_similarities
        or abs(breakdown.semantic_seed) > 0.0001
    )
    has_feedback = (
        recommendation.score_delta is not None
        or abs(breakdown.feedback) > 0.0001
    )
    if not has_semantic and not has_feedback:
        return {}

    fields: dict[str, object] = {}
    if has_semantic:
        top = _top_semantic_similarity(breakdown.semantic_similarities)
        fields.update(
            {
                "semantic_score": round(breakdown.semantic_seed, 4),
                "semantic_similarity": (
                    round(top.similarity, 4)
                    if top is not None
                    else round(breakdown.evidence_score, 4)
                ),
                "top_matching_seed": (
                    _similarity_source_label(top) if top is not None else ""
                ),
                "evidence_status": _semantic_evidence_status(breakdown),
            }
        )
    if has_feedback:
        fields["feedback_delta"] = round(
            recommendation.score_delta
            if recommendation.score_delta is not None
            else breakdown.feedback,
            4,
        )
    return fields


def _top_semantic_similarity(similarities: Sequence[Any]) -> Any | None:
    if not similarities:
        return None
    return max(similarities, key=lambda item: float(getattr(item, "similarity", 0.0)))


def _similarity_source_label(detail: Any | None) -> str:
    if detail is None:
        return ""
    title = getattr(detail, "source_title", None)
    if title:
        return str(title)
    return str(getattr(detail, "source_id", ""))


def _semantic_evidence_status(breakdown: Any) -> str:
    if getattr(breakdown, "fallback", False):
        return "low_evidence"
    evidence_score = float(getattr(breakdown, "evidence_score", 0.0) or 0.0)
    if evidence_score >= 0.75:
        return "high_semantic_evidence"
    if evidence_score >= 0.5:
        return "medium_semantic_evidence"
    if evidence_score >= 0.25:
        return "low_semantic_evidence"
    if abs(float(getattr(breakdown, "semantic_seed", 0.0) or 0.0)) > 0.0001:
        return "minimal_semantic_evidence"
    return "no_semantic_evidence"


def _semantic_provider_summary_for_step(step: WorkflowTraceStep) -> str:
    metadata = step.metadata
    provider = metadata.get("semantic_provider")
    if isinstance(provider, Mapping):
        label = provider.get("provider_label") or provider.get("provider")
        mode = provider.get("provider_mode")
        if label and mode:
            return f"{label} ({mode})"
        return _metadata_text(label)
    label = metadata.get("provider_label") or metadata.get("provider")
    mode = metadata.get("provider_mode")
    if label and mode:
        return f"{label} ({mode})"
    return _metadata_text(label)


def _semantic_model_for_step(step: WorkflowTraceStep) -> str:
    provider = step.metadata.get("semantic_provider")
    if isinstance(provider, Mapping):
        return _metadata_text(provider.get("model"))
    return _metadata_text(step.metadata.get("model"))


def _embedding_cache_summary_for_step(step: WorkflowTraceStep) -> str:
    cache = step.metadata.get("embedding_cache")
    if isinstance(cache, Mapping):
        enabled = "enabled" if cache.get("enabled", True) else "disabled"
        return (
            f"{enabled}; hits={_metadata_text(cache.get('hits'))}; "
            f"misses={_metadata_text(cache.get('misses'))}; "
            f"writes={_metadata_text(cache.get('writes'))}"
        )
    if "cache_enabled" in step.metadata:
        return "enabled" if step.metadata.get("cache_enabled") else "disabled"
    return ""


def _semantic_state_for_step(step: WorkflowTraceStep) -> str:
    metadata = step.metadata
    if step.error_code:
        return step.error_code
    semantic_error = metadata.get("semantic_error")
    if isinstance(semantic_error, Mapping):
        return _metadata_text(
            semantic_error.get("failure_reason")
            or semantic_error.get("error_code")
        )
    if "can_run" in metadata:
        return "ready" if metadata.get("can_run") else _metadata_text(
            metadata.get("error_code") or "not_ready"
        )
    if metadata.get("candidate_pool_diagnostic"):
        return _metadata_text(metadata.get("candidate_pool_diagnostic"))
    fallback_count = _metadata_int(metadata.get("fallback_count"), default=0)
    qualifying_count = _metadata_int(metadata.get("qualifying_count"), default=0)
    if step.skill == "ranking" and fallback_count and not qualifying_count:
        return "low_semantic_evidence"
    return ""


def _first_semantic_trace_error_code(
    trace: Sequence[WorkflowTraceStep],
) -> str | None:
    for step in trace:
        if step.error_code and step.error_code.startswith("semantic_"):
            return step.error_code
        semantic_error = step.metadata.get("semantic_error")
        if isinstance(semantic_error, Mapping):
            code = semantic_error.get("error_code") or semantic_error.get(
                "failure_reason"
            )
            if code:
                return str(code)
    return None


def _semantic_error_notice_for_code(code: str) -> dict[str, str]:
    if code == "semantic_embedding_credentials_missing":
        message = (
            "Semantic recommendations were withheld because embedding credentials "
            "are missing. Set EMBEDDING_API_KEY, or explicitly opt in to "
            "EMBEDDING_REUSE_OPENAI_API_KEY."
        )
    elif code == "semantic_embedding_endpoint_unsafe":
        message = (
            "Semantic recommendations were withheld because the embedding endpoint "
            "is not marked safe. Use the default OpenAI endpoint or configure a "
            "safe HTTPS embedding base URL."
        )
    elif code == "semantic_seed_quality_error":
        message = (
            "Semantic recommendations were withheld because the seed metadata does "
            "not include usable title, abstract, or category text. Add richer seed "
            "papers or use deterministic mode."
        )
    elif code in {
        "semantic_embedding_provider_failed",
        "embedding_provider_failed",
    }:
        message = (
            "Semantic recommendations were withheld because the embedding provider "
            "failed or timed out. Check provider availability, timeout settings, "
            "and credentials."
        )
    elif code == "semantic_embedding_configuration_failed":
        message = (
            "Semantic recommendations were withheld because embedding provider "
            "configuration is invalid. Check provider, model, dimensions, and "
            "cache settings."
        )
    elif code == "semantic_feedback_refinement_failed":
        message = (
            "Semantic feedback refinement failed before score movement could be "
            "applied. Original recommendations should be preserved when available."
        )
    else:
        message = (
            "Semantic recommendations were withheld by a fail-closed semantic "
            f"state: {code}."
        )
    return {"kind": "error", "message": message}


def _semantic_retrieval_state(trace: Sequence[WorkflowTraceStep]) -> str | None:
    for step in trace:
        if step.skill != "arxiv_retrieval":
            continue
        metadata = step.metadata
        if (
            metadata.get("seed_derived_retrieval")
            and _metadata_int(metadata.get("candidate_count"), default=0) == 0
        ):
            return "no_seed_candidates"
    return None


def _all_semantic_recommendations_low_evidence(
    recommendations: Sequence[Recommendation],
) -> bool:
    semantic_recommendations = [
        recommendation
        for recommendation in recommendations
        if recommendation.score_breakdown is not None
        and (
            recommendation.score_breakdown.semantic_similarities
            or abs(recommendation.score_breakdown.semantic_seed) > 0.0001
        )
    ]
    if not semantic_recommendations:
        return False
    return all(
        recommendation.score_breakdown is not None
        and recommendation.score_breakdown.fallback
        for recommendation in semantic_recommendations
    )


def _has_feedback_movement(recommendation: Recommendation) -> bool:
    return (
        recommendation.previous_rank is not None
        or recommendation.rank_delta is not None
        or recommendation.score_delta is not None
        or recommendation.refinement_status is not None
        or recommendation.feedback_error is not None
    )


def _feedback_impact_row(
    recommendation: Recommendation,
    influence: Any | None,
) -> dict[str, object]:
    return {
        "paper_id": recommendation.paper.paper_id,
        "title": recommendation.paper.title,
        "original_rank": recommendation.previous_rank or "",
        "refined_rank": recommendation.rank,
        "rank_delta": recommendation.rank_delta if recommendation.rank_delta is not None else "",
        "score_delta": (
            round(recommendation.score_delta, 4)
            if recommendation.score_delta is not None
            else ""
        ),
        "source_feedback_paper": (
            getattr(influence, "source_paper_id", "") if influence is not None else ""
        ),
        "source_title": (
            getattr(influence, "source_title", "") if influence is not None else ""
        ),
        "similarity": (
            round(getattr(influence, "similarity", 0.0), 4)
            if influence is not None
            else ""
        ),
        "signed_influence": (
            round(getattr(influence, "signed_score_delta", 0.0), 4)
            if influence is not None
            else ""
        ),
        "feedback_value": (
            getattr(getattr(influence, "value", ""), "value", "")
            if influence is not None
            else ""
        ),
        "status": _feedback_status_text(recommendation, influence),
    }


def _feedback_status_text(recommendation: Recommendation, influence: Any | None) -> str:
    if influence is not None:
        status = getattr(influence, "refinement_status", "")
        return str(getattr(status, "value", status))
    if recommendation.feedback_error is not None:
        return f"failed: {recommendation.feedback_error.code}"
    status = recommendation.refinement_status
    if status is not None:
        return status.value
    return "movement"


def _recommendation_mode_label(value: str) -> str:
    normalized = _normalize_recommendation_mode(value)
    return RECOMMENDATION_MODE_LABELS.get(normalized, RECOMMENDATION_MODE_LABELS[RECOMMENDATION_MODE_AUTO])


def _recommendation_mode_help(value: str) -> str:
    normalized = _normalize_recommendation_mode(value)
    if normalized == RECOMMENDATION_MODE_DETERMINISTIC:
        return "semantic ranking disabled for this run"
    if normalized == RECOMMENDATION_MODE_SEMANTIC_SEED:
        return "semantic seed ranking explicitly requested"
    return "seed-only uses semantic; topic plus seed stays deterministic"


def _normalize_recommendation_mode(value: str | None) -> str:
    normalized = (value or RECOMMENDATION_MODE_AUTO).strip().lower().replace("-", "_")
    if normalized == "semantic":
        normalized = RECOMMENDATION_MODE_SEMANTIC_SEED
    if normalized not in RECOMMENDATION_MODE_LABELS:
        return RECOMMENDATION_MODE_AUTO
    return normalized


def _planner_source_for_step(step: WorkflowTraceStep) -> str:
    if step.skill == "query_planning":
        return _metadata_text(step.metadata.get("source"))
    return _metadata_text(step.metadata.get("planner_source"))


def _planner_fallback_for_step(step: WorkflowTraceStep) -> str:
    if step.skill == "query_planning":
        fallback = step.metadata.get("fallback")
    else:
        fallback = step.metadata.get("planner_fallback")
    if fallback is None:
        return ""
    return "yes" if bool(fallback) else "no"


def _cache_summary_for_step(step: WorkflowTraceStep) -> str:
    if step.skill != "arxiv_retrieval":
        return ""
    status = _metadata_text(step.metadata.get("cache_status"))
    hit = step.metadata.get("cache_hit")
    if hit is True:
        return f"hit/{status}" if status else "hit"
    if hit is False:
        return f"miss/{status}" if status else "miss"
    return status


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _metadata_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _seed_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _format_trace_error(step: WorkflowTraceStep) -> str:
    if step.error_code is None:
        return ""
    if step.error_message is None:
        return step.error_code
    return f"{step.error_code}: {step.error_message}"


def _format_result_message(
    result: SkillResult[Any],
    *,
    default: str,
) -> str:
    parts = [result.message or default]
    if result.error is not None:
        parts.append(f"{result.error.code}: {result.error.message}")
    return " ".join(part for part in parts if part)


def _record_action_error(state: MutableMapping[str, Any], exc: Exception) -> None:
    state["ui_runtime_error"] = f"UI action failed: {exc}"


def _import_streamlit() -> Any:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on optional UI extra.
        raise RuntimeError(
            "Streamlit is not installed. Install the UI extras with "
            "`python -m pip install -e .[ui]` or `python -m pip install -e .[all]`."
        ) from exc
    return st


if __name__ == "__main__":  # pragma: no cover - interactive entry point.
    main()
