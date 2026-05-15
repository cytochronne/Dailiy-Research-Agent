"""Command-line entry points for the local Daily arXiv Agent demo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
from typing import Any

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    QueryPlannerMode,
    RetrievalQuery,
    SearchMode,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.evaluation.real_arxiv import (
    DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
    DEFAULT_REAL_ARXIV_LABELS_PATH,
    evaluate_frozen_real_arxiv,
    fetch_real_arxiv_candidates,
    format_real_arxiv_report_markdown,
    write_label_template,
)
from daily_arxiv_agent.orchestrator import (
    DailyArxivAgentOrchestrator,
    RECOMMENDATION_MODE_AUTO,
    RECOMMENDATION_MODE_DETERMINISTIC,
    RECOMMENDATION_MODE_SEMANTIC_SEED,
)
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.skills.followup import FollowupQuery
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


class _FixtureResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FixtureClient:
    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], timeout: float) -> _FixtureResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _FixtureResponse(self.fixture_path.read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        result = _run_demo(args)
    elif args.command == "followup":
        result = _run_followup(args)
    elif args.command == "embedding-cache":
        result = _run_embedding_cache(args)
    elif args.command == "real-eval":
        result = _run_real_eval(args)
    else:  # pragma: no cover - argparse prevents this branch.
        parser.error(f"Unknown command: {args.command}")
    if (
        args.command == "real-eval"
        and getattr(args, "real_eval_command", None) == "run"
        and getattr(args, "output_format", "json") == "markdown"
        and result.data is not None
    ):
        print(format_real_arxiv_report_markdown(result.data))
    elif getattr(args, "output_format", "json") == "briefing":
        print(compact_briefing_output(result))
    else:
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return _exit_code_for_status(result.status)


def _build_parser() -> argparse.ArgumentParser:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(
        prog="daily-arxiv-agent",
        description="Run fixture-backed Daily arXiv Agent workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run recommendation workflow end to end.")
    _add_common_filters(demo)
    demo.add_argument("--fixture", type=Path, help="arXiv Atom XML fixture path.")
    demo.add_argument("--db-path", default=config.db_path)
    demo.add_argument("--profile-id", default="default")
    demo.add_argument("--top-k", type=int, default=5)
    demo.add_argument(
        "--seed",
        action="append",
        default=[],
        help=(
            "Seed paper input. Repeat for multiple seeds. Accepts arXiv IDs, "
            "arXiv URLs, or title text."
        ),
    )
    demo.add_argument(
        "--seed-file",
        action="append",
        default=[],
        type=_existing_file,
        help="Path to a text file containing one seed paper input per line.",
    )
    demo.add_argument(
        "--recommendation-mode",
        choices=[
            RECOMMENDATION_MODE_AUTO,
            RECOMMENDATION_MODE_DETERMINISTIC,
            RECOMMENDATION_MODE_SEMANTIC_SEED,
            "semantic",
            "semantic-seed",
        ],
        default=RECOMMENDATION_MODE_AUTO,
        help=(
            "auto uses semantic ranking for seed-only runs; deterministic disables "
            "semantic ranking; semantic-seed explicitly requests semantic ranking."
        ),
    )
    demo.add_argument(
        "--candidate-pool-size",
        type=int,
        default=config.candidate_pool_size,
        help=(
            "Number of retrieved candidates to collect before ranking. "
            "Top K remains the final recommendation count."
        ),
    )
    demo.add_argument(
        "--search-mode",
        choices=[mode.value for mode in SearchMode],
        default=config.search_mode.value,
        help="Use broad multi-variant search or strict compatibility search.",
    )
    demo.add_argument(
        "--query-planner-mode",
        choices=[mode.value for mode in QueryPlannerMode],
        default=config.query_planner_mode.value,
        help="Requested query-planning strategy for search expansion.",
    )
    demo.add_argument(
        "--page-size",
        type=int,
        default=config.arxiv_page_size,
        help="Maximum papers requested per arXiv API page.",
    )
    demo.add_argument(
        "--max-requests",
        type=int,
        default=config.arxiv_max_requests_per_search,
        help="Maximum arXiv API requests per recommendation run.",
    )
    demo.add_argument(
        "--debug-trace",
        action="store_true",
        help="Include raw query variants and planner rationale in trace metadata.",
    )
    demo.add_argument(
        "--format",
        choices=["json", "briefing"],
        default="json",
        dest="output_format",
        help=(
            "Output full workflow JSON for automation or a compact human-readable "
            "daily briefing."
        ),
    )
    demo.add_argument("--no-cache", action="store_true")
    demo.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Disable SQLite embedding-cache reads and writes for this demo run.",
    )

    followup = subparsers.add_parser("followup", help="Run a local-first follow-up query.")
    _add_common_filters(followup)
    followup.add_argument("--fixture", type=Path, help="Optional arXiv Atom XML fixture path for empty local results.")
    followup.add_argument("--db-path", default=config.db_path)
    followup.add_argument("--top-k", type=int, default=5)
    followup.add_argument("--local-only", action="store_true")

    embedding_cache = subparsers.add_parser(
        "embedding-cache",
        help="Manage the local semantic embedding cache.",
    )
    embedding_cache_subparsers = embedding_cache.add_subparsers(
        dest="cache_command",
        required=True,
    )
    clear_cache = embedding_cache_subparsers.add_parser(
        "clear",
        help="Delete cached embedding vectors without deleting papers or feedback.",
    )
    clear_cache.add_argument("--db-path", default=config.db_path)
    clear_cache.add_argument("--scope", choices=["global", "profile"])
    clear_cache.add_argument("--profile-id")

    real_eval = subparsers.add_parser(
        "real-eval",
        help="Run or refresh the frozen small real-arXiv evaluation.",
    )
    real_eval_subparsers = real_eval.add_subparsers(
        dest="real_eval_command",
        required=True,
    )
    run_real_eval = real_eval_subparsers.add_parser(
        "run",
        help="Evaluate frozen real-arXiv candidates and labels.",
    )
    _add_real_eval_paths(run_real_eval)
    run_real_eval.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        dest="output_format",
        help="Output full JSON or a compact Markdown metrics table.",
    )
    run_real_eval.add_argument(
        "--semantic-provider",
        choices=["fake", "live", "none"],
        default="none",
        help=(
            "Embedding provider for semantic_agent. none keeps the default "
            "offline comparison to lexical rankers; fake is deterministic; "
            "live uses EMBEDDING_* environment configuration."
        ),
    )
    label_template = real_eval_subparsers.add_parser(
        "label-template",
        help="Write a JSONL human-label template from frozen candidates.",
    )
    _add_real_eval_paths(label_template)
    fetch_candidates = real_eval_subparsers.add_parser(
        "fetch-candidates",
        help="Refresh frozen candidates from live arXiv.",
    )
    fetch_candidates.add_argument(
        "--candidates-path",
        type=Path,
        default=DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
    )
    fetch_candidates.add_argument(
        "--store-path",
        type=Path,
        default=Path("data/daily_arxiv_eval.sqlite3"),
    )
    fetch_candidates.add_argument("--request-delay-seconds", type=float, default=3.0)
    return parser


def _add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", default="agent briefing")
    parser.add_argument("--category")
    parser.add_argument("--start-date", type=_parse_date)
    parser.add_argument("--end-date", type=_parse_date)
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help=(
            "Compatibility result limit for strict/follow-up workflows. "
            "For demo candidate gathering, prefer --candidate-pool-size."
        ),
    )


def _add_real_eval_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_REAL_ARXIV_LABELS_PATH,
    )


def _run_demo(args: argparse.Namespace) -> Any:
    orchestrator = _build_orchestrator(args)
    query = RetrievalQuery(
        topic=_optional_text(args.topic),
        category=args.category,
        start_date=args.start_date,
        end_date=args.end_date,
        max_results=args.max_results,
        search_mode=SearchMode(args.search_mode),
        candidate_pool_size=args.candidate_pool_size,
        page_size=args.page_size,
        max_requests=args.max_requests,
        query_planner_mode=QueryPlannerMode(args.query_planner_mode),
    )
    seed_result = _seed_preference_from_args(args)
    if seed_result is not None and seed_result.data is None:
        return _seed_error_result(seed_result)
    seed_preference = seed_result.data if seed_result is not None else None
    if seed_preference is not None:
        orchestrator.store.save_seed_preference(seed_preference)
    return orchestrator.run_recommendation(
        query,
        seed_preference=seed_preference,
        profile_id=args.profile_id,
        top_k=args.top_k,
        use_cache=not args.no_cache,
        include_debug_trace=args.debug_trace,
        recommendation_mode=args.recommendation_mode,
    )


def _run_followup(args: argparse.Namespace) -> Any:
    orchestrator = _build_orchestrator(args)
    query = FollowupQuery(
        topic=_optional_text(args.topic),
        category=args.category,
        start_date=args.start_date,
        end_date=args.end_date,
        max_results=args.max_results,
        fetch_if_empty=not args.local_only,
    )
    return orchestrator.run_followup_query(query, top_k=args.top_k)


def _run_embedding_cache(args: argparse.Namespace) -> SkillResult[dict[str, int]]:
    store = SQLitePaperStore(args.db_path)
    deleted_rows = store.clear_embedding_cache(
        cache_scope=args.scope,
        profile_id=args.profile_id,
    )
    return SkillResult[dict[str, int]](
        status=SkillStatus.SUCCESS,
        data={"deleted_embedding_cache_rows": deleted_rows},
        message=f"Deleted {deleted_rows} embedding cache row(s).",
        metadata={
            "db_path": str(args.db_path),
            "cache_scope": args.scope,
            "profile_id": args.profile_id,
        },
    )


def _run_real_eval(args: argparse.Namespace) -> Any:
    if args.real_eval_command == "run":
        return evaluate_frozen_real_arxiv(
            candidates_path=args.candidates_path,
            labels_path=args.labels_path,
            semantic_provider=args.semantic_provider,
        )
    if args.real_eval_command == "label-template":
        return write_label_template(
            candidates_path=args.candidates_path,
            output_path=args.labels_path,
        )
    if args.real_eval_command == "fetch-candidates":
        return fetch_real_arxiv_candidates(
            output_path=args.candidates_path,
            store_path=args.store_path,
            request_delay_seconds=args.request_delay_seconds,
        )
    raise ValueError(f"Unknown real-eval command: {args.real_eval_command}")


def _build_orchestrator(args: argparse.Namespace) -> DailyArxivAgentOrchestrator:
    store = SQLitePaperStore(args.db_path)
    retrieval_skill = None
    if args.fixture is not None:
        retrieval_skill = ArxivRetrievalSkill(
            store=store,
            client=_FixtureClient(args.fixture),
            request_delay_seconds=0,
        )
    semantic_ranking_skill = None
    if getattr(args, "no_embedding_cache", False):
        config = replace(AppConfig.from_env(), embedding_cache_enabled=False)
        semantic_ranking_skill = SemanticSeedRankingSkill(
            store=store,
            config=config,
        )
    return DailyArxivAgentOrchestrator(
        store=store,
        retrieval_skill=retrieval_skill,
        semantic_ranking_skill=semantic_ranking_skill,
    )


def _seed_preference_from_args(
    args: argparse.Namespace,
) -> SkillResult[Any] | None:
    seeds, has_seed_source = _seed_inputs_from_args(args)
    if not has_seed_source:
        return None
    return SeedParsingSkill().build_preference(
        seeds,
        profile_id=args.profile_id,
    )


def _seed_inputs_from_args(args: argparse.Namespace) -> tuple[list[str], bool]:
    seed_values: list[str] = []
    raw_seed_values = getattr(args, "seed", None) or []
    seed_files = getattr(args, "seed_file", None) or []
    for raw_seed in raw_seed_values:
        seed_values.extend(_nonblank_lines(raw_seed))
    for seed_file in seed_files:
        seed_values.extend(_nonblank_lines(seed_file.read_text()))
    return seed_values, bool(raw_seed_values or seed_files)


def _seed_error_result(seed_result: SkillResult[Any]) -> SkillResult[Any]:
    return SkillResult[Any](
        status=SkillStatus.ERROR,
        data=None,
        evidence_source=seed_result.evidence_source,
        error=seed_result.error,
        message=seed_result.message or "Seed input could not be parsed.",
        metadata={
            "cli_stage": "seed_parsing",
            **seed_result.metadata,
        },
    )


def _nonblank_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _optional_text(value: str | None) -> str | None:
    normalized = " ".join((value or "").split())
    return normalized or None


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


def _existing_file(raw: str) -> Path:
    path = Path(raw)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"seed file does not exist: {raw}")
    return path


def _exit_code_for_status(status: SkillStatus) -> int:
    if status in {SkillStatus.SUCCESS, SkillStatus.EMPTY}:
        return 0
    return 1


def compact_briefing_output(result: Any) -> str:
    """Render a concise human-readable briefing without trace internals."""

    workflow = getattr(result, "data", None)
    briefing = getattr(workflow, "briefing", None)
    lines = [f"Status: {result.status.value}"]
    if result.message:
        lines.append(f"Notice: {result.message}")
    if result.error is not None:
        lines.append(f"Fallback: {result.error.code}: {result.error.message}")
    if briefing is None:
        lines.append("")
        lines.append("No daily briefing was produced.")
        return "\n".join(lines)

    for section in _briefing_sections(briefing):
        lines.append("")
        lines.append(f"## {section['title']}")
        key = section["key"]
        if key == "executive_summary":
            lines.append(str(section["body"]))
        elif key == "top_k_reading_guide":
            lines.extend(_format_top_k_section(section))
        elif key == "evidence_boundary":
            lines.extend(_format_boundary_section(section))
    return "\n".join(lines)


def _briefing_sections(briefing: Any) -> list[dict[str, object]]:
    from daily_arxiv_agent.ui.streamlit_app import enhanced_briefing_sections

    return enhanced_briefing_sections(briefing)


def _format_top_k_section(section: dict[str, object]) -> list[str]:
    lines: list[str] = []
    summary_rows = section["summary_rows"]
    lines.extend(
        _format_row_section(
            summary_rows,
            ["rank", "paper_id", "title", "score", "evidence", "key_reason"],
        )
    )
    paper_briefs = section["paper_briefs"]
    for brief in paper_briefs:
        lines.append("")
        lines.append(f"### Rank {brief['rank']}: {brief['title']}")
        lines.append(
            f"Paper ID: {brief['paper_id']} | Score: {brief['score']} | "
            f"Evidence: {brief['evidence']}"
        )
        lines.append(f"Summary: {brief['summary']}")
        _append_optional_line(lines, "Problem", brief["problem"])
        _append_optional_line(lines, "Approach", brief["approach"])
        _append_optional_line(lines, "Reading guide", brief["reading_guide"])
        _append_optional_line(lines, "Contributions", brief["contributions"])
        _append_optional_line(lines, "Methods", brief["methods"])
        lines.append(f"Relevance: {brief['relevance_rationale']}")
    return lines


def _format_boundary_section(section: dict[str, object]) -> list[str]:
    lines = [
        f"Full text used: {section['full_text_used']}",
        f"Evidence sources: {section['evidence_sources']}",
        f"Unavailable sources: {section['unavailable_sources']}",
    ]
    notes = section["notes"]
    if notes:
        lines.append("Notes: " + "; ".join(str(note) for note in notes))
    abstentions = section["abstentions"]
    if abstentions:
        lines.append("Abstentions: " + "; ".join(str(note) for note in abstentions))
    return lines


def _format_row_section(rows: object, columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    normalized_rows = [row for row in rows if isinstance(row, dict)]
    if not normalized_rows:
        return ["No rows available."]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_cell_text(row.get(column, "")) for column in columns)
        + " |"
        for row in normalized_rows
    ]
    return [header, divider, *body]


def _append_optional_line(lines: list[str], label: str, value: object) -> None:
    text = _cell_text(value)
    if text:
        lines.append(f"{label}: {text}")


def _cell_text(value: object) -> str:
    return " ".join(str(value or "").replace("|", "/").split())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
