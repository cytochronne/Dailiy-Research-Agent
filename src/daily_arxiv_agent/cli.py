"""Command-line entry points for the local Daily arXiv Agent demo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
import json
from pathlib import Path
from typing import Any

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    QueryPlannerMode,
    RetrievalQuery,
    SearchMode,
    SkillStatus,
)
from daily_arxiv_agent.orchestrator import DailyArxivAgentOrchestrator
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.skills.followup import FollowupQuery
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
    else:  # pragma: no cover - argparse prevents this branch.
        parser.error(f"Unknown command: {args.command}")
    if getattr(args, "output_format", "json") == "briefing":
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
    demo.add_argument("--top-k", type=int, default=5)
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

    followup = subparsers.add_parser("followup", help="Run a local-first follow-up query.")
    _add_common_filters(followup)
    followup.add_argument("--fixture", type=Path, help="Optional arXiv Atom XML fixture path for empty local results.")
    followup.add_argument("--db-path", default=config.db_path)
    followup.add_argument("--top-k", type=int, default=5)
    followup.add_argument("--local-only", action="store_true")
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


def _run_demo(args: argparse.Namespace) -> Any:
    orchestrator = _build_orchestrator(args)
    query = RetrievalQuery(
        topic=args.topic,
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
    return orchestrator.run_recommendation(
        query,
        top_k=args.top_k,
        use_cache=not args.no_cache,
        include_debug_trace=args.debug_trace,
    )


def _run_followup(args: argparse.Namespace) -> Any:
    orchestrator = _build_orchestrator(args)
    query = FollowupQuery(
        topic=args.topic,
        category=args.category,
        start_date=args.start_date,
        end_date=args.end_date,
        max_results=args.max_results,
        fetch_if_empty=not args.local_only,
    )
    return orchestrator.run_followup_query(query, top_k=args.top_k)


def _build_orchestrator(args: argparse.Namespace) -> DailyArxivAgentOrchestrator:
    store = SQLitePaperStore(args.db_path)
    retrieval_skill = None
    if args.fixture is not None:
        retrieval_skill = ArxivRetrievalSkill(
            store=store,
            client=_FixtureClient(args.fixture),
            request_delay_seconds=0,
        )
    return DailyArxivAgentOrchestrator(store=store, retrieval_skill=retrieval_skill)


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc


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
