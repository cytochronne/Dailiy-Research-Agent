from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

from daily_arxiv_agent.contracts import SkillStatus
from daily_arxiv_agent.evaluation.real_arxiv import (
    DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
    DEFAULT_REAL_ARXIV_LABELS_PATH,
    DEFAULT_REAL_ARXIV_TOPICS,
    RealArxivCandidate,
    RealArxivTopic,
    _bm25_ranked_ids,
    _metrics_for_ranked_ids,
    _strict_keyword_ranked_ids,
    evaluate_frozen_real_arxiv,
    load_real_arxiv_candidates,
)


def test_frozen_real_arxiv_fixture_has_three_topics_with_fifty_candidates() -> None:
    result = evaluate_frozen_real_arxiv(
        candidates_path=DEFAULT_REAL_ARXIV_CANDIDATES_PATH,
        labels_path=DEFAULT_REAL_ARXIV_LABELS_PATH,
        semantic_provider="fake",
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data is not None
    assert result.data.candidate_count == 150
    assert result.data.label_count == 150
    assert result.data.topics == [
        "llm_tool_agents",
        "retrieval_augmented_generation",
        "vision_language_robotics",
    ]
    assert result.data.methods == [
        "agent",
        "semantic_agent",
        "strict_keyword",
        "bm25",
    ]
    assert len(result.data.rows) == 12
    assert len(result.data.macro_average_rows) == 4


def test_frozen_real_arxiv_candidate_and_label_ids_match() -> None:
    candidate_rows = [
        json.loads(line)
        for line in DEFAULT_REAL_ARXIV_CANDIDATES_PATH.read_text().splitlines()
        if line.strip()
    ]
    label_rows = [
        json.loads(line)
        for line in DEFAULT_REAL_ARXIV_LABELS_PATH.read_text().splitlines()
        if line.strip()
    ]

    candidates_by_topic: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows:
        candidates_by_topic.setdefault(str(row["topic_id"]), []).append(row)
    assert {topic.topic_id for topic in DEFAULT_REAL_ARXIV_TOPICS} == set(candidates_by_topic)
    for rows in candidates_by_topic.values():
        assert len(rows) == 50

    candidate_keys = {
        (str(row["topic_id"]), str(row["paper_id"])) for row in candidate_rows
    }
    label_keys = {(str(row["topic_id"]), str(row["paper_id"])) for row in label_rows}
    assert label_keys == candidate_keys


def test_real_arxiv_loader_reports_invalid_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "bad_candidates.jsonl"
    path.write_text('{"topic_id": "x"}\n')

    result = load_real_arxiv_candidates(path)

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "real_arxiv_candidate_file_invalid"
    assert "bad_candidates.jsonl:1" in result.error.message


def test_strict_keyword_and_bm25_rankers_are_stable() -> None:
    topic = RealArxivTopic(
        topic_id="toy",
        topic="retrieval augmented generation",
        category="cs.CL",
        query="all:retrieval",
        required_terms=["retrieval", "augmented", "generation"],
    )
    candidates = [
        _candidate("toy", "p1", 1, "Unrelated language model", "No matching evidence."),
        _candidate(
            "toy",
            "p2",
            2,
            "Retrieval augmented generation for QA",
            "Retrieval augmented generation grounds answers.",
        ),
        _candidate(
            "toy",
            "p3",
            3,
            "Retrieval methods",
            "Retrieval is discussed without generation.",
        ),
    ]

    assert _strict_keyword_ranked_ids(topic, candidates) == ["p2", "p3", "p1"]
    assert _bm25_ranked_ids(topic, candidates) == ["p2", "p3", "p1"]


def test_real_arxiv_metrics_compute_precision_recall_and_mrr() -> None:
    metrics = _metrics_for_ranked_ids(
        topic_id="toy",
        method="bm25",
        ranked_ids=["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"],
        relevant_ids={"p1", "p6", "p11"},
        candidate_count=12,
    )

    assert metrics.precision_at_5 == 0.2
    assert metrics.recall_at_10 == 0.6667
    assert metrics.mean_reciprocal_rank == 0.5


def _candidate(
    topic_id: str,
    paper_id: str,
    rank: int,
    title: str,
    abstract: str,
) -> RealArxivCandidate:
    return RealArxivCandidate(
        topic_id=topic_id,
        query="all:test",
        fetched_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        category="cs.CL",
        categories=["cs.CL"],
        submitted_date=date(2026, 5, 10),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        candidate_rank=rank,
    )
