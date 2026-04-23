"""SQLite storage for arXiv metadata and retrieval result sets."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from .contracts import FeedbackEvent, PaperMetadata, RetrievalQuery, SeedPreference


class SQLitePaperStore:
    """Small local store for paper metadata and cached retrieval runs."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.initialize()

    def initialize(self) -> None:
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    abstract TEXT,
                    categories_json TEXT NOT NULL,
                    published_date TEXT,
                    updated_date TEXT,
                    arxiv_url TEXT NOT NULL,
                    pdf_url TEXT,
                    provenance_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_runs (
                    query_key TEXT PRIMARY KEY,
                    query_json TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_results (
                    query_key TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (query_key, paper_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );

                CREATE TABLE IF NOT EXISTS seed_preferences (
                    profile_id TEXT PRIMARY KEY,
                    preference_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    event_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    recommendation_run_id TEXT,
                    paper_id TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                """
            )

    def save_papers(self, papers: Iterable[PaperMetadata]) -> None:
        paper_list = list(papers)
        if not paper_list:
            return

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO papers (
                    paper_id,
                    title,
                    authors_json,
                    abstract,
                    categories_json,
                    published_date,
                    updated_date,
                    arxiv_url,
                    pdf_url,
                    provenance_json,
                    payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    abstract = excluded.abstract,
                    categories_json = excluded.categories_json,
                    published_date = excluded.published_date,
                    updated_date = excluded.updated_date,
                    arxiv_url = excluded.arxiv_url,
                    pdf_url = excluded.pdf_url,
                    provenance_json = excluded.provenance_json,
                    payload_json = excluded.payload_json
                """,
                [self._paper_row(paper) for paper in paper_list],
            )

    def save_retrieval(self, query: RetrievalQuery, papers: Iterable[PaperMetadata]) -> None:
        paper_list = list(papers)
        query_key = self.query_key(query)
        self.save_papers(paper_list)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_runs (query_key, query_json, retrieved_at)
                VALUES (?, ?, ?)
                ON CONFLICT(query_key) DO UPDATE SET
                    query_json = excluded.query_json,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    query_key,
                    query.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM retrieval_results WHERE query_key = ?",
                (query_key,),
            )
            connection.executemany(
                """
                INSERT INTO retrieval_results (query_key, paper_id, position)
                VALUES (?, ?, ?)
                """,
                [
                    (query_key, paper.paper_id, position)
                    for position, paper in enumerate(paper_list)
                ],
            )

    def load_retrieval(self, query: RetrievalQuery) -> list[PaperMetadata]:
        query_key = self.query_key(query)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.payload_json
                FROM retrieval_results r
                JOIN papers p ON p.paper_id = r.paper_id
                WHERE r.query_key = ?
                ORDER BY r.position ASC
                """,
                (query_key,),
            ).fetchall()

        return [self._paper_from_payload(row["payload_json"]) for row in rows]

    def list_papers(self) -> list[PaperMetadata]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM papers ORDER BY paper_id ASC"
            ).fetchall()
        return [self._paper_from_payload(row["payload_json"]) for row in rows]

    def find_papers(
        self,
        *,
        topic: str | None = None,
        category: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PaperMetadata]:
        papers = self.list_papers()
        topic_lower = topic.lower() if topic else None

        def matches(paper: PaperMetadata) -> bool:
            if category and category not in paper.categories:
                return False
            if (start_date or end_date) and paper.published_date is None:
                return False
            if start_date and paper.published_date and paper.published_date < start_date:
                return False
            if end_date and paper.published_date and paper.published_date > end_date:
                return False
            if topic_lower:
                haystack = f"{paper.title} {paper.abstract or ''}".lower()
                if topic_lower not in haystack:
                    return False
            return True

        return [paper for paper in papers if matches(paper)]

    def save_seed_preference(self, preference: SeedPreference) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO seed_preferences (
                    profile_id,
                    preference_json,
                    updated_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    preference_json = excluded.preference_json,
                    updated_at = excluded.updated_at
                """,
                (
                    preference.profile_id,
                    preference.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_seed_preference(self, profile_id: str = "default") -> SeedPreference | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT preference_json
                FROM seed_preferences
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()
        if row is None:
            return None
        return SeedPreference.model_validate_json(row["preference_json"])

    def save_feedback_event(self, event: FeedbackEvent) -> None:
        self.save_feedback_events([event])

    def save_feedback_events(self, events: Iterable[FeedbackEvent]) -> None:
        event_list = list(events)
        if not event_list:
            return

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO feedback_events (
                    event_id,
                    profile_id,
                    recommendation_run_id,
                    paper_id,
                    value,
                    created_at,
                    event_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    profile_id = excluded.profile_id,
                    recommendation_run_id = excluded.recommendation_run_id,
                    paper_id = excluded.paper_id,
                    value = excluded.value,
                    created_at = excluded.created_at,
                    event_json = excluded.event_json
                """,
                [
                    (
                        event.event_id,
                        event.profile_id,
                        event.recommendation_run_id,
                        event.paper_id,
                        event.value.value,
                        event.created_at.isoformat(),
                        event.model_dump_json(),
                    )
                    for event in event_list
                ],
            )

    def list_feedback_events(
        self,
        *,
        profile_id: str = "default",
        recommendation_run_id: str | None = None,
    ) -> list[FeedbackEvent]:
        sql = """
            SELECT event_json
            FROM feedback_events
            WHERE profile_id = ?
        """
        params: list[str] = [profile_id]
        if recommendation_run_id is not None:
            sql += " AND recommendation_run_id = ?"
            params.append(recommendation_run_id)
        sql += " ORDER BY created_at ASC, event_id ASC"

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [FeedbackEvent.model_validate_json(row["event_json"]) for row in rows]

    def latest_feedback_by_paper(
        self,
        *,
        profile_id: str = "default",
        recommendation_run_id: str | None = None,
    ) -> dict[str, FeedbackEvent]:
        latest: dict[str, FeedbackEvent] = {}
        for event in self.list_feedback_events(
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
        ):
            latest[event.paper_id] = event
        return latest

    @staticmethod
    def query_key(query: RetrievalQuery) -> str:
        payload = query.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _paper_row(paper: PaperMetadata) -> tuple[str, str, str, str | None, str, str | None, str | None, str, str | None, str, str]:
        payload = paper.model_dump(mode="json")
        return (
            paper.paper_id,
            paper.title,
            json.dumps(paper.authors),
            paper.abstract,
            json.dumps(paper.categories),
            paper.published_date.isoformat() if paper.published_date else None,
            paper.updated_date.isoformat() if paper.updated_date else None,
            str(paper.arxiv_url),
            str(paper.pdf_url) if paper.pdf_url else None,
            paper.provenance.model_dump_json(),
            json.dumps(payload),
        )

    @staticmethod
    def _paper_from_payload(payload_json: str) -> PaperMetadata:
        return PaperMetadata.model_validate(json.loads(payload_json))
