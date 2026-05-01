"""SQLite storage for arXiv metadata and retrieval result sets."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

from .contracts import (
    FeedbackEvent,
    PaperMetadata,
    QueryPlan,
    RetrievalCacheStatus,
    RetrievalQuery,
    RetrievalResultSet,
    RetrievalSourceMetadata,
    SeedPreference,
)


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
                    effective_plan_json TEXT,
                    cache_status TEXT NOT NULL DEFAULT 'complete',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    retrieved_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_results (
                    query_key TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    source_metadata_json TEXT NOT NULL DEFAULT '[]',
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

                CREATE TABLE IF NOT EXISTS paper_full_text_cache (
                    paper_id TEXT PRIMARY KEY,
                    full_text TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                );
                """
            )
            self._ensure_retrieval_schema(connection)

    def _ensure_retrieval_schema(self, connection: sqlite3.Connection) -> None:
        self._ensure_column(
            connection,
            "retrieval_runs",
            "effective_plan_json",
            "effective_plan_json TEXT",
        )
        self._ensure_column(
            connection,
            "retrieval_runs",
            "cache_status",
            "cache_status TEXT NOT NULL DEFAULT 'complete'",
        )
        self._ensure_column(
            connection,
            "retrieval_runs",
            "metadata_json",
            "metadata_json TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(
            connection,
            "retrieval_results",
            "source_metadata_json",
            "source_metadata_json TEXT NOT NULL DEFAULT '[]'",
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})")
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")

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
        self.save_retrieval_result_set(query, papers)

    def save_retrieval_result_set(
        self,
        query: RetrievalQuery,
        papers: Iterable[PaperMetadata],
        *,
        query_plan: QueryPlan | None = None,
        source_metadata_by_paper_id: Mapping[str, list[RetrievalSourceMetadata]]
        | None = None,
        cache_status: RetrievalCacheStatus = RetrievalCacheStatus.COMPLETE,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        paper_list = list(papers)
        query_key = self.effective_query_key(query, query_plan=query_plan)
        source_metadata = source_metadata_by_paper_id or {}
        if (
            cache_status == RetrievalCacheStatus.PARTIAL
            and self._has_complete_retrieval_result_set(query_key)
        ):
            return
        self.save_papers(paper_list)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_runs (
                    query_key,
                    query_json,
                    effective_plan_json,
                    cache_status,
                    metadata_json,
                    retrieved_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_key) DO UPDATE SET
                    query_json = excluded.query_json,
                    effective_plan_json = excluded.effective_plan_json,
                    cache_status = excluded.cache_status,
                    metadata_json = excluded.metadata_json,
                    retrieved_at = excluded.retrieved_at
                """,
                (
                    query_key,
                    query.model_dump_json(),
                    query_plan.model_dump_json() if query_plan is not None else None,
                    cache_status.value,
                    json.dumps(metadata or {}, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.execute(
                "DELETE FROM retrieval_results WHERE query_key = ?",
                (query_key,),
            )
            connection.executemany(
                """
                INSERT INTO retrieval_results (
                    query_key,
                    paper_id,
                    position,
                    source_metadata_json
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        query_key,
                        paper.paper_id,
                        position,
                        self._source_metadata_json(
                            source_metadata.get(paper.paper_id, [])
                        ),
                    )
                    for position, paper in enumerate(paper_list)
                ],
            )

    def load_retrieval(self, query: RetrievalQuery) -> list[PaperMetadata]:
        result_set = self.load_retrieval_result_set(query)
        return result_set.papers if result_set is not None else []

    def load_retrieval_result_set(
        self,
        query: RetrievalQuery,
        *,
        query_plan: QueryPlan | None = None,
        accept_partial: bool = False,
    ) -> RetrievalResultSet | None:
        query_key = self.effective_query_key(query, query_plan=query_plan)
        with self._connect() as connection:
            run = self._load_retrieval_run(connection, query_key)
            if run is None and query_plan is None:
                legacy_key = self.legacy_query_key(query)
                if legacy_key != query_key:
                    run = self._load_retrieval_run(connection, legacy_key)
                    if run is not None:
                        query_key = legacy_key
            if run is None:
                return None

            cache_status = self._cache_status_from_value(run["cache_status"])
            if cache_status == RetrievalCacheStatus.PARTIAL and not accept_partial:
                return None

            rows = connection.execute(
                """
                SELECT p.payload_json, r.source_metadata_json
                FROM retrieval_results r
                JOIN papers p ON p.paper_id = r.paper_id
                WHERE r.query_key = ?
                ORDER BY r.position ASC
                """,
                (query_key,),
            ).fetchall()

        papers = [self._paper_from_payload(row["payload_json"]) for row in rows]
        source_metadata_by_paper_id = {
            paper.paper_id: self._source_metadata_from_json(row["source_metadata_json"])
            for paper, row in zip(papers, rows, strict=True)
        }
        return RetrievalResultSet(
            query=RetrievalQuery.model_validate_json(run["query_json"]),
            papers=papers,
            cache_status=cache_status,
            metadata=self._metadata_from_json(run["metadata_json"]),
            source_metadata_by_paper_id=source_metadata_by_paper_id,
            retrieved_at=datetime.fromisoformat(run["retrieved_at"]),
            effective_query_key=query_key,
        )

    def list_papers(self) -> list[PaperMetadata]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM papers ORDER BY paper_id ASC"
            ).fetchall()
        return [self._paper_from_payload(row["payload_json"]) for row in rows]

    def get_paper(self, paper_id: str) -> PaperMetadata | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM papers
                WHERE paper_id = ?
                """,
                (paper_id,),
            ).fetchone()
        if row is None:
            return None
        return self._paper_from_payload(row["payload_json"])

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

    def save_paper_full_text(
        self,
        paper_id: str,
        full_text: str,
        *,
        source_url: str | None = None,
    ) -> None:
        lines = [" ".join(line.split()) for line in full_text.splitlines()]
        normalized = "\n".join(line for line in lines if line)
        if not normalized:
            normalized = " ".join(full_text.split())
        if not normalized:
            return
        cache_key = self._paper_full_text_cache_key(paper_id, source_url)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO paper_full_text_cache (
                    paper_id,
                    full_text,
                    cached_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    full_text = excluded.full_text,
                    cached_at = excluded.cached_at
                """,
                (
                    cache_key,
                    normalized,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_paper_full_text(
        self,
        paper_id: str,
        *,
        source_url: str | None = None,
    ) -> str | None:
        cache_key = self._paper_full_text_cache_key(paper_id, source_url)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT full_text
                FROM paper_full_text_cache
                WHERE paper_id = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return str(row["full_text"])

    @staticmethod
    def query_key(query: RetrievalQuery) -> str:
        payload = query.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def legacy_query_key(query: RetrievalQuery) -> str:
        payload = {
            "topic": query.topic,
            "category": query.category,
            "start_date": query.start_date.isoformat() if query.start_date else None,
            "end_date": query.end_date.isoformat() if query.end_date else None,
            "start_index": query.start_index,
            "max_results": query.max_results,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def effective_query_key(
        cls,
        query: RetrievalQuery,
        *,
        query_plan: QueryPlan | None = None,
    ) -> str:
        if query_plan is None:
            return cls.query_key(query)
        payload = {
            "query": query.model_dump(mode="json"),
            "query_plan": cls._query_plan_cache_payload(query_plan),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _load_retrieval_run(
        connection: sqlite3.Connection,
        query_key: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT query_json, cache_status, metadata_json, retrieved_at
            FROM retrieval_runs
            WHERE query_key = ?
            """,
            (query_key,),
        ).fetchone()

    def _has_complete_retrieval_result_set(self, query_key: str) -> bool:
        with self._connect() as connection:
            row = self._load_retrieval_run(connection, query_key)
        if row is None:
            return False
        return (
            self._cache_status_from_value(row["cache_status"])
            == RetrievalCacheStatus.COMPLETE
        )

    @staticmethod
    def _cache_status_from_value(value: str | None) -> RetrievalCacheStatus:
        if not value:
            return RetrievalCacheStatus.COMPLETE
        try:
            return RetrievalCacheStatus(value)
        except ValueError:
            return RetrievalCacheStatus.COMPLETE

    @staticmethod
    def _query_plan_cache_payload(query_plan: QueryPlan) -> dict[str, Any]:
        payload = query_plan.model_dump(mode="json")
        planner = payload.get("planner")
        if isinstance(planner, dict):
            planner.pop("generated_at", None)
        return payload

    @staticmethod
    def _source_metadata_json(
        source_metadata: list[RetrievalSourceMetadata],
    ) -> str:
        payload = [
            metadata.model_dump(mode="json")
            for metadata in source_metadata
        ]
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _source_metadata_from_json(payload_json: str) -> list[RetrievalSourceMetadata]:
        try:
            payload = json.loads(payload_json or "[]")
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict):
            payload = [payload] if payload else []
        if not isinstance(payload, list):
            return []
        return [
            RetrievalSourceMetadata.model_validate(item)
            for item in payload
            if isinstance(item, dict)
        ]

    @staticmethod
    def _metadata_from_json(payload_json: str) -> dict[str, Any]:
        try:
            payload = json.loads(payload_json or "{}")
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _paper_full_text_cache_key(paper_id: str, source_url: str | None) -> str:
        normalized_source = " ".join(source_url.split()) if source_url else ""
        if not normalized_source:
            return paper_id
        return f"{paper_id}::{normalized_source}"

    @staticmethod
    def _paper_row(
        paper: PaperMetadata,
    ) -> tuple[
        str,
        str,
        str,
        str | None,
        str,
        str | None,
        str | None,
        str,
        str | None,
        str,
        str,
    ]:
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
