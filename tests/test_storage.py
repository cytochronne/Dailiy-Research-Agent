from datetime import date, datetime, timezone

from daily_arxiv_agent.contracts import (
    FeedbackEvent,
    FeedbackValue,
    PaperMetadata,
    Provenance,
    QueryPlan,
    QueryPlanVariant,
    QueryPlannerMode,
    QueryPlannerProvenance,
    RetrievalCacheStatus,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SearchMode,
)
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


def make_paper(paper_id: str = "2604.00001", category: str = "cs.LG") -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title="Explainable Agents for Daily Research Briefings",
        authors=["Ada Lovelace"],
        abstract="Agent workflows for research-paper recommendation.",
        categories=[category],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query="cat:cs.LG",
        ),
    )


def test_sqlite_store_persists_and_loads_papers(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    paper = make_paper()

    store.save_papers([paper])
    loaded = store.list_papers()

    assert loaded == [paper]


def test_sqlite_store_tracks_retrieval_result_sets(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    query = RetrievalQuery(
        topic="agents",
        category="cs.LG",
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 21),
    )
    paper = make_paper()

    store.save_retrieval(query, [paper])
    loaded = store.load_retrieval(query)

    assert loaded == [paper]


def test_query_key_distinguishes_broad_and_strict_search_modes() -> None:
    strict_query = RetrievalQuery(topic="agents", search_mode=SearchMode.STRICT)
    broad_query = RetrievalQuery(topic="agents", search_mode=SearchMode.BROAD)

    assert SQLitePaperStore.query_key(strict_query) != SQLitePaperStore.query_key(
        broad_query
    )
    assert SQLitePaperStore.query_key(strict_query) == SQLitePaperStore.query_key(
        RetrievalQuery(topic="agents", search_mode=SearchMode.STRICT)
    )


def test_effective_query_key_includes_query_plan_metadata() -> None:
    query = RetrievalQuery(topic="agents", search_mode=SearchMode.BROAD)
    first_plan = make_query_plan("broad_terms", 'all:"agents"')
    second_plan = make_query_plan("title_terms", 'ti:"agents"')

    assert SQLitePaperStore.effective_query_key(
        query,
        query_plan=first_plan,
    ) != SQLitePaperStore.effective_query_key(query, query_plan=second_plan)


def test_effective_query_key_ignores_nonsemantic_planner_timestamp() -> None:
    query = RetrievalQuery(topic="agents", search_mode=SearchMode.BROAD)
    first_plan = make_query_plan(
        "broad_terms",
        'all:"agents"',
        generated_at=datetime(2026, 4, 30, 1, tzinfo=timezone.utc),
    )
    second_plan = make_query_plan(
        "broad_terms",
        'all:"agents"',
        generated_at=datetime(2026, 4, 30, 2, tzinfo=timezone.utc),
    )

    assert SQLitePaperStore.effective_query_key(
        query,
        query_plan=first_plan,
    ) == SQLitePaperStore.effective_query_key(query, query_plan=second_plan)


def test_sqlite_store_saves_effective_retrieval_with_source_metadata(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    query = RetrievalQuery(topic="agents", search_mode=SearchMode.BROAD)
    plan = make_query_plan("broad_terms", 'all:"agents"')
    paper = make_paper()
    source_metadata = RetrievalSourceMetadata(
        variant_label="broad_terms",
        sort_by="relevance",
        variant_index=0,
        position=2,
        first_seen_order=0,
        query='all:"agents"',
    )

    store.save_retrieval_result_set(
        query,
        [paper],
        query_plan=plan,
        source_metadata_by_paper_id={paper.paper_id: [source_metadata]},
    )
    loaded = store.load_retrieval_result_set(query, query_plan=plan)

    assert loaded is not None
    assert loaded.cache_status == RetrievalCacheStatus.COMPLETE
    assert loaded.papers == [paper]
    assert loaded.source_metadata_by_paper_id[paper.paper_id] == [source_metadata]


def test_retrieval_source_metadata_is_scoped_by_effective_plan(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    query = RetrievalQuery(topic="agents", search_mode=SearchMode.BROAD)
    paper = make_paper()
    first_plan = make_query_plan("broad_terms", 'all:"agents"')
    second_plan = make_query_plan("title_terms", 'ti:"agents"')

    store.save_retrieval_result_set(
        query,
        [paper],
        query_plan=first_plan,
        source_metadata_by_paper_id={
            paper.paper_id: [
                RetrievalSourceMetadata(
                    variant_label="broad_terms",
                    sort_by="relevance",
                    variant_index=0,
                    position=0,
                    first_seen_order=0,
                    query='all:"agents"',
                )
            ]
        },
    )
    store.save_retrieval_result_set(
        query,
        [paper],
        query_plan=second_plan,
        source_metadata_by_paper_id={
            paper.paper_id: [
                RetrievalSourceMetadata(
                    variant_label="title_terms",
                    sort_by="submittedDate",
                    variant_index=0,
                    position=4,
                    first_seen_order=0,
                    query='ti:"agents"',
                )
            ]
        },
    )

    first_loaded = store.load_retrieval_result_set(query, query_plan=first_plan)
    second_loaded = store.load_retrieval_result_set(query, query_plan=second_plan)

    assert first_loaded is not None
    assert second_loaded is not None
    assert (
        first_loaded.source_metadata_by_paper_id[paper.paper_id][0].variant_label
        == "broad_terms"
    )
    assert (
        second_loaded.source_metadata_by_paper_id[paper.paper_id][0].variant_label
        == "title_terms"
    )
    assert store.get_paper(paper.paper_id) == paper


def test_partial_retrieval_cache_entry_is_not_loaded_as_complete(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    query = RetrievalQuery(topic="agents", search_mode=SearchMode.BROAD)
    plan = make_query_plan("broad_terms", 'all:"agents"')
    paper = make_paper()

    store.save_retrieval_result_set(
        query,
        [paper],
        query_plan=plan,
        cache_status=RetrievalCacheStatus.PARTIAL,
    )

    assert store.load_retrieval_result_set(query, query_plan=plan) is None
    partial = store.load_retrieval_result_set(
        query,
        query_plan=plan,
        accept_partial=True,
    )

    assert partial is not None
    assert partial.cache_status == RetrievalCacheStatus.PARTIAL
    assert partial.papers == [paper]


def test_sqlite_store_filters_papers_for_followup_queries(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    matching = make_paper("2604.00001", "cs.LG")
    other = make_paper("2604.00002", "cs.IR")
    store.save_papers([matching, other])

    loaded = store.find_papers(
        topic="briefings",
        category="cs.LG",
        start_date=date(2026, 4, 19),
        end_date=date(2026, 4, 21),
    )

    assert loaded == [matching]


def test_sqlite_store_persists_seed_preference_for_later_reuse(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    preference = SeedParsingSkill(metadata_client=None).build_preference(
        ["Agent workflows for research paper recommendation"],
        profile_id="demo",
    ).data

    assert preference is not None
    store.save_seed_preference(preference)
    loaded = store.load_seed_preference("demo")

    assert loaded == preference


def test_sqlite_store_persists_feedback_events_for_later_reuse(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    paper = make_paper()
    event = FeedbackEvent(
        profile_id="demo",
        recommendation_run_id="run-1",
        paper_id=paper.paper_id,
        value=FeedbackValue.LIKE,
        paper=paper,
    )

    store.save_feedback_event(event)
    loaded = store.list_feedback_events(
        profile_id="demo",
        recommendation_run_id="run-1",
    )

    assert loaded == [event]


def test_sqlite_store_returns_latest_feedback_by_paper(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    paper = make_paper()
    first = FeedbackEvent(
        event_id="event-1",
        profile_id="demo",
        recommendation_run_id="run-1",
        paper_id=paper.paper_id,
        value=FeedbackValue.LIKE,
        paper=paper,
    )
    second = FeedbackEvent(
        event_id="event-2",
        profile_id="demo",
        recommendation_run_id="run-1",
        paper_id=paper.paper_id,
        value=FeedbackValue.DISLIKE,
        paper=paper,
    )

    store.save_feedback_events([first, second])
    latest = store.latest_feedback_by_paper(
        profile_id="demo",
        recommendation_run_id="run-1",
    )

    assert latest[paper.paper_id] == second


def test_sqlite_store_caches_full_text_for_selected_papers(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")

    store.save_paper_full_text(
        "2604.00001",
        "  Full paper text for a selected explanation path.  ",
        source_url="https://arxiv.org/pdf/2604.00001v1",
    )
    loaded = store.load_paper_full_text(
        "2604.00001",
        source_url="https://arxiv.org/pdf/2604.00001v1",
    )

    assert loaded == "Full paper text for a selected explanation path."


def test_sqlite_store_scopes_full_text_cache_by_source_url(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")

    store.save_paper_full_text(
        "2604.00001",
        "Version one text.",
        source_url="https://arxiv.org/pdf/2604.00001v1",
    )

    assert (
        store.load_paper_full_text(
            "2604.00001",
            source_url="https://arxiv.org/pdf/2604.00001v2",
        )
        is None
    )


def make_query_plan(
    label: str,
    search_query: str,
    *,
    generated_at: datetime | None = None,
) -> QueryPlan:
    return QueryPlan(
        search_mode=SearchMode.BROAD,
        planner=QueryPlannerProvenance(
            requested_mode=QueryPlannerMode.DETERMINISTIC,
            source="deterministic",
            generated_at=generated_at or datetime.now(timezone.utc),
        ),
        variants=[
            QueryPlanVariant(
                label=label,
                search_query=search_query,
                sort_by="relevance",
            )
        ],
    )
