from datetime import date

from daily_arxiv_agent.contracts import PaperMetadata, Provenance, RetrievalQuery
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
