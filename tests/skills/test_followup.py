from datetime import date

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.skills.followup import FollowupQuery, FollowupSkill
from daily_arxiv_agent.storage import SQLitePaperStore


def make_paper(
    paper_id: str,
    title: str,
    abstract: str | None,
    *,
    category: str = "cs.LG",
    published_date: date = date(2026, 4, 20),
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=[category],
        published_date=published_date,
        updated_date=published_date,
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query="agent briefing",
        ),
    )


class SpyRetrievalSkill:
    def __init__(self, papers: list[PaperMetadata] | None = None) -> None:
        self.papers = papers or []
        self.calls = 0
        self.query_plans = []

    def retrieve(self, query, use_cache=True, query_plan=None):  # noqa: ANN001, ANN201
        self.calls += 1
        self.query_plans.append(query_plan)
        return SkillResult[list[PaperMetadata]](
            status=SkillStatus.SUCCESS if self.papers else SkillStatus.EMPTY,
            data=self.papers,
            evidence_source=EvidenceSource.ABSTRACT,
            provenance=[paper.provenance for paper in self.papers],
            metadata={"query": query.model_dump(mode="json"), "cache_hit": False},
        )


def test_followup_filters_stored_papers_without_refetching(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    matching = make_paper(
        "2604.00001",
        "Explainable Agents for Daily Research Briefings",
        "Agent workflows for research-paper recommendation.",
    )
    unrelated = make_paper(
        "2604.00002",
        "Compiler Register Allocation",
        "Compiler optimization for register pressure.",
        category="cs.PL",
    )
    store.save_papers([matching, unrelated])
    retrieval = SpyRetrievalSkill()

    result = FollowupSkill(store=store, retrieval_skill=retrieval).query(
        FollowupQuery(
            topic="agent workflow",
            category="cs.LG",
            start_date=date(2026, 4, 19),
            end_date=date(2026, 4, 21),
        )
    )

    assert result.status == SkillStatus.SUCCESS
    assert [paper.paper_id for paper in result.data or []] == ["2604.00001"]
    assert retrieval.calls == 0
    assert result.metadata["local_hit"] is True
    assert result.metadata["fetch_attempted"] is False
    assert result.metadata["planner_source"] == "deterministic"


def test_followup_local_filter_uses_planner_normalized_terms(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    matching = make_paper(
        "2604.00001",
        "Explainable Agents for Daily Research Briefings",
        "Research briefing workflows rank papers from preference signals.",
    )
    store.save_papers([matching])
    retrieval = SpyRetrievalSkill()

    result = FollowupSkill(store=store, retrieval_skill=retrieval).query(
        FollowupQuery(topic="agents for workflows", category="cs.LG")
    )

    assert result.status == SkillStatus.SUCCESS
    assert [paper.paper_id for paper in result.data or []] == ["2604.00001"]
    assert retrieval.calls == 0
    assert result.metadata["query_variant_count"] == 1


def test_followup_fetches_only_when_no_stored_papers_match(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    fetched = make_paper(
        "2604.00003",
        "Graph Neural Networks for Research Agents",
        "Graph neural retrieval can support agent recommendations.",
    )
    retrieval = SpyRetrievalSkill([fetched])

    result = FollowupSkill(store=store, retrieval_skill=retrieval).query(
        FollowupQuery(topic="graph neural", category="cs.LG")
    )

    assert result.status == SkillStatus.SUCCESS
    assert [paper.paper_id for paper in result.data or []] == ["2604.00003"]
    assert retrieval.calls == 1
    assert retrieval.query_plans[0] is not None
    assert result.metadata["local_hit"] is False
    assert result.metadata["fetch_attempted"] is True
    assert result.metadata["query_variant_count"] == 1


def test_followup_returns_empty_when_fetched_papers_still_do_not_match(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    fetched = make_paper(
        "2604.00004",
        "Compiler Register Allocation",
        "Compiler optimization for register pressure.",
        category="cs.PL",
    )
    retrieval = SpyRetrievalSkill([fetched])

    result = FollowupSkill(store=store, retrieval_skill=retrieval).query(
        FollowupQuery(topic="graph neural", category="cs.LG")
    )

    assert result.status == SkillStatus.EMPTY
    assert result.data == []
    assert result.metadata["fetch_attempted"] is True


def test_followup_returns_clear_fallback_when_no_local_results_and_no_retrieval(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")

    result = FollowupSkill(store=store).query(FollowupQuery(topic="graph neural"))

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "followup_no_retrieval_skill"
    assert result.data == []
