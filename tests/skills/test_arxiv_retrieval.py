from datetime import date
from io import BytesIO
from pathlib import Path
from urllib import error

import pytest

from daily_arxiv_agent.contracts import (
    QueryPlan,
    QueryPlannerMode,
    QueryPlannerProvenance,
    QueryPlanVariant,
    RetrievalCacheStatus,
    RetrievalQuery,
    SearchMode,
    SkillStatus,
)
from daily_arxiv_agent.skills.arxiv_retrieval import (
    ArxivRetrievalSkill,
    build_arxiv_request_params,
    parse_atom_response,
)
from daily_arxiv_agent.storage import SQLitePaperStore


FIXTURE = Path(__file__).parents[1] / "fixtures" / "arxiv_atom_response.xml"


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.text)


class SequencedClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


class FailingClient:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls += 1
        raise TimeoutError("network unavailable")


class RateLimitedClient:
    def __init__(self, text: str, *, failures_before_success: int) -> None:
        self.text = text
        self.failures_before_success = failures_before_success
        self.calls = 0

    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise error.HTTPError(
                url=url,
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "0"},
                fp=BytesIO(b"rate limited"),
            )
        return FakeResponse(self.text)


def test_parse_atom_fixture_returns_normalized_papers() -> None:
    query = RetrievalQuery(topic="agents", category="cs.LG")

    papers = parse_atom_response(FIXTURE.read_text(), query)

    assert len(papers) == 2
    assert papers[0].paper_id == "2604.00001"
    assert papers[0].title == "Explainable Agents for Daily Research Briefings"
    assert papers[0].authors == ["Ada Lovelace", "Alan Turing"]
    assert papers[0].abstract == "We study agent workflows for explainable research-paper recommendation."
    assert papers[0].categories == ["cs.LG", "cs.AI"]
    assert papers[0].published_date == date(2026, 4, 20)
    assert str(papers[0].arxiv_url) == "https://arxiv.org/abs/2604.00001"
    assert str(papers[0].pdf_url) == "https://arxiv.org/pdf/2604.00001v1"
    assert str(papers[0].provenance.source_url) == "https://arxiv.org/abs/2604.00001v1"
    assert papers[0].provenance.source == "arxiv"


def test_build_arxiv_request_params_supports_date_category_and_topic() -> None:
    query = RetrievalQuery(
        topic="agent briefing",
        category="cs.LG",
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 21),
        start_index=5,
        max_results=25,
    )

    params = build_arxiv_request_params(query)

    assert params["start"] == 5
    assert params["max_results"] == 25
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"
    assert 'all:"agent briefing"' in params["search_query"]
    assert "cat:cs.LG" in params["search_query"]
    assert "submittedDate:[202604180000 TO 202604212359]" in params["search_query"]


def test_retrieval_skill_persists_results_and_reuses_cached_run(tmp_path) -> None:
    query = RetrievalQuery(topic="agents", category="cs.LG")
    client = FakeClient(FIXTURE.read_text())
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    skill = ArxivRetrievalSkill(store=store, client=client, request_delay_seconds=0)

    first = skill.retrieve(query)
    second = skill.retrieve(query)

    assert first.status == SkillStatus.SUCCESS
    assert len(first.data or []) == 2
    assert second.status == SkillStatus.SUCCESS
    assert len(second.data or []) == 2
    assert second.metadata["cache_hit"] is True
    assert len(client.calls) == 1


def test_multi_query_plan_fetches_variants_dedupes_and_records_source_metadata(tmp_path) -> None:
    query = RetrievalQuery(
        topic="agents",
        category="cs.LG",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=10,
        page_size=10,
        max_requests=4,
    )
    plan = make_query_plan(
        QueryPlanVariant(
            label="broad_terms",
            search_query='all:"agents" AND cat:cs.LG',
            sort_by="relevance",
        ),
        QueryPlanVariant(
            label="recent_terms",
            search_query='all:"agents" AND cat:cs.LG',
            sort_by="submittedDate",
        ),
    )
    client = SequencedClient([FIXTURE.read_text(), FIXTURE.read_text()])
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    skill = ArxivRetrievalSkill(store=store, client=client, request_delay_seconds=0)

    result = skill.retrieve(query, query_plan=plan)

    assert result.status == SkillStatus.SUCCESS
    assert [paper.paper_id for paper in result.data or []] == [
        "2604.00001",
        "2604.00002",
    ]
    assert len(client.calls) == 2
    assert [call["params"]["sortBy"] for call in client.calls] == [
        "relevance",
        "submittedDate",
    ]
    assert result.metadata["candidate_count"] == 2
    assert result.metadata["request_count"] == 2
    assert result.metadata["cache_status"] == RetrievalCacheStatus.COMPLETE.value

    loaded = store.load_retrieval_result_set(query, query_plan=plan)
    assert loaded is not None
    assert loaded.cache_status == RetrievalCacheStatus.COMPLETE
    assert [paper.paper_id for paper in loaded.papers] == [
        "2604.00001",
        "2604.00002",
    ]
    first_metadata = loaded.source_metadata_by_paper_id["2604.00001"]
    assert [metadata.variant_label for metadata in first_metadata] == [
        "broad_terms",
        "recent_terms",
    ]
    assert [metadata.sort_by for metadata in first_metadata] == [
        "relevance",
        "submittedDate",
    ]
    assert [metadata.first_seen_order for metadata in first_metadata] == [0, 0]


def test_repeated_plan_retrieval_hits_effective_cache(tmp_path) -> None:
    query = RetrievalQuery(
        topic="agents",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=10,
    )
    plan = make_query_plan(
        QueryPlanVariant(
            label="broad_terms",
            search_query='all:"agents"',
            sort_by="relevance",
        )
    )
    client = SequencedClient([FIXTURE.read_text()])
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    skill = ArxivRetrievalSkill(store=store, client=client, request_delay_seconds=0)

    first = skill.retrieve(query, query_plan=plan)
    second = skill.retrieve(query, query_plan=plan)

    assert first.status == SkillStatus.SUCCESS
    assert second.status == SkillStatus.SUCCESS
    assert second.metadata["cache_hit"] is True
    assert second.metadata["effective_query_key"] == first.metadata["effective_query_key"]
    assert len(client.calls) == 1


def test_candidate_target_stops_before_fetching_extra_variants(tmp_path) -> None:
    query = RetrievalQuery(
        topic="agents",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=1,
        page_size=10,
        max_requests=4,
    )
    plan = make_query_plan(
        QueryPlanVariant(label="first", search_query="all:agents", sort_by="relevance"),
        QueryPlanVariant(label="second", search_query="ti:agents", sort_by="relevance"),
    )
    client = SequencedClient([FIXTURE.read_text()])
    skill = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        client=client,
        request_delay_seconds=0,
    )

    result = skill.retrieve(query, query_plan=plan)

    assert result.status == SkillStatus.SUCCESS
    assert [paper.paper_id for paper in result.data or []] == ["2604.00001"]
    assert len(client.calls) == 1
    assert result.metadata["candidate_count"] == 1


def test_request_budget_exhaustion_reports_actual_unique_candidate_count(tmp_path) -> None:
    query = RetrievalQuery(
        topic="agents",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=100,
        page_size=10,
        max_requests=4,
    )
    plan = make_query_plan(
        QueryPlanVariant(label="first", search_query="all:agents", sort_by="relevance"),
        QueryPlanVariant(label="second", search_query="ti:agents", sort_by="relevance"),
        QueryPlanVariant(label="third", search_query="abs:agents", sort_by="submittedDate"),
        QueryPlanVariant(label="fourth", search_query="cat:cs.LG", sort_by="submittedDate"),
    )
    client = SequencedClient(
        [
            atom_feed([paper_entry("2604.10001", "First Agent Paper")]),
            atom_feed([paper_entry("2604.10001", "First Agent Paper")]),
            atom_feed([paper_entry("2604.10002", "Second Agent Paper")]),
            atom_feed([paper_entry("2604.10002", "Second Agent Paper")]),
        ]
    )
    skill = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        client=client,
        request_delay_seconds=0,
    )

    result = skill.retrieve(query, query_plan=plan)

    assert result.status == SkillStatus.SUCCESS
    assert [paper.paper_id for paper in result.data or []] == [
        "2604.10001",
        "2604.10002",
    ]
    assert result.metadata["request_count"] == 4
    assert result.metadata["candidate_count"] == 2
    assert result.metadata["candidate_target"] == 100
    assert result.metadata["budget_exhausted"] is True


def test_variant_failure_returns_partial_data_without_complete_cache_hit(tmp_path) -> None:
    query = RetrievalQuery(
        topic="agents",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=10,
        page_size=10,
        max_requests=3,
    )
    plan = make_query_plan(
        QueryPlanVariant(label="first", search_query="all:agents", sort_by="relevance"),
        QueryPlanVariant(label="failing", search_query="ti:agents", sort_by="relevance"),
        QueryPlanVariant(label="third", search_query="abs:agents", sort_by="submittedDate"),
    )
    client = SequencedClient(
        [
            atom_feed([paper_entry("2604.10001", "First Agent Paper")]),
            TimeoutError("temporary network failure"),
            atom_feed([paper_entry("2604.10002", "Second Agent Paper")]),
        ]
    )
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    skill = ArxivRetrievalSkill(
        store=store,
        client=client,
        request_delay_seconds=0,
        retry_backoff_seconds=0,
        max_retries=0,
    )

    result = skill.retrieve(query, query_plan=plan)

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "arxiv_partial_failure"
    assert [paper.paper_id for paper in result.data or []] == [
        "2604.10001",
        "2604.10002",
    ]
    assert result.metadata["cache_status"] == RetrievalCacheStatus.PARTIAL.value
    assert result.metadata["partial_failures"]
    assert store.load_retrieval_result_set(query, query_plan=plan) is None
    partial = store.load_retrieval_result_set(
        query,
        query_plan=plan,
        accept_partial=True,
    )
    assert partial is not None
    assert partial.cache_status == RetrievalCacheStatus.PARTIAL


def test_partial_failure_does_not_overwrite_complete_plan_cache(tmp_path) -> None:
    query = RetrievalQuery(
        topic="agents",
        search_mode=SearchMode.BROAD,
        candidate_pool_size=10,
        page_size=10,
        max_requests=2,
    )
    plan = make_query_plan(
        QueryPlanVariant(label="first", search_query="all:agents", sort_by="relevance"),
        QueryPlanVariant(label="second", search_query="ti:agents", sort_by="submittedDate"),
    )
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    priming_skill = ArxivRetrievalSkill(
        store=store,
        client=SequencedClient([FIXTURE.read_text(), FIXTURE.read_text()]),
        request_delay_seconds=0,
    )
    priming_result = priming_skill.retrieve(query, query_plan=plan)
    assert priming_result.status == SkillStatus.SUCCESS

    failing_skill = ArxivRetrievalSkill(
        store=store,
        client=SequencedClient([FIXTURE.read_text(), "<feed><entry>"]),
        request_delay_seconds=0,
    )
    result = failing_skill.retrieve(query, query_plan=plan, use_cache=False)

    assert result.status == SkillStatus.FALLBACK
    loaded = store.load_retrieval_result_set(query, query_plan=plan)
    assert loaded is not None
    assert loaded.cache_status == RetrievalCacheStatus.COMPLETE
    assert [
        metadata.variant_label
        for metadata in loaded.source_metadata_by_paper_id["2604.00001"]
    ] == ["first", "second"]


def test_empty_arxiv_response_returns_successful_empty_result(tmp_path) -> None:
    empty_feed = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    skill = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        client=FakeClient(empty_feed),
        request_delay_seconds=0,
    )

    result = skill.retrieve(RetrievalQuery(topic="unlikely topic"))

    assert result.status == SkillStatus.EMPTY
    assert result.data == []
    assert result.message == "No arXiv papers matched the query."


def test_network_failure_returns_fallback_with_failed_query_metadata(tmp_path) -> None:
    query = RetrievalQuery(topic="agents", category="cs.LG")
    client = FailingClient()
    skill = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        client=client,
        request_delay_seconds=0,
        retry_backoff_seconds=0,
    )

    result = skill.retrieve(query)

    assert result.status == SkillStatus.FALLBACK
    assert result.data == []
    assert result.error is not None
    assert result.error.code == "arxiv_request_failed"
    assert result.error.retryable is True
    assert result.metadata["query"]["topic"] == "agents"
    assert client.calls == 3


def test_rate_limited_request_retries_then_succeeds(tmp_path) -> None:
    query = RetrievalQuery(topic="agents", category="cs.LG")
    client = RateLimitedClient(FIXTURE.read_text(), failures_before_success=1)
    skill = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        client=client,
        request_delay_seconds=0,
        retry_backoff_seconds=0,
    )

    result = skill.retrieve(query, use_cache=False)

    assert result.status == SkillStatus.SUCCESS
    assert len(result.data or []) == 2
    assert client.calls == 2


def test_malformed_atom_response_returns_fallback_without_corrupting_storage(tmp_path) -> None:
    query = RetrievalQuery(topic="agents", category="cs.LG")
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    skill = ArxivRetrievalSkill(
        store=store,
        client=FakeClient("<feed><entry>"),
        request_delay_seconds=0,
    )

    result = skill.retrieve(query)

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "arxiv_parse_failed"
    assert store.list_papers() == []


def test_malformed_atom_response_reuses_cached_results_when_available(tmp_path) -> None:
    query = RetrievalQuery(topic="agents", category="cs.LG")
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    priming_skill = ArxivRetrievalSkill(
        store=store,
        client=FakeClient(FIXTURE.read_text()),
        request_delay_seconds=0,
    )
    priming_result = priming_skill.retrieve(query)

    assert priming_result.status == SkillStatus.SUCCESS

    failing_skill = ArxivRetrievalSkill(
        store=store,
        client=FakeClient("<feed><entry>"),
        request_delay_seconds=0,
    )
    result = failing_skill.retrieve(query, use_cache=False)

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.code == "arxiv_parse_failed"
    assert result.metadata["cache_hit"] is True
    assert [paper.paper_id for paper in result.data or []] == ["2604.00001", "2604.00002"]
    assert [paper.paper_id for paper in store.load_retrieval(query)] == [
        "2604.00001",
        "2604.00002",
    ]


def test_parse_atom_rejects_malformed_xml() -> None:
    with pytest.raises(ValueError):
        parse_atom_response("<feed><entry>", RetrievalQuery(topic="agents"))


def make_query_plan(*variants: QueryPlanVariant) -> QueryPlan:
    return QueryPlan(
        search_mode=SearchMode.BROAD,
        planner=QueryPlannerProvenance(
            requested_mode=QueryPlannerMode.DETERMINISTIC,
            source="deterministic",
        ),
        variants=list(variants),
    )


def atom_feed(entries: list[str]) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  {"".join(entries)}
</feed>
"""


def paper_entry(paper_id: str, title: str) -> str:
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{paper_id}v1</id>
    <updated>2026-04-20T12:00:00Z</updated>
    <published>2026-04-20T10:00:00Z</published>
    <title>{title}</title>
    <summary>Agent retrieval metadata fixture.</summary>
    <author>
      <name>Ada Lovelace</name>
    </author>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/{paper_id}v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/{paper_id}v1" rel="related" type="application/pdf"/>
  </entry>
"""
