from datetime import date
from pathlib import Path

import pytest

from daily_arxiv_agent.contracts import SkillStatus, RetrievalQuery
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


class FailingClient:
    def get(self, url: str, params: dict[str, object], timeout: float) -> FakeResponse:
        raise TimeoutError("network unavailable")


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
    assert str(papers[0].pdf_url) == "https://arxiv.org/pdf/2604.00001"
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
    skill = ArxivRetrievalSkill(
        store=SQLitePaperStore(tmp_path / "papers.sqlite3"),
        client=FailingClient(),
        request_delay_seconds=0,
    )

    result = skill.retrieve(query)

    assert result.status == SkillStatus.FALLBACK
    assert result.data == []
    assert result.error is not None
    assert result.error.code == "arxiv_request_failed"
    assert result.error.retryable is True
    assert result.metadata["query"]["topic"] == "agents"


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


def test_parse_atom_rejects_malformed_xml() -> None:
    with pytest.raises(ValueError):
        parse_atom_response("<feed><entry>", RetrievalQuery(topic="agents"))
