"""arXiv retrieval Skill with Atom parsing and SQLite caching."""

from __future__ import annotations

from datetime import date, datetime
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    RetrievalQuery,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.storage import SQLitePaperStore


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivRetrievalSkill:
    """Retrieve, normalize, persist, and reuse arXiv paper metadata."""

    def __init__(
        self,
        *,
        store: SQLitePaperStore,
        client: Any | None = None,
        request_delay_seconds: float = 3.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.store = store
        self.client = client or _UrllibClient()
        self.request_delay_seconds = request_delay_seconds
        self.timeout_seconds = timeout_seconds

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        use_cache: bool = True,
    ) -> SkillResult[list[PaperMetadata]]:
        cached = self.store.load_retrieval(query) if use_cache else []
        if cached:
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.SUCCESS,
                data=cached,
                evidence_source=EvidenceSource.METADATA,
                provenance=[paper.provenance for paper in cached],
                message="Loaded cached arXiv retrieval results.",
                metadata={
                    "cache_hit": True,
                    "query": query.model_dump(mode="json"),
                    "request_params": build_arxiv_request_params(query),
                },
            )

        params = build_arxiv_request_params(query)
        try:
            if self.request_delay_seconds > 0:
                time.sleep(self.request_delay_seconds)
            response = self.client.get(
                ARXIV_API_URL,
                params=params,
                timeout=self.timeout_seconds,
            )
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
        except Exception as exc:
            cached_after_failure = self.store.load_retrieval(query)
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.FALLBACK,
                data=cached_after_failure,
                evidence_source=EvidenceSource.METADATA,
                provenance=[paper.provenance for paper in cached_after_failure],
                error=SkillError(
                    code="arxiv_request_failed",
                    message=f"arXiv request failed: {exc}",
                    retryable=True,
                ),
                message=(
                    "Using cached arXiv results."
                    if cached_after_failure
                    else "arXiv request failed and no cached results are available."
                ),
                metadata={
                    "cache_hit": bool(cached_after_failure),
                    "query": query.model_dump(mode="json"),
                    "request_params": params,
                },
            )

        try:
            papers = parse_atom_response(response.text, query)
        except ValueError as exc:
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.FALLBACK,
                data=[],
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code="arxiv_parse_failed",
                    message=str(exc),
                    retryable=False,
                ),
                message="arXiv returned a response that could not be parsed.",
                metadata={
                    "cache_hit": False,
                    "query": query.model_dump(mode="json"),
                    "request_params": params,
                },
            )

        self.store.save_retrieval(query, papers)
        status = SkillStatus.SUCCESS if papers else SkillStatus.EMPTY
        return SkillResult[list[PaperMetadata]](
            status=status,
            data=papers,
            evidence_source=EvidenceSource.METADATA,
            provenance=[paper.provenance for paper in papers],
            message=None if papers else "No arXiv papers matched the query.",
            metadata={
                "cache_hit": False,
                "query": query.model_dump(mode="json"),
                "request_params": params,
            },
        )


def build_arxiv_request_params(query: RetrievalQuery) -> dict[str, object]:
    """Build arXiv API query parameters from normalized retrieval inputs."""

    return {
        "search_query": build_arxiv_search_query(query),
        "start": query.start_index,
        "max_results": query.max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }


def build_arxiv_search_query(query: RetrievalQuery) -> str:
    clauses: list[str] = []
    if query.topic:
        escaped_topic = query.topic.replace('"', '\\"')
        clauses.append(f'all:"{escaped_topic}"')
    if query.category:
        clauses.append(f"cat:{query.category}")
    if query.start_date or query.end_date:
        start = _date_floor(query.start_date)
        end = _date_ceiling(query.end_date)
        clauses.append(f"submittedDate:[{start} TO {end}]")
    return " AND ".join(clauses) if clauses else "all:*"


def parse_atom_response(xml_text: str, query: RetrievalQuery) -> list[PaperMetadata]:
    """Parse arXiv Atom XML into normalized paper metadata."""

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed arXiv Atom response: {exc}") from exc

    papers: list[PaperMetadata] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        paper_id = _extract_paper_id(_required_text(entry, "id"))
        title = _clean_text(_required_text(entry, "title"))
        abstract = _clean_text(_optional_text(entry, "summary"))
        authors = [
            _clean_text(name.text or "")
            for name in entry.findall(f"{ATOM_NS}author/{ATOM_NS}name")
            if _clean_text(name.text or "")
        ]
        categories = _unique(
            category.attrib["term"]
            for category in entry.findall(f"{ATOM_NS}category")
            if category.attrib.get("term")
        )
        arxiv_url = f"https://arxiv.org/abs/{paper_id}"
        pdf_url = _pdf_url(entry, paper_id)
        search_query = build_arxiv_search_query(query)
        papers.append(
            PaperMetadata(
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract or None,
                categories=categories,
                published_date=_parse_atom_date(_optional_text(entry, "published")),
                updated_date=_parse_atom_date(_optional_text(entry, "updated")),
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
                provenance=Provenance(
                    source="arxiv",
                    source_url=arxiv_url,
                    query=search_query,
                ),
            )
        )
    return papers


def _date_floor(value: date | None) -> str:
    return f"{value:%Y%m%d}0000" if value else "000101010000"


def _date_ceiling(value: date | None) -> str:
    return f"{value:%Y%m%d}2359" if value else "999912312359"


def _required_text(entry: ET.Element, tag: str) -> str:
    text = _optional_text(entry, tag)
    if not text:
        raise ValueError(f"arXiv entry is missing required field: {tag}")
    return text


def _optional_text(entry: ET.Element, tag: str) -> str:
    child = entry.find(f"{ATOM_NS}{tag}")
    return child.text if child is not None and child.text is not None else ""


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _extract_paper_id(entry_id: str) -> str:
    match = re.search(r"/abs/([^/?#]+)", entry_id)
    raw_id = match.group(1) if match else entry_id.rsplit("/", maxsplit=1)[-1]
    return re.sub(r"v\d+$", "", raw_id)


def _pdf_url(entry: ET.Element, paper_id: str) -> str:
    for link in entry.findall(f"{ATOM_NS}link"):
        title = link.attrib.get("title", "")
        media_type = link.attrib.get("type", "")
        href = link.attrib.get("href")
        if href and (title == "pdf" or media_type == "application/pdf"):
            return f"https://arxiv.org/pdf/{paper_id}"
    return f"https://arxiv.org/pdf/{paper_id}"


def _parse_atom_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


class _TextResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _UrllibClient:
    def get(self, url: str, *, params: dict[str, object], timeout: float) -> _TextResponse:
        query_string = urlencode(params)
        with urlopen(f"{url}?{query_string}", timeout=timeout) as response:
            return _TextResponse(response.read().decode("utf-8"))
