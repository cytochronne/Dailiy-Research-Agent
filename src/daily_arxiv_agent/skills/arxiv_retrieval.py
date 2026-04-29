"""arXiv retrieval Skill with Atom parsing and SQLite caching."""

from __future__ import annotations

from datetime import date, datetime
from email.utils import parsedate_to_datetime
import os
import re
import time
from typing import Any
from urllib import error
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
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
DEFAULT_ARXIV_USER_AGENT = "daily-arxiv-agent/0.1 (+local-debug)"
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class ArxivRetrievalSkill:
    """Retrieve, normalize, persist, and reuse arXiv paper metadata."""

    def __init__(
        self,
        *,
        store: SQLitePaperStore,
        client: Any | None = None,
        request_delay_seconds: float = 3.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 5.0,
        user_agent: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.store = store
        self.request_delay_seconds = max(request_delay_seconds, 0.0)
        self.max_retries = max(max_retries, 0)
        self.retry_backoff_seconds = max(retry_backoff_seconds, 0.0)
        resolved_user_agent = (
            user_agent.strip()
            if user_agent and user_agent.strip()
            else os.getenv("ARXIV_USER_AGENT", DEFAULT_ARXIV_USER_AGENT)
        )
        self.client = client or _UrllibClient(user_agent=resolved_user_agent)
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
            response = self._fetch_response(params)
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
            cached_after_failure = self.store.load_retrieval(query)
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.FALLBACK,
                data=cached_after_failure,
                evidence_source=EvidenceSource.METADATA,
                provenance=[paper.provenance for paper in cached_after_failure],
                error=SkillError(
                    code="arxiv_parse_failed",
                    message=str(exc),
                    retryable=False,
                ),
                message=(
                    "Using cached arXiv results after a parse failure."
                    if cached_after_failure
                    else "arXiv returned a response that could not be parsed."
                ),
                metadata={
                    "cache_hit": bool(cached_after_failure),
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

    def _fetch_response(self, params: dict[str, object]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            delay_seconds = (
                self.request_delay_seconds
                if attempt == 0
                else self._retry_delay_seconds(last_exc, attempt)
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)

            try:
                response = self.client.get(
                    ARXIV_API_URL,
                    params=params,
                    timeout=self.timeout_seconds,
                )
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not _is_retryable_request_exception(exc):
                    raise

        if last_exc is None:  # pragma: no cover - defensive guard.
            raise RuntimeError("arXiv request failed without an exception.")
        raise last_exc

    def _retry_delay_seconds(self, exc: Exception | None, attempt: int) -> float:
        retry_after = _retry_after_seconds(exc)
        backoff = self.retry_backoff_seconds * attempt
        return max(self.request_delay_seconds, retry_after or 0.0, backoff)


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
        source_url = _entry_source_url(entry, paper_id)
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
                    source_url=source_url,
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
            return _normalize_arxiv_url(href)
    return f"https://arxiv.org/pdf/{paper_id}"


def _entry_source_url(entry: ET.Element, paper_id: str) -> str:
    for link in entry.findall(f"{ATOM_NS}link"):
        rel = link.attrib.get("rel", "")
        media_type = link.attrib.get("type", "")
        href = link.attrib.get("href")
        if href and rel == "alternate" and media_type == "text/html":
            return _normalize_arxiv_url(href)

    entry_id = _optional_text(entry, "id")
    if entry_id:
        return _normalize_arxiv_url(entry_id)
    return f"https://arxiv.org/abs/{paper_id}"


def _normalize_arxiv_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if not parts.netloc:
        return value.strip()
    return urlunsplit(
        (
            "https",
            parts.netloc,
            parts.path.rstrip("/"),
            "",
            "",
        )
    )


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
    def __init__(self, *, user_agent: str) -> None:
        self.user_agent = user_agent

    def get(self, url: str, *, params: dict[str, object], timeout: float) -> _TextResponse:
        query_string = urlencode(params)
        req = Request(
            f"{url}?{query_string}",
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/atom+xml",
            },
        )
        with urlopen(req, timeout=timeout) as response:
            return _TextResponse(response.read().decode("utf-8"))


def _is_retryable_request_exception(exc: Exception) -> bool:
    if isinstance(exc, error.HTTPError):
        return exc.code in RETRYABLE_STATUS_CODES
    return isinstance(exc, (error.URLError, TimeoutError))


def _retry_after_seconds(exc: Exception | None) -> float | None:
    if not isinstance(exc, error.HTTPError) or exc.headers is None:
        return None

    raw_value = exc.headers.get("Retry-After")
    if not raw_value:
        return None

    try:
        return max(float(raw_value), 0.0)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=datetime.now().astimezone().tzinfo)
    delta = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
    return max(delta, 0.0)
