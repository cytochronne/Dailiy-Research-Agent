"""arXiv retrieval Skill with Atom parsing and SQLite caching."""

from __future__ import annotations

from dataclasses import dataclass
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
    QueryPlan,
    QueryPlanVariant,
    RetrievalCacheStatus,
    RetrievalQuery,
    RetrievalResultSet,
    RetrievalSourceMetadata,
    SearchMode,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.skills.query_planning import build_deterministic_query_plan
from daily_arxiv_agent.storage import SQLitePaperStore


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
DEFAULT_ARXIV_USER_AGENT = "daily-arxiv-agent/0.1 (+local-debug)"
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class _VariantCursor:
    variant: QueryPlanVariant
    variant_index: int
    next_start: int
    exhausted: bool = False


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
        query_plan: QueryPlan | None = None,
    ) -> SkillResult[list[PaperMetadata]]:
        supplied_query_plan = query_plan is not None
        effective_plan = query_plan or _default_query_plan(query)

        if use_cache:
            cached = self.store.load_retrieval_result_set(
                query,
                query_plan=effective_plan,
            )
            if cached is not None:
                return _result_from_cached_set(query, effective_plan, cached)

            if not supplied_query_plan:
                legacy_cached = self.store.load_retrieval_result_set(query)
                if legacy_cached is not None:
                    return _result_from_cached_set(query, effective_plan, legacy_cached)

        fallback_cache = self.store.load_retrieval_result_set(
            query,
            query_plan=effective_plan,
            accept_partial=True,
        )
        if fallback_cache is None and not supplied_query_plan:
            fallback_cache = self.store.load_retrieval_result_set(
                query,
                accept_partial=True,
            )

        run = self._fetch_planned_results(query, effective_plan)
        papers = run["papers"]
        source_metadata_by_paper_id = run["source_metadata_by_paper_id"]
        partial_failures = run["partial_failures"]
        cache_status = (
            RetrievalCacheStatus.PARTIAL
            if partial_failures
            else RetrievalCacheStatus.COMPLETE
        )
        metadata = _metadata(
            query,
            effective_plan,
            cache_hit=False,
            request_params=run["request_params"],
            request_count=run["request_count"],
            candidate_count=len(papers),
            candidate_target=query.effective_candidate_pool_size,
            cache_status=cache_status,
            budget_exhausted=run["budget_exhausted"],
            source_metadata_by_paper_id=source_metadata_by_paper_id,
            partial_failures=partial_failures,
        )

        if papers or not partial_failures:
            self.store.save_retrieval_result_set(
                query,
                papers,
                query_plan=effective_plan,
                source_metadata_by_paper_id=source_metadata_by_paper_id,
                cache_status=cache_status,
                metadata=metadata,
            )
            if not supplied_query_plan and cache_status == RetrievalCacheStatus.COMPLETE:
                self.store.save_retrieval_result_set(
                    query,
                    papers,
                    metadata=metadata,
                )

        if partial_failures and papers:
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.FALLBACK,
                data=papers,
                evidence_source=EvidenceSource.METADATA,
                provenance=[paper.provenance for paper in papers],
                error=SkillError(
                    code="arxiv_partial_failure",
                    message="One or more arXiv query variants failed; returning partial retrieval results.",
                    retryable=any(failure["retryable"] for failure in partial_failures),
                ),
                message="Returning partial arXiv results after one or more variant failures.",
                metadata=metadata,
            )

        if partial_failures:
            cached_papers = fallback_cache.papers if fallback_cache is not None else []
            failure = partial_failures[0]
            return SkillResult[list[PaperMetadata]](
                status=SkillStatus.FALLBACK,
                data=cached_papers,
                evidence_source=EvidenceSource.METADATA,
                provenance=[paper.provenance for paper in cached_papers],
                error=SkillError(
                    code=failure["code"],
                    message=failure["message"],
                    retryable=failure["retryable"],
                ),
                message=(
                    "Using cached arXiv results."
                    if cached_papers
                    else "arXiv request failed and no cached results are available."
                ),
                metadata={
                    **metadata,
                    "cache_hit": bool(cached_papers),
                    "cache_status": (
                        fallback_cache.cache_status.value
                        if fallback_cache is not None
                        else cache_status.value
                    ),
                },
            )

        status = SkillStatus.SUCCESS if papers else SkillStatus.EMPTY
        return SkillResult[list[PaperMetadata]](
            status=status,
            data=papers,
            evidence_source=EvidenceSource.METADATA,
            provenance=[paper.provenance for paper in papers],
            message=None if papers else "No arXiv papers matched the query.",
            metadata=metadata,
        )

    def _fetch_planned_results(
        self,
        query: RetrievalQuery,
        query_plan: QueryPlan,
    ) -> dict[str, Any]:
        candidate_target = query.effective_candidate_pool_size
        cursors = [
            _VariantCursor(
                variant=variant,
                variant_index=index,
                next_start=query.start_index,
            )
            for index, variant in enumerate(query_plan.variants)
        ]
        papers_by_id: dict[str, PaperMetadata] = {}
        first_seen_order_by_id: dict[str, int] = {}
        ordered_ids: list[str] = []
        source_metadata_by_paper_id: dict[str, list[RetrievalSourceMetadata]] = {}
        request_params: list[dict[str, object]] = []
        partial_failures: list[dict[str, Any]] = []
        request_count = 0

        while (
            request_count < query.max_requests
            and len(ordered_ids) < candidate_target
            and any(not cursor.exhausted for cursor in cursors)
        ):
            made_request = False
            for cursor in cursors:
                if (
                    cursor.exhausted
                    or request_count >= query.max_requests
                    or len(ordered_ids) >= candidate_target
                ):
                    continue

                remaining = max(candidate_target - len(ordered_ids), 1)
                page_size = min(query.page_size, remaining)
                params = build_arxiv_request_params(
                    query,
                    variant=cursor.variant,
                    start_index=cursor.next_start,
                    max_results=page_size,
                )
                request_params.append(params)
                request_count += 1
                made_request = True

                try:
                    response = self._fetch_response(params)
                except Exception as exc:
                    partial_failures.append(
                        _failure_metadata(
                            code="arxiv_request_failed",
                            message=f"arXiv request failed: {exc}",
                            retryable=_is_retryable_request_exception(exc),
                            cursor=cursor,
                        )
                    )
                    cursor.exhausted = True
                    continue

                try:
                    page_papers = parse_atom_response(
                        response.text,
                        query,
                        search_query=cursor.variant.search_query,
                    )
                except ValueError as exc:
                    partial_failures.append(
                        _failure_metadata(
                            code="arxiv_parse_failed",
                            message=str(exc),
                            retryable=False,
                            cursor=cursor,
                        )
                    )
                    cursor.exhausted = True
                    continue

                for position, paper in enumerate(page_papers):
                    if paper.paper_id not in papers_by_id:
                        if len(ordered_ids) >= candidate_target:
                            break
                        papers_by_id[paper.paper_id] = paper
                        first_seen_order_by_id[paper.paper_id] = len(ordered_ids)
                        ordered_ids.append(paper.paper_id)

                    source_metadata_by_paper_id.setdefault(paper.paper_id, []).append(
                        RetrievalSourceMetadata(
                            variant_label=cursor.variant.label,
                            sort_by=cursor.variant.sort_by,
                            variant_index=cursor.variant_index,
                            position=cursor.next_start + position,
                            first_seen_order=first_seen_order_by_id[paper.paper_id],
                            query=cursor.variant.search_query,
                        )
                    )

                if len(page_papers) < page_size:
                    cursor.exhausted = True
                else:
                    cursor.next_start += page_size

            if not made_request:
                break

        papers = [papers_by_id[paper_id] for paper_id in ordered_ids]
        return {
            "papers": papers,
            "source_metadata_by_paper_id": source_metadata_by_paper_id,
            "request_params": request_params,
            "request_count": request_count,
            "partial_failures": partial_failures,
            "budget_exhausted": request_count >= query.max_requests
            and len(papers) < candidate_target,
        }

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


def build_arxiv_request_params(
    query: RetrievalQuery,
    *,
    variant: QueryPlanVariant | None = None,
    start_index: int | None = None,
    max_results: int | None = None,
) -> dict[str, object]:
    """Build arXiv API query parameters from normalized retrieval inputs."""

    return {
        "search_query": (
            variant.search_query
            if variant is not None
            else build_arxiv_search_query(query)
        ),
        "start": query.start_index if start_index is None else start_index,
        "max_results": query.max_results if max_results is None else max_results,
        "sortBy": variant.sort_by if variant is not None else "submittedDate",
        "sortOrder": variant.sort_order if variant is not None else "descending",
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


def parse_atom_response(
    xml_text: str,
    query: RetrievalQuery,
    *,
    search_query: str | None = None,
) -> list[PaperMetadata]:
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
                    query=search_query or build_arxiv_search_query(query),
                ),
            )
        )
    return papers


def _default_query_plan(query: RetrievalQuery) -> QueryPlan:
    max_variants = query.max_requests if query.search_mode == SearchMode.BROAD else 1
    return build_deterministic_query_plan(query, max_variants=max_variants)


def _result_from_cached_set(
    query: RetrievalQuery,
    query_plan: QueryPlan,
    cached: RetrievalResultSet,
) -> SkillResult[list[PaperMetadata]]:
    request_params = _cached_request_params(cached.metadata)
    metadata = _metadata(
        query,
        query_plan,
        cache_hit=True,
        request_params=request_params,
        request_count=0,
        candidate_count=len(cached.papers),
        candidate_target=query.effective_candidate_pool_size,
        cache_status=cached.cache_status,
        budget_exhausted=False,
        source_metadata_by_paper_id=cached.source_metadata_by_paper_id,
        partial_failures=[],
        effective_query_key=cached.effective_query_key,
    )
    return SkillResult[list[PaperMetadata]](
        status=SkillStatus.SUCCESS if cached.papers else SkillStatus.EMPTY,
        data=cached.papers,
        evidence_source=EvidenceSource.METADATA,
        provenance=[paper.provenance for paper in cached.papers],
        message="Loaded cached arXiv retrieval results.",
        metadata=metadata,
    )


def _cached_request_params(metadata: dict[str, Any]) -> list[dict[str, object]]:
    params = (
        metadata.get("request_params_by_variant")
        or metadata.get("request_params")
        or []
    )
    if isinstance(params, dict):
        return [params]
    if not isinstance(params, list):
        return []
    return [item for item in params if isinstance(item, dict)]


def _metadata(
    query: RetrievalQuery,
    query_plan: QueryPlan,
    *,
    cache_hit: bool,
    request_params: list[dict[str, object]],
    request_count: int,
    candidate_count: int,
    candidate_target: int,
    cache_status: RetrievalCacheStatus,
    budget_exhausted: bool,
    source_metadata_by_paper_id: dict[str, list[RetrievalSourceMetadata]],
    partial_failures: list[dict[str, Any]],
    effective_query_key: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "cache_hit": cache_hit,
        "query": query.model_dump(mode="json"),
        "query_plan": query_plan.model_dump(mode="json"),
        "query_variant_count": query_plan.variant_count,
        "request_params": request_params[0] if len(request_params) == 1 else request_params,
        "request_params_by_variant": request_params,
        "request_count": request_count,
        "candidate_count": candidate_count,
        "candidate_target": candidate_target,
        "cache_status": cache_status.value,
        "budget_exhausted": budget_exhausted,
        "partial_failures": partial_failures,
        "source_metadata_by_paper_id": {
            paper_id: [
                source_metadata.model_dump(mode="json")
                for source_metadata in source_metadata_list
            ]
            for paper_id, source_metadata_list in source_metadata_by_paper_id.items()
        },
    }
    metadata["effective_query_key"] = effective_query_key or SQLitePaperStore.effective_query_key(
        query,
        query_plan=query_plan,
    )
    return metadata


def _failure_metadata(
    *,
    code: str,
    message: str,
    retryable: bool,
    cursor: _VariantCursor,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "retryable": retryable,
        "variant_label": cursor.variant.label,
        "variant_index": cursor.variant_index,
        "sort_by": cursor.variant.sort_by,
        "query": cursor.variant.search_query,
    }


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
