"""OpenAI-compatible LLM provider for extraction and briefing generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from email.utils import parsedate_to_datetime
import json
import time
from typing import Any, TypeVar
from urllib import error, request

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExperimentExplanation,
    ExplanationMode,
    LimitationsExplanation,
    MethodExplanation,
    PaperBriefingItem,
    PaperDeepExplanation,
    PaperMetadata,
    Recommendation,
    RetrievalQuery,
)

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
ResponseT = TypeVar("ResponseT")


class OpenAILLMProvider:
    """Call OpenAI-compatible Chat Completions APIs and map output contracts."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        chat_completions_path: str = "/chat/completions",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        output_retries: int = 1,
        briefing_max_retries: int | None = None,
        briefing_retry_backoff_seconds: float | None = None,
        briefing_output_retries: int | None = None,
    ) -> None:
        if api_key is not None and not api_key.strip():
            raise ValueError("LLM_API_KEY cannot be blank when provided.")
        self.api_key = api_key.strip() if api_key else None
        self.model = model
        self.base_url = base_url.rstrip("/")
        normalized_path = chat_completions_path.strip() or "/chat/completions"
        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"
        self.chat_completions_path = normalized_path
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(max_retries, 0)
        self.retry_backoff_seconds = max(retry_backoff_seconds, 0.0)
        self.output_retries = max(output_retries, 0)
        self.briefing_max_retries = max(
            briefing_max_retries if briefing_max_retries is not None else self.max_retries + 2,
            0,
        )
        self.briefing_retry_backoff_seconds = max(
            briefing_retry_backoff_seconds
            if briefing_retry_backoff_seconds is not None
            else self.retry_backoff_seconds,
            0.0,
        )
        self.briefing_output_retries = max(
            briefing_output_retries
            if briefing_output_retries is not None
            else self.output_retries + 1,
            0,
        )

    def extract_paper(
        self,
        paper: PaperMetadata,
        *,
        topic: str,
        recommendation: Recommendation | None = None,
    ) -> PaperBriefingItem:
        evidence_source = (
            EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA
        )
        rank = recommendation.rank if recommendation else 1
        score = recommendation.score if recommendation else 0.0
        rationale = (
            recommendation.rationale
            if recommendation
            else f"Paper metadata is being reviewed for topic '{topic}'."
        )

        if not paper.abstract:
            return PaperBriefingItem(
                paper_id=paper.paper_id,
                title=paper.title,
                rank=rank,
                score=score,
                summary=(
                    f"Metadata only: '{paper.title}' has no abstract available, so the "
                    "briefing is limited to title, category, and provenance fields."
                ),
                contributions=[
                    "Metadata indicates a potentially relevant paper, but abstract-level "
                    "claims are unavailable."
                ],
                methods=[],
                relevance_rationale=f"{rationale} Evidence: {evidence_source.value}.",
                evidence_source=evidence_source,
                provenance=paper.provenance,
                arxiv_url=paper.arxiv_url,
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract structured research-paper briefing fields. "
                        "Return strict JSON with keys: summary, contributions, methods."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_extraction_prompt(
                        paper=paper,
                        topic=topic,
                        recommendation=recommendation,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        summary, contributions, methods = self._complete_with_validation(
            payload,
            validator=_parse_extraction_response,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            output_retries=self.output_retries,
        )

        return PaperBriefingItem(
            paper_id=paper.paper_id,
            title=paper.title,
            rank=rank,
            score=score,
            summary=summary,
            contributions=contributions or ["No explicit contribution extracted."],
            methods=methods,
            relevance_rationale=f"{rationale} Evidence: {evidence_source.value}.",
            evidence_source=evidence_source,
            provenance=paper.provenance,
            arxiv_url=paper.arxiv_url,
        )

    def summarize_briefing(
        self,
        *,
        topic: str,
        items: Sequence[PaperBriefingItem],
    ) -> str:
        if not items:
            return f"No ranked papers were available for '{topic}'."

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write concise executive summaries for daily research briefings. "
                        "Use 2-3 sentences, avoid hype, and stay evidence-grounded."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_summary_prompt(topic=topic, items=items),
                },
            ],
            "temperature": 0.2,
        }
        return self._complete_with_validation(
            payload,
            validator=_parse_summary_response,
            max_retries=self.briefing_max_retries,
            retry_backoff_seconds=self.briefing_retry_backoff_seconds,
            output_retries=self.briefing_output_retries,
        )

    def explain_paper(
        self,
        paper: PaperMetadata,
        *,
        mode: ExplanationMode,
        content: str,
        evidence_source: EvidenceSource,
    ) -> PaperDeepExplanation:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain research papers using only the provided source text. "
                        "Do not invent details. When evidence is missing, explicitly say "
                        f"'Not found in the available {_source_label(evidence_source)} source.' "
                        "Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_explanation_prompt(
                        paper=paper,
                        mode=mode,
                        content=content,
                        evidence_source=evidence_source,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        return self._complete_with_validation(
            payload,
            validator=lambda response: _parse_explanation_response(
                response,
                paper=paper,
                mode=mode,
                evidence_source=evidence_source,
            ),
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            output_retries=self.output_retries,
        )

    def plan_queries(
        self,
        *,
        query: RetrievalQuery,
        deterministic_terms: Sequence[str],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You plan bounded arXiv metadata search queries. "
                        "Return strict JSON only and avoid raw arXiv query syntax."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_query_planning_prompt(
                        query=query,
                        deterministic_terms=deterministic_terms,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        plan = self._complete_with_validation(
            payload,
            validator=_parse_query_planning_response,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            output_retries=self.output_retries,
        )
        plan.setdefault("source", "llm")
        plan.setdefault("model", self.model)
        return plan

    def _chat_completion(
        self,
        payload: dict[str, Any],
        *,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.base_url}{self.chat_completions_path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        raw = self._post_with_retries(
            endpoint=endpoint,
            body=body,
            headers=headers,
            max_retries=self.max_retries if max_retries is None else max(max_retries, 0),
            retry_backoff_seconds=(
                self.retry_backoff_seconds
                if retry_backoff_seconds is None
                else max(retry_backoff_seconds, 0.0)
            ),
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM API returned non-JSON response.") from exc

    def _post_with_retries(
        self,
        *,
        endpoint: str,
        body: bytes,
        headers: dict[str, str],
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> str:
        for attempt in range(max_retries + 1):
            req = request.Request(
                endpoint,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    return resp.read().decode("utf-8")
            except error.HTTPError as exc:  # pragma: no cover - network path
                if attempt < max_retries and exc.code in RETRYABLE_STATUS_CODES:
                    self._sleep_before_retry(exc, attempt + 1, retry_backoff_seconds)
                    continue
                detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
                raise RuntimeError(
                    f"LLM API HTTP {exc.code}: {detail or exc.reason}"
                ) from exc
            except error.URLError as exc:  # pragma: no cover - network path
                if attempt < max_retries:
                    self._sleep_before_retry(exc, attempt + 1, retry_backoff_seconds)
                    continue
                raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

        raise RuntimeError("LLM API request failed after retries.")

    def _complete_with_validation(
        self,
        payload: dict[str, Any],
        *,
        validator: Callable[[dict[str, Any]], ResponseT],
        max_retries: int,
        retry_backoff_seconds: float,
        output_retries: int,
    ) -> ResponseT:
        last_exc: Exception | None = None
        for attempt in range(output_retries + 1):
            response = self._chat_completion(
                payload,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            try:
                return validator(response)
            except Exception as exc:
                last_exc = exc
                if attempt >= output_retries or not _is_retryable_output_exception(exc):
                    raise
                delay_seconds = retry_backoff_seconds * (attempt + 1)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

        if last_exc is None:  # pragma: no cover - defensive guard.
            raise RuntimeError("LLM validation failed without an exception.")
        raise last_exc

    def _sleep_before_retry(
        self,
        exc: Exception,
        attempt: int,
        retry_backoff_seconds: float,
    ) -> None:
        delay_seconds = max(
            retry_backoff_seconds * attempt,
            _retry_after_seconds(exc) or 0.0,
        )
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def _build_extraction_prompt(
    *,
    paper: PaperMetadata,
    topic: str,
    recommendation: Recommendation | None,
) -> str:
    return (
        f"Topic: {topic}\n"
        f"Title: {paper.title}\n"
        f"Paper ID: {paper.paper_id}\n"
        f"Categories: {', '.join(paper.categories)}\n"
        f"Abstract: {paper.abstract or ''}\n"
        f"Ranking rationale: {recommendation.rationale if recommendation else ''}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "summary": "<=2 sentences",\n'
        '  "contributions": ["bullet", "bullet"],\n'
        '  "methods": ["bullet", "bullet"]\n'
        "}\n"
        "Do not include markdown or extra keys."
    )


def _build_summary_prompt(*, topic: str, items: Sequence[PaperBriefingItem]) -> str:
    lines = [f"Topic: {topic}", "Top papers:"]
    for item in items[:5]:
        lines.append(
            f"- Rank {item.rank}: {item.title} | score={item.score:.3f} | "
            f"evidence={item.evidence_source.value} | summary={item.summary}"
        )
    lines.append("Write an executive summary in 2-3 sentences.")
    return "\n".join(lines)


def _build_explanation_prompt(
    *,
    paper: PaperMetadata,
    mode: ExplanationMode,
    content: str,
    evidence_source: EvidenceSource,
) -> str:
    return (
        f"Paper ID: {paper.paper_id}\n"
        f"Title: {paper.title}\n"
        f"Explanation mode: {mode.value}\n"
        f"Evidence source: {evidence_source.value}\n"
        f"Source text:\n{_clip_text(content, limit=20000)}\n\n"
        f"Return JSON only with this schema:\n{_explanation_schema(mode, evidence_source)}\n"
        "No markdown. No extra keys."
    )


def _build_query_planning_prompt(
    *,
    query: RetrievalQuery,
    deterministic_terms: Sequence[str],
) -> str:
    start_date = query.start_date.isoformat() if query.start_date else ""
    end_date = query.end_date.isoformat() if query.end_date else ""
    return (
        "Use only the search intent fields below. Do not add paper-specific evidence.\n"
        f"Topic: {query.topic or ''}\n"
        f"Category filter: {query.category or ''}\n"
        f"Start date filter: {start_date}\n"
        f"End date filter: {end_date}\n"
        f"Search mode: {query.search_mode.value}\n"
        f"Deterministic required terms: {', '.join(deterministic_terms)}\n\n"
        "Return JSON only with these keys:\n"
        "{\n"
        '  "required_terms": ["term"],\n'
        '  "phrases": ["short phrase"],\n'
        '  "related_terms": ["term"],\n'
        '  "suggested_categories": ["cs.LG"],\n'
        '  "exclusions": ["term"],\n'
        '  "rationale": "<=1 sentence"\n'
        "}\n"
        "Limits: required_terms <= 8, phrases <= 4, related_terms <= 8, "
        "suggested_categories <= 6. Terms should be words or short noun phrases, "
        "not Boolean expressions."
    )


def _extract_content_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM API response did not include choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM API response choice is missing message.")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return ""


def _parse_json_content(payload: dict[str, Any]) -> dict[str, Any]:
    text = _extract_content_text(payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM extraction did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM extraction JSON root must be an object.")
    return parsed


def _parse_extraction_response(
    payload: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    parsed = _parse_json_content(payload)
    summary = _clean_text(parsed.get("summary"))
    contributions = _normalize_list(parsed.get("contributions"))
    methods = _normalize_list(parsed.get("methods"))
    if not summary:
        raise RuntimeError("LLM extraction returned empty summary.")
    return summary, contributions, methods


def _parse_summary_response(payload: dict[str, Any]) -> str:
    summary = _clean_text(_extract_content_text(payload))
    if not summary:
        raise RuntimeError("LLM briefing summary returned empty content.")
    return summary


def _parse_query_planning_response(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = _parse_json_content(payload)
    except RuntimeError as exc:
        raise RuntimeError("LLM query planning did not return valid JSON.") from exc

    plan = {
        "required_terms": _normalize_list(parsed.get("required_terms")),
        "phrases": _normalize_list(parsed.get("phrases")),
        "related_terms": _normalize_list(parsed.get("related_terms")),
        "suggested_categories": _normalize_list(parsed.get("suggested_categories")),
        "exclusions": _normalize_list(parsed.get("exclusions")),
        "rationale": _clean_text(parsed.get("rationale")),
    }
    if not plan["required_terms"] and not plan["phrases"] and not plan["related_terms"]:
        raise RuntimeError("LLM query planning returned no usable terms.")
    return plan


def _parse_explanation_response(
    payload: dict[str, Any],
    *,
    paper: PaperMetadata,
    mode: ExplanationMode,
    evidence_source: EvidenceSource,
) -> PaperDeepExplanation:
    parsed = _parse_json_content(payload)
    summary = _clean_text(parsed.get("summary"))
    if not summary:
        raise RuntimeError("LLM explanation returned empty summary.")

    evidence_note = (
        f"This explanation is based on the available {_source_label(evidence_source)} source."
    )
    if mode == ExplanationMode.METHOD:
        return PaperDeepExplanation(
            paper_id=paper.paper_id,
            title=paper.title,
            mode=mode,
            summary=summary,
            evidence_source=evidence_source,
            evidence_note=evidence_note,
            method=MethodExplanation(
                problem=_required_text(
                    parsed.get("problem"),
                    subject="problem statement",
                    evidence_source=evidence_source,
                ),
                method_overview=_required_text(
                    parsed.get("method_overview"),
                    subject="method overview",
                    evidence_source=evidence_source,
                ),
                core_workflow=_required_list(
                    parsed.get("core_workflow"),
                    subject="core workflow",
                    evidence_source=evidence_source,
                ),
                inputs_outputs=_required_list(
                    parsed.get("inputs_outputs"),
                    subject="inputs and outputs",
                    evidence_source=evidence_source,
                ),
                innovation=_required_text(
                    parsed.get("innovation"),
                    subject="claimed innovation",
                    evidence_source=evidence_source,
                ),
            ),
            provenance=paper.provenance,
            arxiv_url=paper.arxiv_url,
        )
    if mode == ExplanationMode.EXPERIMENT:
        return PaperDeepExplanation(
            paper_id=paper.paper_id,
            title=paper.title,
            mode=mode,
            summary=summary,
            evidence_source=evidence_source,
            evidence_note=evidence_note,
            experiment=ExperimentExplanation(
                datasets=_required_list(
                    parsed.get("datasets"),
                    subject="datasets",
                    evidence_source=evidence_source,
                ),
                baselines=_required_list(
                    parsed.get("baselines"),
                    subject="baselines",
                    evidence_source=evidence_source,
                ),
                metrics=_required_list(
                    parsed.get("metrics"),
                    subject="metrics",
                    evidence_source=evidence_source,
                ),
                experimental_setup=_required_text(
                    parsed.get("experimental_setup"),
                    subject="experimental setup",
                    evidence_source=evidence_source,
                ),
                conclusions=_required_list(
                    parsed.get("conclusions"),
                    subject="main conclusions",
                    evidence_source=evidence_source,
                ),
            ),
            provenance=paper.provenance,
            arxiv_url=paper.arxiv_url,
        )
    return PaperDeepExplanation(
        paper_id=paper.paper_id,
        title=paper.title,
        mode=mode,
        summary=summary,
        evidence_source=evidence_source,
        evidence_note=evidence_note,
        limitations=LimitationsExplanation(
            stated_limitations=_required_list(
                parsed.get("stated_limitations"),
                subject="stated limitations",
                evidence_source=evidence_source,
            ),
            assumptions=_required_list(
                parsed.get("assumptions"),
                subject="assumptions",
                evidence_source=evidence_source,
            ),
            missing_validation=_required_list(
                parsed.get("missing_validation"),
                subject="missing validation",
                evidence_source=evidence_source,
            ),
            risks=_required_list(
                parsed.get("risks"),
                subject="possible risks",
                evidence_source=evidence_source,
            ),
        ),
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )


def _normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        cleaned = _clean_text(item)
        if cleaned:
            items.append(cleaned)
    return items


def _retry_after_seconds(exc: Exception) -> float | None:
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


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _is_retryable_output_exception(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError)


def _clip_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n...[truncated]"


def _explanation_schema(
    mode: ExplanationMode,
    evidence_source: EvidenceSource,
) -> str:
    missing = f"Not found in the available {_source_label(evidence_source)} source."
    if mode == ExplanationMode.METHOD:
        return (
            "{\n"
            '  "summary": "<=2 sentences>",\n'
            f'  "problem": "{missing}",\n'
            f'  "method_overview": "{missing}",\n'
            f'  "core_workflow": ["{missing}"],\n'
            f'  "inputs_outputs": ["{missing}"],\n'
            f'  "innovation": "{missing}"\n'
            "}"
        )
    if mode == ExplanationMode.EXPERIMENT:
        return (
            "{\n"
            '  "summary": "<=2 sentences>",\n'
            f'  "datasets": ["{missing}"],\n'
            f'  "baselines": ["{missing}"],\n'
            f'  "metrics": ["{missing}"],\n'
            f'  "experimental_setup": "{missing}",\n'
            f'  "conclusions": ["{missing}"]\n'
            "}"
        )
    return (
        "{\n"
        '  "summary": "<=2 sentences>",\n'
        f'  "stated_limitations": ["{missing}"],\n'
        f'  "assumptions": ["{missing}"],\n'
        f'  "missing_validation": ["{missing}"],\n'
        f'  "risks": ["{missing}"]\n'
        "}"
    )


def _required_text(
    value: Any,
    *,
    subject: str,
    evidence_source: EvidenceSource,
) -> str:
    cleaned = _clean_text(value)
    if cleaned:
        return cleaned
    return _missing_evidence(subject, evidence_source)


def _required_list(
    value: Any,
    *,
    subject: str,
    evidence_source: EvidenceSource,
) -> list[str]:
    normalized = _normalize_list(value)
    if normalized:
        return normalized
    return [_missing_evidence(subject, evidence_source)]


def _source_label(evidence_source: EvidenceSource) -> str:
    if evidence_source == EvidenceSource.FULL_TEXT:
        return "full-text"
    return evidence_source.value


def _missing_evidence(subject: str, evidence_source: EvidenceSource) -> str:
    return f"{subject.capitalize()} was not found in the available {_source_label(evidence_source)} source."
