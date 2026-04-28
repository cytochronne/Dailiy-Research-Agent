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
    PaperBriefingItem,
    PaperMetadata,
    Recommendation,
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
