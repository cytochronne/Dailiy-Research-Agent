"""OpenAI-compatible embedding provider."""

from __future__ import annotations

import json
import math
import re
import time
from ipaddress import ip_address
from typing import Any, Sequence
from urllib import error, request
from urllib.parse import urlsplit

from daily_arxiv_agent.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    normalize_provider_input_text,
)

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class OpenAIEmbeddingProvider:
    """Call OpenAI-compatible Embeddings APIs and validate vector output."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        embeddings_path: str = "/embeddings",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        dimensions: int | None = None,
    ) -> None:
        if api_key is not None and not api_key.strip():
            raise EmbeddingConfigurationError(
                "EMBEDDING_API_KEY cannot be blank when provided."
            )
        if not model.strip():
            raise EmbeddingConfigurationError("EMBEDDING_MODEL cannot be blank.")
        if dimensions is not None and dimensions < 1:
            raise EmbeddingConfigurationError("EMBEDDING_DIMENSIONS must be positive.")

        self.api_key = api_key.strip() if api_key else None
        self.model = model.strip()
        self.base_url = _validate_base_url(base_url)
        self.embeddings_path = _normalize_embeddings_path(embeddings_path)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(max_retries, 0)
        self.retry_backoff_seconds = max(retry_backoff_seconds, 0.0)
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = [normalize_provider_input_text(text) for text in texts]
        if not normalized_texts:
            return []
        if any(not text for text in normalized_texts):
            raise EmbeddingProviderError("embedding input text cannot be blank.")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": normalized_texts,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        raw = self._post_embedding_request(payload)
        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(
                "Embedding API returned non-JSON response."
            ) from exc
        return self._parse_embedding_response(
            response_payload,
            expected_count=len(normalized_texts),
        )

    def _post_embedding_request(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = f"{self.base_url}{self.embeddings_path}"
        return self._post_with_retries(
            endpoint=endpoint,
            body=body,
            headers=headers,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )

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
                    self._sleep_before_retry(attempt + 1, retry_backoff_seconds)
                    continue
                raise EmbeddingProviderError(
                    "Embedding API HTTP "
                    f"{exc.code}: {self._redact_error_detail(str(exc.reason))}"
                ) from exc
            except error.URLError as exc:
                if attempt < max_retries:
                    self._sleep_before_retry(attempt + 1, retry_backoff_seconds)
                    continue
                raise EmbeddingProviderError(
                    "Embedding API request failed: "
                    f"{self._redact_error_detail(str(exc.reason))}"
                ) from exc

        raise EmbeddingProviderError("Embedding API request failed after retries.")

    def _parse_embedding_response(
        self,
        payload: dict[str, Any],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected_count:
            raise EmbeddingProviderError("invalid embedding response: vector count mismatch.")

        ordered_items = _items_in_input_order(data, expected_count)
        vectors: list[list[float]] = []
        expected_dimensions = self.dimensions
        for item in ordered_items:
            if not isinstance(item, dict):
                raise EmbeddingProviderError("invalid embedding response: item shape.")
            vector = _coerce_vector(item.get("embedding"))
            if expected_dimensions is None:
                expected_dimensions = len(vector)
            if len(vector) != expected_dimensions:
                raise EmbeddingProviderError(
                    "invalid embedding response: vector dimensions mismatch."
                )
            vectors.append(vector)
        return vectors

    def _sleep_before_retry(self, attempt: int, retry_backoff_seconds: float) -> None:
        delay_seconds = max(retry_backoff_seconds * attempt, 0.0)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    def _redact_error_detail(self, text: str) -> str:
        return _redact_sensitive_text(text, secrets=(self.api_key,))


def _validate_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    parsed = urlsplit(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise EmbeddingConfigurationError("EMBEDDING_BASE_URL must be an absolute URL.")
    if parsed.scheme == "https":
        return normalized
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return normalized
    if parsed.scheme == "http":
        raise EmbeddingConfigurationError(
            "HTTPS is required for remote embedding provider URLs; "
            "plain HTTP is allowed only for localhost or loopback gateways."
        )
    raise EmbeddingConfigurationError("EMBEDDING_BASE_URL must use HTTPS or HTTP.")


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _normalize_embeddings_path(path: str) -> str:
    normalized = (path or "/embeddings").strip() or "/embeddings"
    if "?" in normalized or "#" in normalized:
        raise EmbeddingConfigurationError(
            "EMBEDDING_PATH must be a path without query or fragment."
        )
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized.rstrip("/") or "/embeddings"


def _items_in_input_order(data: list[Any], expected_count: int) -> list[Any]:
    indexed: list[tuple[int, Any]] = []
    for fallback_index, item in enumerate(data):
        if not isinstance(item, dict):
            return data
        index = item.get("index", fallback_index)
        if not isinstance(index, int) or index < 0 or index >= expected_count:
            raise EmbeddingProviderError("invalid embedding response: item index.")
        indexed.append((index, item))
    if len({index for index, _item in indexed}) != expected_count:
        raise EmbeddingProviderError("invalid embedding response: duplicate item index.")
    return [item for _index, item in sorted(indexed, key=lambda pair: pair[0])]


def _coerce_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise EmbeddingProviderError("invalid embedding response: vector shape.")
    vector: list[float] = []
    for item in value:
        if type(item) not in (float, int):
            raise EmbeddingProviderError("invalid embedding response: vector value.")
        number = float(item)
        if not math.isfinite(number):
            raise EmbeddingProviderError("invalid embedding response: vector value.")
        vector.append(number)
    return vector


_AUTHORIZATION_RE = re.compile(
    r"authorization\s*:\s*bearer\s+[^\s,;]+",
    flags=re.IGNORECASE,
)
_BEARER_RE = re.compile(r"bearer\s+[^\s,;]+", flags=re.IGNORECASE)
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token|secret|key)\s*=\s*[^\s,;]+"
)


def _redact_sensitive_text(
    text: str,
    *,
    secrets: Sequence[str | None] = (),
) -> str:
    redacted = _AUTHORIZATION_RE.sub("Authorization: [REDACTED]", text)
    redacted = _BEARER_RE.sub("[REDACTED]", redacted)
    redacted = _KEY_VALUE_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted
