"""Factory and readiness helpers for embedding providers."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
    SemanticReadiness,
    normalize_embedding_text,
)
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
from daily_arxiv_agent.embeddings.openai_provider import (
    OpenAIEmbeddingProvider,
    _normalize_embeddings_path,
)

LIVE_API_PROVIDER_NAMES = frozenset({"live", "openai"})


def create_embedding_provider(config: AppConfig | None = None) -> EmbeddingProvider:
    """Create the configured embedding provider without fallback substitution."""

    resolved = config or AppConfig.from_env()
    provider = _provider_name(resolved)
    if provider == "fake":
        return FakeEmbeddingProvider(dimensions=resolved.embedding_dimensions)

    endpoint = _validate_endpoint(
        base_url=resolved.embedding_base_url,
        path=resolved.embedding_path,
    )
    api_key = (resolved.embedding_api_key or "").strip() or None
    if provider in LIVE_API_PROVIDER_NAMES and api_key is None:
        raise EmbeddingConfigurationError(
            "EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER is "
            f"{provider!r}. Reusing OPENAI_API_KEY requires "
            "EMBEDDING_REUSE_OPENAI_API_KEY=true."
        )

    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=resolved.embedding_model,
        base_url=endpoint.base_url,
        embeddings_path=endpoint.path,
        timeout_seconds=resolved.embedding_timeout_seconds,
        max_retries=resolved.embedding_max_retries,
        retry_backoff_seconds=resolved.embedding_retry_backoff_seconds,
        dimensions=resolved.embedding_dimensions,
    )


def check_semantic_readiness(
    config: AppConfig | None = None,
    *,
    seed_texts: list[str] | tuple[str, ...] = (),
    cache_enabled: bool | None = None,
) -> SemanticReadiness:
    """Return a trace-safe preflight status for semantic recommendation."""

    resolved = config or AppConfig.from_env()
    provider = _provider_name(resolved)
    provider_mode = "fake" if provider == "fake" else "live"
    endpoint = None
    endpoint_safety = "not_applicable"
    warnings: list[str] = []

    if provider == "fake":
        credential_status = "not_required"
    else:
        endpoint_status = _inspect_endpoint(
            base_url=resolved.embedding_base_url,
            path=resolved.embedding_path,
        )
        endpoint = endpoint_status.endpoint
        endpoint_safety = endpoint_status.safety
        credential_status = _credential_status(resolved, provider, endpoint_status)
        if credential_status == "reused_openai_api_key":
            warnings.append("embedding_api_key_reused_from_openai_api_key")

    seed_quality = _seed_quality(seed_texts)
    cache_flag = resolved.embedding_cache_enabled if cache_enabled is None else cache_enabled
    error_code = _readiness_error_code(
        provider=provider,
        credential_status=credential_status,
        endpoint_safety=endpoint_safety,
        seed_quality=seed_quality,
    )

    return SemanticReadiness(
        provider=provider,
        provider_mode=provider_mode,
        provider_label=f"{provider}:{resolved.embedding_model}",
        credential_status=credential_status,
        model=resolved.embedding_model,
        endpoint=endpoint,
        endpoint_safety=endpoint_safety,
        cache_enabled=cache_flag,
        seed_quality=seed_quality,
        can_run=error_code is None,
        error_code=error_code,
        warnings=warnings,
    )


class _EndpointStatus:
    def __init__(self, *, base_url: str, path: str, safety: str) -> None:
        self.base_url = base_url
        self.path = path
        self.safety = safety

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}{self.path}"


def _provider_name(config: AppConfig) -> str:
    return (config.embedding_provider or "openai").strip().lower() or "openai"


def _validate_endpoint(*, base_url: str, path: str) -> _EndpointStatus:
    status = _inspect_endpoint(base_url=base_url, path=path)
    if status.safety == "invalid_url":
        raise EmbeddingConfigurationError("EMBEDDING_BASE_URL must be an absolute URL.")
    if status.safety == "unsupported_scheme":
        raise EmbeddingConfigurationError("EMBEDDING_BASE_URL must use HTTPS or HTTP.")
    if status.safety == "unsafe_remote_http":
        raise EmbeddingConfigurationError(
            "HTTPS is required for remote embedding provider URLs; "
            "plain HTTP is allowed only for localhost or loopback gateways."
        )
    return status


def _inspect_endpoint(*, base_url: str, path: str) -> _EndpointStatus:
    normalized_base_url = (base_url or "").strip().rstrip("/")
    try:
        normalized_path = _normalize_embeddings_path(path)
    except EmbeddingConfigurationError:
        normalized_path = "/embeddings"
        return _EndpointStatus(
            base_url=normalized_base_url,
            path=normalized_path,
            safety="invalid_url",
        )

    parsed = urlsplit(normalized_base_url)
    if not parsed.scheme or not parsed.netloc:
        return _EndpointStatus(
            base_url=normalized_base_url,
            path=normalized_path,
            safety="invalid_url",
        )
    if parsed.scheme == "https":
        return _EndpointStatus(
            base_url=normalized_base_url,
            path=normalized_path,
            safety="safe",
        )
    if parsed.scheme == "http":
        return _EndpointStatus(
            base_url=normalized_base_url,
            path=normalized_path,
            safety="safe" if _is_loopback_host(parsed.hostname) else "unsafe_remote_http",
        )
    return _EndpointStatus(
        base_url=normalized_base_url,
        path=normalized_path,
        safety="unsupported_scheme",
    )


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


def _credential_status(
    config: AppConfig,
    provider: str,
    endpoint_status: _EndpointStatus,
) -> str:
    if (config.embedding_api_key or "").strip():
        if config.embedding_reuse_openai_api_key:
            return "reused_openai_api_key"
        return "available"
    if provider in LIVE_API_PROVIDER_NAMES:
        return "missing"
    if endpoint_status.safety == "safe" and endpoint_status.base_url.startswith("http://"):
        return "not_required_local_gateway"
    return "missing"


def _seed_quality(seed_texts: list[str] | tuple[str, ...]) -> str:
    if any(normalize_embedding_text(text) for text in seed_texts):
        return "usable"
    return "missing"


def _readiness_error_code(
    *,
    provider: str,
    credential_status: str,
    endpoint_safety: str,
    seed_quality: str,
) -> str | None:
    if endpoint_safety in {"invalid_url", "unsupported_scheme", "unsafe_remote_http"}:
        return "semantic_embedding_endpoint_unsafe"
    if credential_status == "missing" and provider != "fake":
        return "semantic_embedding_credentials_missing"
    if seed_quality != "usable":
        return "semantic_seed_quality_error"
    return None
