"""Embedding provider contracts and trace-safe status models."""

from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import BaseModel, Field


class EmbeddingConfigurationError(ValueError):
    """Raised when embedding provider configuration is invalid."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider call or response is invalid."""


class EmbeddingProvider(Protocol):
    """Minimal adapter boundary for embedding text inputs."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in input order."""


class SemanticReadiness(BaseModel):
    """Trace-safe semantic recommendation preflight state."""

    provider: str
    provider_mode: str
    provider_label: str
    credential_status: str
    model: str
    endpoint: str | None = None
    endpoint_safety: str
    cache_enabled: bool
    seed_quality: str
    can_run: bool
    error_code: str | None = None
    warnings: list[str] = Field(default_factory=list)


def normalize_embedding_text(text: str) -> str:
    """Normalize text for deterministic fake vectors and cache identities."""

    return " ".join(text.split()).lower()


def normalize_provider_input_text(text: str) -> str:
    """Normalize text sent to real embedding providers without changing case."""

    return " ".join(text.split())
