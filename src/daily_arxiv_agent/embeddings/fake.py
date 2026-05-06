"""Deterministic fake embedding provider for tests and local demos."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

from daily_arxiv_agent.embeddings.base import (
    EmbeddingProviderError,
    normalize_embedding_text,
)


class FakeEmbeddingProvider:
    """Produce stable vectors without external credentials or network calls."""

    def __init__(
        self,
        *,
        dimensions: int | None = None,
        vector_map: Mapping[str, Sequence[float]] | None = None,
        synonym_groups: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.dimensions = max(dimensions or 8, 1)
        self.vector_map = {
            normalize_embedding_text(text): _normalize_vector(vector, self.dimensions)
            for text, vector in (vector_map or {}).items()
        }
        self.synonym_groups = {
            group: tuple(normalize_embedding_text(term) for term in terms if term.strip())
            for group, terms in (synonym_groups or {}).items()
        }
        self.calls: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = normalize_embedding_text(text)
            if not normalized:
                raise EmbeddingProviderError("embedding input text cannot be blank.")
            self.calls.append(normalized)
            mapped = self.vector_map.get(normalized)
            if mapped is not None:
                vectors.append(list(mapped))
                continue
            group = self._matching_synonym_group(normalized)
            vectors.append(self._hash_vector(group or normalized))
        return vectors

    def _matching_synonym_group(self, normalized_text: str) -> str | None:
        for group, terms in self.synonym_groups.items():
            if any(term and term in normalized_text for term in terms):
                return f"synonym:{group}"
        return None

    def _hash_vector(self, key: str) -> list[float]:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        values = [
            ((digest[index % len(digest)] / 255.0) * 2.0) - 1.0
            for index in range(self.dimensions)
        ]
        length = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / length for value in values]


def _normalize_vector(vector: Sequence[float], dimensions: int) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != dimensions:
        raise EmbeddingProviderError(
            "fake embedding vector dimensions must match configured dimensions."
        )
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingProviderError("fake embedding vector values must be finite.")
    return values
