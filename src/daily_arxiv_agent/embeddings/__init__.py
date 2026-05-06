"""Embedding provider adapters for semantic recommendation."""

from daily_arxiv_agent.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
    EmbeddingProviderError,
    SemanticReadiness,
)
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
from daily_arxiv_agent.embeddings.openai_provider import OpenAIEmbeddingProvider
from daily_arxiv_agent.embeddings.provider import (
    check_semantic_readiness,
    create_embedding_provider,
)

__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "FakeEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SemanticReadiness",
    "check_semantic_readiness",
    "create_embedding_provider",
]
