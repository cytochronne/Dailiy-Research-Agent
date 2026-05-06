import json
from urllib import error

import pytest

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
)
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
import daily_arxiv_agent.embeddings.openai_provider as openai_embedding_module
from daily_arxiv_agent.embeddings.openai_provider import OpenAIEmbeddingProvider
from daily_arxiv_agent.embeddings.provider import (
    check_semantic_readiness,
    create_embedding_provider,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN201
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_provider_factory_returns_fake_provider_with_stable_vectors() -> None:
    provider = create_embedding_provider(
        AppConfig(
            embedding_provider="fake",
            embedding_api_key=None,
            embedding_dimensions=4,
        )
    )

    assert isinstance(provider, FakeEmbeddingProvider)
    assert provider.embed_texts(["Graph neural retrieval"]) == provider.embed_texts(
        ["  graph   neural retrieval  "]
    )


def test_fake_provider_uses_normalized_vector_map_and_synonym_groups() -> None:
    provider = FakeEmbeddingProvider(
        dimensions=3,
        vector_map={"world model planning": [1.0, 0.0, 0.0]},
        synonym_groups={"planning": ["tree search", "lookahead"]},
    )

    assert provider.embed_texts(["  World   Model Planning "]) == [[1.0, 0.0, 0.0]]
    assert provider.embed_texts(["tree search controller"]) == provider.embed_texts(
        ["lookahead controller"]
    )


def test_provider_factory_builds_openai_provider_with_embedding_config() -> None:
    provider = create_embedding_provider(
        AppConfig(
            embedding_provider="openai",
            embedding_api_key="embed-secret",
            embedding_model="text-embedding-3-large",
            embedding_base_url="https://api.openai.com/v1/",
            embedding_path="embeddings",
            embedding_timeout_seconds=12.5,
            embedding_max_retries=3,
            embedding_retry_backoff_seconds=0.25,
            embedding_dimensions=3072,
        )
    )

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model == "text-embedding-3-large"
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider.embeddings_path == "/embeddings"
    assert provider.timeout_seconds == 12.5
    assert provider.max_retries == 3
    assert provider.retry_backoff_seconds == 0.25
    assert provider.dimensions == 3072


def test_custom_local_provider_can_be_constructed_without_api_key() -> None:
    provider = create_embedding_provider(
        AppConfig(
            embedding_provider="local-gateway",
            embedding_api_key=None,
            embedding_base_url="http://localhost:11434/v1",
            embedding_model="local-embedding-model",
        )
    )

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.api_key is None


def test_openai_provider_requires_embedding_api_key() -> None:
    with pytest.raises(
        EmbeddingConfigurationError,
        match="EMBEDDING_API_KEY is required",
    ):
        create_embedding_provider(
            AppConfig(
                embedding_provider="openai",
                embedding_api_key=None,
            )
        )


def test_embedding_config_does_not_reuse_openai_key_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_REUSE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    config = AppConfig.from_env()

    assert config.embedding_api_key is None
    assert config.embedding_reuse_openai_api_key is False


def test_embedding_config_reuses_openai_key_only_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_REUSE_OPENAI_API_KEY", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    config = AppConfig.from_env()

    assert config.embedding_api_key == "openai-secret"
    assert config.embedding_reuse_openai_api_key is True


def test_remote_http_endpoint_is_rejected_but_localhost_http_is_allowed() -> None:
    with pytest.raises(
        EmbeddingConfigurationError,
        match="HTTPS is required",
    ):
        create_embedding_provider(
            AppConfig(
                embedding_provider="custom",
                embedding_api_key="embed-secret",
                embedding_base_url="http://api.example.com/v1",
            )
        )

    provider = create_embedding_provider(
        AppConfig(
            embedding_provider="custom",
            embedding_api_key=None,
            embedding_base_url="http://127.0.0.1:8080/v1",
        )
    )

    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_openai_provider_constructor_rejects_unsafe_remote_http() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="HTTPS is required"):
        OpenAIEmbeddingProvider(
            api_key="embed-secret",
            model="text-embedding-3-small",
            base_url="http://api.example.com/v1",
        )


def test_openai_provider_posts_embeddings_payload_and_validates_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, object]] = []
    captured_headers: list[dict[str, str]] = []

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        captured_headers.append(dict(req.header_items()))
        return _FakeHTTPResponse(
            {
                "data": [
                    {"index": 0, "embedding": [0.25, 0.75]},
                    {"index": 1, "embedding": [0.5, 0.5]},
                ]
            }
        )

    monkeypatch.setattr(openai_embedding_module.request, "urlopen", fake_urlopen)

    provider = OpenAIEmbeddingProvider(
        api_key="embed-secret",
        model="text-embedding-3-small",
        dimensions=2,
        max_retries=0,
    )

    vectors = provider.embed_texts([" first\ntext ", "second text"])

    assert vectors == [[0.25, 0.75], [0.5, 0.5]]
    assert captured_payloads == [
        {
            "model": "text-embedding-3-small",
            "input": ["first text", "second text"],
            "dimensions": 2,
        }
    ]
    assert captured_headers[0]["Authorization"] == "Bearer embed-secret"


def test_malformed_embedding_response_raises_provider_error_without_payload() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key="embed-secret",
        model="text-embedding-3-small",
        dimensions=2,
    )

    with pytest.raises(EmbeddingProviderError, match="invalid embedding response") as excinfo:
        provider._parse_embedding_response(  # noqa: SLF001
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 2.0, 3.0]},
                    {"index": 1, "embedding": "not-a-vector"},
                ]
            },
            expected_count=2,
        )

    message = str(excinfo.value)
    assert "not-a-vector" not in message
    assert "embed-secret" not in message
    assert "data" not in message


def test_openai_provider_retries_transient_request_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        calls["count"] += 1
        if calls["count"] == 1:
            raise error.URLError("temporary failure using embed-secret")
        return _FakeHTTPResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(openai_embedding_module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_embedding_module.time, "sleep", lambda seconds: None)

    provider = OpenAIEmbeddingProvider(
        api_key="embed-secret",
        model="text-embedding-3-small",
        dimensions=2,
        max_retries=1,
        retry_backoff_seconds=0,
    )

    vectors = provider.embed_texts(["retry me"])

    assert vectors == [[1.0, 0.0]]
    assert calls["count"] == 2


def test_retry_exhaustion_redacts_key_like_and_exact_secret_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        raise error.URLError(
            "Authorization: Bearer embed-secret api_key=abc123 failed for embed-secret"
        )

    monkeypatch.setattr(openai_embedding_module.request, "urlopen", fake_urlopen)

    provider = OpenAIEmbeddingProvider(
        api_key="embed-secret",
        model="text-embedding-3-small",
        max_retries=0,
    )

    with pytest.raises(EmbeddingProviderError) as excinfo:
        provider.embed_texts(["fails"])

    message = str(excinfo.value)
    assert "embed-secret" not in message
    assert "abc123" not in message
    assert "Bearer" not in message


def test_semantic_readiness_reports_missing_credentials_before_provider_creation() -> None:
    readiness = check_semantic_readiness(
        AppConfig(
            embedding_provider="openai",
            embedding_api_key=None,
            embedding_cache_enabled=False,
        ),
        seed_texts=["A usable seed abstract."],
    )

    assert readiness.provider_mode == "live"
    assert readiness.credential_status == "missing"
    assert readiness.cache_enabled is False
    assert readiness.seed_quality == "usable"
    assert readiness.can_run is False
    assert readiness.error_code == "semantic_embedding_credentials_missing"


def test_semantic_readiness_reports_unsafe_endpoint_and_weak_seed_text() -> None:
    readiness = check_semantic_readiness(
        AppConfig(
            embedding_provider="custom",
            embedding_api_key="embed-secret",
            embedding_base_url="http://api.example.com/v1",
        ),
        seed_texts=["  "],
    )

    assert readiness.endpoint_safety == "unsafe_remote_http"
    assert readiness.seed_quality == "missing"
    assert readiness.can_run is False
    assert readiness.error_code == "semantic_embedding_endpoint_unsafe"
    assert readiness.endpoint == "http://api.example.com/v1/embeddings"
    assert "embed-secret" not in readiness.model_dump_json()
