import json
from datetime import date
from urllib import error

import pytest

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    PaperBriefingItem,
    PaperMetadata,
    Provenance,
    Recommendation,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
import daily_arxiv_agent.llm.openai_provider as openai_provider_module
from daily_arxiv_agent.llm.openai_provider import OpenAILLMProvider
from daily_arxiv_agent.llm.provider import create_llm_provider


def test_provider_factory_returns_fake_provider() -> None:
    provider = create_llm_provider(
        AppConfig(
            llm_provider="fake",
            llm_api_key=None,
        )
    )

    assert isinstance(provider, FakeLLMProvider)


def test_provider_factory_requires_api_key_for_openai() -> None:
    with pytest.raises(ValueError, match="LLM_API_KEY or OPENAI_API_KEY is required"):
        create_llm_provider(
            AppConfig(
                llm_provider="openai",
                llm_api_key=None,
            )
        )


def test_provider_factory_allows_local_custom_provider_without_api_key() -> None:
    provider = create_llm_provider(
        AppConfig(
            llm_provider="local-gateway",
            llm_api_key=None,
            llm_base_url="http://localhost:4000/v1",
        )
    )

    assert isinstance(provider, OpenAILLMProvider)


def test_provider_factory_builds_openai_provider_with_key() -> None:
    provider = create_llm_provider(
        AppConfig(
            llm_provider="openai",
            llm_api_key="sk-test",
            llm_model="gpt-5-mini",
            llm_base_url="https://api.openai.com/v1",
            llm_timeout_seconds=15.0,
            llm_max_retries=3,
            llm_retry_backoff_seconds=2.0,
            llm_output_retries=2,
            llm_briefing_max_retries=5,
            llm_briefing_retry_backoff_seconds=4.0,
            llm_briefing_output_retries=3,
        )
    )

    assert isinstance(provider, OpenAILLMProvider)
    assert provider.max_retries == 3
    assert provider.retry_backoff_seconds == 2.0
    assert provider.output_retries == 2
    assert provider.briefing_max_retries == 5
    assert provider.briefing_retry_backoff_seconds == 4.0
    assert provider.briefing_output_retries == 3


def test_provider_factory_accepts_custom_provider_name() -> None:
    provider = create_llm_provider(
        AppConfig(
            llm_provider="deepseek",
            llm_api_key="sk-test",
            llm_model="deepseek-chat",
            llm_base_url="https://api.deepseek.com/v1",
        )
    )

    assert isinstance(provider, OpenAILLMProvider)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ANN201
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _make_paper() -> PaperMetadata:
    return PaperMetadata(
        paper_id="2501.00001",
        title="A Test Paper",
        authors=["Ada Lovelace"],
        abstract="This paper studies a world action model for offline reinforcement learning.",
        categories=["cs.LG"],
        published_date=date(2025, 1, 1),
        updated_date=date(2025, 1, 2),
        arxiv_url="https://arxiv.org/abs/2501.00001",
        pdf_url="https://arxiv.org/pdf/2501.00001",
        provenance=Provenance(
            source="arxiv",
            source_url="https://arxiv.org/abs/2501.00001",
            query='all:"world-action-model"',
        ),
    )


def _make_recommendation() -> Recommendation:
    return Recommendation(
        paper=_make_paper(),
        rank=1,
        score=3.0,
        rationale="Matched explicit terms: world, action, model. Evidence: abstract.",
        evidence_source=EvidenceSource.ABSTRACT,
    )


def _make_briefing_item() -> PaperBriefingItem:
    paper = _make_paper()
    return PaperBriefingItem(
        paper_id=paper.paper_id,
        title=paper.title,
        rank=1,
        score=3.0,
        summary="A concise summary.",
        contributions=["One contribution."],
        methods=["One method."],
        relevance_rationale="Matched explicit terms.",
        evidence_source=EvidenceSource.ABSTRACT,
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )


def test_openai_provider_retries_transient_url_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        calls["count"] += 1
        if calls["count"] == 1:
            raise error.URLError("temporary failure")
        return _FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(openai_provider_module.request, "urlopen", fake_urlopen)

    provider = OpenAILLMProvider(
        api_key="sk-test",
        model="gpt-5-mini",
        max_retries=1,
        retry_backoff_seconds=0,
    )

    payload = provider._chat_completion(
        {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "Reply with ok"}],
        }
    )

    assert payload["choices"][0]["message"]["content"] == "ok"
    assert calls["count"] == 2


def test_extract_paper_retries_invalid_output_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeHTTPResponse({"choices": [{"message": {"content": "not json"}}]})
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "A valid summary.",
                                    "contributions": ["Contribution"],
                                    "methods": ["Method"],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(openai_provider_module.request, "urlopen", fake_urlopen)

    provider = OpenAILLMProvider(
        api_key="sk-test",
        model="gpt-5-mini",
        max_retries=0,
        output_retries=1,
        retry_backoff_seconds=0,
    )

    item = provider.extract_paper(
        _make_paper(),
        topic="world-action-model",
        recommendation=_make_recommendation(),
    )

    assert item.summary == "A valid summary."
    assert calls["count"] == 2


def test_briefing_uses_dedicated_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        calls["count"] += 1
        if calls["count"] == 1:
            raise error.URLError("temporary summary failure")
        return _FakeHTTPResponse({"choices": [{"message": {"content": "Recovered summary."}}]})

    monkeypatch.setattr(openai_provider_module.request, "urlopen", fake_urlopen)

    provider = OpenAILLMProvider(
        api_key="sk-test",
        model="gpt-5-mini",
        max_retries=0,
        briefing_max_retries=1,
        retry_backoff_seconds=0,
        briefing_retry_backoff_seconds=0,
    )

    summary = provider.summarize_briefing(
        topic="world-action-model",
        items=[_make_briefing_item()],
    )

    assert summary == "Recovered summary."
    assert calls["count"] == 2


def test_explain_paper_returns_mode_specific_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req, timeout):  # noqa: ANN001, ANN202
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "A method-focused explanation.",
                                    "problem": "The paper addresses explainable recommendation.",
                                    "method_overview": "It uses a staged agent workflow.",
                                    "core_workflow": [
                                        "Retrieve papers",
                                        "Rank them",
                                        "Explain one paper",
                                    ],
                                    "inputs_outputs": [
                                        "Input: topic query",
                                        "Output: paper explanation",
                                    ],
                                    "innovation": "It preserves evidence labels.",
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(openai_provider_module.request, "urlopen", fake_urlopen)

    provider = OpenAILLMProvider(
        api_key="sk-test",
        model="gpt-5-mini",
        max_retries=0,
        output_retries=0,
        retry_backoff_seconds=0,
    )

    explanation = provider.explain_paper(
        _make_paper(),
        mode=ExplanationMode.METHOD,
        content="Method overview: a staged workflow.",
        evidence_source=EvidenceSource.FULL_TEXT,
    )

    assert explanation.mode == ExplanationMode.METHOD
    assert explanation.method is not None
    assert explanation.method.core_workflow[0] == "Retrieve papers"
    assert explanation.evidence_source == EvidenceSource.FULL_TEXT
