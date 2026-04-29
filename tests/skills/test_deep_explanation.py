from datetime import date
from pathlib import Path

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    PaperMetadata,
    Provenance,
    SkillStatus,
)
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.deep_explanation import PaperDeepExplanationSkill
from daily_arxiv_agent.storage import SQLitePaperStore


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_paper_text.txt"


def make_paper(
    abstract: str | None = (
        "We propose an agent workflow that ranks papers and explains recommendations."
    ),
    *,
    pdf_url: str | None = "https://arxiv.org/pdf/2604.00001",
) -> PaperMetadata:
    return PaperMetadata(
        paper_id="2604.00001",
        title="Explainable Agents for Daily Research Briefings",
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=["cs.LG"],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url="https://arxiv.org/abs/2604.00001",
        pdf_url=pdf_url,
        provenance=Provenance(
            source="arxiv",
            source_url="https://arxiv.org/abs/2604.00001",
            query="agent briefing",
        ),
    )


class FailingExplanationProvider(FakeLLMProvider):
    def explain_paper(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("provider unavailable")


def test_method_mode_uses_full_text_and_returns_structured_fields() -> None:
    text = FIXTURE.read_text()
    result = PaperDeepExplanationSkill(provider=FakeLLMProvider()).explain(
        make_paper(),
        mode=ExplanationMode.METHOD,
        full_text=text,
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.evidence_source == EvidenceSource.FULL_TEXT
    explanation = result.data
    assert explanation is not None
    assert explanation.method is not None
    assert "faster literature review" in explanation.method.problem.lower()
    assert explanation.method.core_workflow
    assert "transparent explanation loop" in explanation.method.innovation.lower()


def test_experiment_mode_extracts_datasets_baselines_metrics_and_conclusions() -> None:
    text = FIXTURE.read_text()
    result = PaperDeepExplanationSkill(provider=FakeLLMProvider()).explain(
        make_paper(),
        mode=ExplanationMode.EXPERIMENT,
        full_text=text,
    )

    assert result.status == SkillStatus.SUCCESS
    explanation = result.data
    assert explanation is not None
    assert explanation.experiment is not None
    assert explanation.experiment.datasets == ["ArxivDailyBench", "ResearchAgentEval."]
    assert explanation.experiment.baselines[0] == "BM25 ranking"
    assert explanation.experiment.metrics[0] == "Recall@5"
    assert "recent cs.LG papers" in explanation.experiment.experimental_setup
    assert explanation.experiment.conclusions


def test_limitations_mode_extracts_limitations_assumptions_validation_and_risks() -> None:
    text = FIXTURE.read_text()
    result = PaperDeepExplanationSkill(provider=FakeLLMProvider()).explain(
        make_paper(),
        mode=ExplanationMode.LIMITATIONS,
        full_text=text,
    )

    assert result.status == SkillStatus.SUCCESS
    explanation = result.data
    assert explanation is not None
    assert explanation.limitations is not None
    assert "small benchmark" in explanation.limitations.stated_limitations[0].lower()
    assert "abstracts and pdfs are aligned" in explanation.limitations.assumptions[0].lower()
    assert "latency" in explanation.limitations.missing_validation[0].lower()
    assert "generated explanations" in explanation.limitations.risks[0].lower()


def test_abstract_only_fallback_labels_evidence_and_avoids_unsupported_experiment_claims() -> None:
    result = PaperDeepExplanationSkill(provider=FakeLLMProvider()).explain(
        make_paper(pdf_url=None),
        mode=ExplanationMode.EXPERIMENT,
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.evidence_source == EvidenceSource.ABSTRACT
    explanation = result.data
    assert explanation is not None
    assert explanation.evidence_source == EvidenceSource.ABSTRACT
    assert explanation.experiment is not None
    assert "available abstract source" in explanation.experiment.datasets[0].lower()
    assert "available abstract source" in explanation.experiment.metrics[0].lower()


def test_pdf_parsing_failure_returns_abstract_only_fallback_when_abstract_exists() -> None:
    skill = PaperDeepExplanationSkill(
        provider=FakeLLMProvider(),
        pdf_text_loader=lambda paper: (_ for _ in ()).throw(RuntimeError("pdf unavailable")),
    )

    result = skill.explain(
        make_paper(),
        mode=ExplanationMode.METHOD,
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert "paper_pdf_parse_failed" in result.error.code
    assert result.evidence_source == EvidenceSource.ABSTRACT
    assert result.message == "Using abstract-only fallback explanation."


def test_cached_full_text_is_reused_before_pdf_loading(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    store.save_paper_full_text(
        "2604.00001",
        FIXTURE.read_text(),
        source_url="https://arxiv.org/pdf/2604.00001v1",
    )
    calls = {"count": 0}

    def fail_if_called(paper):  # noqa: ANN001, ANN202
        calls["count"] += 1
        raise AssertionError("cached full text should be used first")

    result = PaperDeepExplanationSkill(
        provider=FakeLLMProvider(),
        store=store,
        pdf_text_loader=fail_if_called,
    ).explain(
        make_paper(pdf_url="https://arxiv.org/pdf/2604.00001v1"),
        mode=ExplanationMode.METHOD,
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.metadata["content_origin"] == "cached_full_text"
    assert calls["count"] == 0


def test_cached_full_text_does_not_cross_pdf_versions(tmp_path) -> None:
    store = SQLitePaperStore(tmp_path / "papers.sqlite3")
    store.save_paper_full_text(
        "2604.00001",
        "Summary: Old version.\nProblem: Old problem.\nMethod overview: Old method.\n"
        "Core workflow: Old workflow.\nInputs and outputs: Old IO.\nInnovation: Old idea.",
        source_url="https://arxiv.org/pdf/2604.00001v1",
    )
    calls = {"count": 0}

    def versioned_loader(paper):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return FIXTURE.read_text()

    result = PaperDeepExplanationSkill(
        provider=FakeLLMProvider(),
        store=store,
        pdf_text_loader=versioned_loader,
    ).explain(
        make_paper(pdf_url="https://arxiv.org/pdf/2604.00001v2"),
        mode=ExplanationMode.METHOD,
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.metadata["content_origin"] == "pdf_full_text"
    assert calls["count"] == 1


def test_full_text_explanation_uses_pdf_source_in_provenance() -> None:
    text = FIXTURE.read_text()
    result = PaperDeepExplanationSkill(provider=FakeLLMProvider()).explain(
        make_paper(pdf_url="https://arxiv.org/pdf/2604.00001v3"),
        mode=ExplanationMode.METHOD,
        full_text=text,
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data is not None
    assert str(result.data.provenance.source_url) == "https://arxiv.org/pdf/2604.00001v3"
    assert str(result.provenance[0].source_url) == "https://arxiv.org/pdf/2604.00001v3"


def test_provider_failure_returns_deterministic_fallback_explanation() -> None:
    result = PaperDeepExplanationSkill(provider=FailingExplanationProvider()).explain(
        make_paper(pdf_url=None),
        mode=ExplanationMode.METHOD,
    )

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert "llm_explanation_failed" in result.error.code
    assert result.data is not None
    assert result.data.method is not None
