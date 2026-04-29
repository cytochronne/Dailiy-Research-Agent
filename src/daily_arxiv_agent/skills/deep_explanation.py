"""Paper-level deep explanation Skill with full-text and abstract fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib import request

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExplanationMode,
    PaperDeepExplanation,
    PaperMetadata,
    Provenance,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.llm.provider import create_llm_provider
from daily_arxiv_agent.storage import SQLitePaperStore


PdfTextLoader = Callable[[PaperMetadata], str]


@dataclass
class PreparedPaperContent:
    """Resolved source content for deep explanation generation."""

    text: str
    evidence_source: EvidenceSource
    origin: str
    provenance: Provenance
    error: SkillError | None = None


class PaperDeepExplanationSkill:
    """Generate mode-specific paper explanations from cached or prepared content."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        store: SQLitePaperStore | None = None,
        pdf_text_loader: PdfTextLoader | None = None,
    ) -> None:
        self.provider = provider or create_llm_provider()
        self.store = store
        self.pdf_text_loader = pdf_text_loader or _default_pdf_text_loader
        self.fallback_provider = FakeLLMProvider()

    def explain(
        self,
        paper: PaperMetadata,
        *,
        mode: ExplanationMode,
        full_text: str | None = None,
    ) -> SkillResult[PaperDeepExplanation]:
        prepared = self._prepare_content(paper, full_text=full_text)
        errors: list[SkillError] = []
        if prepared.error is not None:
            errors.append(prepared.error)
        explanation_paper = paper.model_copy(update={"provenance": prepared.provenance})

        try:
            explanation = self.provider.explain_paper(
                explanation_paper,
                mode=mode,
                content=prepared.text,
                evidence_source=prepared.evidence_source,
            )
        except Exception as exc:
            explanation = self.fallback_provider.explain_paper(
                explanation_paper,
                mode=mode,
                content=prepared.text,
                evidence_source=prepared.evidence_source,
            )
            errors.append(
                SkillError(
                    code="llm_explanation_failed",
                    message=f"LLM explanation failed: {exc}",
                    retryable=True,
                )
            )

        status = SkillStatus.SUCCESS if not errors else SkillStatus.FALLBACK
        return SkillResult[PaperDeepExplanation](
            status=status,
            data=explanation,
            evidence_source=prepared.evidence_source,
            provenance=[prepared.provenance],
            error=_merge_errors(errors) if errors else None,
            message=_result_message(status, prepared),
            metadata={
                "paper_id": paper.paper_id,
                "mode": mode.value,
                "content_origin": prepared.origin,
                "content_source_url": (
                    str(prepared.provenance.source_url)
                    if prepared.provenance.source_url is not None
                    else None
                ),
            },
        )

    def _prepare_content(
        self,
        paper: PaperMetadata,
        *,
        full_text: str | None,
    ) -> PreparedPaperContent:
        full_text_source_url = _full_text_source_url(paper)
        provided = _normalize_text(full_text)
        if provided:
            return PreparedPaperContent(
                text=provided,
                evidence_source=EvidenceSource.FULL_TEXT,
                origin="provided_full_text",
                provenance=_provenance_with_source_url(paper, full_text_source_url),
            )

        if self.store is not None:
            cached = _normalize_text(
                self.store.load_paper_full_text(
                    paper.paper_id,
                    source_url=str(full_text_source_url) if full_text_source_url else None,
                )
            )
            if cached:
                return PreparedPaperContent(
                    text=cached,
                    evidence_source=EvidenceSource.FULL_TEXT,
                    origin="cached_full_text",
                    provenance=_provenance_with_source_url(paper, full_text_source_url),
                )

        pdf_error: SkillError | None = None
        if paper.pdf_url is not None:
            try:
                extracted = _normalize_text(self.pdf_text_loader(paper))
                if not extracted:
                    raise RuntimeError("PDF parsing returned no text.")
                if self.store is not None:
                    self.store.save_paper_full_text(
                        paper.paper_id,
                        extracted,
                        source_url=str(full_text_source_url) if full_text_source_url else None,
                    )
                return PreparedPaperContent(
                    text=extracted,
                    evidence_source=EvidenceSource.FULL_TEXT,
                    origin="pdf_full_text",
                    provenance=_provenance_with_source_url(paper, full_text_source_url),
                )
            except Exception as exc:
                pdf_error = SkillError(
                    code="paper_pdf_parse_failed",
                    message=f"PDF parsing failed: {exc}",
                    retryable=True,
                )

        abstract = _normalize_text(paper.abstract)
        if abstract:
            if pdf_error is not None:
                return PreparedPaperContent(
                    text=abstract,
                    evidence_source=EvidenceSource.ABSTRACT,
                    origin="abstract_after_pdf_failure",
                    provenance=paper.provenance,
                    error=SkillError(
                        code=pdf_error.code,
                        message=f"{pdf_error.message} Using abstract-only fallback.",
                        retryable=pdf_error.retryable,
                    ),
                )
            return PreparedPaperContent(
                text=abstract,
                evidence_source=EvidenceSource.ABSTRACT,
                origin="abstract_fallback",
                provenance=paper.provenance,
                error=SkillError(
                    code="paper_full_text_unavailable",
                    message="Full-text content was unavailable; using abstract-only fallback.",
                    retryable=False,
                ),
            )

        return PreparedPaperContent(
            text=_metadata_fallback_text(paper),
            evidence_source=EvidenceSource.METADATA,
            origin="metadata_fallback",
            provenance=paper.provenance,
            error=SkillError(
                code="paper_text_unavailable",
                message="Full-text and abstract content were unavailable; using metadata-only fallback.",
                retryable=False,
            ),
        )


def _merge_errors(errors: list[SkillError]) -> SkillError:
    unique_codes: list[str] = []
    messages: list[str] = []
    retryable = False
    for error in errors:
        if error.code not in unique_codes:
            unique_codes.append(error.code)
        if error.message not in messages:
            messages.append(error.message)
        retryable = retryable or error.retryable
    return SkillError(
        code=",".join(unique_codes),
        message="; ".join(messages),
        retryable=retryable,
    )


def _result_message(status: SkillStatus, prepared: PreparedPaperContent) -> str:
    if status == SkillStatus.SUCCESS:
        return "Generated paper explanation from full text."
    if prepared.evidence_source == EvidenceSource.FULL_TEXT:
        return "Using deterministic fallback explanation after provider failure."
    if prepared.evidence_source == EvidenceSource.ABSTRACT:
        return "Using abstract-only fallback explanation."
    return "Using metadata-only fallback explanation."


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    lines = [" ".join(line.split()) for line in value.splitlines()]
    cleaned_lines = [line for line in lines if line]
    if cleaned_lines:
        return "\n".join(cleaned_lines)
    return " ".join(value.split())


def _full_text_source_url(paper: PaperMetadata):
    return paper.pdf_url or paper.provenance.source_url


def _provenance_with_source_url(
    paper: PaperMetadata,
    source_url,
) -> Provenance:
    if source_url is None:
        return paper.provenance
    if paper.provenance.source_url == source_url:
        return paper.provenance
    return paper.provenance.model_copy(update={"source_url": source_url})


def _metadata_fallback_text(paper: PaperMetadata) -> str:
    categories = ", ".join(paper.categories) if paper.categories else "uncategorized"
    return (
        f"Title: {paper.title}. "
        f"Authors: {', '.join(paper.authors) if paper.authors else 'unknown'}. "
        f"Categories: {categories}. "
        f"Provenance source: {paper.provenance.source}."
    )


def _default_pdf_text_loader(paper: PaperMetadata) -> str:
    if paper.pdf_url is None:
        raise RuntimeError("Selected paper does not have a PDF URL.")
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("pymupdf is not installed.") from exc

    with request.urlopen(str(paper.pdf_url), timeout=30.0) as response:  # pragma: no cover - network path
        pdf_bytes = response.read()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:  # pragma: no cover - optional dep
        chunks = [page.get_text("text") for page in document]
    text = "\n".join(chunks)
    normalized = _normalize_text(text)
    if not normalized:
        raise RuntimeError("PDF text extraction returned empty content.")
    return normalized
