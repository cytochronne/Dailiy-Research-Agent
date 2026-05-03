"""Structured extraction Skill for ranked papers."""

from __future__ import annotations

import re

from daily_arxiv_agent.contracts import (
    EvidenceBoundClaim,
    EvidenceSource,
    EvidenceSupportStatus,
    FieldEvidenceStatus,
    PaperBriefingItem,
    Recommendation,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.llm.provider import create_llm_provider


class PaperExtractionSkill:
    """Extract briefing fields for one recommendation through an LLM adapter."""

    def __init__(self, *, provider: LLMProvider | None = None) -> None:
        self.provider = provider or create_llm_provider()

    def extract(
        self,
        recommendation: Recommendation,
        *,
        topic: str,
    ) -> SkillResult[PaperBriefingItem]:
        try:
            item = self.provider.extract_paper(
                recommendation.paper,
                topic=topic,
                recommendation=recommendation,
            )
            item = enrich_briefing_item(item, recommendation, topic=topic)
        except Exception as exc:
            item = _fallback_item(recommendation, topic=topic)
            return SkillResult[PaperBriefingItem](
                status=SkillStatus.FALLBACK,
                data=item,
                evidence_source=item.evidence_source,
                provenance=[recommendation.paper.provenance],
                error=SkillError(
                    code="llm_extraction_failed",
                    message=f"LLM extraction failed: {exc}",
                    retryable=True,
                ),
                message="Using metadata-only fallback extraction.",
                metadata={"topic": topic, "paper_id": recommendation.paper.paper_id},
            )

        return SkillResult[PaperBriefingItem](
            status=SkillStatus.SUCCESS,
            data=item,
            evidence_source=item.evidence_source,
            provenance=[recommendation.paper.provenance],
            metadata={"topic": topic, "paper_id": recommendation.paper.paper_id},
        )


def _fallback_item(recommendation: Recommendation, *, topic: str) -> PaperBriefingItem:
    paper = recommendation.paper
    item = PaperBriefingItem(
        paper_id=paper.paper_id,
        title=paper.title,
        rank=recommendation.rank,
        score=recommendation.score,
        summary=(
            f"Metadata-only fallback for '{paper.title}' while extracting topic "
            f"'{topic}'."
        ),
        contributions=[
            "Structured extraction was unavailable, so no abstract-level claims are made."
        ],
        methods=[],
        relevance_rationale=recommendation.rationale,
        evidence_source=EvidenceSource.METADATA,
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
    )
    return enrich_briefing_item(item, recommendation, topic=topic)


def enrich_briefing_item(
    item: PaperBriefingItem,
    recommendation: Recommendation,
    *,
    topic: str,
) -> PaperBriefingItem:
    """Attach conservative evidence-bound claims to provider extraction output."""

    paper = recommendation.paper
    abstract = paper.abstract or ""
    has_abstract = item.evidence_source == EvidenceSource.ABSTRACT and bool(abstract)
    updates: dict[str, object] = {}

    if item.problem is None:
        updates["problem"] = (
            _abstract_claim(
                _sentence_with_keywords(
                    abstract,
                    "problem",
                    "challenge",
                    "need",
                    "requires",
                    "study",
                    "studies",
                    "addresses",
                    "address",
                ),
                unavailable_reason=(
                    "The abstract does not state a problem framing."
                    if has_abstract
                    else "No abstract is available to support a problem claim."
                ),
            )
            if has_abstract
            else _unavailable_claim(
                "No abstract is available to support a problem claim."
            )
        )

    if item.approach is None:
        updates["approach"] = (
            _abstract_claim(
                _sentence_with_keywords(
                    abstract,
                    "approach",
                    "method",
                    "framework",
                    "workflow",
                    "model",
                    "propose",
                    "proposes",
                    "present",
                    "presents",
                    "introduce",
                    "introduces",
                    "develop",
                    "develops",
                    "use",
                    "uses",
                    "using",
                    "via",
                ),
                unavailable_reason=(
                    "The abstract does not expose an approach or method claim."
                    if has_abstract
                    else "No abstract is available to support an approach claim."
                ),
            )
            if has_abstract
            else _unavailable_claim(
                "No abstract is available to support an approach claim."
            )
        )

    if item.reading_guide is None:
        updates["reading_guide"] = _reading_guide_claim(
            item,
            recommendation,
            topic=topic,
            has_abstract=has_abstract,
        )

    if not item.contribution_claims:
        updates["contribution_claims"] = _claims_from_items(
            item.contributions,
            sources=[EvidenceSource.ABSTRACT] if has_abstract else [],
            unavailable_reason=(
                "The abstract does not provide explicit contribution evidence."
                if has_abstract
                else "No abstract is available to support contribution claims."
            ),
        )

    if not item.method_claims:
        updates["method_claims"] = _claims_from_items(
            item.methods,
            sources=[EvidenceSource.ABSTRACT] if has_abstract else [],
            unavailable_reason=(
                "The abstract does not provide explicit method evidence."
                if has_abstract
                else "No abstract is available to support method claims."
            ),
        )

    if item.relevance_evidence is None:
        updates["relevance_evidence"] = FieldEvidenceStatus(
            status=(
                EvidenceSupportStatus.SUPPORTED
                if has_abstract
                else EvidenceSupportStatus.PARTIAL
            ),
            sources=_ordered_sources([EvidenceSource.RANKING, item.evidence_source]),
            note=(
                "Relevance is supported by ranking rationale and abstract evidence."
                if has_abstract
                else "Relevance is evidence-limited because only metadata and ranking "
                "rationale are available."
            ),
        )

    return item.model_copy(update=updates) if updates else item


def _abstract_claim(
    claim: str,
    *,
    unavailable_reason: str,
) -> EvidenceBoundClaim:
    if claim:
        return EvidenceBoundClaim(
            claim=claim,
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.SUPPORTED,
                sources=[EvidenceSource.ABSTRACT],
            ),
        )
    return _unavailable_claim(unavailable_reason)


def _reading_guide_claim(
    item: PaperBriefingItem,
    recommendation: Recommendation,
    *,
    topic: str,
    has_abstract: bool,
) -> EvidenceBoundClaim:
    if has_abstract:
        claim = (
            f"Read rank {item.rank} for abstract-backed evidence on '{topic}', "
            f"then check the ranking rationale: {recommendation.rationale}"
        )
        return EvidenceBoundClaim(
            claim=claim,
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.PARTIAL,
                sources=_ordered_sources([EvidenceSource.ABSTRACT, EvidenceSource.RANKING]),
                note="Reading guidance combines abstract evidence with ranking context.",
            ),
        )

    return EvidenceBoundClaim(
        claim=(
            f"Treat rank {item.rank} as a metadata-only lead for '{topic}'; "
            "verify the abstract or full text before drawing technical conclusions."
        ),
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.PARTIAL,
            sources=_ordered_sources([EvidenceSource.METADATA, EvidenceSource.RANKING]),
            note="Reading guidance is limited to metadata and ranking context.",
        ),
    )


def _claims_from_items(
    claims: list[str],
    *,
    sources: list[EvidenceSource],
    unavailable_reason: str,
) -> list[EvidenceBoundClaim]:
    supported_claims = [
        claim
        for claim in claims
        if claim.strip() and not _looks_like_abstention(claim)
    ]
    if supported_claims and sources:
        return [
            EvidenceBoundClaim(
                claim=claim,
                evidence=FieldEvidenceStatus(
                    status=EvidenceSupportStatus.SUPPORTED,
                    sources=sources,
                ),
            )
            for claim in supported_claims
        ]
    return [_unavailable_claim(unavailable_reason)]


def _unavailable_claim(reason: str) -> EvidenceBoundClaim:
    return EvidenceBoundClaim(
        claim=None,
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.UNAVAILABLE,
            abstention_reason=reason,
        ),
    )


def _sentence_with_keywords(text: str, *keywords: str) -> str:
    for sentence in _split_sentences(text):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            return sentence
    return ""


def _split_sentences(text: str) -> list[str]:
    stripped = " ".join(text.split())
    if not stripped:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", stripped)
        if sentence.strip()
    ]


def _looks_like_abstention(text: str) -> bool:
    lowered = text.lower()
    abstention_markers = (
        "unavailable",
        "not available",
        "not found",
        "no abstract",
        "no explicit",
        "metadata only",
        "metadata-only",
        "claims are unavailable",
    )
    return any(marker in lowered for marker in abstention_markers)


def _ordered_sources(sources: list[EvidenceSource]) -> list[EvidenceSource]:
    order = {
        EvidenceSource.METADATA: 0,
        EvidenceSource.ABSTRACT: 1,
        EvidenceSource.RANKING: 2,
        EvidenceSource.RETRIEVAL_METADATA: 3,
        EvidenceSource.CANDIDATE_POOL: 4,
        EvidenceSource.FULL_TEXT: 5,
        EvidenceSource.MIXED: 6,
    }
    seen: set[EvidenceSource] = set()
    unique: list[EvidenceSource] = []
    for source in sorted(sources, key=lambda source: order[source]):
        if source not in seen:
            unique.append(source)
            seen.add(source)
    return unique
