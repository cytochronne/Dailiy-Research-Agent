"""Deterministic fake LLM provider for tests and local demos."""

from __future__ import annotations

import re
from typing import Any, Sequence

from daily_arxiv_agent.contracts import (
    BriefingEvidenceBoundary,
    CandidatePoolTrendOverview,
    EvidenceBoundClaim,
    EvidenceSource,
    EvidenceSupportStatus,
    ExperimentExplanation,
    ExplanationMode,
    FieldEvidenceStatus,
    LimitationsExplanation,
    MethodExplanation,
    PaperBriefingItem,
    PaperDeepExplanation,
    PaperMetadata,
    ReadingPriority,
    Recommendation,
    RetrievalQuery,
    TopKComparisonNote,
    TrendAssessmentStatus,
)


class FakeLLMProvider:
    """Produce stable structured output without external credentials."""

    def extract_paper(
        self,
        paper: PaperMetadata,
        *,
        topic: str,
        recommendation: Recommendation | None = None,
    ) -> PaperBriefingItem:
        evidence_source = (
            EvidenceSource.ABSTRACT if paper.abstract else EvidenceSource.METADATA
        )
        rank = recommendation.rank if recommendation else 1
        score = recommendation.score if recommendation else 0.0
        rationale = (
            recommendation.rationale
            if recommendation
            else f"Paper metadata is being reviewed for topic '{topic}'."
        )

        if paper.abstract:
            summary = _first_sentence(paper.abstract)
            contributions = _contributions_from_text(paper.abstract, topic)
            methods = _methods_from_text(paper.abstract)
            problem = _abstract_claim(
                _sentence_with_keywords(
                    paper.abstract,
                    "problem",
                    "challenge",
                    "need",
                    "requires",
                    "study",
                    "studies",
                    "address",
                    "addresses",
                ),
                unavailable_reason="The abstract does not state a problem framing.",
            )
            approach = _abstract_claim(
                _sentence_with_keywords(
                    paper.abstract,
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
                ),
            )
            contribution_claims = _claims_from_items(
                contributions,
                sources=[EvidenceSource.ABSTRACT],
                unavailable_reason=(
                    "The abstract does not provide explicit contribution evidence."
                ),
            )
            method_claims = _claims_from_items(
                methods,
                sources=[EvidenceSource.ABSTRACT],
                unavailable_reason=(
                    "The abstract does not provide explicit method evidence."
                ),
            )
        else:
            summary = (
                f"Metadata only: '{paper.title}' has no abstract available, so the "
                "briefing is limited to title, category, and provenance fields."
            )
            contributions = [
                "Metadata indicates a potentially relevant paper, but abstract-level "
                "claims are unavailable."
            ]
            methods = []
            problem = _unavailable_claim(
                "No abstract is available to support a problem claim."
            )
            approach = _unavailable_claim(
                "No abstract is available to support an approach claim."
            )
            contribution_claims = [
                _unavailable_claim(
                    "No abstract is available to support contribution claims."
                )
            ]
            method_claims = [
                _unavailable_claim("No abstract is available to support method claims.")
            ]

        return PaperBriefingItem(
            paper_id=paper.paper_id,
            title=paper.title,
            rank=rank,
            score=score,
            summary=summary,
            contributions=contributions,
            methods=methods,
            relevance_rationale=f"{rationale} Evidence: {evidence_source.value}.",
            evidence_source=evidence_source,
            provenance=paper.provenance,
            arxiv_url=paper.arxiv_url,
            problem=problem,
            approach=approach,
            reading_guide=_reading_guide_claim(
                rank=rank,
                topic=topic,
                rationale=rationale,
                has_abstract=bool(paper.abstract),
            ),
            contribution_claims=contribution_claims,
            method_claims=method_claims,
            relevance_evidence=FieldEvidenceStatus(
                status=(
                    EvidenceSupportStatus.SUPPORTED
                    if paper.abstract
                    else EvidenceSupportStatus.PARTIAL
                ),
                sources=_ordered_sources([EvidenceSource.RANKING, evidence_source]),
                note=(
                    "Relevance is supported by ranking rationale and abstract evidence."
                    if paper.abstract
                    else "Relevance is evidence-limited because only metadata and "
                    "ranking rationale are available."
                ),
            ),
        )

    def summarize_briefing(
        self,
        *,
        topic: str,
        items: Sequence[PaperBriefingItem],
        trend_overview: CandidatePoolTrendOverview | None = None,
        top_k_comparisons: Sequence[TopKComparisonNote] = (),
        reading_priorities: Sequence[ReadingPriority] = (),
        evidence_boundary: BriefingEvidenceBoundary | None = None,
    ) -> str:
        if not items:
            return f"No ranked papers were available for '{topic}'."
        leader = items[0]
        trend_status = (
            trend_overview.status.value
            if trend_overview is not None
            else TrendAssessmentStatus.NOT_ASSESSED.value
        )
        trend_count = len(trend_overview.signals) if trend_overview else 0
        priority_note = (
            reading_priorities[0].reading_intent
            if reading_priorities
            else "start with the highest ranked paper"
        )
        boundary_note = (
            "full text was not used"
            if evidence_boundary is not None and not evidence_boundary.full_text_used
            else "available structured evidence was used"
        )
        return (
            f"{len(items)} Top-K ranked paper(s) were reviewed for '{topic}'. "
            f"Top-K reading guidance starts with '{leader.title}' because "
            f"{priority_note}. Candidate-pool trend context is {trend_status} "
            f"with {trend_count} bounded signal(s); {boundary_note}."
        )

    def explain_paper(
        self,
        paper: PaperMetadata,
        *,
        mode: ExplanationMode,
        content: str,
        evidence_source: EvidenceSource,
    ) -> PaperDeepExplanation:
        sections = _parse_labeled_sections(content)
        summary = sections.get("summary") or _first_sentence(content) or (
            f"Explanation prepared from the available {_source_label(evidence_source)} source."
        )
        evidence_note = (
            f"This explanation is based on the available {_source_label(evidence_source)} source."
        )

        if mode == ExplanationMode.METHOD:
            return PaperDeepExplanation(
                paper_id=paper.paper_id,
                title=paper.title,
                mode=mode,
                summary=summary,
                evidence_source=evidence_source,
                evidence_note=evidence_note,
                method=MethodExplanation(
                    problem=(
                        sections.get("problem")
                        or _sentence_with_keywords(content, "problem", "address", "task")
                        or _missing_evidence("problem statement", evidence_source)
                    ),
                    method_overview=(
                        sections.get("method_overview")
                        or _sentence_with_keywords(
                            content,
                            "method",
                            "approach",
                            "framework",
                            "pipeline",
                            "propose",
                        )
                        or _missing_evidence("method overview", evidence_source)
                    ),
                    core_workflow=(
                        _split_items(sections.get("core_workflow"))
                        or _collect_sentences(
                            content,
                            "workflow",
                            "pipeline",
                            "step",
                            "module",
                        )
                        or [_missing_evidence("core workflow", evidence_source)]
                    ),
                    inputs_outputs=(
                        _split_items(sections.get("inputs_outputs"))
                        or _collect_sentences(content, "input", "output")
                        or [_missing_evidence("inputs and outputs", evidence_source)]
                    ),
                    innovation=(
                        sections.get("innovation")
                        or _sentence_with_keywords(
                            content,
                            "innovation",
                            "novel",
                            "contribution",
                            "improve",
                        )
                        or _missing_evidence("claimed innovation", evidence_source)
                    ),
                ),
                provenance=paper.provenance,
                arxiv_url=paper.arxiv_url,
            )

        if mode == ExplanationMode.EXPERIMENT:
            return PaperDeepExplanation(
                paper_id=paper.paper_id,
                title=paper.title,
                mode=mode,
                summary=summary,
                evidence_source=evidence_source,
                evidence_note=evidence_note,
                experiment=ExperimentExplanation(
                    datasets=_split_items(sections.get("datasets"))
                    or [_missing_evidence("datasets", evidence_source)],
                    baselines=_split_items(sections.get("baselines"))
                    or [_missing_evidence("baselines", evidence_source)],
                    metrics=_split_items(sections.get("metrics"))
                    or [_missing_evidence("metrics", evidence_source)],
                    experimental_setup=(
                        sections.get("experimental_setup")
                        or _sentence_with_keywords(content, "setup", "experiment", "evaluate")
                        or _missing_evidence("experimental setup", evidence_source)
                    ),
                    conclusions=_split_items(sections.get("conclusions"))
                    or _collect_sentences(content, "result", "conclusion", "improve")
                    or [_missing_evidence("main conclusions", evidence_source)],
                ),
                provenance=paper.provenance,
                arxiv_url=paper.arxiv_url,
            )

        return PaperDeepExplanation(
            paper_id=paper.paper_id,
            title=paper.title,
            mode=mode,
            summary=summary,
            evidence_source=evidence_source,
            evidence_note=evidence_note,
            limitations=LimitationsExplanation(
                stated_limitations=_split_items(sections.get("stated_limitations"))
                or [_missing_evidence("stated limitations", evidence_source)],
                assumptions=_split_items(sections.get("assumptions"))
                or [_missing_evidence("assumptions", evidence_source)],
                missing_validation=_split_items(sections.get("missing_validation"))
                or [_missing_evidence("missing validation", evidence_source)],
                risks=_split_items(sections.get("risks"))
                or [_missing_evidence("possible risks", evidence_source)],
            ),
            provenance=paper.provenance,
            arxiv_url=paper.arxiv_url,
        )

    def plan_queries(
        self,
        *,
        query: RetrievalQuery,
        deterministic_terms: Sequence[str],
    ) -> dict[str, Any]:
        """Return deterministic planner-shaped output for offline tests and demos."""

        required_terms = _dedupe_list(deterministic_terms) or _planning_terms(
            query.topic or ""
        )
        related_terms: list[str] = []
        if "llm" in required_terms:
            related_terms.append("language model")
        if "robotic" in required_terms or "robot" in required_terms:
            related_terms.append("embodied control")
        if "agent" in required_terms:
            related_terms.append("autonomous agent")

        phrases = []
        cleaned_topic = _planning_phrase(query.topic or "")
        if cleaned_topic and len(cleaned_topic.split()) > 1:
            phrases.append(cleaned_topic)
        if "robotic" in required_terms and "manipulation" in required_terms:
            phrases.append("robotic manipulation")

        return {
            "source": "fake_llm",
            "model": "fake",
            "required_terms": required_terms,
            "phrases": _dedupe_list(phrases)[:4],
            "related_terms": _dedupe_list(related_terms)[:4],
            "suggested_categories": [query.category] if query.category else [],
            "exclusions": [],
            "rationale": "Deterministic fake provider expanded the retrieval topic.",
        }


def _first_sentence(text: str) -> str:
    stripped = " ".join(text.split())
    match = re.search(r"(.+?[.!?])(?:\s|$)", stripped)
    return match.group(1) if match else stripped


def _planning_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        normalized = _normalize_planning_token(token)
        if normalized and normalized not in terms:
            terms.append(normalized)
    return terms[:8]


def _planning_phrase(text: str) -> str:
    return " ".join(_planning_terms(text))


def _normalize_planning_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _dedupe_list(values: Sequence[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in items:
            items.append(normalized)
    return items


def _contributions_from_text(text: str, topic: str) -> list[str]:
    sentence = _sentence_with_keywords(
        text,
        "contribution",
        "contributes",
        "propose",
        "proposes",
        "present",
        "presents",
        "introduce",
        "introduces",
        "show",
        "shows",
        "demonstrate",
        "demonstrates",
    )
    if not sentence:
        return []
    if topic:
        return [f"Abstract-backed contribution for '{topic}': {sentence}"]
    return [f"Abstract-backed contribution: {sentence}"]


def _methods_from_text(text: str) -> list[str]:
    method_terms = [
        "workflow",
        "framework",
        "approach",
        "method",
        "ranking",
        "recommendation",
        "retrieval",
        "evaluation",
        "model",
    ]
    lower_text = text.lower()
    matched = [term for term in method_terms if term in lower_text]
    if not matched:
        return []
    return [f"Abstract mentions {term}." for term in matched[:3]]


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
    *,
    rank: int,
    topic: str,
    rationale: str,
    has_abstract: bool,
) -> EvidenceBoundClaim:
    if has_abstract:
        return EvidenceBoundClaim(
            claim=(
                f"Read rank {rank} for abstract-backed evidence on '{topic}', "
                f"then compare it with the ranking rationale: {rationale}"
            ),
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.PARTIAL,
                sources=_ordered_sources([EvidenceSource.ABSTRACT, EvidenceSource.RANKING]),
                note="Reading guidance combines abstract evidence with ranking context.",
            ),
        )
    return EvidenceBoundClaim(
        claim=(
            f"Treat rank {rank} as a metadata-only lead for '{topic}'; verify the "
            "abstract or full text before drawing technical conclusions."
        ),
        evidence=FieldEvidenceStatus(
            status=EvidenceSupportStatus.PARTIAL,
            sources=_ordered_sources([EvidenceSource.METADATA, EvidenceSource.RANKING]),
            note="Reading guidance is limited to metadata and ranking context.",
        ),
    )


def _claims_from_items(
    claims: Sequence[str],
    *,
    sources: list[EvidenceSource],
    unavailable_reason: str,
) -> list[EvidenceBoundClaim]:
    supported_claims = [
        claim for claim in claims if claim.strip() and not _looks_like_abstention(claim)
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


def _looks_like_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "unavailable",
            "not available",
            "not found",
            "no abstract",
            "no explicit",
            "metadata only",
            "metadata-only",
            "claims are unavailable",
        )
    )


def _ordered_sources(sources: Sequence[EvidenceSource]) -> list[EvidenceSource]:
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


def _parse_labeled_sections(text: str) -> dict[str, str]:
    aliases = {
        "summary": "summary",
        "problem": "problem",
        "method overview": "method_overview",
        "core workflow": "core_workflow",
        "inputs and outputs": "inputs_outputs",
        "innovation": "innovation",
        "datasets": "datasets",
        "baselines": "baselines",
        "metrics": "metrics",
        "experimental setup": "experimental_setup",
        "conclusions": "conclusions",
        "stated limitations": "stated_limitations",
        "assumptions": "assumptions",
        "missing validation": "missing_validation",
        "risks": "risks",
    }
    sections: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = aliases.get(label.strip().lower())
        cleaned = " ".join(value.split())
        if key and cleaned:
            sections[key] = cleaned
    return sections


def _split_items(value: str | None) -> list[str]:
    if not value:
        return []
    parts = re.split(r"\s*[;|]\s*|\s*\.\s+(?=[A-Z0-9])", value)
    items = [" ".join(part.split()) for part in parts if part.strip()]
    return items


def _sentence_with_keywords(text: str, *keywords: str) -> str:
    sentences = _split_sentences(text)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            return sentence
    return ""


def _collect_sentences(text: str, *keywords: str) -> list[str]:
    sentences = _split_sentences(text)
    matches = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in keywords)
    ]
    return matches[:3]


def _split_sentences(text: str) -> list[str]:
    stripped = " ".join(text.split())
    if not stripped:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", stripped)
        if sentence.strip()
    ]


def _source_label(evidence_source: EvidenceSource) -> str:
    if evidence_source == EvidenceSource.FULL_TEXT:
        return "full-text"
    return evidence_source.value


def _missing_evidence(subject: str, evidence_source: EvidenceSource) -> str:
    return f"{subject.capitalize()} was not found in the available {_source_label(evidence_source)} source."
