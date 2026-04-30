"""Deterministic fake LLM provider for tests and local demos."""

from __future__ import annotations

import re
from typing import Any, Sequence

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    ExperimentExplanation,
    ExplanationMode,
    LimitationsExplanation,
    MethodExplanation,
    PaperBriefingItem,
    PaperDeepExplanation,
    PaperMetadata,
    Recommendation,
    RetrievalQuery,
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
            contributions = [_contribution_from_text(paper.abstract, topic)]
            methods = _methods_from_text(paper.abstract)
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
        )

    def summarize_briefing(
        self,
        *,
        topic: str,
        items: Sequence[PaperBriefingItem],
    ) -> str:
        if not items:
            return f"No ranked papers were available for '{topic}'."
        leader = items[0]
        return (
            f"{len(items)} ranked paper(s) were reviewed for '{topic}'. "
            f"The top paper is '{leader.title}' with evidence from "
            f"{leader.evidence_source.value}."
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


def _contribution_from_text(text: str, topic: str) -> str:
    if topic:
        return f"Connects the paper's abstract evidence to the requested topic: {topic}."
    return f"Summarizes abstract evidence: {_first_sentence(text)}"


def _methods_from_text(text: str) -> list[str]:
    method_terms = [
        "agent",
        "workflow",
        "ranking",
        "recommendation",
        "retrieval",
        "evaluation",
        "model",
    ]
    lower_text = text.lower()
    matched = [term for term in method_terms if term in lower_text]
    if not matched:
        return ["Method details are only available at abstract level."]
    return [f"Abstract mentions {term}." for term in matched[:3]]


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
