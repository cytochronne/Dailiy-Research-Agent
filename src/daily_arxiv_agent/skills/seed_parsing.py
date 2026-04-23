"""Seed-paper parsing and deterministic preference modeling."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Protocol, Sequence
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    RetrievalQuery,
    SeedPreference,
    SeedRecord,
    SkillError,
    SkillResult,
    SkillStatus,
)
from daily_arxiv_agent.skills.arxiv_retrieval import ARXIV_API_URL, parse_atom_response


class SeedMetadataClient(Protocol):
    """Fetch arXiv metadata for a normalized paper id."""

    def get_metadata(self, paper_id: str) -> PaperMetadata | None:
        """Return metadata for a paper id, or None when unavailable."""


class DeterministicTextVectorizer:
    """Small replaceable vectorizer used for deterministic local ranking."""

    def vectorize(self, text: str) -> dict[str, float]:
        counts = Counter(_tokenize(text))
        return {term: float(count) for term, count in counts.items() if term}


class ArxivSeedMetadataClient:
    """Fetch one seed paper's metadata through the arXiv API."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def get_metadata(self, paper_id: str) -> PaperMetadata | None:
        params = {
            "id_list": paper_id,
            "start": 0,
            "max_results": 1,
        }
        with urlopen(
            f"{ARXIV_API_URL}?{urlencode(params)}",
            timeout=self.timeout_seconds,
        ) as response:
            xml_text = response.read().decode("utf-8")
        papers = parse_atom_response(xml_text, RetrievalQuery(topic=paper_id, max_results=1))
        return papers[0] if papers else None


class SeedParsingSkill:
    """Normalize seed inputs and build a reusable preference representation."""

    def __init__(
        self,
        *,
        metadata_client: SeedMetadataClient | None = None,
        vectorizer: DeterministicTextVectorizer | None = None,
    ) -> None:
        self.metadata_client = metadata_client or ArxivSeedMetadataClient()
        self.vectorizer = vectorizer or DeterministicTextVectorizer()

    def build_preference(
        self,
        seeds: Sequence[str],
        *,
        profile_id: str = "default",
    ) -> SkillResult[SeedPreference]:
        normalized: list[_NormalizedSeed] = []
        invalid_inputs: list[str] = []
        seen: set[str] = set()
        duplicate_count = 0

        for raw_seed in seeds:
            parsed = _normalize_seed(raw_seed)
            if parsed is None:
                invalid_inputs.append(raw_seed)
                continue
            if parsed.identity in seen:
                duplicate_count += 1
                continue
            seen.add(parsed.identity)
            normalized.append(parsed)

        if not normalized:
            return SkillResult[SeedPreference](
                status=SkillStatus.ERROR,
                evidence_source=EvidenceSource.METADATA,
                error=SkillError(
                    code="invalid_seed_input",
                    message="No valid seed papers were provided.",
                    retryable=False,
                ),
                message="Provide at least one arXiv ID, arXiv URL, or title seed.",
                metadata={
                    "invalid_inputs": invalid_inputs,
                    "duplicate_count": duplicate_count,
                },
            )

        records: list[SeedRecord] = []
        fetch_failures: list[str] = []
        for seed in normalized:
            if seed.input_type in {"arxiv_id", "arxiv_url"}:
                try:
                    paper = self.metadata_client.get_metadata(seed.paper_id or "")
                except Exception:
                    paper = None
                    fetch_failures.append(seed.paper_id or seed.input_text)

                if paper is not None:
                    records.append(_record_from_paper(seed, paper))
                    continue

                records.append(_fallback_record(seed))
                if seed.paper_id not in fetch_failures:
                    fetch_failures.append(seed.paper_id or seed.input_text)
                continue

            records.append(_record_from_title(seed))

        preference_text = "\n\n".join(record.preference_text for record in records)
        preference = SeedPreference(
            profile_id=profile_id,
            seeds=records,
            preference_text=preference_text,
            vector=self.vectorizer.vectorize(preference_text),
        )

        status = SkillStatus.SUCCESS
        error: SkillError | None = None
        message = "Built seed-paper preference representation."
        if invalid_inputs or fetch_failures:
            status = SkillStatus.FALLBACK
            error = SkillError(
                code="seed_preference_partial_fallback",
                message=(
                    "Some seed inputs were invalid or could not be resolved; "
                    "available seed text was used."
                ),
                retryable=bool(fetch_failures),
            )
            message = "Built seed-paper preference representation with fallback input."

        evidence_source = (
            EvidenceSource.ABSTRACT
            if any(record.abstract for record in records)
            else EvidenceSource.METADATA
        )

        return SkillResult[SeedPreference](
            status=status,
            data=preference,
            evidence_source=evidence_source,
            provenance=[
                record.paper.provenance
                for record in records
                if record.paper is not None
            ],
            error=error,
            message=message,
            metadata={
                "profile_id": profile_id,
                "seed_count": len(records),
                "duplicate_count": duplicate_count,
                "invalid_inputs": invalid_inputs,
                "fetch_failures": fetch_failures,
            },
        )


def build_paper_preference_text(paper: PaperMetadata) -> str:
    """Build the text used to compare a candidate paper to seed preferences."""

    parts = [
        paper.title,
        paper.abstract or "",
        " ".join(paper.categories),
    ]
    return " ".join(part for part in parts if part).strip()


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute cosine similarity for sparse dict vectors."""

    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    if dot == 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class _NormalizedSeed:
    def __init__(
        self,
        *,
        identity: str,
        input_text: str,
        input_type: str,
        paper_id: str | None = None,
    ) -> None:
        self.identity = identity
        self.input_text = input_text
        self.input_type = input_type
        self.paper_id = paper_id


def _normalize_seed(raw_seed: str) -> _NormalizedSeed | None:
    value = raw_seed.strip()
    if not value:
        return None

    arxiv_id = _extract_arxiv_id(value)
    if arxiv_id:
        input_type = "arxiv_url" if _looks_like_url(value) else "arxiv_id"
        return _NormalizedSeed(
            identity=f"arxiv:{arxiv_id}",
            input_text=value,
            input_type=input_type,
            paper_id=arxiv_id,
        )

    if _looks_like_url(value):
        return None

    title = " ".join(value.split())
    if len(_tokenize(title)) < 2:
        return None
    return _NormalizedSeed(
        identity=f"title:{title.lower()}",
        input_text=title,
        input_type="title",
    )


def _extract_arxiv_id(value: str) -> str | None:
    if _looks_like_url(value):
        parsed = urlparse(value)
        if not parsed.netloc.endswith("arxiv.org"):
            return None
        match = re.match(r"^/(abs|pdf)/([^/?#]+)", parsed.path)
        if not match:
            return None
        return _canonical_arxiv_id(match.group(2))

    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", value):
        return _canonical_arxiv_id(value)
    if re.match(r"^[a-z-]+(?:\.[A-Z]{2})?/\d{7}(v\d+)?$", value):
        return _canonical_arxiv_id(value)
    return None


def _canonical_arxiv_id(value: str) -> str:
    return re.sub(r"(?:\.pdf)?v\d+$", "", value.removesuffix(".pdf"))


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _record_from_paper(seed: _NormalizedSeed, paper: PaperMetadata) -> SeedRecord:
    return SeedRecord(
        identity=seed.identity,
        input_text=seed.input_text,
        input_type=seed.input_type,
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        paper=paper,
        preference_text=build_paper_preference_text(paper),
    )


def _record_from_title(seed: _NormalizedSeed) -> SeedRecord:
    return SeedRecord(
        identity=seed.identity,
        input_text=seed.input_text,
        input_type=seed.input_type,
        title=seed.input_text,
        preference_text=seed.input_text,
    )


def _fallback_record(seed: _NormalizedSeed) -> SeedRecord:
    paper_id = seed.paper_id or seed.input_text
    return SeedRecord(
        identity=seed.identity,
        input_text=seed.input_text,
        input_type=seed.input_type,
        paper_id=seed.paper_id,
        title=paper_id,
        preference_text=paper_id,
    )


def _tokenize(text: str) -> list[str]:
    return [_normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
