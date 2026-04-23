from datetime import date

from daily_arxiv_agent.contracts import PaperMetadata, Provenance, SkillStatus
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill


def make_paper(paper_id: str = "2604.00001") -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title="Explainable Agents for Daily Research Briefings",
        authors=["Ada Lovelace"],
        abstract="We study agent workflows for research-paper recommendation.",
        categories=["cs.LG", "cs.AI"],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query=f"id_list:{paper_id}",
        ),
    )


class FakeSeedMetadataClient:
    def __init__(
        self,
        papers: dict[str, PaperMetadata],
        *,
        failing_ids: set[str] | None = None,
    ) -> None:
        self.papers = papers
        self.failing_ids = failing_ids or set()
        self.calls: list[str] = []

    def get_metadata(self, paper_id: str) -> PaperMetadata | None:
        self.calls.append(paper_id)
        if paper_id in self.failing_ids:
            raise TimeoutError("arXiv unavailable")
        return self.papers.get(paper_id)


def test_arxiv_id_seed_resolves_metadata_and_builds_preference() -> None:
    paper = make_paper()
    client = FakeSeedMetadataClient({paper.paper_id: paper})

    result = SeedParsingSkill(metadata_client=client).build_preference([paper.paper_id])

    assert result.status == SkillStatus.SUCCESS
    preference = result.data
    assert preference is not None
    assert preference.seeds[0].paper_id == paper.paper_id
    assert preference.seeds[0].paper == paper
    assert "agent workflows" in preference.preference_text.lower()
    assert preference.vector["agent"] > 0
    assert client.calls == [paper.paper_id]


def test_arxiv_url_seed_normalizes_to_same_identity_as_id() -> None:
    paper = make_paper()
    client = FakeSeedMetadataClient({paper.paper_id: paper})

    result = SeedParsingSkill(metadata_client=client).build_preference(
        [
            paper.paper_id,
            f"https://arxiv.org/abs/{paper.paper_id}v2",
        ]
    )

    assert result.status == SkillStatus.SUCCESS
    preference = result.data
    assert preference is not None
    assert len(preference.seeds) == 1
    assert preference.seeds[0].identity == f"arxiv:{paper.paper_id}"
    assert result.metadata["duplicate_count"] == 1
    assert client.calls == [paper.paper_id]


def test_title_only_seed_contributes_without_metadata_fetch() -> None:
    client = FakeSeedMetadataClient({})

    result = SeedParsingSkill(metadata_client=client).build_preference(
        ["Graph Neural Retrieval for Scientific Agents"]
    )

    assert result.status == SkillStatus.SUCCESS
    preference = result.data
    assert preference is not None
    assert preference.seeds[0].input_type == "title"
    assert preference.seeds[0].paper is None
    assert "graph neural retrieval" in preference.preference_text.lower()
    assert client.calls == []


def test_duplicate_title_seeds_collapse_to_one_contribution() -> None:
    result = SeedParsingSkill(metadata_client=FakeSeedMetadataClient({})).build_preference(
        [
            "Graph Neural Retrieval for Scientific Agents",
            "Graph Neural Retrieval for Scientific Agents",
        ]
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.data is not None
    assert len(result.data.seeds) == 1
    assert result.metadata["duplicate_count"] == 1


def test_invalid_seed_input_returns_structured_error() -> None:
    result = SeedParsingSkill(metadata_client=FakeSeedMetadataClient({})).build_preference(
        ["https://example.com/not-arxiv"]
    )

    assert result.status == SkillStatus.ERROR
    assert result.error is not None
    assert result.error.code == "invalid_seed_input"
    assert result.data is None


def test_metadata_fetch_failure_falls_back_to_available_seed_text() -> None:
    client = FakeSeedMetadataClient({}, failing_ids={"2604.99999"})

    result = SeedParsingSkill(metadata_client=client).build_preference(["2604.99999"])

    assert result.status == SkillStatus.FALLBACK
    assert result.error is not None
    assert result.error.retryable is True
    assert result.data is not None
    assert result.data.seeds[0].paper_id == "2604.99999"
    assert result.data.preference_text == "2604.99999"
    assert result.metadata["fetch_failures"] == ["2604.99999"]
