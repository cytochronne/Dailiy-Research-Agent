from datetime import date

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    EvidenceSource,
    PaperMetadata,
    Provenance,
    QueryPlan,
    QueryPlannerMode,
    QueryPlannerProvenance,
    QueryPlanVariant,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SearchMode,
    SeedPreference,
    SeedRecord,
    SkillStatus,
)
from daily_arxiv_agent.embeddings.base import EmbeddingProviderError
from daily_arxiv_agent.embeddings.fake import FakeEmbeddingProvider
from daily_arxiv_agent.skills.seed_parsing import (
    DeterministicTextVectorizer,
    build_paper_preference_text,
)
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


def make_paper(
    paper_id: str,
    title: str,
    abstract: str | None,
    *,
    category: str = "cs.LG",
    published_date: date = date(2026, 4, 20),
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        abstract=abstract,
        categories=[category],
        published_date=published_date,
        updated_date=published_date,
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        provenance=Provenance(
            source="arxiv",
            source_url=f"https://arxiv.org/abs/{paper_id}",
            query="semantic seed ranking",
        ),
    )


def make_seed_preference(
    papers: list[PaperMetadata],
    *,
    profile_id: str = "default",
) -> SeedPreference:
    records = [
        SeedRecord(
            identity=f"arxiv:{paper.paper_id}",
            input_text=paper.paper_id,
            input_type="arxiv_id",
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            paper=paper,
            preference_text=build_paper_preference_text(paper),
        )
        for paper in papers
    ]
    preference_text = "\n\n".join(record.preference_text for record in records)
    return SeedPreference(
        profile_id=profile_id,
        seeds=records,
        preference_text=preference_text,
        vector=DeterministicTextVectorizer().vectorize(preference_text),
    )


def seed_query_plan() -> QueryPlan:
    return QueryPlan(
        search_mode=SearchMode.BROAD,
        planner=QueryPlannerProvenance(
            requested_mode=QueryPlannerMode.DETERMINISTIC,
            source="seed_derived",
        ),
        variants=[
            QueryPlanVariant(
                label="seed_terms",
                search_query='all:"program repair"',
                sort_by="relevance",
            )
        ],
        required_terms=["graph", "neural", "program", "repair"],
        phrases=["graph neural program repair"],
    )


def semantic_config(*, dimensions: int = 3) -> AppConfig:
    return AppConfig(
        embedding_provider="fake",
        embedding_model="fake-semantic",
        embedding_dimensions=dimensions,
    )


def test_semantic_seed_ranking_demotes_lexical_distractor(tmp_path) -> None:
    seed = make_paper(
        "2604.30001",
        "Graph Neural Program Repair",
        "Neural models repair code defects from failing tests.",
        category="cs.SE",
    )
    related = make_paper(
        "2604.30002",
        "Learning Patches from Execution Traces",
        "Models synthesize bug fixes from test failures and runtime traces.",
        category="cs.SE",
    )
    lexical_distractor = make_paper(
        "2604.30003",
        "Graph Neural Program Repair Index",
        (
            "Graph neural program repair terms appear repeatedly, but the paper "
            "only indexes citation metadata."
        ),
        category="cs.SE",
    )
    unrelated = make_paper(
        "2604.30004",
        "Compiler Register Allocation",
        "A survey of register pressure in optimizing compilers.",
        category="cs.PL",
    )
    vectors = {
        build_paper_preference_text(seed): [1.0, 0.0, 0.0],
        build_paper_preference_text(related): [0.98, 0.02, 0.0],
        build_paper_preference_text(lexical_distractor): [0.0, 1.0, 0.0],
        build_paper_preference_text(unrelated): [-1.0, 0.0, 0.0],
    }
    provider = FakeEmbeddingProvider(dimensions=3, vector_map=vectors)
    skill = SemanticSeedRankingSkill(
        embedding_provider=provider,
        store=SQLitePaperStore(tmp_path / "semantic.sqlite3"),
        config=semantic_config(),
        minimum_semantic_similarity=0.4,
    )

    result = skill.rank(
        [lexical_distractor, related, seed, unrelated],
        seed_preference=make_seed_preference([seed], profile_id="demo"),
        query_plan=seed_query_plan(),
        retrieval_query=RetrievalQuery(category="cs.SE"),
        retrieval_source_metadata_by_paper_id={
            lexical_distractor.paper_id: [
                RetrievalSourceMetadata(
                    variant_label="seed_terms",
                    sort_by="relevance",
                    variant_index=0,
                    position=0,
                    first_seen_order=0,
                )
            ],
            related.paper_id: [
                RetrievalSourceMetadata(
                    variant_label="seed_terms",
                    sort_by="relevance",
                    variant_index=0,
                    position=1,
                    first_seen_order=1,
                )
            ],
        },
        top_k=3,
    )

    assert result.status == SkillStatus.SUCCESS
    recommendations = result.data or []
    assert seed.paper_id not in [item.paper.paper_id for item in recommendations]
    assert recommendations[0].paper.paper_id == related.paper_id
    assert [item.paper.paper_id for item in recommendations].index(
        related.paper_id
    ) < [item.paper.paper_id for item in recommendations].index(
        lexical_distractor.paper_id
    )
    first_breakdown = recommendations[0].score_breakdown
    assert first_breakdown is not None
    assert first_breakdown.semantic_seed > 0
    assert first_breakdown.evidence_score >= 0.4
    assert len(first_breakdown.semantic_similarities) == 1
    assert first_breakdown.semantic_similarities[0].source_id == seed.paper_id
    assert result.metadata["ranking_mode"] == "semantic_seed"
    assert "semantic_seed" in result.metadata["score_signals"]


def test_semantic_embedding_cache_reuses_seed_and_candidate_vectors(tmp_path) -> None:
    seed = make_paper(
        "2604.31001",
        "Vision Language Robot Planning",
        "Embodied agents plan manipulation tasks from visual context.",
        category="cs.RO",
    )
    candidate = make_paper(
        "2604.31002",
        "Embodied Task Planning from Scene Goals",
        "Robots infer manipulation steps from visual goals.",
        category="cs.RO",
    )
    vectors = {
        build_paper_preference_text(seed): [1.0, 0.0, 0.0],
        build_paper_preference_text(candidate): [0.9, 0.1, 0.0],
    }
    provider = FakeEmbeddingProvider(dimensions=3, vector_map=vectors)
    skill = SemanticSeedRankingSkill(
        embedding_provider=provider,
        store=SQLitePaperStore(tmp_path / "semantic.sqlite3"),
        config=semantic_config(),
    )
    preference = make_seed_preference([seed], profile_id="demo")

    first = skill.rank([candidate], seed_preference=preference, top_k=1)
    second = skill.rank([candidate], seed_preference=preference, top_k=1)
    other_profile = skill.rank(
        [candidate],
        seed_preference=preference,
        profile_id="other-profile",
        top_k=1,
    )

    assert first.status == SkillStatus.SUCCESS
    assert second.status == SkillStatus.SUCCESS
    assert other_profile.status == SkillStatus.SUCCESS
    assert len(provider.calls) == 3
    assert first.metadata["embedding_cache"]["misses"] == 2
    assert first.metadata["embedding_cache"]["writes"] == 2
    assert second.metadata["embedding_cache"]["hits"] == 2
    assert second.metadata["embedding_cache"]["misses"] == 0
    assert other_profile.metadata["embedding_cache"]["hits"] == 1
    assert other_profile.metadata["embedding_cache"]["misses"] == 1


def test_multi_seed_diagnostics_preserve_each_seed_similarity(tmp_path) -> None:
    robot_seed = make_paper(
        "2604.32001",
        "Vision Language Robot Planning",
        "Embodied agents plan manipulation tasks from visual context.",
        category="cs.RO",
    )
    compiler_seed = make_paper(
        "2604.32002",
        "Compiler Register Allocation",
        "Graph coloring allocates registers under pressure.",
        category="cs.PL",
    )
    robot_candidate = make_paper(
        "2604.32003",
        "Embodied Task Planning",
        "Robots infer manipulation steps from scene goals.",
        category="cs.RO",
    )
    compiler_candidate = make_paper(
        "2604.32004",
        "Spill-Aware Register Assignment",
        "Compilers allocate registers while reducing spill cost.",
        category="cs.PL",
    )
    vectors = {
        build_paper_preference_text(robot_seed): [1.0, 0.0, 0.0],
        build_paper_preference_text(compiler_seed): [0.0, 1.0, 0.0],
        build_paper_preference_text(robot_candidate): [0.9, 0.1, 0.0],
        build_paper_preference_text(compiler_candidate): [0.1, 0.95, 0.0],
    }
    skill = SemanticSeedRankingSkill(
        embedding_provider=FakeEmbeddingProvider(dimensions=3, vector_map=vectors),
        store=SQLitePaperStore(tmp_path / "semantic.sqlite3"),
        config=semantic_config(),
    )

    result = skill.rank(
        [robot_candidate, compiler_candidate],
        seed_preference=make_seed_preference([robot_seed, compiler_seed]),
        top_k=2,
    )

    assert result.status == SkillStatus.SUCCESS
    recommendations = result.data or []
    assert len(recommendations) == 2
    top_matching_seed_ids = set()
    for recommendation in recommendations:
        breakdown = recommendation.score_breakdown
        assert breakdown is not None
        assert len(breakdown.semantic_similarities) == 2
        top_detail = max(
            breakdown.semantic_similarities,
            key=lambda item: item.similarity,
        )
        top_matching_seed_ids.add(top_detail.source_id)

    assert top_matching_seed_ids == {robot_seed.paper_id, compiler_seed.paper_id}
    assert result.metadata["semantic_context"]["aggregation"] == "max_per_seed"


class RaisingEmbeddingProvider:
    def embed_texts(self, texts):  # noqa: ANN001, ANN201
        raise EmbeddingProviderError("embedding service unavailable")


def test_embedding_provider_failure_returns_semantic_error_without_recommendations(
    tmp_path,
) -> None:
    seed = make_paper(
        "2604.33001",
        "Vision Language Robot Planning",
        "Embodied agents plan manipulation tasks from visual context.",
        category="cs.RO",
    )
    candidate = make_paper(
        "2604.33002",
        "Embodied Task Planning",
        "Robots infer manipulation steps from scene goals.",
        category="cs.RO",
    )
    skill = SemanticSeedRankingSkill(
        embedding_provider=RaisingEmbeddingProvider(),
        store=SQLitePaperStore(tmp_path / "semantic.sqlite3"),
        config=semantic_config(),
    )

    result = skill.rank(
        [candidate],
        seed_preference=make_seed_preference([seed]),
        top_k=1,
    )

    assert result.status == SkillStatus.ERROR
    assert result.data == []
    assert result.error is not None
    assert result.error.code == "semantic_embedding_provider_failed"
    assert result.metadata["ranking_mode"] == "semantic_seed"
    assert result.metadata["semantic_error"]["failure_reason"] == (
        "embedding_provider_failed"
    )
