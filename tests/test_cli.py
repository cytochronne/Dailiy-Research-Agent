from __future__ import annotations

from datetime import date
import json

import daily_arxiv_agent.cli as cli_module
from daily_arxiv_agent.cli import compact_briefing_output, main
from daily_arxiv_agent.contracts import (
    BriefingEvidenceBoundary,
    BriefingTableRow,
    CandidatePoolTrendOverview,
    DailyBriefing,
    EmbeddingCacheScope,
    EmbeddingIdentity,
    EvidenceBoundClaim,
    EvidenceSource,
    EvidenceSupportStatus,
    FieldEvidenceStatus,
    PaperBriefingItem,
    PaperMetadata,
    Provenance,
    ReadingPriority,
    Recommendation,
    RetrievalQuery,
    SkillError,
    SkillResult,
    SkillStatus,
    TopKComparisonNote,
    TrendAssessmentStatus,
    TrendSignal,
    TrendSignalStrength,
    TrendSignalType,
)
from daily_arxiv_agent.orchestrator import RecommendationWorkflow
from daily_arxiv_agent.storage import SQLitePaperStore


def make_paper() -> PaperMetadata:
    return PaperMetadata(
        paper_id="2604.00001",
        title="Agent Workflows for Research Recommendation",
        authors=["Ada Lovelace"],
        abstract="Daily research agents rank and summarize new papers.",
        categories=["cs.LG"],
        published_date=date(2026, 4, 20),
        updated_date=date(2026, 4, 20),
        arxiv_url="https://arxiv.org/abs/2604.00001",
        pdf_url="https://arxiv.org/pdf/2604.00001",
        provenance=Provenance(
            source="arxiv",
            source_url="https://arxiv.org/abs/2604.00001",
            query="agent briefing",
        ),
    )


def make_cli_result(*, status: SkillStatus = SkillStatus.SUCCESS) -> SkillResult[RecommendationWorkflow]:
    paper = make_paper()
    item = PaperBriefingItem(
        paper_id=paper.paper_id,
        title=paper.title,
        rank=1,
        score=8.5,
        summary="Agent workflows can structure daily research recommendation.",
        contributions=["Connects ranking evidence to a briefing workflow."],
        methods=["Staged retrieval, ranking, and synthesis."],
        relevance_rationale="Matched agent and briefing terms.",
        evidence_source=EvidenceSource.ABSTRACT,
        provenance=paper.provenance,
        arxiv_url=paper.arxiv_url,
        problem=EvidenceBoundClaim(
            claim="Daily paper monitoring needs traceable recommendation context.",
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.SUPPORTED,
                sources=[EvidenceSource.ABSTRACT],
            ),
        ),
        approach=EvidenceBoundClaim(
            claim="The workflow stages retrieval, ranking, and briefing.",
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.SUPPORTED,
                sources=[EvidenceSource.ABSTRACT],
            ),
        ),
        reading_guide=EvidenceBoundClaim(
            claim="Read first for the workflow shape and evidence labels.",
            evidence=FieldEvidenceStatus(
                status=EvidenceSupportStatus.PARTIAL,
                sources=[EvidenceSource.ABSTRACT, EvidenceSource.RANKING],
                note="Reading guidance combines abstract evidence with ranking context.",
            ),
        ),
    )
    briefing = DailyBriefing(
        topic="agent briefing",
        executive_summary="Top papers emphasize traceable agent briefing workflows.",
        summary_table=[
            BriefingTableRow(
                rank=1,
                paper_id=paper.paper_id,
                title=paper.title,
                score=8.5,
                key_reason="Matched agent and briefing terms.",
                evidence_source=EvidenceSource.ABSTRACT,
                arxiv_url=paper.arxiv_url,
            )
        ],
        highlighted_paper=item,
        items=[item],
        evidence_source=EvidenceSource.MIXED,
        trend_overview=CandidatePoolTrendOverview(
            status=TrendAssessmentStatus.AVAILABLE,
            summary="Agent workflow appears across candidate abstracts.",
            candidate_count=6,
            abstract_count=5,
            metadata_only_count=1,
            top_k_count=1,
            signals=[
                TrendSignal(
                    label="agent workflow",
                    signal_type=TrendSignalType.HOTSPOT,
                    strength=TrendSignalStrength.MODERATE,
                    support_count=4,
                    candidate_count=6,
                    top_k_count=1,
                    evidence_sources=[
                        EvidenceSource.CANDIDATE_POOL,
                        EvidenceSource.ABSTRACT,
                    ],
                    summary="Repeated across abstracts and titles.",
                )
            ],
            evidence_sources=[EvidenceSource.CANDIDATE_POOL, EvidenceSource.ABSTRACT],
        ),
        top_k_comparisons=[
            TopKComparisonNote(
                dimension="ranking context",
                note="Rank 1 leads on abstract-backed relevance.",
                paper_ids=[paper.paper_id],
                ranks=[1],
                evidence=FieldEvidenceStatus(
                    status=EvidenceSupportStatus.SUPPORTED,
                    sources=[EvidenceSource.RANKING, EvidenceSource.ABSTRACT],
                ),
            )
        ],
        reading_priorities=[
            ReadingPriority(
                priority=1,
                reading_intent="start with abstract-backed workflow evidence",
                paper_id=paper.paper_id,
                rank=1,
                reason="It has the strongest score and abstract support.",
                evidence=FieldEvidenceStatus(
                    status=EvidenceSupportStatus.SUPPORTED,
                    sources=[EvidenceSource.RANKING, EvidenceSource.ABSTRACT],
                ),
            )
        ],
        evidence_boundary=BriefingEvidenceBoundary(
            evidence_sources=[
                EvidenceSource.METADATA,
                EvidenceSource.ABSTRACT,
                EvidenceSource.RANKING,
                EvidenceSource.CANDIDATE_POOL,
            ],
            unavailable_sources=[EvidenceSource.FULL_TEXT],
            full_text_used=False,
            notes=["No PDF or full-text evidence was used."],
            abstentions=[
                EvidenceBoundClaim(
                    claim=None,
                    evidence=FieldEvidenceStatus(
                        status=EvidenceSupportStatus.UNAVAILABLE,
                        abstention_reason=(
                            "PDF and full-text evidence were not used in the default briefing."
                        ),
                    ),
                )
            ],
        ),
    )
    workflow = RecommendationWorkflow(
        run_id="run-cli-briefing",
        topic="agent briefing",
        query=RetrievalQuery(topic="agent briefing"),
        papers=[paper],
        recommendations=[
            Recommendation(
                paper=paper,
                rank=1,
                score=8.5,
                rationale="Matched agent and briefing terms.",
                evidence_source=EvidenceSource.ABSTRACT,
            )
        ],
        briefing=briefing,
    )
    if status == SkillStatus.FALLBACK:
        return SkillResult(
            status=status,
            data=workflow,
            evidence_source=EvidenceSource.MIXED,
            error=SkillError(
                code="llm_briefing_failed",
                message="LLM briefing generation failed.",
                retryable=True,
            ),
            message="Using deterministic fallback briefing.",
        )
    return SkillResult(
        status=status,
        data=workflow,
        evidence_source=EvidenceSource.MIXED,
    )


def test_cli_compact_briefing_output_uses_required_section_order(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_module, "_run_demo", lambda args: make_cli_result())

    exit_code = main(["demo", "--format", "briefing"])

    output = capsys.readouterr().out
    headings = [
        "## Executive Summary",
        "## Top-K Reading Guide",
        "## Evidence Boundary",
    ]
    positions = [output.index(heading) for heading in headings]
    assert exit_code == 0
    assert positions == sorted(positions)
    assert "## Trend / Hotspot Overview" not in output
    assert "## Top-K Comparison" not in output
    assert "## Reading Priorities" not in output
    assert "Status: success" in output
    assert "Full text used: no" in output
    assert "No PDF or full-text evidence was used." in output


def test_cli_json_output_remains_default_and_keeps_enhanced_fields(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli_module, "_run_demo", lambda args: make_cli_result())

    exit_code = main(["demo"])

    payload = json.loads(capsys.readouterr().out)
    briefing = payload["data"]["briefing"]
    assert exit_code == 0
    assert payload["status"] == "success"
    assert briefing["trend_overview"]["status"] == "available"
    assert briefing["trend_overview"]["signals"][0]["label"] == "agent workflow"
    assert briefing["top_k_comparisons"][0]["dimension"] == "ranking context"
    assert briefing["reading_priorities"][0]["reading_intent"] == (
        "start with abstract-backed workflow evidence"
    )
    assert briefing["evidence_boundary"]["full_text_used"] is False


def test_cli_compact_briefing_output_surfaces_fallback_notice() -> None:
    output = compact_briefing_output(make_cli_result(status=SkillStatus.FALLBACK))

    assert "Status: fallback" in output
    assert "Notice: Using deterministic fallback briefing." in output
    assert "Fallback: llm_briefing_failed" in output
    assert "## Evidence Boundary" in output


def test_cli_demo_builds_seed_preference_from_inline_and_file(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    class SpyStore:
        def __init__(self) -> None:
            self.saved = []

        def save_seed_preference(self, preference):  # noqa: ANN001, ANN201
            self.saved.append(preference)

    class SpyOrchestrator:
        def __init__(self) -> None:
            self.store = SpyStore()
            self.calls = []

        def run_recommendation(self, query, **kwargs):  # noqa: ANN001, ANN003, ANN201
            self.calls.append({"query": query, **kwargs})
            return make_cli_result()

    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("Embodied task planning from scene goals\n\n")
    orchestrator = SpyOrchestrator()
    monkeypatch.setattr(cli_module, "_build_orchestrator", lambda args: orchestrator)

    exit_code = main(
        [
            "demo",
            "--topic",
            "",
            "--profile-id",
            "cli-profile",
            "--seed",
            "Learning patches from execution traces",
            "--seed-file",
            str(seed_file),
            "--recommendation-mode",
            "semantic-seed",
            "--no-embedding-cache",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    call = orchestrator.calls[0]
    preference = call["seed_preference"]
    assert exit_code == 0
    assert payload["status"] == "success"
    assert call["query"].topic is None
    assert call["profile_id"] == "cli-profile"
    assert call["recommendation_mode"] == "semantic-seed"
    assert preference.profile_id == "cli-profile"
    assert [seed.input_text for seed in preference.seeds] == [
        "Learning patches from execution traces",
        "Embodied task planning from scene goals",
    ]
    assert orchestrator.store.saved == [preference]


def test_cli_semantic_seed_mode_runs_with_fake_embeddings(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_CACHE_ENABLED", "false")

    exit_code = main(
        [
            "demo",
            "--fixture",
            "tests/fixtures/arxiv_atom_response.xml",
            "--db-path",
            str(tmp_path / "semantic.sqlite3"),
            "--topic",
            "",
            "--seed",
            "Agent workflows for research paper recommendation",
            "--recommendation-mode",
            "semantic-seed",
            "--top-k",
            "1",
            "--candidate-pool-size",
            "2",
            "--page-size",
            "2",
            "--max-requests",
            "1",
            "--no-cache",
            "--no-embedding-cache",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    trace = payload["data"]["trace"]
    ranking_step = next(step for step in trace if step["skill"] == "ranking")
    assert exit_code == 0
    assert payload["status"] == "success"
    assert [step["skill"] for step in trace][:4] == [
        "query_planning",
        "semantic_readiness",
        "arxiv_retrieval",
        "ranking",
    ]
    assert ranking_step["metadata"]["ranking_mode"] == "semantic_seed"
    assert ranking_step["metadata"]["semantic_provider"]["provider_mode"] == "fake"
    assert ranking_step["metadata"]["embedding_cache"]["enabled"] is False


def test_cli_semantic_mode_without_seed_errors_clearly(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")

    exit_code = main(
        [
            "demo",
            "--db-path",
            str(tmp_path / "semantic.sqlite3"),
            "--topic",
            "robotic manipulation",
            "--recommendation-mode",
            "semantic-seed",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "semantic_seed_quality_error"
    assert payload["data"]["recommendations"] == []
    assert [step["skill"] for step in payload["data"]["trace"]] == [
        "query_planning",
        "semantic_readiness",
    ]


def test_cli_semantic_mode_missing_real_embedding_credentials_surfaces_config_error(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_REUSE_OPENAI_API_KEY", "false")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = main(
        [
            "demo",
            "--db-path",
            str(tmp_path / "semantic.sqlite3"),
            "--topic",
            "",
            "--seed",
            "Learning patches from execution traces",
            "--recommendation-mode",
            "semantic-seed",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "semantic_embedding_credentials_missing"
    assert payload["data"]["recommendations"] == []


def test_cli_embedding_cache_clear_removes_only_embedding_rows(tmp_path, capsys) -> None:
    db_path = tmp_path / "cache.sqlite3"
    store = SQLitePaperStore(db_path)
    identity = EmbeddingIdentity(
        provider="fake",
        model="fake-semantic",
        dimensions=2,
        input_version="semantic-paper-v1",
        input_hash="abc123",
        cache_scope=EmbeddingCacheScope.GLOBAL,
    )
    store.save_embedding(identity, [1.0, 0.0])

    exit_code = main(["embedding-cache", "clear", "--db-path", str(db_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "success"
    assert payload["data"]["deleted_embedding_cache_rows"] == 1
    assert store.load_embedding(identity) is None
