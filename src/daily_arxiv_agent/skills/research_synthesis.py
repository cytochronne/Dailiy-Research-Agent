"""Public Skill facade for briefing synthesis and paper explanation."""

from __future__ import annotations

from collections.abc import Sequence

from daily_arxiv_agent.contracts import (
    DailyBriefing,
    ExplanationMode,
    PaperBriefingItem,
    PaperDeepExplanation,
    PaperMetadata,
    QueryPlan,
    Recommendation,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SkillResult,
)
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.skills.briefing import DailyBriefingSkill
from daily_arxiv_agent.skills.deep_explanation import PaperDeepExplanationSkill
from daily_arxiv_agent.skills.extraction import PaperExtractionSkill
from daily_arxiv_agent.storage import SQLitePaperStore


class ResearchSynthesisSkill:
    """Coordinate the public synthesis/explanation Skill surface."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        store: SQLitePaperStore | None = None,
        extraction_skill: PaperExtractionSkill | None = None,
        briefing_skill: DailyBriefingSkill | None = None,
        deep_explanation_skill: PaperDeepExplanationSkill | None = None,
    ) -> None:
        self.extraction_skill = extraction_skill or PaperExtractionSkill(
            provider=provider
        )
        self.briefing_skill = briefing_skill or DailyBriefingSkill(provider=provider)
        self.deep_explanation_skill = (
            deep_explanation_skill
            or PaperDeepExplanationSkill(provider=provider, store=store)
        )

    def extract_paper(
        self,
        recommendation: Recommendation,
        *,
        topic: str,
    ) -> SkillResult[PaperBriefingItem]:
        return self.extraction_skill.extract(recommendation, topic=topic)

    def generate_briefing(
        self,
        *,
        topic: str,
        recommendations: Sequence[Recommendation],
        extraction_results: Sequence[SkillResult[PaperBriefingItem]] | None = None,
        candidate_papers: Sequence[PaperMetadata] | None = None,
        query_plan: QueryPlan | None = None,
        retrieval_query: RetrievalQuery | None = None,
        retrieval_source_metadata_by_paper_id: dict[
            str, Sequence[RetrievalSourceMetadata]
        ]
        | None = None,
        ranking_metadata: dict[str, object] | None = None,
    ) -> SkillResult[DailyBriefing]:
        return self.briefing_skill.generate(
            topic=topic,
            recommendations=recommendations,
            extraction_results=extraction_results,
            candidate_papers=candidate_papers,
            query_plan=query_plan,
            retrieval_query=retrieval_query,
            retrieval_source_metadata_by_paper_id=(
                retrieval_source_metadata_by_paper_id
            ),
            ranking_metadata=ranking_metadata,
        )

    def explain_paper(
        self,
        paper: PaperMetadata,
        *,
        mode: ExplanationMode,
        full_text: str | None = None,
    ) -> SkillResult[PaperDeepExplanation]:
        return self.deep_explanation_skill.explain(
            paper,
            mode=mode,
            full_text=full_text,
        )
