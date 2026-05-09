"""Public Skill facade for discovery, recommendation, and refinement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from daily_arxiv_agent.config import AppConfig
from daily_arxiv_agent.contracts import (
    FeedbackEvent,
    PaperMetadata,
    QueryPlan,
    Recommendation,
    RetrievalQuery,
    RetrievalSourceMetadata,
    SeedPreference,
    SkillResult,
)
from daily_arxiv_agent.embeddings.base import SemanticReadiness
from daily_arxiv_agent.llm.base import LLMProvider
from daily_arxiv_agent.skills.arxiv_retrieval import ArxivRetrievalSkill
from daily_arxiv_agent.skills.feedback import FeedbackInput, FeedbackRefinementSkill
from daily_arxiv_agent.skills.followup import FollowupQuery, FollowupSkill
from daily_arxiv_agent.skills.query_planning import QueryPlanningSkill
from daily_arxiv_agent.skills.ranking import TopicRankingSkill
from daily_arxiv_agent.skills.seed_parsing import SeedParsingSkill
from daily_arxiv_agent.skills.semantic_seed_ranking import SemanticSeedRankingSkill
from daily_arxiv_agent.storage import SQLitePaperStore


class DiscoveryRecommendationSkill:
    """Coordinate the public discovery/recommendation Skill surface.

    Existing single-purpose Skills remain the internal implementation units so their
    tests, imports, and fallback behavior stay stable.
    """

    def __init__(
        self,
        *,
        store: SQLitePaperStore | None = None,
        provider: LLMProvider | None = None,
        config: AppConfig | None = None,
        seed_parsing_skill: SeedParsingSkill | None = None,
        query_planning_skill: QueryPlanningSkill | None = None,
        retrieval_skill: ArxivRetrievalSkill | None = None,
        ranking_skill: TopicRankingSkill | None = None,
        semantic_ranking_skill: SemanticSeedRankingSkill | None = None,
        feedback_skill: FeedbackRefinementSkill | None = None,
        followup_skill: FollowupSkill | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.store = store or SQLitePaperStore(self.config.db_path)
        self.seed_parsing_skill = seed_parsing_skill or SeedParsingSkill()
        self.query_planning_skill = query_planning_skill or QueryPlanningSkill(
            provider=provider
        )
        self.retrieval_skill = retrieval_skill or ArxivRetrievalSkill(
            store=self.store,
            request_delay_seconds=self.config.arxiv_request_delay_seconds,
        )
        self.ranking_skill = ranking_skill or TopicRankingSkill()
        self.semantic_ranking_skill = semantic_ranking_skill or SemanticSeedRankingSkill(
            store=self.store,
            config=self.config,
        )
        self.feedback_skill = feedback_skill or FeedbackRefinementSkill(store=self.store)
        self.followup_skill = followup_skill or FollowupSkill(
            store=self.store,
            retrieval_skill=self.retrieval_skill,
        )

    def build_seed_preference(
        self,
        seeds: Sequence[str],
        *,
        profile_id: str = "default",
    ) -> SkillResult[SeedPreference]:
        return self.seed_parsing_skill.build_preference(
            seeds,
            profile_id=profile_id,
        )

    def plan_query(self, query: RetrievalQuery) -> SkillResult[QueryPlan]:
        return self.query_planning_skill.plan(query)

    def plan_query_from_seed(
        self,
        query: RetrievalQuery,
        seed_preference: SeedPreference,
    ) -> SkillResult[QueryPlan]:
        return self.query_planning_skill.plan_from_seed(query, seed_preference)

    def retrieve_papers(
        self,
        query: RetrievalQuery,
        *,
        use_cache: bool = True,
        query_plan: QueryPlan | None = None,
    ) -> SkillResult[list[PaperMetadata]]:
        return self.retrieval_skill.retrieve(
            query,
            use_cache=use_cache,
            query_plan=query_plan,
        )

    def rank_recommendations(
        self,
        papers: Sequence[PaperMetadata],
        *,
        topic: str | None = None,
        seed_preference: SeedPreference | None = None,
        feedback_events: Sequence[FeedbackEvent] | None = None,
        top_k: int = 5,
        query_plan: QueryPlan | None = None,
        retrieval_query: RetrievalQuery | None = None,
        retrieval_source_metadata_by_paper_id: Mapping[
            str, Sequence[RetrievalSourceMetadata | Mapping[str, object]]
        ]
        | None = None,
    ) -> SkillResult[list[Recommendation]]:
        return self.ranking_skill.rank(
            papers,
            topic=topic,
            seed_preference=seed_preference,
            feedback_events=feedback_events,
            top_k=top_k,
            query_plan=query_plan,
            retrieval_query=retrieval_query,
            retrieval_source_metadata_by_paper_id=(
                retrieval_source_metadata_by_paper_id
            ),
        )

    def check_semantic_readiness(
        self,
        seed_preference: SeedPreference | None,
    ) -> SemanticReadiness:
        return self.semantic_ranking_skill.check_readiness(seed_preference)

    def rank_semantic_recommendations(
        self,
        papers: Sequence[PaperMetadata],
        *,
        topic: str | None = None,
        seed_preference: SeedPreference | None = None,
        feedback_events: Sequence[FeedbackEvent] | None = None,
        top_k: int = 5,
        query_plan: QueryPlan | None = None,
        retrieval_query: RetrievalQuery | None = None,
        retrieval_source_metadata_by_paper_id: Mapping[
            str, Sequence[RetrievalSourceMetadata | Mapping[str, object]]
        ]
        | None = None,
        profile_id: str | None = None,
    ) -> SkillResult[list[Recommendation]]:
        return self.semantic_ranking_skill.rank(
            papers,
            topic=topic,
            seed_preference=seed_preference,
            feedback_events=feedback_events,
            top_k=top_k,
            query_plan=query_plan,
            retrieval_query=retrieval_query,
            retrieval_source_metadata_by_paper_id=(
                retrieval_source_metadata_by_paper_id
            ),
            profile_id=profile_id,
        )

    def record_feedback(
        self,
        feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]],
        *,
        recommendations: Sequence[Recommendation] = (),
        papers: Sequence[PaperMetadata] = (),
        profile_id: str = "default",
        recommendation_run_id: str | None = None,
    ) -> SkillResult[list[FeedbackEvent]]:
        return self.feedback_skill.record_feedback(
            feedback,
            recommendations=recommendations,
            papers=papers,
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
        )

    def refine_feedback(
        self,
        recommendations: Sequence[Recommendation],
        *,
        feedback: Sequence[FeedbackInput | FeedbackEvent | dict[str, object]] = (),
        papers: Sequence[PaperMetadata] = (),
        profile_id: str = "default",
        recommendation_run_id: str | None = None,
        semantic_context: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SkillResult[list[Recommendation]]:
        return self.feedback_skill.refine(
            recommendations,
            feedback=feedback,
            papers=papers,
            profile_id=profile_id,
            recommendation_run_id=recommendation_run_id,
            semantic_context=semantic_context,
            top_k=top_k,
        )

    def query_followup(
        self,
        query: FollowupQuery,
    ) -> SkillResult[list[PaperMetadata]]:
        return self.followup_skill.query(query)
