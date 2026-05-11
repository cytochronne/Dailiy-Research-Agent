---
name: discovery-recommendation
description: "Plan arXiv discovery, retrieve candidate papers, rank recommendations, and refine them with seed papers or feedback."
author: cytochronne
version: 0.1.0
tags:
  - arxiv
  - recommendation
  - retrieval
  - ranking
  - research-agent
---

# Discovery Recommendation

Use this skill when a user needs research-paper discovery or recommendation over arXiv metadata. It turns a research interest into query plans, candidate papers, ranked recommendations, feedback refinement, and follow-up filtering.

## Repository Entry Points

- Implementation: `src/daily_arxiv_agent/skills/discovery_recommendation.py`
- Public class: `daily_arxiv_agent.skills.discovery_recommendation.DiscoveryRecommendationSkill`
- Shared contracts: `daily_arxiv_agent.contracts`
- Store: `daily_arxiv_agent.storage.SQLitePaperStore`
- Technical notes: `docs/discovery-recommendation-skill.md`

## Setup

Work inside the repository's conda environment:

```bash
conda run -n daily-arxiv-agent python -m pip install -e . --no-build-isolation --no-deps
```

For deterministic local tests and demos, use fixture-backed retrieval and fake providers. Do not require live arXiv, LLM, or embedding credentials unless the user explicitly asks for a live run.

## Workflow

1. Build an optional seed preference with `build_seed_preference(...)` when the user provides arXiv IDs, arXiv URLs, or paper-title text.
2. Convert the user topic/category/date range into a bounded `QueryPlan` with `plan_query(...)`, or use `plan_query_from_seed(...)` for seed-heavy discovery.
3. Retrieve candidate papers with `retrieve_papers(...)`; prefer cache reuse unless freshness matters.
4. Rank papers with `rank_recommendations(...)` for deterministic ranking, or check `check_semantic_readiness(...)` before calling `rank_semantic_recommendations(...)`.
5. Record and apply user feedback with `record_feedback(...)` and `refine_feedback(...)`.
6. For follow-up questions, call `query_followup(...)` before starting a full new retrieval workflow.

Always inspect the returned `SkillResult.status`, `SkillResult.error`, `SkillResult.metadata`, and evidence source before presenting output. Preserve fallback and empty states instead of treating them as successful recommendations.

## Minimal Example

```python
from daily_arxiv_agent.contracts import RetrievalQuery
from daily_arxiv_agent.skills.discovery_recommendation import DiscoveryRecommendationSkill
from daily_arxiv_agent.storage import SQLitePaperStore

store = SQLitePaperStore("data/daily_arxiv.sqlite3")
skill = DiscoveryRecommendationSkill(store=store)

query = RetrievalQuery(topic="multimodal agents", category="cs.LG", max_results=10)
plan = skill.plan_query(query)
papers = skill.retrieve_papers(query, query_plan=plan.data if plan.data else None)
recommendations = skill.rank_recommendations(
    papers.data or [],
    topic=query.topic,
    query_plan=plan.data,
    retrieval_query=query,
    top_k=5,
)
```

## Validation

Run the focused facade tests after modifying this skill or its implementation:

```bash
conda run -n daily-arxiv-agent python -m pytest tests/skills/test_public_skill_facades.py -q
```
