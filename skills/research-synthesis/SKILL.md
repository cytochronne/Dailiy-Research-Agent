---
name: research-synthesis
description: "Turn ranked arXiv recommendations into evidence-bounded briefings and selected-paper explanations."
author: cytochronne
version: 0.1.0
tags:
  - arxiv
  - synthesis
  - briefing
  - explanation
  - research-agent
---

# Research Synthesis

Use this skill when a user already has selected or ranked research papers and needs readable research output: per-paper briefing items, daily briefing summaries, evidence boundaries, or selected-paper deep explanations.

## Repository Entry Points

- Implementation: `src/daily_arxiv_agent/skills/research_synthesis.py`
- Public class: `daily_arxiv_agent.skills.research_synthesis.ResearchSynthesisSkill`
- Shared contracts: `daily_arxiv_agent.contracts`
- LLM provider boundary: `daily_arxiv_agent.llm.base.LLMProvider`
- Technical notes: `docs/research-synthesis-skill.md`

## Setup

Work inside the repository's conda environment:

```bash
conda run -n daily-arxiv-agent python -m pip install -e . --no-build-isolation --no-deps
```

For deterministic local tests and demos, use `daily_arxiv_agent.llm.fake.FakeLLMProvider`. Live LLM providers require explicit credentials in the environment.

## Workflow

1. Start from ranked `Recommendation` objects produced by the discovery/recommendation workflow.
2. Convert each recommendation into a structured `PaperBriefingItem` with `extract_paper(...)`.
3. Combine recommendations, extraction results, candidate-pool context, query metadata, and ranking metadata with `generate_briefing(...)`.
4. For a selected paper, call `explain_paper(...)` with an `ExplanationMode` of `method`, `experiment`, or `limitations`.
5. Clearly report whether output is backed by metadata, abstract text, ranking evidence, candidate-pool context, or full text.

Always inspect the returned `SkillResult.status`, `SkillResult.error`, and evidence source. If a provider fails, preserve deterministic fallback output and make the evidence boundary visible.

## Minimal Example

```python
from daily_arxiv_agent.llm.fake import FakeLLMProvider
from daily_arxiv_agent.skills.research_synthesis import ResearchSynthesisSkill

skill = ResearchSynthesisSkill(provider=FakeLLMProvider())

extraction_results = [
    skill.extract_paper(recommendation, topic="multimodal agents")
    for recommendation in recommendations
]
briefing = skill.generate_briefing(
    topic="multimodal agents",
    recommendations=recommendations,
    extraction_results=extraction_results,
    candidate_papers=[recommendation.paper for recommendation in recommendations],
)
```

## Validation

Run the focused facade tests after modifying this skill or its implementation:

```bash
conda run -n daily-arxiv-agent python -m pytest tests/skills/test_public_skill_facades.py -q
```
