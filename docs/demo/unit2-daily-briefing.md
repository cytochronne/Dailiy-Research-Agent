# Unit 2 Daily Briefing MVP

## Run Context

- Topic: `agent briefing`
- Ranking mode: deterministic explicit keyword ranking
- LLM provider: `fake`
- Evidence boundary: metadata and abstract only

## Executive Summary

2 ranked paper(s) were reviewed for `agent briefing`. The top paper is `Explainable Agents for Daily Research Briefings` with evidence from `abstract`.

## Summary Table

| Rank | Paper | Score | Evidence | Key reason |
|---:|---|---:|---|---|
| 1 | [Explainable Agents for Daily Research Briefings](https://arxiv.org/abs/2604.00001) | 8.0 | abstract | Matched explicit terms: agent, briefing. Evidence: abstract. |
| 2 | [Daily Research Recommendation Workflows](https://arxiv.org/abs/2604.00002) | 1.0 | abstract | Matched explicit terms: briefing. Evidence: abstract. |

## Highlighted Paper

**Explainable Agents for Daily Research Briefings**

- Paper ID: `2604.00001`
- Evidence source: abstract
- Summary: We propose an agent workflow for daily research briefings.
- Contributions: Connects the paper's abstract evidence to the requested topic: agent briefing.
- Methods: Abstract mentions agent; abstract mentions workflow.
- Relevance rationale: Matched explicit terms: agent, briefing. Evidence: abstract.

## Ranked Briefing Items

### 1. Explainable Agents for Daily Research Briefings

- arXiv: https://arxiv.org/abs/2604.00001
- Evidence source: abstract
- Score: 8.0
- Summary: We propose an agent workflow for daily research briefings.
- Relevance rationale: Matched explicit terms: agent, briefing. Evidence: abstract.

### 2. Daily Research Recommendation Workflows

- arXiv: https://arxiv.org/abs/2604.00002
- Evidence source: abstract
- Score: 1.0
- Summary: We study daily briefing recommendation workflows for research paper discovery.
- Relevance rationale: Matched explicit terms: briefing. Evidence: abstract.

## Acceptance Question

Is keyword ranking and briefing output good enough for the MVP?

## Enhanced Surface Update

The MVP shape remains valid for backward compatibility: executive summary, compact summary table, highlighted paper, and ranked briefing items are still present.

The current enhanced briefing surface adds sections after the MVP fields:

1. Top-K reading guide, with the compact summary table as the index.
2. Candidate-pool trend or hotspot overview when enough candidate evidence is available.
3. Top-K comparison notes.
4. Goal-aware reading priorities.
5. Evidence boundary and explicit abstentions.

The default briefing is still abstract, metadata, ranking, and retrieval-metadata only. It does not parse PDFs or make full-text claims. See `docs/demo/enhanced-briefing-demo.md` for the current compact CLI and UI-oriented example.
