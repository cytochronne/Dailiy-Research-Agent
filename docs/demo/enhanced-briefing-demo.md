# Enhanced Briefing Demo

## Run Context

- Topic: `agent briefing`
- Search mode: broad candidate pool, deterministic query planning
- LLM provider: `fake`
- Evidence boundary: metadata, abstracts, ranking scores, retrieval metadata, and candidate-pool summaries only
- PDF parsing: not used in the default briefing workflow

## CLI Commands

Raw JSON remains the automation format:

```bash
daily-arxiv-agent demo \
  --fixture tests/fixtures/arxiv_atom_response.xml \
  --topic "agent briefing" \
  --category cs.LG \
  --search-mode broad \
  --query-planner-mode deterministic \
  --candidate-pool-size 20 \
  --top-k 2 \
  --no-cache
```

Compact human-readable briefing:

```bash
daily-arxiv-agent demo \
  --fixture tests/fixtures/arxiv_atom_response.xml \
  --topic "agent briefing" \
  --category cs.LG \
  --search-mode broad \
  --query-planner-mode deterministic \
  --candidate-pool-size 20 \
  --top-k 2 \
  --format briefing \
  --no-cache
```

## Executive Summary

Top papers emphasize traceable agent briefing workflows. The output is a reading guide over Top-K papers plus bounded candidate-pool context, not a full-text survey.

## Top-K Reading Guide

The compact summary table appears first as the index.

| Rank | Paper | Score | Evidence | Key reason |
|---:|---|---:|---|---|
| 1 | Agent Workflows for Research Recommendation | 8.500 | abstract | Matched agent and briefing terms. |
| 2 | Metadata-Only Retrieval Agents | 4.000 | metadata | Matched metadata and ranking signals. |

### 1. Agent Workflows for Research Recommendation

- Summary: Agent workflows can structure daily research recommendation.
- Problem: Daily paper monitoring needs traceable recommendation context.
- Approach: The workflow stages retrieval, ranking, and briefing.
- Reading guide: Read first for the workflow shape and evidence labels.
- Evidence: abstract plus ranking context.

### 2. Metadata-Only Retrieval Agents

- Summary: Only metadata was available for this paper.
- Problem: unavailable from the default evidence scope because no abstract was available.
- Approach: unavailable from the default evidence scope because no abstract was available.
- Reading guide: Treat this as a lead until abstract or full text is checked.
- Evidence: metadata plus ranking context.

## Trend / Hotspot Overview

When enough candidates are present, the briefing reports bounded candidate-pool signals such as repeated topics, categories, and query-echo limitations.

| Signal | Type | Strength | Support | Top-K | Boundary |
|---|---|---|---:|---:|---|
| agent workflow | hotspot | moderate | 4 | 1 | candidate-pool and abstract evidence |

If the candidate pool is missing or too small, this section says `not_assessed` or `insufficient_candidate_data` and does not show an empty hotspot table.

## Top-K Comparison

- Ranking context: Rank 1 leads on abstract-backed relevance.
- Evidence coverage: Rank 1 is abstract-backed while rank 2 is metadata-limited.

## Reading Priorities

1. Start with abstract-backed workflow evidence for rank 1 because it has the strongest score and abstract support.
2. Treat metadata-only papers as leads and verify abstract or full text before making technical claims.

## Evidence Boundary

- Full text used: no
- Evidence sources: metadata, abstract, ranking, retrieval metadata, candidate pool
- Unavailable source: full text
- Explicit abstention: PDF and full-text evidence are not used in the default briefing.

## UI State Matrix

| State | Expected surface |
|---|---|
| Default success | Executive summary, Top-K compact index, detailed paper briefs, trend overview, comparisons, priorities, evidence boundary |
| Metadata-limited paper | Problem and approach fields show an evidence-limited note instead of blank claims |
| Insufficient trends | Trend section explains that the candidate pool is too small and hides empty hotspot rows |
| No trend data | Trend section reports `not_assessed` and keeps Top-K guidance visible |
| Fallback briefing | Existing warning notice appears; deterministic sections still render |
| Narrow screen | Tables remain compact and detailed paper briefs are available through labeled expanders |
