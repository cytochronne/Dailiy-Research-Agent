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

## Before / After Briefing Example

MVP-style briefing:

> Two papers were found for agent briefing. The top result is relevant to the topic and should be read first.

Why this is insufficient:

| Gap | Impact |
|-----|--------|
| no paper-level problem or approach fields | user cannot tell what each paper contributes |
| no Top-K comparison | ranks look interchangeable apart from score |
| no trend status | candidate-pool context is invisible |
| no evidence boundary | output can be mistaken for a full-text survey |

Enhanced briefing:

| Section | Example output |
|---------|----------------|
| Top-K brief | Rank 1 explains agent workflow evidence; rank 2 is explicitly metadata-limited |
| Comparison | Rank 1 is abstract-backed while rank 2 needs follow-up before technical claims |
| Trend status | `available` when candidate-pool signals are supported; `not_assessed` when no candidate pool was supplied |
| Reading priority | start with the abstract-backed workflow paper, then treat metadata-only papers as leads |
| Evidence boundary | metadata, abstract, ranking, retrieval metadata, and candidate-pool evidence only; no PDF/full-text evidence |

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

## Quality Evaluation Hook

The enhanced briefing can be checked offline with:

```python
from daily_arxiv_agent.evaluation.metrics import evaluate_briefing_quality

result = evaluate_briefing_quality(
    briefing,
    expected_top_k_paper_ids=["2604.20001", "2604.20002", "2604.20003"],
)
```

Expected passing signals:

| Metric | Expected value |
|--------|----------------|
| `quality_passed` | `True` |
| `top_k_coverage` | `1.0` when all expected Top-K IDs have detailed items |
| `trend_status` | explicit status, including `not_assessed` when candidate-pool data is absent |
| `reading_priority_present` | `True` |
| `evidence_boundary_present` | `True` |
| `claim_support_coverage` | high coverage from abstract, metadata, ranking, or candidate-pool sources |
| `forbidden_evidence_claims` | empty in default mode |

Failure examples:

| Failure | Reported field |
|---------|----------------|
| structurally complete but generic text | `generic_claim_locations`, `claim_specificity_score`, `claim_specificity_low` |
| missing reading guidance | `missing_sections`, `reading_priorities_missing` |
| no evidence boundary | `missing_sections`, `evidence_boundary_missing` |
| default briefing says full text was used | `forbidden_evidence_claims`, `default_mode_full_text_used` |

The quality hook uses local `DailyBriefing` data and fixture-backed papers only. It does not require live arXiv, live LLM calls, or PDF parsing.

## UI State Matrix

| State | Expected surface |
|---|---|
| Default success | Executive summary, Top-K compact index, detailed paper briefs, trend overview, comparisons, priorities, evidence boundary |
| Metadata-limited paper | Problem and approach fields show an evidence-limited note instead of blank claims |
| Insufficient trends | Trend section explains that the candidate pool is too small and hides empty hotspot rows |
| No trend data | Trend section reports `not_assessed` and keeps Top-K guidance visible |
| Fallback briefing | Existing warning notice appears; deterministic sections still render |
| Narrow screen | Tables remain compact and detailed paper briefs are available through labeled expanders |
