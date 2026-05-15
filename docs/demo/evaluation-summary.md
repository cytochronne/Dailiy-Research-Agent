# Unit 8 Evaluation Summary

This file is the Unit 8 manual acceptance artifact for lightweight evaluation and final reporting.

## Evaluation Scope

Unit 8 adds deterministic helpers in `daily_arxiv_agent.evaluation.metrics`. The goal is not to create a benchmark suite; it is to make the final demo auditable with small, repeatable checks.

| Evaluation area | Helper | What it reports |
|-----------------|--------|-----------------|
| Recommendation overlap | `evaluate_recommendations(...)` | matched expected IDs, missing expected IDs, `precision_at_k`, `recall_at_k`, and mean reciprocal rank |
| Fixture-backed recommendation evaluation | `evaluate_recommendation_fixture(...)` | validates a dictionary fixture and returns the same overlap metrics or a structured validation error |
| Search-quality gate | `evaluate_search_quality(...)` | candidate count, relevant candidate coverage, Top-K IDs, precision/recall, rationale coverage, and retrieval budget status |
| Feedback movement | `evaluate_feedback_movement(...)` | moved-up, moved-down, unchanged, new, and removed paper IDs with rank and score deltas |
| Explanation completeness | `check_explanation_completeness(...)` | present and missing required sections for method, experiment, and limitations explanations |
| Enhanced briefing quality | `evaluate_briefing_quality(...)` | required enhanced sections, Top-K brief coverage, trend status, reading priorities, evidence boundary, claim specificity, claim-support coverage, and forbidden full-text/PDF claims |

## Frozen Real-arXiv Ranking Evaluation

The repository includes a small frozen real-arXiv evaluation in `data/evaluation/`.
It is separate from the deterministic workflow fixtures: the fixture tests verify
contract behavior, while this frozen set gives a small external relevance check.
The candidate snapshot contains three topics, 50 candidates per topic, and one
binary human relevance label per candidate.

Run it with:

```bash
conda run -n daily-arxiv-agent daily-arxiv-agent real-eval run --format markdown
```

The live semantic comparison uses the configured embedding provider:

```bash
conda run -n daily-arxiv-agent daily-arxiv-agent real-eval run --semantic-provider live --format markdown
```

Current frozen-snapshot metrics with live `embedding-3` semantic ranking:

| Topic | Method | Precision@5 | Recall@10 | MRR | Relevant |
|-------|--------|-------------|-----------|-----|----------|
| llm_tool_agents | agent | 0.400 | 0.667 | 0.333 | 6 |
| llm_tool_agents | semantic_agent | 0.600 | 0.500 | 1.000 | 6 |
| llm_tool_agents | strict_keyword | 0.400 | 0.500 | 1.000 | 6 |
| llm_tool_agents | bm25 | 0.400 | 0.833 | 1.000 | 6 |
| retrieval_augmented_generation | agent | 1.000 | 1.000 | 1.000 | 6 |
| retrieval_augmented_generation | semantic_agent | 0.800 | 0.667 | 1.000 | 6 |
| retrieval_augmented_generation | strict_keyword | 1.000 | 1.000 | 1.000 | 6 |
| retrieval_augmented_generation | bm25 | 1.000 | 1.000 | 1.000 | 6 |
| vision_language_robotics | agent | 0.800 | 0.833 | 1.000 | 6 |
| vision_language_robotics | semantic_agent | 0.800 | 1.000 | 1.000 | 6 |
| vision_language_robotics | strict_keyword | 0.800 | 0.667 | 1.000 | 6 |
| vision_language_robotics | bm25 | 1.000 | 1.000 | 1.000 | 6 |
| macro_average | agent | 0.733 | 0.833 | 0.778 | 18 |
| macro_average | semantic_agent | 0.733 | 0.722 | 1.000 | 18 |
| macro_average | strict_keyword | 0.733 | 0.722 | 1.000 | 18 |
| macro_average | bm25 | 0.800 | 0.944 | 1.000 | 18 |

Candidate refresh and labeling are manual steps, not part of CI:

```bash
conda run -n daily-arxiv-agent daily-arxiv-agent real-eval fetch-candidates
conda run -n daily-arxiv-agent daily-arxiv-agent real-eval label-template
```

## Deterministic Demo Fixtures

The fixture-backed recommendation demo uses `tests/fixtures/arxiv_atom_response.xml`.

Expected result for topic `agents`, category `cs.LG`, and top-k `2`:

| Rank | Paper ID | Expected role |
|------|----------|---------------|
| 1 | `2604.00001` | relevant agent briefing paper |
| 2 | `2604.00002` | secondary topic-tracking paper |

A simple overlap fixture can mark `2604.00001` as the expected relevant paper and evaluate the top result:

```python
from daily_arxiv_agent.evaluation.metrics import evaluate_recommendation_fixture

result = evaluate_recommendation_fixture(
    {
        "recommendations": [
            {"paper_id": "2604.00001", "rank": 1, "score": 9.0},
            {"paper_id": "2604.00002", "rank": 2, "score": 0.0},
        ],
        "expected_relevant_paper_ids": ["2604.00001"],
        "k": 1,
    }
)
```

Expected metrics:

| Metric | Value |
|--------|-------|
| matched paper IDs | `["2604.00001"]` |
| missing relevant IDs | `[]` |
| precision@1 | `1.0` |
| recall@1 | `1.0` |
| mean reciprocal rank | `1.0` |

## Hybrid Search Quality Gate

Unit 7 adds `tests/fixtures/arxiv_search_quality_response.xml` to exercise the improved search pipeline offline. The fixture contains:

| Paper ID | Fixture role |
|----------|--------------|
| `2604.20001` | exact multi-keyword topic match |
| `2604.20002` | related language-model / embodied-control match |
| `2604.20003` | related planning-agent manipulation match |
| `2604.20004` | recent weak candidate |
| `2604.20005` | category-matched weak candidate |
| `2604.20006` | unrelated noisy candidate |

For topic `multimodal llm agents for robotic manipulation`, strict phrase retrieval only covers `2604.20001`. Broad planned retrieval covers all three expected relevant papers, and hybrid ranking places `2604.20001`, `2604.20002`, and `2604.20003` in the top three while leaving the unrelated noisy candidate out of Top-K.

Expected search-quality metrics for the broad fixture run:

| Metric | Value |
|--------|-------|
| relevant candidate coverage | `1.0` |
| Top-K paper IDs | `["2604.20001", "2604.20002", "2604.20003"]` |
| precision@3 | `1.0` |
| recall@3 | `1.0` |
| rationale coverage | `1.0` |
| budget exhausted | `true`, because the default four-request budget is fully used before reaching the large candidate target |

The same tests verify that a syntactically valid but semantically divergent LLM query plan falls back to deterministic planning before retrieval, so fixture quality cannot be degraded by an unrelated planner output that fails the required-term guard.

## Enhanced Briefing Quality Gate

Unit 6 adds a deterministic briefing-quality evaluator for the richer daily briefing shape. The helper accepts a `DailyBriefing` object or serialized dictionary and checks the enhanced output without live arXiv, live LLM calls, or PDF parsing.

Required enhanced sections:

| Section | Check |
|---------|-------|
| executive summary | non-empty briefing summary |
| summary table | compact Top-K index exists |
| Top-K items | expected Top-K paper IDs have detailed brief items |
| trend overview | explicit status such as `available`, `not_assessed`, or `insufficient_candidate_data` |
| Top-K comparisons | at least one evidence-bounded comparison note |
| reading priorities | goal-aware reading guidance exists |
| evidence boundary | structured sources, unavailable sources, notes, or abstentions exist |

Semantic checks:

| Metric | Passing behavior |
|--------|------------------|
| Top-K coverage | every expected Top-K ID has a detailed briefing item |
| trend signal coverage | `available` trends include supported candidate-pool signals; `not_assessed` is accepted when no candidate pool is present |
| claim-support coverage | textual claims are supported by allowed evidence sources or are explicit abstentions |
| claim specificity | briefing text includes paper-specific differences, problem/approach/contribution claims, or goal-specific reading priorities |
| evidence boundary | default briefing must not mark full text as used or imply PDF/full-text evidence |

Generic or vacuous text fails even when all sections exist. For example, repeated claims such as "This paper is useful" or "Rank 1 and rank 2 are both useful papers" are reported in `generic_claim_locations` and lower `claim_specificity_score`.

Default-mode full-text/PDF violations are reported in `forbidden_evidence_claims`. Examples include `evidence_boundary.full_text_used`, `evidence_boundary.evidence_sources`, or generated text such as "Full-text evidence shows...".

Fixture-backed Unit 6 tests reuse `tests/fixtures/arxiv_search_quality_response.xml` so the quality gate covers exact, related, weak, and unrelated candidates offline:

| Fixture role | Paper ID |
|--------------|----------|
| exact Top-K match | `2604.20001` |
| related Top-K match | `2604.20002` |
| related Top-K match | `2604.20003` |
| weak candidate | `2604.20004`, `2604.20005` |
| unrelated noisy candidate | `2604.20006` |

## Feedback Movement Check

Feedback evaluation compares before/after recommendation rows rather than judging subjective relevance.

Example interpretation:

| Paper ID | Before rank | After rank | Movement |
|----------|-------------|------------|----------|
| `2604.00002` | 2 | 1 | moved up |
| `2604.00001` | 1 | 2 | moved down |
| `2604.00003` | 3 | 3 | unchanged |

The helper also preserves which papers were explicitly liked or disliked, so the report can explain whether a movement came after user feedback.

## Explanation Completeness Check

The completeness helper checks that mode-specific explanation objects contain the sections expected by the final demo:

| Mode | Required section examples |
|------|---------------------------|
| `method` | problem, method overview, core workflow, inputs/outputs, innovation |
| `experiment` | datasets, baselines, metrics, experimental setup, conclusions |
| `limitations` | stated limitations, assumptions, missing validation, risks |

Missing sections are reported explicitly. This is useful for abstract-only fallback runs, where the system should label missing experiment details instead of inventing them.

## Validation Behavior

Malformed evaluation fixtures return a structured `SkillResult` error instead of raising through the demo:

| Field | Expected value |
|-------|----------------|
| status | `error` |
| error code | `evaluation_fixture_invalid` |
| retryable | `false` |

This matches the rest of the project contract: invalid or incomplete inputs remain inspectable by the Agent/UI layer.

## Verification

Targeted Unit 8 check:

```bash
conda run -n daily-arxiv-agent python -m pytest tests/test_evaluation.py
conda run -n daily-arxiv-agent python -m pytest tests/test_real_arxiv_evaluation.py
```

Historical Unit 8 implementation result: `9 passed`.

Targeted Unit 7 search-quality check:

```bash
conda run -n daily-arxiv-agent python -m pytest tests/test_evaluation.py
```

Result during Unit 7 implementation: `13 passed`.

Targeted Unit 6 enhanced-briefing-quality check:

```bash
conda run -n daily-arxiv-agent python -m pytest -q tests/test_evaluation.py
```

Result during Unit 6 implementation: `20 passed`.

Full project check:

```bash
conda run -n daily-arxiv-agent python -m pytest -q
```

Result after Unit 6 implementation: `195 passed`.

## Reporting Notes

- The final demo can run entirely from local fixtures and fake LLM behavior.
- Live arXiv and live LLM calls are supported but should be labeled non-deterministic in the final report.
- Recommendation quality is evaluated with transparent expected-ID overlap, not hidden human preference claims.
- Feedback value is shown through observable rank/score movement.
- Explanation quality is framed as section completeness plus evidence labeling, not free-form subjective grading.
- Enhanced briefing quality is framed as structural completeness, evidence-bounded support, specificity, and forbidden-claim detection.
