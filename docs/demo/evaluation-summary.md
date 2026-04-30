# Unit 8 Evaluation Summary

This file is the Unit 8 manual acceptance artifact for lightweight evaluation and final reporting.

## Evaluation Scope

Unit 8 adds deterministic helpers in `daily_arxiv_agent.evaluation.metrics`. The goal is not to create a benchmark suite; it is to make the final demo auditable with small, repeatable checks.

| Evaluation area | Helper | What it reports |
|-----------------|--------|-----------------|
| Recommendation overlap | `evaluate_recommendations(...)` | matched expected IDs, missing expected IDs, `precision_at_k`, `recall_at_k`, and mean reciprocal rank |
| Fixture-backed recommendation evaluation | `evaluate_recommendation_fixture(...)` | validates a dictionary fixture and returns the same overlap metrics or a structured validation error |
| Feedback movement | `evaluate_feedback_movement(...)` | moved-up, moved-down, unchanged, new, and removed paper IDs with rank and score deltas |
| Explanation completeness | `check_explanation_completeness(...)` | present and missing required sections for method, experiment, and limitations explanations |

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
```

Result during implementation: `9 passed`.

Full project check:

```bash
conda run -n daily-arxiv-agent python -m pytest
```

Result during implementation: `108 passed`.

## Reporting Notes

- The final demo can run entirely from local fixtures and fake LLM behavior.
- Live arXiv and live LLM calls are supported but should be labeled non-deterministic in the final report.
- Recommendation quality is evaluated with transparent expected-ID overlap, not hidden human preference claims.
- Feedback value is shown through observable rank/score movement.
- Explanation quality is framed as section completeness plus evidence labeling, not free-form subjective grading.
