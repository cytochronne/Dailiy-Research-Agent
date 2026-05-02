# Improved Search Demo

This file is the Unit 7 manual acceptance artifact for the hybrid arXiv search quality gate.

## Offline Scenario

Fixture: `tests/fixtures/arxiv_search_quality_response.xml`

Topic:

```text
multimodal llm agents for robotic manipulation
```

Expected relevant papers:

| Paper ID | Why it is relevant |
|----------|--------------------|
| `2604.20001` | exact title and abstract match for the full topic |
| `2604.20002` | related language-model and embodied-control match |
| `2604.20003` | related planning-agent and manipulation match |

Weak and noisy candidates are also present in the fixture:

| Paper ID | Fixture role |
|----------|--------------|
| `2604.20004` | recent weak candidate |
| `2604.20005` | category-matched weak candidate |
| `2604.20006` | unrelated noisy candidate |

## Before And After

Strict phrase retrieval only finds the exact match:

| Search mode | Relevant candidates covered |
|-------------|-----------------------------|
| strict | `2604.20001` |
| broad planned search | `2604.20001`, `2604.20002`, `2604.20003` |

Hybrid ranking then keeps the evidence-bearing papers above weak or unrelated candidates:

| Rank | Paper ID | Role |
|------|----------|------|
| 1 | `2604.20001` | exact match |
| 2 | `2604.20002` | related language-model / embodied-control match |
| 3 | `2604.20003` | related planning-agent manipulation match |

The unrelated fixture paper `2604.20006` is retrieved as noise but does not enter Top-K when enough evidence-bearing candidates exist.

## Quality Gate Metrics

The Unit 7 test evaluates:

| Metric | Expected value |
|--------|----------------|
| relevant candidate coverage | `1.0` |
| Top-K IDs | `["2604.20001", "2604.20002", "2604.20003"]` |
| precision@3 | `1.0` |
| recall@3 | `1.0` |
| rationale coverage | `1.0` |
| budget exhausted | `true` with the default four-request budget and a large candidate target |

The budget-exhausted flag is expected in this offline scenario: the fixture intentionally uses a candidate target larger than the number of unique papers returned by four planned requests.

## Planner Guard

The same offline test includes a syntactically valid but semantically divergent LLM plan for compiler register allocation. The query-planning guard rejects it because it does not preserve enough deterministic required terms from the user topic, then falls back to the deterministic plan before retrieval. This keeps a bad planner output from degrading fixture coverage.

## Verification

```bash
conda run -n daily-arxiv-agent python -m pytest tests/test_evaluation.py
```

Unit 7 implementation result: `13 passed`.
