# Final Course Demo Script

This script is the Unit 8 manual acceptance artifact for presenting the complete Daily arXiv Research Briefing Agent.

## Demo Goal

Show one local workflow that connects the Agent and Skills from retrieval through recommendation, feedback refinement, selected-paper explanation, and lightweight evaluation.

Use fixture-backed or fake-LLM paths when the presentation must be deterministic. Live arXiv and live LLM runs are optional extensions, not requirements for the course demo.

## Preconditions

```bash
conda activate daily-arxiv-agent
python -m pip install -e '.[ui]' --no-build-isolation
export LLM_PROVIDER=fake
```

If using the CLI without Streamlit, the editable install from Unit 0 is sufficient:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

## 1. Open With the Agent Workflow

Run the deterministic CLI workflow:

```bash
DEMO_DB="$(mktemp -t daily-arxiv-unit8-demo.XXXXXX.sqlite3)"
daily-arxiv-agent demo \
  --fixture tests/fixtures/arxiv_atom_response.xml \
  --db-path "$DEMO_DB" \
  --topic agents \
  --category cs.LG \
  --max-results 5 \
  --top-k 2 \
  --no-cache
```

Call out these points:

- The Agent calls retrieval, ranking, extraction, and briefing in order.
- The workflow trace keeps each Skill visible.
- Evidence labels distinguish metadata, abstract, and full-text-backed outputs.
- Fixture mode proves the workflow without depending on live network access.
- The temporary DB keeps saved seed preferences and feedback from changing the fixture result.

Expected ranked result:

| Rank | Paper ID | Title |
|------|----------|-------|
| 1 | `2604.00001` | Explainable Agents for Daily Research Briefings |
| 2 | `2604.00002` | Retrieval-Augmented Topic Tracking |

## 2. Show the Streamlit Demo Surface

```bash
python -m streamlit run src/daily_arxiv_agent/ui/streamlit_app.py
```

Use these UI inputs:

| Field | Demo value |
|-------|------------|
| Topic | `agent briefing` |
| Category | `cs.LG` |
| Top K | `5` |
| Seed papers | `Explainable Agents for Daily Research Briefings` |

Walk through:

1. Run recommendations.
2. Point to the run ID and workflow trace.
3. Show ranked recommendations and the daily briefing table.
4. Open fallback/error details if any trace row reports them.

## 3. Demonstrate Feedback Refinement

In the UI, mark one recommendation as liked and another as disliked, then apply feedback refinement.

Call out:

- Feedback is stored as structured paper-level events.
- Refined recommendations expose `previous_rank`, `rank_delta`, and `score_delta`.
- The ranking rationale explains how feedback moved related papers.

## 4. Demonstrate Deep Explanation

Select paper `2604.00001` and run all three explanation modes:

- `method`
- `experiment`
- `limitations`

Call out:

- Full-text evidence is preferred when supplied or cached.
- Abstract-only and metadata-only fallback paths are explicitly labeled.
- Mode-specific fields prevent a single generic summary from hiding missing evidence.

## 5. Show Evaluation Artifacts

Open `docs/demo/evaluation-summary.md`.

Use it to explain the Unit 8 evaluation scope:

- Recommendation overlap against expected relevant paper IDs.
- Before/after feedback rank and score movement.
- Explanation completeness by required mode-specific sections.
- Structured validation errors for malformed evaluation fixtures.

Then show the automated check:

```bash
conda run -n daily-arxiv-agent python -m pytest tests/test_evaluation.py
```

## 6. Course Deliverable Mapping

| Course deliverable theme | Project evidence |
|--------------------------|------------------|
| Agent orchestration | `DailyArxivAgentOrchestrator` runs retrieval, ranking, briefing, feedback, follow-up, and explanation workflows with trace visibility. |
| Skill decomposition | Retrieval, seed parsing, ranking, extraction, briefing, feedback refinement, follow-up, deep explanation, and evaluation are separate modules with test coverage. |
| Human feedback loop | Unit 4 feedback events and Unit 8 movement metrics show before/after recommendation changes. |
| Explainability | Evidence labels, workflow traces, rationales, and selected-paper explanation modes make outputs inspectable. |
| Deterministic evaluation | Unit 8 metrics avoid live API dependency and provide repeatable checks for final reporting. |

## Close

End by showing:

- `docs/demo/staged-acceptance.md` for unit-by-unit verification.
- `README.md` for setup and run instructions.
- `docs/demo/evaluation-summary.md` for final reporting support.
