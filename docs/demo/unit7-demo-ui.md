# Unit 7 Demo UI Note

This file is the Unit 7 manual acceptance note for the Streamlit demo surface.

## Recommended Demo Setup

Install the optional UI dependency set:

```bash
conda run -n daily-arxiv-agent python -m pip install -e '.[ui]'
```

Use fake LLM mode when you want deterministic local LLM behavior without calling a live LLM API. This does not disable live arXiv retrieval or seed-paper metadata lookup.

```bash
export LLM_PROVIDER=fake
conda run -n daily-arxiv-agent python -m streamlit run src/daily_arxiv_agent/ui/streamlit_app.py
```

If a live provider is desired instead, keep `LLM_PROVIDER=openai` (or another OpenAI-compatible value) and export the corresponding API key before launching Streamlit.

## Demo Flow

### 1. Initial Recommendation Workflow

- Enter a topic such as `agent briefing` and category `cs.LG`.
- Optionally add seed papers one per line as arXiv IDs, arXiv URLs, or titles.
- Run the recommendation workflow.
- Confirm that the page shows:
  - a run ID and workflow metrics
  - ranked recommendations
  - a daily briefing summary table
  - workflow trace rows with evidence labels and fallback/error details when present

### 2. Feedback Refinement

- Mark at least one recommendation as `like` and one as `dislike`.
- Apply feedback refinement.
- Confirm that the refined table shows:
  - `previous_rank`
  - `rank_delta`
  - `score_delta`
  - updated rationale text tied to the feedback event

### 3. Selected-Paper Deep Explanation

- Choose one paper from the current recommendation set.
- Run one of the three explanation modes: `method`, `experiment`, or `limitations`.
- Confirm that the page shows:
  - explanation summary
  - evidence source / evidence note
  - structured fields for the selected mode
  - explanation trace visibility in the expander

### 4. Follow-up Query

- Run a follow-up topic/category/date filter.
- Confirm that the page either:
  - reuses local results and shows the follow-up trace, or
  - reports a clear fallback/error path without breaking the UI

## Runtime Behavior Worth Checking

- If the UI is switched to the environment-backed provider without an API key, the page should show a clear runtime error instead of crashing on import.
- If retrieval or explanation falls back, the trace should keep the fallback visible.
- If no recommendation rows are available, the page should render a clear empty-state message.
