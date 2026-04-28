# Unit 5 Agent Orchestrator and Follow-up Demo

Manual acceptance artifact for Unit 5: Agent Orchestrator and Follow-up Queries.

## Scenario

The Agent runs a fixture-backed recommendation workflow for topic `agents` and category `cs.LG`, then runs a follow-up query against the stored papers.

Command used for the recommendation workflow:

```bash
PYTHONPATH=src python -m daily_arxiv_agent.cli demo \
  --fixture tests/fixtures/arxiv_atom_response.xml \
  --db-path /tmp/unit5-demo.sqlite3 \
  --topic agents \
  --category cs.LG \
  --max-results 5 \
  --top-k 2 \
  --no-cache
```

## Recommendation Workflow Trace

| Step | Skill | Status | Evidence | Input Summary | Output Summary |
|------|-------|--------|----------|---------------|----------------|
| 1 | `arxiv_retrieval` | `success` | `metadata` | topic `agents`, category `cs.LG`, max results `5` | 2 papers retrieved |
| 2 | `ranking` | `success` | `abstract` | topic `agents`, no seed, 0 feedback events, top-k `2` | 2 recommendations ranked |
| 3 | `extraction` | `success` | `abstract` | 2 recommendations | 2 briefing items extracted |
| 4 | `briefing` | `success` | `abstract` | topic `agents`, 2 extracted items | briefing generated |

## Recommendation Output

| Rank | Paper ID | Score | Evidence | Title |
|------|----------|-------|----------|-------|
| 1 | `2604.00001` | 9.0000 | `abstract` | Explainable Agents for Daily Research Briefings |
| 2 | `2604.00002` | 0.0000 | `abstract` | Retrieval-Augmented Topic Tracking |

Executive summary:

```text
2 ranked paper(s) were reviewed for 'agents'. The top paper is 'Explainable Agents for Daily Research Briefings' with evidence from abstract.
```

## Follow-up Query

Command used after the recommendation workflow populated the local SQLite store:

```bash
PYTHONPATH=src python -m daily_arxiv_agent.cli followup \
  --db-path /tmp/unit5-demo.sqlite3 \
  --topic 'agent workflow' \
  --category cs.LG \
  --start-date 2026-04-19 \
  --end-date 2026-04-21 \
  --max-results 5 \
  --top-k 2
```

Follow-up trace:

| Step | Skill | Status | Evidence | Local Hit | Fetch Attempted | Output Summary |
|------|-------|--------|----------|-----------|-----------------|----------------|
| 1 | `followup_filter` | `success` | `abstract` | `true` | `false` | 1 paper matched |
| 2 | `ranking` | `success` | `abstract` | n/a | n/a | 1 follow-up recommendation |

Follow-up result:

| Rank | Paper ID | Score | Title |
|------|----------|-------|-------|
| 1 | `2604.00001` | 7.0000 | Explainable Agents for Daily Research Briefings |

## Feedback Refinement Workflow

The orchestrator exposes `run_feedback_refinement(...)`, which delegates to `FeedbackRefinementSkill` and records a trace step named `feedback_refinement`. Automated coverage verifies that a like event for an agent-workflow anchor moves a similar recommendation above an unrelated compiler paper and persists the event in SQLite.

## Fallback Visibility

Automated coverage injects a failing retrieval Skill. The workflow returns top-level `fallback` status and preserves a first trace step with:

```text
skill=arxiv_retrieval
status=error
fallback=true
error_code=retrieval_skill_failed
```

This keeps the user-facing workflow inspectable even when one Skill fails.

## Verification

```bash
conda run -n daily-arxiv-agent python -m pytest
```

Result: `64 passed`.

## Acceptance Notes

- `DailyArxivAgentOrchestrator.run_recommendation(...)` is the shared entry point for retrieval, ranking, extraction, and briefing.
- `DailyArxivAgentOrchestrator.run_feedback_refinement(...)` records feedback and returns updated recommendations with trace data.
- `DailyArxivAgentOrchestrator.run_followup_query(...)` filters local stored papers first and only calls retrieval when no local paper matches.
- `daily-arxiv-agent demo` and `daily-arxiv-agent followup` provide non-UI verification paths through the same orchestrator methods.
