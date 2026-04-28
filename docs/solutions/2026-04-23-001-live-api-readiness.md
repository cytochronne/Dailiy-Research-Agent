---
title: "Live API Readiness Change Note (arXiv + OpenAI LLM)"
type: note
status: active
date: 2026-04-23
related_plan: docs/plans/2026-04-21-001-feat-daily-arxiv-agent-plan.md
---

# Live API Readiness Change Note (arXiv + OpenAI LLM)

## Purpose

Record out-of-band implementation changes that enable real API execution before later units (Unit 6+), so downstream tasks can continue the staged plan without re-discovering environment and contract implications.

## What Changed

### 1) LLM provider path now supports real OpenAI API

- Added a concrete provider:
  - `src/daily_arxiv_agent/llm/openai_provider.py`
- Updated provider factory:
  - `src/daily_arxiv_agent/llm/provider.py`
- Behavior:
  - `LLM_PROVIDER=fake` -> deterministic fake provider (test-safe)
  - `LLM_PROVIDER=openai` or `LLM_PROVIDER=live` -> OpenAI-backed provider

### 2) Runtime config defaults now target live LLM by default

- Updated config defaults and env parsing:
  - `src/daily_arxiv_agent/config.py`
- Key changes:
  - `llm_provider` default: `openai`
  - `llm_model` default: `gpt-5-mini`
  - Added `llm_base_url` default: `https://api.openai.com/v1`
  - Added `llm_timeout_seconds` default: `30.0`
  - API key lookup supports both `LLM_API_KEY` and `OPENAI_API_KEY`

### 3) Env template and docs updated for real API runs

- Updated `.env.example` with real-API defaults and required variables.
- Updated `README.md` with:
  - real LLM env export examples
  - real demo command (no fixture, live arXiv + live LLM)

### 4) Test suite hardened to remain deterministic

- Updated tests to avoid accidental live dependency:
  - `tests/test_orchestrator.py` forces `LLM_PROVIDER=fake` for CLI fixture integration test.
  - `tests/test_contracts.py` updated expected config defaults and alias behavior.
  - Added `tests/test_llm_provider.py` for provider factory behavior.

## Current Runtime Semantics

### Retrieval (arXiv)

- Real arXiv retrieval already existed and remains default when CLI does not pass `--fixture`.
- Fixture mode still available for deterministic demos/tests.

### LLM

- Default runtime now expects OpenAI credentials.
- If `LLM_PROVIDER=openai` and key is missing, provider creation raises a clear configuration `ValueError`.
- Fallback behavior in extraction/briefing remains unchanged (Skill-level fallback envelopes still apply on provider/API errors).

## Downstream Impact for Unit 6/7/8

### Unit 6 (Deep Explanation)

- Reuse the same provider boundary (`LLMProvider`) instead of adding direct API calls inside Skill logic.
- Keep deep explanation tests deterministic by injecting fake provider or setting `LLM_PROVIDER=fake` in test scope.

### Unit 7 (Streamlit UI)

- UI should surface clear runtime config errors when live key is missing (do not crash page render loop).
- Preserve ability to run demo in `fake` mode for offline/classroom fallback.

### Unit 8 (Evaluation)

- Evaluation fixtures should not require live API access.
- Any live-run metrics should be clearly labeled as non-deterministic and optional.

## Operational Guidance

### Real end-to-end run

```bash
export LLM_PROVIDER=openai
export LLM_API_KEY="<your-api-key>"
export LLM_MODEL="gpt-5-mini"

daily-arxiv-agent demo \
  --topic agents \
  --category cs.LG \
  --max-results 10 \
  --top-k 5 \
  --no-cache
```

### Deterministic local/offline run

```bash
export LLM_PROVIDER=fake

daily-arxiv-agent demo \
  --fixture tests/fixtures/arxiv_atom_response.xml \
  --topic agents \
  --category cs.LG \
  --max-results 10 \
  --top-k 5 \
  --no-cache
```

## Compatibility Contract

- Contracts in `src/daily_arxiv_agent/contracts.py` are unchanged by this note.
- Orchestrator and Skill public method signatures used by Unit 5 remain unchanged.
- The change is primarily in provider selection defaults and live provider availability.

## Verification Snapshot

- Local automated suite after this change: `64 passed`.

