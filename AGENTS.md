# Daily arXiv Research Briefing Agent

## Agent Overview

This repository contains the Daily arXiv Research Briefing Agent, a local
Agent + Skills system for daily arXiv paper discovery, recommendation,
feedback refinement, briefing generation, and selected-paper explanation.

The Agent is designed for course/demo use with zero server infrastructure. It
can run from the command line or a local Streamlit UI, and it keeps each major
capability behind an independently testable Skill boundary.

## Agent Entry Points

- CLI command: `daily-arxiv-agent`
- Main demo workflow: `daily-arxiv-agent demo`
- Follow-up workflow: `daily-arxiv-agent followup`
- Streamlit UI: `src/daily_arxiv_agent/ui/streamlit_app.py`
- Python orchestrator: `daily_arxiv_agent.orchestrator.DailyArxivAgentOrchestrator`

## Component Skills

This Agent integrates two StudyClawHub-published Skills:

- `skills/discovery-recommendation`
  - Public class: `daily_arxiv_agent.skills.discovery_recommendation.DiscoveryRecommendationSkill`
  - Purpose: query planning, arXiv retrieval, recommendation ranking, seed-paper personalization, feedback refinement, and follow-up filtering.
- `skills/research-synthesis`
  - Public class: `daily_arxiv_agent.skills.research_synthesis.ResearchSynthesisSkill`
  - Purpose: paper extraction, daily briefing generation, evidence-boundary reporting, and selected-paper deep explanation.

## Usage

Install the project in editable mode inside the dedicated conda environment:

```bash
conda run -n daily-arxiv-agent python -m pip install -e . --no-build-isolation --no-deps
```

Run a fixture-backed or local demo through the CLI:

```bash
conda run -n daily-arxiv-agent daily-arxiv-agent demo --topic "agent briefing"
```

Run the local UI after installing the optional UI dependencies:

```bash
conda run -n daily-arxiv-agent python -m pip install -e '.[ui]' --no-build-isolation
conda run -n daily-arxiv-agent streamlit run src/daily_arxiv_agent/ui/streamlit_app.py
```

## Evidence and Safety Rules

- Keep arXiv metadata, abstracts, ranking evidence, candidate-pool context, and full-text evidence clearly separated.
- Do not present metadata-only or abstract-only output as full-paper analysis.
- Use fake providers and fixtures for deterministic demos and tests unless live API access is explicitly requested.
- Keep API keys and private credentials out of git. Use `.env` or shell environment variables for local secrets.
- Preserve `SkillResult` fallback, empty, and error states in user-facing output.

## Development Environment

- Always use the dedicated conda environment `daily-arxiv-agent` for this repository.
- Run Python commands with `conda run -n daily-arxiv-agent ...` or activate the environment first with `conda activate daily-arxiv-agent`.
- Do not run project development, dependency installation, or tests from conda `base` unless the user explicitly asks for it.
- Expected interpreter: `/home/cytochrome/pan1/miniconda3/envs/daily-arxiv-agent/bin/python`.
- Standard test command: `conda run -n daily-arxiv-agent python -m pytest -q`.
