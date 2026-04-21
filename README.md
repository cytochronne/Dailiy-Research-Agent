# Daily arXiv Research Briefing Agent

Local Agent + Skills project for daily arXiv paper monitoring, recommendation, feedback refinement, briefing generation, and selected-paper explanation.

## Current Status

This repository is being built in staged units from `docs/plans/2026-04-21-001-feat-daily-arxiv-agent-plan.md`.

Current stage: Unit 0, project scaffold and delivery workflow.

## Setup

Use the dedicated conda environment for all Python work:

```bash
conda activate daily-arxiv-agent
```

If the environment does not exist yet, create it with the Unit 0 dependencies:

```bash
conda env create -f environment.yml
conda activate daily-arxiv-agent
python -m pip install -e . --no-build-isolation --no-deps
```

All future Python packages should be installed into this conda environment. Prefer conda packages when available; use pip only inside the active conda environment when a package is not available through conda.

Copy `.env.example` to `.env` for local settings. Keep real secrets out of git.

## Test

```bash
conda run -n daily-arxiv-agent python -m pytest
```

## Planned Demo

The final local demo will show:

- Agent workflow trace
- arXiv retrieval results
- ranked recommendations
- daily briefing
- like/dislike refinement
- selected-paper deep explanation modes

## Staged Delivery Rule

Each implementation unit is developed and verified independently. After a unit is implemented:

1. Run the unit's automated checks.
2. Produce its manual acceptance artifact under `docs/demo/`.
3. Ask for user acceptance.
4. Commit and push only after acceptance.
5. Start the next unit only after the pushed commit is confirmed.

See `docs/demo/staged-acceptance.md` for the full acceptance checklist.
