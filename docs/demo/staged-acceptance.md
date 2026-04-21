# Staged Acceptance Workflow

This project is developed one implementation unit at a time.

## Gate

Each unit must pass this gate before the next unit starts:

1. Implement only the current unit's scope.
2. Run the automated checks listed in the plan.
3. Produce the unit's manual acceptance artifact under `docs/demo/`.
4. Ask the user to accept or request changes.
5. Commit only after the user accepts.
6. Push only after the accepted commit exists locally.
7. Start the next unit only after the pushed commit is confirmed.

## Current Plan

Plan file: `docs/plans/2026-04-21-001-feat-daily-arxiv-agent-plan.md`

## Unit 0 Acceptance Checklist

- [x] Project installs in editable mode.
- [x] Contract tests pass.
- [x] README explains setup, tests, demo direction, and staged delivery.
- [x] README and docs specify that all Python work runs inside the `daily-arxiv-agent` conda environment.
- [x] `environment.yml` defines the reproducible Unit 0 conda environment.
- [x] `.env.example` documents local configuration without real secrets.
- [x] Shared contracts define paper metadata, provenance, evidence source, Skill status, Skill error, recommendation, and Skill result envelopes.
- [x] User accepts Unit 0 before commit and push.

## Unit 0 Verification Record

- Conda environment: `daily-arxiv-agent`
- Editable install: passed with `conda run -n daily-arxiv-agent python -m pip install -e . --no-build-isolation --no-deps`
- Tests: `8 passed` with `conda run -n daily-arxiv-agent python -m pytest`
