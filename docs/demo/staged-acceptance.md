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

## Unit 1 Acceptance Checklist

- [x] Fixture-backed Atom parsing returns normalized paper metadata.
- [x] Retrieval query construction supports topic, category, date range, pagination, and submitted-date sorting.
- [x] SQLite storage persists paper metadata and retrieval result sets.
- [x] Follow-up filtering can reuse stored papers by topic, category, and date range.
- [x] Empty arXiv responses return a successful empty result.
- [x] Network/API failures return structured fallback results with failed query metadata.
- [x] Malformed Atom responses return structured fallback results without corrupting storage.
- [x] Manual acceptance artifact exists at `docs/demo/unit1-sample-retrieval.md`.
- [x] User accepts Unit 1 before commit and push.

## Unit 1 Verification Record

- Conda environment: `daily-arxiv-agent`
- Tests: `18 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Manual artifact: fixture-backed sample retrieval for topic `agents`, category `cs.LG`, submitted-date range `2026-04-18` to `2026-04-21`

## Unit 2 Acceptance Checklist

- [x] Deterministic keyword ranking ranks title/abstract matches above unrelated papers.
- [x] Top-K recommendations include rank, score, rationale, evidence source, and paper provenance.
- [x] Fewer papers than Top-K returns all available papers without error.
- [x] Missing abstracts use metadata evidence labels and avoid fabricated method details.
- [x] Structured extraction runs behind the LLM provider boundary with a deterministic fake provider.
- [x] LLM extraction failures return fallback extraction output with a clear error.
- [x] Daily briefing generation includes an executive summary, summary table, highlighted paper, and all ranked paper references.
- [x] Briefing-level LLM summary failures return fallback briefing output with a clear error.
- [x] Extraction fallback status propagates to the daily briefing result.
- [x] Manual acceptance artifact exists at `docs/demo/unit2-daily-briefing.md`.
- [x] User requested code review, commit, and push for Unit 2.

## Unit 2 Verification Record

- Conda environment: `daily-arxiv-agent`
- Tests: `29 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Review: `/ce-code-review` found one briefing fallback propagation issue; fixed before commit.
- Commit: `10ad446 feat(ranking): add topic briefing MVP`
- Manual artifact: generated daily briefing MVP for topic `agent briefing` using deterministic keyword ranking and the fake LLM provider.
