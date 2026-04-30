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

## Unit 3 Acceptance Checklist

- [x] Seed parsing accepts arXiv ID, arXiv URL, and title-only seed input.
- [x] arXiv ID/URL seeds normalize into one canonical paper identity when duplicated.
- [x] Title-only seeds contribute preference text without requiring metadata fetch success.
- [x] Seed preference representation includes reusable deterministic vector-like features.
- [x] Ranking supports seed-only recommendation with seed-similarity rationale.
- [x] Ranking supports hybrid topic + seed recommendation in one call.
- [x] Invalid seed input returns a structured validation error without workflow crash.
- [x] Seed metadata fetch failure falls back to available seed text.
- [x] SQLite persists and reloads seed preference data for later reuse.
- [x] Manual acceptance artifact exists at `docs/demo/unit3-seed-personalization.md`.
- [x] User accepts Unit 3 before commit and push.

## Unit 3 Verification Record

- Conda environment: `daily-arxiv-agent`
- Tests: `38 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Review: `/ce-code-review` completed; one non-blocking URL host-validation hardening suggestion remains.
- Commit: `8fbd0c3 feat(unit3): add seed-paper personalization workflow`
- Manual artifact: seed-paper recommendation list generated in `docs/demo/unit3-seed-personalization.md`, including duplicate normalization and seed-similarity ranking evidence.

## Unit 4 Acceptance Checklist

- [x] Like feedback increases scores for similar candidate papers.
- [x] Dislike feedback decreases scores for similar candidate papers.
- [x] Refined recommendations include previous rank, new rank, score delta, and rationale.
- [x] Feedback on a paper outside the current result set is recorded without breaking refinement.
- [x] Conflicting feedback on the same paper follows the documented latest-wins rule.
- [x] Invalid feedback values return a structured validation error.
- [x] SQLite persists feedback events for later recommendation calls.
- [x] Manual acceptance artifact exists at `docs/demo/unit4-feedback-refinement.md`.
- [x] User accepts Unit 4 before commit and push.

## Unit 4 Verification Record

- Conda environment: `daily-arxiv-agent`
- Tests: `48 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Commit: `6c1deea feat(unit4): add feedback refinement loop`
- Manual artifact: before/after feedback comparison generated in `docs/demo/unit4-feedback-refinement.md`, including rank movement, score deltas, and rationale text.

## Unit 5 Acceptance Checklist

- [x] Recommendation workflow calls retrieval, ranking, extraction, and briefing in order.
- [x] Workflow trace records each Skill call, input summary, output summary, evidence source, fallback status, and structured error details.
- [x] Feedback refinement workflow records feedback and returns updated recommendations through the orchestrator.
- [x] Follow-up topic/date queries filter stored papers without unnecessary retrieval.
- [x] Empty local follow-up results trigger retrieval when configured, or a clear fallback when retrieval is unavailable.
- [x] Skill failures appear in workflow trace and return top-level fallback output.
- [x] CLI fixture demo runs a workflow end to end.
- [x] Manual acceptance artifact exists at `docs/demo/unit5-workflow-trace.md`.
- [x] User accepts Unit 5 before commit and push.

## Unit 5 Verification Record

- Conda environment: `daily-arxiv-agent`
- Tests: `64 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Commit: `da4e665 fix(unit5): preserve workflow error and cache fallback semantics`
- Manual artifact: workflow trace output generated in `docs/demo/unit5-workflow-trace.md`, including recommendation trace, follow-up local reuse, feedback refinement coverage, and fallback visibility.

## Unit 6 Acceptance Checklist

- [x] Method mode returns problem, method overview, core workflow, inputs/outputs, and innovation.
- [x] Experiment mode returns datasets, baselines, metrics, setup, and conclusions when full-text evidence is available.
- [x] Limitations mode returns stated limitations, assumptions, missing validation, and risks when source text supports them.
- [x] Abstract-only fallback clearly labels the evidence source and avoids unsupported experiment claims.
- [x] PDF parsing failure falls back to abstract-only output when an abstract is available.
- [x] Missing selected paper returns a structured not-found error through the orchestrator.
- [x] Selected-paper explanation can run after a recommendation workflow.
- [x] Manual acceptance artifact exists at `docs/demo/unit6-deep-explanation.md`.
- [x] User accepts Unit 6 before commit and push.

## Unit 6 Verification Record

- Conda environment: `daily-arxiv-agent`
- Tests: `91 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Commit: `00dce6c feat(unit6): add selected-paper deep explanation workflow`
- Manual artifact: three deterministic explanation outputs generated in `docs/demo/unit6-deep-explanation.md`, covering method, experiment, and limitations modes from full-text evidence plus documented fallback behavior.

## Unit 7 Acceptance Checklist

- [x] Streamlit app module exists at `src/daily_arxiv_agent/ui/streamlit_app.py`.
- [x] App imports without side effects that require live API credentials.
- [x] Recommendation helpers render ranked rows from structured recommendation objects.
- [x] Workflow trace helpers render status, evidence labels, and fallback details.
- [x] Empty recommendation states render a clear user-facing message.
- [x] Fallback Skill results render a clear warning/error message without breaking the page.
- [x] Manual acceptance artifact exists at `docs/demo/unit7-demo-ui.md`.
- [x] User accepts Unit 7 before commit and push.

## Unit 7 Verification Record

- Conda environment: `daily-arxiv-agent`
- Tests: `99 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Manual artifact: `docs/demo/unit7-demo-ui.md` records the recommended demo setup, expected UI flow, and runtime checks for recommendation, feedback, explanation, and follow-up paths.
- Commit: `c9f2213 feat(unit7): add Streamlit demo UI`

## Unit 8 Acceptance Checklist

- [x] Recommendation evaluation compares ranked results against expected relevant paper IDs.
- [x] Feedback evaluation detects before/after rank and score movement after likes/dislikes.
- [x] Explanation completeness checks report present and missing required sections.
- [x] Empty recommendation lists return a meaningful zero-data evaluation result.
- [x] Malformed evaluation fixtures return a structured validation error.
- [x] Manual acceptance artifact exists at `docs/demo/final-demo-script.md`.
- [x] Manual acceptance artifact exists at `docs/demo/evaluation-summary.md`.
- [x] User accepts Unit 8 before commit and push.

## Unit 8 Verification Record

- Conda environment: `daily-arxiv-agent`
- Targeted tests: `9 passed` with `conda run -n daily-arxiv-agent python -m pytest tests/test_evaluation.py`
- Full tests: `108 passed` with `conda run -n daily-arxiv-agent python -m pytest`
- Manual artifact: `docs/demo/final-demo-script.md` gives the final presentation flow.
- Manual artifact: `docs/demo/evaluation-summary.md` records lightweight metric scope, sample fixture interpretation, validation behavior, and reporting notes.
- Commit: pending.
