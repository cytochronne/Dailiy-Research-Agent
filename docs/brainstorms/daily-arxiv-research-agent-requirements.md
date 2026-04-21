---
date: 2026-04-21
topic: daily-arxiv-research-agent
---

# Daily arXiv Research Briefing Agent

## Problem Frame

The project needs a course-ready Agent + Skills system that can monitor new arXiv papers, recommend papers according to explicit topics or implicit seed-paper interests, generate a structured daily briefing, refine recommendations from user feedback, and explain selected papers in depth.

The main risk is overbuilding a platform before the end-to-end research workflow works. The first useful system should make the Agent workflow visible, keep every Skill independently testable, and demonstrate a full recommendation-feedback-explanation loop with clear outputs for reports and presentation.

## Requirements

**Agent and Skill Architecture**
- R1. The system must use an Agent + Skills architecture where each Skill has a single responsibility, explicit inputs, explicit outputs, and can be tested independently.
- R2. The Agent must orchestrate the complete recommendation workflow: user input, arXiv retrieval, ranking, information extraction, daily briefing generation, feedback recording, preference update, and refined recommendations.
- R3. The Agent must orchestrate the paper-level explanation workflow: selected paper, selected explanation mode, full or abstract-level content preparation, and final explanation output.
- R4. The system must expose the workflow clearly enough for presentation, including which Skill produced each major intermediate result.

**arXiv Retrieval and Paper Metadata**
- R5. The system must retrieve new arXiv papers by date or time range.
- R6. Retrieval must support filtering by topic, keyword, and arXiv category.
- R7. Retrieved paper metadata must include title, authors, abstract, category, publication or submission date, paper URL, and PDF URL when available.
- R8. Retrieval must respect arXiv API usage constraints, including reasonable request pacing and avoiding unnecessary bulk PDF downloads.

**Recommendation and Ranking**
- R9. The system must support explicit topic or keyword based ranking.
- R10. The system must support seed-paper-based personalized recommendation without requiring explicit keywords.
- R11. Seed papers must support at least arXiv ID, arXiv URL, and title input. PDF upload can be added after the core loop works.
- R12. Ranking output must include Top-K recommendations with a relevance score or a short ranking rationale for each paper.
- R13. The first implementation should use an explainable hybrid ranking strategy rather than training a custom model from scratch.

**Feedback Refinement**
- R14. Users must be able to mark recommended papers as like or dislike.
- R15. Feedback must update the user preference representation for later ranking.
- R16. The system must produce a refined recommendation list after feedback.
- R17. Refined recommendations must show enough rationale to explain why rankings changed after feedback.

**Structured Extraction and Daily Briefing**
- R18. The system must extract or generate structured paper fields for briefing: concise summary, key contributions, methods, and relevance rationale.
- R19. The daily briefing must include a summary table, a ranked recommendation list, brief introductions for recommended papers, and a highlighted most relevant or most worth-attention paper.
- R20. Briefing output must be structured enough to render in a UI and to reuse in reports.

**Follow-up Queries**
- R21. Users must be able to continue filtering prior or newly retrieved papers by topic.
- R22. Users must be able to continue filtering prior or newly retrieved papers by date range.
- R23. Follow-up queries should reuse the existing retrieved paper set when possible instead of always re-fetching from arXiv.

**Paper-Level Deep Explanation**
- R24. Users must be able to select one paper from the recommendation list for deeper explanation.
- R25. The system must support three explanation modes: method/framework explanation, experiment/results explanation, and limitations analysis.
- R26. Method/framework explanation must cover the problem addressed, overall method, core modules or workflow, inputs and outputs, and claimed innovation.
- R27. Experiment/results explanation must cover datasets, baselines, metrics, experimental setup, and main conclusions when that information is available.
- R28. Limitations analysis must cover stated limitations, implicit assumptions, missing validation, and possible risks.
- R29. Deep explanation should use full paper text when available. Abstract-only explanation is acceptable only as a clearly labeled fallback.

**Evaluation and Delivery**
- R30. The system must include evaluation hooks for recommendation quality, ranking changes after feedback, and deep explanation quality.
- R31. The final demo must show the Agent workflow, recommendation results, feedback loop effect, and one selected paper explained in all supported modes.
- R32. The code and documentation must support course deliverables: group Agent report, individual Skill report, code submission, and presentation.

**Reliability and Evidence Boundaries**
- R33. The system must handle empty retrieval results, invalid seed-paper inputs, arXiv/API failures, PDF parsing failures, and LLM failures without crashing the Agent workflow.
- R34. User-facing outputs must clearly label whether a briefing or explanation is based on metadata, abstract text, or full paper text.
- R35. The system must avoid presenting unavailable experimental details or limitations as facts; when evidence is missing, it must say the information was not found in the available source.
- R36. Each generated recommendation, briefing item, and deep explanation must preserve paper provenance, including the arXiv page or paper URL used.

## Success Criteria

- A user can provide either keywords or seed papers and receive a daily briefing with ranked Top-K papers.
- A user can like and dislike papers, rerun refinement, and see a changed recommendation list with clear change rationale.
- A user can select a recommended paper and generate all three explanation modes.
- Each core Skill can be tested independently with fixture inputs and structured outputs.
- Missing data, empty results, and upstream failures produce clear fallback output instead of a broken workflow.
- The presentation can show a readable Agent workflow and concrete intermediate artifacts, not only final text.
- The project avoids heavy production infrastructure while still demonstrating the full requirements in `development_checklist.md`.

## Scope Boundaries

- The first version is a local or single-user course demo, not a production multi-user SaaS.
- The first version should not require user accounts, authentication, billing, or deployment automation.
- The first version should not batch-download full PDFs for every retrieved arXiv paper.
- The first version should not train a custom recommendation model from scratch.
- The first version should not depend on a complex database service unless planning shows SQLite is insufficient.
- PDF upload for seed papers is useful but secondary to arXiv ID, URL, and title seed support.

## Key Decisions

- Build the first version as a Python Agent + Skills MVP: This keeps the system easy to test, explain, and present.
- Prefer explicit Skill contracts over a heavy agent framework at the start: The course requirement is clearer if each Skill has inspectable inputs and outputs.
- Use metadata and abstracts for daily retrieval and ranking, then parse full text only for selected paper explanation: This balances quality with arXiv usage constraints and implementation time.
- Use explainable hybrid ranking for MVP: Combining topic/keyword similarity, seed-paper similarity, recency/category signals, and feedback adjustment is easier to debug and present than a trained recommender.
- Treat feedback refinement as preference-vector adjustment first: Like/dislike feedback should visibly change ranking without needing a large training dataset.
- Use a lightweight UI for demonstration: The UI should prioritize showing workflow, recommendations, feedback changes, and paper explanations over product polish.

## High-Level Technical Direction

```text
User input
  -> arXiv Retrieval Skill
  -> Seed Paper Parsing / Preference Modeling Skill
  -> Personalized Ranking Skill
  -> Paper Information Extraction Skill
  -> Daily Briefing Skill
  -> Feedback Update Skill
  -> Refined Ranking Skill
  -> Paper Deep Explanation Skill
```

The recommended implementation shape is a local Python application with independently testable Skill modules, structured data contracts, lightweight persistent storage, and a simple demonstration UI. Planning should choose the exact libraries and file layout.

## Dependencies / Assumptions

- arXiv metadata retrieval will use the official arXiv API or another arXiv-supported access path.
- The system will use an LLM for structured extraction, briefing generation, and deep explanation.
- Embedding-based similarity is acceptable for the first personalized recommendation version.
- The project is currently early-stage and `development_checklist.md` is the primary source of requirements.
- The first planning pass should assume a single-user local workflow unless the user explicitly expands scope.

## Alternatives Considered

- Full web platform with FastAPI, Postgres, and a frontend: More production-like, but likely too much infrastructure before the workflow is proven.
- Research-heavy full-text RAG system from day one: Higher explanation ceiling, but PDF parsing and vector infrastructure may dominate the project.
- Abstract-only explanation for all modes: Simpler, but weak for experiment/result and limitations modes, so it should only be a fallback.

## Outstanding Questions

### Resolve Before Planning

None.

### Deferred to Planning

- [Affects R13][Technical] Choose the initial ranking method and scoring formula.
- [Affects R18][Technical] Choose the structured output schema for paper extraction and briefing.
- [Affects R29][Technical] Choose the PDF/full-text extraction approach and fallback behavior.
- [Affects R30][Technical] Define lightweight evaluation fixtures and metrics for the course demo.
- [Affects R4][Technical] Decide whether workflow visualization is generated from logs, state objects, or a static diagram in the UI.
- [Affects R18, R25][Needs research] Choose the LLM provider/model strategy available in the development environment.
- [Affects R33][Technical] Define the standard error/fallback shape returned by each Skill.

## Next Steps

-> /ce:plan for structured implementation planning.
