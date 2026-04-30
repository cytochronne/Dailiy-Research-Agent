---
date: 2026-04-30
topic: improve-arxiv-search
---

# Improve arXiv Search Quality

## Problem Frame

The current topic-based search is too narrow for realistic paper discovery. It sends the raw topic to arXiv as one mostly literal query, retrieves a small first-page candidate set, and then ranks only those papers with simple keyword overlap. This causes two visible failures: users miss relevant papers when they provide several keywords, and the recommendation workflow has too few candidates to rank well.

The improved system should treat topic input as an intent signal rather than a literal phrase. It should retrieve a broader candidate pool, merge multiple arXiv result sets, and use hybrid ranking to surface the most relevant papers while preserving explainable evidence and arXiv-safe behavior.

## Requirements

**Candidate Retrieval**
- R1. The system must separate final recommendation count from candidate retrieval size, so users can request a small Top-K while the retrieval Skill gathers a larger candidate pool.
- R2. The retrieval flow must support multi-page arXiv metadata fetching for a single user request, with request pacing and without bulk PDF downloads.
- R3. The retrieval flow must support multiple arXiv query variants for the same user topic, including relevance-oriented and recency-oriented result sets.
- R4. The system must merge candidates from multiple query variants and remove duplicate papers by arXiv paper identity before ranking.
- R5. Retrieval metadata must preserve enough provenance to show which query variant or strategy found each paper.

**Topic Understanding**
- R6. The system must no longer rely only on a full-topic literal query such as `all:"complete topic text"`.
- R7. The system must support deterministic query expansion from user topic text into important terms, phrases, and fielded arXiv searches over title, abstract, and all-fields metadata.
- R8. The system should support synonym, abbreviation, and related-term expansion when the expansion source is explainable or inspectable.
- R9. The system must include a deterministic fallback query plan that works without an LLM provider.
- R10. When an LLM provider is configured, the system should optionally use an LLM query planner to produce expanded keywords, phrases, related terms, suggested arXiv categories, and possible exclusion terms.
- R11. LLM-generated query plans must be bounded and auditable: the UI or trace should expose the generated terms and the final arXiv query variants.

**Hybrid Ranking**
- R12. Ranking must score papers using more than raw keyword overlap, combining lexical relevance, phrase coverage, arXiv result-order or query-source signals when available, recency, category fit, seed-paper similarity, and feedback adjustment.
- R13. Ranking rationale must remain explainable, showing the major signals that affected each paper's score.
- R14. The system should support semantic similarity ranking when an embedding or comparable vector signal is available, but it must still produce useful rankings without that dependency.
- R15. Ranking must avoid filling the Top-K with clearly unrelated zero-evidence papers unless the result set is too small, and such fallback inclusions must be labeled.

**User Control and Workflow Visibility**
- R16. The UI should distinguish between final Top-K recommendations and candidate pool size.
- R17. The UI should expose a search mode that lets users choose a broader recall-oriented search or a stricter precision-oriented search.
- R18. Workflow traces must show candidate count before ranking, query strategy, cache behavior, ranking mode, and any LLM query-planner fallback.
- R19. Existing follow-up queries should benefit from the same improved topic matching behavior where practical, while still preferring local stored papers before live retrieval.

## Success Criteria

- A multi-keyword topic that currently returns too few or weak results returns a broader candidate pool and a more relevant Top-K list.
- Users can retrieve substantially more than 10 metadata candidates while still seeing a concise Top-K recommendation list.
- Changing from strict to broad search visibly changes candidate coverage without hiding the query strategy.
- The workflow still works in fake/offline LLM mode through deterministic query expansion and deterministic ranking.
- Recommendation rows include understandable relevance rationales rather than opaque model scores.
- The implementation continues to respect arXiv API usage constraints and does not bulk-download PDFs.

## Scope Boundaries

- This improvement does not require downloading full PDFs for routine search or ranking.
- This improvement does not require training a custom recommendation model.
- This improvement does not require replacing arXiv as the primary metadata source.
- This improvement does not require a production-scale search index or external database service.
- LLM query planning is an enhancement, not a hard dependency for search.

## Key Decisions

- Use Approach B plus Approach C together: Hybrid retrieval/ranking should be the main reliable path, while LLM query planning improves topic understanding when available.
- Keep deterministic fallback behavior: The course/demo workflow must remain testable and usable without live LLM calls.
- Prefer larger metadata candidate pools over PDF-heavy retrieval: Metadata and abstracts are enough for first-pass discovery, and selected-paper full-text explanation remains a separate workflow.
- Make broad recall the recommended default: Missing relevant papers is currently more harmful than including a few candidates that ranking can demote.
- Preserve explainability: Search quality should improve without turning recommendation output into an opaque black box.

## High-Level Technical Direction

```text
User topic
  -> Query planner
       -> deterministic expansion
       -> optional LLM expansion
  -> Multi-query arXiv metadata retrieval
       -> relevance-oriented queries
       -> recency-oriented queries
       -> category/date filters
       -> paginated candidate pool
  -> Candidate merge and dedupe
  -> Hybrid ranking
       -> lexical and phrase score
       -> optional semantic score
       -> recency/category/seed/feedback signals
  -> Top-K recommendations with traceable rationales
```

## Dependencies / Assumptions

- arXiv API supports fielded search, Boolean query composition, pagination with `start` and `max_results`, submitted-date filtering, and sorting by relevance or submitted date.
- Current local SQLite storage can continue to cache normalized paper metadata and retrieval result sets, though planning may need to adjust cache keys for query plans and multi-query retrieval.
- Existing fake/live LLM provider boundaries can be reused for optional query planning if planning confirms they fit the required structured output.

## Outstanding Questions

### Resolve Before Planning

None.

### Deferred to Planning

- [Affects R1, R2][Technical] Choose default candidate pool size and page size that improve recall while keeping arXiv request pacing reasonable.
- [Affects R7, R8][Technical] Define the deterministic query expansion rules and stopword handling.
- [Affects R10, R11][Technical] Define the LLM query-plan schema and validation rules.
- [Affects R12, R14][Technical] Choose the first hybrid scoring formula, including how to use arXiv result ordering, and whether semantic similarity is available in the local environment.
- [Affects R17][Product] Decide the exact UI labels and defaults for broad versus strict search modes.
- [Affects R18][Technical] Decide how query-plan and ranking-signal details should appear in workflow trace metadata without overwhelming the UI.

## Next Steps

-> /ce:plan for structured implementation planning.
