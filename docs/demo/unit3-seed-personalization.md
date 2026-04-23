# Unit 3 Seed-Paper Personalization

## Run Context

- Personalization mode: deterministic seed-paper vector ranking
- Seed metadata mode: fixture-backed arXiv metadata client for acceptance repeatability
- Evidence boundary: metadata and abstract only
- Profile ID: `default`

## Seed Inputs

| Input | Normalized contribution | Result |
|---|---|---|
| `2604.00001` | `arxiv:2604.00001` | Resolved to metadata for `Explainable Agents for Daily Research Briefings` |
| `https://arxiv.org/abs/2604.00001v2` | `arxiv:2604.00001` | Collapsed as a duplicate of the ID seed |
| `Agent workflows for research paper recommendation` | `title:agent workflows for research paper recommendation` | Added as title-only preference text |

Preference build status: `success`

Preference metadata:

```json
{
  "profile_id": "default",
  "seed_count": 2,
  "duplicate_count": 1,
  "invalid_inputs": [],
  "fetch_failures": []
}
```

## Seed-Based Recommendation List

| Rank | Paper | Score | Evidence | Rationale |
|---:|---|---:|---|---|
| 1 | [Agent Workflows for Research Paper Recommendation](https://arxiv.org/abs/2604.01001) | 7.5443 | abstract | Seed-paper similarity: 0.754. Evidence: abstract. |
| 2 | [Explainable Scientific Search with Agent Briefings](https://arxiv.org/abs/2604.01002) | 6.2663 | abstract | Seed-paper similarity: 0.627. Evidence: abstract. |
| 3 | [A Survey of Compiler Register Allocation](https://arxiv.org/abs/2604.01003) | 0.695 | abstract | Seed-paper similarity: 0.070. Evidence: abstract. |

## Notes

- The top result shares the seed interest terms `agent`, `workflow`, `research`, `paper`, and `recommendation`.
- The arXiv URL seed normalizes to the same paper identity as the equivalent ID and does not double-count the seed paper.
- The unrelated compiler paper is still returned to fill Top-K, but its low score and rationale make the weak match visible.
- Automated tests cover arXiv ID resolution, URL normalization, title-only seeds, duplicate collapse, invalid seed errors, metadata-fetch fallback, seed-only ranking, hybrid topic+seed ranking, and SQLite preference reuse.

## Acceptance Question

Does seed-based personalization behave plausibly?
