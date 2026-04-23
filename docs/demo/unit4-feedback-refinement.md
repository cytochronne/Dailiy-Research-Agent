# Unit 4 Feedback Refinement Demo

Manual acceptance artifact for Unit 4: Feedback Refinement Loop.

## Scenario

Starting point: an initial recommendation list has one compiler paper above one agent recommendation paper. The user then provides two feedback events for the same recommendation run:

- Like `2604.00001`: Agent Workflows for Research Recommendation
- Dislike `2604.00004`: Compiler Register Allocation Survey

Conflict rule: latest feedback wins per paper. This demo has no conflicting events, but the implementation and tests cover that rule.

## Before Feedback

| Rank | Paper ID | Score | Title |
|------|----------|-------|-------|
| 1 | `2604.00003` | 3.2000 | Compiler Optimization Benchmarks |
| 2 | `2604.00002` | 2.4000 | Feedback Agents for Paper Recommendation |
| 3 | `2604.00005` | 1.6000 | Graph Neural Weather Forecasting |

## After Feedback

| New Rank | Previous Rank | Rank Delta | Paper ID | New Score | Score Delta | Title |
|----------|---------------|------------|----------|-----------|-------------|-------|
| 1 | 2 | +1 | `2604.00002` | 6.0876 | +3.6876 | Feedback Agents for Paper Recommendation |
| 2 | 3 | +1 | `2604.00005` | 1.5487 | -0.0513 | Graph Neural Weather Forecasting |
| 3 | 1 | -2 | `2604.00003` | -0.1675 | -3.3675 | Compiler Optimization Benchmarks |

## Change Rationales

`2604.00002` moved from rank 2 to rank 1 because it is similar to the liked agent-workflow paper and only weakly similar to the disliked compiler paper.

```text
Feedback adjustment: liked 2604.00001 moved similar papers up (+4.291); disliked 2604.00004 moved similar papers down (-0.603). Previous rank: 2; score delta: +3.6876.
```

`2604.00003` moved from rank 1 to rank 3 because it is strongly similar to the disliked compiler paper.

```text
Feedback adjustment: liked 2604.00001 moved similar papers up (+0.632); disliked 2604.00004 moved similar papers down (-4.000). Previous rank: 1; score delta: -3.3675.
```

## Acceptance Notes

- Feedback events are tied to `profile_id`, `recommendation_run_id`, `paper_id`, and optional paper metadata.
- SQLite persistence stores complete `FeedbackEvent` payloads for later reuse.
- Refined recommendations expose `previous_rank`, `previous_score`, `score_delta`, and `rank_delta`.
- `TopicRankingSkill.rank(...)` can accept persisted feedback events so later recommendation calls are influenced by previous feedback.
- Invalid feedback values return a structured `SkillResult` error with code `invalid_feedback_value`.
