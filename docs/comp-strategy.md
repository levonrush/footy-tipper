# Competition strategy

Picking the most likely winner maximizes expected correct tips. Winning a competition means beating other people, which is a different objective once the ladder position matters.

When leading, correlated tips can protect a gap. When chasing, a carefully chosen disagreement can buy the variance required to catch up. The strategy layer searches those deviations without altering the canonical model predictions.

## Simulation contract

[`comp_strategy.py`](../pipeline/common/use_predictions/comp_strategy.py) performs a deterministic per-round simulation:

1. Draw each game's outcome from the calibrated model probability.
2. Simulate rival tips around the market favourite with heterogeneous skill.
3. Approximate future rounds from the priced round opportunity metrics.
4. Enumerate subsets of at most the configured number of flips inside the uncertainty band.
5. Compare every candidate against the baseline using the same random draws.
6. Apply a deviation only when its estimated competition-win gain clears the configured threshold.

The layer returns `unavailable` on failure and preserves the base model tips.

## Modes and ownership

| Mode | Email behavior | Database behavior |
| --- | --- | --- |
| `off` | No strategy advice | No applied deviations |
| `advisory` (default) | Explain proposed deviations; send base tips | Audit decision only |
| `auto` | Apply selected deviations to outgoing tips | `predictions_table` remains unchanged |

In `auto`, outgoing scoreline/margin presentation is re-clamped to the adjusted tip. The underlying `predictions_table` is never rewritten: it remains the model record, while `comp_strategy_decisions` records baseline tip, strategy tip, mode, scenario, points gap, and both competition-win estimates for each game.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOOTY_TIPPER_COMP_STRATEGY` | `advisory` | `off`, `advisory`, or `auto` |
| `FOOTY_TIPPER_COMP_GAP` | joker gap or `0` | Points behind the leader; negative means ahead |
| `FOOTY_TIPPER_COMP_FIELD_SIZE` | `75` | Rival count in the simulated competition |
| `FOOTY_TIPPER_COMP_MAX_FLIPS` | `2` | Maximum current-round deviations |
| `FOOTY_TIPPER_COMP_SIMULATIONS` | `8000` | Monte Carlo draws |
| `FOOTY_TIPPER_COMP_FLIP_BAND_LO` / `HI` | `0.38` / `0.62` | Eligible model-probability band |
| `FOOTY_TIPPER_COMP_MIN_PWIN_GAIN` | `0.002` | Required estimated gain before deviating |
| `FOOTY_TIPPER_COMP_USER_EDGE` | `0.15` | Expected extra correct tips per future round |
| `FOOTY_TIPPER_COMP_ROUNDS_LEFT` | all priced | Optional future-round cap |

Update the points gap and field size from the actual competition. Defaults make the model runnable; they do not know your pub.

## Expected transitions

- **Leading:** marginal anti-market tips may be flipped back toward the field to reduce catch-up variance.
- **Neutral:** most rounds retain the model argmax because a deviation has little relative value.
- **Chasing:** high-edge contrarian candidates may be selected when their estimated catch-up benefit exceeds the threshold.
- **Mathematically cooked:** the system should report near-zero win probability instead of inventing a heroic two-flip rescue.

## Idempotency and audit

Simulations are seeded from round/game state, so an unchanged rerun produces the same comparison. Production runs record decision rows, but do not mutate model predictions. Email idempotency is shared by the Drive delivery marker and `email_sends`; confirming a manual live dispatch does not authorize a different simulation input and should not be used to fish for a better result.

## Relationship to betting decisions

Competition strategy optimizes relative tipping score. Value picks use expected monetary value, and staking uses a bounded Kelly fraction. A contrarian comp tip is not automatically a value bet, and a value bet is not automatically the right tipping-comp deviation. See [Principled odds integration](principled-odds-integration.md).
