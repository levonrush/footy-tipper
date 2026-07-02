# Competition Strategy Layer

Pure argmax tipping maximizes expected correct tips. A tipping comp is won on
*relative* score against the field, which changes the optimal play:

- **Leading**: mirror the field's tips on coin-flip games. If everyone scores
  the same, nobody can catch you — deviation is variance you don't need.
- **Trailing**: you need variance. High-EV contrarian picks (where the model
  disagrees with the market) buy catch-up probability that favourite-tipping
  never will.

`pipeline/common/use_predictions/comp_strategy.py` implements this as a
search over small deviations from the model's argmax tips, scored by
simulated P(win comp).

## How it works

1. **Reality model**: game outcomes for the current round are drawn from the
   *calibrated model probabilities* (our best estimate of what will happen).
2. **Field model**: N rivals tip the market favourite, flipping to the
   underdog at a per-rival skill rate (`~N(0.12, 0.06)` clipped). Their
   correctness is evaluated against the same simulated outcomes, so
   correlation between your tips and the field's emerges naturally.
3. **Future rounds**: approximated from market prices via the joker round
   metrics (mu/sigma per remaining round). Your future mean gets a small
   edge (`FOOTY_TIPPER_COMP_USER_EDGE`, default 0.15 correct tips/round)
   reflecting the model's honest-eval advantage over market favourites.
4. **Search**: every subset of ≤ `FOOTY_TIPPER_COMP_MAX_FLIPS` flips among
   games with calibrated prob inside the flip band ([0.38, 0.62] by default)
   is scored against the *same* simulated draws (exactly paired comparisons).
   The best set must beat the baseline by `FOOTY_TIPPER_COMP_MIN_PWIN_GAIN`
   (default 0.002) or the model's tips stand.

Everything is deterministic per round (seeded from `GAME_SEED_BASE`), and
every failure path returns "unavailable" — the model's tips are never lost.

## Modes

`FOOTY_TIPPER_COMP_STRATEGY`:

- `off` — layer disabled.
- `advisory` (default) — deviations are logged and shown in the email
  closing with the P(win comp) math, but the sent tips stay pure model.
- `auto` — deviations are applied to the outgoing email's tips. The
  `predictions_table` is **never** modified; margins/scorelines are
  re-clamped to the adjusted tip.

Every run (production sends) writes one row per game to the
`comp_strategy_decisions` table: baseline tip, strategy tip, mode, scenario,
points gap, and both P(win) numbers — so you can audit every deviation after
the season.

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `FOOTY_TIPPER_COMP_STRATEGY` | `advisory` | `off` / `advisory` / `auto` |
| `FOOTY_TIPPER_COMP_GAP` | joker gap or 0 | Points behind the leader (negative = leading) |
| `FOOTY_TIPPER_COMP_FIELD_SIZE` | 75 | Rivals in the comp |
| `FOOTY_TIPPER_COMP_MAX_FLIPS` | 2 | Max deviations per round |
| `FOOTY_TIPPER_COMP_SIMULATIONS` | 8000 | Monte Carlo draws |
| `FOOTY_TIPPER_COMP_FLIP_BAND_LO/HI` | 0.38 / 0.62 | Only games this uncertain may be flipped |
| `FOOTY_TIPPER_COMP_MIN_PWIN_GAIN` | 0.002 | Required P(win) lift before deviating |
| `FOOTY_TIPPER_COMP_USER_EDGE` | 0.15 | Your expected extra correct tips per future round |
| `FOOTY_TIPPER_COMP_ROUNDS_LEFT` | all priced | Cap on modelled future rounds |

`FOOTY_TIPPER_COMP_GAP` falls back to `FOOTY_TIPPER_JOKER_POINTS_GAP` so the
joker and strategy layers share one view of the standings. Update it weekly.

## Expected behaviour

- Big lead, plenty of season left → flips marginal anti-market picks to the
  market side (shadowing the field).
- Neutral → rarely deviates; the model's tips are usually optimal.
- Chasing → takes the model's highest-edge contrarian picks; if the deficit
  is too large for two flips to matter, it says so honestly (P(win) ≈ 0
  regardless — the joker round is the bigger lever there).
