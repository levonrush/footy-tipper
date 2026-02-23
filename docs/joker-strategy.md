# Joker Strategy

This project assumes a single-use, double-points joker per season.

The joker system has two layers:
- round scoring (live, current-season)
- policy learning (historical simulation during training)

## 1) Round Scoring (Live)

At send time, upcoming rounds are scored from market odds:
- remove overround in head-to-head odds
- estimate per-match tip correctness probability
- aggregate per round:
  - `mu` = expected correct tips
  - `variance` / `sigma` = swing potential

Decision modes:
- `points`: maximize `mu`
- `protect`: maximize `mu - lambda*sigma`
- `chase`: maximize `variance`

Guardrails prevent premature `PLAY` calls:
- minimum priced rounds in scope
- minimum odds coverage per round
- minimum lead over next-best round

## 2) Policy Learning (Training-Time)

During `footy-tipper train`, the pipeline runs historical Monte Carlo backtests and saves:
- `models/joker_policy.json`

Backtest idea:
- use historical round-level market-derived `mu`/`sigma`
- simulate many seasons/scenarios (lead/neutral/chase state)
- compare strategy outcomes against a simulated field
- choose recommended strategy per scenario

## 3) Auto Strategy In Current Season

Set:
- `FOOTY_TIPPER_JOKER_STRATEGY=auto`
- `FOOTY_TIPPER_JOKER_POINTS_GAP=<your gap to leader>`

Then send-time logic maps your state:
- ahead by enough -> lead scenario
- near even -> neutral scenario
- behind by enough -> chase scenario

and applies the strategy recommended by `joker_policy.json`.

## 4) Key Env Vars

Core behavior:
- `FOOTY_TIPPER_JOKER_STRATEGY` (`auto`, `points`, `protect`, `chase`)
- `FOOTY_TIPPER_JOKER_POINTS_GAP`
- `FOOTY_TIPPER_JOKER_RISK_LAMBDA`

Guardrails:
- `FOOTY_TIPPER_JOKER_MIN_ROUNDS_WITH_ODDS`
- `FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE`
- `FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO`

Policy artifact control:
- `FOOTY_TIPPER_JOKER_POLICY_PATH` (override location)

Backtest tuning:
- `FOOTY_TIPPER_JOKER_BACKTEST_SIMULATIONS`
- `FOOTY_TIPPER_JOKER_BACKTEST_FIELD_SIZE`
- `FOOTY_TIPPER_JOKER_BACKTEST_SEED`
- plus scenario thresholds/gaps in `pipeline/common/model_training/joker_policy.py`

## 5) Interpretation Rules

Treat `PLAY` as a strong recommendation only when:
- enough rounds are priced
- your selected objective has clear separation

If coverage is thin (common early in market cycle), `HOLD` is expected and correct.

## 6) Literature Link

Source material and notes:
- `lit-review/Optimal Joker-Round Selection in Footy Tipping Competitions.pdf`
- `lit-review/deep-research-report.md`

This implementation is the practical version of that work: rigorous enough to be useful, simple enough to run weekly.
