# Joker strategy

The project assumes one double-points joker per season. The trick is not finding a round that looks friendly; it is deciding whether that round is better than the opportunities still alive, under the objective you actually care about.

## Two layers

Training builds a historical policy in `models/joker_policy.json`. It aggregates round-level opportunity, simulates season/field scenarios, and records which strategy performed best for lead, neutral, and chase positions.

Send time scores the currently priced rounds using the model's game probabilities where available, with market-derived fallbacks. For each round it calculates expected correct tips (`mu`), variance, standard deviation, coverage, and separation from the next-best round.

| Strategy | Objective | Typical use |
| --- | --- | --- |
| `points` | Maximize expected joker points | Neutral/default state |
| `protect` | Maximize `mu - lambda * sigma` | Protecting a lead |
| `chase` | Prefer high-variance opportunity | Needing a catch-up path |
| `auto` | Read scenario and trained policy | Normal configured operation |

The recommendation reports both the no-joker baseline and the estimated lift. Near-ties resolve toward expected points rather than pretending Monte Carlo noise is insight.

## Guardrails

A `PLAY` call requires:

- enough remaining rounds with odds;
- sufficient game-level odds coverage in the candidate round;
- enough separation from the next-best candidate;
- an unused joker for that competition year.

Thin markets should produce `HOLD`. That is the guardrail working.

## Runtime state transition

`joker_usage` is keyed by `competition_year` and stores the played round, timestamp, and source.

```text
unplayed -> recommendation HOLD -> unplayed
unplayed -> test/dry run PLAY -> unplayed
unplayed -> failed production delivery -> unplayed
unplayed -> successful production delivery + PLAY -> used(round)
used(same round) -> sticky already-locked message
used(other round) -> forced HOLD
```

The write happens only after a successful production email. Test sends read state but never mutate it. A repeated production invocation is also constrained by the separate `email_sends` ledger, so joker use and email delivery remain replay-safe.

## Configuration

| Variable | Meaning |
| --- | --- |
| `FOOTY_TIPPER_JOKER_STRATEGY` | `auto`, `points`, `protect`, or `chase` |
| `FOOTY_TIPPER_JOKER_POINTS_GAP` | Points behind the leader; negative means ahead |
| `FOOTY_TIPPER_JOKER_RISK_LAMBDA` | Variance penalty for protect mode |
| `FOOTY_TIPPER_JOKER_MIN_ROUNDS_WITH_ODDS` | Minimum priced future-round count |
| `FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE` | Minimum odds coverage per candidate round |
| `FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO` | Required separation before `PLAY` |
| `FOOTY_TIPPER_JOKER_POLICY_PATH` | Optional policy artifact override |
| `FOOTY_TIPPER_JOKER_BACKTEST_SIMULATIONS` | Training-time simulation count |
| `FOOTY_TIPPER_JOKER_BACKTEST_FIELD_SIZE` | Simulated field size |
| `FOOTY_TIPPER_JOKER_BACKTEST_SEED` | Reproducible policy seed |

The points gap is operational state; update it with the real competition table before relying on `auto`.

## Weekly runbook

1. Refresh predictions and current-round markets.
2. Confirm odds coverage and the configured points gap.
3. Run `footy-tipper send --test --dry-run --skip-drive`.
4. Read the objective, candidate round, baseline, lift, and guardrail explanation.
5. Use the production command only when both the email and joker transition are intended.
6. After delivery, verify `joker_usage` and `email_sends` if the call was `PLAY`.

## Limits

- Future-round prices are often incomplete and can move materially.
- The field model is a scenario, not telemetry from the actual competitors.
- Different competitions implement joker deadlines and scoring differently; the current code assumes one double-points use per season.
- A mathematically favoured `PLAY` can still lose. Reg regrets to advise that variance has not signed the code of conduct.

Research lineage: [Research and history](research-and-history.md#joker-and-competition-strategy) and the archival [Optimal Joker report](../lit-review/README.md#optimal-joker-round-selection-in-footy-tipping-competitions).
