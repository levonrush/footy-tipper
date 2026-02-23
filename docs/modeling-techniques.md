# Modeling Techniques

This doc explains the modelling choices and tradeoffs in Footy Tipper.

## 1) Tier-A Baseline Layer

Tier-A provides a stable prior using team state and market-aware context.

Why it matters:
- gives sane predictions when features are sparse/noisy
- improves robustness early season and around roster disruption

## 2) Tier-B Score Models

Two separate Poisson regressors model:
- expected home score
- expected away score

Why score models first:
- score distributions can generate both margin and win probability
- easier to simulate than direct class-only models

## 3) Blend Instead of Replace

The system blends Tier-A and Tier-B expected scores using learned weights.

Why blend:
- pure model can overfit
- pure baseline can underfit
- blend gives smoother bias/variance behavior

## 4) Probability Stacking + Calibration

From blended scores, the pipeline creates conditional home-win probability signals:
- Tier-A conditional probability
- Tier-B conditional probability
- market conditional probability

These are stacked with logistic regression, then calibrated with a beta calibrator.

Why this stack:
- different signals dominate in different matches
- calibration improves downstream decision quality (value picks, joker calls)

## 5) Bivariate Poisson Simulation

Inference simulates outcomes using bivariate Poisson with shared component `lambda3`.

Why not independent Poisson only:
- independent score assumptions often understate shared game conditions
- `lambda3` captures correlated scoring environments

## 6) Decision Layer, Not Just Prediction Layer

Predictions are converted into decisions:
- value picks from expected value (`p * odds - 1`)
- Kelly-derived stake sizing with caps/floors
- joker timing recommendation

This is intentional: the model is built for tipping decisions, not just leaderboard metrics.

## 7) Practical Design Constraints

Key constraints considered during implementation:
- strict train/infer split (`Final` vs `Pre Game`)
- no hardcoded season end year
- graceful degradation when Google/OpenAI deps are missing
- offseason-safe behavior when no pre-game fixtures exist

## 8) Known Modeling Risks

- odds availability by round can vary; decision confidence should reflect coverage
- feed latency can create temporary mismatch between market and model snapshot
- competition-winning objectives depend on field behavior, not only expected points

That last point is why joker policy includes simulation of field dynamics.
