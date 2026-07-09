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

## 9) 2026-07 Upgrades

Changes shipped in the July 2026 modelling pass (each gated on the honest
nested eval in `footy-tipper evaluate`; reports persist to `reports/`):

- **Line/spread market in the stacker.** The handicap market is the
  bookmaker's own margin model. The stacker now sees the line cover
  probability (Shin), line overround, and the model-vs-line spread
  disagreement (`calibration.build_line_market_features`). Stacker feature
  layouts are versioned, so pre-upgrade pickles still load.
- **Margin blend for the tie-breaker.** A 3-coefficient ridge (OOF model
  margin + market spread + Tier-A margin) saved in the manifest overrides
  the simulated margin at inference; games without a line fall back to the
  simulation. Margin/scoreline are also importance-reweighted to agree with
  the calibrated win probability and sign-clamped to the tip.
- **LOSO calibrator.** The beta calibrator is fit on leave-one-season-out
  stacker predictions, removing the stacker-overfit flattery in calibration.
- **Tier-A tuning (default on).** Alpha/carryover grid-searched on
  sequential log-loss; evaluate tunes only on pre-holdout seasons.
- **Negative-binomial simulation.** Per-side dispersion estimated from OOF
  residuals; the simulation draws gamma-mixed Poisson scores when set
  (NRL points are 2/4/6-lumpy, so Poisson understates margin variance).
- **Joker uses model probabilities** for the current round and reports the
  no-joker baseline + lift; strategy ties within epsilon resolve to points.
- **Competition strategy layer** (`docs/comp-strategy.md`): P(win comp)
  optimized tip deviations — shadow the field when leading, contrarian when
  chasing. Advisory by default, `auto` applies to outgoing emails.

Honest benchmark moved from 62.0% accuracy / 0.6450 log-loss / 26.5%
P(win comp in a 75-field) to 63.4% / 0.6413 / ~35% (2024-2026 held out;
market favourite baseline 61.2%).
