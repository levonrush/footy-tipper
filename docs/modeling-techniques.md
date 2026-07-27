# Models, markets, and evaluation

The model is an ensemble because NRL games are not obliged to be convenient. One component remembers team strength, one models scores, one models the winner directly, the market contributes its own information, and a calibrated meta-layer settles the argument.

![Production model, market, calibration, margin, and simulation flow](diagrams/model-stack.svg)

[Editable Mermaid source](diagrams/model-stack.mmd)

## Tier A: dynamic baseline

Tier A builds sequential team state from matches available before each game. Its tuned alpha/carryover settings provide expected home and away scores plus a conditional home-win probability. It is deliberately stable when richer features are sparse, especially early in a season.

It is not a full Bayesian attack/defence hierarchy. The research case for that model remains exploratory; calling the current baseline Bayesian would be borrowing a dinner jacket.

## Tier B: home and away score models

Separate tuned Poisson pipelines predict expected home and away scores from fixture, ladder, recent performance, and lineup-derived predictors. Training creates out-of-fold (OOF) score means so blend and meta-model fitting are not judged only on in-sample predictions.

Learned weights blend Tier A and Tier B score means separately by side. These means generate a conditional home-win signal after marginalizing over lineup-selection uncertainty with deterministic per-game Monte Carlo draws.

## Tier C: direct winner model

Tier C is a binary classifier trained on non-draw outcomes with the same selected predictor frame. Its OOF probabilities enter stacker training; the final `binary_model.pkl` supplies the inference signal. This gives the ensemble a direct classification view alongside score-derived probabilities.

If the binary artifact is absent in an older compatible model bundle, inference can continue without it.

## Markets stay separate

Head-to-head prices are de-vigged to a conditional market probability and enter
the market-covered probability pool. Complete fresh spread and totals families
act separately on expected score means before simulation. None of those market
signals are buried inside Tier-B score predictors.

That separation makes the ensemble interpretable and avoids feeding the bookmaker into a score model, then counting the bookmaker again in the stacker. The full rationale and bibliography are in [Principled odds integration](principled-odds-integration.md).

![Naive and current odds integration](diagrams/odds-before-after.svg)

[Editable Mermaid source](diagrams/odds-before-after.mmd)

## Probability pooling and calibration

Two constrained logit pools combine:

- Tier-A conditional probability
- OOF Tier-B conditional probability
- OOF Tier-C probability
- genuine market conditional probability in the market-covered pool only

Pool weights are nonnegative, sum to one, and have no intercept. The
learned market pool is retained only when fully nested season-out predictions
provide material log-loss improvement over the strongest individual
comparator on the same covered rows, stay within the accuracy and Brier
tolerances, and pass recent-season stability checks. The raw market participates
in that comparison, but a rejected pool becomes a one-hot selection of the
strongest Tier A/B/C model expert rather than a raw market-only winner
forecast. Tier B is the safe default when nested evidence is unavailable.

The no-market pool is trained on counterfactually masked OOF rows and is
retained only when it beats Tier B in season-out log loss.
Positive-temperature, no-intercept calibration is fitted to the selected
leave-one-season-out path, so 50% stays neutral and calibration cannot reverse
a tip.

At inference, genuinely missing H2H odds route to the no-market artifact or
the manifest-declared Tier-B fallback. Market-covered games use the
manifest-recorded learned pool or its selected Tier A/B/C expert fallback.
Compatibility guards prevent older artifacts from reversing unanimous
Tier/market evidence.

## Margin and coherent scorelines

Winner probability and expected margin are related but not identical decisions:

- a small ridge blend uses OOF model margin, bookmaker spread, and Tier-A margin when enough honest line rows exist;
- complete spread/total markets adjust expected score means before simulation;
- games without usable paired prices remain entirely model-based;
- stored margin is always the displayed home score minus away score.

Score simulation uses a bivariate Poisson shared component `lambda3`. When OOF residuals support valid overdispersion estimates, gamma-mixed Poisson draws supply a negative-binomial fallback per side; otherwise the ordinary Poisson path remains. Deterministic game-specific seeds prevent a rerun from flipping a tip through random-number drift.

## Artifacts

| File | Contents |
| --- | --- |
| `home_model.pkl`, `away_model.pkl` | Tier-B score pipelines |
| `binary_model.pkl` | Tier-C classifier |
| `stacker.pkl` | selected constrained market-covered logit pool: learned weights or a one-hot Tier A/B/C fallback |
| `win_prob_calibrator.pkl` | positive-temperature calibrator for the selected market-covered path |
| `stacker_no_market.pkl` | constrained Tier A/B/C no-market pool, when selected |
| `win_prob_calibrator_no_market.pkl` | positive-temperature no-market calibrator |
| `model_manifest.json` | predictor schema, blend weights, Tier-A config, `lambda3`, dispersion, uncertainty, margin metadata |
| `joker_policy.json` | historical joker-policy backtest summary |
| `training-receipt.json` | release ID, Git SHA, tuning count, training scope, runtime versions, artifact sizes, and hashes; written after every other staged artifact |

`footy-tipper update-model` trains these artifacts into staging, validates them,
runs the nested season-out acceptance gate against the staged models and
database, publishes and re-downloads a create-only Drive release, then
dispatches `model-check.yml` so the GitHub Actions production image loads that
exact candidate. A failed evaluation cannot publish or activate. Only a
successful hosted check permits pointer activation. The default search remains
100 Bayesian candidates with outer candidate parallelism and single-threaded
LightGBM fits. Actions consumes models but never trains them.

## Honest evaluation

Use:

```bash
footy-tipper advanced model evaluate --skip-prepare --seasons 3
```

The evaluator holds out each recent season in turn and fits blend weights,
pool, and calibrator only on earlier seasons. In addition to the operational
market/no-market routing, every held-out non-draw row is counterfactually
forced through the no-market path. On comparable market-covered rows, the
learned market pool must prove robust incremental value over the strongest
individual comparator; if it does not, the strongest Tier A/B/C expert is
selected instead. The acceptance gate also compares the counterfactual
no-market path with Tier A/B/C so sparse current odds cannot hide a weak
model-only fallback. It reports calibration, tipping, market comparison,
score/margin behavior, ROI simulations, and competition-policy evidence where
coverage allows. Reports are written under `reports/`.

The checked-in [`reports/eval-latest.json`](../reports/eval-latest.json) records the current 2024–2026 nested holdout summary: 552 pooled non-draw games, 63.4% tipping accuracy, 0.6413 log loss, and 61.2% market-favourite accuracy on covered games. The competition simulation reported about a 35.0% win probability in its configured field scenario. These are historical evaluation results, not a promise about the next round.

Training-time metrics remain useful diagnostics, but the meta-layer has seen the season's OOF rows; do not substitute them for the nested evaluation.

## Decisions downstream

The calibrated prediction becomes:

- a tip and confidence statement;
- expected value `p * decimal_odds - 1` on either side;
- Kelly-derived staking, bounded by configured minimums and maximums;
- a points or competition-win joker recommendation;
- advisory or automatic competition strategy, depending on configuration.

See [Competition strategy](comp-strategy.md) and [Joker strategy](joker-strategy.md) for the stateful decision contracts.

## Limitations

- Historical odds and line coverage are incomplete and time-varying.
- H2H, spread, and totals freshness are tracked per market family; stale
  families are masked rather than silently reused.
- Player identities and old team-list layouts are noisier than match-level IDs.
- A score model cannot fully represent rugby league's discrete scoring and tactical state; dispersion and shared components only soften the assumption.
- `lambda3` may estimate near zero when the evidence does not support a shared component.
- A competition-win policy depends on field size, points gap, opponent behavior, and joker rules; it is scenario-dependent.
- The nrl.com/odds feed replacement is on the runtime path, but source coverage and markup remain operational dependencies; the legacy XML feed is the rollback.

For the research lineage and what has not shipped, see [Research and history](research-and-history.md).
