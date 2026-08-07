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

Pool weights are nonnegative, sum to one, and have no intercept. The learned
market pool is retained only when fully nested season-out predictions provide
material log-loss improvement over the strongest individual comparator on the
same covered rows, stay within the accuracy and Brier tolerances, and pass
recent-season stability checks. The raw market participates in that comparison,
but a rejected pool becomes a one-hot selection of the strongest Tier A/B/C
model expert rather than a raw market-only winner forecast. Tier B is the safe
default when nested evidence is unavailable. The manifest records
`learned_weights` for both regimes, so a rejected pool can still be inspected
for what it had learned.

On current data that gate selects Tier C alone, giving the market zero weight,
and a run of experiments in August 2026 established that this is the right
answer rather than an artifact of the gate's shape.

### A shrinkage path, tried and withdrawn

The gate's failure mode looks wrong on inspection: "the learned mixture did not
clearly beat the best expert" is a weaker claim than "the best expert alone is
the right model", and collapsing straight to a corner of the simplex treats
them as the same. The comparison also used to be asymmetric, judging the pool
against a bar that included the market while drawing the fallback from a set
that excluded it, so a narrow miss against the market shipped a third choice
worse than either.

The fix tried was a shrinkage path, `normalize(s * learned + (1 - s) *
onehot(fallback))` for `s` in `{0, 0.25, 0.5, 0.75, 1}`, with the bar and the
fallback made the same expert. Three walk-forward runs rejected it:

| rung objective | walk-forward outcome |
| --- | --- |
| mean per-season P(finish first) | acceptance FAIL; one fold took the full pool, 7 tips worse |
| P(first) plus a recent-window guard | selects zero shrinkage in every fold |
| log loss, parsimonious | acceptance FAIL; 8 tips worse, 0.23 worse on P(first) |

Every rule that admitted market weight produced a model worse out of sample
than Tier C alone, on tipping accuracy and on competition placement together.
The cause is regime change rather than noise. Nested evidence is dominated by
2010 to 2023, when the market tipped about 71%; from 2024 it tips about 62%,
and the rival field tips the market, so borrowing from it costs both accuracy
and separation from the field at once. Held out over 2024 to 2026: Tier C 381
of 587 tips with P(first) 0.564, the market 368 of 587 with P(first) 0.0005.

The path, the asymmetry fix and the competition scorer are recoverable from
history around commits `079f999` through `b8b3d02`. They are worth revisiting
if the market regains its edge; they are not worth carrying while it has not.

### Competition placement as a scoreboard

`comp_placement_metrics` scores any forecaster against a simulated rival field
that tips the market favourite, and the evaluation report carries it for the
deployed model and every expert. It does not select anything.

Two properties of it are worth knowing. Within one realized season, maximising
P(first) is exactly equivalent to maximising tips correct, because rival scores
do not depend on our tips. Deviating from the field for its own sake buys
nothing at selection time, and the in-season case for it lives in the
competition-strategy layer, which knows the live points gap. Across seasons it
is not the same as pooled accuracy: P(first) saturates, so clearing the field
in a strong season is worth more than the same average accuracy spread evenly.
Seasons are therefore scored separately and averaged, never pooled.

Those two facts together produce a result worth stating plainly, because it is
easy to mistake for a bug. The rival field tips the market favourite, so the
field is strong in exactly the seasons the market is strong. A model that
tracks the market therefore banks its extra tips in seasons that were
unwinnable anyway, and gives them back in the seasons it could have won. On
2010 to 2026 the raw market wins more tips than Tier C in twelve of seventeen
seasons and still scores worse on mean P(first), 0.058 against 0.150, because
Tier C's good seasons are ones where it clears a weakened field. No
within-season monotonicity is broken by this; the whole effect is composition
across seasons.

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

The checked-in [`reports/eval-latest.json`](../reports/eval-latest.json) records the 2024 to 2026 nested holdout summary: 580 pooled non-draw games, 64.5% tipping accuracy, 0.6327 log loss, 0.2213 Brier, and 62.3% market-favourite accuracy on covered games. Margin MAE is 14.13 against 14.27 for the market line. These are historical evaluation results, not a promise about the next round.

### Margin distribution

The evaluation also scores the *distribution* the simulator draws from, not just
its point margin, using CRPS, randomised PIT, and coverage reported with width
(`distributional_metrics.py`). Randomisation in the PIT is required because
margins are integers; the plain transform is non-uniform even under a correctly
specified model. Over-dispersion and the shared component are refitted on prior
seasons only, so held-out seasons stay honest.

Comparators are deliberately strong rather than a floor. On the 2024 to 2026
holdout the pooled CRPS is:

| Method | CRPS | 50% cov | 90% cov | 90% width |
|---|---|---|---|---|
| Normal approximation | 10.20 | 0.50 | 0.87 | 55.8 |
| Empirical replay of past errors | 10.22 | 0.48 | 0.88 | 56.9 |
| Score model (Poisson family) | 10.25 | 0.47 | 0.84 | 51.4 |
| Score model reconciled to the calibrated probability | 10.50 | 0.48 | 0.83 | 51.3 |

Two honest negatives follow. The 100k-draw Poisson-family simulator does not
beat a two-parameter normal approximation fitted to prior-season residuals, so
the negative-binomial dispersion and shared component are not currently earning
their complexity on this metric. And reconciling the score means onto the
calibrated win probability, which is what ships, *costs* CRPS and calibration:
the win-probability stack and the score model disagree, and the price of making
them agree is now measured rather than assumed. All four methods under-cover at
the 90% level.

The market line appears as a point forecast, where CRPS reduces to MAE (14.27),
so it is not a like-for-like comparison against a distribution and should not be
read as one.

### The displayed scoreline

The three integers that reach `predictions_table` are not measured by the distribution
table above, so `margin_distribution.reconciliation` in the report measures them
directly. Two switches on `simulate_game` decide how they are produced, and the report
scores every combination on identical per-game seeds, plus the importance-reweighting
that predates both:

- `reconcile`: `"on_conflict"` moves the score means onto the calibrated probability
  only where the score model would otherwise put the other side in front;
  `"always"` moves them on every game.
- `display`: `"median"` takes the median margin and the median total, each the
  MAE-optimal estimate of its own quantity, and splits the total around the margin;
  `"mode"` takes the most common exact scoreline on the tipped side.

| How the three integers are produced | Margin MAE | Home MAE | Away MAE | Total MAE | CRPS |
|---|---|---|---|---|---|
| Solve on conflict, median (**deployed**) | 14.12 | 9.25 | 8.76 | 11.36 | 10.23 |
| Solve on conflict, modal scoreline | 14.24 | 10.26 | 9.61 | 14.78 | 10.23 |
| Reweighted, modal scoreline | 14.31 | 10.28 | 9.60 | 14.74 | 10.19 |
| Solve every game, modal scoreline | 14.46 | 10.37 | 9.69 | 14.92 | 10.50 |
| Solve every game, median | 14.49 | 9.40 | 8.93 | 11.41 | 10.50 |

Both defaults follow that table rather than preference. The reasoning behind each:

A mode of a two-dimensional discrete distribution carries heavy sampling noise, and
`simulate_game` already returned `median_margin` while the display ignored it. Reading
the median instead is worth about three and a half points of total MAE and a point of
per-side score MAE.

The requirement the scoreline has to meet is that it never contradicts the tip, and on
most games the score model already satisfies it. The calibrated probability is Tier C,
which models no scores at all, so reconciling a game that was already coherent hands the
scoreline to a model that has never predicted one. Margin MAE reflects that directly:
14.13 unreconciled, 14.12 solving only on conflict, 14.49 solving every game.

Non-contradiction is enforced explicitly rather than inferred: a median margin of zero,
or one whose sign disagrees with the tip, is pushed a single point onto the tipped side.
Both cases only arise inside the near-tie band.

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
- On the current holdout the score model's CRPS does not beat a normal
  approximation, so the Poisson-family machinery is not yet demonstrably worth
  its complexity for distributional accuracy. It is retained because the
  simulation also supplies the scoreline and draw probability, which a margin
  distribution alone does not give.
- A competition-win policy depends on field size, points gap, opponent behavior, and joker rules; it is scenario-dependent.
- The nrl.com/odds feed replacement is on the runtime path, but source coverage and markup remain operational dependencies; the legacy XML feed is the rollback.

For the research lineage and what has not shipped, see [Research and history](research-and-history.md).
