# Principled odds integration

Bookmaker prices are powerful because they summarize public information and trading pressure. That is exactly why they need their own lane: if odds enter a base score model and then enter the stacker again, the market gets two votes while pretending to be two witnesses.

![Odds integration before and after](diagrams/odds-before-after.svg)

[Editable Mermaid source](diagrams/odds-before-after.mmd)

## The old problem

Earlier Tier-B configurations included many raw and derived odds columns alongside football-performance and lineup predictors. The stacker also received a market probability. That caused three problems:

1. Market variables could dominate tree splits and crowd out the residual football signal.
2. The stacker saw market information directly and embedded in Tier B.
3. Model-versus-market edge was entangled rather than measurable.

The old architecture is genuine history. It is not the current predictor contract.

## Current separation

Tier B excludes market columns through the predictor configuration. Odds are transformed only for the market/meta path:

- head-to-head odds -> fair conditional probability;
- line prices/amounts -> line cover, overround, implied spread, and disagreement features;
- missing odds -> a separately trained no-market pool (or validated Tier-B fallback);
- offered prices -> downstream expected-value and staking decisions.

The probability layer uses two explicit pools. The market pool combines Tier A,
Tier B, Tier C, and a genuine H2H market probability. The no-market pool
combines only Tier A/B/C, so a missing price is never represented as a
fictitious 50% market opinion.

## Removing the overround

For decimal odds `o_i`, raw implied probabilities are `q_i = 1/o_i` and normally sum above one. The pipeline prefers a Shin adjustment, then falls back through power/basic normalization and ultimately a two-price ratio when necessary.

Shin's model treats some overround as protection against informed trading. It is useful here not because every NRL book is a textbook market, but because the ensemble needs a reproducible, de-vigged market input. See [Shin's original Economic Journal article](https://academic.oup.com/ej/article-abstract/103/420/1141/5157258).

## Constrained pooling

Let `p_A`, `p_B`, `p_C`, and `p_M` be the conditional home-win
probabilities. Each pool combines their logits with nonnegative weights that
sum to one and no intercept:

```text
logit(p_pool) = w_A logit(p_A) + w_B logit(p_B)
              + w_C logit(p_C) [+ w_M logit(p_M)]
```

This makes each expert monotone, preserves 50% when every expert is neutral,
and prevents the pool from reversing a unanimous set of inputs. A
positive-temperature, no-intercept calibrator preserves those same
properties.

## Current stacker inputs

The market pool uses:

- `logit(tier_a)`
- `logit(tier_b)`
- `logit(tier_c)`
- `logit(market)`

The learned market pool is retained only when fully nested season-out
predictions show material log-loss improvement over the strongest individual
comparator on the same market-covered rows, remain within the accuracy and
Brier tolerances, and pass the recent-season stability checks. The raw market is
included in that demanding comparison. If the gate rejects the learned weights,
the production fallback is a one-hot selection of the strongest Tier A/B/C
expert, with a matching direction-preserving calibrator. If nested evidence is
unavailable, Tier B is the safe default. Neither case becomes a raw market-only
winner tip.

On current data the gate selects Tier C alone and the market carries zero
weight. That was checked rather than assumed: a shrinkage path between the
learned weights and the one-hot fallback was built and tested under three
selection rules, and every rule that admitted market weight produced a model
worse out of sample than Tier C on both tipping accuracy and competition
placement. The market's own tipping accuracy fell from about 71% before 2024 to
about 62% after, while the rival field still tips it, so borrowing from the
market now costs accuracy and separation from the field at the same time. The
detail, including the numbers, is in `docs/modeling-techniques.md`; the code is
recoverable from history if the market regains its edge.

Competition placement is recorded in the evaluation report for the deployed
model and every expert, but it does not select.

The no-market pool uses the first three inputs and is trained
counterfactually by masking market data on every OOF row. It is selected only
when nested season-out log loss beats Tier B; otherwise the manifest declares
Tier B as the no-market fallback. Both calibrators are fitted to LOSO pool
predictions.

This is close in spirit to the [Super Learner ensemble](https://doi.org/10.2202/1544-6115.1309): use honest base predictions and learn a constrained combination. It is not a formal claim that the implementation satisfies every Super Learner theorem.

## Line market and margin

The handicap and totals markets are separate score forecasts. The current
system:

- fits a small ridge margin blend from honest model margin, market spread, and Tier-A margin;
- accepts a line or total only with a complete pair of valid decimal prices;
- adjusts the expected home/away score means before simulation while
  preserving the blended total;
- persists the displayed margin as home score minus away score.

Winner probability remains calibrated separately, and winner, scoreline, and
margin are one coherent public prediction.

## Value and staking

After inference, the decision layer compares the calibrated probability with the offered decimal price:

```text
expected_value = probability * decimal_odds - 1
```

`probability` here is two-way, `win / (win + lose)`, matching the two-way
decimal quote it is compared against. `predictions_table` stores a three-way
triple, so the stored win probability carries a `(1 - draw_prob)` factor;
pricing against that understated every edge by the draw mass, which is roughly
the size of the edge threshold itself. Every published number, in the email,
on the site, in the CLI and in staking, now uses the two-way convention, which
is also the quantity the model is calibrated and scored on.

Positive edge must clear the configured threshold. A Kelly-derived fraction is then reduced and bounded by minimum/maximum stake controls; `normalized` mode reports fractions, while `bankroll` mode can also report `stake_amount`.

No calibration method turns a positive estimated edge into guaranteed profit. Coverage, price timing, limits, and model error still matter.

## What is not implemented

A full residual score model with bookmaker means as learned Poisson offsets is
not implemented. The current manifest blends are deliberately smaller:

```text
lambda_home_market = (T + M) / 2
lambda_away_market = (T - M) / 2
```

They nudge prediction-time means only when complete live markets exist; they do
not train Tier B against later closing information.

## Evaluation rules

- Compare model and market on the same non-draw, odds-covered rows.
- Retain a learned market pool only when nested season-out results prove
  incremental value over the strongest individual comparator and remain
  stable in recent held-out seasons; if rejected, select the strongest Tier
  A/B/C model expert, never the raw market alone.
- Counterfactually mask odds on every held-out row and gate the no-market path
  against the strongest Tier A/B/C expert.
- Prefer nested season-out metrics over training-time summaries.
- Treat opening, prediction-time, and closing prices as different information sets.
- Do not train a 24-hour-before-kickoff process against closing information without an explicit leakage analysis.
- Report both probability quality and decision outcomes; ROI alone is noisy and selection-dependent.

The current checked-in nested report is summarized in [Models and evaluation](modeling-techniques.md#honest-evaluation).

## Primary references

- H. S. Shin, [“Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims”](https://academic.oup.com/ej/article-abstract/103/420/1141/5157258), *The Economic Journal* 103(420), 1993.
- R. H. Koning and Renske Zijm, [“Betting Market Efficiency and Prediction in Binary Choice Models”](https://doi.org/10.1007/s10479-022-04722-3), *Annals of Operations Research* 325, 2023 (published online 2022).
- M. J. van der Laan, E. C. Polley, and A. E. Hubbard, [“Super Learner”](https://doi.org/10.2202/1544-6115.1309), *Statistical Applications in Genetics and Molecular Biology* 6(1), 2007.

The previously listed “Lopez (2017), Predicting NFL game outcomes using betting market information” citation could not be verified and has been removed. The Egidi and Gabry player-performance paper is relevant to hierarchical sports modelling, not evidence for this market-integration architecture; it is indexed correctly in [Research and history](research-and-history.md#primary-references).
