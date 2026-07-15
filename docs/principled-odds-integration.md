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
- missing odds -> explicit indicator and neutral fallback;
- offered prices -> downstream expected-value and staking decisions.

The stacker combines Tier A, Tier B, Tier C, and market signals. The market appears as a named expert rather than a hidden ingredient.

## Removing the overround

For decimal odds `o_i`, raw implied probabilities are `q_i = 1/o_i` and normally sum above one. The pipeline prefers a Shin adjustment, then falls back through power/basic normalization and ultimately a two-price ratio when necessary.

Shin's model treats some overround as protection against informed trading. It is useful here not because every NRL book is a textbook market, but because the ensemble needs a reproducible, de-vigged market input. See [Shin's original Economic Journal article](https://academic.oup.com/ej/article-abstract/103/420/1141/5157258).

## Explicit disagreement

Let `p_A`, `p_B`, and `p_M` be Tier-A, Tier-B, and market conditional home-win probabilities. The stacker receives logit differences such as:

```text
delta_A = logit(p_A) - logit(p_M)
delta_B = logit(p_B) - logit(p_M)
```

These features say how far the football models disagree with the market on a symmetric log-odds scale. The stacker can learn whether particular disagreement patterns historically contained signal rather than assuming every difference is an edge.

## Current stacker inputs

The version-aware regularized logistic stacker can use:

- `logit(tier_a)`
- `logit(tier_b)`
- `logit(tier_c)`
- `logit(market)`
- `odds_missing`
- Tier A/market and Tier B/market disagreement
- line cover probability
- line overround
- model-versus-implied-spread disagreement

The regularization value `C` is cross-validated over the implementation grid; it is no longer the fixed future-work item described by the older document. Tier inputs used to fit the stacker are OOF where coverage permits, and the beta calibrator is fitted to LOSO stacker predictions when enough seasons exist.

This is close in spirit to the [Super Learner ensemble](https://doi.org/10.2202/1544-6115.1309): use honest base predictions and learn a constrained combination. It is not a formal claim that the implementation satisfies every Super Learner theorem.

## Line market and margin

The handicap market is a separate margin forecast. The current system:

- adds line-derived features to the win-probability stacker;
- fits a small ridge margin blend from honest model margin, market spread, and Tier-A margin;
- uses that blend only for games with a finite line;
- falls back to the simulated margin without line coverage.

Winner probability remains calibrated separately, and the displayed scoreline is made coherent with the chosen winner.

## Value and staking

After inference, the decision layer compares the calibrated probability with the offered decimal price:

```text
expected_value = probability * decimal_odds - 1
```

Positive edge must clear the configured threshold. A Kelly-derived fraction is then reduced and bounded by minimum/maximum stake controls; `normalized` mode reports fractions, while `bankroll` mode can also report `stake_amount`.

No calibration method turns a positive estimated edge into guaranteed profit. Coverage, price timing, limits, and model error still matter.

## What is not implemented

The most coherent score-market extension needs both spread and total markets. If expected market total `T` and expected home margin `M` are available, market score means can be approximated by:

```text
lambda_home_market = (T + M) / 2
lambda_away_market = (T - M) / 2
```

Those means could become Poisson offsets, leaving Tier B to model residual score strength. The production feed has line inputs but not a dependable totals contract, so market totals and score offsets remain **not implemented**. The feed-migration prototype identifies historical totals as a possible future source; it is not wired to preparation or inference.

## Evaluation rules

- Compare model and market on the same non-draw, odds-covered rows.
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
