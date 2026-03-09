# Principled Odds Integration

> **Why we stopped feeding betting odds to LightGBM, and what we do instead.**

This document explains a significant change to how market/bookmaker information flows through the prediction pipeline. It's based on a literature review of probabilistic sports modelling, specifically Egidi et al. (2018), Shin (1993), Koning & Zijm (2022), and Lopez (2017).

---

## The Problem: Naïve Odds Features

Before this change, the pipeline treated bookmaker odds like any other feature — they were normalised, log-transformed, entropy-computed, and fed directly into the Tier-B LightGBM score models alongside 300+ performance and lineup features. The feature block looked like this:

```python
# The old way — odds sitting in the LightGBM feature list
"home_market_prob_basic", "away_market_prob_basic",
"home_market_prob_power", "away_market_prob_power",
"market_overround_h2h", "home_market_logit_basic",
"market_entropy_basic", "market_prob_delta_basic",
"implied_spread_home", "implied_spread_away",
"team_head_to_head_odds_home", "team_line_odds_home",
# ... and more
```

This approach — which the literature calls **naïve odds integration** — has several concrete problems.

### Problem 1: Collinearity and feature domination

Betting markets are extraordinarily good at encoding team quality, injuries, home-ground advantage, and rest. An odds-implied probability already synthesises a huge fraction of the signal in the rest of the feature space. When you give LightGBM both the odds and the performance statistics, the odds consistently dominate tree splits, because they're so correlated with the outcome. The performance features — the ones that could capture *edge over market* — get crowded out.

### Problem 2: Information double-counting

The stacker already takes `logit(market)` as an explicit input alongside `logit(tier_a)` and `logit(tier_b)`. Having odds also in Tier B means the market probability flows into the stacker *twice*: once directly as a stacker input, and once embedded inside the Tier-B output. The stacker can't disentangle these, so it can't correctly learn the relative credibility of the market versus the data-driven signal.

### Problem 3: No principled way to measure edge

If the outcome is "home wins" and the market probability is 0.65, the interesting question is: *does our model have reason to think the true probability is higher or lower than 0.65?* When the odds sit inside the base learner, the model can't answer this cleanly. The market signal and the performance signal are fused, so any disagreement is implicit rather than explicit.

---

## The Architecture Before and After

### Before

```
Raw odds ──────────────────────────────────────────────────────┐
         (20+ derived features: prob, logit, entropy, spread)  │
                                                               ↓
Performance features ──────────────────────────────────► Tier-B LightGBM ──► mu_home, mu_away
Lineup features ───────────────────────────────────────►                         │
                                                                                  │
Tier-A ELO ratings ─────────────────────────────────────────────────────► blend ─┤
                                                                                  │
                                                                                  ↓
Raw odds ──────────────────────────────────────────────► Stacker (logistic)
Tier-A prob ──────────────────────────────────────────►     [tier_a, tier_b, market, missing]
Tier-B prob ──────────────────────────────────────────►         │
                                                                  ↓
                                                           Win probability
```

Odds appear **twice**: inside Tier-B and again at the stacker. The model has no clean separation between "what does the market think" and "what do our performance features say."

### After

```
Performance features ──────────────────────────────────► Tier-B LightGBM ──► mu_home, mu_away
Lineup features ───────────────────────────────────────►    (no odds)            │
                                                                                  │
Tier-A ELO ratings ─────────────────────────────────────────────────────► blend ─┤
                                                                                  │
                                                                                  ↓ tier_b_prob
                                                                                  │
Shin-adjusted odds ────────────────────────────────────► derive market_prob       │
                                                              │                   │
                                                              └──────────────────►│
                                                                                  ↓
                                                           Stacker (logistic):
                                                            [logit(tier_a),
                                                             logit(tier_b),
                                                             logit(market),
                                                             odds_missing,
                                                             logit(tier_a) - logit(market),  ← new
                                                             logit(tier_b) - logit(market)]  ← new
                                                                  │
                                                                  ↓
                                                           Win probability
```

Odds now enter the pipeline **once**, at the stacker level, in a carefully designed form. The base learners produce a pure data-driven signal. The stacker's job is to optimally blend that signal with the market, learning the right weight under cross-validation.

---

## Shin Normalization: Getting a Better Market Probability

Before we can use the market probability, we need to estimate it correctly. The raw implied probabilities from decimal odds overstate probability because bookmakers build margin (overround) into their prices.

The **naïve (basic) approach** is proportional rescaling:

$$
q_i = \frac{1}{o_i}, \quad p_i^{\text{basic}} = \frac{q_i}{q_H + q_A}
$$

where $o_i$ is the decimal odds and $q_i$ is the raw implied probability. This divides out the overround proportionally — simple but unprincipled.

A more sophisticated approach is **power normalization**, which finds a power $k$ such that:

$$
q_H^k + q_A^k = 1
$$

and then sets $p_i^{\text{power}} = q_i^k$. This asymmetrically adjusts for overround in a way that penalises favourites less, which better reflects empirical calibration.

But the most theoretically grounded approach is **Shin normalization** (Shin, 1993). Shin's key insight is that the overround isn't just vigorish — it also reflects the bookmaker pricing against *informed bettors* (punters with private information). He models the market as a mixture:

$$
\text{overround} = \text{vig component} + \text{insider-protection component}
$$

Let $z \in [0, 0.5]$ be the fraction of bettors who are informed. The fair probability under the Shin model is:

$$
p_i^{\text{Shin}} = \frac{\sqrt{z^2 + 4(1-z) \left(\frac{q_i}{\sum_j q_j}\right)^2} - z}{2(1-z)}
$$

The insider-trading parameter $z$ is estimated from the odds themselves using the closed-form discriminant of the overround equation:

$$
z = \frac{W - \sqrt{W^2 - 4(W-1) \cdot \frac{\sum_i q_i^2}{W}}}{2(W-1)}, \quad W = \sum_i q_i
$$

In practice, $z$ is small (typically 0–8% for well-liquid NRL markets) but non-zero, meaning the Shin-adjusted probabilities are slightly more conservative than power-normalised probabilities for strong favourites.

**Why does this matter?** The market probability is now the most principled input to our stacker. If we're computing "model disagrees with market by X log-odds points," we want the cleanest possible market estimate — Shin gives us that.

The pipeline uses a fallback chain: Shin → power → basic → raw odds ratio. In practice, Shin is always available when odds are.

---

## The Disagreement Features: Making Edge Explicit

The most intellectually interesting change is the addition of two new stacker features:

$$
\delta_A = \text{logit}(p_A) - \text{logit}(p_M)
$$
$$
\delta_B = \text{logit}(p_B) - \text{logit}(p_M)
$$

where $p_A$ is Tier-A probability, $p_B$ is Tier-B probability, and $p_M$ is the market (Shin-adjusted) probability. These are **log-odds disagreements** — they directly encode how much our model diverges from the bookmaker's view.

### Why log-odds (logit) and not raw probability differences?

Log-odds have nicer properties for regression. A probability of 0.52 vs 0.48 is barely different in absolute terms, but the teams might be playing very different quality of football — the market is just very uncertain. A probability of 0.92 vs 0.80 is a massive disagreement that the raw difference of 0.12 doesn't fully capture.

The logit transformation maps $(0, 1) \rightarrow (-\infty, +\infty)$ and compresses near-certainty predictions symmetrically:

$$
\text{logit}(p) = \log\frac{p}{1-p}
$$

| $p_{\text{model}}$ | $p_{\text{market}}$ | $\Delta p$ | $\delta = \Delta\text{logit}$ |
|---|---|---|---|
| 0.55 | 0.50 | +0.05 | +0.20 |
| 0.70 | 0.65 | +0.05 | +0.27 |
| 0.90 | 0.85 | +0.05 | +0.64 |

The same raw probability difference carries much more information when both probabilities are extreme.

### What does the stacker learn from these features?

The stacker is trained on historical outcomes using cross-validated out-of-fold predictions. For each game, it sees:

- What did Tier-A think?
- What did Tier-B think?
- What did the market think?
- How much did each model *disagree* with the market?

The disagreement features allow the stacker to learn conditional patterns like:

> *"When Tier-B is more confident than the market (large positive $\delta_B$), that tends to be informative — weight Tier-B more. But when Tier-A disagrees with the market, it's less reliable."*

Or equivalently: the stacker can learn **when to trust the market** and **when to trust the model**. Without explicit disagreement features, this learning is implicit and entangled with the baseline probabilities.

In the residual modelling framing of Koning & Zijm (2022), $\delta_B$ is essentially the **log-odds residual** — what our model says after the market has been accounted for. A positive $\delta_B$ means our model sees value on the home team; a negative $\delta_B$ means value on the away team.

---

## The Stacker as a Super Learner

The final architecture treats the market as a **separate expert** in an ensemble — what the machine learning literature calls a Super Learner (van der Laan et al., 2007).

The stacker fits a logistic regression on six inputs:

```
X = [logit(tier_a),              # ELO-based conditional win probability
     logit(tier_b),              # LightGBM-based conditional win probability
     logit(market_shin),         # Bookmaker market (Shin-adjusted)
     odds_missing,               # Binary flag: no market data available
     logit(tier_a) - logit(mkt), # Tier A value-over-market signal
     logit(tier_b) - logit(mkt)] # Tier B value-over-market signal

y = home_win (binary, non-draw games only)
```

The logistic regression learns coefficients $\boldsymbol{\beta}$ such that:

$$
P(\text{home win}) = \sigma\left(\beta_0 + \beta_A \cdot \text{logit}(p_A) + \beta_B \cdot \text{logit}(p_B) + \beta_M \cdot \text{logit}(p_M) + \beta_\text{miss} \cdot \mathbb{1}_\text{miss} + \beta_{\delta A} \cdot \delta_A + \beta_{\delta B} \cdot \delta_B\right)
$$

**Key things to look for in the logged coefficients:**

1. $\beta_M$ should be the largest positive coefficient — the market is highly informative
2. $\beta_{\delta B} > 0$ means the stacker is learning that Tier-B disagreement is real signal
3. If $\beta_M \gg \beta_A + \beta_B$, the market is dominating and regularisation should be tightened (reduce `C`)
4. $\beta_\text{miss}$ will be negative (when odds are missing, home-team advantage tends to be underestimated)

Every training run now prints these coefficients:

```
Stacker coefficients: tier_a=0.412, tier_b=0.831, market=1.203,
                      odds_missing=-0.094, disagree_tier_a=-0.041, disagree_tier_b=0.287
```

A healthy stacker has market weight visibly larger than the data-driven weights (the market is better-calibrated) but the data-driven weights nonzero (we have real signal beyond the market).

---

## Model vs. Market: The Benchmark

The training run now also computes and prints:

```
Non-draw calibrated log loss (train):  0.6712
Market log loss (benchmark):           0.6804
Model log loss (vs market):            0.6712  (better than market)
```

The **market log-loss** is the irreducible floor for a well-calibrated tipping model. If the closing market had perfect calibration, matching it would be the best we could do without private information. Our model should aim to **beat the market's log-loss on the training set** and ideally also on a forward holdout.

The log-loss comparison uses the same subset of games (non-draw, odds-available), so the comparison is apples-to-apples.

---

## Inference: Value-Over-Market Summary

After each inference run, the pipeline prints a market edge summary:

```
Model vs. market edge summary (threshold ±5%):
  Model favours home stronger than market: 2 game(s)
  Model favours away stronger than market: 1 game(s)
  [HOME] Panthers vs Eels: model=68.3%, market=61.9%, edge=+6.4%
  [AWAY] Storm vs Roosters: model=37.1%, market=44.2%, edge=-7.1%
```

This is the **value-over-market** view. Games where the model and market agree strongly aren't interesting — you're just confirming what the bookmaker already knows. Games where the model diverges by more than 5% are where the data-driven signal has something to say beyond the market consensus.

This doesn't mean you should bet the edge — tipping competitions and betting have different objectives. But it tells you which tips are genuine model calls versus market echoes.

---

## What We Didn't Implement (and Why)

### Poisson Offset

The most theoretically principled extension would be to use the market-implied expected scores as an **offset** in the Tier-B Poisson models:

$$
\text{score}_{\text{home}} \sim \text{Poisson}\left(\exp\left[\log(\lambda_M^{\text{home}}) + \mathbf{x}^\top \boldsymbol{\beta}\right]\right)
$$

where $\log(\lambda_M^{\text{home}})$ is included as an offset (fixed coefficient of 1.0). This forces the baseline to exactly match the market's expected score, with LightGBM explaining only the *deviation* from that baseline.

The challenge: we need $\lambda_M^{\text{home}}$ (expected home points), which requires both a **spread** market and a **total points** market. We have the line/spread in our data but not a total market column. Without the total, we can't separate the spread into $\lambda_{\text{home}}$ and $\lambda_{\text{away}}$ — we only know their difference, not their sum.

If a totals market column is ever added to the feed, this becomes straightforward:

```r
lambda_home_market = (market_total + market_spread) / 2
lambda_away_market = (market_total - market_spread) / 2
log_offset_home = log(lambda_home_market)
log_offset_away = log(lambda_away_market)
```

### Stacker C Tuning

The stacker's L2 regularisation strength (`C = 0.25`) is currently fixed. Cross-validating `C` across `[0.1, 0.25, 0.5, 1.0]` would let the training process decide how much to regularise market weight. Left for a future training run with sufficient time budget.

---

## Summary of Changes

| Component | Before | After |
|---|---|---|
| Tier-B LightGBM features | 300+ features including ~25 odds-derived columns | 300+ features, zero odds columns |
| Market probability estimate | Power normalization (preferred), basic fallback | Shin (1993) adjustment (preferred), power, basic fallback |
| Stacker inputs | `[logit(A), logit(B), logit(M), missing]` | `[logit(A), logit(B), logit(M), missing, δ_A, δ_B]` |
| Market log-loss benchmark | Not computed | Printed every training run |
| Inference edge summary | Not computed | Printed every inference run |

The net effect: the market is used **once**, at the right level of the architecture, in a form that explicitly separates "what does the market say" from "what does our model say beyond the market."

---

## References

- Shin, H.S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *The Economic Journal*, 103(420), 1141–1153.
- Egidi, L., Gabry, J., & Goodrich, B. (2018). Hierarchical models for basketball and soccer. *Journal of Quantitative Analysis in Sports*.
- Koning, R.H. & Zijm, R. (2022). Efficiency in football betting markets. Working paper.
- Lopez, M.J. (2017). Predicting NFL game outcomes using betting market information. Working paper.
- van der Laan, M.J., Polley, E.C., & Hubbard, A.E. (2007). Super Learner. *Statistical Applications in Genetics and Molecular Biology*, 6(1).
