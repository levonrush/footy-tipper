# Research and history

Footy Tipper grew in the usual dignified fashion: build a model, discover leakage, automate it, discover a new class of leakage, write a character called Reg R-ai-gan, then eventually insist that every research idea declare whether it actually shipped.

This page is the curated bridge between formal reports, historical notebooks, production code, and the Medium story. Raw research exports remain archival; broken internal citation tokens in them are not usable references.

## Research synthesis

### Formal model and evaluation

The central research direction was a tiered probabilistic system: a stable sequential team-strength prior, richer score regressors, a direct outcome classifier, market-aware stacking, calibration, and a coherent score distribution. Production now implements Tier A/B/C, learned score blends, a regularized stacker, beta calibration, bivariate-Poisson dependence, negative-binomial overdispersion when estimable, and nested season-out evaluation.

The biggest deliberate non-implementation is a full dynamic hierarchical Bayesian attack/defence model. Tier A captures sequential state and carryover but does not claim that richer latent-state structure.

### Lineups

The lineup reports argued that ingestion must preserve versions and decision time, then translate raw names into role strength, continuity, cohesion, churn, and uncertainty. Production versions official articles, repairs old zero-entry snapshots, selects training lineups as-of a pre-kickoff cutoff, and marginalizes Tier-B probability over learned selection uncertainty.

Player-performance ratings sourced from match-centre history remain incomplete. Current role-group strength uses the history available inside the existing match/lineup pipeline.

### Odds and market information

The odds review rejected naïve market variables inside Tier B. Production treats the market as a separate expert: de-vigged head-to-head probabilities, line-market inputs, explicit model/market disagreement, cross-validated stacking, LOSO calibration, and a line-aware margin blend. A totals market and market-implied score offsets remain unimplemented.

### Joker and competition strategy

The joker work reframed “best round” as a sequential, stateful decision. Production scores round opportunity, applies coverage/separation guardrails, backtests scenario policies, and writes joker use only after a successful live send. The competition layer separately simulates limited deviations that maximize estimated competition-win probability while preserving canonical model tips.

### Feed migration

The migration research found plausible nrl.com draw and match-centre replacements, a derivable ladder, richer player/team data, and historical/live odds options. Prototype modules on `feed-migration` target the existing cache schemas. They are **not connected to the CLI or R preparation**, so this remains next-state work rather than the system described in present tense.

## Research -> production matrix

| Research proposition | Status | Production evidence or next gate |
| --- | --- | --- |
| Sequential Tier-A team-strength baseline | Shipped | `tier_a_baseline.py`; tuned with past-season-only rules in evaluation |
| Separate home/away Tier-B score models | Shipped | `home_model.pkl`, `away_model.pkl` |
| Direct binary winner model | Shipped | Tier C OOF signal and `binary_model.pkl` |
| Learned Tier A/B score blend | Shipped | manifest home/away weights |
| Market isolated from Tier-B predictors | Shipped | predictor filter plus meta-layer market inputs |
| Shin/fallback de-vigging | Shipped | fair-market probability path |
| Model/market and line disagreement | Shipped | stacker meta-features |
| Cross-validated regularized stacker | Shipped | version-aware `stacker.pkl` |
| Beta calibration on LOSO stack predictions | Shipped | `win_prob_calibrator.pkl`; logged fallback for sparse season groups |
| Bivariate score dependence | Shipped | estimated `lambda3`, possibly near zero when unsupported |
| Negative-binomial overdispersion | Shipped | manifest side dispersions with Poisson fallback |
| Line-aware margin blend | Shipped | optional manifest ridge coefficients; simulation fallback without lines |
| Versioned team-list snapshots and repair | Shipped | lineup snapshot/entry/run tables and zero-entry repair |
| As-of lineup selection | Shipped | default 24-hour historical cutoff |
| Role/continuity/cohesion/churn features | Shipped | shared lineup feature builder |
| Lineup uncertainty marginalization | Shipped | deterministic per-game Monte Carlo |
| Player match-performance ratings | Partial | match-centre research/prototype exists; full production feature path does not |
| Joker opportunity, guardrails, and single-use ledger | Shipped | `joker_policy.json`, `joker_usage` |
| Competition-win deviation search | Shipped | advisory default; audit table; model predictions unchanged |
| Dynamic hierarchical Bayesian attack/defence | Exploratory | formal research only; current Tier A is simpler |
| Market totals and Poisson score offsets | Not implemented | requires dependable prediction-time totals contract |
| nrl.com draw/match-centre feed replacement | Partial | prototype modules exist; no CLI/R/Actions wiring or completed shadow cutover |
| Referee, weather, travel, and rest expansion | Exploratory | candidate sources/features; train/infer symmetry and evaluation still required |

## Evidence discipline

- Use [`reports/eval-latest.json`](../reports/eval-latest.json) for the latest checked-in nested evaluation, not a Medium-era headline.
- Use `Final` rows for training and `Pre Game` rows for inference.
- Match prediction-time information sets in historical tests; closing odds and final lineups can leak.
- Treat the AI-assisted reports as research maps. Verify citations and claims at the primary source before relying on them.
- Keep “Shipped,” “Partial,” “Exploratory,” and “Not implemented” attached to claims that could otherwise be mistaken for architecture.

## Primary references

- H. S. Shin, [“Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims”](https://academic.oup.com/ej/article-abstract/103/420/1141/5157258), *The Economic Journal* 103(420), 1993.
- R. H. Koning and Renske Zijm, [“Betting Market Efficiency and Prediction in Binary Choice Models”](https://doi.org/10.1007/s10479-022-04722-3), *Annals of Operations Research* 325, 2023 (published online 2022).
- M. J. van der Laan, E. C. Polley, and A. E. Hubbard, [“Super Learner”](https://doi.org/10.2202/1544-6115.1309), 2007.
- L. Egidi and J. Gabry, [“Bayesian Hierarchical Models for Predicting Individual Performance in Soccer”](https://doi.org/10.1515/jqas-2017-0066), *Journal of Quantitative Analysis in Sports*, 2018. This supports hierarchical player-performance modelling; it is not cited as an odds-combination paper.

The previously listed Lopez citation could not be verified and has been removed rather than upgraded by confidence alone.

## The Medium series

These essays are the narrative record, not the runtime specification. The unrelated surfboat article is intentionally excluded.

1. [Origin and problem framing](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292) — turns pub-comp frustration into a data-science problem and establishes the first end-to-end ambition.
2. [Preseason baseline, Elo, and Round 1 lessons](https://medium.com/@levonrush/the-footy-tipper-preseason-model-training-and-the-countdown-to-round-1-b0400407f50) — builds the early baseline, explores Elo and feature ideas, then meets the small-sample cruelty of a new season.
3. [Leakage, rolling validation, SMEs, and research-led design](https://medium.com/@levonrush/the-footy-tipper-3-early-season-woes-smes-research-lead-design-and-the-double-diamond-approach-62052f4b631a) — diagnoses early-season failure, brings domain experts into the loop, and shifts toward temporal validation and deliberate discovery.
4. [Automation and engineering a maintainable product](https://medium.com/@levonrush/the-footy-tipper-4-to-err-is-human-to-automate-divine-25a29661d4e4) — moves repetitive weekly work into a pipeline and confronts the difference between a notebook and an operated product.
5. [The R, Python, and SQLite rebuild](https://medium.com/@levonrush/the-footy-tipper-5-refining-the-set-play-11aeb0023af9) — reorganizes responsibilities across languages and a transparent database so preparation, modelling, and delivery can evolve independently.
6. [Model selection, tuning, feature selection, and validation](https://medium.com/@levonrush/the-footy-tipper-6-a-rugby-league-coachs-guide-to-advanced-modelling-6e5142571378) — treats modelling like coaching: choose a structure, tune it, cut passengers, and measure it on games it has not seen.
7. [Reg R-ai-gan and usable email delivery](https://medium.com/@levonrush/the-footy-tipper-7-taking-ai-too-far-3adf4164384b) — gives the delivery layer a character and tests how far generative copy can go before usefulness leaves the building.
8. [Tech debt, the off-season rebuild, and research-to-production discipline](https://medium.com/@levonrush/the-footy-tipper-8-grand-final-heartbreak-the-off-season-rebuild-and-another-long-season-ahead-f16535abf3a2) — uses the offseason to repay shortcuts, improve evaluation, and demand a production answer from research recommendations.
9. [Product maturity, Agile practice, and knowing when to stop](https://medium.com/@levonrush/the-footy-tipper-9-shifting-goal-posts-b966add1cb2d) — reflects on changing definitions of done, iterative delivery, and the discipline of ending a feature hunt.
10. [Epilogue: lessons about data-science practice and leadership](https://medium.com/@levonrush/the-footy-tipper-epilogue-mad-monday-df4c5043840f) — extracts lessons about teams, communication, judgment, and the work around the model.
11. [Reprise: agent-led research, planning, and rebuilding](https://medium.com/@levonrush/the-footy-tipper-reprise-changing-the-game-d4629064fb1f) — returns with agents as collaborators for auditing, research synthesis, implementation planning, and another round of rebuilding.

For current behavior, return to the [documentation map](README.md). For the untouched experiments, see [Research notebooks](../research/README.md) and the [literature-review index](../lit-review/README.md).
