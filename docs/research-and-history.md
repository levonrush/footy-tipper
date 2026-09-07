# Research and history

Footy Tipper grew in the usual dignified fashion: build a model, discover leakage, automate it, discover a new class of leakage, write a character called Reg R-ai-gan, then eventually insist that every research idea declare whether it actually shipped.

This page is the curated bridge between formal reports, historical notebooks, production code, and the Medium story. Raw research exports remain archival; broken internal citation tokens in them are not usable references.

## Research synthesis

### Formal model and evaluation

The central research direction was a tiered probabilistic system: a stable sequential team-strength prior, richer score regressors, a direct outcome classifier, market-aware pooling, calibration, and a coherent score distribution. Production now implements Tier A/B/C, learned score blends, separate constrained market/no-market pools, direction-preserving calibration, bivariate-Poisson dependence, negative-binomial overdispersion when estimable, and nested season-out evaluation.

The biggest deliberate non-implementation is a full dynamic hierarchical Bayesian attack/defence model. Tier A captures sequential state and carryover but does not claim that richer latent-state structure.

### Lineups

The lineup reports argued that ingestion must preserve versions and decision time, then translate raw names into role strength, continuity, cohesion, churn, and uncertainty. Production versions official articles, repairs old zero-entry snapshots, selects training lineups as-of a pre-kickoff cutoff, and marginalizes Tier-B probability over learned selection uncertainty.

Player-performance ratings sourced from match-centre history remain incomplete. Current role-group strength uses the history available inside the existing match/lineup pipeline.

### Odds and market information

The odds review rejected naïve market variables inside Tier B. Production
treats genuine H2H prices as a separate expert in a constrained market pool,
uses a separately validated no-market path when prices are absent, applies
direction-preserving LOSO calibration, and uses valid line/total markets only
to nudge pre-simulation score means.

### Joker and competition strategy

The joker work reframed “best round” as a sequential, stateful decision. Production scores round opportunity, applies coverage/separation guardrails, backtests scenario policies, and writes joker use only after a successful live send. The competition layer separately simulates limited deviations that maximize estimated competition-win probability while preserving canonical model tips.

Both layers are scoped to a competition that runs for the regular season only, so both switch off for the finals. That boundary was implicit until the finals work made it explicit; see below.

### Finals

The comps stop in September and the football does not, so the finals run as special editions. The reframe is that the two competition-aware layers have nothing left to optimise, and the slot they occupy is better spent on the question a knockout series can actually answer: who wins the premiership.

Production simulates the remaining bracket from the ladder seeds, pricing drawn fixtures with the calibrated model and undrawn ones from Tier-A ratings. The ratings needed calibrating first: raw Tier-A probabilities are severely overconfident, so a bracket simulated on them would have reported a minor premier as an overwhelming favourite. The bracket rules are pinned against three real seasons rather than asserted. See [Finals special edition](finals-edition.md).

### Feed migration

The migration research identified nrl.com draw/match-centre replacements, a derivable ladder/performance path, and historical/live odds sources. That work shipped on `main` in PR #34: Python refreshes the sources into the existing cache schemas before R preparation, with `FOOTY_TIPPER_FEED_SOURCE=feed` retaining the XML rollback. Richer player identity features and totals-based score offsets remain separate follow-on work.

### Club Context

The missing concept was not news sentiment but **psychosocial match context**: a confirmed acute leadership, serious-human, tribute/milestone, or club-crisis event with a plausible pathway to motivation, cohesion, distraction, tactics, or variance. The product name is Club Context.

Research on football coach changes warns against the intuitive story. A 24-study systematic review found mixed short-term results, a 331-dismissal matched study found no detectable treatment effect, and earlier control-group work attributes much of the apparent bounce to regression to the mean. Those results do not settle the NRL question; they determine the default: sign-neutral features, matched and nested evaluation, and no production effect until prospective evidence earns one.

The implemented shadow boundary is a rights-aware event registry, immutable round-cutoff snapshots, a shared shadow feature transformer, materiality reporting, and a sourced Context Watch card that explicitly does not alter the probability. The first model-era ablation paired 3,180 held-out games, including 89 context-exposed games: its small event-cohort log-loss improvement was uncertain, aggregate probability quality was slightly worse, and the matched margin and dispersion checks were null. The recorded decision is therefore to remain shadow-only and collect a prospective season. See [Club Context](club-context.md) and [Source policy](source-policy.md) for effect sizes and limitations.

## Research -> production matrix

| Research proposition | Status | Production evidence or next gate |
| --- | --- | --- |
| Sequential Tier-A team-strength baseline | Shipped | `tier_a_baseline.py`; tuned with past-season-only rules in evaluation |
| Separate home/away Tier-B score models | Shipped | `home_model.pkl`, `away_model.pkl` |
| Direct binary winner model | Shipped | Tier C OOF signal and `binary_model.pkl` |
| Learned Tier A/B score blend | Shipped | manifest home/away weights |
| Market isolated from Tier-B predictors | Shipped | predictor filter plus meta-layer market inputs |
| Shin/fallback de-vigging | Shipped | fair-market probability path |
| Constrained market/no-market pooling | Shipped | version-aware market and no-market stacker artifacts |
| Direction-preserving LOSO calibration | Shipped | market and no-market calibrator artifacts |
| Bivariate score dependence | Shipped | estimated `lambda3`, possibly near zero when unsupported |
| Negative-binomial overdispersion | Shipped | manifest side dispersions with Poisson fallback |
| Line-aware margin blend | Shipped | optional manifest ridge coefficients; simulation fallback without lines |
| Versioned team-list snapshots and repair | Shipped | lineup snapshot/entry/run tables and zero-entry repair |
| As-of lineup selection | Shipped | default 24-hour historical cutoff |
| Role/continuity/cohesion/churn features | Shipped | shared lineup feature builder |
| Lineup uncertainty marginalization | Shipped | deterministic per-game Monte Carlo |
| Player match-performance ratings | Partial | match-centre ingestion is production; identity-linked rating features are not |
| Joker opportunity, guardrails, and single-use ledger | Shipped | `joker_policy.json`, `joker_usage`; suppressed for the finals |
| Competition-win deviation search | Shipped | advisory default; audit table; model predictions unchanged; suppressed for the finals |
| Finals round-stage classification | Shipped | `rounds.py`; shared by ingestion, prediction, and delivery |
| Premiership bracket simulation | Shipped | `premiership.py`; bracket rules pinned against 2023-2025; fails soft to no section |
| Calibrated Tier-A ratings for undrawn matchups | Shipped | Platt scaling fitted from the ratings walk; Brier 0.274 to 0.225 on 2015 onward |
| Simulated-distribution summaries | Shipped | `prediction_distributions`; margin bands and line/totals cover probabilities, read from the existing simulation |
| Joker playable in the final priced round | Not implemented | `min_rounds_with_odds` forces HOLD, so an unused joker expires worthless |
| Finals-aware ladder rate features | Not implemented | `wins / round_id` deflates rate features in finals; consistent train/serve, needs a retrain |
| Dynamic hierarchical Bayesian attack/defence | Exploratory | formal research only; current Tier A is simpler |
| Full bookmaker-offset residual score model | Not implemented | current valid-market blends only nudge prediction-time means |
| nrl.com draw/match-centre feed replacement | Shipped | Python ingestion runs before R prep; parity evidence checked in; XML retained as rollback |
| Referee, weather, travel, and rest expansion | Exploratory | candidate sources/features; train/infer symmetry and evaluation still required |
| Evidence-gated Club Context registry and cutoff snapshots | Shadow | reviewed facts/provenance can be frozen and surfaced; cannot change production probabilities |
| Club Context score/winner features | Shadow candidate | sign-neutral shared transformer; requires paired nested season-out and prospective evidence |
| Generic news sentiment, raw article text, or embeddings | Rejected | too noisy, hard to interpret, rights-sensitive, and unnecessary for the event hypothesis |

## Evidence discipline

- Use [`reports/eval-latest.json`](../reports/eval-latest.json) for the latest checked-in nested evaluation, not a Medium-era headline.
- Use `Final` rows for training and `Pre Game` rows for inference.
- Match prediction-time information sets in historical tests; closing odds and final lineups can leak.
- Treat the AI-assisted reports as research maps. Verify citations and claims at the primary source before relying on them.
- Keep “Shipped,” “Shadow,” “Partial,” “Exploratory,” “Rejected,” and “Not implemented” attached to claims that could otherwise be mistaken for architecture.

## Primary references

- H. S. Shin, [“Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims”](https://academic.oup.com/ej/article-abstract/103/420/1141/5157258), *The Economic Journal* 103(420), 1993.
- R. H. Koning and Renske Zijm, [“Betting Market Efficiency and Prediction in Binary Choice Models”](https://doi.org/10.1007/s10479-022-04722-3), *Annals of Operations Research* 325, 2023 (published online 2022).
- M. J. van der Laan, E. C. Polley, and A. E. Hubbard, [“Super Learner”](https://doi.org/10.2202/1544-6115.1309), 2007.
- L. Egidi and J. Gabry, [“Bayesian Hierarchical Models for Predicting Individual Performance in Soccer”](https://doi.org/10.1515/jqas-2017-0066), *Journal of Quantitative Analysis in Sports*, 2018. This supports hierarchical player-performance modelling; it is not cited as an odds-combination paper.
- E. Lundkvist et al., [“The sacking illusion: A counterfactual analysis of mid-season coaching changes using points and expected points in European football”](https://doi.org/10.1080/02640414.2026.2698238), *Journal of Sports Sciences*, 2026.
- H. Sousa et al., [“Effects of changing the head coach on soccer team's performance: A systematic review”](https://doi.org/10.5114/biolsport.2024.131816), *Biology of Sport* 41(2), 2024.
- J. C. van Ours and M. A. van Tuijl, [“In-Season Head-Coach Dismissals and the Performance of Professional Football Teams”](https://ideas.repec.org/a/bla/ecinqu/v54y2016i1p591-604.html), *Economic Inquiry* 54(1), 2016.

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
