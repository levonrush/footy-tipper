# Literature-review index

These reports are research inputs, not runtime specifications. Some are human-written summaries, some are AI-assisted exports, and several contain recommendations that were later changed, partially implemented, or rejected. The production status below is the useful part.

## Reports

### Deep Research Report on Upgrading an NRL Match Prediction Pipeline

- [PDF](Deep%20Research%20Report%20on%20Upgrading%20an%20NRL%20Match%20Prediction%20Pipeline.pdf)
- [archival Markdown export](deep-research-report.md)
- **Thesis:** combine dynamic team strength, distributional score models, market signals, calibrated stacking, and honest temporal evaluation.
- **Status:** Partial/Shipped mix. Tier A/B, Tier C, stacking, beta calibration, bivariate simulation, dispersion, and nested evaluation shipped. A full dynamic Bayesian attack/defence model and market score offsets did not.
- **Informed:** [Architecture](../docs/how-it-works.md), [Models and evaluation](../docs/modeling-techniques.md), and [Research and history](../docs/research-and-history.md).

The Markdown file is a raw archival export and contains broken internal AI citation tokens. Do not treat those tokens as references; use the repaired primary bibliography in the curated research guide.

### Data acquisition options for historical and upcoming NRL team lists and squad announcements

- [PDF](Data%20acquisition%20options%20for%20historical%20and%20upcoming%20NRL%20team%20lists%20and%20squad%20announcements.pdf)
- [text extract](Data%20acquisition%20options%20for%20historical%20and%20upcoming%20NRL%20team%20lists%20and%20squad%20announcements.txt)
- **Thesis:** official team-list pages can support a polite, versioned historical/current ingestion path when snapshots and parser drift are managed explicitly.
- **Status:** Shipped for nrl.com Team Lists/Late Mail discovery, modern/legacy parsing, versioned snapshots, and repair behavior.
- **Informed:** [Lineup integration](../docs/lineup-integration.md).

### Incorporating NRL Team Lineups into Match Prediction Systems (2012–Present)

- [PDF](Incorporating%20NRL%20Team%20Lineups%20into%20Match%20Prediction%20Systems%20%282012%E2%80%93Present%29.pdf)
- [text extract](Incorporating%20NRL%20Team%20Lineups%20into%20Match%20Prediction%20Systems%20%282012%E2%80%93Present%29.txt)
- **Thesis:** lineup value comes from roles, continuity, cohesion, player strength, snapshot timing, and uncertainty—not a single “star out” flag.
- **Status:** Partial. Versioned as-of lineups, role-group aggregates, continuity/cohesion, churn, and uncertainty marginalization shipped. A richer player-performance model remains future work.
- **Informed:** [Lineup integration](../docs/lineup-integration.md) and [Models and evaluation](../docs/modeling-techniques.md).

### Optimal Joker-Round Selection in Footy Tipping Competitions

- [PDF](Optimal%20Joker-Round%20Selection%20in%20Footy%20Tipping%20Competitions.pdf)
- **Thesis:** joker timing is a sequential decision under uncertain future round quality and relative competition state.
- **Status:** Partial/Shipped. Round opportunity metrics, guardrails, historical simulations, an `auto` policy, and single-use state shipped. The field model remains scenario-based rather than learned from actual rival telemetry.
- **Informed:** [Joker strategy](../docs/joker-strategy.md) and [Competition strategy](../docs/comp-strategy.md).

### Principled Alternatives to Naïve Odds Features

- [PDF](Principled%20Alternatives%20to%20Na%C3%AFve%20Odds%20Features.pdf)
- **Thesis:** isolate the market as a calibrated ensemble signal, measure disagreement explicitly, and avoid counting odds inside both base learner and stacker.
- **Status:** Partial/Shipped. Shin/fallback de-vigging, market separation, disagreements, line features, cross-validated stacker, LOSO calibration, and margin blend shipped. Totals-derived score means and Poisson offsets did not.
- **Informed:** [Principled odds integration](../docs/principled-odds-integration.md).

## Reading order

Start with the [curated research synthesis](../docs/research-and-history.md). Open the source reports when you need the argument, alternatives, or provenance behind a specific production decision. Treat all report citations as candidates to verify against primary sources before quoting them.
