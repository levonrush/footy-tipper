# Club Context: research, evidence, and shadow design

> **Status: SHADOW ONLY.** Club Context can collect reviewed events, freeze the facts available at prediction time, build sign-neutral candidate features, and surface an experimental Context Watch card. It does **not** alter the production probability, tip, margin, scoreline, value pick, stake, or joker decision. The model-era retrospective evaluation found no clear out-of-sample lift, so predictive materiality remains unestablished.

**Club Context** is the reader-facing name. **Psychosocial match context** is the research term: a confirmed, acute off-field event that could plausibly affect motivation, cohesion, distraction, leadership, tactics, or performance variance around a particular game.

It is not a weekly sentiment score. “The vibes are good” is not a datum, and a dramatic headline is not an effect estimate. The layer records what was verifiably known, keeps the direction of any effect unknown, and asks the model evaluation whether narrowly defined event cohorts contain useful signal.

![Club Context evidence, cutoff, shadow evaluation, reader safety, and provenance](diagrams/club-context.svg)

[Editable Mermaid source](diagrams/club-context.mmd)

## Why this needs a separate layer

Match, ladder, market, and lineup inputs describe most of the observable football state. They do not describe an interim coach taking charge, a confirmed club sanction, or a formal tribute match. Those events are real information, but three traps make them dangerous predictors:

1. **Story bias:** the memorable win after an event is easier to recall than the loss or ordinary performance.
2. **Regression to the mean:** an underperforming club commonly changes its coach after a bad run, when results were already likely to recover.
3. **Information leakage:** the most persuasive “they played for him” evidence is often a post-match interview published after the outcome.

The architecture therefore separates discovery, reviewed facts, prediction-time snapshots, model evaluation, and reader copy. An event can be worth mentioning without being allowed to move a probability.

## What the research says

The clearest evidence concerns head-coach changes in association football. It is useful prior evidence, not proof that the same effect transports to the NRL.

- Sousa et al.'s [2024 systematic review](https://doi.org/10.5114/biolsport.2024.131816) included 24 studies and found mixed short-term results; changing the head coach did not guarantee improved performance.
- Lundkvist et al.'s [2026 matched study of 331 mid-season dismissals](https://pubmed.ncbi.nlm.nih.gov/42421473/) compared clubs with near-identical recent performance trajectories. It found no detectable improvement in points or expected points over the following ten matches relative to controls.
- Van Ours and Van Tuijl's [14-season Dutch study](https://ideas.repec.org/a/bla/ecinqu/v54y2016i1p591-604.html) found that changed and unchanged control clubs both improved after the trigger point, consistent with regression to the mean rather than a universal “new coach bounce.”

The correct prior is therefore **no universal signed effect**. A particular transition might matter through tactics, selection, or leadership, but the system must learn that from leakage-safe NRL evidence rather than assigning an automatic boost.

### The Jai Arrow sequence

The 2026 South Sydney sequence motivated the broader question, but it is hypothesis-generating rather than causal proof. Souths lost immediately after the retirement announcement in the [24 May match](https://www.abc.net.au/news/2026-05-24/nrl-live-blog-cameron-munster-origin-training-cowboys-rabbitohs/106715988), then won 48–6 during the formal tribute match, described in the NRL's [post-match report](https://www.nrl.com/news/2026/06/12/the-ultimate-team-mate-rabbitohs-pay-tribute-to-arrow-after-emotion-charged-win/). Opponent absences, team lists, underlying form, venue, rest, and market expectations remain confounders.

Those moments are separate event phases, not one persistent “MND boost.” The post-match article may help a researcher validate that a tribute occurred; its language can never become a feature for the match it describes.

## Event taxonomy

Only acute, identifiable events enter the registry. Routine commentary, generic morale, transfer rumours, training-ground speculation, and article sentiment do not.

| Category | Included examples | Important phase distinctions |
| --- | --- | --- |
| Leadership change | Dismissal, resignation, interim appointment, planned succession, temporary leave | Announcement, effective transition, first match, resolution |
| Serious human event | Confirmed serious illness, bereavement, or comparable disruption | Announcement, ongoing period, formal return or resolution |
| Tribute, farewell, or milestone | A club-confirmed tribute match, farewell, or special appearance | Announcement and the actual match pulse are separate |
| Club crisis, sanction, or governance disruption | Confirmed sanction, insolvency-scale disruption, board or governance crisis | Announcement, effective date, ongoing phase, resolution |

Category and phase are versioned. When an event evolves, the new phase is a new registry event with its own known/effective/expiry times. It does not rewrite the old observation.

## Evidence gate

Discovery is intentionally wider than eligibility. ABC RSS and GDELT may identify candidates, but neither makes an event true. An event becomes eligible only when all of these conditions hold:

- its review status is approved;
- it is confirmed, not rumour or unconfirmed reporting;
- confidence meets the current minimum threshold;
- the taxonomy version is current;
- it was known by the applicable round decision cutoff;
- its effective window covers the match; and
- it has either an official NRL/club confirmation or two reputable, genuinely independent sources whose use is permitted.

Two syndicated copies from the same publisher or wire are one source, not two. Missing publication time, ambiguous entity mapping, unresolved rights, or a failed classifier leaves the candidate in review and out of features and reader output.

The registry stores factual summaries and links, not article bodies. See [Source and content-use policy](source-policy.md) for the rights statuses, adapter rules, and monetisation gate.

The discovery implementation uses ordinary RSS/JSON HTTP requests from Python's standard library. It requires no Chrome, headless browser, Playwright, Selenium, or local browser installation.

## Time and leakage contract

Club Context freezes one information set per round.

| Run | Decision time |
| --- | --- |
| Historical training/evaluation | `11:00 Australia/Sydney` on the local calendar day of the round's earliest fixture, shared by every match in that round |
| Live, test, refresh, or preview | One actual `decision_at_utc`, captured after ingestion and immediately before feature construction |

The distinction matters for a Thursday-to-Sunday round. News first published on Friday cannot affect Sunday's historical feature row because the whole round was frozen on Thursday morning. That conservative rule mirrors the product's weekly decision and prevents later matches receiving information the sent email did not have.

Every prediction run writes a new immutable context snapshot. A later refresh may create a newer run; it cannot update or delete the context tied to the earlier run. `known_at_utc` is the earliest supportable availability time, while `effective_from_utc` and `expires_at_utc` describe the event window. The source record also preserves whether `known_at_utc` came from a publisher timestamp, first observation, or manual verification.

Post-match reporting can validate a research label for a future audit. It can never be moved backward across the decision cutoff or presented as contemporaneous evidence.

## Persisted contracts

The Club Context tables are additive siblings of the prediction tables:

| Table | Contract |
| --- | --- |
| `context_article_snapshots` | Canonical URL, publisher, publication/first-observed times, content hash, acquisition method, extraction version, and rights status; no article body |
| `context_events` | Category, phase, known/effective/expiry times, confidence, salience, sensitivity, factual summary, confirmation, and review state |
| `context_event_sources` | Evidence role, source independence, official/reputable flags, and the article link supporting an event |
| `context_event_entities` | Team, player, coach, or club relationship and normalized team mapping |
| `context_ingestion_runs` | Mode, adapter outcomes, counts, status, and coverage diagnostics |
| `context_prediction_runs` | One immutable decision time, season/round/mode, context hash, and schema/taxonomy/feature versions |
| `prediction_context` | Immutable game-event links, observed facts, source provenance, and the exact sign-neutral feature snapshot |

Context tables never enter `prediction_table.sql`. A missing or invalid context table must leave the published tips contract intact.

## Sign-neutral feature design

The shared transformer is used by training, inference, and evaluation. It emits the same compact measurements for home and away sides plus their differences:

- category and phase counts;
- recency decay and games since the event;
- official confirmation and independent-source diversity;
- maximum confidence and salience;
- uncertainty and sensitivity indicators; and
- explicit data-available/missing flags.

The vector contains no predetermined positive/negative effect. It also contains no generic sentiment, raw text, language-model embedding, post-match quote, or generated copy. Home-minus-away differences let a fitted model determine direction from historical evidence without encoding “emotion equals extra points.”

Club Context is a candidate feature family for the existing Tier-B score models and Tier-C binary classifier. It is not another stacking expert. Leadership or tribute cohorts may separately justify a bounded change to simulated residual dispersion, but only if held-out evidence shows a repeatable variance effect.

## Shadow evaluation and materiality

The production stack and a context-feature candidate are fitted on identically seeded, expanding season-out folds. With no input file, the command below generates both candidates, restricts comparison to their shared held-out rows, and then scores the pair. `--context-input` is an optional reproducibility path for rescoring an existing paired CSV/JSON file. Neither path can activate a model or modify a production artifact.

```bash
footy-tipper advanced model evaluate --context-ablation
```

The default report paths are:

- `reports/club-context-shadow-predictions-latest.csv`
- `reports/club-context-materiality-latest.json`
- `reports/club-context-materiality-latest.md`

An explicitly supplied empty paired file produces `not_ready`, not a simulated win for the feature. The nested run reports:

- log loss, Brier score, calibration, and accuracy;
- score and margin error plus residual variance when those outputs exist;
- tip flips, including flips from correct to wrong and vice versa;
- event-linked and all-game cohorts;
- category and time-since-event cohorts;
- results with and without market inputs;
- event-cluster confidence intervals with a fixed seed; and
- event-study pre-trends and placebo checks when relative-match data exists.

The underlying research set has two deliberate parts: a complete model-era census of NRL mid-season coaching changes, and a manually audited registry of identifiable serious-event, tribute, and crisis cases. Automated prospective collection measures source coverage for every category; it does not silently turn discovery hits into labels.

Market odds are a particularly important comparator because public news may already be priced. The report must state how much apparent signal disappears after odds and lineup changes are included. A feature that only repeats the market may add reader context without adding model skill.

### Retrospective result — 7 September 2026

The audited registry contains 32 effective in-season first-grade coaching handovers from 2008–2026, plus manually reviewed serious-human, tribute/milestone, and club-crisis cases. The seeded expanding-season evaluation produced 3,180 paired held-out games from 2011–2026. Club Context was active in 89 games across 36 event clusters (2.8% of the paired sample); 81 of those games were linked to leadership changes. The other category samples are only two or three games each and cannot support category-specific conclusions.

All deltas below are context candidate minus baseline; negative log-loss, Brier, and error deltas are better.

| Cohort | Games | Δ log loss | Δ Brier | Δ accuracy | Tip flips | Δ margin MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All held-out games | 3,180 | +0.00011 | +0.000078 | -0.25 points | 88 (40 to correct, 48 to wrong) | -0.011 points |
| Context-exposed games | 89 | -0.00725 | -0.00314 | -3.37 points | 5 (1 to correct, 4 to wrong) | -0.004 points |

The event-cluster bootstrap put the event-cohort log-loss delta's 95% interval at **-0.01911 to +0.00486**, crossing zero. Its estimated probability of improvement was 87.8%, short of the predeclared 95% evidentiary threshold. Removing market inputs produced almost the same event-cohort delta (-0.00722), so this sample does not show the apparent signal being absorbed by prices; it also does not establish an independent effect. Historical market coverage is nearly complete and the constrained stack often selected a football expert instead of a learned market blend, so this comparison should be read cautiously.

Score dispersion did not move materially: the context-to-baseline event-cohort residual-variance ratio was **1.001**. There is therefore no evidence for even a bounded simulation-dispersion adjustment.

The research-only event study tells the same cautionary story. Across 37 event matches, the affected side performed 0.77 margin points worse than its three nearest controls after orienting the nested baseline residual to that side; the event-cluster 95% interval was **-6.38 to +5.14 points**, and the permutation placebo p-value was 0.804. The raw pre-event residual was substantially negative, consistent with events—especially sackings—being selected after underperformance and with subsequent mean reversion. This is not evidence for a universal new-coach bounce.

The Jai Arrow rows illustrate why individual stories remain useful but insufficient. For the announcement-period match at North Queensland, the shadow candidate moved the Cowboys from 49.1% to 50.5% and happened to correct the tip. For the formal Souths tribute match, it moved Souths from 71.6% to 68.1% and did not change the correct Souths tip. One success and one probability reduction do not identify a stable motivational effect.

The all-game candidate can differ even on a row with no active event because adding predictors refits LightGBM and changes the learned trees. That is why the report shows both aggregate harm and the sparse event cohort, and why event-linked flips—not cherry-picked global flips—are the relevant examples.

**Decision:** retain shadow mode. The small event-cohort probability signal is uncertain, aggregate probability quality is slightly worse, tip flips skewed harmful, and neither the matched study nor dispersion test shows material benefit. Prospective collection remains worthwhile because the rare non-leadership categories are underpowered.

### Activation gate

The current implementation has no activation switch. A future model release may include Club Context only after a separate, explicit decision documents all of the following:

1. leakage-safe historical event-cohort improvement with uncertainty reported;
2. no material harm to overall calibration;
3. stable direction after market and lineup controls;
4. a full prospective season confirming the historical direction;
5. adequate source coverage and train/infer feature parity; and
6. completed rights review for the intended deployment, especially a monetised one.

Passing one memorable-match case, in-sample fit, or aggregate accuracy comparison is not enough.

## Explainability and the reader product

The product keeps two statements distinct:

- **Observed context:** a verified event existed, with a factual summary and source.
- **Model impact:** a fitted, measured contribution changed the prediction.

During shadow mode only the first statement is available. An eligible event may produce a sourced **Context Watch** card in the email and site, carrying the fixed disclosure that it is experimental and does not alter the displayed probability. No event, missing source data, or a context failure renders the existing no-context email/site path unchanged.

Reg remains the narrator, but prose is built only from locked facts and passes a deterministic safety layer. The layer prohibits:

- causal claims or certainty that a team is “playing for” someone;
- diagnosis speculation;
- treating illness, death, bereavement, or trauma as a betting edge;
- jokes or banter about a sensitive human event; and
- using a sensitive event to prompt banner imagery.

Validation failure falls back to deterministic factual copy. Once a future release activates the feature family, context may appear in `why_line` only when its real TreeSHAP contribution clears the normal explanation threshold. An observed event is not itself a model explanation.

## Operator workflow

```bash
# Current discovery into the review queue; fail-soft unless --strict is chosen.
footy-tipper advanced data context refresh

# Intentional import/repair of the audited historical registry.
footy-tipper advanced data context backfill

# Read-only schema, eligibility, coverage, and provenance checks.
footy-tipper advanced data context validate

# Generate and score paired baseline/candidate season-out rows; never activate production.
footy-tipper advanced model evaluate --context-ablation
```

Normal weekly prediction must survive a discovery outage, rights-disabled adapter, sparse coverage, or invalid event. Strict mode exists for diagnosing ingestion; it is not the production default.

`update-model`, advanced inference, and hosted prediction idempotently import the checked-in reviewed catalogue before discovery. This bootstraps a fresh SQLite runtime and picks up newly reviewed entries without promoting discovery headlines automatically.

## Acceptance and test invariants

The test suite protects the boundaries that matter most:

- canonical-URL deduplication, timestamps, entities, phases, confidence, and independent-source rules;
- source outage and rights-disabled adapter behavior;
- the Arrow announcement and later tribute as distinct pulses;
- post-match “played for him” reporting excluded from pre-match features;
- news after the round cutoff excluded even from a later weekend fixture;
- train/infer/evaluate feature parity and seeded reproducibility;
- test-mode non-mutation and immutable sent snapshots;
- fail-soft prediction and byte-identical no-context rendering; and
- prohibited Reg language plus deterministic safe fallback.

The materiality report remains publishable even if the result is “no predictive value.” That negative is a product result: the Context Watch card can still explain the week without pretending the story earned a probability adjustment.

## Research and policy references

- E. Lundkvist et al., [“The sacking illusion: A counterfactual analysis of mid-season coaching changes using points and expected points in European football”](https://doi.org/10.1080/02640414.2026.2698238), *Journal of Sports Sciences*, 2026.
- H. Sousa et al., [“Effects of changing the head coach on soccer team's performance: A systematic review”](https://doi.org/10.5114/biolsport.2024.131816), *Biology of Sport* 41(2), 2024.
- J. C. van Ours and M. A. van Tuijl, [“In-Season Head-Coach Dismissals and the Performance of Professional Football Teams”](https://ideas.repec.org/a/bla/ecinqu/v54y2016i1p591-604.html), *Economic Inquiry* 54(1), 2016.
- [NRL Terms of Use](https://www.nrl.com/terms-of-use).
- [ABC Terms of Use](https://www.abc.net.au/conditions.htm).
- [GDELT Terms of Use](https://www.gdeltproject.org/about.html#termsofuse).
