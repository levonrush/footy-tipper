# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- A second hosted scheduling clock using Google Apps Script, a guarded
  `watchdog=true` workflow input, repository-scoped token boundary, automatic
  failure issue, dedicated operations runbook, and dual-clock Mermaid diagram.
- Distributional scoring for the margin (`pipeline/common/model_training/distributional_metrics.py`): CRPS on ensemble samples, randomised PIT for the discrete outcome, and coverage returned welded to interval width. Wired into the nested evaluation and persisted in `reports/eval-*.json`, scored against a normal approximation, an empirical replay of past errors, and the market line. Nothing previously scored the score distribution, so `lambda3`, the negative-binomial dispersion, and the market score blends were unfalsifiable.
- `research/phd-methods-transfer.ipynb` documenting which PhD methods were ported, which were rejected, and the results, generated with embedded outputs by `research/build_phd_transfer_notebook.py`.
- Scoring for the *displayed* scoreline, which no metric reached before: the evaluation now replays the importance-reweighting the constraint-native solve replaced, on identical per-game seeds, and reports margin, per-side score, and total MAE plus a seed-matched margin CRPS for both. Recorded under `margin_distribution.reconciliation`. Supported by `crps_weighted_ensemble`, which scores a weighted ensemble on the same scale as an unweighted one.
- Per-prediction explainability (`pipeline/common/explain/`, PR #44). Every tip now carries the exact decision chain (expert probabilities, simplex pool weights, temperature, consensus guard, and the score-mean path) plus TreeSHAP attribution from LightGBM's native `pred_contrib`, grouped into feature families. Stored in `prediction_explanations`, a sibling of `predictions_table` rather than extra columns on it, so a broken explanations table costs a sentence rather than a send. Surfaced through `footy-tipper advanced explain round|cohort|report`, a one-line `why` in the email and on the site's tip card, and out-of-fold capture via `advanced model evaluate --explain`. `FOOTY_TIPPER_EXPLAIN=false` skips the write. See [docs/explainability.md](docs/explainability.md).
- `FOOTY_TIPPER_TRAINING_SEED` (default `20100308`), which seeds `BayesSearchCV` and every LightGBM fit. Training was previously unseeded, so two runs on identical data produced different models: repeated runs scored 63.1% and 64.4% tipping accuracy, one failing the release acceptance gate and one passing it. Two trains now produce identical metrics, an identical manifest, and byte-identical booster trees. Recorded as `training_seed` in the manifest. Never A/B two feature sets on unseeded runs; the comparison measures search noise rather than the change.
- `FOOTY_TIPPER_NAN_PASSTHROUGH` (default off), sending missing values to LightGBM as `NaN` instead of `0.0` so the booster can learn an explicit missing branch. Resolved when the pipeline is built and baked into the fitted transformer, so a NaN-trained model cannot silently revert to zero-fill at serve time. It targets the performance-stat era gap; team-list features are zero-filled upstream and unaffected.

### Changed
- Scheduled delivery now uses targeted off-boundary GitHub polls plus
  independent DST-aware Google recovery slots. Both clocks ask the same
  Drive-backed gate, and existing concurrency, marker, ledger, odds, and SMTP
  protections remain authoritative.
- The calibrated win probability and the simulated scoreline are now one object. Score means are solved along a total-preserving ray so the simulated distribution's own win probability is the calibrated one, replacing the post-hoc importance reweighting and removing the mirroring fallback for a side the simulation could not produce.
- The displayed scoreline is the median of the simulation, not its mode. `simulate_game` already returned `median_margin` and the display discarded it in favour of the most common exact scoreline, which is a high-variance statistic on a two-dimensional discrete distribution. The new `pf.scoreline_from_samples` splits the median total around the median margin, and pushes a zero or wrong-signed margin one point onto the tipped side so the scoreline can never contradict the tip. Worth 3.4 points of total MAE and about a point of per-side score MAE on the 2024 to 2026 holdout.
- Reconciliation now runs only where it is needed. `simulate_game(reconcile="on_conflict")` moves the score means onto the calibrated probability only where the score model would otherwise put the other side in front; the requirement is non-contradiction, and most games already satisfy it. Reconciling regardless handed every scoreline to Tier C, which models no scores, and cost 0.37 points of margin MAE for nothing.
- Both of those defaults are chosen on measurement, not preference. The nested evaluation scores all four combinations of the two switches plus the legacy reweighting on identical per-game seeds, and the report names the deployed one, so it cannot describe a configuration it did not measure. Net against the arrangement predating this work: margin MAE 14.31 to 14.12, home 10.28 to 9.25, away 9.60 to 8.76, total 14.74 to 11.36, CRPS 10.19 to 10.23.
- `lambda3` and the per-side negative-binomial dispersion now compose. A non-zero shared component previously discarded the dispersion outright; the dispersion is rescaled so the marginal variance is preserved while the shared component carries the covariance.
- `train.py` and `evaluate.py` print a summary panel through the shared reporter. Their stdout is captured by the parent, so a long run previously finished showing only a tick with the numbers buried in the CLI log.

### Fixed
- Raw `crowd` was a declared predictor, but attendance is only known after kickoff: populated on 3,538 of 3,593 training rows and 0 of 8 inference rows, then zero-filled by `to_df`. Every training row taught the model something no serving row carries. Removed; `venue_avg_crowd`, a trailing per-venue mean over prior games, was already in the list and covers the intent.
- Target leak in the ladder form columns for 2018 to 2024. The legacy XML feed stored end-of-season values in `recent_form`, `season_form`, `current_streak`, `day_record`, `night_record` and `players_used` on every round row, and those columns are declared predictors, so a round-2 row read how the season finished. Distinct values per team-season ran about 1.4 across roughly 12.5 home games in that window against about 7 either side. `refresh` only ever rewrites the current season, so the frozen values survived into `training_data`. New `advanced data nrl rebuild-ladders` re-derives `feed_cache_ladders` as-of-round from data already in SQLite, with no network. Validated by rebuilding 2026, which the live path had already built from real bye rows: identical across all 33 columns. `feed_cache_performance` is deliberately not rebuilt, since 46 of its columns are non-derivable from the modern match centre.
- Fold-level optimism in the nested evaluation. The expanding-window OOF loops transformed every fold through the preprocessor fitted on the whole corpus, so a fold predicting season Y was encoded by an encoder that had already seen Y's categories. Fold weights were honest; the encoding was not. Each fold now fits its own preprocessor, and TreeSHAP capture realigns on that fold's one-hot widths. Tier-A base rates are pinned to pre-holdout seasons for the same reason its alpha/carryover grid already was. The report now records holdout discipline per component, including what is *not* held out: hyperparameters and predictor selection still come from a full-corpus search, because re-running a 100-candidate `BayesSearchCV` inside every fold is not affordable for a routine evaluation.
- `subsample` was a live dimension in the Bayesian search but had no effect. LightGBM ignores `bagging_fraction` unless `bagging_freq` is greater than zero, and it defaults to zero; verified by fitting two models differing only in `subsample` and getting identical predictions. One of eight search dimensions was inert and the manifest's reported `subsample` was meaningless. `subsample_freq` is now searched alongside it, with `0` reproducing the previous behaviour.
- The lineup as-of cutoff compared a venue-local kickoff against true-UTC article publish times. `start_time` is venue-local wall clock serialised as-if-UTC while `source_published_at_utc` is true UTC, so the documented 24 hour leakage guard was really running at about 13 to 14 hours for Australian venues, and `lineup_source_age_hours` was inflated by the same offset. Now measured from `start_time_utc`.
- `FOOTY_TIPPER_RSCRIPT` pins the R interpreter used for data preparation. Bare `Rscript` resolves through PATH, so an activated conda env carrying its own R shadows the one `R_LIBS_USER` was built against. Package shared objects are named `.so` by a CRAN build and `.dylib` by a conda build and are tied to the R minor version, so the mismatch surfaces as a missing shared object rather than as a version error.
- Finals team lists were being dropped. `parse_round_id` only matched a numeric "Round N", so "Finals Week 1" and "Grand Final" articles stored `round_id = NULL` and every entry on them was discarded. The round is now resolved from `round_name` per season, since finals round numbers depend on how many regular rounds the season had. Team-list coverage goes from 1,133 to 1,168 training games, completing 2022 and 2023.
- Lineup continuity across coverage gaps. The history accumulators skipped uncovered games rather than resetting, so a 2018 game's "retained since last week" could be computed against a 2012 lineup. Every fixture is now tracked and the carried lineup is dropped when any were skipped.
- The three duplicated post-merge lineup fill blocks in `train.py`, `inference.py`, and `evaluate.py` had drifted, and all three filled `lineup_features_missing` with `0.0`, announcing complete team lists for a game the builder never saw. Collapsed into one helper defaulting to `1.0`. `evaluate.py` also no longer swallows a failed lineup merge silently.
- Train/serve skew in the probability meta-layer. `train.py` computed the lineup-marginalised Tier-B probability and discarded it, fitting the pools and calibrator on the unmarginalised value while inference served the marginalised one; `evaluate.py` had the same gap. Both now marginalise, via a vectorised path that reproduces the scalar values exactly.
- `install.R` no longer breaks a working R library. It now tests whether each package loads rather than whether it is listed, installs only what fails, and re-verifies afterwards. Previously an opportunistic upgrade could remove and fail to rebuild `dplyr`, aborting data preparation even though the restored library was fine.

## [1.0.0] - 2026-07-22

### Documentation
- Rewrote the operator documentation for the 1.0 guided CLI, exact advanced toolbox, one-command model update, immutable releases, and Drive-backed delivery marker.
- Added the operator command hierarchy diagram and refreshed production/operations diagrams for the split model/runtime state contract.
- Audited repository Markdown against the current eleven-command CLI, R preparation modes, SQL contracts, Tier A/B/C model path, line markets, LOSO calibration, dispersion fallback, state sync, and Actions workflows.
- Added task-oriented documentation, research/literature indexes, a curated research-to-production matrix, the complete eleven-part Medium series, and canonical Mermaid diagram sources with SVG previews.
- Corrected the runtime start-year default to `2010`, the test recipient fallback to `levon_rush@hotmail.com`, and provider ownership: Claude/Anthropic writes optional email copy while OpenAI generates an optional banner.
- Reconciled the feed documentation with the shipped cutover: Python nrl.com/odds ingestion is production and the credentialled XML feed is the rollback path.
- Documented local-authoritative model training, Drive publication, Actions prediction-only ownership, the 11:00 Sydney gate, and the safe local training runbook.
- Added a private linked Notion project hub; repository Markdown remains canonical.

### Changed
- Replaced the eleven pipeline-shaped top-level commands with `status`, `setup`, `tips`, `update-model`, and `advanced`. Retired names now fail with an exact replacement instead of forwarding.
- Everyday `tips` commands dispatch and wait for exact GitHub Actions modes; manual live delivery requires typed `SEND ROUND N` confirmation.
- Production model publication is now a resumable `update-model` transaction. It trains into staging on local hardware, validates and re-downloads an immutable release, dispatches an exact-release check in the GitHub Actions production image, activates a pointer, and requests a no-email refresh without disabling Actions or requiring local Docker.
- Split mutable Actions runtime DB/schedule synchronization from immutable model publication. In-flight prediction jobs cannot upload or overwrite models.

#### CLI 1.0 migration map

| Retired top-level shape | 1.0 replacement |
| --- | --- |
| `footy-tipper prep` | `footy-tipper advanced data prepare all` |
| `footy-tipper train` | `footy-tipper update-model` for production; `footy-tipper advanced model train` for an isolated technical run |
| `footy-tipper infer` | `footy-tipper advanced model infer` |
| `footy-tipper send` | `footy-tipper advanced delivery preview|test|live` |
| `footy-tipper predict` | `footy-tipper advanced local-run preview|test|live` |
| `footy-tipper lineups` | `footy-tipper advanced data lineups refresh|backfill` |
| `footy-tipper nrl-data` | `footy-tipper advanced data nrl refresh|backfill|validate` |
| `footy-tipper odds` | `footy-tipper advanced data odds refresh|backfill` |
| `footy-tipper site` | `footy-tipper advanced site build|publish` |
| `footy-tipper evaluate` | `footy-tipper advanced model evaluate` |
| `footy-tipper state` | `footy-tipper advanced cloud pull-runtime|push-runtime|schedule|gate` |
- Retired hosted GitHub Actions training. Local training again uses the 100-candidate default, with Bayesian search parallelism outside single-threaded LightGBM fits.
- Scheduled prediction now polls every 15 minutes and becomes eligible at 11:00 `Australia/Sydney` on the first-game day of the next unsent round.
- Actions test, refresh, and live prediction modes all disable auto-training and fail clearly when published artifacts are missing.

### Reliability / Hardening
- Added schema-versioned status output, plain-language failures, secret redaction, stable exit codes, and an opt-in `--debug` traceback.
- Added a Drive-backed pending/sent/uncertain delivery marker so SMTP success followed by runner/state-push failure cannot cause an automatic duplicate.
- Actions uses a separate exact-allowlist machine runner; unknown workflow modes fail and no wildcard can default to live.
- Model releases include a last-written training receipt with provenance, runtime versions, training range, sizes, and hashes. Release objects are create-only and verified after re-download before pointer activation.
- Local model updates now hold an exclusive process lock, terminate/reap the trainer on interruption, and revalidate stage and active-pointer evidence before resuming.
- Every human production send, including advanced live aliases, routes through the single serialized Actions workflow and binds manual authorization to the exact outgoing round.
- Production delivery validates credentials and freezes the recipient envelope before claiming a pending marker; partial SMTP refusal leaves the round pending/uncertain instead of risking an automatic duplicate.
- Explicit activation and rollback rerun the selected release in the hosted production image; a malformed pointer is archived before a confirmed repair.
- Model publication validates the required home, away, and manifest artifacts before uploading Drive state.
- Drive pulls stage and validate model archives before replacing the local last-known-good artifact set.

## [0.1.0] - 2026-02-18

### Added
- Added `CHANGELOG.md` and `AGENTS.md` for project history and agent/operator guidance.
- Added first-class CLI entrypoint (`footy-tipper`) with subcommands:
  - `prep`, `train`, `infer`, `send`, `predict`
- Added packaging entrypoint (`pyproject.toml`) so `footy-tipper` is available as an installed command via `pip install -e .`.
- Added dynamic season controls (the original start-year default was `2018`; it is now `2010`):
  - `FOOTY_TIPPER_START_YEAR`
  - `FOOTY_TIPPER_END_YEAR` (defaults to current calendar year)
  - `FOOTY_TIPPER_INCLUDE_PERFORMANCE` (default `true`)
- Added test-send workflow with single-recipient mode (the fallback address has since changed).
- Added the original dedicated `CLI.md` documentation (later removed in favour of `docs/cli-reference.md`).
- Added the original provider-backed email generation and fallback path (copy generation has since moved to Claude; OpenAI now handles the optional banner).
- Added safe no-op behavior for Google Drive/Sheets actions when dependencies or credentials are missing.
- Added joker-round recommendation logic (market-odds based) with configurable strategy:
  - `FOOTY_TIPPER_JOKER_STRATEGY` (`auto`, `points`, `protect`, `chase`)
  - `FOOTY_TIPPER_JOKER_RISK_LAMBDA` (for protect objective)
- Added joker decision guardrails so `PLAY` is only triggered with enough priced-round coverage and a minimum signal gap:
  - `FOOTY_TIPPER_JOKER_MIN_ROUNDS_WITH_ODDS`
  - `FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE`
  - `FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO`
- Added training-time Monte Carlo joker backtest artifact (`models/joker_policy.json`) and automatic strategy selection from that policy for send runs.
- Added policy state controls:
  - `FOOTY_TIPPER_JOKER_POINTS_GAP`
  - `FOOTY_TIPPER_JOKER_POLICY_PATH`
- Added joker summary block in send emails (plain + HTML) and joker context injection into Reg/OpenAI copy prompts.
- Added `pipeline/common/sql/joker_round_candidates.sql` for upcoming-round joker scoring inputs.
- Added joker-round unit tests (`tests/test_joker_rounds.py`) and joker-policy training tests (`tests/test_joker_policy_training.py`).
- Added a new docs hub under `docs/` with focused pages for:
  - getting started
  - CLI reference
  - architecture/how-it-works
  - modeling techniques
  - joker strategy
  - operations/reliability
- Added lineup ingestion pipeline (`pipeline/lineups.py`) and lineup feature modules (`pipeline/common/lineups/*`) to capture NRL Team Lists/Late Mail snapshots into SQLite (`lineup_article_snapshots`, `lineup_entries`).
- Added lineup-aware predictor blocks (availability, spine/bench composition, continuity, freshness, and home-away deltas) to training/inference.
- Added lineup uncertainty features (expected named/spine/interchange counts + selection uncertainty) and Monte Carlo marginalization of Tier-B conditional win probabilities.
- Added dedicated lineup integration docs (`docs/lineup-integration.md`) and lineup config examples in `secrets.env.example`.
- Added lineup parsing/feature unit tests:
  - `tests/test_lineup_ingest_parsing.py`
  - `tests/test_lineup_features.py`

### Changed
- Season year span is now automatic instead of hardcoded to 2025 (`pipeline/common/data-prep/data_config.R`).
- Data prep bootstrap no longer depends on a machine-specific local `setwd`; it resolves project root from script path (`pipeline/data-prep.R`).
- Training/inference split is now explicit by `game_state_name` (`Final` / `Pre Game`) with validation (`pipeline/common/data-prep/data-pipeline.R`).
- Inference set now selects the earliest pre-game round from the latest available competition year.
- Prediction query now scopes to latest pre-game season before choosing min round (`pipeline/common/sql/prediction_table.sql`).
- `send.py` now exits cleanly when there are no pre-game fixtures; no offseason crash path.
- Email subject now includes both round and competition year (`pipeline/send.py`).
- Predict workflow can now run end-to-end from one CLI command (`footy-tipper predict`).
- Python/Conda OpenAI dependency range updated to v1-compatible (`requirements.txt`, `environment.yml`).
- Conda Python version aligned with Docker Python 3.11 (`environment.yml`).
- README redesigned to be lightweight and operator-friendly, with deep technical content moved to `docs/`.
- Legacy docs `CLI.md` and `cli/README.md` acted as pointers during the transition and were later removed.
- CLI now includes lineup orchestration controls:
  - new `lineups` command
  - lineup refresh runs before `prep`, `train`, `infer`, and `predict` by default
  - new flags: `--skip-lineups`, `--lineups-mode`, `--lineups-max-articles`, `--lineups-include-sitemap-in-recent`, `--lineups-strict`
- `train` now auto-bootstraps a historical lineup backfill when the configured training window has not been backfilled yet; auto-train from `infer`/`predict` inherits the same behavior unless lineups are explicitly skipped.
- `infer` and `predict` now auto-run training when required model artifacts are missing (unless `--skip-auto-train` is provided), so default commands are one-step for operators.
- Lineup features now use as-of snapshot cutoffs (default 24h before kickoff for training rows) to better match real run timing.
- Legacy wrappers included lineup refresh at that stage and were later removed:
  - `footy-tipper-train.py`
  - `footy-tipper-predict.py`

### Fixed
- Fixed `get_predictions` missing return in prediction utilities (`pipeline/common/model_prediciton/prediction_functions.py`).
- Fixed inference path to safely handle empty inference datasets.
- Fixed venue lumping assignment to ensure the model uses the intended transformed `venue_name` field (`pipeline/common/data-prep/clean-data.R`).
- Fixed turnaround feature leakage across seasons by grouping by team and competition year (`pipeline/common/data-prep/feature-engineering.R`).
- Fixed `matchup_form` to honor `form_period` instead of hardcoded window.
- Fixed fixture home/away split logic to avoid reliance on implicit group ordering (`pipeline/common/data-prep/get-data.R`).
- Fixed `install.R` to actually install required R packages.
- Fixed lineup package import behavior so missing scraper deps (`bs4`/`lxml`) no longer crash `train`/`infer`; lineup refresh now fail-soft skips unless strict mode is enabled.
- Fixed lineup feature timestamp parsing so Unix-second `start_time` values from SQLite match lineup snapshots correctly.
- Improved lineup ingestion observability with live progress logs for topic discovery, sitemap scanning, and article processing.
- Fixed historical lineup parsing coverage by adding legacy text-template support for older official NRL team-list pages (not just the modern structured template).
- Fixed lineup backfill repair behavior so reruns can upgrade existing zero-entry snapshots in place when newer parsing logic can now extract lineup rows from the same article content hash.
- Expanded lineup-derived feature engineering with role-group strength aggregates, halves-pair continuity, rolling lineup stability, bench spine-cover, and snapshot change-rate features.

### Reliability / Hardening
- Fixture fetch now skips unavailable future years instead of hard failing entire runs (`pipeline/common/data-prep/get-data.R`).
- Pipeline now fails fast with clear messages for missing required training rows or missing performance feed when performance is enabled.
- `sending_functions.py` now gracefully degrades when optional providers (Google/OpenAI) are unavailable.

### Notes
- Current behavior is now season-rollover safe for 2026+ without code edits, as long as source feeds are available.
