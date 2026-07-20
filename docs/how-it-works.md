# Architecture

Footy Tipper is a small production system wearing a tipping-comp scarf. Python owns source ingestion, orchestration, lineups, modelling, inference, decisions, and delivery; R reads the Python-owned feed caches and builds broad feature tables; SQLite is the hand-off and operational ledger.

![Current end-to-end production architecture](diagrams/current-production.svg)

[Editable Mermaid source](diagrams/current-production.mmd)

## Entrypoints and ownership

| Boundary | Owner | Contract |
| --- | --- | --- |
| Operator CLI | `footy-tipper` -> [`pipeline/cli.py`](../pipeline/cli.py) | The supported day-to-day interface. |
| Source ingestion | `pipeline/common/nrl_data/` and `pipeline/common/odds/` | Refresh nrl.com match data and market snapshots into compatible SQLite caches. |
| Provider preparation | [`pipeline/data-prep.R`](../pipeline/data-prep.R) and `pipeline/common/data-prep/` | Read cached inputs and write prepared match tables. |
| Lineup ingestion | [`pipeline/lineups.py`](../pipeline/lineups.py) and `pipeline/common/lineups/` | Discover, parse, version, normalize, and repair official team-list snapshots. |
| Training | [`pipeline/train.py`](../pipeline/train.py) | Fit score, binary, stack, calibration, dispersion, margin, and joker artifacts. |
| Inference | [`pipeline/inference.py`](../pipeline/inference.py) | Rebuild pre-game context, apply artifacts, simulate, and upsert predictions. |
| Delivery | `pipeline/common/use_predictions/` | Select tips/value, size stakes, decide joker, render copy/site, send, and record state. |
| Published state | `pipeline/ops/` and Google Drive | Publish validated local models and synchronize DB/model/schedule state around Actions prediction. |

Removed wrapper scripts and historical `CLI.md` files are not alternate entrypoints. Their removal belongs in the [changelog](../CHANGELOG.md).

## Current external inputs

The production default is `FOOTY_TIPPER_FEED_SOURCE=python`. Before R preparation, the CLI invokes:

- nrl.com draw JSON for fixture identity, round/state, scores, venue, and kickoff;
- nrl.com match centres for team/player match statistics used to derive ladder and performance cache rows;
- Australia Sports Betting for the one-time historical odds backfill;
- Betfair Exchange for available live pre-game match, line, and totals snapshots.

The Python path writes season-scoped `feed_cache_fixtures`, `feed_cache_ladders`, and `feed_cache_performance` rows plus the odds history/snapshot contracts. Smart refreshes preserve frozen seasons and fail-soft network paths leave the last usable cache in place. `FOOTY_TIPPER_FEED_SOURCE=feed` is the explicit rollback: R fetches the legacy credentialled XML endpoints using `PASSWORD`, `BASE_URL`, and the `NRL_*_EXTENTION` values.

Lineups are fetched separately from official nrl.com Team Lists and Late Mail articles. The parser supports modern structured pages and older 2012–2018 text layouts.

## Preparation and SQLite contracts

`pipeline/data-prep.R` writes or incrementally upserts:

| Table | Role |
| --- | --- |
| `footy_tipping_data` | Full chronological match context used for feature state. |
| `training_data` | Only rows where `game_state_name == "Final"`. |
| `inference_data` | Only rows where `game_state_name == "Pre Game"`, restricted to the next season/round scope. |
| `odds_snapshots` | Append-only pre-game market observations from each preparation run. |

That explicit Final/Pre Game split is a leakage boundary, not a convenience.

Preparation modes control ingestion scope and table writes:

- `full`: refresh the requested source scope and replace prepared tables.
- `train`: bootstrap missing historical nrl.com/odds coverage, smart-refresh current inputs, and replace prepared tables.
- `infer`: smart-refresh a narrow context window, upsert context/training rows by `game_id`, replace the inference batch, and append odds snapshots.

Lineup ingestion owns:

- `lineup_article_snapshots`
- `lineup_entries`
- `lineup_ingestion_runs`

Prediction and operator state add:

- `predictions_table`, upserted through [`insert_into_table.sql`](../pipeline/common/sql/insert_into_table.sql)
- `email_sends`, the live-send idempotency ledger
- `joker_usage`, the once-per-season joker transition

[`prediction_table.sql`](../pipeline/common/sql/prediction_table.sql) always selects the latest season with pre-game context and the minimum round in that season.

## Model flow

![Tier A, B, C, market, calibration, margin, and simulation flow](diagrams/model-stack.svg)

[Editable Mermaid source](diagrams/model-stack.mmd)

Training produces these principal files under `models/`:

| Artifact | Purpose | Required for production inference? |
| --- | --- | --- |
| `home_model.pkl` | Tier-B home-score model | Yes |
| `away_model.pkl` | Tier-B away-score model | Yes |
| `model_manifest.json` | predictor layout, blend weights, Tier-A config, `lambda3`, dispersion, lineup uncertainty controls, and optional margin/joker metadata | Yes |
| `binary_model.pkl` | Tier-C direct home-win classifier | Optional compatibility fallback |
| `stacker.pkl` | regularized logistic signal combiner | Optional compatibility fallback |
| `win_prob_calibrator.pkl` | beta probability calibrator | Optional compatibility fallback |
| `joker_policy.json` | backtested joker policy summary | Optional for core inference |

The CLI regards home, away, and manifest files as the minimum artifact set. Inference can load older compatible state without Tier C/stacker/calibrator; it falls back toward the Tier-B conditional probability. A normal current training run writes all of them.

Production training is local-authoritative. The normal search budget is 100 Bayesian candidates; candidate fits parallelize across the operator's cores while each LightGBM fit stays single-threaded. After a successful no-auto-train inference validation, `state push` validates the required artifact set and publishes it to Drive. Actions never trains: all test, refresh, and live prediction modes pass `--skip-auto-train` and consume the published model archive.

## Lineups and as-of state

![Versioned lineup ingestion and as-of feature selection](diagrams/lineup-as-of.svg)

[Editable Mermaid source](diagrams/lineup-as-of.mmd)

Training selects the latest eligible snapshot at or before the configured cutoff, normally 24 hours before kickoff. Inference selects the most recent known pre-game snapshot. Both paths derive the same versioned feature families and fill safe defaults when coverage is absent. See [Lineup integration](lineup-integration.md).

## Delivery and state ownership

`send` reads the current prediction view and computes:

- primary tip and calibrated probability
- expected-value opportunities from offered odds
- Kelly-derived stake fractions with configured floors/caps
- joker recommendation and competition strategy advice
- Claude-generated or deterministic email prose
- an optional OpenAI-generated banner

A production send records the season/round before another live send is allowed, persists an eligible joker use only after successful delivery, refreshes the local site, and can upload Drive state and a gzipped DB backup. Test and dry-run modes do not perform the live state transition.

The local operator owns production training. GitHub Actions owns prediction scheduling and delivery, polling every 15 minutes and opening the next unsent round at 11:00 `Australia/Sydney` on its first-game day. Google Drive owns published mutable cross-run state. Git owns code and hand-written documentation. Generated `docs/site/` output is a publication artifact, not a runtime source of truth.

![GitHub Actions gate, Drive state, delivery, backup, and Pages flow](diagrams/operations-state.svg)

[Editable Mermaid source](diagrams/operations-state.mmd)

## Production feed and rollback

![Python nrl.com and odds production feeds with the legacy XML rollback](diagrams/feed-migration.svg)

[Editable Mermaid source](diagrams/feed-migration.mmd)

The Python nrl.com/odds path cut over on `main` in PR #34 and is invoked by `prep`, `train`, `infer`, and `predict` unless `--skip-nrl-data` or the legacy feed source is selected. R deliberately keeps the same cache boundary. Dashed paths in the diagram mark the XML rollback and optional future feature extensions, not an unfinished production cutover. See [Data-source migration](data-source-migration.md) for parity evidence and recovery behavior.

## Failure boundaries

- No pre-game rows: inference writes an empty batch and exits cleanly; send does not fail the pipeline.
- Lineup ingestion: logs and continues by default; only strict mode fails the command.
- Performance feed: when explicitly enabled, missing required data fails fast and names the problem.
- Missing optional Google/Claude/OpenAI dependencies: Drive/banner/copy paths skip or fall back as documented; core prediction remains usable.
- Missing required models: standalone infer/predict auto-train unless `--skip-auto-train` requests a hard failure; Actions always requests that hard failure so hosted hardware cannot train implicitly.
- Re-runs: data writes use replacement/upsert or append-only snapshot contracts; production email and joker use have explicit idempotency ledgers.
