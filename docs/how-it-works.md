# Architecture

Footy Tipper is a small production system wearing a tipping-comp scarf. R owns provider preparation and broad feature tables; Python owns orchestration, lineups, modelling, inference, decisions, and delivery; SQLite is the hand-off and operational ledger.

![Current end-to-end production architecture](diagrams/current-production.svg)

[Editable Mermaid source](diagrams/current-production.mmd)

## Entrypoints and ownership

| Boundary | Owner | Contract |
| --- | --- | --- |
| Operator CLI | `footy-tipper` -> [`pipeline/cli.py`](../pipeline/cli.py) | The supported day-to-day interface. |
| Provider preparation | [`pipeline/data-prep.R`](../pipeline/data-prep.R) and `pipeline/common/data-prep/` | Refresh provider caches and write prepared match tables. |
| Lineup ingestion | [`pipeline/lineups.py`](../pipeline/lineups.py) and `pipeline/common/lineups/` | Discover, parse, version, normalize, and repair official team-list snapshots. |
| Training | [`pipeline/train.py`](../pipeline/train.py) | Fit score, binary, stack, calibration, dispersion, margin, and joker artifacts. |
| Inference | [`pipeline/inference.py`](../pipeline/inference.py) | Rebuild pre-game context, apply artifacts, simulate, and upsert predictions. |
| Delivery | `pipeline/common/use_predictions/` | Select tips/value, size stakes, decide joker, render copy/site, send, and record state. |
| Cloud state | `pipeline/ops/` and Google Drive | Synchronize mutable DB/model/schedule state around Actions jobs. |

Removed wrapper scripts and historical `CLI.md` files are not alternate entrypoints. Their removal belongs in the [changelog](../CHANGELOG.md).

## Current external inputs

The production preparation path currently reads credentialled XML endpoints configured by:

- `BASE_URL`
- `NRL_FIXTURES_EXTENTION`
- `NRL_ROUND_LADDER_EXTENTION`
- `NRL_PERFORMANCE_EXTENTION`
- `PASSWORD`

The fixtures feed also carries head-to-head and line-market columns where available. R persists season-scoped provider caches as `feed_cache_fixtures`, `feed_cache_ladders`, and `feed_cache_performance`, so smart refreshes can avoid refetching frozen seasons.

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

Preparation modes control cache and table writes:

- `full`: refresh every requested season and replace prepared tables.
- `train`: smart-refresh missing/current seasons and replace prepared tables.
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

| Artifact | Purpose | Required to trigger no auto-train? |
| --- | --- | --- |
| `home_model.pkl` | Tier-B home-score model | Yes |
| `away_model.pkl` | Tier-B away-score model | Yes |
| `model_manifest.json` | predictor layout, blend weights, Tier-A config, `lambda3`, dispersion, lineup uncertainty controls, and optional margin/joker metadata | Yes |
| `binary_model.pkl` | Tier-C direct home-win classifier | Optional compatibility fallback |
| `stacker.pkl` | regularized logistic signal combiner | Optional compatibility fallback |
| `win_prob_calibrator.pkl` | beta probability calibrator | Optional compatibility fallback |
| `joker_policy.json` | backtested joker policy summary | Optional for core inference |

The CLI regards home, away, and manifest files as the minimum artifact set. Inference can load older compatible state without Tier C/stacker/calibrator; it falls back toward the Tier-B conditional probability. A normal current training run writes all of them.

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

GitHub Actions owns scheduling. Google Drive owns mutable cross-run state. Git owns code and hand-written documentation. Generated `docs/site/` output is a publication artifact, not a runtime source of truth.

![GitHub Actions gate, Drive state, delivery, backup, and Pages flow](diagrams/operations-state.svg)

[Editable Mermaid source](diagrams/operations-state.mmd)

## Current versus target feeds

![Current production feeds and the target nrl.com migration](diagrams/feed-migration.svg)

[Editable Mermaid source](diagrams/feed-migration.mmd)

The modules under `pipeline/common/nrl_data/` on `feed-migration` are a **prototype/WIP**. They are not invoked by `footy-tipper`, `pipeline/data-prep.R`, or the current Actions workflows. Their cache-schema compatibility is useful groundwork; it is not production cutover. Dashed paths in the diagram mark this unfinished route. See [Data-source migration](data-source-migration.md) for parity gates and ownership decisions.

## Failure boundaries

- No pre-game rows: inference writes an empty batch and exits cleanly; send does not fail the pipeline.
- Lineup ingestion: logs and continues by default; only strict mode fails the command.
- Performance feed: when explicitly enabled, missing required data fails fast and names the problem.
- Missing optional Google/Claude/OpenAI dependencies: Drive/banner/copy paths skip or fall back as documented; core prediction remains usable.
- Missing required models: infer/predict auto-train unless `--skip-auto-train` requests a hard failure.
- Re-runs: data writes use replacement/upsert or append-only snapshot contracts; production email and joker use have explicit idempotency ledgers.
