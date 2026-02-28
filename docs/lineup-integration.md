# Lineup Integration Guide

This doc covers the lineup-aware upgrade: ingestion, storage, model features, and operations.

## Why This Is Feasible For Your 24h Manual Run

Your current operating pattern (manual run within 24 hours of the first game) is a good fit:

- Team-list articles are already published by then.
- Late-mail updates may still land, but you can rerun ingestion quickly.
- Inference is now designed to degrade safely if lineup scraping fails.

In practice, the minimum reliable sequence is:

1. `footy-tipper lineups --lineups-mode recent`
2. `footy-tipper infer`
3. `footy-tipper send --test --dry-run` (optional sanity check)

If you prefer one command, use `footy-tipper predict`.
It now auto-trains when model artifacts are missing.
For historical training coverage, `footy-tipper train` also bootstraps a one-time backfill when needed.

## What Was Added

### New ingestion entrypoint

- `pipeline/lineups.py`
- Pulls article URLs from Team Lists hub (`recent`) or sitemap archives (`backfill`).
- Fetches and parses structured team sheets from NRL team-list/late-mail articles.
- Supports both:
  - current `.match-header` / `.team-list` article templates
  - legacy 2012-2018 text-style team-list pages inside `.s-cms-content`
- Writes snapshots and normalized player rows into SQLite.

### New lineup storage tables

Created automatically when ingestion runs:

- `lineup_article_snapshots`
- `lineup_entries`
- `lineup_ingestion_runs`

`lineup_article_snapshots` stores per-fetch metadata (`article_url`, timestamps, parse status, hash).  
`lineup_entries` stores normalized rows (`team`, `player`, `jersey`, `position`, `squad_group`, `year/round`).
`lineup_ingestion_runs` stores backfill/recent run summaries so the CLI can tell whether historical bootstrap has already been done.

Important repair behavior:
- if a previously scraped snapshot already exists with the same content hash but `entry_count = 0`, a later rerun can now repair that snapshot in place and insert the newly parsed lineup rows
- this matters for historical backfills because many older pages were discovered correctly before the legacy parser existed, but had no extracted entries

### New feature generation

Model features are built in Python from latest lineup snapshots per team/year/round and merged into train/infer datasets:

- availability: `lineup_data_available_home/away`
- roster size: named/interchange/reserve counts
- composition: spine counts, spine complete flags, bench hooker counts, bench spine-cover counts
- freshness: lineup source age in hours
- continuity: retained player ratio vs previous lineup, starter/spine retention, same-halves-pair and same-spine flags
- role-group strength: historical experience and margin-rating aggregates for spine, halves, middles, edges, outside backs, and interchange
- cohesion/stability: named/spine pair cohesion, repeated halves pairing, recent named/spine stability over the last four matches
- change tracking: snapshot window hours plus named/spine change counts and per-snapshot change rates
- matchup deltas: home-away differences for all major lineup metrics
- uncertainty: expected named/spine/interchange counts and lineup-selection uncertainty
- quality flag: `lineup_features_missing`

Player lists are also attached as strings:

- `lineup_home_players`
- `lineup_away_players`

These list columns are currently for traceability/debugging and are not used as model predictors.

### Selection uncertainty and win-probability marginalization

The model now estimates lineup-selection uncertainty from historical snapshot transitions (earlier squad -> latest squad in the same match week) and uses it in two places:

- features:
  - `lineup_expected_named_count_*`
  - `lineup_expected_spine_count_*`
  - `lineup_expected_interchange_count_*`
  - `lineup_selection_uncertainty_*`
- probability stack input:
  - Tier-B conditional home win probability is Monte Carlo-marginalized over lineup uncertainty.

## CLI Behavior

Lineup refresh now runs before `prep`, `train`, `infer`, and `predict` unless explicitly skipped.

`train` has one extra default:
- if historical lineup backfill has not been bootstrapped for the configured training window, it runs a one-time backfill before the normal recent refresh

`predict` stays fast for weekly use:
- it runs the normal recent refresh
- if models are missing and auto-training is triggered, that training run inherits the same historical lineup bootstrap logic

New command:

- `footy-tipper lineups`

New options (also available on `prep/train/infer/predict`):

- `--skip-lineups`
- `--lineups-mode recent|backfill`
- `--lineups-max-articles N`
- `--lineups-include-sitemap-in-recent`
- `--lineups-strict`

## Environment Variables

Optional runtime controls:

- `FOOTY_TIPPER_LINEUPS_ENABLED` (default: `true`)
- `FOOTY_TIPPER_LINEUPS_MODE` (`recent` or `backfill`; default: `recent`)
- `FOOTY_TIPPER_LINEUPS_MAX_ARTICLES` (default: mode-dependent)
- `FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES` (default: `2000` for train bootstrap)
- `FOOTY_TIPPER_LINEUPS_INCLUDE_SITEMAP_IN_RECENT` (default: `false`)
- `FOOTY_TIPPER_LINEUPS_STRICT` (default: `false`)
- `FOOTY_TIPPER_LINEUPS_AUTO_BACKFILL` (default: `true`)
- `FOOTY_TIPPER_LINEUPS_AS_OF_HOURS_BEFORE_KICKOFF` (default: `24`)
- `FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES` (default: `64`)
- `FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE` (default: `0.12`)

## Python Dependencies

Lineup scraping requires:

- `beautifulsoup4`
- `lxml`

If these are missing, lineup refresh skips in fail-soft mode (unless strict mode is enabled).

## Reliability + Failure Modes

- Default mode is fail-soft: lineup ingestion errors do not crash train/infer.
- Strict mode is opt-in (`--lineups-strict` or env).
- Existing train/infer contracts remain intact (`Final` for training, `Pre Game` for inference).
- If no lineup data exists, lineup predictors are filled with safe defaults and model execution continues.
- Partial coverage is normal if only some current-round team-list articles have been published when you run.

## Backfill Strategy

Manual backfill is still available when you want it:

```bash
footy-tipper lineups --lineups-mode backfill --start-year 2018 --end-year 2026 --lineups-max-articles 2000
```

Backfill now emits progress logs during:
- topic-page discovery
- sitemap discovery/scanning
- article processing progress counters
- repair notices when old zero-entry snapshots are upgraded with parsed lineup rows

Run once, then switch back to `recent` for weekly operation.

## Notes / Limits

- Parsing is still schema-sensitive, but now covers both the modern structured template and the older text-style official team-list pages used across much of 2012-2018.
- Some older "Late Mail" / commentary-only pages still do not contain a full structured squad and may remain zero-entry snapshots.
- NRLW/non-NRL competitions are filtered out from lineup ingestion.
- Commercial/licensed feeds are still the best long-term option for stronger ID stability and legal clarity.
