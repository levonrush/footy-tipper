# Data-source migration: nrl.com production and XML rollback

> **Status: CUT OVER on `main` (PR #34, merged 2026-07-15).** `footy-tipper prep`, `train`, `infer`, and `predict` invoke Python nrl.com and odds ingestion before R preparation. `FOOTY_TIPPER_FEED_SOURCE=python` is the default; `feed` is the legacy credentialled XML rollback.

![Python nrl.com and odds production feeds with the legacy XML rollback](diagrams/feed-migration.svg)

[Editable Mermaid source](diagrams/feed-migration.mmd)

## Production contract

The Python ingestion path owns source refresh while preserving the SQLite cache boundary consumed by R:

| Source | Production role | Persisted contract |
| --- | --- | --- |
| nrl.com draw JSON | Fixtures, game state, scores, venue, kickoff, byes, and match-centre identity | `feed_cache_fixtures` |
| nrl.com match centres | Team/player match statistics | Source tables plus derived `feed_cache_ladders` and `feed_cache_performance` |
| Australia Sports Betting workbook | Historical opening/closing head-to-head, line, and totals markets | `odds_history`; fill missing fixture-cache odds only |
| Betfair Exchange | Live pre-game head-to-head, line, and totals snapshots when configured | `odds_history` and current fixture-cache odds |

The CLI runs `nrl-data refresh` and `odds live` before normal R preparation. A training run also performs one-time historical nrl.com and odds backfills when their completion markers are absent. Individual network/parse failures fail soft by default and preserve usable cache state; the downstream performance-data requirement still fails clearly when enabled coverage is insufficient.

[`pipeline/data-prep.R`](../pipeline/data-prep.R) does not refetch the production sources. It reads `feed_cache_fixtures`, `feed_cache_ladders`, and `feed_cache_performance`, then builds the prepared training/inference tables. This compatibility boundary kept the feature pipeline stable through cutover.

## Temporal and odds safety

Historical workbook rows include opening and closing observations. They are stored explicitly in `odds_history`; closing values fill only fixture-cache gaps for historical training. Live Betfair observations update upcoming fixtures and retain their observation time. Prediction-time odds must not be silently replaced with later closing prices.

Missing odds remain supported through the model's explicit missing-market path. Betfair authentication, network, or market-matching failure leaves existing values untouched unless strict odds ingestion is requested.

## Legacy XML rollback

Set `FOOTY_TIPPER_FEED_SOURCE=feed` to bypass Python nrl.com/odds orchestration and let R fetch the previous XML endpoints. That path requires:

- `PASSWORD`
- `BASE_URL`
- `NRL_FIXTURES_EXTENTION`
- `NRL_ROUND_LADDER_EXTENTION`
- `NRL_PERFORMANCE_EXTENTION` when performance is enabled

The rollback writes the same `feed_cache_*` contracts. It is a recovery option, not the production default and not a second source to merge into the same preparation run.

## Cutover evidence

The migration gates were completed before the default changed:

- Supported CLI commands invoke Python ingestion, with `--skip-nrl-data` for an intentional cache-only run.
- Fixture, ladder, and performance rows satisfy the R-facing schemas and incremental refresh rules.
- [`nrl_data_parity_modern.csv`](../reports/nrl_data_parity_modern.csv) records 100% modern fixture parity, 100% ladder-core parity, and aligned 2019–2025 performance rows.
- [`evaluate-feed-migration-20260714.json`](../reports/evaluate-feed-migration-20260714.json) records the nested evaluation used for the model-path comparison.
- Network/markup errors preserve the last usable cache, and `FOOTY_TIPPER_FEED_SOURCE=feed` provides operational rollback.

## Operator commands

```bash
footy-tipper nrl-data refresh
footy-tipper nrl-data backfill --start-year 2012
footy-tipper nrl-data validate --report-path reports/nrl-data-check.csv
footy-tipper odds live
footy-tipper odds backfill

# Explicit legacy recovery only:
FOOTY_TIPPER_FEED_SOURCE=feed footy-tipper prep
```

Backfill is an intentional historical repair operation; normal prediction uses the narrow current refresh. `validate` is read-only evidence generation. Use `--strict` while diagnosing ingestion itself, not as a permanent default for the weekly prediction path.

## Remaining extensions

The source cutover is complete; these are independent modelling/data additions rather than migration blockers:

- player-performance ratings joined to versioned lineup identities;
- totals-based Poisson score offsets after coverage and temporal-safety evaluation;
- richer referee, weather, travel, and rest features;
- ongoing parity monitoring when nrl.com markup or provider behavior changes.

Each extension still requires train/infer symmetry and honest season-out evaluation. Keep Drive backups and an independent cold SQLite copy before large backfills or schema changes.

## Source notes

- [nrl.com draw JSON example](https://www.nrl.com/draw/data?competition=111&round=19&season=2026)
- [Australia Sports Betting historical data](https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/)
- [Betfair historical data services](https://developer.betfair.com/historical-data-services-api/)
- [Open-Meteo forecast and historical weather](https://open-meteo.com/)

External endpoints and commercial terms can change. The checked-in parity files record cutover evidence, not a permanent service guarantee.
