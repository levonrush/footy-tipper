# Data-source migration: current feed and nrl.com target

> **Status: CUT OVER on `main` (PR #34, merged 2026-07-15).** `footy-tipper prep/train/infer/predict` invoke the `pipeline/common/nrl_data/` refresh and the odds ingestion before R preparation; `FOOTY_TIPPER_FEED_SOURCE=python` is the default and `feed` restores the legacy XML path. Parity evidence: `reports/nrl_data_parity_modern.csv` (fixtures 100%, ladder core 100%, performance rows aligned 2019-2025) and the honest benchmark in `reports/evaluate-feed-migration-20260714.json`.

The sections below record the migration design as reviewed on 2026-07-13; the cutover gates listed later in this document have been satisfied.

![Current production feeds and unfinished target paths](diagrams/feed-migration.svg)

[Editable Mermaid source](diagrams/feed-migration.mmd)

## Current production contract

[`get-data.R`](../pipeline/common/data-prep/get-data.R) fetches three XML endpoint families:

| Feed | Contents | Downstream ownership |
| --- | --- | --- |
| Fixtures | IDs, round/state/time/venue, teams, scores, head-to-head odds, line odds/amount | Match backbone, targets, market features, `odds_snapshots` |
| Round ladder | W/D/L, points for/against, scoring splits, form/streak/context | Lagged ladder predictor families |
| Performance leaderboard | Season-to-date team attack/defence/possession/discipline metrics | Lagged performance predictor families when enabled |

Fetched seasons are cached in `feed_cache_fixtures`, `feed_cache_ladders`, and `feed_cache_performance`. These SQLite schemas are the proposed compatibility boundary: replacement ingestion should write the same contracts so R feature engineering does not change during provider cutover.

## Prototype modules

The branch adds draw, match-centre, derived ladder/performance, cache-write, refresh, and validation code under `pipeline/common/nrl_data/`. It can explore:

- nrl.com draw JSON for fixtures, state, scores, venue, kickoff, byes, and match-centre URLs;
- match-centre embedded data for team/player stats, attendance, weather, ground, and officials;
- derived round ladders from fixture history;
- a cache writer that targets existing `feed_cache_*` layouts;
- overlap validation against cached provider history.

These modules are **next-state components**, not the current hot path. Any module docstring suggesting otherwise should be read as prototype intent until a CLI/R call graph proves the integration.

## Odds remain an explicit decision

Potential sources identified by the research include:

- [Australia Sports Betting historical NRL results and odds](https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/) for personal-use research/backfill, including totals and price movement where covered;
- [Betfair Developer Program](https://developer.betfair.com/) for account-backed live exchange integration;
- [The Odds API](https://the-odds-api.com/) as a paid bookmaker aggregation option.

No live provider has been selected or wired. Historical license/terms, team/game matching, snapshot timing, and production reliability must be reviewed before ingestion. The existing `odds_snapshots` ledger should remain the prediction-time observation contract regardless of source.

## Cutover gates

1. **Call graph:** a supported CLI command explicitly invokes the new refresh path.
2. **Schema parity:** fixture/ladder/performance rows satisfy the existing cache columns and types.
3. **Historical overlap:** season/round/team totals and derived values reconcile against frozen provider caches with explained exceptions.
4. **Temporal safety:** training features use only information available before each match; prediction-time odds are not silently replaced with closing prices.
5. **Incremental behavior:** frozen seasons stay frozen; current/missing seasons refresh safely and idempotently.
6. **Failure behavior:** network/markup errors preserve the last good cache and report an actionable state.
7. **Performance toggle:** enabled performance still fails clearly when required coverage is absent.
8. **Operations:** Actions container dependencies, Drive state, schedule derivation, and recovery are exercised.
9. **Shadow period:** old and new paths run over an overlap window before production selection changes.
10. **Documentation:** only after the CLI and R wiring ships may the target be described in present tense.

## Migration sequence

### Phase 0 — protect the corpus

- Keep Drive backups and an independent cold copy of the SQLite database.
- While the provider remains available, run an intentional full refresh and verify cache coverage.

### Phase 1 — establish parity

- Complete fixture, ladder, and performance writers against the existing cache schemas.
- Add repeatable overlap reports and thresholds.
- Decide the live odds provider and snapshot timing contract.
- Add a feature-flagged CLI/R integration with rollback to the current feed.

### Phase 2 — shadow and cut over

- Run both paths for current rounds.
- Compare prepared tables, model features, and predictions, not just raw records.
- Switch the default only after operational recovery and missing-data behavior are proven.

### Phase 3 — optional expansion

- totals and line-movement market features;
- EWMA match-centre form;
- player performance joined to versioned lineups;
- referee, weather, rest, and travel features.

Each expansion requires train/infer symmetry and honest evaluation; it should not ride along invisibly with the provider replacement.

## Source notes

- [nrl.com draw JSON example](https://www.nrl.com/draw/data?competition=111&round=19&season=2026)
- [Australia Sports Betting historical data](https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/)
- [Betfair historical data services](https://developer.betfair.com/historical-data-services-api/)
- [Open-Meteo forecast and historical weather](https://open-meteo.com/)

External endpoints and commercial terms can change. The verification date records the research snapshot, not a service guarantee.
