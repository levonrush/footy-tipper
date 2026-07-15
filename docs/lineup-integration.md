# Lineup integration

A team list is not a fact that appears once. It is a sequence: Tuesday squad, reshuffle, late mail, final 17. Footy Tipper stores that sequence so training can know what was knowable at the time instead of quietly borrowing Thursday from Saturday.

![Versioned lineup ingestion and as-of feature selection](diagrams/lineup-as-of.svg)

[Editable Mermaid source](diagrams/lineup-as-of.mmd)

## Source and parser contract

[`pipeline/lineups.py`](../pipeline/lineups.py) discovers official nrl.com Team Lists and Late Mail articles in two modes:

- `recent`: refresh the team-list hub, optionally including sitemap URLs.
- `backfill`: crawl topic pages and sitemap archives for the requested year window.

The parser supports modern `.match-header`/`.team-list` pages and older 2012–2018 text-style team lists inside `.s-cms-content`. NRLW and non-NRL competitions are filtered out.

Required parser dependencies are `beautifulsoup4` and `lxml`. Missing packages or fetch/parse failures log and continue by default. `--lineups-strict` is the only CLI mode that turns ingestion errors into command failure.

## Storage contract

| Table | Grain and responsibility |
| --- | --- |
| `lineup_article_snapshots` | One content-version record with URL, article/fetch timestamps, hash, parse status, and entry count. |
| `lineup_entries` | Normalized team/player/jersey/position/squad rows linked to a snapshot and match round. |
| `lineup_ingestion_runs` | Recent/backfill run summaries used for coverage and bootstrap decisions. |

Content hashes prevent duplicate article versions. A special repair path handles a previously stored snapshot whose hash is unchanged but `entry_count = 0`: improved parser logic can populate that same snapshot and its entries instead of creating a duplicate article row.

## As-of selection

Training and inference intentionally select different eligible endpoints:

- **Training:** latest snapshot known at or before `kickoff - FOOTY_TIPPER_LINEUPS_AS_OF_HOURS_BEFORE_KICKOFF`, default 24 hours.
- **Inference:** latest available pre-game snapshot at run time.

Historical backtests therefore do not use final lineups that were unavailable at the configured decision time. Within-week snapshot changes still inform churn and uncertainty features.

## Feature families

The same feature builder runs for training and inference. It produces home/away values and matchup deltas for:

- availability and missingness;
- named, interchange, reserve, and spine counts;
- bench hooker and spine-cover composition;
- snapshot age and observation window;
- retained-player, starter, spine, halves-pair, and full-spine continuity;
- role-group experience and margin ratings for spine, halves, middles, edges, outside backs, and interchange;
- named and spine cohesion/stability across recent matches;
- within-week named/spine changes and change rates;
- expected squad composition and selection uncertainty.

`lineup_home_players` and `lineup_away_players` are retained for traceability, not used as model predictors. When no eligible lineup exists, numeric features receive safe defaults and `lineup_features_missing` exposes the gap.

## Selection uncertainty

Historical transitions from earlier to later snapshots estimate how much a published squad tends to move. Tier B's conditional win probability is then marginalized over noisy score means with:

- `FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES` (default `64`)
- `FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE` (default `0.12`)

The random stream is derived from `game_id`, so the same inputs reproduce the same probability.

## CLI behavior

Lineup refresh runs before `prep`, `train`, `infer`, and `predict` unless `--skip-lineups` is set. `train` additionally checks historical coverage and, by default, performs a one-time backfill before the normal recent refresh. Auto-training from `infer`/`predict` inherits that bootstrap unless lineups were skipped.

```bash
footy-tipper lineups
footy-tipper lineups --lineups-mode recent --lineups-max-articles 80
footy-tipper lineups --lineups-mode backfill --start-year 2010 --end-year 2026 --lineups-max-articles 2000
```

Shared flags:

- `--skip-lineups`
- `--lineups-mode recent|backfill`
- `--lineups-max-articles N`
- `--lineups-include-sitemap-in-recent`
- `--lineups-strict`

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOOTY_TIPPER_LINEUPS_ENABLED` | `true` | Global ingestion switch. |
| `FOOTY_TIPPER_LINEUPS_MODE` | `recent` | Recent refresh or historical backfill. |
| `FOOTY_TIPPER_LINEUPS_MAX_ARTICLES` | mode-dependent | Per-run article ceiling. |
| `FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES` | `2000` | Train-bootstrap ceiling. |
| `FOOTY_TIPPER_LINEUPS_INCLUDE_SITEMAP_IN_RECENT` | `false` | Broaden recent discovery. |
| `FOOTY_TIPPER_LINEUPS_STRICT` | `false` | Fail the command on ingestion errors. |
| `FOOTY_TIPPER_LINEUPS_AUTO_BACKFILL` | `true` | Allow training bootstrap. |
| `FOOTY_TIPPER_LINEUPS_AS_OF_HOURS_BEFORE_KICKOFF` | `24` | Historical decision cutoff. |
| `FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES` | `64` | Uncertainty integration draws. |
| `FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE` | `0.12` | Score-mean noise scale. |

## State transitions and idempotency

1. Discover URLs for the requested mode/window.
2. Fetch content and calculate the hash.
3. Skip an already healthy identical snapshot.
4. Repair an identical zero-entry snapshot if the current parser now extracts rows.
5. Otherwise insert a new content version and normalized entries.
6. Record run counts/status in `lineup_ingestion_runs`.
7. Select an eligible as-of version during feature building.

This supports safe reruns: broad backfills do not multiply identical articles, and parser improvements can heal old sparse state.

## Failure and coverage limits

- Some Late Mail/commentary pages do not contain a full squad and may legitimately remain zero-entry snapshots.
- Partial current-round coverage is normal before all clubs publish.
- Article markup remains schema-sensitive; monitor parse-rate shifts.
- Missing or sparse lineup data never blocks train/infer in default mode.
- Commercial feeds may provide stronger stable player IDs and clearer service guarantees; official-page scraping should remain polite, cache-aware, and reviewable.
