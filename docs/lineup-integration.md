# Lineup integration

A team list is not a fact that appears once. It is a sequence: Tuesday squad, reshuffle, late mail, final 17. Footy Tipper stores that sequence so training can know what was knowable at the time instead of quietly borrowing Thursday from Saturday.

![Versioned lineup ingestion and as-of feature selection](diagrams/lineup-as-of.svg)

[Editable Mermaid source](diagrams/lineup-as-of.mmd)

## Source and parser contract

[`pipeline/lineups.py`](../pipeline/lineups.py) discovers official nrl.com Team Lists and Late Mail articles in two modes:

- `recent`: refresh the team-list hub, optionally including sitemap URLs.
- `backfill`: crawl topic pages and sitemap archives for the requested year window.

The parser supports modern `.match-header`/`.team-list` pages and older 2012–2018 text-style team lists inside `.s-cms-content`. NRLW and non-NRL competitions are filtered out.

Required parser dependencies are `beautifulsoup4` and `lxml`. Missing packages or fetch/parse failures log and continue by default. Explicit strict diagnosis in the advanced command is the only mode that turns ingestion errors into command failure.

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

The cutoff is measured from `start_time_utc`, the true UTC kickoff. It used to
be measured from `start_time`, which is venue-local wall clock serialised
as-if-UTC, while article publish times are true UTC. Comparing the two shifted
the cutoff by the venue's offset, so the documented 24 hour guard was really
running at about 13 to 14 hours for Australian venues. Anyone tuning the horizon
should know the pre-fix numbers were measured against the shorter one.

`lineup_source_age_hours` is built on a separate path and **still** measures from
`start_time`, so it carries the same inflation. It is a declared predictor, so
correcting it changes a feature value under the already-shipped pickles. It is
therefore deferred to land alongside a retrain rather than on its own.

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

Current-lineup refresh is part of normal cloud prediction and staged model training. `update-model` also checks historical coverage and can perform the one-time backfill required by the configured training window. Missing model artifacts in an everyday tips command never trigger training.

```bash
footy-tipper advanced data lineups refresh
footy-tipper advanced data lineups refresh --max-articles 80
footy-tipper advanced data lineups backfill --start-year 2010 --end-year 2026 --max-articles 2000
```

Use leaf `--help` for the technical limits, sitemap, and strict flags. Those controls are deliberately kept out of the beginner interface.

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

## Historical coverage is structurally limited

Team-list features reach roughly a third of training games (1,168 of 3,593 as at 2026-08-12, after finals articles were recovered). This is expected, not a join defect, and the investigation below should not be repeated.

`lineup_entries` holds about 201 fixtures per season back to 2008, but most of those rows come from `match_state = FullTime` match-centre pages (3,945 of 4,797 snapshots). For a completed match the match-centre parser keeps only jersey 1 to 17 non-replacement players, which is the side that actually played. That is post-match information, and the 2024 backfill stamped it with the CMS `updated` date rather than a publication date. The as-of cutoff is what keeps it out of training, so it is a leakage guard rather than a tunable coverage knob.

Usable pre-kickoff coverage therefore depends entirely on how nrl.com published its round team-list articles in a given era, which changed over time:

| Season | Median publish lead vs kickoff | Games covered |
| --- | --- | --- |
| 2012 | +35 h | 59.7% |
| 2015, 2016 | -36 h (published mid-round) | 0.5%, 0.0% |
| 2018 | +9 h | 32.8% |
| 2022-2025 | +108 h | ~95% |

Two checks confirmed the attribution is correct rather than a parsing bug. Comparing each article's named players against the played roster gives a mean Jaccard overlap of 0.85 to 1.00 against the round it is filed under, versus 0.71 to 0.92 against the following round, in every season. Pre-round articles also show *lower* overlap with played teams (2012: 0.849) than mid-round ones (2015: 0.956), which is the expected signature of late team changes. So `parse_round_id` files articles correctly, and the 2015 to 2017 "Updated Round N team lists" pages are genuinely retrospective.

Do not relax `FOOTY_TIPPER_LINEUPS_AS_OF_HOURS_BEFORE_KICKOFF` to raise historical coverage. It would admit post-match rosters into training.

## Failure and coverage limits

- Some Late Mail/commentary pages do not contain a full squad and may legitimately remain zero-entry snapshots.
- Partial current-round coverage is normal before all clubs publish.
- Article markup remains schema-sensitive; monitor parse-rate shifts.
- Missing or sparse lineup data never blocks train/infer in default mode.
- Commercial feeds may provide stronger stable player IDs and clearer service guarantees; official-page scraping should remain polite, cache-aware, and reviewable.
