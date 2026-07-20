# CLI reference

`footy-tipper` is the supported operator interface. It dispatches to [`pipeline/cli.py`](../pipeline/cli.py); removed wrapper scripts and the old `CLI.md` are history, not current entrypoints.

```bash
footy-tipper --help
# Equivalent when the environment has not installed the console wrapper:
python -m pipeline.cli --help
```

## Command map

| Command | Purpose | Default composition |
| --- | --- | --- |
| `prep` | Refresh inputs and write prepared SQLite tables | full requested-season refresh + lineups |
| `train` | Train production artifacts | lineup bootstrap/refresh + smart prep + training |
| `infer` | Predict pre-game rows | lineup refresh + narrow prep + auto-train if needed |
| `send` | Render/distribute existing predictions | Drive upload + Claude copy + idempotent live send |
| `predict` | Run the weekly prediction workflow | infer composition + send |
| `lineups` | Run lineup ingestion only | recent team-list refresh |
| `nrl-data` | Manage nrl.com match-data ingestion | refresh, backfill, or validate Python-owned caches |
| `odds` | Manage market ingestion | Betfair live snapshot or historical workbook backfill |
| `site` | Build the static site | write `docs/site/` locally |
| `evaluate` | Run nested season-out evaluation | prepare, then hold out three recent seasons |
| `state` | Manage Drive-backed runtime state | explicit `push`, `pull`, `gate`, or `schedule` action |

## Shared data flags

`prep`, `train`, `infer`, `predict`, `lineups`, and `evaluate` expose some or all of these flags:

| Flag | Behavior |
| --- | --- |
| `--start-year YEAR` | Override `FOOTY_TIPPER_START_YEAR` (default `2010`). |
| `--end-year YEAR` | Override `FOOTY_TIPPER_END_YEAR` (default current year). |
| `--include-performance` / `--without-performance` | Force performance features on/off. Environment default is on. |
| `--require-odds` / `--allow-missing-odds` | Drop rows without head-to-head odds or keep them. Missing odds are allowed by default. |
| `--skip-nrl-data` | Skip the normal nrl.com and odds refresh before preparation. Available on prep/train/infer/predict. |

The preparation modes are not cosmetic:

- `full` forces a fresh provider pull for all seasons in scope.
- `train` smart-refreshes missing/current seasons, then rebuilds prepared tables.
- `infer` narrows the season window and incrementally upserts prepared rows; it includes one prior context season by default.

## `prep`

```bash
footy-tipper prep [--prep-mode full|train|infer]
                  [--infer-context-years N]
                  [shared data flags]
                  [lineup flags]
```

Default mode: `full`. It writes `footy_tipping_data`, `training_data`, and `inference_data` in `data/footy-tipper-db.sqlite`.

```bash
footy-tipper prep
footy-tipper prep --prep-mode infer --infer-context-years 1
footy-tipper prep --prep-mode train --skip-lineups
```

## `train`

```bash
footy-tipper train [--prep-mode train|full] [--skip-prep]
                   [shared data flags] [lineup flags]
```

Default mode: `train`. Unless skipped, it performs one-time historical nrl.com/odds and lineup bootstraps when coverage is absent, refreshes current inputs, prepares data, and trains. `--skip-prep` uses existing SQLite tables.

The default tuning budget is `FOOTY_TIPPER_TUNE_ITER=100`. Bayesian search parallelizes candidate fits across available cores while each LightGBM fit uses one thread. Production artifacts are trained locally, validated, then published with `state push`; there is no hosted training workflow.

```bash
footy-tipper train
footy-tipper train --start-year 2010 --end-year 2026
footy-tipper train --skip-prep
```

## `infer`

```bash
footy-tipper infer [--prep-mode infer|full] [--infer-context-years N]
                   [--skip-prep] [--skip-auto-train]
                   [shared data flags] [lineup flags]
```

Default mode: `infer`. Required artifacts are checked before prediction; missing artifacts trigger `train` unless `--skip-auto-train` is present. Auto-training inherits data/lineup bootstrap behavior unless those inputs were skipped. Scheduled Actions always supplies `--skip-auto-train`, making missing published artifacts a hard operational failure.

## `send`

```bash
footy-tipper send [--test] [--test-email ADDRESS]
                  [--skip-drive] [--with-llm|--no-llm]
                  [--dry-run] [--force-resend]
```

- `--test` addresses one recipient instead of the production list.
- `--test-email` resolves from the flag, then `FOOTY_TIPPER_TEST_EMAIL`, then `levon_rush@hotmail.com`.
- `--dry-run` prints rendered output without sending.
- `--skip-drive` omits the prediction upload.
- `--with-llm` is the default and asks Claude for prose. `--no-llm` uses deterministic copy.
- `--force-resend` bypasses the production `(season, round)` email ledger.

The hidden `--use-openai` and `--without-openai` aliases remain temporarily for muscle memory. They control the Claude copy path; OpenAI itself is optional banner generation.

After a successful production send, the workflow records `email_sends`, applies an eligible joker transition, refreshes the site, and can upload a gzipped database backup. Missing optional provider dependencies degrade as described in [operations](operations-reliability.md).

## `predict`

```bash
footy-tipper predict [infer flags] [--skip-send]
                     [send flags]
```

This is the normal weekly composition: `prep -> infer -> send`. It exposes the inference, lineup, and delivery flags, including `--skip-prep`, `--skip-auto-train`, `--test`, `--dry-run`, and `--force-resend`.

```bash
footy-tipper predict
footy-tipper predict --test --dry-run --skip-drive
footy-tipper predict --skip-send
```

## `lineups`

```bash
footy-tipper lineups [shared data flags]
                     [--lineups-mode recent|backfill]
                     [--lineups-max-articles N]
                     [--lineups-include-sitemap-in-recent]
                     [--lineups-strict]
```

`recent` is the default. `backfill` also crawls sitemap archives and is the historical repair mode.

```bash
footy-tipper lineups --lineups-mode recent --lineups-max-articles 80
footy-tipper lineups --lineups-mode backfill --start-year 2010 --end-year 2026 --lineups-max-articles 2000
```

Lineup ingestion fails soft unless strict mode is requested. The environment-only bootstrap ceiling defaults to `FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES=2000`.

## `nrl-data`

```bash
footy-tipper nrl-data refresh [--season YEAR] [--max-pages N] [--strict]
footy-tipper nrl-data backfill [--start-year YEAR] [--end-year YEAR] [--max-pages N] [--strict]
footy-tipper nrl-data validate [--start-year YEAR] [--end-year YEAR] [--report-path PATH]
```

`refresh` updates current nrl.com draw/match-centre data and derived cache rows. `backfill` repairs historical match-centre coverage from 2012 onward. `validate` writes parity evidence without changing source state. The default `FOOTY_TIPPER_FEED_SOURCE=python` path invokes refresh automatically before R preparation; `FOOTY_TIPPER_FEED_SOURCE=feed` bypasses it and uses the legacy XML rollback.

## `odds`

```bash
footy-tipper odds live [--strict]
footy-tipper odds backfill [--xlsx-path PATH] [--url URL] [--strict]
```

`live` records available Betfair Exchange match, line, and totals markets for upcoming fixtures. `backfill` imports the Australia Sports Betting historical workbook. Both update `odds_history` and compatible fixture-cache fields; normal orchestration runs them fail-soft unless strict mode is explicitly requested.

## `site`

```bash
footy-tipper site [--publish]
```

The command builds current-round, archive, and season-result pages under `docs/site/`. `--publish` commits and pushes that generated directory, so use it only when publication is intended. The expected Pages URL remains an expected endpoint while it returns 404.

## `evaluate`

```bash
footy-tipper evaluate [shared data flags] [--seasons N] [--skip-prep]
```

Default holdout count: `FOOTY_TIPPER_EVAL_SEASONS` or `3`. Each held-out season gets blend weights, stacker, and calibrator fitted only on earlier seasons. This is the evaluation result to use for performance claims; training-time meta metrics are less conservative.

## `state`

```bash
footy-tipper state push
footy-tipper state pull
footy-tipper state gate
footy-tipper state schedule
```

| Action | Contract |
| --- | --- |
| `push` | Validate the DB and required home/away/manifest artifacts, then upload SQLite, `models/`, and `schedule.json` to Drive state. |
| `pull` | Stage and validate downloaded SQLite/model state before replacing the local last-known-good files. |
| `gate` | Read the schedule and print the workflow decision: send, refresh, or skip. `send` opens at 11:00 Sydney on the first-game day. |
| `schedule` | Print the locally derived next-match schedule without changing Drive state. |

These actions have no nested flags; configuration comes from the environment and service-account file.

For a production model publication, temporarily disable `predict.yml`, run `state pull`, train locally, validate with `infer --skip-prep --skip-lineups --skip-nrl-data --skip-auto-train`, inspect `state schedule`, and only then run `state push`. Re-enable prediction whether the train succeeds or fails.

## Lineup flags shared by prep/train/infer/predict

- `--skip-lineups`
- `--lineups-mode recent|backfill`
- `--lineups-max-articles N`
- `--lineups-include-sitemap-in-recent`
- `--lineups-strict`

The standalone `lineups` command omits only `--skip-lineups`, because skipping the command would be performance art.
