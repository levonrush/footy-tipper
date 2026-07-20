# AGENTS

This file is for coding/automation agents working on `footy-tipper`.

## Purpose
- Predict NRL match outcomes and scorelines using an R + Python + SQLite pipeline.
- Produce weekly tips and optionally send results to Google Drive + email list.

## High-Level Architecture
- CLI wrapper:
  - `footy-tipper` (root executable) -> `pipeline/cli.py`
  - Preferred operator interface for day-to-day runs.
- Entrypoints:
  - `footy-tipper train` / `footy-tipper predict` (or `python -m pipeline.cli ...`); legacy wrapper scripts were removed.
- Lineup ingestion:
  - `pipeline/lineups.py` fetches Team Lists/Late Mail articles, parses structured lineups, and writes:
    - `lineup_article_snapshots`
    - `lineup_entries`
    - `lineup_ingestion_runs`
  - Ingestion must support both modern structured NRL team-list pages and the older 2012-2018 text-style team-list pages.
  - Re-running lineup backfill should be able to repair previously stored zero-entry snapshots when newer parsing logic can now extract rows from the same article hash.
  - `pipeline/train.py` and `pipeline/inference.py` merge lineup-derived features from these tables.
  - Current lineup feature families include squad size/composition, uncertainty, continuity, role-group strength (spine/halves/middles/edges/outside backs/interchange), cohesion, and within-week snapshot churn.
- Data prep:
  - `pipeline/data-prep.R` sources `pipeline/common/data-prep/*.R`.
  - Writes three SQLite tables to `data/footy-tipper-db.sqlite`:
    - `footy_tipping_data`
    - `training_data`
    - `inference_data`
- Training:
  - `pipeline/train.py` builds Tier-A baseline features, trains Tier-B home/away Poisson models and the Tier-C binary classifier, blends expected scores, fits the regularized stacker + LOSO beta calibrator, estimates `lambda3`/dispersion, and saves artifacts in `models/`.
  - Key artifacts: `home_model.pkl`, `away_model.pkl`, `binary_model.pkl`, `stacker.pkl`, `win_prob_calibrator.pkl`, `model_manifest.json`, `joker_policy.json`.
  - Production training runs on the operator's local hardware. The default Bayesian search budget is `FOOTY_TIPPER_TUNE_ITER=100`; search fits parallelize externally while each LightGBM fit remains single-threaded.
  - Google Drive `state/` is the publication channel for validated models. GitHub Actions must not train or auto-train.
- Inference:
  - `pipeline/inference.py` loads artifacts + manifest, rebuilds Tier-A baseline context, applies blend/stack/calibration, simulates outcomes with bivariate Poisson (`lambda3`), and upserts into `predictions_table`.
- Distribution:
  - `footy-tipper send` (pipeline/cli.py) reads the prediction view, computes EV-based value picks with Kelly-derived staking, and handles upload/email via `pipeline/common/use_predictions/` modules (joker, staking, scoreboard, email_copy, email_render, distribution, site).
- Production feeds:
  - The default `FOOTY_TIPPER_FEED_SOURCE=python` path invokes `pipeline/common/nrl_data/` and odds ingestion before R preparation. R reads the resulting `feed_cache_*` tables.
  - `FOOTY_TIPPER_FEED_SOURCE=feed` restores the credentialled XML path as an explicit rollback.
- Documentation:
  - Repository Markdown is canonical. The private Notion hub is a curated navigation/story layer and must link back to it.

## Critical Runtime Config
- Season controls:
  - `FOOTY_TIPPER_START_YEAR` (default: `2010`)
  - `FOOTY_TIPPER_END_YEAR` (default: current year)
  - `FOOTY_TIPPER_INCLUDE_PERFORMANCE` (default: `true`)
- Lineup controls:
  - `FOOTY_TIPPER_LINEUPS_ENABLED` (default: `true`)
  - `FOOTY_TIPPER_LINEUPS_MODE` (`recent` or `backfill`, default: `recent`)
  - `FOOTY_TIPPER_LINEUPS_MAX_ARTICLES` (default depends on mode)
  - `FOOTY_TIPPER_LINEUPS_BACKFILL_MAX_ARTICLES` (default: `2000`; used by train bootstrap)
  - `FOOTY_TIPPER_LINEUPS_INCLUDE_SITEMAP_IN_RECENT` (default: `false`)
  - `FOOTY_TIPPER_LINEUPS_STRICT` (default: `false`; when true, ingestion errors fail the command)
  - `FOOTY_TIPPER_LINEUPS_AUTO_BACKFILL` (default: `true`; train bootstraps historical lineup scrape when needed)
  - `FOOTY_TIPPER_LINEUPS_AS_OF_HOURS_BEFORE_KICKOFF` (default: `24`; training as-of cutoff for lineup snapshots)
  - `FOOTY_TIPPER_LINEUP_MONTE_CARLO_SAMPLES` (default: `64`; uncertainty marginalization samples)
  - `FOOTY_TIPPER_LINEUP_MU_NOISE_SCALE` (default: `0.12`; score-mean noise scale for uncertainty marginalization)
  - Python deps for scraping: `beautifulsoup4`, `lxml` (missing deps should fail soft unless strict mode is enabled)
- Feed controls:
  - `FOOTY_TIPPER_FEED_SOURCE` (`python` default; `feed` selects legacy XML rollback)
  - `FOOTY_TIPPER_NRL_DATA_ENABLED` (default: `true`)
  - `FOOTY_TIPPER_NRL_DATA_AUTO_BACKFILL` (default: `true`)
  - `FOOTY_TIPPER_NRL_DATA_MAX_PAGES` (optional match-centre cap)
- Legacy XML values, required only for `FOOTY_TIPPER_FEED_SOURCE=feed`:
  - `PASSWORD`
  - `BASE_URL`
  - `NRL_FIXTURES_EXTENTION`
  - `NRL_ROUND_LADDER_EXTENTION`
  - `NRL_PERFORMANCE_EXTENTION`
- Odds values:
  - `BETFAIR_APP_KEY`, `BETFAIR_USERNAME`, `BETFAIR_PASSWORD` (live pre-game odds; optional/fail-soft)
  - `FOOTY_TIPPER_ODDS_HIST_URL` (optional historical workbook override)
- Send/integration values:
  - `FOLDER_ID`, `FOLDER_URL`
  - `ANTHROPIC_API_KEY`, optional `CLAUDE_MODEL` (email copy)
  - `OPENAI_KEY`, optional `OPENAI_MODEL` (banner generation only)
  - `MY_EMAIL`, `EMAIL_PASSWORD`
  - `FOOTY_TIPPER_TEST_EMAIL` (optional; default test recipient is `levon_rush@hotmail.com`)
  - Value-pick/staking controls (optional):
    - `FOOTY_TIPPER_MIN_VALUE_EDGE`
    - `FOOTY_TIPPER_KELLY_FRACTION`
    - `FOOTY_TIPPER_MAX_STAKE_FRACTION`
    - `FOOTY_TIPPER_MIN_STAKE_FRACTION`
    - `FOOTY_TIPPER_STAKE_MODE` (`normalized` or `bankroll`)
    - `FOOTY_TIPPER_BANKROLL` (used for `stake_amount` output)
- Service account token path expected by scripts:
  - `service-account-token.json`

## Behavior to Preserve
- No hardcoded season end year.
- No hardcoded local machine paths in runtime scripts.
- Training and inference datasets are split explicitly by:
  - `game_state_name == "Final"` for training
  - `game_state_name == "Pre Game"` for inference
- Offseason-safe execution:
  - If no pre-game rows exist, send step exits cleanly without failing pipeline.
- Provider-safe execution:
  - Missing Google/Claude/OpenAI dependencies should degrade gracefully where documented (skip/deterministic copy/static banner), not crash the core prediction path.
- Training ownership:
  - GitHub Actions runs prediction/distribution only and passes `--skip-auto-train` in every mode.
  - Missing required model artifacts in Actions must fail clearly.
  - A local model publication must pull Drive state, train, validate with `--skip-auto-train`, then push; do not push after a failed train/validation.
- Lineup-safe execution:
  - Lineup ingestion should fail soft by default.
  - Train/infer must continue if lineup tables are unavailable or sparse.
  - `--lineups-strict` / `FOOTY_TIPPER_LINEUPS_STRICT=true` is the only mode that should fail hard.
  - Historical lineup backfills should prefer repairing existing sparse/zero-entry snapshots over creating duplicate article rows.

## Data and SQL Contracts
- `prediction_table.sql` should always target:
  - latest `competition_year` with pre-game rows
  - minimum `round_id` within that season
- `predictions_table` is upserted by `pipeline/common/sql/insert_into_table.sql`.

## Common Commands
- Preferred CLI:
  - `footy-tipper lineups`
  - `footy-tipper prep`
  - `footy-tipper train`
  - `footy-tipper infer`
  - `footy-tipper send`
  - `footy-tipper send --test --test-email levon_rush@hotmail.com`
  - `footy-tipper predict`
  - `footy-tipper nrl-data refresh|backfill|validate`
  - `footy-tipper odds live|backfill`
  - `footy-tipper evaluate`
  - `footy-tipper site`
  - `footy-tipper state pull|push|gate|schedule`
- Simplicity defaults:
  - `footy-tipper train` should bootstrap historical lineups when needed, then run lineup refresh + prep + training without extra flags.
  - `footy-tipper predict` should run lineup refresh + inference + send workflow without extra flags.
  - `infer`/`predict` should auto-train if required model artifacts are missing (unless explicitly disabled via `--skip-auto-train`), and that auto-train path should inherit lineup bootstrap behavior unless `--skip-lineups` is set.
  - The standalone CLI keeps that convenience behavior, but Actions prediction always supplies `--skip-auto-train` so cloud hardware cannot become a training fallback.
- Local production training:
  - Temporarily disable `predict.yml` to prevent a Drive-state race.
  - Run `state pull`, `FOOTY_TIPPER_TUNE_ITER=100 footy-tipper train`, a no-prep/no-ingestion `infer --skip-auto-train` validation, `state schedule`, then `state push`.
  - Re-enable `predict.yml` even when training fails; request a manual `refresh` only after a successful push.
- Honest evaluation:
  - `footy-tipper evaluate --skip-prep` (nested season-out metrics)
- Static site:
  - `footy-tipper site` (writes docs/site/), `footy-tipper site --publish`
- Data prep only:
  - `Rscript pipeline/data-prep.R`
- Quick syntax checks:
  - `python -m compileall -q pipeline`
  - `Rscript -e "parse(file='pipeline/data-prep.R')"`
- Lineup backfill:
  - `footy-tipper lineups --lineups-mode backfill --start-year 2010 --end-year 2026 --lineups-max-articles 2000`

## Safety / Repo Hygiene
- Never commit secrets (`secrets.env`, service-account token, passwords, API keys).
- Never commit generated model artifacts; publish validated models through Drive state.
- Keep `.gitignore` protections intact.
- Prefer additive, testable changes; avoid broad refactors without smoke checks.
- If changing feature columns, update both:
  - data-prep outputs
  - `pipeline/common/model_training/training_config.py` predictors

## Known Risks
- Performance feed availability can vary by season; when `FOOTY_TIPPER_INCLUDE_PERFORMANCE=true`, missing performance data should fail fast with a clear error.
- Google/Claude/OpenAI integrations are operationally optional for local runs, but production workflows should monitor unexpected skipped integration messages.
