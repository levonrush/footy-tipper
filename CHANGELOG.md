# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Documentation
- Audited repository Markdown against the current nine-command CLI, R preparation modes, SQL contracts, Tier A/B/C model path, line markets, LOSO calibration, dispersion fallback, state sync, and Actions workflows.
- Added task-oriented documentation, research/literature indexes, a curated research-to-production matrix, the complete eleven-part Medium series, and canonical Mermaid diagram sources with SVG previews.
- Corrected the runtime start-year default to `2010`, the test recipient fallback to `levon_rush@hotmail.com`, and provider ownership: Claude/Anthropic writes optional email copy while OpenAI generates an optional banner.
- Documented `pipeline/common/nrl_data/` as a feed-migration prototype that is not yet invoked by the CLI or R preparation.
- Added a private linked Notion project hub; repository Markdown remains canonical.

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
