# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-02-18

### Added
- Added `CHANGELOG.md` and `AGENTS.md` for project history and agent/operator guidance.
- Added first-class CLI entrypoint (`footy-tipper`) with subcommands:
  - `prep`, `train`, `infer`, `send`, `predict`
- Added packaging entrypoint (`pyproject.toml`) so `footy-tipper` is available as an installed command via `pip install -e .`.
- Added dynamic season controls:
  - `FOOTY_TIPPER_START_YEAR` (default `2018`)
  - `FOOTY_TIPPER_END_YEAR` (defaults to current calendar year)
  - `FOOTY_TIPPER_INCLUDE_PERFORMANCE` (default `true`)
- Added test-send workflow with single-recipient mode (default test address: `levon.rush@gmail.com`).
- Added dedicated CLI documentation (`CLI.md`).
- Added fallback email generation when OpenAI is unavailable or not configured.
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
- Legacy docs `CLI.md` and `cli/README.md` now act as pointers to `docs/` pages.

### Fixed
- Fixed `get_predictions` missing return in prediction utilities (`pipeline/common/model_prediciton/prediction_functions.py`).
- Fixed inference path to safely handle empty inference datasets.
- Fixed venue lumping assignment to ensure the model uses the intended transformed `venue_name` field (`pipeline/common/data-prep/clean-data.R`).
- Fixed turnaround feature leakage across seasons by grouping by team and competition year (`pipeline/common/data-prep/feature-engineering.R`).
- Fixed `matchup_form` to honor `form_period` instead of hardcoded window.
- Fixed fixture home/away split logic to avoid reliance on implicit group ordering (`pipeline/common/data-prep/get-data.R`).
- Fixed `install.R` to actually install required R packages.

### Reliability / Hardening
- Fixture fetch now skips unavailable future years instead of hard failing entire runs (`pipeline/common/data-prep/get-data.R`).
- Pipeline now fails fast with clear messages for missing required training rows or missing performance feed when performance is enabled.
- `sending_functions.py` now gracefully degrades when optional providers (Google/OpenAI) are unavailable.

### Notes
- Current behavior is now season-rollover safe for 2026+ without code edits, as long as source feeds are available.
