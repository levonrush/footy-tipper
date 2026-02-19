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
- README updated to reflect current token filename, script behavior, and season configuration controls.

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
