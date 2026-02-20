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
  - `footy-tipper-train.py`: runs R data prep, then Python training.
  - `footy-tipper-predict.py`: runs R data prep, inference, then send.
- Data prep:
  - `pipeline/data-prep.R` sources `pipeline/common/data-prep/*.R`.
  - Writes three SQLite tables to `data/footy-tipper-db.sqlite`:
    - `footy_tipping_data`
    - `training_data`
    - `inference_data`
- Training:
  - `pipeline/train.py` builds Tier-A baseline features, trains Tier-B home/away Poisson models, blends expected scores, fits stacker + beta calibrator, estimates `lambda3`, and saves artifacts in `models/`.
  - Key artifacts: `home_model.pkl`, `away_model.pkl`, `stacker.pkl`, `win_prob_calibrator.pkl`, `model_manifest.json`.
- Inference:
  - `pipeline/inference.py` loads artifacts + manifest, rebuilds Tier-A baseline context, applies blend/stack/calibration, simulates outcomes with bivariate Poisson (`lambda3`), and upserts into `predictions_table`.
- Distribution:
  - `pipeline/send.py` reads prediction view via SQL, computes EV-based value picks with Kelly-derived staking, and handles upload/email.

## Critical Runtime Config
- Season controls:
  - `FOOTY_TIPPER_START_YEAR` (default: `2018`)
  - `FOOTY_TIPPER_END_YEAR` (default: current year)
  - `FOOTY_TIPPER_INCLUDE_PERFORMANCE` (default: `true`)
- Feed/API environment values expected in `secrets.env`:
  - `PASSWORD`
  - `BASE_URL`
  - `NRL_FIXTURES_EXTENTION`
  - `NRL_ROUND_LADDER_EXTENTION`
  - `NRL_PERFORMANCE_EXTENTION`
- Send/integration values:
  - `FOLDER_ID`, `FOLDER_URL`
  - `OPENAI_KEY`, optional `OPENAI_MODEL`
  - `MY_EMAIL`, `EMAIL_PASSWORD`
  - `FOOTY_TIPPER_TEST_EMAIL` (optional; default test recipient is `levon.rush@gmail.com`)
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
  - Missing Google/OpenAI dependencies should degrade gracefully (skip/fallback), not crash.

## Data and SQL Contracts
- `prediction_table.sql` should always target:
  - latest `competition_year` with pre-game rows
  - minimum `round_id` within that season
- `predictions_table` is upserted by `pipeline/common/sql/insert_into_table.sql`.

## Common Commands
- Preferred CLI:
  - `footy-tipper prep`
  - `footy-tipper train`
  - `footy-tipper infer`
  - `footy-tipper send`
  - `footy-tipper send --test --test-email levon.rush@gmail.com`
  - `footy-tipper predict`
- Train:
  - `python footy-tipper-train.py`
- Predict + send:
  - `python footy-tipper-predict.py`
- Data prep only:
  - `Rscript pipeline/data-prep.R`
- Quick syntax checks:
  - `python -m compileall -q pipeline footy-tipper-train.py footy-tipper-predict.py`
  - `Rscript -e "parse(file='pipeline/data-prep.R')"`

## Safety / Repo Hygiene
- Never commit secrets (`secrets.env`, service-account token, passwords, API keys).
- Keep `.gitignore` protections intact.
- Prefer additive, testable changes; avoid broad refactors without smoke checks.
- If changing feature columns, update both:
  - data-prep outputs
  - `pipeline/common/model_training/training_config.py` predictors

## Known Risks
- Performance feed availability can vary by season; when `FOOTY_TIPPER_INCLUDE_PERFORMANCE=true`, missing performance data should fail fast with a clear error.
- Google/OpenAI integrations are operationally optional for local runs, but production workflows should monitor skipped-send messages.
