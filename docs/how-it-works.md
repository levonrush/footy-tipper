# How It Works

This page explains the end-to-end pipeline and the data contracts that keep it stable.

## High-Level Flow

```mermaid
flowchart TB
  subgraph O[Orchestration + Data Prep]
    CLI[CLI + Entrypoints] --> PREP[R Data Prep]
    PREP --> DB[(SQLite)]
  end

  subgraph M[Model Training + Inference]
    DB --> TRAIN[Train Tiered Models]
    TRAIN --> ART[Model Artifacts]
    DB --> INFER[Inference]
    ART --> INFER
  end

  subgraph S[Delivery]
    INFER --> PRED[predictions_table]
    PRED --> SEND[Value Picks + Joker + Email]
  end
```

Mermaid source file: `images/workflow.mmd`

## Data Prep (`pipeline/data-prep.R`)

Data prep writes these tables in `data/footy-tipper-db.sqlite`:
- `footy_tipping_data`: full context table
- `training_data`: `game_state_name == "Final"`
- `inference_data`: `game_state_name == "Pre Game"`

This split is a core safety feature. It prevents accidental train/infer leakage.

## Training (`pipeline/train.py`)

Training currently does this:
1. Build Tier-A baseline features.
2. Train Tier-B home/away Poisson score models.
3. Blend Tier-A and Tier-B score expectations.
4. Estimate shared score covariance (`lambda3`) for bivariate simulation.
5. Fit win-probability stacker + beta calibrator.
6. Save model artifacts to `models/`.
7. Run joker backtest simulations and save `models/joker_policy.json`.

Main artifacts:
- `models/home_model.pkl`
- `models/away_model.pkl`
- `models/stacker.pkl`
- `models/win_prob_calibrator.pkl`
- `models/model_manifest.json`
- `models/joker_policy.json`

## Inference (`pipeline/inference.py`)

Inference:
1. Loads `inference_data`.
2. Rebuilds Tier-A context.
3. Loads manifest + trained artifacts.
4. Produces blended expected scores.
5. Applies stack + calibration to home win probabilities.
6. Simulates outcomes/scorelines with bivariate Poisson.
7. Upserts into `predictions_table`.

## Send Layer (`pipeline/send.py`)

Send does three decisions:
- winner/probability communication
- value picks and stake sizing
- joker recommendation for the current round

Outputs can be uploaded to Drive and sent via email list/test email.

## Why This Architecture

- R handles data wrangling and feature engineering where your pipeline already has history.
- Python handles model fitting/inference/artifact lifecycle.
- SQLite gives a transparent, debuggable state store.
- Each phase is runnable independently but still works as one weekly pipeline.
