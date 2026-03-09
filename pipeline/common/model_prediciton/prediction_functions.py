from __future__ import annotations

import sqlite3

import dill as pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam


def get_inference_data(db_path, sql_file):
    """Retrieve data for inference from an SQLite database."""
    print("Getting inference data...")
    con = sqlite3.connect(str(db_path))
    with open(sql_file, "r") as file:
        query = file.read()
    inference_data = pd.read_sql_query(query, con)
    con.close()
    return inference_data


def get_table_data(db_path, table_name):
    """Read a full table from SQLite."""
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
    con.close()
    return df


def predict_scores(model, data):
    """Predict expected scores."""
    return model.predict(data)


def compute_outcome_probs_independent(mu_home, mu_away):
    draw_prob = float(skellam.pmf(0, mu_home, mu_away))
    home_win_prob = float(1.0 - skellam.cdf(0, mu_home, mu_away))
    away_win_prob = float(max(0.0, 1.0 - home_win_prob - draw_prob))
    return home_win_prob, away_win_prob, draw_prob


def conditional_home_win_prob(mu_home, mu_away):
    home_win, away_win, _ = compute_outcome_probs_independent(mu_home, mu_away)
    non_draw = max(1e-9, home_win + away_win)
    return home_win / non_draw


def marginalized_conditional_home_win_prob(
    mu_home,
    mu_away,
    lineup_uncertainty_home=0.0,
    lineup_uncertainty_away=0.0,
    n_samples=64,
    mu_noise_scale=0.12,
    rng=None,
):
    """
    Approximate p(home win | non-draw) by marginalising over lineup uncertainty.

    The uncertainty terms are expected to be in [0, 0.25] from p(1-p) style
    features. We convert them to multiplicative score-mean noise and average the
    conditional win probabilities across Monte Carlo draws.
    """
    base = conditional_home_win_prob(mu_home, mu_away)
    n_samples = int(max(1, n_samples))
    mu_noise_scale = float(max(0.0, mu_noise_scale))
    if n_samples <= 1 or mu_noise_scale <= 0:
        return base

    if rng is None:
        rng = np.random.default_rng()

    uh = float(max(0.0, lineup_uncertainty_home))
    ua = float(max(0.0, lineup_uncertainty_away))
    std_home = mu_noise_scale * np.sqrt(uh)
    std_away = mu_noise_scale * np.sqrt(ua)
    if std_home <= 1e-9 and std_away <= 1e-9:
        return base

    # Lognormal multipliers keep score means positive and centred near 1.0.
    mult_home = np.exp(rng.normal(loc=-0.5 * (std_home ** 2), scale=std_home, size=n_samples))
    mult_away = np.exp(rng.normal(loc=-0.5 * (std_away ** 2), scale=std_away, size=n_samples))

    probs = []
    for mh, ma in zip(mult_home, mult_away):
        sample_mu_home = max(1e-6, float(mu_home) * float(mh))
        sample_mu_away = max(1e-6, float(mu_away) * float(ma))
        probs.append(conditional_home_win_prob(sample_mu_home, sample_mu_away))

    return float(np.mean(probs)) if probs else base


def derive_market_home_probability(df: pd.DataFrame) -> np.ndarray:
    """Get market-implied home win probability with robust fallbacks.

    Preference order: Shin-adjusted → power-normalised → basic-normalised → raw odds ratio.
    Shin (1993) adjustment is most principled as it corrects for insider-trading bias
    in the bookmaker overround.
    """
    p = pd.Series(np.nan, index=df.index)

    if "home_market_prob_shin" in df.columns:
        p = p.fillna(pd.to_numeric(df["home_market_prob_shin"], errors="coerce"))

    if "home_market_prob_power" in df.columns:
        p = p.fillna(pd.to_numeric(df["home_market_prob_power"], errors="coerce"))

    if "home_market_prob_basic" in df.columns:
        p = p.fillna(pd.to_numeric(df["home_market_prob_basic"], errors="coerce"))

    if "team_head_to_head_odds_home" in df.columns and "team_head_to_head_odds_away" in df.columns:
        home_odds = pd.to_numeric(df["team_head_to_head_odds_home"], errors="coerce")
        away_odds = pd.to_numeric(df["team_head_to_head_odds_away"], errors="coerce")
        qh = 1.0 / home_odds
        qa = 1.0 / away_odds
        p_basic = qh / (qh + qa)
        p = p.fillna(p_basic)

    return np.clip(pd.to_numeric(p, errors="coerce").fillna(0.5).to_numpy(dtype=float), 1e-6, 1 - 1e-6)


def simulate_game(home_score_avg, away_score_avg, n_simulations=100000, lambda3=0.0, rng=None):
    """Simulate outcomes and scoreline under independent or bivariate Poisson."""
    if rng is None:
        rng = np.random.default_rng()

    home_score_avg = float(max(home_score_avg, 1e-9))
    away_score_avg = float(max(away_score_avg, 1e-9))
    lambda3 = float(max(lambda3, 0.0))

    if lambda3 > 0:
        shared = min(lambda3, 0.95 * min(home_score_avg, away_score_avg))
        lam1 = max(home_score_avg - shared, 1e-9)
        lam2 = max(away_score_avg - shared, 1e-9)
        shared_sim = rng.poisson(shared, size=n_simulations)
        home_goals_sim = rng.poisson(lam1, size=n_simulations) + shared_sim
        away_goals_sim = rng.poisson(lam2, size=n_simulations) + shared_sim
    else:
        home_goals_sim = rng.poisson(home_score_avg, size=n_simulations)
        away_goals_sim = rng.poisson(away_score_avg, size=n_simulations)

    home_wins = (home_goals_sim > away_goals_sim).sum()
    away_wins = (home_goals_sim < away_goals_sim).sum()
    draws = (home_goals_sim == away_goals_sim).sum()

    total_games = float(n_simulations)
    probabilities = {
        "home_win_prob": home_wins / total_games,
        "away_win_prob": away_wins / total_games,
        "draw_prob": draws / total_games,
    }

    scorelines = list(zip(home_goals_sim, away_goals_sim))
    predicted_scoreline = max(set(scorelines), key=scorelines.count)

    return probabilities, predicted_scoreline


def calculate_bayes_factor(probabilities):
    home_win_prob = probabilities["home_win_prob"]
    away_win_prob = probabilities["away_win_prob"]
    return home_win_prob / away_win_prob if away_win_prob != 0 else np.inf


def map_bayes_factor_to_evidence(bayes_factor):
    if bayes_factor < 1:
        return "Negative evidence"
    if 1 <= bayes_factor < 3:
        return "Anecdotal evidence"
    if 3 <= bayes_factor < 10:
        return "Moderate evidence"
    if 10 <= bayes_factor < 30:
        return "Strong evidence"
    if 30 <= bayes_factor < 100:
        return "Very strong evidence"
    return "Decisive evidence"


def predict_match_outcome_and_scoreline_with_bayes(
    home_model=None,
    away_model=None,
    inference_data=None,
    predictors=None,
    n_simulations=100000,
    mu_home=None,
    mu_away=None,
    lambda3=0.0,
    calibrated_home_win_conditional=None,
):
    """
    Predict match outcomes and scorelines.

    Backward compatible mode:
    - pass home_model/away_model/inference_data/predictors.

    Enhanced mode:
    - pass inference_data with precomputed mu_home/mu_away arrays and optional
      calibrated_home_win_conditional.
    """
    if inference_data is None:
        raise ValueError("inference_data is required.")

    if inference_data.empty:
        empty_outcomes = pd.DataFrame(
            columns=[
                "game_id",
                "home_team_result",
                "home_team_win_prob",
                "home_team_lose_prob",
                "draw_prob",
                "bayes_factor",
                "evidence_strength",
            ]
        )
        empty_margins = pd.DataFrame(
            columns=["game_id", "predicted_home_score", "predicted_away_score", "predicted_margin"]
        )
        return empty_outcomes, empty_margins

    working = inference_data.copy().reset_index(drop=True)

    if mu_home is None or mu_away is None:
        if home_model is None or away_model is None or predictors is None:
            raise ValueError("Need either precomputed mu arrays or models + predictors.")
        working["home_goals_avg"] = predict_scores(home_model, working[predictors])
        working["away_goals_avg"] = predict_scores(away_model, working[predictors])
    else:
        working["home_goals_avg"] = np.asarray(mu_home, dtype=float)
        working["away_goals_avg"] = np.asarray(mu_away, dtype=float)

    if calibrated_home_win_conditional is None:
        calibrated_home_win_conditional = np.full(len(working), np.nan)
    calibrated_home_win_conditional = np.asarray(calibrated_home_win_conditional, dtype=float)

    rng = np.random.default_rng()
    results = []
    for idx, row in working.iterrows():
        probabilities, predicted_scoreline = simulate_game(
            row["home_goals_avg"],
            row["away_goals_avg"],
            n_simulations=n_simulations,
            lambda3=lambda3,
            rng=rng,
        )

        calibrated_cond = calibrated_home_win_conditional[idx]
        if not np.isnan(calibrated_cond):
            calibrated_cond = float(np.clip(calibrated_cond, 1e-6, 1 - 1e-6))
            non_draw = max(0.0, 1.0 - probabilities["draw_prob"])
            probabilities["home_win_prob"] = calibrated_cond * non_draw
            probabilities["away_win_prob"] = (1.0 - calibrated_cond) * non_draw

        home_team_result = "Win" if probabilities["home_win_prob"] > probabilities["away_win_prob"] else "Loss"
        bayes_factor = calculate_bayes_factor(probabilities)
        evidence_strength = map_bayes_factor_to_evidence(bayes_factor)

        results.append(
            {
                "game_id": row["game_id"],
                "home_team_win_prob": probabilities["home_win_prob"],
                "home_team_lose_prob": probabilities["away_win_prob"],
                "draw_prob": probabilities["draw_prob"],
                "predicted_home_score": predicted_scoreline[0],
                "predicted_away_score": predicted_scoreline[1],
                "predicted_margin": predicted_scoreline[0] - predicted_scoreline[1],
                "home_team_result": home_team_result,
                "bayes_factor": bayes_factor,
                "evidence_strength": evidence_strength,
            }
        )

    results_df = pd.DataFrame(results)
    outcome_df = results_df[
        [
            "game_id",
            "home_team_result",
            "home_team_win_prob",
            "home_team_lose_prob",
            "draw_prob",
            "bayes_factor",
            "evidence_strength",
        ]
    ]
    margin_df = results_df[["game_id", "predicted_home_score", "predicted_away_score", "predicted_margin"]]
    return outcome_df, margin_df


def get_predictions(db_path, sql_file):
    con = sqlite3.connect(str(db_path))
    with open(sql_file, "r") as file:
        query = file.read()
    predictions = pd.read_sql_query(query, con)
    con.close()
    return predictions


def load_models(model_name, project_root):
    model_path = project_root / "models" / f"{model_name}.pkl"
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        print(f"{model_name} model pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        raise
    return pipeline


def _ensure_prediction_table_columns(con):
    expected_columns = {
        "draw_prob": "REAL",
        "bayes_factor": "REAL",
        "evidence_strength": "TEXT",
        "predicted_home_score": "INTEGER",
        "predicted_away_score": "INTEGER",
        "predicted_margin": "INTEGER",
    }
    existing_columns = {row[1] for row in con.execute("PRAGMA table_info(predictions_table)").fetchall()}
    for column_name, column_ddl in expected_columns.items():
        if column_name not in existing_columns:
            con.execute(f"ALTER TABLE predictions_table ADD COLUMN {column_name} {column_ddl}")


def save_predictions_to_db(predictions_df, db_path, create_table_sql_file, insert_into_table_sql_file):
    print("Saving predictions to database...")
    con = sqlite3.connect(str(db_path))

    with open(create_table_sql_file, "r") as file:
        create_table_query = file.read()
    con.execute(create_table_query)
    _ensure_prediction_table_columns(con)

    with open(insert_into_table_sql_file, "r") as file:
        insert_into_table_query = file.read()

    for _, row in predictions_df.iterrows():
        con.execute(
            insert_into_table_query,
            (
                row["game_id"],
                row["home_team_result"],
                row["home_team_win_prob"],
                row["home_team_lose_prob"],
                row["draw_prob"],
                row["bayes_factor"],
                row["evidence_strength"],
                row["predicted_home_score"],
                row["predicted_away_score"],
                row["predicted_margin"],
            ),
        )

    con.commit()
    con.close()
