"""Value-pick selection and Kelly staking."""

import math
import os

import pandas as pd

from pipeline.common.odds.validity import valid_decimal_odds


# The 'get_tipper_picks' function calculates the odds thresholds and returns a DataFrame of tipper picks.
def get_tipper_picks(predictions, prod_run=False):
    output_columns = [
        "game_id",
        "team",
        "opponent",
        "side",
        "price",
        "price_min",
        "model_prob",
        "edge",
        "kelly_full",
        "kelly_fraction",
        "kelly_capped_fraction",
        "stake_fraction",
        "stake_amount",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=output_columns)

    min_edge_default = 0.03 if prod_run else 0.02
    min_edge = float(os.getenv("FOOTY_TIPPER_MIN_VALUE_EDGE", str(min_edge_default)))
    kelly_multiplier = float(os.getenv("FOOTY_TIPPER_KELLY_FRACTION", "0.5"))
    max_stake_fraction = float(os.getenv("FOOTY_TIPPER_MAX_STAKE_FRACTION", "0.05"))
    min_stake_fraction = float(os.getenv("FOOTY_TIPPER_MIN_STAKE_FRACTION", "0.0"))
    stake_mode = os.getenv("FOOTY_TIPPER_STAKE_MODE", "normalized").strip().lower()
    if stake_mode not in {"normalized", "bankroll"}:
        stake_mode = "normalized"
    bankroll_env = os.getenv("FOOTY_TIPPER_BANKROLL", "")

    bankroll = None
    if bankroll_env.strip():
        try:
            bankroll_value = float(bankroll_env)
            if bankroll_value > 0:
                bankroll = bankroll_value
        except ValueError:
            bankroll = None

    predictions = predictions.copy()

    # Use expected value (p * odds - 1) for the model's predicted winner only.
    # Only tips the model expects to win are eligible as value picks.
    records = []
    for _, row in predictions.iterrows():
        game_id = row.get("game_id")
        home_team = row.get("team_home")
        away_team = row.get("team_away")
        home_prob = pd.to_numeric(pd.Series([row.get("home_team_win_prob")]), errors="coerce").iloc[0]
        away_prob = pd.to_numeric(pd.Series([row.get("home_team_lose_prob")]), errors="coerce").iloc[0]
        home_odds = pd.to_numeric(pd.Series([row.get("team_head_to_head_odds_home")]), errors="coerce").iloc[0]
        away_odds = pd.to_numeric(pd.Series([row.get("team_head_to_head_odds_away")]), errors="coerce").iloc[0]

        side_candidates = []
        predicted_result = row.get("home_team_result")
        for side, team, opp, prob, odds in [
            ("home", home_team, away_team, home_prob, home_odds),
            ("away", away_team, home_team, away_prob, away_odds),
        ]:
            # Only evaluate sides the model tips to win
            if side == "home" and predicted_result != "Win":
                continue
            if side == "away" and predicted_result != "Loss":
                continue

            if (
                pd.isna(prob)
                or not math.isfinite(float(prob))
                or not valid_decimal_odds(odds)
                or prob <= 0
                or prob >= 1
            ):
                continue

            fair_odds = 1 / prob
            edge = (prob * odds) - 1.0
            denominator = odds - 1.0
            kelly_full = edge / denominator if denominator > 0 else 0.0
            kelly_full = max(0.0, kelly_full)
            kelly_fractional = max(0.0, kelly_full * kelly_multiplier)
            kelly_capped = min(max_stake_fraction, kelly_fractional)
            if kelly_capped < min_stake_fraction:
                kelly_capped = 0.0

            side_candidates.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "opponent": opp,
                    "side": side,
                    "price": odds,
                    "price_min": fair_odds,
                    "model_prob": prob,
                    "edge": edge,
                    "kelly_full": kelly_full,
                    "kelly_fraction": kelly_fractional,
                    "kelly_capped_fraction": kelly_capped,
                }
            )

        if not side_candidates:
            continue

        best = max(side_candidates, key=lambda x: x["edge"])
        if best["edge"] >= min_edge and best["kelly_capped_fraction"] > 0:
            records.append(best)

    if not records:
        return pd.DataFrame(columns=output_columns)

    tipper_picks = pd.DataFrame.from_records(records)
    if stake_mode == "normalized":
        total_weight = float(tipper_picks["kelly_capped_fraction"].sum())
        if total_weight > 0:
            tipper_picks["stake_fraction"] = tipper_picks["kelly_capped_fraction"] / total_weight
        else:
            tipper_picks["stake_fraction"] = 0.0
    else:
        tipper_picks["stake_fraction"] = tipper_picks["kelly_capped_fraction"]

    if bankroll is not None:
        tipper_picks["stake_amount"] = tipper_picks["stake_fraction"] * bankroll
    else:
        tipper_picks["stake_amount"] = pd.NA

    tipper_picks = tipper_picks.sort_values(["stake_fraction", "edge"], ascending=False).reset_index(drop=True)
    return tipper_picks[output_columns]
