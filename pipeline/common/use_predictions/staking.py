"""Value-pick selection and Kelly staking.

Head-to-head picks are the year-round section. Line and totals picks are finals
extras: with one to four games there is room to price the other two markets, and
the simulation already produced the cover probabilities they need.
"""

import math
import os

import pandas as pd

from pipeline.common.odds.validity import valid_decimal_odds
from pipeline.common.use_predictions.probabilities import two_way_home_probability

MARKET_PICK_COLUMNS = [
    "game_id",
    "fixture",
    "market",
    "selection",
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


def _staking_config(prod_run=False):
    """Resolve the shared value/staking knobs once."""
    min_edge_default = 0.03 if prod_run else 0.02
    stake_mode = os.getenv("FOOTY_TIPPER_STAKE_MODE", "normalized").strip().lower()
    if stake_mode not in {"normalized", "bankroll"}:
        stake_mode = "normalized"

    bankroll = None
    bankroll_env = os.getenv("FOOTY_TIPPER_BANKROLL", "")
    if bankroll_env.strip():
        try:
            bankroll_value = float(bankroll_env)
            if bankroll_value > 0:
                bankroll = bankroll_value
        except ValueError:
            bankroll = None

    return {
        "min_edge": float(os.getenv("FOOTY_TIPPER_MIN_VALUE_EDGE", str(min_edge_default))),
        "kelly_multiplier": float(os.getenv("FOOTY_TIPPER_KELLY_FRACTION", "0.5")),
        "max_stake_fraction": float(os.getenv("FOOTY_TIPPER_MAX_STAKE_FRACTION", "0.05")),
        "min_stake_fraction": float(os.getenv("FOOTY_TIPPER_MIN_STAKE_FRACTION", "0.0")),
        "stake_mode": stake_mode,
        "bankroll": bankroll,
    }


def _kelly(probability, odds, config):
    """Edge and bounded Kelly stake for one priced selection, or None."""
    if (
        probability is None
        or pd.isna(probability)
        or not math.isfinite(float(probability))
        or not valid_decimal_odds(odds)
        or probability <= 0
        or probability >= 1
    ):
        return None
    probability = float(probability)
    odds = float(odds)
    edge = (probability * odds) - 1.0
    denominator = odds - 1.0
    kelly_full = max(0.0, edge / denominator if denominator > 0 else 0.0)
    kelly_fractional = max(0.0, kelly_full * config["kelly_multiplier"])
    kelly_capped = min(config["max_stake_fraction"], kelly_fractional)
    if kelly_capped < config["min_stake_fraction"]:
        kelly_capped = 0.0
    return {
        "price": odds,
        "price_min": 1 / probability,
        "model_prob": probability,
        "edge": edge,
        "kelly_full": kelly_full,
        "kelly_fraction": kelly_fractional,
        "kelly_capped_fraction": kelly_capped,
    }


def _finalise(records, config, output_columns):
    """Normalise stakes across the selected picks and order them."""
    if not records:
        return pd.DataFrame(columns=output_columns)

    picks = pd.DataFrame.from_records(records)
    if config["stake_mode"] == "normalized":
        total_weight = float(picks["kelly_capped_fraction"].sum())
        picks["stake_fraction"] = (
            picks["kelly_capped_fraction"] / total_weight if total_weight > 0 else 0.0
        )
    else:
        picks["stake_fraction"] = picks["kelly_capped_fraction"]

    if config["bankroll"] is not None:
        picks["stake_amount"] = picks["stake_fraction"] * config["bankroll"]
    else:
        picks["stake_amount"] = pd.NA

    picks = picks.sort_values(["stake_fraction", "edge"], ascending=False).reset_index(drop=True)
    return picks[output_columns]


def _numeric(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def get_market_picks(predictions, distributions=None, prod_run=False):
    """Line and totals value picks, priced off the simulated distribution.

    Returns an empty frame whenever the distribution summaries are missing, which
    is the normal state until inference has run since the table was introduced.
    """
    if (
        predictions is None
        or getattr(predictions, "empty", True)
        or distributions is None
        or getattr(distributions, "empty", True)
        or "game_id" not in distributions.columns
    ):
        return pd.DataFrame(columns=MARKET_PICK_COLUMNS)

    config = _staking_config(prod_run)
    by_game = {
        int(row["game_id"]): row
        for _, row in distributions.iterrows()
        if pd.notna(row.get("game_id"))
    }

    records = []
    for _, row in predictions.iterrows():
        game_id = row.get("game_id")
        if pd.isna(game_id) or int(game_id) not in by_game:
            continue
        summary = by_game[int(game_id)]
        fixture = f"{row.get('team_home')} v {row.get('team_away')}"
        for candidate in _line_candidates(row, summary) + _total_candidates(row, summary):
            priced = _kelly(candidate["probability"], candidate["odds"], config)
            if priced is None:
                continue
            if priced["edge"] < config["min_edge"] or priced["kelly_capped_fraction"] <= 0:
                continue
            records.append(
                {
                    "game_id": int(game_id),
                    "fixture": fixture,
                    "market": candidate["market"],
                    "selection": candidate["selection"],
                    **priced,
                }
            )
    return _finalise(records, config, MARKET_PICK_COLUMNS)


def _line_candidates(row, summary):
    """Both sides of the handicap, when a fresh line and a cover probability exist."""
    cover = summary.get("p_home_covers_line")
    handicap = _numeric(row.get("team_line_amount_home"))
    if cover is None or pd.isna(cover) or pd.isna(handicap):
        return []
    return [
        {
            "market": "Line",
            "selection": f"{row.get('team_home')} {handicap:+g}",
            "probability": float(cover),
            "odds": _numeric(row.get("team_line_odds_home")),
        },
        {
            "market": "Line",
            "selection": f"{row.get('team_away')} {-handicap:+g}",
            "probability": 1.0 - float(cover),
            "odds": _numeric(row.get("team_line_odds_away")),
        },
    ]


def _total_candidates(row, summary):
    over = summary.get("p_total_over")
    total_line = _numeric(row.get("total_line"))
    if over is None or pd.isna(over) or pd.isna(total_line):
        return []
    return [
        {
            "market": "Total",
            "selection": f"Over {total_line:g}",
            "probability": float(over),
            "odds": _numeric(row.get("total_over_odds")),
        },
        {
            "market": "Total",
            "selection": f"Under {total_line:g}",
            "probability": 1.0 - float(over),
            "odds": _numeric(row.get("total_under_odds")),
        },
    ]


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

    config = _staking_config(prod_run)
    min_edge = config["min_edge"]

    predictions = predictions.copy()

    # Use expected value (p * odds - 1) for the model's predicted winner only.
    # Only tips the model expects to win are eligible as value picks.
    records = []
    for _, row in predictions.iterrows():
        game_id = row.get("game_id")
        home_team = row.get("team_home")
        away_team = row.get("team_away")
        # Two-way, to match the market price it is about to be compared against:
        # a decimal H2H price is a two-way quote, so pricing against the
        # draw-deflated probability understates every edge by the draw mass.
        home_prob = two_way_home_probability(
            row.get("home_team_win_prob"), row.get("home_team_lose_prob")
        )
        away_prob = None if pd.isna(home_prob) else 1.0 - home_prob
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

            priced = _kelly(prob, odds, config)
            if priced is None:
                continue

            side_candidates.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "opponent": opp,
                    "side": side,
                    **priced,
                }
            )

        if not side_candidates:
            continue

        best = max(side_candidates, key=lambda x: x["edge"])
        if best["edge"] >= min_edge and best["kelly_capped_fraction"] > 0:
            records.append(best)

    return _finalise(records, config, output_columns)
