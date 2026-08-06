"""Season-to-date tipping results for the email and site.

Joins stored predictions to completed (Final) games so readers can see how
the model has actually been going: last round record, season record, and the
market-favourite benchmark on the same games.
"""

import sqlite3

import pandas as pd

from pipeline.common.use_predictions.probabilities import (
    tipped_home as _tipped_home_from_probs,
    two_way_home_probability,
)

_SCOREBOARD_QUERY = """
SELECT
    CAST(ft.game_id AS INTEGER) AS game_id,
    CAST(ft.competition_year AS INTEGER) AS competition_year,
    CAST(ft.round_id AS INTEGER) AS round_id,
    ft.round_name,
    ft.team_home,
    ft.team_away,
    CAST(ft.team_final_score_home AS REAL) AS team_final_score_home,
    CAST(ft.team_final_score_away AS REAL) AS team_final_score_away,
    CAST(ft.team_head_to_head_odds_home AS REAL) AS odds_home,
    CAST(ft.team_head_to_head_odds_away AS REAL) AS odds_away,
    p.home_team_win_prob,
    p.home_team_lose_prob,
    p.home_team_result
FROM predictions_table p
JOIN footy_tipping_data ft ON ft.game_id = p.game_id
WHERE ft.game_state_name = 'Final'
  AND ft.competition_year = (
      SELECT MAX(CAST(competition_year AS INTEGER))
      FROM footy_tipping_data ft2
      JOIN predictions_table p2 ON p2.game_id = ft2.game_id
      WHERE ft2.game_state_name = 'Final'
  )
"""


def get_season_scoreboard(db_path):
    """Return season-to-date tipping results, or None when nothing is settled.

    Result dict keys:
        competition_year, season_games, season_correct, season_accuracy,
        market_games, market_correct, market_accuracy (None without odds),
        last_round_id, last_round_name, last_round_games, last_round_correct.
    """
    try:
        con = sqlite3.connect(str(db_path))
        try:
            settled = pd.read_sql_query(_SCOREBOARD_QUERY, con)
        finally:
            con.close()
    except Exception as exc:
        print(f"Scoreboard query failed ({exc}).")
        return None

    if settled.empty:
        return None

    settled = settled.dropna(subset=["home_team_win_prob", "team_final_score_home", "team_final_score_away"])
    # Draws can't be tipped correctly or incorrectly in a two-way comp; skip them.
    settled = settled[settled["team_final_score_home"] != settled["team_final_score_away"]]
    if settled.empty:
        return None

    home_won = settled["team_final_score_home"] > settled["team_final_score_away"]
    # Use the stored tip (home_team_result) — it's what was actually emailed.
    model_tipped_home = settled["home_team_result"].astype(str).str.strip().eq("Win")
    no_result = ~settled["home_team_result"].astype(str).str.strip().isin(["Win", "Loss"])
    if no_result.any():
        # Compare the two sides rather than testing the home probability against
        # 0.5: both carry the same (1 - draw) factor, so `win > 0.5` would hand
        # the away team any game whose conditional sits just above a half.
        fallback = _tipped_home_from_probs(
            settled["home_team_win_prob"], settled["home_team_lose_prob"]
        )
        model_tipped_home = model_tipped_home.where(~no_result, fallback)
    settled = settled.assign(model_correct=(model_tipped_home == home_won))

    has_odds = (settled["odds_home"] > 1.0) & (settled["odds_away"] > 1.0)
    market_tipped_home = settled["odds_home"] < settled["odds_away"]
    market_correct = (market_tipped_home == home_won) & has_odds

    season_games = int(len(settled))
    season_correct = int(settled["model_correct"].sum())
    market_games = int(has_odds.sum())

    last_round_id = int(settled["round_id"].max())
    last_round = settled[settled["round_id"] == last_round_id]
    last_round_name = None
    if not last_round.empty:
        name_value = last_round.iloc[0].get("round_name")
        if pd.notna(name_value) and str(name_value).strip():
            last_round_name = str(name_value).strip()
    if not last_round_name:
        last_round_name = f"Round {last_round_id}"

    return {
        "competition_year": int(settled.iloc[0]["competition_year"]),
        "season_games": season_games,
        "season_correct": season_correct,
        "season_accuracy": season_correct / season_games,
        "market_games": market_games,
        "market_correct": int(market_correct.sum()),
        "market_accuracy": (int(market_correct.sum()) / market_games) if market_games > 0 else None,
        "last_round_id": last_round_id,
        "last_round_name": last_round_name,
        "last_round_games": int(len(last_round)),
        "last_round_correct": int(last_round["model_correct"].sum()),
    }


def get_season_results(db_path):
    """Per-game settled results for the season results page.

    Returns a DataFrame (newest round first) with the tip, the actual result,
    and whether the tip landed — or an empty frame when nothing is settled.
    """
    try:
        con = sqlite3.connect(str(db_path))
        try:
            settled = pd.read_sql_query(_SCOREBOARD_QUERY, con)
        finally:
            con.close()
    except Exception as exc:
        print(f"Season results query failed ({exc}).")
        return pd.DataFrame()

    if settled.empty:
        return settled

    settled = settled.dropna(subset=["home_team_win_prob", "team_final_score_home", "team_final_score_away"])
    if settled.empty:
        return settled

    home_won = settled["team_final_score_home"] > settled["team_final_score_away"]
    draw = settled["team_final_score_home"] == settled["team_final_score_away"]
    tipped_home = settled["home_team_result"].astype(str).str.strip().eq("Win")
    home_prob = two_way_home_probability(
        settled["home_team_win_prob"], settled["home_team_lose_prob"]
    )

    settled = settled.assign(
        tipped_team=settled["team_home"].where(tipped_home, settled["team_away"]),
        winner=settled["team_home"].where(home_won, settled["team_away"]).where(~draw, "Draw"),
        tip_correct=(tipped_home == home_won) & ~draw,
        is_draw=draw,
        # Two-way both ways round. Using `1 - home_team_win_prob` for away tips
        # would fold the draw mass into the away side only, inflating away-tip
        # confidence relative to home-tip confidence on identical numbers.
        tip_prob=home_prob.where(tipped_home, 1.0 - home_prob),
    )
    return settled.sort_values(["round_id", "game_id"], ascending=[False, True]).reset_index(drop=True)


def scoreboard_summary_line(scoreboard):
    """One-line summary for logs, prompts, and plain-text email."""
    if not isinstance(scoreboard, dict):
        return None
    parts = [
        f"{scoreboard['last_round_name']}: {scoreboard['last_round_correct']}/{scoreboard['last_round_games']}",
        f"Season: {scoreboard['season_correct']}/{scoreboard['season_games']} ({scoreboard['season_accuracy']:.0%})",
    ]
    if scoreboard.get("market_accuracy") is not None:
        parts.append(f"Market favourite: {scoreboard['market_accuracy']:.0%}")
    return " | ".join(parts)
