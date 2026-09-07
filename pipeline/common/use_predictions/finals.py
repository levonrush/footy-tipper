"""Finals-only presentation logic: stakes, ledgers, history and theming.

The regular-season email answers a weekly question for a comp that is running.
In September the comp is over, so the finals edition drops the joker and the
competition-strategy note and spends those slots on what is actually at stake.

Everything here is presentation. Nothing in this module changes a tip, and every
lookup fails soft: a finals email missing its head-to-head line is a worse email,
not a failed send.
"""

import os
import sqlite3

import pandas as pd

from pipeline.common import rounds
from pipeline.common.use_predictions.premiership import premiership_race
from pipeline.common.use_predictions.scoreboard import settled_predictions
from pipeline.common.use_predictions.staking import get_market_picks

# `auto` classifies from the round name, `on` forces the finals treatment for
# out-of-season rehearsals, `off` restores the regular email unconditionally.
FINALS_MODE_ENV = "FOOTY_TIPPER_FINALS_MODE"
VALID_FINALS_MODES = ("auto", "on", "off")


def resolve_finals_mode():
    mode = str(os.getenv(FINALS_MODE_ENV, "auto") or "auto").strip().lower()
    return mode if mode in VALID_FINALS_MODES else "auto"


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

# Regular-round values are exactly the colours the template hardcoded before the
# finals work, so a regular email still renders byte for byte as it did.
_REGULAR_THEME = {
    "accent": "#0f766e",
    "value_accent": "#16a34a",
    "feature_accent": "#f59e0b",
    "header_gradient": "linear-gradient(135deg, #115e59 0%, #0369a1 100%)",
    "ribbon_background": None,
    "ribbon_text": None,
    "ribbon_label": None,
}

_FINALS_THEMES = {
    rounds.FINALS_WEEK_1: {
        "accent": "#0369a1",
        "header_gradient": "linear-gradient(135deg, #0f172a 0%, #0369a1 100%)",
        "ribbon_label": "FINALS WEEK 1 · SPECIAL EDITION",
    },
    rounds.FINALS_WEEK_2: {
        "accent": "#4338ca",
        "header_gradient": "linear-gradient(135deg, #0f172a 0%, #4338ca 100%)",
        "ribbon_label": "SEMI FINALS · SPECIAL EDITION",
    },
    rounds.PRELIMINARY: {
        "accent": "#7e22ce",
        "header_gradient": "linear-gradient(135deg, #0f172a 0%, #7e22ce 100%)",
        "ribbon_label": "PRELIMINARY FINALS · SPECIAL EDITION",
    },
    rounds.GRAND_FINAL: {
        "accent": "#b45309",
        "header_gradient": "linear-gradient(135deg, #78350f 0%, #b45309 100%)",
        "ribbon_label": "GRAND FINAL · SPECIAL EDITION",
    },
}


def theme(stage):
    """Colour tokens for a round stage."""
    if stage not in _FINALS_THEMES:
        return dict(_REGULAR_THEME)
    finals = dict(_REGULAR_THEME)
    finals.update(_FINALS_THEMES[stage])
    finals.update(
        {
            "value_accent": "#16a34a",
            "feature_accent": finals["accent"],
            "ribbon_background": "#0f172a",
            "ribbon_text": "#fbbf24",
        }
    )
    return finals


# ---------------------------------------------------------------------------
# Stakes
# ---------------------------------------------------------------------------

# Week one runs two qualifying finals (a double chance) and two elimination
# finals (sudden death) on the same weekend, so the seeds decide the stakes.
_QUALIFYING_PAIRS = {frozenset((1, 4)), frozenset((2, 3))}
_ELIMINATION_PAIRS = {frozenset((5, 8)), frozenset((6, 7))}

_SUDDEN_DEATH = "Sudden death. The loser's season ends here."


def knockout_stakes(stage, position_home=None, position_away=None):
    """One sentence on what the fixture is worth, or None outside finals."""
    if stage == rounds.GRAND_FINAL:
        return "The premiership. Eighty minutes, one trophy, no next week."
    if stage == rounds.PRELIMINARY:
        return "Winner plays in the Grand Final. Loser goes home one week short."
    if stage == rounds.FINALS_WEEK_2:
        return _SUDDEN_DEATH
    if stage != rounds.FINALS_WEEK_1:
        return None

    seeds = _seed_pair(position_home, position_away)
    if seeds in _QUALIFYING_PAIRS:
        return (
            "Qualifying final. Winner gets a week off and a home preliminary final. "
            "Loser drops into sudden death."
        )
    if seeds in _ELIMINATION_PAIRS:
        return "Elimination final. " + _SUDDEN_DEATH
    return "Finals football. There is no next round to fix it in."


def _seed_pair(position_home, position_away):
    try:
        return frozenset((int(position_home), int(position_away)))
    except (TypeError, ValueError):
        return frozenset()


# ---------------------------------------------------------------------------
# Ledgers
# ---------------------------------------------------------------------------


def _summarise(frame):
    if frame.empty:
        return None
    games = int(len(frame))
    correct = int(frame["model_correct"].sum())
    market_games = int(frame["has_odds"].sum())
    market_correct = int(frame["market_correct"].sum())
    return {
        "games": games,
        "correct": correct,
        "accuracy": correct / games,
        "market_games": market_games,
        "market_correct": market_correct,
        "market_accuracy": (market_correct / market_games) if market_games else None,
    }


def stage_ledger(db_path):
    """Split the season's settled tips into regular season and finals.

    Finals week one wants "here is how the comp finished"; later weeks want a
    running finals record next to it.
    """
    settled = settled_predictions(db_path)
    if settled.empty:
        return None

    stages = [
        rounds.round_stage(name, round_id)
        for name, round_id in zip(settled["round_name"], settled["round_id"])
    ]
    settled = settled.assign(stage=stages)
    regular = settled[settled["stage"] == rounds.REGULAR]
    finals = settled[settled["stage"] != rounds.REGULAR]

    return {
        "competition_year": int(settled.iloc[0]["competition_year"]),
        "regular_season": _summarise(regular),
        "finals": _summarise(finals),
        "best_call": _best_call(regular),
        "worst_call": _worst_call(regular),
    }


def _best_call(frame):
    """The biggest upset the model called and got right."""
    priced = frame[frame["model_correct"] & frame["market_probability_for_tip"].notna()]
    if priced.empty:
        return None
    row = priced.loc[priced["market_probability_for_tip"].idxmin()]
    return _call_summary(row)


def _worst_call(frame):
    """The most confident tip that missed."""
    missed = frame[~frame["model_correct"] & frame["tip_probability"].notna()]
    if missed.empty:
        return None
    row = missed.loc[missed["tip_probability"].idxmax()]
    return _call_summary(row)


def _call_summary(row):
    return {
        "round_name": str(row.get("round_name") or f"Round {int(row['round_id'])}"),
        "tipped_team": str(row["tipped_team"]),
        "opponent": str(
            row["team_away"] if row["tipped_team"] == row["team_home"] else row["team_home"]
        ),
        "tip_probability": _optional_float(row.get("tip_probability")),
        "market_probability": _optional_float(row.get("market_probability_for_tip")),
        "correct": bool(row["model_correct"]),
        "score": f"{int(row['team_final_score_home'])}-{int(row['team_final_score_away'])}",
    }


def _optional_float(value):
    return None if value is None or pd.isna(value) else float(value)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

_HISTORY_QUERY = """
SELECT CAST(competition_year AS INTEGER) AS competition_year,
       round_name,
       CAST(round_id AS INTEGER) AS round_id,
       team_home,
       team_away,
       CAST(team_final_score_home AS REAL) AS score_home,
       CAST(team_final_score_away AS REAL) AS score_away
FROM footy_tipping_data
WHERE game_state_name = 'Final'
  AND team_final_score_home IS NOT NULL
  AND ((team_home = ? AND team_away = ?) OR (team_home = ? AND team_away = ?))
ORDER BY CAST(competition_year AS REAL), CAST(round_id AS REAL)
"""


def head_to_head(db_path, home_team, away_team, competition_year):
    """This season's series between the two clubs, plus their finals history.

    Returns None when they have never met in the available record.
    """
    try:
        con = sqlite3.connect(str(db_path))
        try:
            meetings = pd.read_sql_query(
                _HISTORY_QUERY,
                con,
                params=(home_team, away_team, away_team, home_team),
            )
        finally:
            con.close()
    except Exception as exc:
        print(f"Head-to-head lookup failed ({exc}).")
        return None

    if meetings.empty:
        return None

    decided = meetings[meetings["score_home"] != meetings["score_away"]]
    winners = decided["team_home"].where(
        decided["score_home"] > decided["score_away"], decided["team_away"]
    )
    stages = [
        rounds.round_stage(name, round_id)
        for name, round_id in zip(decided["round_name"], decided["round_id"])
    ]
    decided = decided.assign(winner=winners, stage=stages)

    # Everything is read as at the requested season so an out-of-season rehearsal
    # cannot quote a result from the future.
    decided = decided[decided["competition_year"] <= int(competition_year)]
    if decided.empty:
        return None
    this_season = decided[decided["competition_year"] == int(competition_year)]
    finals_meetings = decided[decided["stage"] != rounds.REGULAR]
    last = decided.iloc[-1]

    return {
        "season_meetings": int(len(this_season)),
        "season_home_wins": int((this_season["winner"] == home_team).sum()),
        "season_away_wins": int((this_season["winner"] == away_team).sum()),
        "finals_meetings": int(len(finals_meetings)),
        "finals_home_wins": int((finals_meetings["winner"] == home_team).sum()),
        "finals_away_wins": int((finals_meetings["winner"] == away_team).sum()),
        "last_meeting": None if last is None else _meeting_summary(last),
    }


def _meeting_summary(row):
    """One past meeting, with the score read from the winner's side.

    "Penrith won 18-28" is a home-away scoreline attached to an away winner, and
    it reads as a typo. Winner first is the only orientation that cannot mislead.
    """
    home_score, away_score = int(row["score_home"]), int(row["score_away"])
    winning, losing = (
        (home_score, away_score)
        if home_score > away_score
        else (away_score, home_score)
    )
    return {
        "competition_year": int(row["competition_year"]),
        "round_name": str(row["round_name"]),
        "winner": str(row["winner"]),
        "score": f"{winning}-{losing}",
        "home_team": str(row["team_home"]),
        "away_team": str(row["team_away"]),
    }


def head_to_head_line(history, home_team, away_team):
    """A one-line reader summary of `head_to_head`, or None."""
    if not history:
        return None
    parts = []
    if history["season_meetings"]:
        home_wins = history["season_home_wins"]
        away_wins = history["season_away_wins"]
        if home_wins == away_wins:
            parts.append(
                f"They split {history['season_meetings']} meeting(s) this season."
            )
        else:
            leader, count = (
                (home_team, home_wins) if home_wins > away_wins else (away_team, away_wins)
            )
            parts.append(
                f"{leader} won {count} of {history['season_meetings']} this season."
            )
    else:
        parts.append("They have not met this season.")

    if history["finals_meetings"]:
        parts.append(
            f"In finals: {home_team} {history['finals_home_wins']}, "
            f"{away_team} {history['finals_away_wins']} "
            f"from {history['finals_meetings']} meeting(s)."
        )
    last = history.get("last_meeting")
    if last:
        parts.append(
            f"Last time: {last['winner']} won {last['score']} in "
            f"{last['round_name']} {last['competition_year']}."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def finals_context(predictions, db_path=None):
    """Everything the email and site need to decide how finals this round is.

    `is_finals` respects `FOOTY_TIPPER_FINALS_MODE`; `stage` always reports what
    the round actually is, so a forced-off finals round still logs honestly.
    """
    mode = resolve_finals_mode()
    stage = rounds.REGULAR
    round_name = None
    competition_year = None
    if predictions is not None and not getattr(predictions, "empty", True):
        round_name = predictions.iloc[0].get("round_name")
        round_id = predictions.iloc[0].get("round_id")
        competition_year = predictions.iloc[0].get("competition_year")
        stage = rounds.round_stage(round_name, round_id)

    if mode == "on" and stage == rounds.REGULAR:
        # Rehearsal: treat an ordinary round as week one so the whole finals path
        # can be exercised out of season.
        stage = rounds.FINALS_WEEK_1
    is_finals = stage != rounds.REGULAR and mode != "off"

    return {
        "mode": mode,
        "stage": stage,
        "is_finals": is_finals,
        "week": rounds.stage_week(stage),
        "display_name": rounds.stage_display_name(stage) or (
            str(round_name) if round_name else None
        ),
        "round_name": None if round_name is None else str(round_name),
        "competition_year": None if competition_year is None else int(competition_year),
        "theme": theme(stage if is_finals else rounds.REGULAR),
        "ledger": stage_ledger(db_path) if (is_finals and db_path) else None,
    }


def finals_payload(db_path, predictions):
    """Assemble everything the finals email and site need, or the plain context.

    A regular round returns early with `is_finals` false and nothing else built,
    so no finals work runs for the other twenty-seven weeks of the year. Each
    finals extra is gathered independently: one failing does not cost the others.
    """
    context = finals_context(predictions, db_path)
    if not context["is_finals"]:
        return context

    context["distributions"] = {}
    context["stakes"] = {}
    context["head_to_head"] = {}
    context["premiership"] = None
    context["market_picks"] = None
    if predictions is None or getattr(predictions, "empty", True):
        return context

    stage = context["stage"]
    for _, row in predictions.iterrows():
        game_id = _game_id(row)
        if game_id is None:
            continue
        stakes = knockout_stakes(
            stage, row.get("position_home"), row.get("position_away")
        )
        if stakes:
            context["stakes"][game_id] = stakes

    try:
        from pipeline.common.model_prediciton import distributions as pdist

        stored = pdist.load_distributions(db_path, predictions["game_id"].tolist())
        context["distributions"] = {
            int(record["game_id"]): dict(record)
            for _, record in stored.iterrows()
            if pd.notna(record.get("game_id"))
        }
        context["market_picks"] = get_market_picks(predictions, stored)
    except Exception as exc:
        print(f"Finals distribution extras skipped ({exc}).")

    try:
        year = context["competition_year"]
        for _, row in predictions.iterrows():
            game_id = _game_id(row)
            if game_id is None:
                continue
            home, away = str(row.get("team_home")), str(row.get("team_away"))
            line = head_to_head_line(
                head_to_head(db_path, home, away, year), home, away
            )
            if line:
                context["head_to_head"][game_id] = line
    except Exception as exc:
        print(f"Finals head-to-head skipped ({exc}).")

    context["premiership"] = premiership_race(db_path, predictions)
    return context


def _game_id(row):
    try:
        return int(row.get("game_id"))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Copy helpers
# ---------------------------------------------------------------------------

_SUBJECTS = {
    rounds.FINALS_WEEK_1: "FINALS WEEK 1: sudden death starts now",
    rounds.FINALS_WEEK_2: "SEMI FINALS: win or the season's over",
    rounds.PRELIMINARY: "PRELIMINARY FINALS: one game from the big dance",
    rounds.GRAND_FINAL: "GRAND FINAL: the big dance",
}


def finals_subject(stage, competition_year):
    """Stage-branded subject line, or None for a regular round."""
    headline = _SUBJECTS.get(stage)
    if not headline:
        return None
    return f"{headline} · Footy Tipper {competition_year}"
