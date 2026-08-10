"""Feature-family taxonomy for explainability.

Groups the ~600 raw predictors into a handful of families with plain-English
labels. Reason codes and cohort analyses are only readable at this altitude:
"team-list strength" means something, "lineup_avg_spine_margin_rating_delta"
does not.

This deliberately lives here rather than in ``training_config``: that module is
imported by train/inference/evaluate and must stay a pure config surface, while
this one is presentation and must tolerate predictors it has never seen (hence
the "unclassified" fallback rather than an exception).

Rule order matters and is asserted by tests/test_explain_families.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bump when the rules change, so stored explanations can be spotted as stale.
FAMILY_TAXONOMY_VERSION = 1

UNCLASSIFIED = "unclassified"

# Kept short: these are column labels in an 80-column console table as well as
# clause subjects in the one-line email why.
FAMILY_LABELS = {
    "tier_a_baseline": "Team strength baseline",
    "elo": "Elo ratings",
    "ladder": "Ladder and season totals",
    "team_match_stats": "Historical match stats",
    "player_form": "Player recent form",
    "role_ratings": "Positional role strength",
    "lineup": "Team list and selection",
    "recent_form_stats": "Recent match-stat form",
    "referee": "Referee",
    "weather": "Weather",
    "travel_rest": "Travel and rest",
    "venue_crowd": "Venue and crowd",
    "season_state": "Season trajectory",
    "schedule_context": "Schedule context",
    "broadcast": "Broadcast slot",
    "team_identity": "Team identity",
    UNCLASSIFIED: "Unclassified",
}

# Season-trajectory features that do not carry the "season_" prefix.
_SEASON_STATE_EXTRAS = frozenset({
    "matchup_form",
    "form_delta",
    "points_for_form_delta",
    "points_against_form_delta",
    "diff_form_delta",
    "attack_delta",
    "defence_delta",
    "home_prev_result_diff",
    "away_prev_result_diff",
    "prev_result_diff",
    "position_diff",
})

# When/where the game is played, as opposed to who is playing.
_SCHEDULE_CONTEXT = frozenset({
    "round_id",
    "round_name",
    "game_number",
    "game_state_name",
    "start_time",
    "start_time_utc",
    "start_hour",
    "game_day",
    "competition_year",
    "corona_season",
    "state_of_origin",
    "post_origin",
})

_WEATHER_EXTRAS = frozenset({"ground_condition", "weather_missing"})
_VENUE_EXTRAS = frozenset({"city", "crowd", "crowd_features_missing"})
_TEAM_IDENTITY = frozenset({"team_home", "team_away"})


def _starts(*prefixes):
    return lambda name: name.startswith(prefixes)


# First match wins. The order encodes three real traps:
#   * "elo" before "ladder", so elo probabilities are not read as ladder state;
#   * "player_form"/"role_ratings" before the broad "lineup_" catch-all;
#   * "recent_form_stats" excludes form_delta exactly as training_config's
#     filter_predictors does, because that column is season state, not a
#     match-stat rolling mean.
RULES = (
    ("tier_a_baseline", _starts("baseline_")),
    ("elo", lambda name: "elo" in name),
    ("ladder", lambda name: name.endswith("_ladder") or name.startswith("ladder_")),
    (
        "team_match_stats",
        lambda name: "_performance" in name or name.startswith("performance_"),
    ),
    ("player_form", _starts("lineup_form_", "lineup_spine_form_")),
    ("role_ratings", _starts("lineup_rating_")),
    ("lineup", _starts("lineup_")),
    (
        "recent_form_stats",
        lambda name: name.startswith("form_") and name != "form_delta",
    ),
    ("referee", lambda name: name.startswith("ref_") or name == "referee_name"),
    (
        "weather",
        lambda name: name.startswith("wx_") or name in _WEATHER_EXTRAS,
    ),
    (
        "travel_rest",
        lambda name: name.startswith(("travel_", "tz_", "turn_around"))
        or name == "rest_delta",
    ),
    (
        "venue_crowd",
        lambda name: name.startswith("venue_") or name in _VENUE_EXTRAS,
    ),
    (
        "season_state",
        lambda name: name.startswith("season_") or name in _SEASON_STATE_EXTRAS,
    ),
    ("schedule_context", lambda name: name in _SCHEDULE_CONTEXT),
    ("broadcast", _starts("broadcast_channel")),
    ("team_identity", lambda name: name in _TEAM_IDENTITY),
)

FAMILIES = tuple(family for family, _ in RULES)

# Suffixes are checked before prefixes: "home_win_rate_away_ladder" is the away
# team's home-win rate, so the side that owns the number is away.
_SIDE_SUFFIXES = (
    ("_home_ladder", "home"),
    ("_away_ladder", "away"),
    ("_home_performance", "home"),
    ("_away_performance", "away"),
    ("_delta", "delta"),
    ("_diff", "delta"),
    ("_home", "home"),
    ("_away", "away"),
)


def family_for(name: str) -> str:
    """Family of a raw predictor. Never raises; unknown names are unclassified."""
    name = str(name)
    for family, matches in RULES:
        if matches(name):
            return family
    return UNCLASSIFIED


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", " ").capitalize())


def side_for(name: str) -> str:
    """Which side of the fixture a predictor describes: home/away/delta/neutral."""
    name = str(name)
    for suffix, side in _SIDE_SUFFIXES:
        if name.endswith(suffix):
            return side
    if name.startswith("home_"):
        return "home"
    if name.startswith("away_"):
        return "away"
    return "neutral"


def family_map(predictors) -> dict:
    return {str(name): family_for(name) for name in predictors}


def family_counts(predictors) -> dict:
    counts = {}
    for family in family_map(predictors).values():
        counts[family] = counts.get(family, 0) + 1
    return counts


def group_by_family(values, feature_names, *, families=None) -> pd.DataFrame:
    """Sum a (n_rows, n_features) contribution matrix into per-family columns.

    Summing is exact: SHAP contributions are additive on the link scale, so a
    family's contribution is the sum of its members' contributions.
    """
    values = np.asarray(values, dtype=float)
    feature_names = [str(name) for name in feature_names]
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            f"contribution matrix {values.shape} does not match "
            f"{len(feature_names)} feature names"
        )
    mapping = family_map(feature_names)
    order = [f for f in (families or FAMILIES + (UNCLASSIFIED,))]
    present = [f for f in order if f in set(mapping.values())]

    columns = {}
    for family in present:
        idx = [i for i, name in enumerate(feature_names) if mapping[name] == family]
        columns[family] = values[:, idx].sum(axis=1)
    return pd.DataFrame(columns, index=pd.RangeIndex(values.shape[0]))
