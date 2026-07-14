"""Derived per-round team performance rows in the feed_cache_performance schema.

Semantics verified against the cached feed: each (competition_year, round_id,
team) row holds that round's SINGLE-GAME stats (not season-to-date); bye
rounds are all-zero rows. R lags the table by one round at read time, so the
`_performance` features at round r are the team's round r-1 game stats.

Feed naming quirk: `<stat>` columns hold metres/values and `<stat>_occur`
columns hold counts (e.g. `kicks` = kicking metres, `kicks_occur` = number of
kicks).

Sources per column:
- ("team", name): match_team_stats long row (modern match centre groups)
- ("players", column): sum of match_player_stats column over the team's players
- ("fixture", kind): derived from the fixture result
- ("ratio", num_spec, den_spec, scale): ratio of two other specs

Columns whose source no longer exists (era-specific leaderboard extras like
`shifts`, `receipts`, `x20_att_gl_possession_seconds`) are intentionally
absent: the cache writer NULL-fills them and the parity validator reports the
definitive non-derivable list, which gets pruned from predictors at the
cutover retrain.
"""

from __future__ import annotations

from collections import defaultdict

PERF_STAT_MAP: dict[str, tuple] = {
    # defence
    "tackle_made": ("players", "tackles_made"),
    "tackle_missed": ("players", "missed_tackles"),
    "ineffective_tackle": ("players", "ineffective_tackles"),
    "intercept": ("players", "intercepts"),
    "one_on_one_steal": ("players", "one_on_one_steal"),
    "one_on_one_lost": ("players", "one_on_one_lost"),
    "effective_tackle_percentage": ("team", "effective_tackle_pct"),
    # attack (team stats reproduce the feed's decimal values exactly;
    # player sums are the fallback for eras without the stat group)
    "all_runs": ("first", ("team", "all_runs"), ("players", "all_runs")),
    "all_run_metres": ("first", ("team", "all_run_metres"), ("players", "all_run_metres")),
    "post_contact_metres": ("first", ("team", "post_contact_metres"), ("players", "post_contact_metres")),
    "linebreak": ("first", ("team", "line_breaks"), ("players", "line_breaks")),
    "lb_assist": ("players", "line_break_assists"),
    "tackle_break": ("first", ("team", "tackle_breaks"), ("players", "tackle_breaks")),
    "offloads": ("first", ("team", "offloads"), ("players", "offloads")),
    "dummy_pass": ("players", "dummy_passes"),
    "dh_run": ("players", "dummy_half_run_metres"),
    "dh_run_occur": ("players", "dummy_half_runs"),
    "hit_up": ("players", "hit_up_run_metres"),
    "hit_up_occur": ("players", "hit_ups"),
    "decoys": ("players", "decoys"),
    "supports": ("players", "supports"),
    "half_break": ("players", "half_breaks"),
    # kicking: bare column = metres, _occur = count
    "kicks": ("first", ("team", "kicking_metres"), ("players", "kick_metres")),
    "kicks_occur": ("players", "kicks"),
    "kick_return": ("players", "kick_return_metres"),
    "kick_bomb": ("players", "bomb_kicks"),
    "kick_chip": ("players", "chip_kicks"),
    "kick_crossfield": ("players", "cross_field_kicks"),
    "kick_grubber": ("players", "grubber_kicks"),
    "kick_forces_dropout": ("players", "forced_drop_out_kicks"),
    "x40_20_kick": ("players", "forty_twenty_kicks"),
    "kick_defused": ("players", "kicks_defused"),
    "kick_dead": ("players", "kicks_dead"),
    "charge_downs": ("players", "charge_downs"),
    # scoring
    "try": ("players", "tries"),
    "try_assist": ("players", "try_assists"),
    "points": ("fixture", "points_for"),
    # feed convention: goals = conversions + penalty goals
    "all_goals_made": ("sum", ("players", "conversions"), ("players", "penalty_goals")),
    "all_goals_attempted": ("players", "goal_attempts"),
    "conversion_made": ("players", "conversions"),
    "conversion_attempted": ("players", "conversion_attempts"),
    "penalty_shot_made": ("players", "penalty_goals"),
    "field_goal_made": ("players", "field_goals"),
    "one_point_field_goal": ("players", "one_point_field_goals"),
    "two_point_field_goal": ("players", "two_point_field_goals"),
    "goal_conversion_rate": (
        "ratio",
        ("players", "goals"),
        ("players", "goal_attempts"),
        100.0,
    ),
    # discipline / errors
    "error": ("players", "errors"),
    "handling_errors": ("players", "handling_errors"),
    "penalties": ("players", "penalties"),
    "ruck_infringement": ("players", "ruck_infringements"),
    "offside_penalties": ("players", "offside_within_ten_metres"),
    "sin_bin": ("players", "sin_bins"),
    "sent_off": ("players", "send_offs"),
    "on_report": ("players", "on_report"),
    "pass_intercepted": ("players", "pass_intercepted"),
    # game control
    "possession": ("team", "possession_pct"),
    "set_completion_rate": ("team", "completion_rate"),
    "minutes_played": ("players", "minutes_played"),
    "play_the_ball": ("players", "play_the_ball_total"),
    "ptbaverage": ("team", "average_play_the_ball_speed"),
    "line_dropout": ("team", "line_dropouts"),
    "line_engaged": ("players", "line_engaged_runs"),
    "receipts": ("players", "receipts"),
    # result flags
    "wins": ("fixture", "win"),
    "draws": ("fixture", "draw"),
    "losses": ("fixture", "loss"),
}

# player columns referenced above that may be missing in some eras; sums treat
# missing columns as absent rather than erroring
_OPTIONAL_PLAYER_COLUMNS = {
    "decoys",
    "supports",
    "half_breaks",
    "chip_kicks",
    "kicks_dead",
    "charge_downs",
    "goal_attempts",
    "pass_intercepted",
}


def _resolve(
    spec: tuple,
    team_stats: dict[str, float],
    player_sums: dict[str, float],
    fixture_values: dict[str, float],
) -> float | None:
    kind = spec[0]
    if kind == "team":
        return team_stats.get(spec[1])
    if kind == "players":
        return player_sums.get(spec[1])
    if kind == "fixture":
        return fixture_values.get(spec[1])
    if kind == "first":
        for sub_spec in spec[1:]:
            value = _resolve(sub_spec, team_stats, player_sums, fixture_values)
            if value is not None:
                return value
        return None
    if kind == "sum":
        parts = [
            _resolve(sub_spec, team_stats, player_sums, fixture_values)
            for sub_spec in spec[1:]
        ]
        known = [part for part in parts if part is not None]
        return sum(known) if known else None
    if kind == "ratio":
        numerator = _resolve(spec[1], team_stats, player_sums, fixture_values)
        denominator = _resolve(spec[2], team_stats, player_sums, fixture_values)
        if numerator is None or not denominator:
            return None
        return round(numerator / denominator * spec[3], 2)
    raise ValueError(f"Unknown performance stat spec: {spec}")


def build_game_performance(
    team_stats: dict[str, float],
    player_sums: dict[str, float],
    fixture_values: dict[str, float],
) -> dict[str, float]:
    """One team-game's leaderboard-schema stats."""
    row: dict[str, float] = {}
    for column, spec in PERF_STAT_MAP.items():
        value = _resolve(spec, team_stats, player_sums, fixture_values)
        if value is not None:
            row[column] = float(value)
    return row


def build_season_performance(
    fixtures: list[dict],
    byes: list[dict],
    season: int,
    team_stats_by_game: dict[int, dict[str, dict[str, float]]],
    player_sums_by_game: dict[int, dict[str, dict[str, float]]],
) -> list[dict]:
    """Raw feed_cache_performance rows for a season.

    `team_stats_by_game` / `player_sums_by_game`: game_id -> side -> stats.
    Only regular-season rounds are emitted (the feed leaderboard covers the
    premiership rounds; finals rounds have no leaderboard rows).
    """
    from .ladder import is_regular_round

    rows: list[dict] = []
    played_rounds: dict[int, set[str]] = defaultdict(set)

    for fixture in fixtures:
        if fixture.get("game_state_name") != "Final":
            continue
        round_id = int(float(fixture["round_id"]))
        if not is_regular_round(fixture.get("round_name"), round_id):
            continue
        game_id = int(float(fixture["game_id"]))
        home_score = float(fixture.get("team_final_score_home") or 0)
        away_score = float(fixture.get("team_final_score_away") or 0)

        for side, team, points_for, points_against in (
            ("home", fixture["team_home"], home_score, away_score),
            ("away", fixture["team_away"], away_score, home_score),
        ):
            fixture_values = {
                "points_for": points_for,
                "win": float(points_for > points_against),
                "draw": float(points_for == points_against),
                "loss": float(points_for < points_against),
            }
            row = build_game_performance(
                (team_stats_by_game.get(game_id) or {}).get(side, {}),
                (player_sums_by_game.get(game_id) or {}).get(side, {}),
                fixture_values,
            )
            row.update(
                {
                    "team": team,
                    "round_id": round_id,
                    "competition_year": int(season),
                }
            )
            rows.append(row)
            played_rounds[round_id].add(team)

    # bye rounds: the feed emits all-zero rows for teams without a game
    zero_template = {column: 0.0 for column in PERF_STAT_MAP}
    for bye in byes:
        round_id = int(float(bye["round_id"]))
        team = bye["team"]
        if team in played_rounds.get(round_id, set()):
            continue
        row = dict(zero_template)
        row.update(
            {
                "team": team,
                "round_id": round_id,
                "competition_year": int(season),
            }
        )
        rows.append(row)

    rows.sort(key=lambda r: (r["round_id"], r["team"]))
    return rows


def load_team_stats_by_game(con, season: int) -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for game_id, side, stat_name, value in con.execute(
        "SELECT game_id, side, stat_name, value FROM match_team_stats "
        "WHERE competition_year = ?",
        (int(season),),
    ):
        result[int(game_id)][side][stat_name] = value
    return result


def load_player_sums_by_game(
    con, season: int
) -> dict[int, dict[str, dict[str, float]]]:
    columns = {
        row[1]
        for row in con.execute("PRAGMA table_info(match_player_stats)")
    }

    def _player_columns(spec) -> set[str]:
        kind = spec[0]
        if kind == "players":
            return {spec[1]}
        if kind in ("first", "sum"):
            found: set[str] = set()
            for sub_spec in spec[1:]:
                found |= _player_columns(sub_spec)
            return found
        if kind == "ratio":
            return _player_columns(spec[1]) | _player_columns(spec[2])
        return set()

    wanted: set[str] = set()
    for spec in PERF_STAT_MAP.values():
        wanted |= _player_columns(spec)
    available = sorted(wanted & columns)
    if not available:
        return {}
    select_cols = ", ".join(f"SUM({col}) AS {col}" for col in available)
    result: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in con.execute(
        f"SELECT game_id, side, {select_cols} FROM match_player_stats "
        "WHERE competition_year = ? GROUP BY game_id, side",
        (int(season),),
    ):
        game_id, side = int(row[0]), row[1]
        result[game_id][side] = {
            col: row[i + 2] for i, col in enumerate(available) if row[i + 2] is not None
        }
    return result


def load_game_scoring(con, season: int) -> dict[int, dict[str, dict]]:
    """Scoring + squad aggregates for the ladder builder.

    Feed convention (validated): goals = conversions + penalty goals; the
    match centre 'goals' player column is unpopulated in the modern era.
    """
    result: dict[int, dict[str, dict]] = defaultdict(dict)
    columns = {
        row[1] for row in con.execute("PRAGMA table_info(match_player_stats)")
    }
    fg_expr = "COALESCE(SUM(field_goals), 0)"
    if {"one_point_field_goals", "two_point_field_goals"} <= columns:
        fg_expr = (
            "MAX(COALESCE(SUM(field_goals), 0), "
            "COALESCE(SUM(one_point_field_goals), 0) + "
            "COALESCE(SUM(two_point_field_goals), 0))"
        )
    for game_id, side, tries, goals, field_goals in con.execute(
        f"""
        SELECT game_id, side, COALESCE(SUM(tries), 0),
               COALESCE(SUM(conversions), 0) + COALESCE(SUM(penalty_goals), 0),
               {fg_expr}
        FROM match_player_stats
        WHERE competition_year = ?
        GROUP BY game_id, side
        """,
        (int(season),),
    ):
        players = {
            int(row[0])
            for row in con.execute(
                "SELECT player_id FROM match_player_stats "
                "WHERE game_id = ? AND side = ? AND "
                "COALESCE(minutes_played, 0) > 0",
                (game_id, side),
            )
        }
        result[int(game_id)][side] = {
            "tries": tries,
            "goals": goals,
            "field_goals": field_goals,
            "players": players,
        }
    return result
