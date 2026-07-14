"""Derived round-by-round ladder in the feed_cache_ladders schema.

The feed's raw ladder row for round r is the season-to-date table INCLUDING
round r (R lags it by one round at read time). Reproduced conventions,
verified against cached feed seasons:

- one row per team per round 1..max_round, finals rounds carry the frozen
  regular-season ladder forward (17 identical rows per finals round);
- form strings are most-recent-first and encode byes as "B"
  (recent_form = first 5 chars of season_form);
- day/night records are "W-D-L" strings split on venue-local kickoff hour;
- current_streak is like "3W"/"2L" over played games (byes skipped);
- competition points: win 2, draw 1, bye 2.

Tries/goals/field-goals for+against and players_used need match centre player
stats (`game_scoring`); when absent they accumulate as zero, which the parity
validator reports per column.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

# Validated against the feed: games before 17:00 venue-local count as day
DAY_NIGHT_SPLIT_HOUR = 17
CLOSE_GAME_MARGIN = 6

LADDER_COLUMNS = [
    "position",
    "team",
    "wins",
    "draws",
    "losses",
    "byes",
    "competition_points",
    "points_for",
    "points_against",
    "points_difference",
    "home_wins",
    "home_draws",
    "home_losses",
    "away_wins",
    "away_draws",
    "away_losses",
    "recent_form",
    "season_form",
    "tries_for",
    "tries_conceded",
    "goals_for",
    "goals_conceded",
    "field_goals_for",
    "field_goals_conceded",
    "players_used",
    "average_winning_margin",
    "average_losing_margin",
    "close_games",
    "day_record",
    "night_record",
    "current_streak",
    "round_id",
    "competition_year",
]


def is_regular_round(round_name: str | None, round_id: int) -> bool:
    if round_name:
        return str(round_name).strip().lower().startswith("round")
    return round_id <= 27


def _local_hour(start_time_epoch: float | None) -> int | None:
    if start_time_epoch is None:
        return None
    # start_time is venue-local wall clock serialised as-if-UTC
    return dt.datetime.fromtimestamp(
        float(start_time_epoch), tz=dt.timezone.utc
    ).hour


class _TeamTally:
    def __init__(self, team: str) -> None:
        self.team = team
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.byes = 0
        self.points_for = 0.0
        self.points_against = 0.0
        self.home_wins = 0
        self.home_draws = 0
        self.home_losses = 0
        self.away_wins = 0
        self.away_draws = 0
        self.away_losses = 0
        self.results_recent_first: list[str] = []
        self.tries_for = 0.0
        self.tries_conceded = 0.0
        self.goals_for = 0.0
        self.goals_conceded = 0.0
        self.field_goals_for = 0.0
        self.field_goals_conceded = 0.0
        self.players: set[int] = set()
        self.winning_margins: list[float] = []
        self.losing_margins: list[float] = []
        self.close_games = 0
        self.day_w = 0
        self.day_d = 0
        self.day_l = 0
        self.night_w = 0
        self.night_d = 0
        self.night_l = 0

    @property
    def competition_points(self) -> int:
        return self.wins * 2 + self.draws + self.byes * 2

    def record_bye(self) -> None:
        self.byes += 1
        self.results_recent_first.insert(0, "B")

    def record_game(
        self,
        is_home: bool,
        points_for: float,
        points_against: float,
        local_hour: int | None,
        scoring_for: dict | None,
        scoring_against: dict | None,
    ) -> None:
        margin = points_for - points_against
        if margin > 0:
            result = "W"
            self.wins += 1
            self.winning_margins.append(margin)
        elif margin < 0:
            result = "L"
            self.losses += 1
            self.losing_margins.append(-margin)
        else:
            result = "D"
            self.draws += 1

        self.results_recent_first.insert(0, result)
        self.points_for += points_for
        self.points_against += points_against

        if is_home:
            self.home_wins += result == "W"
            self.home_draws += result == "D"
            self.home_losses += result == "L"
        else:
            self.away_wins += result == "W"
            self.away_draws += result == "D"
            self.away_losses += result == "L"

        if abs(margin) <= CLOSE_GAME_MARGIN:
            self.close_games += 1

        if local_hour is not None:
            is_day = local_hour < DAY_NIGHT_SPLIT_HOUR
            if is_day:
                self.day_w += result == "W"
                self.day_d += result == "D"
                self.day_l += result == "L"
            else:
                self.night_w += result == "W"
                self.night_d += result == "D"
                self.night_l += result == "L"

        if scoring_for:
            self.tries_for += scoring_for.get("tries", 0) or 0
            self.goals_for += scoring_for.get("goals", 0) or 0
            self.field_goals_for += scoring_for.get("field_goals", 0) or 0
            self.players.update(scoring_for.get("players") or ())
        if scoring_against:
            self.tries_conceded += scoring_against.get("tries", 0) or 0
            self.goals_conceded += scoring_against.get("goals", 0) or 0
            self.field_goals_conceded += scoring_against.get("field_goals", 0) or 0

    def current_streak(self) -> str | None:
        run_result = None
        run_length = 0
        for result in self.results_recent_first:
            if result == "B":
                continue
            if run_result is None:
                if result == "D":
                    return None
                run_result = result
            if result != run_result:
                break
            run_length += 1
        if run_result is None or run_length == 0:
            return None
        return f"{run_length}{run_result}"

    def snapshot(self, round_id: int, season: int) -> dict:
        season_form = "".join(self.results_recent_first)
        avg_win = (
            round(sum(self.winning_margins) / len(self.winning_margins), 1)
            if self.winning_margins
            else 0.0
        )
        avg_loss = (
            round(sum(self.losing_margins) / len(self.losing_margins), 1)
            if self.losing_margins
            else 0.0
        )
        return {
            "team": self.team,
            "wins": float(self.wins),
            "draws": float(self.draws),
            "losses": float(self.losses),
            "byes": float(self.byes),
            "competition_points": float(self.competition_points),
            "points_for": self.points_for,
            "points_against": self.points_against,
            "points_difference": self.points_for - self.points_against,
            "home_wins": float(self.home_wins),
            "home_draws": float(self.home_draws),
            "home_losses": float(self.home_losses),
            "away_wins": float(self.away_wins),
            "away_draws": float(self.away_draws),
            "away_losses": float(self.away_losses),
            "recent_form": season_form[:5],
            "season_form": season_form,
            "tries_for": self.tries_for,
            "tries_conceded": self.tries_conceded,
            "goals_for": self.goals_for,
            "goals_conceded": self.goals_conceded,
            "field_goals_for": self.field_goals_for,
            "field_goals_conceded": self.field_goals_conceded,
            "players_used": float(len(self.players)),
            "average_winning_margin": avg_win,
            "average_losing_margin": avg_loss,
            "close_games": float(self.close_games),
            "day_record": f"{self.day_w}-{self.day_d}-{self.day_l}",
            "night_record": f"{self.night_w}-{self.night_d}-{self.night_l}",
            "current_streak": self.current_streak(),
            "round_id": int(round_id),
            "competition_year": int(season),
        }


def _rank_rows(rows: list[dict]) -> None:
    rows.sort(
        key=lambda row: (
            -row["competition_points"],
            -row["points_difference"],
            -row["points_for"],
            row["team"],
        )
    )
    for position, row in enumerate(rows, start=1):
        row["position"] = float(position)


def build_season_ladder(
    fixtures: list[dict],
    byes: list[dict],
    season: int,
    game_scoring: dict | None = None,
) -> list[dict]:
    """Raw (unlagged) ladder rows for every team and round of a season.

    `game_scoring` maps game_id -> {"home": {...}, "away": {...}} with keys
    tries/goals/field_goals/players, sourced from match_player_stats.
    """
    game_scoring = game_scoring or {}

    fixtures_by_round: dict[int, list[dict]] = defaultdict(list)
    round_names: dict[int, str | None] = {}
    max_round = 0
    teams: set[str] = set()
    for fixture in fixtures:
        round_id = int(float(fixture["round_id"]))
        fixtures_by_round[round_id].append(fixture)
        round_names.setdefault(round_id, fixture.get("round_name"))
        max_round = max(max_round, round_id)
        teams.add(fixture["team_home"])
        teams.add(fixture["team_away"])

    byes_by_round: dict[int, list[str]] = defaultdict(list)
    for bye in byes:
        round_id = int(float(bye["round_id"]))
        byes_by_round[round_id].append(bye["team"])
        teams.add(bye["team"])
        max_round = max(max_round, round_id)

    tallies = {team: _TeamTally(team) for team in sorted(teams)}
    ladder_rows: list[dict] = []
    frozen_snapshot: list[dict] | None = None

    for round_id in range(1, max_round + 1):
        regular = is_regular_round(round_names.get(round_id), round_id)

        if regular:
            for fixture in fixtures_by_round.get(round_id, []):
                if fixture.get("game_state_name") != "Final":
                    continue
                game_id = fixture.get("game_id")
                scoring = game_scoring.get(
                    int(float(game_id)) if game_id is not None else None, {}
                )
                local_hour = _local_hour(fixture.get("start_time"))
                home_score = float(fixture.get("team_final_score_home") or 0)
                away_score = float(fixture.get("team_final_score_away") or 0)
                tallies[fixture["team_home"]].record_game(
                    True,
                    home_score,
                    away_score,
                    local_hour,
                    scoring.get("home"),
                    scoring.get("away"),
                )
                tallies[fixture["team_away"]].record_game(
                    False,
                    away_score,
                    home_score,
                    local_hour,
                    scoring.get("away"),
                    scoring.get("home"),
                )
            for team in byes_by_round.get(round_id, []):
                tallies[team].record_bye()

            round_rows = [
                tally.snapshot(round_id, season) for tally in tallies.values()
            ]
            _rank_rows(round_rows)
            frozen_snapshot = round_rows
        else:
            # Finals: ladder freezes; carry the last regular-season table.
            if frozen_snapshot is None:
                continue
            round_rows = [dict(row, round_id=int(round_id)) for row in frozen_snapshot]

        ladder_rows.extend(round_rows)

    return ladder_rows
