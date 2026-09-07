import os
import sqlite3
import tempfile
import unittest

import pandas as pd

from pipeline.common import rounds
from pipeline.common.use_predictions import premiership as pr


# Real NRL finals series, encoded as ladder seeds so the bracket rules can be
# checked without the runtime database. Each entry is
# `stage -> {frozenset(seed pair): winning seed}`.
HISTORIC_SERIES = {
    2023: {
        "week1": {(2, 3): 2, (1, 4): 1, (6, 7): 7, (5, 8): 5},
        "week2": {(3, 7): 3, (4, 5): 4},
        "week3": {(1, 3): 1, (2, 4): 2},
    },
    2024: {
        "week1": {(2, 3): 2, (1, 4): 1, (5, 8): 5, (6, 7): 7},
        "week2": {(4, 5): 4, (3, 7): 3},
        "week3": {(1, 3): 1, (2, 4): 2},
    },
    2025: {
        "week1": {(2, 3): 2, (6, 7): 7, (5, 8): 5, (1, 4): 4},
        "week2": {(1, 5): 5, (3, 7): 7},
        "week3": {(2, 5): 2, (4, 7): 4},
    },
}

TEAMS = {seed: f"Team {seed}" for seed in range(1, 9)}


class BracketRuleTests(unittest.TestCase):
    """The pairing rules are pinned to what the NRL actually played."""

    def _week_one_outcomes(self, series):
        results = {frozenset(pair): winner for pair, winner in series["week1"].items()}

        def resolve(pair):
            winner = results[frozenset(pair)]
            loser = pair[0] if winner == pair[1] else pair[1]
            return winner, loser

        return resolve(pr.QUALIFYING_1), resolve(pr.QUALIFYING_2), resolve(
            pr.ELIMINATION_1
        ), resolve(pr.ELIMINATION_2)

    def test_semi_finals_reproduce_2023_to_2025(self):
        for year, series in HISTORIC_SERIES.items():
            with self.subTest(year=year):
                (_, qf1_l), (_, qf2_l), (ef1_w, _), (ef2_w, _) = self._week_one_outcomes(
                    series
                )
                predicted = {
                    frozenset(pair)
                    for pair in pr.semi_final_pairs((qf1_l, qf2_l), (ef1_w, ef2_w))
                }
                actual = {frozenset(pair) for pair in series["week2"]}
                self.assertEqual(predicted, actual)

    def test_preliminary_finals_reproduce_2023_to_2025(self):
        for year, series in HISTORIC_SERIES.items():
            with self.subTest(year=year):
                (qf1_w, qf1_l), (qf2_w, qf2_l), (ef1_w, _), (ef2_w, _) = (
                    self._week_one_outcomes(series)
                )
                semis = pr.semi_final_pairs((qf1_l, qf2_l), (ef1_w, ef2_w))
                week2 = {frozenset(pair): winner for pair, winner in series["week2"].items()}
                semi_winners = (week2[frozenset(semis[0])], week2[frozenset(semis[1])])
                predicted = {
                    frozenset(pair)
                    for pair in pr.preliminary_final_pairs((qf1_w, qf2_w), semi_winners)
                }
                actual = {frozenset(pair) for pair in series["week3"]}
                self.assertEqual(predicted, actual)


def _build_db(db_path, played_through=None):
    """A 2025-shaped season whose finals follow the 2025 results.

    `played_through` is the last finals round left as completed; everything after
    it is pre-game.
    """
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE footy_tipping_data (
            game_id INTEGER PRIMARY KEY, competition_year INTEGER, round_id REAL,
            round_name TEXT, game_number REAL, start_time REAL, game_state_name TEXT,
            team_home TEXT, team_away TEXT,
            position_home_ladder REAL, position_away_ladder REAL,
            team_final_score_home REAL, team_final_score_away REAL
        )
        """
    )
    rows = []
    game_id = 0
    # A regular season long enough for the ratings to separate the teams, and to
    # establish that round 27 is the last regular round.
    for round_id in range(1, 28):
        for offset in range(0, 8, 2):
            home, away = TEAMS[offset + 1], TEAMS[offset + 2]
            game_id += 1
            # Better seeds win more, so the ratings order matches the ladder.
            rows.append(
                (game_id, 2025, round_id, f"Round {round_id}", offset / 2 + 1, game_id,
                 "Final", home, away, offset + 1, offset + 2, 26, 18)
            )

    series = HISTORIC_SERIES[2025]
    finals = [
        (28, "Finals Week 1", series["week1"]),
        (29, "Finals Week 2", series["week2"]),
        (30, "Finals Week 3", series["week3"]),
        (31, "Grand Final", {(2, 4): 4}),
    ]
    for round_id, round_name, games in finals:
        for number, (pair, winner) in enumerate(games.items(), start=1):
            home, away = pair
            game_id += 1
            completed = played_through is not None and round_id <= played_through
            state = "Final" if completed else "Pre Game"
            home_score = away_score = None
            if completed:
                home_score, away_score = (30, 12) if winner == home else (12, 30)
            rows.append(
                (game_id, 2025, round_id, round_name, number, game_id, state,
                 TEAMS[home], TEAMS[away], home, away, home_score, away_score)
            )
    con.executemany(
        "INSERT INTO footy_tipping_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()


def _predictions(db_path, round_id, model_probabilities=None):
    con = sqlite3.connect(db_path)
    frame = pd.read_sql_query(
        "SELECT game_id, competition_year, CAST(round_id AS INTEGER) AS round_id, "
        "round_name, team_home, team_away FROM footy_tipping_data "
        "WHERE CAST(round_id AS INTEGER) = ?",
        con,
        params=(round_id,),
    )
    con.close()
    probabilities = model_probabilities or {}
    frame["home_team_win_prob"] = [
        probabilities.get(team) for team in frame["team_home"]
    ]
    frame["home_team_lose_prob"] = [
        None if probabilities.get(team) is None else 1.0 - probabilities[team]
        for team in frame["team_home"]
    ]
    return frame


class PremiershipRaceTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)

    def tearDown(self):
        os.remove(self.db_path)

    def test_week_one_prices_every_finalist(self):
        _build_db(self.db_path, played_through=27)
        race = pr.premiership_race(self.db_path, _predictions(self.db_path, 28), simulations=4000)
        self.assertTrue(race["available"], race["reason"])
        self.assertEqual(race["stage"], rounds.FINALS_WEEK_1)
        self.assertEqual(len(race["teams"]), 8)
        self.assertTrue(all(team["alive"] for team in race["teams"]))
        self.assertAlmostEqual(
            sum(team["p_premiership"] for team in race["teams"]), 1.0, places=9
        )

    def test_probabilities_are_ordered_and_sum_to_one_at_every_stage(self):
        for played_through, round_id, stage, alive in (
            (27, 28, rounds.FINALS_WEEK_1, 8),
            (28, 29, rounds.FINALS_WEEK_2, 6),
            (29, 30, rounds.PRELIMINARY, 4),
            (30, 31, rounds.GRAND_FINAL, 2),
        ):
            with self.subTest(stage=stage):
                _build_db(self.db_path, played_through=played_through)
                race = pr.premiership_race(
                    self.db_path, _predictions(self.db_path, round_id), simulations=4000
                )
                self.assertTrue(race["available"], race["reason"])
                self.assertEqual(race["stage"], stage)
                self.assertAlmostEqual(
                    sum(team["p_premiership"] for team in race["teams"]), 1.0, places=9
                )
                self.assertEqual(
                    sum(1 for team in race["teams"] if team["alive"]), alive
                )
                # An eliminated team cannot win the premiership.
                for team in race["teams"]:
                    if not team["alive"]:
                        self.assertEqual(team["p_premiership"], 0.0)
                        self.assertEqual(team["p_grand_final"], 0.0)
                ordered = [team["p_premiership"] for team in race["teams"]]
                self.assertEqual(ordered, sorted(ordered, reverse=True))
                os.remove(self.db_path)
                handle, self.db_path = tempfile.mkstemp(suffix=".sqlite")
                os.close(handle)

    def test_qualifying_final_winners_stay_alive_through_their_bye(self):
        """Week two has six live teams: four playing and two resting."""
        _build_db(self.db_path, played_through=28)
        race = pr.premiership_race(self.db_path, _predictions(self.db_path, 29), simulations=2000)
        alive = {team["team"] for team in race["teams"] if team["alive"]}
        # 2025 week one: seeds 2 and 4 won their qualifying finals.
        self.assertIn(TEAMS[2], alive)
        self.assertIn(TEAMS[4], alive)
        # Seeds 6 and 8 lost elimination finals and are gone.
        self.assertNotIn(TEAMS[6], alive)
        self.assertNotIn(TEAMS[8], alive)

    def test_grand_finalists_both_reach_the_grand_final_with_certainty(self):
        _build_db(self.db_path, played_through=30)
        race = pr.premiership_race(self.db_path, _predictions(self.db_path, 31), simulations=2000)
        live = [team for team in race["teams"] if team["alive"]]
        self.assertEqual(len(live), 2)
        for team in live:
            self.assertEqual(team["p_grand_final"], 1.0)

    def test_the_same_inputs_produce_the_same_table(self):
        _build_db(self.db_path, played_through=27)
        predictions = _predictions(self.db_path, 28)
        first = pr.premiership_race(self.db_path, predictions, simulations=4000)
        second = pr.premiership_race(self.db_path, predictions, simulations=4000)
        self.assertEqual(first["teams"], second["teams"])

    def test_model_probabilities_take_precedence_over_ratings(self):
        """A scheduled fixture must use the calibrated model, not the ratings."""
        _build_db(self.db_path, played_through=27)
        baseline = pr.premiership_race(
            self.db_path, _predictions(self.db_path, 28), simulations=8000
        )
        # Tell the model the top seed is certain to lose its qualifying final.
        forced = _predictions(self.db_path, 28, model_probabilities={TEAMS[1]: 0.001})
        skewed = pr.premiership_race(self.db_path, forced, simulations=8000)
        self.assertEqual(skewed["scheduled_games_priced_by_model"], 1)
        top_seed = {team["seed"]: team for team in skewed["teams"]}[1]
        self.assertLess(
            top_seed["p_premiership"],
            {team["seed"]: team for team in baseline["teams"]}[1]["p_premiership"],
        )

    def test_a_bracket_that_is_not_the_top_eight_fails_soft(self):
        _build_db(self.db_path, played_through=27)
        con = sqlite3.connect(self.db_path)
        con.execute(
            "UPDATE footy_tipping_data SET position_away_ladder = 9 "
            "WHERE CAST(round_id AS INTEGER) = 28 AND position_away_ladder = 8"
        )
        con.commit()
        con.close()
        race = pr.premiership_race(self.db_path, _predictions(self.db_path, 28))
        self.assertFalse(race["available"])
        self.assertIn("1 to 8", race["reason"])
        self.assertEqual(race["teams"], [])

    def test_missing_finals_fixtures_fail_soft(self):
        _build_db(self.db_path, played_through=27)
        con = sqlite3.connect(self.db_path)
        con.execute("DELETE FROM footy_tipping_data WHERE CAST(round_id AS INTEGER) >= 29")
        con.execute("DELETE FROM footy_tipping_data WHERE CAST(round_id AS INTEGER) = 28 AND game_number > 2")
        con.commit()
        con.close()
        race = pr.premiership_race(self.db_path, _predictions(self.db_path, 28))
        self.assertFalse(race["available"])
        self.assertIn("week-one", race["reason"])

    def test_empty_predictions_fail_soft(self):
        _build_db(self.db_path, played_through=27)
        race = pr.premiership_race(self.db_path, pd.DataFrame())
        self.assertFalse(race["available"])

    def test_a_missing_database_fails_soft(self):
        race = pr.premiership_race("/nonexistent/footy.sqlite", pd.DataFrame())
        self.assertFalse(race["available"])


if __name__ == "__main__":
    unittest.main()
