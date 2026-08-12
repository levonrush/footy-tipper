import unittest

import pandas as pd

from pipeline.common.lineups.features import build_lineup_match_features


def _team_entries(snapshot_id, year, round_id, team_key, players, published_at):
    rows = []
    for player_name, jersey_number, listed_position, squad_group in players:
        player_key = player_name.lower().replace(" ", "_")
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "competition_year": year,
                "round_id": round_id,
                "team_key": team_key,
                "player_name": player_name,
                "player_key": player_key,
                "jersey_number": jersey_number,
                "listed_position": listed_position,
                "squad_group": squad_group,
                "source_published_at_utc": published_at,
                "inserted_at_utc": published_at,
            }
        )
    return rows


class LineupFeatureTests(unittest.TestCase):
    def test_build_lineup_features_with_retention_and_deltas(self):
        matches = pd.DataFrame(
            [
                {
                    "game_id": 101,
                    "competition_year": 2026,
                    "round_id": 1,
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2026-03-01T02:15:00Z",
                    "team_home": "Knights",
                    "team_away": "Cowboys",
                    "team_final_score_home": 20,
                    "team_final_score_away": 10,
                },
                {
                    "game_id": 102,
                    "competition_year": 2026,
                    "round_id": 2,
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2026-03-08T02:15:00Z",
                    "team_home": "Knights",
                    "team_away": "Broncos",
                    "team_final_score_home": 18,
                    "team_final_score_away": 16,
                },
                {
                    "game_id": 103,
                    "competition_year": 2026,
                    "round_id": 3,
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2026-03-15T02:15:00Z",
                    "team_home": "Knights",
                    "team_away": "Bulldogs",
                    "team_final_score_home": 22,
                    "team_final_score_away": 18,
                },
            ]
        )

        knights_r1 = [
            ("Kalyn Ponga", 1, "Fullback", "backs"),
            ("Fletcher Sharpe", 6, "Five-Eighth", "backs"),
            ("Dylan Brown", 7, "Halfback", "backs"),
            ("Phoenix Crossland", 9, "Hooker", "forwards"),
            ("Sandon Smith", 14, "Hooker", "interchange"),
            ("Jack Hetherington", 18, "Prop", "reserves"),
        ]
        knights_r2 = [
            ("Kalyn Ponga", 1, "Fullback", "backs"),
            ("Fletcher Sharpe", 6, "Five-Eighth", "backs"),
            ("Dylan Brown", 7, "Halfback", "backs"),
            ("Phoenix Crossland", 9, "Hooker", "forwards"),
            ("Sandon Smith", 14, "Hooker", "interchange"),
            ("Jack Hetherington", 18, "Prop", "reserves"),
        ]
        knights_r2_late = [
            ("Kalyn Ponga", 1, "Fullback", "backs"),
            ("Fletcher Sharpe", 6, "Five-Eighth", "backs"),
            ("Tyson Gamble", 7, "Halfback", "backs"),
            ("Phoenix Crossland", 9, "Hooker", "forwards"),
            ("Sandon Smith", 14, "Hooker", "interchange"),
            ("Jack Hetherington", 18, "Prop", "reserves"),
        ]
        cowboys_r1 = [
            ("Scott Drinkwater", 1, "Fullback", "backs"),
            ("Jake Clifford", 6, "Five-Eighth", "backs"),
            ("Tom Dearden", 7, "Halfback", "backs"),
            ("Reece Robson", 9, "Hooker", "forwards"),
            ("Sam McIntyre", 14, "Hooker", "interchange"),
            ("Jordan McLean", 18, "Prop", "reserves"),
        ]
        broncos_r2 = [
            ("Reece Walsh", 1, "Fullback", "backs"),
            ("Ezra Mam", 6, "Five-Eighth", "backs"),
            ("Ben Hunt", 7, "Halfback", "backs"),
            ("Billy Walters", 14, "Hooker", "interchange"),
            ("Xavier Willison", 18, "Prop", "reserves"),
        ]
        knights_r3 = [
            ("Kalyn Ponga", 1, "Fullback", "backs"),
            ("Fletcher Sharpe", 6, "Five-Eighth", "backs"),
            ("Tyson Gamble", 7, "Halfback", "backs"),
            ("Phoenix Crossland", 9, "Hooker", "forwards"),
            ("Sandon Smith", 14, "Hooker", "interchange"),
            ("Jack Hetherington", 18, "Prop", "reserves"),
        ]
        bulldogs_r3 = [
            ("Connor Tracey", 1, "Fullback", "backs"),
            ("Matt Burton", 6, "Five-Eighth", "backs"),
            ("Toby Sexton", 7, "Halfback", "backs"),
            ("Reed Mahoney", 9, "Hooker", "forwards"),
            ("Kurt Mann", 14, "Lock", "interchange"),
            ("Samuel Hughes", 18, "Prop", "reserves"),
        ]

        entries = pd.DataFrame(
            _team_entries(1, 2026, 1, "knights", knights_r1, "2026-02-25T04:00:00Z")
            + _team_entries(1, 2026, 1, "cowboys", cowboys_r1, "2026-02-25T04:00:00Z")
            + _team_entries(2, 2026, 2, "knights", knights_r2, "2026-03-03T04:00:00Z")
            + _team_entries(3, 2026, 2, "knights", knights_r2_late, "2026-03-04T04:00:00Z")
            + _team_entries(4, 2026, 2, "broncos", broncos_r2, "2026-03-04T04:00:00Z")
            + _team_entries(5, 2026, 3, "knights", knights_r3, "2026-03-10T04:00:00Z")
            + _team_entries(5, 2026, 3, "bulldogs", bulldogs_r3, "2026-03-10T04:00:00Z")
        )

        features = build_lineup_match_features(matches, entries).set_index("game_id")

        self.assertEqual(features.loc[101, "lineup_data_available_home"], 1.0)
        self.assertEqual(features.loc[101, "lineup_named_count_home"], 5.0)
        self.assertEqual(features.loc[101, "lineup_interchange_count_home"], 1.0)
        self.assertEqual(features.loc[101, "lineup_reserve_count_home"], 1.0)
        self.assertEqual(features.loc[101, "lineup_spine_count_home"], 4.0)
        self.assertEqual(features.loc[101, "lineup_spine_complete_home"], 1.0)
        self.assertEqual(features.loc[101, "lineup_bench_hooker_count_home"], 1.0)
        self.assertEqual(features.loc[101, "lineup_bench_spine_cover_count_home"], 1.0)
        self.assertAlmostEqual(features.loc[101, "lineup_retained_ratio_home"], 0.0, places=6)
        self.assertGreater(features.loc[101, "lineup_expected_named_count_home"], 0.0)
        self.assertGreater(features.loc[101, "lineup_expected_spine_count_home"], 0.0)
        self.assertGreater(features.loc[101, "lineup_selection_uncertainty_home"], 0.0)
        self.assertEqual(features.loc[101, "lineup_snapshot_count_home"], 1.0)
        self.assertEqual(features.loc[101, "lineup_named_change_count_home"], 0.0)
        self.assertEqual(features.loc[101, "lineup_avg_named_experience_home"], 0.0)
        self.assertEqual(features.loc[101, "lineup_avg_named_margin_rating_home"], 0.0)
        self.assertEqual(features.loc[101, "lineup_named_cohesion_home"], 0.0)

        self.assertAlmostEqual(features.loc[102, "lineup_retained_ratio_home"], 5 / 6, places=6)
        self.assertAlmostEqual(features.loc[102, "lineup_starters_retained_ratio_home"], 4 / 5, places=6)
        self.assertAlmostEqual(features.loc[102, "lineup_spine_retained_ratio_home"], 3 / 4, places=6)
        self.assertEqual(features.loc[102, "lineup_spine_same_as_prev_home"], 0.0)
        self.assertEqual(features.loc[102, "lineup_halves_pair_same_as_prev_home"], 0.0)
        self.assertAlmostEqual(features.loc[102, "lineup_avg_named_experience_home"], 4 / 5, places=6)
        self.assertAlmostEqual(features.loc[102, "lineup_avg_spine_experience_home"], 3 / 4, places=6)
        self.assertAlmostEqual(features.loc[102, "lineup_avg_halves_experience_home"], 0.5, places=6)
        self.assertAlmostEqual(features.loc[102, "lineup_avg_middles_experience_home"], 1.0, places=6)
        self.assertAlmostEqual(features.loc[102, "lineup_avg_outside_backs_experience_home"], 1.0, places=6)
        self.assertAlmostEqual(features.loc[102, "lineup_avg_interchange_experience_home"], 1.0, places=6)
        self.assertGreater(features.loc[102, "lineup_avg_named_margin_rating_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_spine_margin_rating_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_halves_margin_rating_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_middles_margin_rating_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_outside_backs_margin_rating_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_interchange_margin_rating_home"], 0.0)
        self.assertEqual(features.loc[102, "lineup_debutant_count_home"], 1.0)
        self.assertGreater(features.loc[102, "lineup_named_cohesion_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_spine_cohesion_home"], 0.0)
        self.assertEqual(features.loc[102, "lineup_halves_pair_cohesion_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_recent_named_stability_home"], 0.0)
        self.assertGreater(features.loc[102, "lineup_recent_spine_stability_home"], 0.0)
        self.assertEqual(features.loc[102, "lineup_snapshot_count_home"], 2.0)
        self.assertEqual(features.loc[102, "lineup_snapshot_window_hours_home"], 24.0)
        self.assertEqual(features.loc[102, "lineup_named_change_count_home"], 2.0)
        self.assertEqual(features.loc[102, "lineup_named_change_rate_home"], 2.0)
        self.assertEqual(features.loc[102, "lineup_spine_change_count_home"], 2.0)
        self.assertEqual(features.loc[102, "lineup_spine_change_rate_home"], 2.0)
        self.assertGreater(features.loc[102, "lineup_avg_named_experience_delta"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_halves_experience_delta"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_named_margin_rating_delta"], 0.0)
        self.assertGreater(features.loc[102, "lineup_avg_halves_margin_rating_delta"], 0.0)
        self.assertGreater(features.loc[102, "lineup_named_cohesion_delta"], 0.0)
        self.assertGreater(features.loc[102, "lineup_recent_named_stability_delta"], 0.0)
        self.assertGreater(features.loc[102, "lineup_snapshot_count_delta"], 0.0)
        self.assertGreater(features.loc[102, "lineup_spine_count_delta"], 0.0)
        self.assertEqual(features.loc[102, "lineup_features_missing"], 0.0)

        self.assertEqual(features.loc[103, "lineup_halves_pair_same_as_prev_home"], 1.0)
        self.assertGreater(features.loc[103, "lineup_halves_pair_cohesion_home"], 0.0)
        self.assertGreater(features.loc[103, "lineup_recent_named_stability_home"], 0.0)
        self.assertGreater(features.loc[103, "lineup_recent_spine_stability_home"], 0.0)

    def test_build_lineup_features_handles_empty_lineup_rows(self):
        matches = pd.DataFrame(
            [
                {
                    "game_id": 999,
                    "competition_year": 2026,
                    "round_id": 1,
                    "team_home": "Knights",
                    "team_away": "Cowboys",
                    "start_time": "2026-03-01T02:15:00Z",
                }
            ]
        )
        features = build_lineup_match_features(matches, pd.DataFrame())
        self.assertEqual(len(features), 1)
        self.assertEqual(float(features.iloc[0]["lineup_features_missing"]), 1.0)
        self.assertEqual(features.iloc[0]["lineup_home_players"], "")
        self.assertEqual(features.iloc[0]["lineup_away_players"], "")

    def test_build_lineup_features_handles_unix_second_start_times(self):
        matches = pd.DataFrame(
            [
                {
                    "game_id": 301,
                    "competition_year": 2026,
                    "round_id": 1,
                    "game_number": 1,
                    "game_state_name": "Pre Game",
                    "start_time": 1772302500.0,
                    "team_home": "Newcastle Knights",
                    "team_away": "North Queensland Cowboys",
                }
            ]
        )

        entries = pd.DataFrame(
            _team_entries(1, 2026, 1, "knights", [("Kalyn Ponga", 1, "Fullback", "backs")], "2026-02-25T04:00:00Z")
            + _team_entries(1, 2026, 1, "cowboys", [("Scott Drinkwater", 1, "Fullback", "backs")], "2026-02-25T04:00:00Z")
        )

        features = build_lineup_match_features(matches, entries).set_index("game_id")

        self.assertEqual(features.loc[301, "lineup_data_available_home"], 1.0)
        self.assertEqual(features.loc[301, "lineup_data_available_away"], 1.0)
        self.assertEqual(features.loc[301, "lineup_features_missing"], 0.0)

    def test_post_kickoff_snapshot_is_never_selected_for_a_final_game(self):
        """The as-of cutoff is the only guard against training on played sides.

        Historical match-centre pages carry the actually-played 17 stamped with
        the backfill date, so a snapshot dated after kickoff must never be
        chosen for a completed match.
        """
        matches = pd.DataFrame(
            [
                {
                    "game_id": 401,
                    "competition_year": 2010,
                    "round_id": 5,
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2010-04-01T09:00:00Z",
                    "team_home": "Knights",
                    "team_away": "Cowboys",
                }
            ]
        )
        # Published years after the match, exactly like the 2024 backfill.
        entries = pd.DataFrame(
            _team_entries(1, 2010, 5, "knights", [("Kurt Gidley", 1, "Fullback", "backs")], "2024-09-07T12:29:36Z")
            + _team_entries(1, 2010, 5, "cowboys", [("Matt Bowen", 1, "Fullback", "backs")], "2024-09-07T12:29:36Z")
        )

        features = build_lineup_match_features(matches, entries).set_index("game_id")

        self.assertEqual(features.loc[401, "lineup_features_missing"], 1.0)
        self.assertEqual(features.loc[401, "lineup_data_available_home"], 0.0)
        self.assertEqual(features.loc[401, "lineup_data_available_away"], 0.0)

    def test_as_of_cutoff_uses_true_utc_kickoff_not_venue_local(self):
        """The cutoff must be measured from the real kickoff instant.

        `start_time` is venue-local wall clock serialised as-if-UTC, while
        article publish times are true UTC. Building the cutoff from
        `start_time` shifted the guard by the venue's offset, so a documented
        24h horizon ran at about 13h for Australian venues. A snapshot
        published 18h before a Sydney kickoff is inside that broken window and
        outside the real one, so it pins the direction of the fix.
        """
        true_kickoff = pd.Timestamp("2026-03-25T08:50:00Z")  # 19:50 Sydney
        local_as_utc = true_kickoff + pd.Timedelta(hours=11)
        published = true_kickoff - pd.Timedelta(hours=18)

        def _matches(include_utc):
            row = {
                "game_id": 501,
                "competition_year": 2026,
                "round_id": 3,
                "game_number": 1,
                "game_state_name": "Final",
                "start_time": local_as_utc.isoformat(),
                "team_home": "Roosters",
                "team_away": "Storm",
            }
            if include_utc:
                row["start_time_utc"] = true_kickoff.isoformat()
            return pd.DataFrame([row])

        entries = pd.DataFrame(
            _team_entries(1, 2026, 3, "roosters", [("James Tedesco", 1, "Fullback", "backs")], published.isoformat())
            + _team_entries(1, 2026, 3, "storm", [("Ryan Papenhuyzen", 1, "Fullback", "backs")], published.isoformat())
        )

        with_utc = build_lineup_match_features(_matches(True), entries).set_index("game_id")
        self.assertEqual(with_utc.loc[501, "lineup_features_missing"], 1.0)

        # Without the true-UTC column the cutoff falls back to local-as-UTC,
        # which is the permissive behaviour the fix removes.
        without_utc = build_lineup_match_features(_matches(False), entries).set_index("game_id")
        self.assertEqual(without_utc.loc[501, "lineup_features_missing"], 0.0)

    def test_finals_round_id_is_recovered_from_round_name(self):
        """parse_round_id finds no number in "Finals Week 1", so entries arrive
        with a NULL round_id and used to be dropped wholesale."""
        matches = pd.DataFrame(
            [
                {
                    "game_id": 501,
                    "competition_year": 2026,
                    "round_id": 28,
                    "round_name": "Finals Week 1",
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2026-09-05T09:00:00Z",
                    "team_home": "Knights",
                    "team_away": "Cowboys",
                }
            ]
        )
        entries = pd.DataFrame(
            _team_entries(1, 2026, None, "knights", [("Kalyn Ponga", 1, "Fullback", "backs")], "2026-09-01T04:00:00Z")
            + _team_entries(1, 2026, None, "cowboys", [("Scott Drinkwater", 1, "Fullback", "backs")], "2026-09-01T04:00:00Z")
        )
        entries["round_name"] = "Finals Week 1"

        features = build_lineup_match_features(matches, entries).set_index("game_id")

        self.assertEqual(features.loc[501, "lineup_features_missing"], 0.0)
        self.assertEqual(features.loc[501, "lineup_data_available_home"], 1.0)

    def test_continuity_resets_across_a_coverage_gap(self):
        """Retention must compare against the previous fixture, not the
        previous *covered* fixture, which can be seasons earlier."""
        matches = pd.DataFrame(
            [
                {
                    "game_id": 601,
                    "competition_year": 2012,
                    "round_id": 1,
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2012-03-01T09:00:00Z",
                    "team_home": "Knights",
                    "team_away": "Cowboys",
                },
                {  # uncovered: no lineup entries for this round
                    "game_id": 602,
                    "competition_year": 2012,
                    "round_id": 2,
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2012-03-08T09:00:00Z",
                    "team_home": "Knights",
                    "team_away": "Broncos",
                },
                {
                    "game_id": 603,
                    "competition_year": 2012,
                    "round_id": 3,
                    "game_number": 1,
                    "game_state_name": "Final",
                    "start_time": "2012-03-15T09:00:00Z",
                    "team_home": "Knights",
                    "team_away": "Cowboys",
                },
            ]
        )
        squad = [("Kurt Gidley", 1, "Fullback", "backs"), ("Jarrod Mullen", 6, "Five-Eighth", "backs")]
        entries = pd.DataFrame(
            _team_entries(1, 2012, 1, "knights", squad, "2012-02-25T04:00:00Z")
            + _team_entries(1, 2012, 1, "cowboys", squad, "2012-02-25T04:00:00Z")
            + _team_entries(3, 2012, 3, "knights", squad, "2012-03-11T04:00:00Z")
            + _team_entries(3, 2012, 3, "cowboys", squad, "2012-03-11T04:00:00Z")
        )

        features = build_lineup_match_features(matches, entries).set_index("game_id")

        # Round 2 was skipped, so round 3 has no usable "last week" to compare
        # against even though round 1 named an identical side.
        self.assertEqual(features.loc[603, "lineup_starters_retained_ratio_home"], 0.0)
        self.assertEqual(features.loc[603, "lineup_spine_same_as_prev_home"], 0.0)


class LineupColumnFillTests(unittest.TestCase):
    def test_merge_miss_defaults_to_missing_not_present(self):
        """A left-merge miss must not announce complete team lists."""
        from pipeline.common.lineups.features import (
            fill_lineup_feature_columns,
            lineup_coverage_fraction,
        )

        frame = pd.DataFrame(
            {
                "game_id": [1, 2],
                "lineup_features_missing": [0.0, None],
                "lineup_named_count_home": [17.0, None],
                "lineup_home_players": ["a;b", None],
            }
        )
        filled = fill_lineup_feature_columns(frame)

        self.assertEqual(filled.loc[1, "lineup_features_missing"], 1.0)
        self.assertEqual(filled.loc[1, "lineup_named_count_home"], 0.0)
        self.assertEqual(filled.loc[1, "lineup_home_players"], "")
        self.assertAlmostEqual(lineup_coverage_fraction(filled), 0.5)


if __name__ == "__main__":
    unittest.main()
