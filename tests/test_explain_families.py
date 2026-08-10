import json
import pathlib
import unittest

import numpy as np

from pipeline.common.explain import families as fam

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "models" / "model_manifest.json"


def _deployed_predictors():
    if not MANIFEST.exists():
        return None
    with open(MANIFEST) as fh:
        return json.load(fh).get("predictors") or None


class FamilyTaxonomyTests(unittest.TestCase):
    def test_every_family_has_a_label(self):
        for family in fam.FAMILIES:
            self.assertIn(family, fam.FAMILY_LABELS)
            self.assertTrue(fam.family_label(family).strip())
        self.assertIsInstance(fam.FAMILY_TAXONOMY_VERSION, int)

    def test_training_config_predictors_are_all_classified(self):
        # The live guard: a new feature block added to training_config without a
        # taxonomy rule shows up here rather than silently as "unclassified"
        # noise in every cohort table.
        from pipeline.common.model_training import training_config as tc

        unclassified = sorted(
            name for name in tc.predictors if fam.family_for(name) == fam.UNCLASSIFIED
        )
        self.assertEqual(unclassified, [])

    def test_deployed_manifest_predictors_are_all_classified(self):
        predictors = _deployed_predictors()
        if predictors is None:
            self.skipTest("no model_manifest.json in this checkout")
        unclassified = sorted(
            name for name in predictors if fam.family_for(name) == fam.UNCLASSIFIED
        )
        self.assertEqual(unclassified, [])

    def test_rule_order_traps(self):
        # Elo before ladder.
        self.assertEqual(fam.family_for("home_elo_prob"), "elo")
        # Player form and role ratings before the broad lineup catch-all.
        self.assertEqual(fam.family_for("lineup_form_fantasy_home"), "player_form")
        self.assertEqual(fam.family_for("lineup_spine_form_fantasy_away"), "player_form")
        self.assertEqual(fam.family_for("lineup_rating_halves_delta"), "role_ratings")
        self.assertEqual(fam.family_for("lineup_spine_complete_home"), "lineup")
        # form_delta is season state, not a match-stat rolling mean.
        self.assertEqual(fam.family_for("form_delta"), "season_state")
        self.assertEqual(fam.family_for("form_line_breaks_delta"), "recent_form_stats")
        # Ladder columns that also start with season_.
        self.assertEqual(fam.family_for("season_form_home_ladder"), "ladder")
        self.assertEqual(fam.family_for("season_form_home"), "season_state")

    def test_unknown_predictor_falls_back_without_raising(self):
        self.assertEqual(fam.family_for("something_brand_new"), fam.UNCLASSIFIED)
        self.assertTrue(fam.family_label(fam.UNCLASSIFIED))

    def test_side_uses_suffix_before_prefix(self):
        # The away team's home-win rate belongs to the away side.
        self.assertEqual(fam.side_for("home_win_rate_away_ladder"), "away")
        self.assertEqual(fam.side_for("home_win_rate_home_ladder"), "home")
        self.assertEqual(fam.side_for("home_elo"), "home")
        self.assertEqual(fam.side_for("elo_diff"), "delta")
        self.assertEqual(fam.side_for("round_name"), "neutral")

    def test_deployed_predictors_are_side_balanced(self):
        predictors = _deployed_predictors()
        if predictors is None:
            self.skipTest("no model_manifest.json in this checkout")
        sides = [fam.side_for(name) for name in predictors]
        self.assertEqual(sides.count("home"), sides.count("away"))

    def test_group_by_family_sums_members_exactly(self):
        names = ["baseline_mu_home", "home_elo", "away_elo", "lineup_spine_count_home"]
        values = np.array([[1.0, 2.0, -0.5, 4.0], [0.0, 1.0, 1.0, -2.0]])

        grouped = fam.group_by_family(values, names)

        self.assertEqual(list(grouped.index), [0, 1])
        np.testing.assert_allclose(grouped["tier_a_baseline"], [1.0, 0.0])
        np.testing.assert_allclose(grouped["elo"], [1.5, 2.0])
        np.testing.assert_allclose(grouped["lineup"], [4.0, -2.0])
        # Grouping is a partition: family totals must equal the row total.
        np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1))

    def test_group_by_family_rejects_mismatched_names(self):
        with self.assertRaises(ValueError):
            fam.group_by_family(np.zeros((2, 3)), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
