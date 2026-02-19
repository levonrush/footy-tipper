import unittest

import numpy as np
import pandas as pd

from pipeline.common.model_training.cv import InSeasonSplit


class InSeasonSplitTests(unittest.TestCase):
    def test_split_is_round_blocked_within_each_season(self):
        X = pd.DataFrame(
            {
                "round_id": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
                "feature": list(range(12)),
            }
        )
        groups = np.array([2024] * 6 + [2025] * 6)

        cv = InSeasonSplit(n_splits=3)
        splits = list(cv.split(X, groups=groups))

        self.assertGreater(len(splits), 0)
        for train_idx, test_idx in splits:
            train_rounds = set(X.iloc[train_idx]["round_id"].tolist())
            test_rounds = set(X.iloc[test_idx]["round_id"].tolist())
            self.assertTrue(train_rounds.isdisjoint(test_rounds))

            train_seasons = set(groups[train_idx].tolist())
            test_seasons = set(groups[test_idx].tolist())
            self.assertEqual(len(train_seasons), 1)
            self.assertEqual(len(test_seasons), 1)
            self.assertEqual(train_seasons, test_seasons)

    def test_split_requires_groups(self):
        X = pd.DataFrame({"round_id": [1, 2, 3]})
        cv = InSeasonSplit(n_splits=2)
        with self.assertRaises(ValueError):
            list(cv.split(X))

    def test_split_requires_round_id_column(self):
        X = pd.DataFrame({"feature": [1, 2, 3]})
        groups = np.array([2024, 2024, 2024])
        cv = InSeasonSplit(n_splits=2)
        with self.assertRaises(ValueError):
            list(cv.split(X, groups=groups))

    def test_split_accepts_remainder_round_id_column(self):
        X = pd.DataFrame(
            {
                "remainder__round_id": [1, 1, 2, 2, 3, 3],
                "feature": [10, 11, 12, 13, 14, 15],
            }
        )
        groups = np.array([2026] * 6)
        cv = InSeasonSplit(n_splits=3)

        splits = list(cv.split(X, groups=groups))
        self.assertGreater(len(splits), 0)


if __name__ == "__main__":
    unittest.main()
