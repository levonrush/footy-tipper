import unittest

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover - environment without the modelling stack
    HAVE_LIGHTGBM = False

if HAVE_LIGHTGBM:
    from pipeline.common.explain import contributions as xc
    from pipeline.common.model_training.modelling_functions import sanitize_feature_names


CAT_COLS = ["round_name", "venue_name"]
NUM_COLS = ["home_elo", "away_elo", "lineup_spine_count_home", "travel_km_delta"]


def _frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            # Levels deliberately carry spaces, so a name-parsing implementation
            # of the group map would break here.
            "round_name": rng.choice(["Round 1", "Finals Week 1", "Round 12"], n),
            "venue_name": rng.choice(["Suncorp Stadium", "AAMI Park"], n),
            "home_elo": rng.normal(1500, 60, n),
            "away_elo": rng.normal(1500, 60, n),
            "lineup_spine_count_home": rng.integers(0, 5, n).astype(float),
            "travel_km_delta": rng.normal(0, 400, n),
        }
    )


def _build_pipeline(estimator, X, y, wrap=False):
    """Mirror modelling_functions.create_pipeline's shape on a toy problem."""
    preprocessor = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough",
    )

    def to_df(X_array):
        cols = preprocessor.get_feature_names_out(preprocessor.feature_names_in_)
        df = pd.DataFrame(X_array, columns=sanitize_feature_names(cols))
        return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    final = _FakeSearch(estimator) if wrap else estimator
    pipe = Pipeline(
        [
            ("one_hot", preprocessor),
            ("to_df", FunctionTransformer(func=to_df, validate=False)),
            ("model", final),
        ]
    )
    return pipe.fit(X, y)


class _FakeSearch:
    """Stands in for BayesSearchCV: exposes the fitted model as best_estimator_."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.best_estimator_ = self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)


@unittest.skipUnless(HAVE_LIGHTGBM, "lightgbm/sklearn not available")
class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = _frame()
        rng = np.random.default_rng(1)
        signal = (cls.X["home_elo"] - cls.X["away_elo"]) / 50.0
        cls.y_binary = (signal + rng.normal(0, 0.5, len(cls.X)) > 0).astype(int)
        cls.y_count = np.maximum(20 + signal * 2, 1.0)

        cls.clf_pipe = _build_pipeline(
            lgb.LGBMClassifier(n_estimators=25, num_leaves=4, verbose=-1),
            cls.X,
            cls.y_binary,
        )
        cls.reg_pipe = _build_pipeline(
            lgb.LGBMRegressor(objective="poisson", n_estimators=25, num_leaves=4, verbose=-1),
            cls.X,
            cls.y_count,
            wrap=True,
        )

    def test_final_estimator_resolves_wrapped_and_bare_pipelines(self):
        self.assertIsInstance(xc.final_estimator(self.clf_pipe), lgb.LGBMClassifier)
        self.assertIsInstance(xc.final_estimator(self.reg_pipe), lgb.LGBMRegressor)

    def test_link_is_inferred_per_model_type(self):
        self.assertEqual(xc.link_for(self.clf_pipe), xc.LINK_LOG_ODDS)
        self.assertEqual(xc.link_for(self.reg_pipe), xc.LINK_LOG_MEAN)

    def test_feature_alignment_matches_the_booster(self):
        for pipe in (self.clf_pipe, self.reg_pipe):
            xc.verify_feature_alignment(pipe)
            ct = pipe.named_steps["one_hot"]
            expected = sanitize_feature_names(
                ct.get_feature_names_out(ct.feature_names_in_)
            )
            self.assertEqual(list(expected), list(xc.booster_of(pipe).feature_name()))

    def test_group_map_is_a_total_disjoint_partition(self):
        for pipe in (self.clf_pipe, self.reg_pipe):
            raw_names, widths = xc.onehot_group_map(pipe)
            self.assertEqual(len(raw_names), len(CAT_COLS) + len(NUM_COLS))
            self.assertEqual(int(widths.sum()), xc.booster_of(pipe).num_feature())
            # Categorical blocks are as wide as the encoder has levels; every
            # passthrough column maps one to one.
            self.assertTrue((widths[len(CAT_COLS):] == 1).all())
            self.assertEqual(set(raw_names), set(CAT_COLS) | set(NUM_COLS))

    def test_contributions_are_additive_against_the_raw_score(self):
        for pipe in (self.clf_pipe, self.reg_pipe):
            contribs = xc.raw_contributions(pipe, self.X, chunk_rows=7)
            transformed = xc.transform_frame(pipe, self.X)
            raw_score = xc.booster_of(pipe).predict(transformed, raw_score=True)
            np.testing.assert_allclose(
                contribs.prediction_link, raw_score, rtol=0, atol=1e-9
            )

    def test_categorical_group_equals_the_sum_of_its_one_hot_columns(self):
        pipe = self.clf_pipe
        contribs = xc.raw_contributions(pipe, self.X)
        transformed = xc.transform_frame(pipe, self.X)
        full = np.asarray(
            xc.booster_of(pipe).predict(transformed, pred_contrib=True), dtype=float
        )[:, :-1]

        names = list(xc.booster_of(pipe).feature_name())
        cols = [i for i, name in enumerate(names) if name.startswith("encoder__round_name")]
        expected = full[:, cols].sum(axis=1)

        idx = list(contribs.feature_names).index("round_name")
        np.testing.assert_allclose(contribs.values[:, idx], expected, atol=1e-12)

    def test_chunking_does_not_change_the_answer(self):
        small = xc.raw_contributions(self.clf_pipe, self.X, chunk_rows=3)
        large = xc.raw_contributions(self.clf_pipe, self.X, chunk_rows=10_000)
        np.testing.assert_allclose(small.values, large.values, atol=0)
        np.testing.assert_allclose(small.base_value, large.base_value, atol=0)

    def test_split_counts_cover_every_raw_predictor(self):
        splits = xc.raw_split_counts(self.clf_pipe)
        self.assertEqual(set(splits), set(CAT_COLS) | set(NUM_COLS))
        self.assertEqual(
            sum(splits.values()),
            int(np.asarray(xc.booster_of(self.clf_pipe).feature_importance("split")).sum()),
        )
        # The planted signal has to have been used.
        self.assertGreater(splits["home_elo"] + splits["away_elo"], 0)

    def test_align_to_reorders_and_reports_missing_predictors(self):
        contribs = xc.raw_contributions(self.clf_pipe, self.X)
        ordered = xc.align_to(contribs, ["travel_km_delta", "home_elo"])
        idx = list(contribs.feature_names).index("home_elo")
        np.testing.assert_allclose(ordered[:, 1], contribs.values[:, idx])
        with self.assertRaises(KeyError):
            xc.align_to(contribs, ["not_a_predictor"])


if __name__ == "__main__":
    unittest.main()
