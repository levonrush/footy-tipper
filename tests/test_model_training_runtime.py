import os
import inspect
import re
import unittest
from unittest.mock import patch

try:
    import lightgbm as lgb
    from pipeline.common.model_training import modelling_functions as mf
except ModuleNotFoundError:  # Smoke workflow installs the full training stack.
    lgb = None
    mf = None


@unittest.skipIf(lgb is None, "lightgbm training dependency is not installed")
class ModelTrainingRuntimeTests(unittest.TestCase):
    @staticmethod
    def _create_search():
        estimator = mf.score_regressor()
        pipeline = mf.create_pipeline(
            estimator=estimator,
            search_spaces={"n_estimators": [20]},
            use_rfe=False,
            cv=2,
            opt_metric="neg_mean_absolute_error",
            cat_cols=[],
        )
        return pipeline.named_steps["hyperparamtuning"]

    def test_search_defaults_to_full_local_tuning_budget(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOOTY_TIPPER_TUNE_ITER", None)
            search = self._create_search()

        self.assertEqual(search.n_iter, 100)
        self.assertEqual(search.n_jobs, -1)
        self.assertEqual(search.estimator.get_params()["n_jobs"], 1)

    def test_search_accepts_explicit_tuning_budget_override(self):
        with patch.dict(os.environ, {"FOOTY_TIPPER_TUNE_ITER": "17"}):
            search = self._create_search()

        self.assertEqual(search.n_iter, 17)

    @staticmethod
    def _fitted_to_df():
        """Build the pipeline and fit its preprocessor, then hand back a
        callable that pushes one all-NaN row through the to_df step."""
        import pandas as pd

        pipeline = mf.create_pipeline(
            estimator=mf.score_regressor(),
            search_spaces={"n_estimators": [20]},
            use_rfe=False,
            cv=2,
            opt_metric="neg_mean_absolute_error",
            cat_cols=[],
        )
        preprocessor = pipeline.named_steps["one_hot"]
        preprocessor.fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        to_df = pipeline.named_steps["to_df"].func

        def transform_nan_row():
            import numpy as np

            arr = preprocessor.transform(pd.DataFrame({"a": [np.nan], "b": [1.0]}))
            return to_df(arr)

        return transform_nan_row

    def test_missing_values_are_zero_filled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOOTY_TIPPER_NAN_PASSTHROUGH", None)
            transform_nan_row = self._fitted_to_df()

        self.assertEqual(float(transform_nan_row().iloc[0, 0]), 0.0)

    def test_nan_passthrough_is_baked_in_at_build_time(self):
        """The setting must survive into the fitted transformer.

        If to_df re-read the environment when transforming, a model trained on
        NaN would silently switch to zero-fill at serve time.
        """
        import numpy as np

        with patch.dict(os.environ, {"FOOTY_TIPPER_NAN_PASSTHROUGH": "true"}):
            transform_nan_row = self._fitted_to_df()

        # Environment is restored here, so a transform-time read would zero-fill.
        self.assertTrue(np.isnan(float(transform_nan_row().iloc[0, 0])))

    def test_every_stochastic_estimator_is_seeded(self):
        """An unseeded search made releases irreproducible: two runs on the
        same data differed by ~1.3pp accuracy, flipping the acceptance gate."""
        search = self._create_search()

        self.assertEqual(search.random_state, mf.training_seed())
        self.assertEqual(
            search.estimator.get_params()["random_state"], mf.training_seed()
        )

    def test_training_seed_is_overridable_and_falls_back_cleanly(self):
        with patch.dict(os.environ, {"FOOTY_TIPPER_TRAINING_SEED": "1234"}):
            self.assertEqual(mf.training_seed(), 1234)
        with patch.dict(os.environ, {"FOOTY_TIPPER_TRAINING_SEED": "not-a-number"}):
            self.assertEqual(mf.training_seed(), mf.TRAINING_SEED)

    def test_binary_classifiers_are_seeded(self):
        source = inspect.getsource(mf)
        unseeded = re.findall(
            r"lgb\.LGBM(?:Regressor|Classifier)\((?:(?!random_state)[^()])*\)",
            source,
        )
        self.assertEqual(unseeded, [], f"unseeded estimators: {unseeded}")

    def test_no_lightgbm_fit_uses_nested_internal_parallelism(self):
        source = inspect.getsource(mf)
        nested = re.findall(
            r"LGBM(?:Regressor|Classifier)\([^)]*?n_jobs\s*=\s*-1",
            source,
            flags=re.DOTALL,
        )
        self.assertEqual(nested, [])


if __name__ == "__main__":
    unittest.main()
