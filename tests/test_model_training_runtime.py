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
