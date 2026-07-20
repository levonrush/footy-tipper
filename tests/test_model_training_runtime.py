import os
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
    def test_search_uses_bounded_iterations_and_outer_parallelism(self):
        estimator = mf.score_regressor()
        self.assertEqual(estimator.get_params()["n_jobs"], 1)

        with patch.dict(os.environ, {"FOOTY_TIPPER_TUNE_ITER": "10"}):
            pipeline = mf.create_pipeline(
                estimator=estimator,
                search_spaces={"n_estimators": [20]},
                use_rfe=False,
                cv=2,
                opt_metric="neg_mean_absolute_error",
                cat_cols=[],
            )

        search = pipeline.named_steps["hyperparamtuning"]
        self.assertEqual(search.n_iter, 10)
        self.assertEqual(search.n_jobs, -1)
        self.assertEqual(search.estimator.get_params()["n_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
