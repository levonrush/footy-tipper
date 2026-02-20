import dill as pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


class IdentityCalibrator:
    def fit(self, probs, y):
        return self

    def predict(self, probs):
        clipped = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        return clipped


class BetaCalibrator:
    """Beta calibration: sigmoid(a * log(p) + b * log(1-p) + c)."""

    def __init__(self, c=1.0, max_iter=1000):
        self._model = LogisticRegression(C=c, max_iter=max_iter)
        self._is_fitted = False

    def fit(self, probs, y):
        p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=int)

        if len(np.unique(y)) < 2:
            self._is_fitted = False
            return self

        X = np.column_stack([np.log(p), np.log(1 - p)])
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, probs):
        p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        if not self._is_fitted:
            return p

        X = np.column_stack([np.log(p), np.log(1 - p)])
        return self._model.predict_proba(X)[:, 1]


class LogisticStacker:
    """Simple meta-model for combining Tier A/Tier B/market probabilities."""

    def __init__(self, c=1.0, max_iter=1000):
        self._model = LogisticRegression(C=c, max_iter=max_iter)
        self._is_fitted = False

    @staticmethod
    def _to_meta_features(tier_a, tier_b, market, odds_missing):
        eps = 1e-6
        pa = np.clip(np.asarray(tier_a, dtype=float), eps, 1 - eps)
        pb = np.clip(np.asarray(tier_b, dtype=float), eps, 1 - eps)
        pm = np.clip(np.asarray(market, dtype=float), eps, 1 - eps)
        miss = np.asarray(odds_missing, dtype=float)

        return np.column_stack(
            [
                np.log(pa / (1 - pa)),
                np.log(pb / (1 - pb)),
                np.log(pm / (1 - pm)),
                miss,
            ]
        )

    def fit(self, tier_a, tier_b, market, odds_missing, y):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self._is_fitted = False
            return self

        X = self._to_meta_features(tier_a, tier_b, market, odds_missing)
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, tier_a, tier_b, market, odds_missing):
        X = self._to_meta_features(tier_a, tier_b, market, odds_missing)
        if not self._is_fitted:
            return np.clip(np.asarray(tier_b, dtype=float), 1e-6, 1 - 1e-6)
        return self._model.predict_proba(X)[:, 1]


def save_artifact(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)
