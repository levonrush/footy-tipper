import dill as pickle
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

PROB_EPS = 1e-4
MAX_LOGIT = 8.0


def _clip_probs(values):
    clipped = np.clip(np.asarray(values, dtype=float), PROB_EPS, 1 - PROB_EPS)
    return np.nan_to_num(clipped, nan=0.5, posinf=1 - PROB_EPS, neginf=PROB_EPS)


def _safe_logit(values):
    probs = _clip_probs(values)
    logits = np.log(probs / (1.0 - probs))
    return np.clip(np.nan_to_num(logits, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT), -MAX_LOGIT, MAX_LOGIT)


class IdentityCalibrator:
    def fit(self, probs, y):
        return self

    def predict(self, probs):
        return _clip_probs(probs)


class BetaCalibrator:
    """Beta calibration: sigmoid(a * log(p) + b * log(1-p) + c)."""

    def __init__(self, c=0.25, max_iter=1000):
        self._model = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs")
        self._is_fitted = False

    def fit(self, probs, y):
        p = _clip_probs(probs)
        y = np.asarray(y, dtype=int)

        if len(np.unique(y)) < 2:
            self._is_fitted = False
            return self

        X = np.column_stack(
            [
                np.clip(np.log(p), -MAX_LOGIT, 0.0),
                np.clip(np.log(1 - p), -MAX_LOGIT, 0.0),
            ]
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=-MAX_LOGIT)

        if not np.isfinite(X).all():
            self._is_fitted = False
            return self

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self._model.fit(X, y)
            self._is_fitted = True
        except Exception:
            self._is_fitted = False
        return self

    def predict(self, probs):
        p = _clip_probs(probs)
        if not self._is_fitted:
            return p

        X = np.column_stack(
            [
                np.clip(np.log(p), -MAX_LOGIT, 0.0),
                np.clip(np.log(1 - p), -MAX_LOGIT, 0.0),
            ]
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=-MAX_LOGIT)
        return self._model.predict_proba(X)[:, 1]


class LogisticStacker:
    """Meta-model for combining Tier A/Tier B/market probabilities.

    Uses LogisticRegressionCV to cross-validate the L2 regularisation strength C,
    preventing the market term from dominating unchecked.
    """

    _DEFAULT_CS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    def __init__(self, cs=None, max_iter=2000):
        cs = cs if cs is not None else self._DEFAULT_CS
        self._model = LogisticRegressionCV(
            Cs=cs,
            cv=5,
            max_iter=max_iter,
            solver="lbfgs",
            scoring="neg_log_loss",
        )
        self._is_fitted = False

    @staticmethod
    def _to_meta_features(tier_a, tier_b, market, odds_missing, tier_c=None):
        pa = _clip_probs(tier_a)
        pb = _clip_probs(tier_b)
        pm = _clip_probs(market)
        miss = np.nan_to_num(np.asarray(odds_missing, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)

        la = _safe_logit(pa)
        lb = _safe_logit(pb)
        lm = _safe_logit(pm)

        X = np.column_stack(
            [
                la,         # Tier A log-odds
                lb,         # Tier B log-odds
                lm,         # Market log-odds
                miss,       # Odds missing flag
                la - lm,    # Tier A disagreement with market (value-over-market signal)
                lb - lm,    # Tier B disagreement with market (value-over-market signal)
            ]
        )
        if tier_c is not None:
            lc = _safe_logit(_clip_probs(tier_c)).reshape(-1, 1)
            X = np.column_stack([X, lc])  # Tier C: binary classifier log-odds

        return np.nan_to_num(X, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT)

    def fit(self, tier_a, tier_b, market, odds_missing, y, tier_c=None):
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self._is_fitted = False
            return self

        X = self._to_meta_features(tier_a, tier_b, market, odds_missing, tier_c=tier_c)
        if not np.isfinite(X).all():
            self._is_fitted = False
            return self

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self._model.fit(X, y)
            self._is_fitted = True
        except Exception:
            self._is_fitted = False
        return self

    def predict(self, tier_a, tier_b, market, odds_missing, tier_c=None):
        X = self._to_meta_features(tier_a, tier_b, market, odds_missing, tier_c=tier_c)
        if not self._is_fitted:
            return _clip_probs(tier_b)
        return self._model.predict_proba(X)[:, 1]


def save_artifact(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)
