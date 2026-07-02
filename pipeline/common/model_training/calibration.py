import dill as pickle
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold

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
        self._cs = cs if cs is not None else self._DEFAULT_CS
        self._max_iter = max_iter
        self._model = self._build_model(cv=5)
        self._is_fitted = False

    def _build_model(self, cv):
        return LogisticRegressionCV(
            Cs=self._cs,
            cv=cv,
            max_iter=self._max_iter,
            solver="lbfgs",
            scoring="neg_log_loss",
        )

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

    def fit(self, tier_a, tier_b, market, odds_missing, y, tier_c=None, groups=None):
        """Fit the meta-model.

        When `groups` (e.g. competition_year per row) is provided, the internal
        regularisation search uses season-grouped CV instead of random k-fold,
        keeping the meta-layer consistent with the time-aware CV used elsewhere.
        """
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self._is_fitted = False
            return self

        X = self._to_meta_features(tier_a, tier_b, market, odds_missing, tier_c=tier_c)
        if not np.isfinite(X).all():
            self._is_fitted = False
            return self

        cv = 5
        if groups is not None:
            groups = np.asarray(groups)
            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                splitter = GroupKFold(n_splits=min(5, n_groups))
                cv = list(splitter.split(X, y, groups))
        self._model = self._build_model(cv=cv)

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


def loso_stacker_predictions(tier_a, tier_b, market, odds_missing, y, groups, tier_c=None):
    """Leave-one-season-out stacker predictions for calibrator training.

    For each season group, fit a fresh LogisticStacker on the other seasons and
    predict the held-out one, so every returned prediction is out-of-sample for
    the stacker. A calibrator fit on these is not flattered by stacker overfit.
    Returns an array with NaN for rows that could not be predicted, or None when
    fewer than 3 season groups exist (callers fall back to in-sample fitting).
    """
    tier_a = np.asarray(tier_a, dtype=float)
    tier_b = np.asarray(tier_b, dtype=float)
    market = np.asarray(market, dtype=float)
    odds_missing = np.asarray(odds_missing, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=float)
    tier_c = None if tier_c is None else np.asarray(tier_c, dtype=float)

    finite_groups = np.isfinite(groups)
    unique_groups = np.unique(groups[finite_groups])
    if len(unique_groups) < 3:
        return None

    preds = np.full(len(tier_a), np.nan)
    for group in unique_groups:
        hold = groups == group
        train = finite_groups & ~hold
        if train.sum() < 50 or len(np.unique(y[train])) < 2:
            continue
        stacker = LogisticStacker()
        stacker.fit(
            tier_a=tier_a[train],
            tier_b=tier_b[train],
            market=market[train],
            odds_missing=odds_missing[train],
            tier_c=None if tier_c is None else tier_c[train],
            y=y[train],
            groups=groups[train],
        )
        if not stacker._is_fitted:
            continue
        preds[hold] = stacker.predict(
            tier_a[hold],
            tier_b[hold],
            market[hold],
            odds_missing[hold],
            tier_c=None if tier_c is None else tier_c[hold],
        )

    if not np.isfinite(preds).any():
        return None
    return preds


def save_artifact(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)
