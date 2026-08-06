import dill as pickle
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold

from pipeline.common.model_training import comp_sim
from pipeline.common.odds.validity import valid_decimal_odds

PROB_EPS = 1e-4
MAX_LOGIT = 8.0
PROBABILITY_STACK_VERSION = 3

# Sanity clip for the model-vs-line spread disagreement (points).
MAX_SPREAD_DISAGREEMENT = 30.0
# Scale divisor bringing the disagreement (points) to roughly unit variance so
# the shared L2 penalty treats it comparably to the logit-scale features.
SPREAD_DISAGREEMENT_SCALE = 10.0

# Typical NRL total for centring the market total line.
TOTAL_LINE_CENTER = 40.0
TOTAL_LINE_SCALE = 10.0

LINE_MARKET_FEATURE_NAMES = [
    "line_cover_logit",
    "line_overround_centered",
    "spread_disagreement",
    "line_missing",
    "h2h_move_logit",
    "line_move_points",
    "total_line_centered",
    "totals_missing",
]

# Bumped when the extra-feature layout changes; pinned into the manifest so
# inference can detect a stacker trained on a different layout.
MARKET_EXTRA_VERSION = 2


def build_line_market_features(df, model_margin):
    """Market meta-features for the stacker's extra channel.

    The handicap market is the bookmaker's own margin model; these features
    let the stacker weigh it and the model's disagreement with it, plus odds
    movement (open vs latest, from odds_history via R) and the totals market
    line. All values fail soft to zeros (with missing flags set) when columns
    are absent, so offseason frames and old databases keep working.
    """
    n = len(df)
    model_margin = np.asarray(model_margin, dtype=float)

    def _col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    cover = _col("home_line_cover_prob_shin")
    cover = cover.fillna(_col("home_line_cover_prob_power"))
    cover = cover.fillna(_col("home_line_cover_prob_basic"))

    overround = _col("line_overround_basic")
    # Market's expected home margin is the negated handicap.
    market_spread = -_col("implied_spread_home")

    missing = (
        cover.isna() | market_spread.isna() | ~np.isfinite(model_margin)
    ).to_numpy(dtype=float)

    disagreement = np.clip(
        np.nan_to_num(model_margin, nan=0.0)
        - market_spread.fillna(0.0).to_numpy(dtype=float),
        -MAX_SPREAD_DISAGREEMENT,
        MAX_SPREAD_DISAGREEMENT,
    )
    disagreement = np.where(missing > 0, 0.0, disagreement)

    h2h_move = _col("h2h_move_logit")
    line_move = _col("line_move_points")

    total_line = _col("market_total_line")
    if total_line.isna().all():
        total_line = _col("total_line")
    totals_missing = total_line.isna().to_numpy(dtype=float)
    overround_centered = np.nan_to_num(
        overround.to_numpy(dtype=float) - 1.0,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X = np.column_stack(
        [
            np.where(cover.isna().to_numpy(), 0.0, _safe_logit(cover.fillna(0.5))),
            np.clip(overround_centered, -1.0, 1.0) * 10.0,
            disagreement / SPREAD_DISAGREEMENT_SCALE,
            missing,
            np.clip(np.nan_to_num(h2h_move.to_numpy(dtype=float), nan=0.0), -2.0, 2.0),
            np.clip(
                np.nan_to_num(line_move.to_numpy(dtype=float), nan=0.0), -12.0, 12.0
            )
            / SPREAD_DISAGREEMENT_SCALE,
            np.nan_to_num(
                (total_line.to_numpy(dtype=float) - TOTAL_LINE_CENTER)
                / TOTAL_LINE_SCALE,
                nan=0.0,
            ),
            totals_missing,
        ]
    )
    assert X.shape == (n, len(LINE_MARKET_FEATURE_NAMES))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _clip_probs(values):
    clipped = np.clip(np.asarray(values, dtype=float), PROB_EPS, 1 - PROB_EPS)
    return np.nan_to_num(clipped, nan=0.5, posinf=1 - PROB_EPS, neginf=PROB_EPS)


def _safe_logit(values):
    probs = _clip_probs(values)
    logits = np.log(probs / (1.0 - probs))
    return np.clip(
        np.nan_to_num(logits, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT),
        -MAX_LOGIT,
        MAX_LOGIT,
    )


def valid_h2h_mask(df):
    """Return rows backed by a complete, valid two-sided H2H market.

    The raw decimal prices are the source of truth. Derived market
    probabilities can contain historical 0.5 fallbacks, so they must never be
    used to infer market availability.
    """
    n = len(df)
    if (
        "team_head_to_head_odds_home" not in df.columns
        or "team_head_to_head_odds_away" not in df.columns
    ):
        return np.zeros(n, dtype=bool)

    home = pd.to_numeric(df["team_head_to_head_odds_home"], errors="coerce").to_numpy(
        dtype=float
    )
    away = pd.to_numeric(df["team_head_to_head_odds_away"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.fromiter(
        (
            valid_decimal_odds(home_price) and valid_decimal_odds(away_price)
            for home_price, away_price in zip(home, away)
        ),
        dtype=bool,
        count=n,
    )

    if "odds_missing" in df.columns:
        missing = (
            pd.to_numeric(df["odds_missing"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
        valid &= missing < 0.5
    return valid


def fresh_game_mask(df, fresh_game_ids):
    """Return rows whose game id has a fresh live-odds snapshot."""
    if "game_id" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    fresh = {int(game_id) for game_id in fresh_game_ids}
    game_ids = pd.to_numeric(df["game_id"], errors="coerce").to_numpy(dtype=float)
    is_fresh = np.fromiter(
        (np.isfinite(game_id) and int(game_id) in fresh for game_id in game_ids),
        dtype=bool,
        count=len(df),
    )
    return is_fresh


def valid_fresh_h2h_mask(df, fresh_game_ids):
    """Require both valid raw prices and a fresh live snapshot for each game."""
    return valid_h2h_mask(df) & fresh_game_mask(df, fresh_game_ids)


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


class TemperatureCalibrator:
    """Symmetric calibration via a strictly positive temperature.

    ``sigmoid(logit(p) / temperature)`` has no intercept, fixes 0.5 at 0.5,
    preserves the predicted side, and cannot reverse probability ordering.
    """

    def __init__(self, min_temperature=0.1, max_temperature=10.0):
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.temperature_ = 1.0
        self._is_fitted = False

    def fit(self, probs, y):
        p = _clip_probs(probs)
        y = np.asarray(y, dtype=int)
        finite = np.isfinite(p) & np.isfinite(y)
        p = p[finite]
        y = y[finite]
        if len(p) < 2 or len(np.unique(y)) < 2:
            self.temperature_ = 1.0
            self._is_fitted = False
            return self

        logits = _safe_logit(p)
        lower = float(np.log(self.min_temperature))
        upper = float(np.log(self.max_temperature))

        def objective(log_temperature):
            temperature = float(np.exp(log_temperature))
            z = logits / temperature
            return float(np.mean(np.logaddexp(0.0, z) - y * z))

        try:
            result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
            if result.success and np.isfinite(result.x):
                self.temperature_ = float(np.exp(result.x))
                self._is_fitted = True
            else:
                self.temperature_ = 1.0
                self._is_fitted = False
        except Exception:
            self.temperature_ = 1.0
            self._is_fitted = False
        return self

    def predict(self, probs):
        p = _clip_probs(probs)
        temperature = self.temperature_ if self._is_fitted else 1.0
        z = _safe_logit(p) / max(float(temperature), PROB_EPS)
        return _clip_probs(1.0 / (1.0 + np.exp(-z)))


class SimplexLogitPool:
    """Non-negative, no-intercept logit pool over model experts.

    Weights are constrained to the probability simplex. This makes the pool
    monotone in every expert, symmetric around 0.5, and immune to the
    collinear disagreement terms used by the legacy logistic stacker.
    """

    def __init__(self, include_market=True, max_iter=2000):
        self.include_market = bool(include_market)
        self.max_iter = int(max_iter)
        self.expert_names_ = ()
        self.weights_ = np.array([], dtype=float)
        self._is_fitted = False

    def _expert_matrix(self, tier_a, tier_b, tier_c=None, market=None):
        columns = [_safe_logit(tier_a), _safe_logit(tier_b)]
        names = ["tier_a", "tier_b"]
        if tier_c is not None:
            columns.append(_safe_logit(tier_c))
            names.append("tier_c")
        if self.include_market:
            if market is None:
                raise ValueError(
                    "market probabilities are required by this simplex pool"
                )
            columns.append(_safe_logit(market))
            names.append("market")
        X = np.column_stack(columns)
        return np.nan_to_num(X, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT), tuple(
            names
        )

    def fit(self, tier_a, tier_b, y, tier_c=None, market=None):
        y = np.asarray(y, dtype=int)
        X, names = self._expert_matrix(tier_a, tier_b, tier_c=tier_c, market=market)
        self.expert_names_ = names
        n_experts = X.shape[1]
        if len(y) != len(X) or len(y) == 0 or not np.isfinite(X).all():
            self.weights_ = np.full(n_experts, 1.0 / n_experts)
            self._is_fitted = False
            return self

        initial = np.full(n_experts, 1.0 / n_experts)

        def objective(weights):
            weights = np.asarray(weights, dtype=float)
            if (
                not np.isfinite(weights).all()
                or (weights < -1e-6).any()
                or (weights > 1.0 + 1e-6).any()
            ):
                return 1e6
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                logits = X @ weights
            if not np.isfinite(logits).all():
                return 1e6
            return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

        try:
            result = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_experts,
                constraints={
                    "type": "eq",
                    "fun": lambda weights: np.sum(weights) - 1.0,
                },
                options={"maxiter": self.max_iter, "ftol": 1e-12},
            )
            if result.success and np.isfinite(result.x).all():
                weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
                total = float(weights.sum())
                if total <= 0:
                    raise ValueError("simplex optimizer returned zero total weight")
                self.weights_ = weights / total
                self._is_fitted = True
                return self
        except Exception:
            pass

        # Deterministic safe fallback: choose the best individual expert.
        losses = [
            float(np.mean(np.logaddexp(0.0, X[:, idx]) - y * X[:, idx]))
            for idx in range(n_experts)
        ]
        self.weights_ = np.zeros(n_experts, dtype=float)
        self.weights_[int(np.argmin(losses))] = 1.0
        self._is_fitted = True
        return self

    def predict(self, tier_a, tier_b, tier_c=None, market=None):
        X, names = self._expert_matrix(tier_a, tier_b, tier_c=tier_c, market=market)
        if not self._is_fitted or tuple(names) != tuple(self.expert_names_):
            return _clip_probs(tier_b)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = X @ np.asarray(self.weights_, dtype=float)
        logits = np.nan_to_num(logits, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT)
        return _clip_probs(1.0 / (1.0 + np.exp(-logits)))

    @property
    def weight_map(self):
        return {
            name: float(weight)
            for name, weight in zip(
                self.expert_names_, np.asarray(self.weights_, dtype=float)
            )
        }

    def select_expert(self, expert_name):
        """Set a deterministic one-hot simplex for a validated expert."""
        if expert_name not in self.expert_names_:
            raise ValueError(f"unknown simplex expert: {expert_name}")
        self.weights_ = np.zeros(len(self.expert_names_), dtype=float)
        self.weights_[self.expert_names_.index(expert_name)] = 1.0
        self._is_fitted = True
        return self

    def blend_toward_expert(self, expert_name, learned_weights, shrinkage):
        """Move the learned weights a fraction of the way toward one expert.

        `shrinkage=1.0` keeps the learned pool, `shrinkage=0.0` reproduces
        `select_expert` exactly, and the values between are the intermediate
        rungs the selection gate chooses from. Rejecting a learned mixture is a
        reason to trust it less, not a reason to throw all of it away.

        Every point on the path is still a simplex, so the deployed artifact
        keeps the same nonnegative sum-to-one contract the release validator
        enforces, and no lambda is stored on the object: the manifest records
        it, the pickle stays a plain weight vector.
        """
        if expert_name not in self.expert_names_:
            raise ValueError(f"unknown simplex expert: {expert_name}")
        shrinkage = float(shrinkage)
        if not np.isfinite(shrinkage) or shrinkage < 0.0 or shrinkage > 1.0:
            raise ValueError(f"shrinkage must lie in [0, 1]: {shrinkage}")

        target = np.zeros(len(self.expert_names_), dtype=float)
        target[self.expert_names_.index(expert_name)] = 1.0
        if shrinkage == 0.0:
            # Bitwise identical to select_expert, so the status quo stays exact.
            self.weights_ = target
            self._is_fitted = True
            return self

        learned = np.asarray(learned_weights, dtype=float)
        if learned.shape != target.shape:
            raise ValueError("learned weights do not match the pool's experts")
        weights = np.clip(shrinkage * learned + (1.0 - shrinkage) * target, 0.0, 1.0)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("shrunk weights did not sum to a positive total")
        self.weights_ = weights / total
        self._is_fitted = True
        return self


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
    def _to_meta_features(
        tier_a, tier_b, market, odds_missing, tier_c=None, extra=None
    ):
        pa = _clip_probs(tier_a)
        pb = _clip_probs(tier_b)
        pm = _clip_probs(market)
        miss = np.nan_to_num(
            np.asarray(odds_missing, dtype=float), nan=0.0, posinf=1.0, neginf=0.0
        )

        la = _safe_logit(pa)
        lb = _safe_logit(pb)
        lm = _safe_logit(pm)

        X = np.column_stack(
            [
                la,  # Tier A log-odds
                lb,  # Tier B log-odds
                lm,  # Market log-odds
                miss,  # Odds missing flag
                la - lm,  # Tier A disagreement with market (value-over-market signal)
                lb - lm,  # Tier B disagreement with market (value-over-market signal)
            ]
        )
        if tier_c is not None:
            lc = _safe_logit(_clip_probs(tier_c)).reshape(-1, 1)
            X = np.column_stack([X, lc])  # Tier C: binary classifier log-odds
        if extra is not None:
            X = np.column_stack([X, np.asarray(extra, dtype=float)])

        return np.nan_to_num(X, nan=0.0, posinf=MAX_LOGIT, neginf=-MAX_LOGIT)

    def fit(
        self,
        tier_a,
        tier_b,
        market,
        odds_missing,
        y,
        tier_c=None,
        groups=None,
        extra=None,
    ):
        """Fit the meta-model.

        When `groups` (e.g. competition_year per row) is provided, the internal
        regularisation search uses season-grouped CV instead of random k-fold,
        keeping the meta-layer consistent with the time-aware CV used elsewhere.
        `extra` appends additional meta-feature columns (e.g. line-market
        features); the fitted feature layout is versioned so old pickles keep
        their original inputs.
        """
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self._is_fitted = False
            return self

        self._feature_version = 2 if extra is not None else 1
        self._n_extra = 0 if extra is None else int(np.asarray(extra).shape[1])
        X = self._to_meta_features(
            tier_a, tier_b, market, odds_missing, tier_c=tier_c, extra=extra
        )
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

    def predict(self, tier_a, tier_b, market, odds_missing, tier_c=None, extra=None):
        # Pickles from before the extra-feature layout have no version attr.
        version = getattr(self, "_feature_version", 1)
        if version < 2:
            extra = None
        elif extra is None:
            # Fitted with extra features but none supplied: neutral zeros.
            extra = np.zeros(
                (len(np.asarray(tier_a)), int(getattr(self, "_n_extra", 0)))
            )
        X = self._to_meta_features(
            tier_a, tier_b, market, odds_missing, tier_c=tier_c, extra=extra
        )
        if not self._is_fitted:
            return _clip_probs(tier_b)
        return self._model.predict_proba(X)[:, 1]


def loso_stacker_predictions(
    tier_a, tier_b, market, odds_missing, y, groups, tier_c=None, extra=None
):
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
    extra = None if extra is None else np.asarray(extra, dtype=float)

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
            extra=None if extra is None else extra[train],
        )
        if not stacker._is_fitted:
            continue
        preds[hold] = stacker.predict(
            tier_a[hold],
            tier_b[hold],
            market[hold],
            odds_missing[hold],
            tier_c=None if tier_c is None else tier_c[hold],
            extra=None if extra is None else extra[hold],
        )

    if not np.isfinite(preds).any():
        return None
    return preds


def _normalise_shrinkage_grid(shrinkage_grid):
    """Sorted, de-duplicated grid with 0.0 always present as the safe rung."""
    values = {0.0}
    for value in shrinkage_grid:
        value = float(value)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"shrinkage grid values must lie in [0, 1]: {value}")
        values.add(value)
    return tuple(sorted(values))


def loso_simplex_pool_predictions(
    tier_a,
    tier_b,
    y,
    groups,
    *,
    tier_c=None,
    market=None,
    include_market=True,
    shrinkage_grid=None,
    fallback_expert=None,
):
    """Leave-one-season-out predictions from a simplex logit pool.

    With `shrinkage_grid`, returns `{shrinkage: predictions}` covering the whole
    path from the one-hot fallback to the learned pool. The pool is fitted once
    per season and every rung reuses those weights, so a five-point grid costs
    one extra matrix product per fold rather than five extra optimiser runs.
    """
    tier_a = np.asarray(tier_a, dtype=float)
    tier_b = np.asarray(tier_b, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=float)
    tier_c = None if tier_c is None else np.asarray(tier_c, dtype=float)
    market = None if market is None else np.asarray(market, dtype=float)

    path_mode = shrinkage_grid is not None
    if path_mode:
        if fallback_expert is None:
            raise ValueError("a shrinkage grid requires a fallback expert")
        grid = _normalise_shrinkage_grid(shrinkage_grid)

    finite_groups = np.isfinite(groups)
    unique_groups = np.unique(groups[finite_groups])
    if len(unique_groups) < 3:
        return None

    if path_mode:
        preds = {value: np.full(len(tier_a), np.nan) for value in grid}
    else:
        preds = np.full(len(tier_a), np.nan)

    for group in unique_groups:
        hold = groups == group
        train = finite_groups & ~hold
        if train.sum() < 50 or hold.sum() == 0:
            continue
        pool = SimplexLogitPool(include_market=include_market)
        pool.fit(
            tier_a=tier_a[train],
            tier_b=tier_b[train],
            tier_c=None if tier_c is None else tier_c[train],
            market=None if market is None else market[train],
            y=y[train],
        )
        if not pool._is_fitted:
            continue
        hold_args = (
            tier_a[hold],
            tier_b[hold],
        )
        hold_kwargs = {
            "tier_c": None if tier_c is None else tier_c[hold],
            "market": None if market is None else market[hold],
        }
        if not path_mode:
            preds[hold] = pool.predict(*hold_args, **hold_kwargs)
            continue

        learned_weights = np.asarray(pool.weights_, dtype=float).copy()
        for value in grid:
            pool.blend_toward_expert(fallback_expert, learned_weights, value)
            preds[value][hold] = pool.predict(*hold_args, **hold_kwargs)

    if path_mode:
        if not any(np.isfinite(array).any() for array in preds.values()):
            return None
        return preds
    if not np.isfinite(preds).any():
        return None
    return preds


def loso_temperature_predictions(probs, y, groups):
    """Cross-fit temperature calibration by season for honest selection."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=float)
    finite = np.isfinite(probs) & np.isfinite(groups)
    unique_groups = np.unique(groups[finite])
    if len(unique_groups) < 3:
        return None

    preds = np.full(len(probs), np.nan)
    for group in unique_groups:
        hold = finite & (groups == group)
        train = finite & (groups != group)
        if train.sum() < 50 or hold.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        calibrator = TemperatureCalibrator().fit(probs[train], y[train])
        preds[hold] = calibrator.predict(probs[hold])
    if not np.isfinite(preds).any():
        return None
    return preds


def nested_loso_simplex_predictions(
    tier_a,
    tier_b,
    y,
    groups,
    *,
    tier_c=None,
    market=None,
    include_market=True,
    shrinkage_grid=None,
    fallback_expert=None,
):
    """Fully nested season-out pool + temperature predictions.

    For each outer held season, the pool is fitted only on prior input rows and
    its temperature is learned from inner LOSO predictions within that outer
    training set. Held-season labels cannot influence either layer.

    With `shrinkage_grid`, returns `{shrinkage: predictions}`. Each rung gets its
    own temperature, fitted on that rung's inner predictions, because a shrunk
    pool is a different forecaster from the learned one and borrowing the
    learned pool's temperature would misstate every intermediate rung.

    `fallback_expert` is supplied by the caller rather than re-chosen per fold.
    Picking it inside the fold would need the competition objective, and with it
    the market prices and outcomes, threaded through this helper. The residual
    optimism is one choice among three experts, made on aggregate performance;
    the per-fold weights, temperatures and predictions all remain honest.
    """
    tier_a = np.asarray(tier_a, dtype=float)
    tier_b = np.asarray(tier_b, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=float)
    tier_c = None if tier_c is None else np.asarray(tier_c, dtype=float)
    market = None if market is None else np.asarray(market, dtype=float)

    path_mode = shrinkage_grid is not None
    if path_mode:
        if fallback_expert is None:
            raise ValueError("a shrinkage grid requires a fallback expert")
        grid = _normalise_shrinkage_grid(shrinkage_grid)

    finite_groups = np.isfinite(groups)
    unique_groups = np.unique(groups[finite_groups])
    if len(unique_groups) < 4:
        return None

    if path_mode:
        predictions = {value: np.full(len(tier_a), np.nan) for value in grid}
    else:
        predictions = np.full(len(tier_a), np.nan)

    for outer_group in unique_groups:
        hold = finite_groups & (groups == outer_group)
        train = finite_groups & (groups != outer_group)
        if train.sum() < 50 or hold.sum() == 0:
            continue

        inner_predictions = loso_simplex_pool_predictions(
            tier_a[train],
            tier_b[train],
            y[train],
            groups[train],
            tier_c=None if tier_c is None else tier_c[train],
            market=None if market is None else market[train],
            include_market=include_market,
            shrinkage_grid=shrinkage_grid,
            fallback_expert=fallback_expert,
        )
        if inner_predictions is None:
            continue

        pool = SimplexLogitPool(include_market=include_market).fit(
            tier_a[train],
            tier_b[train],
            y[train],
            tier_c=None if tier_c is None else tier_c[train],
            market=None if market is None else market[train],
        )
        if not pool._is_fitted:
            continue
        hold_args = (tier_a[hold], tier_b[hold])
        hold_kwargs = {
            "tier_c": None if tier_c is None else tier_c[hold],
            "market": None if market is None else market[hold],
        }

        if not path_mode:
            inner_rows = np.isfinite(inner_predictions)
            if inner_rows.sum() < 50:
                continue
            calibrator = TemperatureCalibrator().fit(
                inner_predictions[inner_rows], y[train][inner_rows]
            )
            predictions[hold] = calibrator.predict(pool.predict(*hold_args, **hold_kwargs))
            continue

        learned_weights = np.asarray(pool.weights_, dtype=float).copy()
        for value in grid:
            inner_rung = inner_predictions[value]
            inner_rows = np.isfinite(inner_rung)
            if inner_rows.sum() < 50:
                continue
            calibrator = TemperatureCalibrator().fit(
                inner_rung[inner_rows], y[train][inner_rows]
            )
            pool.blend_toward_expert(fallback_expert, learned_weights, value)
            predictions[value][hold] = calibrator.predict(
                pool.predict(*hold_args, **hold_kwargs)
            )

    if path_mode:
        if not any(np.isfinite(array).any() for array in predictions.values()):
            return None
        return predictions
    if not np.isfinite(predictions).any():
        return None
    return predictions


def comp_placement_metrics(probabilities, market, y, groups, rows=None):
    """Score a forecaster the way the competition scores it: P(finish first).

    Tips are `probabilities > 0.5`; the rival field tips the market favourite
    with per-rival flip noise, exactly as `comp_sim` models it.

    Two properties of this objective are deliberate and load-bearing.

    First, *within* one realized season, maximising `p_first` is exactly
    equivalent to maximising tips correct: rival scores do not depend on our
    tips, so `p_first` is monotone in our score. Deviating from the field for
    its own sake buys nothing here. The genuine value of differentiating when
    behind depends on the live points gap and belongs in `comp_strategy`.

    Second, *across* seasons it is not the same as pooled accuracy. `p_first`
    is S-shaped in the season score, so clearing the field in a strong season
    is worth far more than the same average accuracy spread evenly, and
    finishing fortieth rather than twentieth costs nothing extra. That is why
    seasons are scored separately and then averaged, never pooled: the mean of
    per-season `p_first` is the quantity that tracks winning a comp.

    Games without a usable market price are dropped rather than handed a
    fabricated favourite, and the count is reported.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    market = np.asarray(market, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=float)

    usable = np.isfinite(probabilities) & np.isfinite(groups)
    if rows is not None:
        usable &= np.asarray(rows, dtype=bool)
    excluded_no_market = int((usable & ~np.isfinite(market)).sum())
    usable &= np.isfinite(market)

    seasons = []
    for group in np.unique(groups[usable]):
        season_rows = usable & (groups == group)
        if not season_rows.any():
            continue
        placement = comp_sim.simulate_comp_placement(
            probabilities[season_rows], market[season_rows], y[season_rows]
        )
        if placement is None:
            continue
        seasons.append(
            {
                "group": float(group),
                "games": int(season_rows.sum()),
                "tips_correct": int(placement["user_score"]),
                "accuracy": float(placement["user_score"]) / float(season_rows.sum()),
                "p_first": float(placement["p_first"]),
                "expected_rank": float(placement["expected_rank"]),
            }
        )

    if not seasons:
        return None
    return {
        "mean_p_first": float(np.mean([season["p_first"] for season in seasons])),
        "mean_expected_rank": float(np.mean([season["expected_rank"] for season in seasons])),
        "tips_correct": int(sum(season["tips_correct"] for season in seasons)),
        "games": int(sum(season["games"] for season in seasons)),
        "seasons": seasons,
        "excluded_no_market": excluded_no_market,
    }


DEFAULT_SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

# Experts a rejected pool may collapse onto. The raw market stays a comparator
# but is never deployed as the model on its own.
DEPLOYABLE_EXPERT_NAMES = ("tier_a", "tier_b", "tier_c")


def strongest_deployable_expert(
    expert_probabilities,
    y,
    groups=None,
    *,
    market=None,
    objective="log_loss",
    rows=None,
):
    """Name the expert a rejected pool should shrink toward.

    Chosen up front because the shrinkage path has to be built around a target
    before the nested predictions exist. Tier A/B/C probabilities are already
    out-of-fold, so ranking them directly is honest.

    Ranked on log loss, deliberately, even though rungs of the path are ranked
    on P(finish first). This target is the safety default the whole path falls
    back to, and P(first) is far too noisy to choose it: it is close to bimodal
    per season, so a couple of lucky seasons can promote a clearly worse
    expert. A walk-forward fold did exactly that, picking Tier A on a 0.024
    P(first) edge while Tier A was 0.15 worse on log loss, and the resulting
    model regressed against the deployed one. Deciding *how far to move* from
    the default is a different question from deciding *what the default is*.
    """
    experts = {
        name: np.asarray(probabilities, dtype=float)
        for name, probabilities in expert_probabilities.items()
        if name in DEPLOYABLE_EXPERT_NAMES
    }
    if not experts:
        raise ValueError("no deployable expert is available")

    y = np.asarray(y, dtype=int)
    usable = np.ones(len(y), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    for probabilities in experts.values():
        usable &= np.isfinite(probabilities)

    market = None if market is None else np.asarray(market, dtype=float)
    use_comp = bool(objective == "p_first" and market is not None and groups is not None)

    def key(name):
        metrics = _pool_selection_metrics(y, experts[name], usable)
        if use_comp:
            placement = comp_placement_metrics(experts[name], market, y, groups, rows=usable)
            if placement is not None:
                return (-placement["mean_p_first"], -metrics["accuracy"], metrics["log_loss"], name)
        return (metrics["log_loss"], -metrics["accuracy"], metrics["brier"], name)

    return min(experts, key=key)


def _recent_mean_p_first(placement, recent_groups):
    """Mean P(finish first) over the most recent seasons only.

    `comp_placement_metrics` returns seasons in ascending group order, so the
    tail is the recent window.
    """
    if not placement or not placement.get("seasons"):
        return None
    window = placement["seasons"][-max(1, int(recent_groups)) :]
    if not window:
        return None
    return float(np.mean([season["p_first"] for season in window]))


def _pool_selection_rows(nested, expert_probabilities):
    """Rows where a nested pool and every comparator are directly comparable."""
    if nested is None:
        return None
    rows = np.isfinite(np.asarray(nested, dtype=float))
    for probabilities in expert_probabilities.values():
        rows &= np.isfinite(np.asarray(probabilities, dtype=float))
    return rows


def _pool_selection_metrics(y, probabilities, rows):
    selected_y = np.asarray(y, dtype=int)[rows]
    probabilities = _clip_probs(np.asarray(probabilities, dtype=float)[rows])
    logits = _safe_logit(probabilities)
    loss = float(np.mean(np.logaddexp(0.0, logits) - selected_y * logits))
    return {
        "accuracy": float(((probabilities > 0.5) == selected_y.astype(bool)).mean()),
        "log_loss": loss,
        "brier": float(np.mean((probabilities - selected_y) ** 2)),
    }


def select_market_pool(
    pool,
    nested_pool_predictions,
    y,
    expert_probabilities,
    *,
    groups=None,
    recent_groups=3,
    min_log_loss_improvement=0.005,
    accuracy_tolerance=0.01,
    brier_tolerance=0.005,
    objective="log_loss",
    market_probabilities=None,
):
    """Choose how much of the learned pool to deploy, on competition evidence.

    `nested_pool_predictions` may be a single array (the learned pool, scored
    the legacy way) or `{shrinkage: predictions}` covering the path from the
    one-hot fallback to the learned pool. The path exists because "the learned
    mixture did not clearly beat the best expert" is not the same claim as "the
    best expert alone is the right model", and collapsing straight to a corner
    of the simplex acts as though it were.

    Candidates are ranked by mean per-season P(finish first), which is what a
    tipping competition actually pays out on, with log loss and Brier demoted
    to a guard: a candidate that wins tips on badly calibrated probabilities
    would degrade the joker and competition-strategy layers downstream, both of
    which consume these numbers directly.

    The raw market stays a comparator and never becomes the deployed model on
    its own, but shrinkage means market evidence is throttled rather than
    discarded outright whenever the learned pool falls short.
    """
    y = np.asarray(y, dtype=int)
    if isinstance(nested_pool_predictions, dict):
        nested_path = {
            float(shrinkage): np.asarray(predictions, dtype=float)
            for shrinkage, predictions in nested_pool_predictions.items()
        }
        nested_path = {
            shrinkage: predictions
            for shrinkage, predictions in nested_path.items()
            if np.isfinite(predictions).any()
        }
        nested = nested_path[max(nested_path)] if nested_path else None
    elif nested_pool_predictions is None:
        nested_path, nested = {}, None
    else:
        nested = np.asarray(nested_pool_predictions, dtype=float)
        nested_path = {1.0: nested}
    learned_weights = (
        np.asarray(pool.weights_, dtype=float).copy()
        if getattr(pool, "weights_", None) is not None
        else None
    )
    experts = {
        name: np.asarray(probabilities, dtype=float)
        for name, probabilities in expert_probabilities.items()
    }
    if not experts:
        raise ValueError("at least one expert is required for market selection")
    fallback_names = [
        name for name in ("tier_a", "tier_b", "tier_c") if name in experts
    ]
    if "market" in experts and not fallback_names:
        raise ValueError(
            "market selection requires at least one Tier A/B/C fallback expert"
        )

    if nested is None:
        fallback_name = (
            "tier_b"
            if "tier_b" in experts
            else (fallback_names[0] if fallback_names else next(iter(experts)))
        )
        pool.select_expert(fallback_name)
        return {
            "selected": fallback_name,
            "selection_rows": 0,
            "reason": "nested_loso_unavailable",
            "pool": None,
            "best_expert": None,
            "fallback_applied": True,
            "fallback_expert": {"name": fallback_name},
            "fallback_reason": "nested_loso_unavailable",
        }

    selection_rows = _pool_selection_rows(nested, experts)
    for rung in nested_path.values():
        selection_rows &= np.isfinite(rung)
    if not selection_rows.any():
        fallback_name = (
            "tier_b"
            if "tier_b" in experts
            else (fallback_names[0] if fallback_names else next(iter(experts)))
        )
        pool.select_expert(fallback_name)
        return {
            "selected": fallback_name,
            "selection_rows": 0,
            "reason": "nested_loso_unavailable",
            "pool": None,
            "best_expert": None,
            "fallback_applied": True,
            "fallback_expert": {"name": fallback_name},
            "fallback_reason": "nested_loso_unavailable",
        }

    def metrics(probabilities, rows=selection_rows):
        return _pool_selection_metrics(y, probabilities, rows)

    pool_metrics = metrics(nested)
    expert_metrics = {
        name: metrics(probabilities) for name, probabilities in experts.items()
    }

    groups = None if groups is None else np.asarray(groups, dtype=float)
    market_p = market_probabilities
    if market_p is None:
        market_p = experts.get("market")
    market_p = None if market_p is None else np.asarray(market_p, dtype=float)
    # Competition placement is always reported when a rival field and seasons
    # exist, but it only *selects* when asked for. Mean per-season P(first) is
    # close to bimodal, so a handful of seasons swings it, and walk-forward
    # runs showed the rung it picks flipping between the endpoints across
    # retrains of near-identical data. It is a good scoreboard and a poor
    # objective; log loss is stable and the guards do the rest.
    can_report_comp = bool(market_p is not None and groups is not None)
    use_comp = bool(objective == "p_first" and can_report_comp)

    def comp(probabilities, rows=selection_rows):
        if not can_report_comp:
            return None
        return comp_placement_metrics(probabilities, market_p, y, groups, rows=rows)

    expert_comp = {name: comp(probabilities) for name, probabilities in experts.items()}

    def rank_key(name, metric_source=expert_metrics, comp_source=expert_comp):
        entry = metric_source[name]
        placement = comp_source.get(name)
        if use_comp and placement is not None:
            return (-placement["mean_p_first"], -entry["accuracy"], entry["log_loss"], name)
        return (entry["log_loss"], -entry["accuracy"], entry["brier"], name)

    def stability_rank(name):
        entry = expert_metrics[name]
        return (entry["log_loss"], -entry["accuracy"], entry["brier"], name)

    best_name = min(expert_metrics, key=rank_key)
    best = expert_metrics[best_name]
    # The bar and the fallback are now the same object. Judging the pool against
    # a comparator we are not allowed to deploy meant a narrow miss against the
    # market shipped a third choice that was worse than either.
    #
    # The target is ranked on log loss rather than on P(first): it is the safety
    # default, and P(first) is too noisy per season to choose one. See
    # strongest_deployable_expert for what happens when it is not.
    fallback_candidates = fallback_names or list(expert_metrics)
    fallback_name = min(fallback_candidates, key=stability_rank)
    fallback = expert_metrics[fallback_name]
    best_deployable = expert_metrics[fallback_name]
    best_deployable_comp = expert_comp.get(fallback_name)

    recent_group_rows = []
    if groups is not None:
        available_groups = np.unique(groups[selection_rows & np.isfinite(groups)])
        for group in available_groups[-max(1, int(recent_groups)) :]:
            group_rows = selection_rows & (groups == group)
            if group_rows.any():
                recent_group_rows.append((float(group), group_rows))

    def group_guard(probabilities):
        """Per-season non-regression against the best deployable expert.

        The old rule asked a convex blend to strictly beat its own best
        component in every recent season, which roughly two hundred games at a
        time is close to a coin flip. This asks only that it does not fall
        materially behind, using the same tolerances the release gate applies.
        """
        results = []
        for group, group_rows in recent_group_rows:
            group_pool = metrics(probabilities, group_rows)
            group_experts = {
                name: metrics(values, group_rows) for name, values in experts.items()
            }
            group_deployable = {
                name: group_experts[name]
                for name in (fallback_names or list(group_experts))
            }
            group_best_name = min(
                group_deployable,
                key=lambda name: (
                    group_deployable[name]["log_loss"],
                    -group_deployable[name]["accuracy"],
                    group_deployable[name]["brier"],
                    name,
                ),
            )
            acceptance = acceptance_against_experts(
                group_pool,
                {group_best_name: group_deployable[group_best_name]},
                accuracy_tolerance=accuracy_tolerance,
                loss_tolerance=brier_tolerance,
            )
            results.append(
                {
                    "group": group,
                    "pool": group_pool,
                    "best_expert": {"name": group_best_name, **group_deployable[group_best_name]},
                    "passed": bool(acceptance["passed"]),
                }
            )
        return results

    # A one-hot pool has no fitted parameters, so its held-out prediction on any
    # row is simply that expert's own probability. That makes the zero rung
    # always constructible, including from a legacy single-array call, and
    # guarantees the search always has the status quo available.
    if 0.0 not in nested_path:
        nested_path[0.0] = experts[fallback_name]

    path = []
    for shrinkage in sorted(nested_path):
        rung = nested_path[shrinkage]
        rung_metrics = metrics(rung)
        rung_comp = comp(rung)
        stability = group_guard(rung)
        calibration_ok = acceptance_against_experts(
            rung_metrics,
            {fallback_name: best_deployable},
            accuracy_tolerance=accuracy_tolerance,
            loss_tolerance=brier_tolerance,
        )
        comp_ok = True
        recent_comp_ok = True
        if use_comp and rung_comp is not None and best_deployable_comp is not None:
            comp_ok = rung_comp["mean_p_first"] >= (
                best_deployable_comp["mean_p_first"] - float(min_log_loss_improvement)
            )
            # Averaging P(first) over every season lets a rung ride on ancient
            # history. The market's own tipping accuracy fell about nine points
            # after 2023, so a market-heavy rung can win the all-time mean while
            # losing the seasons that resemble the one being predicted. Require
            # it to hold up over the same recent window the calibration guard
            # already uses.
            recent_rung = _recent_mean_p_first(rung_comp, recent_groups)
            recent_target = _recent_mean_p_first(best_deployable_comp, recent_groups)
            if recent_rung is not None and recent_target is not None:
                recent_comp_ok = recent_rung >= (
                    recent_target - float(min_log_loss_improvement)
                )
        # Zero shrinkage is the deployed status quo and is always available, so
        # the search can never end with nothing to fall back to.
        admissible = shrinkage == 0.0 or bool(
            calibration_ok["passed"]
            and comp_ok
            and recent_comp_ok
            and all(item["passed"] for item in stability)
        )
        path.append(
            {
                "shrinkage": float(shrinkage),
                "pool": rung_metrics,
                "comp": rung_comp,
                "recent_group_stability": stability,
                "calibration_guard": calibration_ok,
                "comp_guard": {
                    "pooled_passed": bool(comp_ok),
                    "recent_passed": bool(recent_comp_ok),
                    "recent_groups": int(recent_groups),
                    "selects": bool(use_comp),
                    "recent_mean_p_first": _recent_mean_p_first(rung_comp, recent_groups),
                    "recent_target_mean_p_first": _recent_mean_p_first(
                        best_deployable_comp, recent_groups
                    ),
                },
                "admissible": admissible,
            }
        )

    admissible_rungs = [row for row in path if row["admissible"]]

    if use_comp:
        def selection_key(row):
            placement = row["comp"]
            primary = -placement["mean_p_first"] if placement else row["pool"]["log_loss"]
            return (primary, row["shrinkage"], row["pool"]["log_loss"])

        chosen = min(admissible_rungs, key=selection_key)
    else:
        # Parsimony on a stable statistic: take the smallest shrinkage whose log
        # loss is within the improvement threshold of the best admissible rung.
        # Moving further than the evidence supports is how the market weight
        # ended up at either extreme in the first place.
        best_loss = min(row["pool"]["log_loss"] for row in admissible_rungs)
        cutoff = best_loss + float(min_log_loss_improvement)
        chosen = min(
            (row for row in admissible_rungs if row["pool"]["log_loss"] <= cutoff),
            key=lambda row: (row["shrinkage"], row["pool"]["log_loss"]),
        )
    selected_shrinkage = float(chosen["shrinkage"])
    keep_learned = selected_shrinkage == 1.0

    if selected_shrinkage == 0.0:
        pool.select_expert(fallback_name)
    elif not keep_learned and learned_weights is not None:
        pool.blend_toward_expert(fallback_name, learned_weights, selected_shrinkage)

    if keep_learned:
        selected = "learned"
    elif selected_shrinkage == 0.0:
        selected = fallback_name
    else:
        selected = "shrunk"

    fallback_reason = None
    if selected_shrinkage < 1.0:
        fallback_reason = (
            "learned_pool_rejected_use_strongest_non_market_expert"
            if "market" in experts
            else "learned_pool_rejected_use_strongest_expert"
        )

    return {
        "selected": selected,
        "selected_shrinkage": selected_shrinkage,
        "shrinkage_grid": [float(value) for value in sorted(nested_path)],
        "shrinkage_target": fallback_name,
        "objective": "p_first" if use_comp else "log_loss",
        "learned_weights": (
            None
            if learned_weights is None
            else {
                name: float(weight)
                for name, weight in zip(getattr(pool, "expert_names_", ()), learned_weights)
            }
        ),
        "path": path,
        "selection_rows": int(selection_rows.sum()),
        "reason": "nested_loso_gate",
        "pool": pool_metrics,
        "pool_comp": comp(nested),
        "best_expert": {
            "name": best_name,
            **best,
        },
        "best_deployable_expert": {
            "name": fallback_name,
            **best_deployable,
        },
        "expert_comp": expert_comp,
        "fallback_applied": bool(not keep_learned),
        "fallback_expert": {
            "name": fallback_name,
            **fallback,
        },
        "fallback_reason": fallback_reason,
        "expert_metrics": expert_metrics,
        # Keeps its long-standing meaning: how the *learned* pool held up season
        # by season. Per-rung stability lives in `path`, so this stays readable
        # as "why the full mixture was or was not kept".
        "recent_group_stability": path[-1]["recent_group_stability"],
        "min_log_loss_improvement": float(min_log_loss_improvement),
        "accuracy_tolerance": float(accuracy_tolerance),
        "brier_tolerance": float(brier_tolerance),
    }


def select_no_market_pool(
    pool,
    nested_pool_predictions,
    y,
    expert_probabilities,
    *,
    groups=None,
    recent_groups=3,
    min_log_loss_improvement=0.005,
    accuracy_tolerance=0.01,
    brier_tolerance=0.005,
    objective="log_loss",
    market_probabilities=None,
):
    """Select the counterfactual A/B/C pool without weakening Tier-B safety.

    A learned mixture is eligible only when its fully nested log loss strictly
    beats Tier B. Eligible mixtures then face the same shrinkage-path selection
    as the market pool, so a mixture that narrowly fails is throttled toward the
    strongest A/B/C expert rather than discarded for it.

    The market never appears as an expert here, but it is still what the rival
    field tips, so `market_probabilities` is accepted purely to score the
    competition objective.
    """
    y = np.asarray(y, dtype=int)
    if isinstance(nested_pool_predictions, dict):
        nested_path = {
            float(shrinkage): np.asarray(predictions, dtype=float)
            for shrinkage, predictions in nested_pool_predictions.items()
            if np.isfinite(np.asarray(predictions, dtype=float)).any()
        }
        nested = nested_path[max(nested_path)] if nested_path else None
    elif nested_pool_predictions is None:
        nested = None
    else:
        nested = np.asarray(nested_pool_predictions, dtype=float)
    experts = {
        name: np.asarray(probabilities, dtype=float)
        for name, probabilities in expert_probabilities.items()
    }
    if "tier_b" not in experts:
        raise ValueError("Tier B is required for no-market eligibility")

    selection_rows = _pool_selection_rows(nested, experts)
    if selection_rows is None or not selection_rows.any():
        pool.select_expert("tier_b")
        return {
            "strategy": "tier_b",
            "selected": "tier_b",
            "eligible": False,
            "selection_rows": 0,
            "reason": "nested_loso_unavailable",
            "eligibility": {
                "criterion": "pool_log_loss < tier_b_log_loss",
                "passed": False,
                "pool_log_loss": None,
                "tier_b_log_loss": None,
            },
            "pool": None,
            "best_expert": None,
            "expert_metrics": {},
            "recent_group_stability": [],
        }

    pool_metrics = _pool_selection_metrics(y, nested, selection_rows)
    expert_metrics = {
        name: _pool_selection_metrics(y, probabilities, selection_rows)
        for name, probabilities in experts.items()
    }
    tier_b_log_loss = expert_metrics["tier_b"]["log_loss"]
    eligible = pool_metrics["log_loss"] < tier_b_log_loss
    eligibility = {
        "criterion": "pool_log_loss < tier_b_log_loss",
        "passed": bool(eligible),
        "pool_log_loss": pool_metrics["log_loss"],
        "tier_b_log_loss": tier_b_log_loss,
    }
    if not eligible:
        pool.select_expert("tier_b")
        return {
            "strategy": "tier_b",
            "selected": "tier_b",
            "eligible": False,
            "selection_rows": int(selection_rows.sum()),
            "reason": "pool_did_not_beat_tier_b",
            "eligibility": eligibility,
            "pool": pool_metrics,
            "best_expert": {
                "name": "tier_b",
                **expert_metrics["tier_b"],
            },
            "expert_metrics": expert_metrics,
            "recent_group_stability": [],
        }

    selection = select_market_pool(
        pool,
        nested_pool_predictions,
        y,
        experts,
        groups=groups,
        recent_groups=recent_groups,
        min_log_loss_improvement=min_log_loss_improvement,
        accuracy_tolerance=accuracy_tolerance,
        brier_tolerance=brier_tolerance,
        objective=objective,
        market_probabilities=market_probabilities,
    )
    return {
        **selection,
        "strategy": "simplex",
        "eligible": True,
        "eligibility": eligibility,
    }


def fit_selected_pool_calibrator(
    selection,
    expert_probabilities,
    y,
    learned_calibrator,
    loso_path_predictions=None,
):
    """Apply the identical post-selection temperature policy everywhere.

    Given the shrinkage path, the deployed temperature is always fitted on the
    held-out predictions of the weights actually being deployed. That is one
    rule for every rung, and it is behaviour-preserving at the status quo: a
    one-hot pool has no fitted parameters, so its held-out prediction on any row
    is just that expert's own probability, which makes the fit numerically
    identical to the in-sample fit used below.
    """
    if loso_path_predictions is not None and "selected_shrinkage" in selection:
        shrinkage = float(selection["selected_shrinkage"])
        rung = loso_path_predictions.get(shrinkage)
        if rung is not None:
            rung = np.asarray(rung, dtype=float)
            finite = np.isfinite(rung)
            if finite.sum() >= 50:
                return TemperatureCalibrator().fit(
                    rung[finite], np.asarray(y, dtype=int)[finite]
                )

    if selection.get("selected") == "learned":
        return learned_calibrator
    selected = selection.get("selected")
    if selected == "shrunk":
        # A shrunk pool without usable path predictions: the learned
        # temperature is the closest honest match, and it was fitted on the
        # same LOSO rows.
        return learned_calibrator
    if selected not in expert_probabilities:
        raise ValueError(f"selected pool expert is unavailable: {selected}")
    return TemperatureCalibrator().fit(expert_probabilities[selected], y)


def fit_selected_market_calibrator(
    selection,
    expert_probabilities,
    y,
    learned_calibrator,
    loso_path_predictions=None,
):
    """Backward-compatible name for the shared pool calibration policy."""
    return fit_selected_pool_calibrator(
        selection,
        expert_probabilities,
        y,
        learned_calibrator,
        loso_path_predictions=loso_path_predictions,
    )


def acceptance_against_experts(
    model_metrics,
    expert_metrics,
    *,
    accuracy_tolerance=0.01,
    loss_tolerance=0.005,
):
    """Apply the release thresholds on one directly comparable regime."""
    if not expert_metrics:
        return {"passed": True, "reason": "no_games"}
    best_accuracy = max(metrics["accuracy"] for metrics in expert_metrics.values())
    best_log_loss = min(metrics["log_loss"] for metrics in expert_metrics.values())
    best_brier = min(metrics["brier"] for metrics in expert_metrics.values())
    result = {
        "best_expert_accuracy": best_accuracy,
        "best_expert_log_loss": best_log_loss,
        "best_expert_brier": best_brier,
        "accuracy_pass": model_metrics["accuracy"]
        >= best_accuracy - float(accuracy_tolerance),
        "log_loss_pass": model_metrics["log_loss"]
        <= best_log_loss + float(loss_tolerance),
        "brier_pass": model_metrics["brier"] <= best_brier + float(loss_tolerance),
    }
    result["passed"] = bool(
        result["accuracy_pass"] and result["log_loss_pass"] and result["brier_pass"]
    )
    return result


def apply_consensus_guard(
    combined, tier_a, tier_b, tier_c=None, market=None, valid_market=None
):
    """Replace a side-reversing ensemble result with Tier B.

    The guard applies only when every available expert is strictly on the same
    side of 0.5 and the combined probability crosses to the opposite side.
    """
    combined = _clip_probs(combined)
    tier_a = _clip_probs(tier_a)
    tier_b = _clip_probs(tier_b)
    n = len(combined)
    experts = [tier_a, tier_b]
    available = [np.ones(n, dtype=bool), np.ones(n, dtype=bool)]

    if tier_c is not None:
        tier_c_arr = np.asarray(tier_c, dtype=float)
        experts.append(_clip_probs(tier_c_arr))
        available.append(np.isfinite(tier_c_arr))
    if market is not None:
        market_arr = np.asarray(market, dtype=float)
        market_available = np.isfinite(market_arr)
        if valid_market is not None:
            market_available &= np.asarray(valid_market, dtype=bool)
        experts.append(_clip_probs(market_arr))
        available.append(market_available)

    all_home = np.ones(n, dtype=bool)
    all_away = np.ones(n, dtype=bool)
    for values, present in zip(experts, available):
        all_home &= ~present | (values > 0.5)
        all_away &= ~present | (values < 0.5)

    reversed_side = (all_home & (combined < 0.5)) | (all_away & (combined > 0.5))
    guarded = combined.copy()
    guarded[reversed_side] = tier_b[reversed_side]
    return guarded, reversed_side


def predict_probability_regimes(
    *,
    tier_a,
    tier_b,
    tier_c,
    market,
    valid_market,
    market_stacker,
    market_calibrator,
    no_market_stacker=None,
    no_market_calibrator=None,
    no_market_strategy="tier_b",
    legacy_extra=None,
):
    """Route each row through a genuine-market or model-only probability path.

    Legacy stackers remain usable for rows with real H2H prices. Missing-market
    rows never receive a fabricated 0.5 input: they use the selected no-market
    pool or Tier B directly.
    """
    tier_a = np.asarray(tier_a, dtype=float)
    tier_b = np.asarray(tier_b, dtype=float)
    tier_c = None if tier_c is None else np.asarray(tier_c, dtype=float)
    market = np.asarray(market, dtype=float)
    valid_market = np.asarray(valid_market, dtype=bool)
    result = _clip_probs(tier_b)
    routes = {
        "market": 0,
        "no_market_pool": 0,
        "tier_b": int((~valid_market).sum()),
        "consensus_guarded": 0,
    }

    market_rows = np.flatnonzero(valid_market)
    if len(market_rows) and market_stacker is not None:
        try:
            if isinstance(market_stacker, SimplexLogitPool):
                predicted = market_stacker.predict(
                    tier_a[market_rows],
                    tier_b[market_rows],
                    tier_c=None if tier_c is None else tier_c[market_rows],
                    market=market[market_rows],
                )
            else:
                predicted = market_stacker.predict(
                    tier_a[market_rows],
                    tier_b[market_rows],
                    market[market_rows],
                    np.zeros(len(market_rows), dtype=float),
                    tier_c=None if tier_c is None else tier_c[market_rows],
                    extra=None
                    if legacy_extra is None
                    else np.asarray(legacy_extra)[market_rows],
                )
            if market_calibrator is not None:
                predicted = market_calibrator.predict(predicted)
            result[market_rows] = _clip_probs(predicted)
            routes["market"] = len(market_rows)
        except Exception:
            # A malformed/partial artifact must fail safe to Tier B.
            result[market_rows] = _clip_probs(tier_b[market_rows])
            routes["tier_b"] += len(market_rows)
    else:
        routes["tier_b"] += len(market_rows)

    no_market_rows = np.flatnonzero(~valid_market)
    if (
        len(no_market_rows)
        and no_market_strategy == "simplex"
        and no_market_stacker is not None
    ):
        try:
            predicted = no_market_stacker.predict(
                tier_a[no_market_rows],
                tier_b[no_market_rows],
                tier_c=None if tier_c is None else tier_c[no_market_rows],
            )
            if no_market_calibrator is not None:
                predicted = no_market_calibrator.predict(predicted)
            result[no_market_rows] = _clip_probs(predicted)
            routes["no_market_pool"] = len(no_market_rows)
            routes["tier_b"] -= len(no_market_rows)
        except Exception:
            result[no_market_rows] = _clip_probs(tier_b[no_market_rows])

    result, guarded = apply_consensus_guard(
        result,
        tier_a,
        tier_b,
        tier_c=tier_c,
        market=market,
        valid_market=valid_market,
    )
    routes["consensus_guarded"] = int(guarded.sum())
    return result, routes


def save_artifact(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)
