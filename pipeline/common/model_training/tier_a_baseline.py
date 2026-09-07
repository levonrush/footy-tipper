import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skellam


SORT_COLS = ["competition_year", "round_id", "start_time", "game_number", "game_id"]


@dataclass
class TierABaselineConfig:
    alpha: float = 0.2
    carryover: float = 0.6
    min_rate: float = 0.5
    max_rate: float = 80.0
    base_home: float | None = None
    base_away: float | None = None


def _clip_rate(value: float, min_rate: float, max_rate: float) -> float:
    return float(np.clip(value, min_rate, max_rate))


def _get_float(value, default):
    try:
        val = float(value)
    except Exception:
        return float(default)
    if np.isnan(val):
        return float(default)
    return val


def _match_probabilities(mu_home: float, mu_away: float) -> tuple[float, float, float, float]:
    # diff = home - away; draw is diff == 0
    draw_prob = float(skellam.pmf(0, mu_home, mu_away))
    home_win = float(1.0 - skellam.cdf(0, mu_home, mu_away))
    away_win = float(max(0.0, 1.0 - home_win - draw_prob))

    non_draw = max(1e-9, home_win + away_win)
    home_win_conditional = home_win / non_draw
    return home_win, away_win, draw_prob, home_win_conditional


def default_baseline_config_from_env() -> TierABaselineConfig:
    return TierABaselineConfig(
        alpha=float(os.getenv("FOOTY_TIPPER_BASELINE_ALPHA", "0.2")),
        carryover=float(os.getenv("FOOTY_TIPPER_BASELINE_CARRYOVER", "0.6")),
        min_rate=float(os.getenv("FOOTY_TIPPER_BASELINE_MIN_RATE", "0.5")),
        max_rate=float(os.getenv("FOOTY_TIPPER_BASELINE_MAX_RATE", "80")),
    )


def _resolve_base_rates(df: pd.DataFrame, config: TierABaselineConfig) -> tuple[float, float]:
    if config.base_home is not None and config.base_away is not None:
        return float(config.base_home), float(config.base_away)

    finals = df[df["game_state_name"] == "Final"]
    if finals.empty:
        # Conservative historical defaults for NRL-like scoring.
        return 22.0, 20.0

    base_home = _get_float(finals["team_final_score_home"].mean(), 22.0)
    base_away = _get_float(finals["team_final_score_away"].mean(), 20.0)
    return base_home, base_away


# The attack/defence ratings are the reason the baseline says what it says, so
# they are returned alongside it. They are read before the post-match update
# below, which keeps them leak-safe by construction.
BASELINE_FEATURE_COLUMNS = [
    "game_id",
    "baseline_mu_home",
    "baseline_mu_away",
    "baseline_draw_prob",
    "baseline_home_win_prob_conditional",
    "tier_a_attack_home",
    "tier_a_defence_home",
    "tier_a_attack_away",
    "tier_a_defence_away",
]


REQUIRED_BASELINE_COLUMNS = {
    "game_id",
    "competition_year",
    "round_id",
    "start_time",
    "game_number",
    "team_home",
    "team_away",
    "game_state_name",
}


_LOGIT_CLIP = 1e-4
# Two parameters fitted on thousands of games; a season's worth is not enough to
# trust the slope, so below this the ratings stay raw and say so.
_MIN_CALIBRATION_GAMES = 200


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _LOGIT_CLIP, 1.0 - _LOGIT_CLIP)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    # exp overflows to +inf for strongly negative z, which is the correct answer
    # here (the sigmoid goes to zero), so the warning is noise.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_logit_calibration(raw_probabilities, outcomes, max_iter: int = 100, tol: float = 1e-10):
    """Platt scaling of Tier-A probabilities: `sigmoid(a + b * logit(p))`.

    The ratings rank matchups well (about 63% tipping accuracy on 2015 onward) but
    their probabilities are severely overconfident: the raw 0.9-1.0 bucket wins
    about 72% of the time, and nearly half of all games are pushed outside
    [0.1, 0.9]. Simulating a finals bracket on the raw numbers would report a
    minor premier as an 85% premiership chance. Two parameters fitted by Newton's
    method on completed matches fix the spread without touching the ordering.

    Returns `(a, b)`, or None when there is not enough history to fit.
    """
    x = _logit(raw_probabilities)
    y = np.asarray(outcomes, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < _MIN_CALIBRATION_GAMES or len(np.unique(y)) < 2:
        return None

    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2, dtype=float)
    # `matmul` reports whatever floating-point flags are already set on the
    # thread, and the ratings walk that produced these probabilities makes
    # thousands of Skellam calls that legitimately underflow. Silencing the
    # inherited flags here keeps a clean fit from printing alarming warnings;
    # the finiteness and slope checks below remain the real guard.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for _ in range(max_iter):
            probabilities = _sigmoid(design @ beta)
            weights = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
            gradient = design.T @ (y - probabilities)
            hessian = design.T @ (design * weights[:, None])
            # Ridge term keeps the solve well conditioned on near-separable inputs.
            hessian[np.diag_indices_from(hessian)] += 1e-8
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return None
            beta = beta + step
            if not np.all(np.isfinite(beta)):
                return None
            if np.max(np.abs(step)) < tol:
                break
    if beta[1] <= 0.0:
        # A non-positive slope would invert the ratings. Refuse rather than lie.
        return None
    return float(beta[0]), float(beta[1])


def apply_logit_calibration(probability, calibration, include_intercept: bool = True):
    """Apply a fitted Platt scaling.

    The fit is done on home teams, so its intercept absorbs whatever home bias the
    base rates did not. At a neutral venue there is no home side, so only the slope
    (the confidence shrinkage) applies. Dropping the intercept there also makes the
    transform exactly antisymmetric, which is what keeps the two sides of a Grand
    Final summing to one.
    """
    if calibration is None:
        return float(probability)
    intercept, slope = calibration
    offset = intercept if include_intercept else 0.0
    return float(_sigmoid(offset + slope * _logit(probability)))


@dataclass(frozen=True)
class TierARatings:
    """Team attack/defence multipliers after the last observed completed match.

    The same numbers the baseline features are read from, exposed as end state so
    a caller can price a matchup that is not in the fixture list (a finals bracket
    that has not been drawn yet, for example).
    """

    attack: dict[str, float]
    defence: dict[str, float]
    base_home: float
    base_away: float
    calibration: tuple[float, float] | None = None

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def expected_scores(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        base_home = self.base_home
        base_away = self.base_away
        if neutral:
            # No home ground: split the difference rather than pretending one side
            # of a decider at a neutral venue holds the advantage.
            base_home = base_away = 0.5 * (self.base_home + self.base_away)
        attack_home = self.attack.get(str(home_team), 1.0)
        defence_home = self.defence.get(str(home_team), 1.0)
        attack_away = self.attack.get(str(away_team), 1.0)
        defence_away = self.defence.get(str(away_team), 1.0)
        return (
            base_home * attack_home * defence_away,
            base_away * attack_away * defence_home,
        )

    def home_win_probability(self, home_team: str, away_team: str, neutral: bool = False) -> float:
        """Draw-excluded, calibrated probability that `home_team` wins.

        Matches the draw-excluded convention every published number uses. A neutral
        decider is priced from both orientations and averaged, so the two sides of
        the same fixture always sum to one.
        """
        raw = self._raw_home_win_probability(home_team, away_team, neutral=neutral)
        return apply_logit_calibration(
            raw, self.calibration, include_intercept=not neutral
        )

    def _raw_home_win_probability(self, home_team: str, away_team: str, neutral: bool = False) -> float:
        mu_home, mu_away = self.expected_scores(home_team, away_team, neutral=neutral)
        _, _, _, home_win_conditional = _match_probabilities(mu_home, mu_away)
        return home_win_conditional

    def knows(self, team: str) -> bool:
        return str(team) in self.attack and str(team) in self.defence


def _prepare(df: pd.DataFrame, config: TierABaselineConfig | None):
    if config is None:
        config = default_baseline_config_from_env()

    missing = sorted(REQUIRED_BASELINE_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError("Tier-A baseline requires columns: " + ", ".join(missing))

    if df.empty:
        return config, None, None, None

    ordered = df.sort_values(SORT_COLS).reset_index(drop=True)
    base_home, base_away = _resolve_base_rates(ordered, config)
    return config, ordered, base_home, base_away


def _accumulate(ordered: pd.DataFrame, config: TierABaselineConfig, base_home: float, base_away: float):
    """Walk the fixtures once, returning per-row features and the end ratings.

    Both public entry points share this loop so the feature output cannot drift
    from the ratings the premiership simulation prices hypothetical matchups with.
    """
    attack: dict[str, float] = {}
    defence: dict[str, float] = {}

    rows = []
    current_season = None

    for row in ordered.itertuples(index=False):
        season = int(getattr(row, "competition_year"))
        if current_season is None:
            current_season = season
        elif season != current_season:
            # Offseason shrinkage back toward league-average team strength.
            for team_key in list(attack.keys()):
                attack[team_key] = 1.0 + config.carryover * (attack[team_key] - 1.0)
                defence[team_key] = 1.0 + config.carryover * (defence[team_key] - 1.0)
            current_season = season

        home_team = str(getattr(row, "team_home"))
        away_team = str(getattr(row, "team_away"))

        attack_home = attack.get(home_team, 1.0)
        defence_home = defence.get(home_team, 1.0)
        attack_away = attack.get(away_team, 1.0)
        defence_away = defence.get(away_team, 1.0)

        mu_home = _clip_rate(base_home * attack_home * defence_away, config.min_rate, config.max_rate)
        mu_away = _clip_rate(base_away * attack_away * defence_home, config.min_rate, config.max_rate)

        _, _, draw_prob, home_win_conditional = _match_probabilities(mu_home, mu_away)

        rows.append(
            {
                "game_id": getattr(row, "game_id"),
                "baseline_mu_home": mu_home,
                "baseline_mu_away": mu_away,
                "baseline_draw_prob": draw_prob,
                "baseline_home_win_prob_conditional": home_win_conditional,
                "tier_a_attack_home": attack_home,
                "tier_a_defence_home": defence_home,
                "tier_a_attack_away": attack_away,
                "tier_a_defence_away": defence_away,
            }
        )

        if getattr(row, "game_state_name") != "Final":
            continue

        try:
            score_home = float(getattr(row, "team_final_score_home"))
            score_away = float(getattr(row, "team_final_score_away"))
        except Exception:
            continue

        if np.isnan(score_home) or np.isnan(score_away):
            continue

        obs_attack_home = score_home / max(base_home * defence_away, 1e-6)
        obs_defence_away = score_home / max(base_home * attack_home, 1e-6)
        obs_attack_away = score_away / max(base_away * defence_home, 1e-6)
        obs_defence_home = score_away / max(base_away * attack_away, 1e-6)

        attack[home_team] = (1.0 - config.alpha) * attack_home + config.alpha * float(np.clip(obs_attack_home, 0.25, 4.0))
        defence[away_team] = (1.0 - config.alpha) * defence_away + config.alpha * float(np.clip(obs_defence_away, 0.25, 4.0))

        attack[away_team] = (1.0 - config.alpha) * attack_away + config.alpha * float(np.clip(obs_attack_away, 0.25, 4.0))
        defence[home_team] = (1.0 - config.alpha) * defence_home + config.alpha * float(np.clip(obs_defence_home, 0.25, 4.0))

    return rows, attack, defence


def compute_tier_a_baseline_features(df: pd.DataFrame, config: TierABaselineConfig | None = None) -> pd.DataFrame:
    """Compute leak-safe dynamic team-strength baseline features for each match row."""
    config, ordered, base_home, base_away = _prepare(df, config)
    if ordered is None:
        return pd.DataFrame(columns=BASELINE_FEATURE_COLUMNS)
    rows, _attack, _defence = _accumulate(ordered, config, base_home, base_away)
    return pd.DataFrame(rows)


def compute_tier_a_ratings(df: pd.DataFrame, config: TierABaselineConfig | None = None) -> TierARatings:
    """Team ratings after the last completed match in `df`.

    Runs the identical accumulation as `compute_tier_a_baseline_features` and keeps
    the end state instead of the per-row features.
    """
    config, ordered, base_home, base_away = _prepare(df, config)
    if ordered is None:
        return TierARatings({}, {}, 22.0, 20.0)
    rows, attack, defence = _accumulate(ordered, config, base_home, base_away)
    return TierARatings(
        dict(attack),
        dict(defence),
        float(base_home),
        float(base_away),
        _fit_ratings_calibration(ordered, rows),
    )


def _fit_ratings_calibration(ordered: pd.DataFrame, rows: list) -> tuple[float, float] | None:
    """Fit Platt scaling from the same walk that produced the ratings.

    Every row was scored before its own result was folded in, so the fit is
    leak-safe by construction. Draws carry no two-way outcome and are dropped.
    """
    if not rows:
        return None
    try:
        baseline = pd.DataFrame(rows)[["game_id", "baseline_home_win_prob_conditional"]]
        results = ordered.loc[
            ordered["game_state_name"] == "Final",
            ["game_id", "team_final_score_home", "team_final_score_away"],
        ]
        merged = results.merge(baseline, on="game_id", how="inner").dropna()
        home = pd.to_numeric(merged["team_final_score_home"], errors="coerce")
        away = pd.to_numeric(merged["team_final_score_away"], errors="coerce")
        decided = home != away
        return fit_logit_calibration(
            merged.loc[decided, "baseline_home_win_prob_conditional"],
            (home > away)[decided],
        )
    except Exception:
        # Ratings without a calibration are still usable; a failed fit must not
        # take the baseline features down with it.
        return None


DEFAULT_TUNE_ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_TUNE_CARRYOVERS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def tune_baseline_hyperparams(
    df: pd.DataFrame,
    alphas=DEFAULT_TUNE_ALPHAS,
    carryovers=DEFAULT_TUNE_CARRYOVERS,
    config_template: TierABaselineConfig | None = None,
) -> tuple[TierABaselineConfig, pd.DataFrame]:
    """Grid-search alpha/carryover on sequential conditional-win-prob log-loss.

    The ratings are updated strictly after each Final, so every prediction is
    out-of-sample by construction; the first observed season is excluded from
    scoring as ratings warm-up. Returns (best_config, results_frame).
    """
    template = config_template or default_baseline_config_from_env()
    base_cols = df[
        ["game_id", "competition_year", "game_state_name", "team_final_score_home", "team_final_score_away"]
    ].copy()

    results = []
    best_cfg, best_ll = None, np.inf
    for alpha in alphas:
        for carryover in carryovers:
            cfg = TierABaselineConfig(
                alpha=float(alpha),
                carryover=float(carryover),
                min_rate=template.min_rate,
                max_rate=template.max_rate,
                base_home=template.base_home,
                base_away=template.base_away,
            )
            feats = compute_tier_a_baseline_features(df, cfg)
            merged = base_cols.merge(feats, on="game_id", how="inner")
            finals = merged[merged["game_state_name"] == "Final"]
            years = pd.to_numeric(finals["competition_year"], errors="coerce")
            score_home = pd.to_numeric(finals["team_final_score_home"], errors="coerce")
            score_away = pd.to_numeric(finals["team_final_score_away"], errors="coerce")
            scored = (
                (years > years.min())
                & score_home.notna()
                & score_away.notna()
                & (score_home != score_away)
            )
            if scored.sum() < 100:
                continue
            y = (score_home[scored] > score_away[scored]).to_numpy(dtype=float)
            p = np.clip(
                pd.to_numeric(
                    finals.loc[scored, "baseline_home_win_prob_conditional"], errors="coerce"
                ).fillna(0.5).to_numpy(dtype=float),
                1e-6,
                1 - 1e-6,
            )
            ll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            acc = float(np.mean((p > 0.5) == (y > 0.5)))
            results.append({"alpha": float(alpha), "carryover": float(carryover), "log_loss": ll, "accuracy": acc, "games": int(scored.sum())})
            if ll < best_ll:
                best_cfg, best_ll = cfg, ll

    return best_cfg or template, pd.DataFrame(results)


def baseline_config_to_dict(config: TierABaselineConfig, base_home: float, base_away: float) -> dict:
    return {
        "alpha": float(config.alpha),
        "carryover": float(config.carryover),
        "min_rate": float(config.min_rate),
        "max_rate": float(config.max_rate),
        "base_home": float(base_home),
        "base_away": float(base_away),
    }


def baseline_config_from_dict(payload: dict) -> TierABaselineConfig:
    return TierABaselineConfig(
        alpha=float(payload.get("alpha", 0.2)),
        carryover=float(payload.get("carryover", 0.6)),
        min_rate=float(payload.get("min_rate", 0.5)),
        max_rate=float(payload.get("max_rate", 80.0)),
        base_home=float(payload.get("base_home", 22.0)),
        base_away=float(payload.get("base_away", 20.0)),
    )
