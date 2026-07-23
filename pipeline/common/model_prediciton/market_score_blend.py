"""Apply genuine, fresh line and totals markets to expected score means."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.common.odds.validity import valid_price_pair


MIN_SCORE_MEAN = 1e-6


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _available_flag(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Treat an explicit missing flag as authoritative; old frames may omit it."""
    if name not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(values) & (values < 0.5)


def _paired_price_mask(
    frame: pd.DataFrame,
    first_name: str,
    second_name: str,
) -> np.ndarray:
    first = _numeric_column(frame, first_name)
    second = _numeric_column(frame, second_name)
    return np.fromiter(
        (
            valid_price_pair(first_price, second_price)
            for first_price, second_price in zip(first, second)
        ),
        dtype=bool,
        count=len(frame),
    )


def apply_market_score_mean_blends(
    frame: pd.DataFrame,
    mu_home,
    mu_away,
    baseline_mu_home,
    baseline_mu_away,
    *,
    fresh_market,
    fresh_line_market=None,
    fresh_total_market=None,
    margin_blend=None,
    total_blend=None,
):
    """Return score means adjusted only by complete, fresh market families.

    Totals are applied first. A valid margin target then splits the resulting
    total into home and away means, so the line changes the simulated
    scoreline rather than becoming a contradictory post-simulation override.
    """
    home = np.maximum(np.asarray(mu_home, dtype=float).copy(), MIN_SCORE_MEAN)
    away = np.maximum(np.asarray(mu_away, dtype=float).copy(), MIN_SCORE_MEAN)
    tier_a_home = np.asarray(baseline_mu_home, dtype=float)
    tier_a_away = np.asarray(baseline_mu_away, dtype=float)
    fresh = np.asarray(fresh_market, dtype=bool)
    fresh_line = np.asarray(
        fresh if fresh_line_market is None else fresh_line_market,
        dtype=bool,
    )
    fresh_total = np.asarray(
        fresh if fresh_total_market is None else fresh_total_market,
        dtype=bool,
    )

    n_rows = len(frame)
    arrays = (
        home,
        away,
        tier_a_home,
        tier_a_away,
        fresh,
        fresh_line,
        fresh_total,
    )
    if any(len(values) != n_rows for values in arrays):
        raise ValueError("score means, baseline means, and freshness must match frame rows")

    line_applied = np.zeros(n_rows, dtype=bool)
    total_applied = np.zeros(n_rows, dtype=bool)

    if isinstance(total_blend, dict):
        total_line = _numeric_column(frame, "market_total_line")
        if not np.isfinite(total_line).any():
            total_line = _numeric_column(frame, "total_line")
        total_valid = (
            fresh_total
            & _available_flag(frame, "totals_missing")
            & _paired_price_mask(frame, "total_over_odds", "total_under_odds")
            & np.isfinite(total_line)
            & (total_line > 0.0)
        )
        model_total = home + away
        target_total = (
            float(total_blend.get("intercept", 0.0))
            + float(total_blend.get("coef_model_total", 0.0)) * model_total
            + float(total_blend.get("coef_market_total", 0.0)) * total_line
        )
        total_valid &= np.isfinite(target_total) & (target_total > 2 * MIN_SCORE_MEAN)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(total_valid, target_total / model_total, 1.0)
        # The market can nudge pace, but cannot make one feed response rewrite
        # the score model wholesale.
        scale = np.where(total_valid, np.clip(scale, 0.75, 1.35), 1.0)
        home = np.maximum(home * scale, MIN_SCORE_MEAN)
        away = np.maximum(away * scale, MIN_SCORE_MEAN)
        total_applied = total_valid & (np.abs(scale - 1.0) > 1e-12)

    if isinstance(margin_blend, dict):
        implied_spread_home = _numeric_column(frame, "implied_spread_home")
        line_valid = (
            fresh_line
            & _available_flag(frame, "line_odds_missing")
            & _paired_price_mask(
                frame,
                "team_line_odds_home",
                "team_line_odds_away",
            )
            & np.isfinite(implied_spread_home)
        )
        model_margin = np.asarray(mu_home, dtype=float) - np.asarray(
            mu_away, dtype=float
        )
        tier_a_margin = tier_a_home - tier_a_away
        market_margin = -implied_spread_home
        target_margin = (
            float(margin_blend.get("intercept", 0.0))
            + float(margin_blend.get("coef_model_margin", 0.0)) * model_margin
            + float(margin_blend.get("coef_market_spread", 0.0)) * market_margin
            + float(margin_blend.get("coef_tier_a_margin", 0.0)) * tier_a_margin
        )
        line_valid &= np.isfinite(target_margin)

        adjusted_total = home + away
        max_abs_margin = np.maximum(adjusted_total - 2 * MIN_SCORE_MEAN, 0.0)
        bounded_margin = np.clip(target_margin, -max_abs_margin, max_abs_margin)
        candidate_home = np.maximum(
            (adjusted_total + bounded_margin) / 2.0,
            MIN_SCORE_MEAN,
        )
        candidate_away = np.maximum(
            (adjusted_total - bounded_margin) / 2.0,
            MIN_SCORE_MEAN,
        )
        home = np.where(line_valid, candidate_home, home)
        away = np.where(line_valid, candidate_away, away)
        line_applied = line_valid

    return home, away, {
        "line_applied": line_applied,
        "total_applied": total_applied,
        "line_count": int(line_applied.sum()),
        "total_count": int(total_applied.sum()),
    }
