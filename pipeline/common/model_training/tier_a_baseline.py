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


def compute_tier_a_baseline_features(df: pd.DataFrame, config: TierABaselineConfig | None = None) -> pd.DataFrame:
    """Compute leak-safe dynamic team-strength baseline features for each match row."""
    if config is None:
        config = default_baseline_config_from_env()

    required = {"game_id", "competition_year", "round_id", "start_time", "game_number", "team_home", "team_away", "game_state_name"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Tier-A baseline requires columns: " + ", ".join(missing))

    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "baseline_mu_home",
                "baseline_mu_away",
                "baseline_draw_prob",
                "baseline_home_win_prob_conditional",
            ]
        )

    ordered = df.sort_values(SORT_COLS).reset_index(drop=True)
    base_home, base_away = _resolve_base_rates(ordered, config)

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

    return pd.DataFrame(rows)


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
