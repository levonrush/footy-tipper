"""Per-game explanations: ranked drivers plus a one-line plain-English why.

Combines the two halves of the story:

* the exact arithmetic chain (trace.py), which says WHICH model decided the tip
* TreeSHAP over that model's features (contributions.py), which says why

Driver text is composed at the family level, never the feature level:
"team-list strength" reads; "lineup_avg_spine_margin_rating_delta" does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from pipeline.common.explain import contributions as xc
from pipeline.common.explain import families as fam
from pipeline.common.explain import trace as xt
from pipeline.common.explain import units

# Percentage points. Below this a clause is noise and clutters the sentence.
DEFAULT_MIN_POINTS = 0.8
DEFAULT_MAX_CLAUSES = 3

NO_DOMINANT_DRIVER = "No single factor dominates: this one is close."
MARKET_ONLY_WHY = "Tip follows the market price; model features do not move it."


@dataclass(frozen=True)
class Driver:
    """One ranked reason, in the units of whatever it is explaining."""

    key: str
    label: str
    family: str
    points: float  # signed, toward the tipped side
    share: float  # signed fraction of total absolute contribution
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GameExplanation:
    game_id: int
    team_home: str
    team_away: str
    probability: xt.ProbabilityTrace
    score: xt.ScoreTrace
    prob_drivers: tuple = ()
    margin_drivers: tuple = ()
    prob_families: tuple = ()
    margin_families: tuple = ()
    why_line: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def tipped_team(self) -> str:
        return self.team_home if self.probability.tipped_home else self.team_away

    def as_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "team_home": self.team_home,
            "team_away": self.team_away,
            "tipped_team": self.tipped_team,
            "why_line": self.why_line,
            "probability": self.probability.as_dict(),
            "score": self.score.as_dict(),
            "prob_drivers": [d.as_dict() for d in self.prob_drivers],
            "margin_drivers": [d.as_dict() for d in self.margin_drivers],
            "prob_families": [d.as_dict() for d in self.prob_families],
            "margin_families": [d.as_dict() for d in self.margin_families],
            "meta": dict(self.meta),
        }


def _rank(values, keys, labels, family_of, *, sign, details=None, top=None):
    """Signed drivers ordered by magnitude, flipped toward the tipped side."""
    values = np.asarray(values, dtype=float) * sign
    total = float(np.abs(values).sum())
    order = np.argsort(-np.abs(values))
    if top:
        order = order[:top]
    drivers = []
    for i in order:
        if not np.isfinite(values[i]) or values[i] == 0.0:
            continue
        drivers.append(
            Driver(
                key=str(keys[i]),
                label=str(labels[i]),
                family=family_of(keys[i]),
                points=float(values[i]),
                share=float(values[i] / total) if total > 0 else 0.0,
                detail="" if details is None else str(details[i]),
            )
        )
    return tuple(drivers)


def one_line_why(
    explanation: GameExplanation,
    *,
    max_drivers=DEFAULT_MAX_CLAUSES,
    min_points=DEFAULT_MIN_POINTS,
) -> str:
    """The email/site sentence. A pure function: no LLM, no network."""
    probability = explanation.probability
    if probability.attribution_source == xt.ATTRIBUTION_EXPERTS:
        return MARKET_ONLY_WHY

    drivers = [d for d in explanation.prob_families if abs(d.points) >= min_points]
    if not drivers:
        return NO_DOMINANT_DRIVER
    drivers = drivers[:max_drivers]

    tipped = explanation.tipped_team
    forwards = [d for d in drivers if d.points > 0]
    against = [d for d in drivers if d.points < 0]

    def clause(driver):
        return f"{driver.label.lower()} ({driver.points:+.0f} pts)"

    # Lead with the biggest single reason, since that is what the clause order
    # already shows. A tip carried by the model's base rate against its own
    # features must not read as "favoured on" a small positive.
    if forwards and drivers[0].points > 0:
        head = f"{tipped} favoured on " + _join([clause(d) for d in forwards])
        tail = ""
        if against:
            verb = "works" if len(against) == 1 else "work"
            tail = "; " + _join([clause(d) for d in against]) + f" {verb} against them"
    else:
        head = f"{tipped} tipped despite " + _join([clause(d) for d in against or forwards])
        tail = ""
        if against and forwards:
            tail = "; " + _join([clause(d) for d in forwards]) + " in their favour"

    prefix = ""
    if probability.guard_fired:
        # The ensemble tried to reverse the side and was overruled, so these
        # drivers come from the score models, not the classifier. Say so.
        prefix = "Guard override: "
    return f"{prefix}{head}{tail}."


def _join(parts):
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def explain_games(
    *,
    inference_data: pd.DataFrame,
    predictors,
    stack: xt.ProbabilityStack,
    home_model,
    away_model,
    tier_a,
    tier_b,
    published_cond,
    tier_c=None,
    market=None,
    valid_market=None,
    routes=None,
    mu_model_home=None,
    mu_model_away=None,
    mu_blended_home=None,
    mu_blended_away=None,
    mu_final_home=None,
    mu_final_away=None,
    blend_weight_home=1.0,
    blend_weight_away=1.0,
    score_market_diagnostics=None,
    sim_diagnostics=None,
    outcomes=None,
    top_drivers=8,
) -> list:
    """Build a GameExplanation per row of inference_data.

    Every input is a value inference already computed. Nothing is re-predicted
    except the TreeSHAP pass, and nothing is re-simulated at all.
    """
    frame = inference_data.reset_index(drop=True)
    game_ids = frame["game_id"].to_numpy()
    X = frame[list(predictors)]

    prob_contribs = xc.raw_contributions(stack.binary_model, X) if stack.binary_model else None
    home_contribs = xc.raw_contributions(home_model, X)
    away_contribs = xc.raw_contributions(away_model, X)

    mu_home_used, mu_away_used = _resolve_simulated_means(
        frame, sim_diagnostics, mu_final_home, mu_final_away, mu_blended_home, mu_blended_away
    )

    traces = xt.build_probability_traces(
        game_ids=game_ids,
        stack=stack,
        tier_a=tier_a,
        tier_b=tier_b,
        tier_c=tier_c,
        market=market,
        valid_market=valid_market,
        routes=routes,
        published_cond=published_cond,
        draw_prob=_column(outcomes, "draw_prob", len(frame)),
        mu_home=mu_home_used,
        mu_away=mu_away_used,
    )

    names = home_contribs.feature_names
    labels = [fam.family_label(fam.family_for(name)) for name in names]
    diagnostics = score_market_diagnostics or {}
    sim_lookup = _sim_lookup(sim_diagnostics)

    explanations = []
    for i, trace in enumerate(traces):
        margin_contrib = units.margin_points(
            home_contribs.values[i], mu_home_used[i], away_contribs.values[i], mu_away_used[i]
        )
        sign = 1.0 if trace.tipped_home else -1.0

        if trace.attribution_source == xt.ATTRIBUTION_BINARY and prob_contribs is not None:
            scale = (
                units.PCT_SCALE
                * trace.published_cond
                * (1.0 - trace.published_cond)
                * trace.feature_multiplier
            )
            prob_contrib = prob_contribs.values[i] * scale
        elif trace.attribution_source == xt.ATTRIBUTION_SCORE:
            # Published probability came from the score models on this row, so
            # its drivers are the margin drivers priced in probability terms.
            prob_contrib = margin_contrib * trace.feature_multiplier * units.PCT_SCALE
        else:
            prob_contrib = np.zeros_like(margin_contrib)

        values = [_value_detail(frame, name, i) for name in names]
        prob_drivers = _rank(prob_contrib, names, names, fam.family_for, sign=sign,
                             details=values, top=top_drivers)
        margin_drivers = _rank(margin_contrib, names, names, fam.family_for, sign=sign,
                               details=values, top=top_drivers)

        prob_families = _family_drivers(prob_contrib, names, sign)
        margin_families = _family_drivers(margin_contrib, names, sign)

        score = xt.ScoreTrace(
            game_id=int(trace.game_id),
            mu_model_home=_at(mu_model_home, i),
            mu_model_away=_at(mu_model_away, i),
            mu_baseline_home=_frame_value(frame, "baseline_mu_home", i),
            mu_baseline_away=_frame_value(frame, "baseline_mu_away", i),
            blend_weight_home=float(blend_weight_home),
            blend_weight_away=float(blend_weight_away),
            mu_blended_home=_at(mu_blended_home, i),
            mu_blended_away=_at(mu_blended_away, i),
            mu_final_home=float(mu_home_used[i]),
            mu_final_away=float(mu_away_used[i]),
            line_applied=_mask_at(diagnostics.get("line_applied"), i),
            total_applied=_mask_at(diagnostics.get("total_applied"), i),
            reconciled=bool(sim_lookup.get(int(trace.game_id), {}).get("reconciled", False)),
            displayed_home=_outcome_int(outcomes, "predicted_home_score", trace.game_id),
            displayed_away=_outcome_int(outcomes, "predicted_away_score", trace.game_id),
            displayed_margin=_outcome_int(outcomes, "predicted_margin", trace.game_id),
            tier_a_attack_home=_frame_value(frame, "tier_a_attack_home", i),
            tier_a_defence_home=_frame_value(frame, "tier_a_defence_home", i),
            tier_a_attack_away=_frame_value(frame, "tier_a_attack_away", i),
            tier_a_defence_away=_frame_value(frame, "tier_a_defence_away", i),
        )

        explanation = GameExplanation(
            game_id=int(trace.game_id),
            team_home=str(frame.at[i, "team_home"]),
            team_away=str(frame.at[i, "team_away"]),
            probability=trace,
            score=score,
            prob_drivers=prob_drivers,
            margin_drivers=margin_drivers,
            prob_families=prob_families,
            margin_families=margin_families,
            meta={"taxonomy_version": fam.FAMILY_TAXONOMY_VERSION},
        )
        explanations.append(replace(explanation, why_line=one_line_why(explanation)))
    return explanations


def _family_drivers(contrib, names, sign):
    grouped = fam.group_by_family(np.asarray(contrib)[None, :], names)
    keys = list(grouped.columns)
    values = grouped.iloc[0].to_numpy(dtype=float)
    labels = [fam.family_label(key) for key in keys]
    return _rank(values, keys, labels, lambda key: key, sign=sign)


def _resolve_simulated_means(frame, sim_diagnostics, *candidates):
    """The means actually simulated, preferring what simulation reported."""
    lookup = _sim_lookup(sim_diagnostics)
    if lookup:
        ids = frame["game_id"].to_numpy()
        home = np.array([lookup.get(int(g), {}).get("mu_home_used", np.nan) for g in ids])
        away = np.array([lookup.get(int(g), {}).get("mu_away_used", np.nan) for g in ids])
        if np.isfinite(home).all() and np.isfinite(away).all():
            return home, away
    for home, away in zip(candidates[0::2], candidates[1::2]):
        if home is not None and away is not None:
            return np.asarray(home, dtype=float), np.asarray(away, dtype=float)
    n = len(frame)
    return np.full(n, np.nan), np.full(n, np.nan)


def _sim_lookup(sim_diagnostics):
    if sim_diagnostics is None or len(sim_diagnostics) == 0:
        return {}
    return {
        int(record["game_id"]): record
        for record in sim_diagnostics.to_dict(orient="records")
    }


def _column(frame, name, n):
    if frame is None or name not in getattr(frame, "columns", []):
        return np.zeros(n)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _at(values, i):
    if values is None:
        return float("nan")
    return float(np.asarray(values, dtype=float)[i])


def _mask_at(mask, i):
    if mask is None:
        return False
    try:
        return bool(np.asarray(mask)[i])
    except (IndexError, TypeError):
        return False


def _frame_value(frame, column, i):
    if column not in frame.columns:
        return float("nan")
    try:
        return float(pd.to_numeric(frame.at[i, column], errors="coerce"))
    except (TypeError, ValueError):
        return float("nan")


def _outcome_int(outcomes, column, game_id):
    if outcomes is None or column not in getattr(outcomes, "columns", []):
        return 0
    match = outcomes.loc[outcomes["game_id"] == game_id, column]
    if match.empty:
        return 0
    try:
        return int(match.iloc[0])
    except (TypeError, ValueError):
        return 0


def _value_detail(frame, name, i):
    if name not in frame.columns:
        return ""
    value = frame.at[i, name]
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return f"{name} = {value}"
    return f"{name} = {float(number):.4g}"
