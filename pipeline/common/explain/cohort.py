"""Cohort-level attribution: what the model uses, and where that helps.

Three analyses ship here, all computable from contributions alone:

A. family_signal   which families carry directional skill, not just volume
B. dead_features   which predictors are provably or effectively unused
E. coverage_gaps   where the model has nothing to say, i.e. where a new
                   dataset would actually buy something

Two analyses deliberately live elsewhere (disagreement, confidently-wrong):
they compare the model against outcomes it may have been fitted on, so they are
only trustworthy on genuinely out-of-fold contributions.

Every result carries a ``source`` label. In-sample contributions overstate how
useful a family looks, and a table that does not say which it is will be read
as honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.common.explain import families as fam
from pipeline.common.explain import units

# Probability contributions are reported in percentage points, because that is
# the unit the tips are read in ("78%"), not in raw probability.
PCT = units.PCT_SCALE

SOURCE_IN_SAMPLE = "in-sample-deployed"
SOURCE_NESTED_OOF = "evaluate-nested-season-out"

# Contributions below this are numerically indistinguishable from unused.
_ZERO = 1e-9

# Every feature family that can be wholly absent carries a missingness flag.
# Cross-tabbing contribution against the flag shows whether the model actually
# uses the family when it is present, or has learned to lean on the flag.
FAMILY_MISSING_FLAGS = {
    "team_match_stats": "performance_features_missing",
    "lineup": "lineup_features_missing",
    "player_form": "lineup_form_missing_home",
    "recent_form_stats": "form_features_missing_home",
    "referee": "ref_missing",
    "weather": "weather_missing",
    "travel_rest": "travel_missing",
    "venue_crowd": "crowd_features_missing",
}


@dataclass(frozen=True)
class CohortInputs:
    """Contributions plus the outcomes needed to score them.

    Assembled either from the deployed models over all history (fast, in-sample)
    or from the nested season-out evaluate loop (slow, honest).
    """

    source: str
    feature_names: tuple
    prob_logit: np.ndarray  # (n, n_features) published-logit contributions
    prob_base: np.ndarray  # (n,) published-logit base value
    home_log_mu: np.ndarray  # (n, n_features) log-mean contributions
    away_log_mu: np.ndarray
    home_base: np.ndarray
    away_base: np.ndarray
    p_model: np.ndarray  # (n,) conditional home-win probability
    y: np.ndarray  # (n,) 1 when the home team won
    non_draw: np.ndarray  # (n,) bool
    mu_home: np.ndarray
    mu_away: np.ndarray
    actual_margin: np.ndarray
    # NaN wherever no usable H2H price existed; never a fabricated 0.5.
    market_prob: np.ndarray = None
    split_counts: dict = field(default_factory=dict)
    frame: pd.DataFrame = None  # raw predictor values, for missingness cross-tabs
    meta: dict = field(default_factory=dict)

    @property
    def n_games(self) -> int:
        return int(len(self.y))

    @property
    def honest(self) -> bool:
        return self.source == SOURCE_NESTED_OOF

    def prob_points(self) -> np.ndarray:
        """Per-feature contribution in percentage points of win probability."""
        return PCT * units.prob_points(self.prob_logit, self.p_model[:, None])

    def margin_points(self) -> np.ndarray:
        """Per-feature contribution in points of predicted margin."""
        return units.margin_points(
            self.home_log_mu, self.mu_home[:, None], self.away_log_mu, self.mu_away[:, None]
        )


def wilson_interval(successes, n, z=1.96):
    """Binomial CI that stays sane at small n, unlike the normal approximation."""
    n = int(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_loss(y, p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _direction(lo, hi, n_speaks):
    """Verdict from the agreement CI: helps, hurts, or nothing demonstrated."""
    if not n_speaks or math.isnan(lo) or math.isnan(hi):
        return "silent"
    if lo > 0.5:
        return "helps"
    if hi < 0.5:
        return "hurts"
    return "unclear"


def _family_indices(feature_names):
    mapping = fam.family_map(feature_names)
    grouped = {}
    for i, name in enumerate(feature_names):
        grouped.setdefault(mapping[name], []).append(i)
    return grouped


def family_signal(inputs: CohortInputs) -> pd.DataFrame:
    """Per family: how loudly it speaks, and whether speaking helps.

    ``mean_abs_prob_points`` is volume. ``lift_log_loss`` is skill: the family's
    contribution is added to the model's base rate on its own and scored, so a
    positive lift means the family moves probability toward the truth. A loud
    family with a lift indistinguishable from zero is noise the model is paying
    variance for.
    """
    grouped = _family_indices(inputs.feature_names)
    prob_points = inputs.prob_points()
    margin_points = inputs.margin_points()
    # Family-level denominator, so shares across families sum to one.
    total_abs_prob = np.sum(
        [np.abs(prob_points[:, idx].sum(axis=1)) for idx in grouped.values()], axis=0
    )

    nd = inputs.non_draw.astype(bool)
    y_nd = inputs.y[nd]
    p_base = units.sigmoid(inputs.prob_base[nd])
    base_loss = _log_loss(y_nd, p_base)

    base_margin = np.exp(inputs.home_base) - np.exp(inputs.away_base)
    base_mae = float(np.mean(np.abs(inputs.actual_margin - base_margin)))

    rows = []
    for family, idx in grouped.items():
        fam_prob_points = prob_points[:, idx].sum(axis=1)
        fam_margin_points = margin_points[:, idx].sum(axis=1)
        fam_logit = inputs.prob_logit[:, idx].sum(axis=1)

        # Skill: score this family's contribution alone against the base rate.
        p_family = units.sigmoid(inputs.prob_base[nd] + fam_logit[nd])
        lift_log_loss = base_loss - _log_loss(y_nd, p_family)

        family_margin = np.exp(
            inputs.home_base + inputs.home_log_mu[:, idx].sum(axis=1)
        ) - np.exp(inputs.away_base + inputs.away_log_mu[:, idx].sum(axis=1))
        lift_margin_mae = base_mae - float(
            np.mean(np.abs(inputs.actual_margin - family_margin))
        )

        # Direction: does the family push toward the side that actually won?
        speaks = nd & (np.abs(fam_logit) > _ZERO)
        agree = int(
            np.sum((fam_logit[speaks] > 0) == (inputs.y[speaks] > 0.5))
        )
        n_speaks = int(speaks.sum())
        lo, hi = wilson_interval(agree, n_speaks)

        rows.append(
            {
                "family": family,
                "label": fam.family_label(family),
                "n_features": len(idx),
                "mean_abs_prob_points": float(np.mean(np.abs(fam_prob_points))),
                "share_abs_prob": float(
                    np.mean(
                        np.divide(
                            np.abs(fam_prob_points),
                            total_abs_prob,
                            out=np.zeros_like(fam_prob_points),
                            where=total_abs_prob > _ZERO,
                        )
                    )
                ),
                "signed_mean_prob_points": float(np.mean(fam_prob_points)),
                "lift_log_loss": float(lift_log_loss),
                "agreement_rate": (agree / n_speaks) if n_speaks else float("nan"),
                "agreement_lo": lo,
                "agreement_hi": hi,
                "agreement_n": n_speaks,
                "mean_abs_margin_points": float(np.mean(np.abs(fam_margin_points))),
                "lift_margin_mae": float(lift_margin_mae),
                # "unclear" is the important verdict: a CI straddling 0.5 means
                # the family has no demonstrated directional value, however
                # loudly it speaks. "hurts" means it points at the loser more
                # often than chance, which magnitude alone would never reveal.
                "direction": _direction(lo, hi, n_speaks),
            }
        )

    return pd.DataFrame(rows).sort_values("lift_log_loss", ascending=False).reset_index(
        drop=True
    )


def dead_features(inputs: CohortInputs, *, soft_threshold=0.05, rare_rate=0.10,
                  strong_points=1.0) -> dict:
    """Three tiers of unused, because each implies a different action.

    never_split     provably unused by every booster: safe to delete outright
    soft_dead       split at least once but never moves anything: prune candidates
    rare_but_strong low coverage, high peak: niche work on a handful of games,
                    and exactly the features a volume-based cut would destroy
    """
    prob_points = np.abs(inputs.prob_points())
    margin_points = np.abs(inputs.margin_points())
    nonzero = (np.abs(inputs.prob_logit) > _ZERO) | (
        (np.abs(inputs.home_log_mu) > _ZERO) | (np.abs(inputs.away_log_mu) > _ZERO)
    )

    per_feature = pd.DataFrame(
        {
            "feature": list(inputs.feature_names),
            "family": [fam.family_for(name) for name in inputs.feature_names],
            "side": [fam.side_for(name) for name in inputs.feature_names],
            "split_count": [
                int(inputs.split_counts.get(name, 0)) for name in inputs.feature_names
            ],
            "nonzero_rate": nonzero.mean(axis=0),
            "mean_abs_prob_points": prob_points.mean(axis=0),
            "max_abs_prob_points": prob_points.max(axis=0),
            "q99_abs_prob_points": np.quantile(prob_points, 0.99, axis=0),
            "mean_abs_margin_points": margin_points.mean(axis=0),
            "q99_abs_margin_points": np.quantile(margin_points, 0.99, axis=0),
        }
    )

    never = per_feature["split_count"] == 0
    quiet = (per_feature["q99_abs_prob_points"] < soft_threshold) & (
        per_feature["q99_abs_margin_points"] < soft_threshold
    )
    rare = (per_feature["nonzero_rate"] < rare_rate) & (
        per_feature["max_abs_prob_points"] >= strong_points
    )

    per_feature["tier"] = np.select(
        [never, quiet & ~never, rare & ~never & ~quiet],
        ["never_split", "soft_dead", "rare_but_strong"],
        default="active",
    )

    by_family = (
        per_feature.groupby("family")["tier"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["never_split", "soft_dead", "rare_but_strong", "active"], fill_value=0)
        .reset_index()
    )

    return {
        "source": inputs.source,
        "n_features": int(len(per_feature)),
        "never_split": sorted(per_feature.loc[never, "feature"].tolist()),
        "soft_dead": sorted(per_feature.loc[quiet & ~never, "feature"].tolist()),
        "rare_but_strong": per_feature.loc[rare & ~never & ~quiet]
        .sort_values("max_abs_prob_points", ascending=False)["feature"]
        .tolist(),
        "per_feature": per_feature.sort_values(
            "mean_abs_prob_points", ascending=False
        ).reset_index(drop=True),
        "by_family": by_family,
    }


def coverage_gaps(inputs: CohortInputs, *, quantiles=5) -> dict:
    """Where the model is wrong and has nothing to say about it.

    A high-error cohort in which no family contributes much is a data gap: the
    model is not making a bad call, it is making a call with no information. A
    high-error cohort with loud attribution is a modelling problem instead. The
    two want completely different fixes, which is why they are separated here.
    """
    predicted_margin = inputs.mu_home - inputs.mu_away
    residual = np.abs(inputs.actual_margin - predicted_margin)
    total_margin_attr = np.abs(inputs.margin_points()).sum(axis=1)
    total_prob_attr = np.abs(inputs.prob_points()).sum(axis=1)

    finite = np.isfinite(residual)
    buckets = np.full(len(residual), np.nan)
    if finite.any():
        buckets[finite] = pd.qcut(
            residual[finite], quantiles, labels=False, duplicates="drop"
        )
    residual_table = (
        pd.DataFrame(
            {
                "bucket": buckets,
                "residual": residual,
                "total_abs_margin_points": total_margin_attr,
                "total_abs_prob_points": total_prob_attr,
            }
        )
        .groupby("bucket", observed=True)
        .agg(
            games=("residual", "size"),
            mean_residual=("residual", "mean"),
            mean_abs_margin_attribution=("total_abs_margin_points", "mean"),
            mean_abs_prob_attribution=("total_abs_prob_points", "mean"),
        )
        .reset_index()
    )

    grouped = _family_indices(inputs.feature_names)
    prob_points = inputs.prob_points()

    missingness = []
    if inputs.frame is not None:
        for family, flag in FAMILY_MISSING_FLAGS.items():
            if family not in grouped or flag not in inputs.frame.columns:
                continue
            flag_values = pd.to_numeric(
                inputs.frame[flag], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            missing = flag_values > 0.5
            if missing.all() or (~missing).all():
                continue
            fam_abs = np.abs(prob_points[:, grouped[family]].sum(axis=1))
            present_mean = float(fam_abs[~missing].mean())
            missing_mean = float(fam_abs[missing].mean())
            missingness.append(
                {
                    "family": family,
                    "label": fam.family_label(family),
                    "flag": flag,
                    "games_present": int((~missing).sum()),
                    "games_missing": int(missing.sum()),
                    "mean_abs_prob_points_present": present_mean,
                    "mean_abs_prob_points_missing": missing_mean,
                    "uses_when_present": present_mean - missing_mean,
                }
            )

    sides = [fam.side_for(name) for name in inputs.feature_names]
    margin_points = inputs.margin_points()
    side_rows = []
    for side in ("home", "away", "delta", "neutral"):
        idx = [i for i, value in enumerate(sides) if value == side]
        if not idx:
            continue
        side_rows.append(
            {
                "side": side,
                "n_features": len(idx),
                "mean_abs_prob_points": float(
                    np.mean(np.abs(prob_points[:, idx]).sum(axis=1))
                ),
                "mean_abs_margin_points": float(
                    np.mean(np.abs(margin_points[:, idx]).sum(axis=1))
                ),
            }
        )

    return {
        "source": inputs.source,
        "residual_buckets": residual_table,
        "missingness": pd.DataFrame(missingness),
        "side_balance": pd.DataFrame(side_rows),
    }


class FoldCollector:
    """Capture per-fold TreeSHAP during evaluate's expanding-window OOF loop.

    Passed as `on_fold` to the OOF generators, so contributions come from the
    same fold models that produced the out-of-fold predictions. No extra model
    is fitted; the only added work is one pred_contrib call per fold.

    The raw-feature layout comes from the fitted preprocessor, which is exact,
    rather than from parsing the booster's feature names. Each fold now fits its
    own encoder so it cannot see its test season's categories, so the one-hot
    widths differ per fold while the raw predictor list does not. The reduceat
    offsets are therefore rebuilt per fold and the raw layout is asserted
    unchanged, which keeps every fold writing into the same columns here.
    """

    def __init__(self, preprocessor_steps, n_rows):
        from pipeline.common.explain import contributions as xc

        self.feature_names, self._widths = xc.onehot_group_map(preprocessor_steps)
        self._starts = np.concatenate(([0], np.cumsum(self._widths)[:-1]))
        self.values = np.zeros((int(n_rows), len(self.feature_names)), dtype=float)
        self.base_value = np.zeros(int(n_rows), dtype=float)
        self.captured = np.zeros(int(n_rows), dtype=bool)
        self.split_counts = {name: 0 for name in self.feature_names}

    def _offsets_for(self, fold_preprocessor, test_year):
        """Reduceat offsets for this fold's encoder, in the shared raw layout."""
        if fold_preprocessor is None:
            return self._starts
        from pipeline.common.explain import contributions as xc

        names, widths = xc.onehot_group_map(fold_preprocessor)
        if tuple(names) != tuple(self.feature_names):
            raise ValueError(
                f"fold {test_year}: raw predictor layout changed between folds"
            )
        return np.concatenate(([0], np.cumsum(widths)[:-1]))

    def __call__(self, fold_model, X_test_t, test_mask, test_year, fold_preprocessor=None):
        booster = fold_model.booster_
        contrib = np.asarray(booster.predict(X_test_t, pred_contrib=True), dtype=float)
        rows = np.flatnonzero(np.asarray(test_mask, dtype=bool))
        if len(rows) != len(contrib):
            raise ValueError(
                f"fold {test_year}: {len(rows)} masked rows but {len(contrib)} predictions"
            )
        starts = self._offsets_for(fold_preprocessor, test_year)
        self.values[rows] = np.add.reduceat(contrib[:, :-1], starts, axis=1)
        self.base_value[rows] = contrib[:, -1]
        self.captured[rows] = True

        splits = np.add.reduceat(
            np.asarray(booster.feature_importance("split"), dtype=float), starts
        )
        for name, count in zip(self.feature_names, splits):
            self.split_counts[name] += int(count)


def build_deployed_cohort(project_root, db_path, *, models_dir=None, years=None):
    """Attribute the deployed models over all history. Fast, and in-sample.

    Answers "what does this model use?", which needs no held-out data. It does
    NOT honestly answer "where is it wrong", because the models saw these rows
    during training; run `evaluate --explain` for that.
    """
    import pathlib

    # Reused rather than reimplemented: this is the exact merge sequence
    # train.py performs, and the feature frame has to match it column for
    # column or the attribution describes a different model than the deployed one.
    from pipeline.evaluate import _load_training_frame
    from pipeline.common.model_prediciton import prediction_functions as pf
    from pipeline.common.model_training import tier_a_baseline as tb
    from pipeline.common.model_training import training_config as tc
    from pipeline.common.explain import contributions as xc
    from pipeline.common.explain import trace as xt

    project_root = pathlib.Path(project_root)
    models_dir = pathlib.Path(models_dir) if models_dir else project_root / "models"

    stack = xt.load_probability_stack(models_dir)
    if stack.binary_model is None:
        raise RuntimeError(f"binary_model.pkl not found in {models_dir}")
    home_model = pf.load_models("home_model", project_root, models_dir=models_dir)
    away_model = pf.load_models("away_model", project_root, models_dir=models_dir)

    baseline_payload = stack.manifest.get("tier_a_baseline") or {}
    baseline_cfg = (
        tb.baseline_config_from_dict(baseline_payload)
        if baseline_payload
        else tb.default_baseline_config_from_env()
    )
    data, _ = _load_training_frame(project_root, db_path, baseline_cfg=baseline_cfg)

    if years:
        year = pd.to_numeric(data["competition_year"], errors="coerce")
        data = data[year.isin(list(years))].reset_index(drop=True)
    if data.empty:
        raise RuntimeError("No training rows left after filtering; nothing to attribute.")

    predictors = stack.manifest.get("predictors") or tc.filter_predictors(
        include_performance=tc.include_performance, predictor_list=tc.predictors
    )
    data = tc.align_predictor_columns(data, predictors)
    X = data[predictors]

    prob = xc.raw_contributions(stack.binary_model, X)
    home = xc.raw_contributions(home_model, X)
    away = xc.raw_contributions(away_model, X)
    if not (prob.feature_names == home.feature_names == away.feature_names):
        raise RuntimeError("models disagree on their feature layout; retrain before explaining")

    multiplier = stack.chain_multiplier
    if not xt.is_logit_linear(stack.stacker, stack.calibrator):
        # Legacy stacker/calibrator: the published logit is not a linear
        # function of the expert logit, so say so rather than implying the
        # multiplier below is exact.
        print(
            "WARNING: deployed stack is not a simplex pool with temperature "
            "scaling; probability attribution is approximate."
        )

    home_score = data["team_final_score_home"].to_numpy(dtype=float)
    away_score = data["team_final_score_away"].to_numpy(dtype=float)

    split_counts = {}
    for name in prob.feature_names:
        split_counts[name] = 0
    for pipe in (stack.binary_model, home_model, away_model):
        for name, count in xc.raw_split_counts(pipe).items():
            split_counts[name] = split_counts.get(name, 0) + count

    published_logit = multiplier * (prob.base_value + prob.values.sum(axis=1))

    return CohortInputs(
        source=SOURCE_IN_SAMPLE,
        feature_names=prob.feature_names,
        prob_logit=multiplier * prob.values,
        prob_base=multiplier * prob.base_value,
        home_log_mu=home.values,
        away_log_mu=away.values,
        home_base=home.base_value,
        away_base=away.base_value,
        p_model=units.sigmoid(published_logit),
        y=(home_score > away_score).astype(int),
        non_draw=home_score != away_score,
        mu_home=np.exp(home.prediction_link),
        mu_away=np.exp(away.prediction_link),
        actual_margin=home_score - away_score,
        market_prob=pf.derive_market_home_probability(data),
        split_counts=split_counts,
        frame=data,
        meta={
            "games": int(len(data)),
            "predictor_count": len(predictors),
            "chain_multiplier": float(multiplier),
            "stack": stack.describe(),
            "years": sorted(
                pd.to_numeric(data["competition_year"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            ),
        },
    )


def market_disagreement(inputs: CohortInputs, *, top_fraction=0.10, lead_share=0.30) -> dict:
    """Where the model departs from the market, on whose say-so, and who wins.

    The payoff question. Restricted to the decile of games where the model and
    the market disagree most, it asks which family supplied the departure and
    then scores the model against the market on exactly those games. A family
    with a materially negative edge and enough games behind it is one the model
    would be better off ignoring.

    Only meaningful on out-of-fold contributions: in-sample, every family looks
    like it beats the market.
    """
    if inputs.market_prob is None:
        return {"source": inputs.source, "available": False, "reason": "no market probabilities"}

    valid = np.isfinite(inputs.market_prob) & inputs.non_draw.astype(bool)
    if valid.sum() < 20:
        return {"source": inputs.source, "available": False, "reason": "too few market games"}

    deviation = np.full(len(inputs.y), np.nan)
    deviation[valid] = units.logit(inputs.p_model[valid]) - units.logit(
        inputs.market_prob[valid]
    )
    cut = np.nanquantile(np.abs(deviation[valid]), 1.0 - top_fraction)
    selected = valid & (np.abs(deviation) >= cut)

    prob_points = inputs.prob_points()
    grouped = _family_indices(inputs.feature_names)
    sign = np.sign(deviation)

    # Denominator is the family-level total, not the per-feature one. Within a
    # family, opposing feature contributions cancel, so a per-feature sum is a
    # much larger number and would make every family's share look negligible.
    family_points = {family: prob_points[:, idx].sum(axis=1) for family, idx in grouped.items()}
    total = np.sum([np.abs(values) for values in family_points.values()], axis=0)

    rows = []
    for family, idx in grouped.items():
        fam_points = family_points[family]
        signed_share = np.divide(
            fam_points * sign,
            total,
            out=np.zeros_like(fam_points),
            where=np.isfinite(total) & (total > _ZERO),
        )
        leads = selected & (signed_share > lead_share)
        n_leads = int(leads.sum())
        row = {
            "family": family,
            "label": fam.family_label(family),
            "deviation_share": float(np.mean(signed_share[selected])) if selected.any() else 0.0,
            "games_leading": n_leads,
        }
        if n_leads >= 10:
            model_acc = float(np.mean((inputs.p_model[leads] > 0.5) == (inputs.y[leads] > 0.5)))
            market_acc = float(
                np.mean((inputs.market_prob[leads] > 0.5) == (inputs.y[leads] > 0.5))
            )
            row.update(
                {
                    "model_accuracy": model_acc,
                    "market_accuracy": market_acc,
                    "edge_when_family_leads": model_acc - market_acc,
                    "model_log_loss": _log_loss(inputs.y[leads], inputs.p_model[leads]),
                    "market_log_loss": _log_loss(inputs.y[leads], inputs.market_prob[leads]),
                }
            )
        else:
            row.update(
                {
                    "model_accuracy": float("nan"),
                    "market_accuracy": float("nan"),
                    "edge_when_family_leads": float("nan"),
                    "model_log_loss": float("nan"),
                    "market_log_loss": float("nan"),
                }
            )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("edge_when_family_leads", ascending=False)
    return {
        "source": inputs.source,
        "available": True,
        "honest": inputs.honest,
        "market_games": int(valid.sum()),
        "disagreement_games": int(selected.sum()),
        "overall_model_accuracy": float(
            np.mean((inputs.p_model[selected] > 0.5) == (inputs.y[selected] > 0.5))
        ),
        "overall_market_accuracy": float(
            np.mean((inputs.market_prob[selected] > 0.5) == (inputs.y[selected] > 0.5))
        ),
        "families": table.reset_index(drop=True),
    }


def confidently_wrong(inputs: CohortInputs, *, threshold=0.70, worst=15) -> dict:
    """Which families show up when the model is confident and wrong.

    Confidence threshold matches the green badge in the email and the HIGH band
    in the inference log, so this measures the tips a reader actually trusts.
    Reports the standardized difference rather than the raw one, or the loudest
    families would top the table by construction.
    """
    nd = inputs.non_draw.astype(bool)
    tipped_home = inputs.p_model >= 0.5
    confident = nd & (np.maximum(inputs.p_model, 1.0 - inputs.p_model) >= threshold)
    correct = tipped_home == (inputs.y > 0.5)
    wrong = confident & ~correct
    right = confident & correct

    prob_points = inputs.prob_points()
    sign = np.where(tipped_home, 1.0, -1.0)
    grouped = _family_indices(inputs.feature_names)

    rows = []
    for family, idx in grouped.items():
        toward_tip = prob_points[:, idx].sum(axis=1) * sign
        if wrong.sum() < 5 or right.sum() < 5:
            continue
        mean_wrong = float(np.mean(toward_tip[wrong]))
        mean_right = float(np.mean(toward_tip[right]))
        pooled_sd = float(np.std(toward_tip[confident])) or float("nan")
        rows.append(
            {
                "family": family,
                "label": fam.family_label(family),
                "mean_when_wrong": mean_wrong,
                "mean_when_right": mean_right,
                "difference": mean_wrong - mean_right,
                "standardized": (mean_wrong - mean_right) / pooled_sd
                if pooled_sd == pooled_sd
                else float("nan"),
            }
        )

    worst_games = []
    if wrong.any():
        order = np.argsort(-np.maximum(inputs.p_model, 1.0 - inputs.p_model))
        for i in order:
            if not wrong[i]:
                continue
            drivers = prob_points[i] * sign[i]
            top = np.argsort(-np.abs(drivers))[:3]
            worst_games.append(
                {
                    "row": int(i),
                    "game_id": _meta_game_id(inputs, i),
                    "confidence": float(max(inputs.p_model[i], 1.0 - inputs.p_model[i])),
                    "tipped_home": bool(tipped_home[i]),
                    "home_won": bool(inputs.y[i] > 0.5),
                    "actual_margin": float(inputs.actual_margin[i]),
                    "top_drivers": [
                        {
                            "feature": inputs.feature_names[j],
                            "family": fam.family_for(inputs.feature_names[j]),
                            "points": float(drivers[j]),
                        }
                        for j in top
                    ],
                }
            )
            if len(worst_games) >= worst:
                break

    return {
        "source": inputs.source,
        "honest": inputs.honest,
        "threshold": threshold,
        "confident_games": int(confident.sum()),
        "confident_wrong": int(wrong.sum()),
        "confident_accuracy": float(right.sum() / confident.sum()) if confident.any() else float("nan"),
        "families": pd.DataFrame(rows).sort_values("standardized", ascending=False).reset_index(
            drop=True
        )
        if rows
        else pd.DataFrame(),
        "worst_games": worst_games,
    }


def _meta_game_id(inputs, i):
    if inputs.frame is not None and "game_id" in inputs.frame.columns:
        try:
            return int(inputs.frame["game_id"].iloc[i])
        except Exception:
            return None
    return None


ANALYSES = {
    "families": family_signal,
    "dead": dead_features,
    "coverage": coverage_gaps,
    "disagreement": market_disagreement,
    "confident-wrong": confidently_wrong,
}


def run_analyses(inputs: CohortInputs, which="all") -> dict:
    names = list(ANALYSES) if which in (None, "all") else [which]
    unknown = [name for name in names if name not in ANALYSES]
    if unknown:
        raise ValueError(f"unknown analysis: {unknown[0]}")
    results = {name: ANALYSES[name](inputs) for name in names}
    results["source"] = inputs.source
    results["n_games"] = inputs.n_games
    results["meta"] = dict(inputs.meta)
    return results
