"""Exact per-prediction feature attribution for the deployed LightGBM models.

Uses LightGBM's native ``pred_contrib=True``, which is exact TreeSHAP. The
``shap`` package would give the same numbers for these models while adding a
dependency, and requirements.txt pins library versions precisely so the model
pickles stay loadable, so we do not add one.

The models are sklearn Pipelines (see modelling_functions.create_pipeline):

    home_model / away_model : [one_hot, to_df, hyperparamtuning -> BayesSearchCV]
    binary_model            : [one_hot, to_df, binary_clf]

so attribution has to walk back from the 804 transformed columns to the 611 raw
predictors. That mapping is built arithmetically from the fitted
ColumnTransformer rather than by parsing "encoder__<col>_<level>" strings:
both column names (broadcast_channel1) and category values (Finals Week 1)
contain underscores and spaces, so string parsing is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.common.model_training.modelling_functions import sanitize_feature_names

LINK_LOG_ODDS = "log_odds"
LINK_LOG_MEAN = "log_mean"


@dataclass(frozen=True)
class RawContributions:
    """Per-row, per-raw-predictor contributions on the model's link scale."""

    feature_names: tuple
    values: np.ndarray  # (n_rows, n_raw)
    base_value: np.ndarray  # (n_rows,)
    link: str

    @property
    def prediction_link(self) -> np.ndarray:
        """Link-scale prediction. Equals base + sum(values) by SHAP additivity."""
        return self.base_value + self.values.sum(axis=1)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=list(self.feature_names))


def final_estimator(pipe):
    """The LightGBM estimator, whatever wrapper the last pipeline step uses."""
    step = pipe.steps[-1][1]
    return getattr(step, "best_estimator_", step)


def booster_of(pipe):
    return final_estimator(pipe).booster_


def link_for(pipe) -> str:
    """log-odds for the classifier, log-mean for the Poisson score models."""
    estimator = final_estimator(pipe)
    objective = str(getattr(estimator, "objective", "") or "")
    if hasattr(estimator, "predict_proba") or "binary" in objective:
        return LINK_LOG_ODDS
    return LINK_LOG_MEAN


def transform_frame(pipe, X_raw) -> pd.DataFrame:
    """Run the preprocessing steps only, giving the booster's input frame."""
    return pipe[:-1].transform(X_raw)


def onehot_group_map(pipe):
    """Map raw predictors to their transformed column blocks.

    Returns (raw_names, widths). The ColumnTransformer lays out the encoder
    block first, one contiguous run of ``len(categories_[i])`` columns per
    categorical predictor, then the remainder passthrough one column each. The
    partition is therefore contiguous, total and disjoint by construction,
    which is what lets the aggregation below use a single reduceat.
    """
    ct = pipe.named_steps["one_hot"]
    encoder = ct.named_transformers_["encoder"]
    feature_names_in = list(ct.feature_names_in_)

    cat_cols = [str(col) for col in ct.transformers_[0][2]]
    raw_names = list(cat_cols)
    widths = [len(levels) for levels in encoder.categories_]

    remainder_spec = ct.transformers_[1][2]
    for col in remainder_spec:
        # sklearn reports remainder columns as names or as positional indices
        # depending on version; accept both.
        name = feature_names_in[col] if isinstance(col, (int, np.integer)) else str(col)
        raw_names.append(name)
        widths.append(1)

    return tuple(raw_names), np.asarray(widths, dtype=int)


def verify_feature_alignment(pipe) -> None:
    """Assert the transformed-space mapping matches the booster exactly.

    Cheap enough to run once per model load, and it is the single assumption
    every attribution number rests on.
    """
    ct = pipe.named_steps["one_hot"]
    booster = booster_of(pipe)
    expected = sanitize_feature_names(ct.get_feature_names_out(ct.feature_names_in_))
    actual = booster.feature_name()
    if list(expected) != list(actual):
        raise ValueError(
            "transformed feature names do not match the booster "
            f"({len(expected)} built vs {len(actual)} in model)"
        )
    _, widths = onehot_group_map(pipe)
    if int(widths.sum()) != booster.num_feature():
        raise ValueError(
            f"one-hot group widths sum to {int(widths.sum())} but the booster "
            f"has {booster.num_feature()} features"
        )


def _group_starts(widths: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(widths)[:-1]))


def raw_contributions(pipe, X_raw, *, link=None, chunk_rows=1024, verify=True):
    """Exact TreeSHAP contributions aggregated to raw predictors.

    A categorical predictor's contribution is the sum over its one-hot columns.
    That is exact, not an approximation: SHAP values are additive on the link
    scale, and rows whose level was unseen at fit time (handle_unknown='ignore')
    simply contribute an all-zero block.
    """
    if verify:
        verify_feature_alignment(pipe)

    booster = booster_of(pipe)
    raw_names, widths = onehot_group_map(pipe)
    starts = _group_starts(widths)
    link = link or link_for(pipe)

    n_rows = len(X_raw)
    values = np.empty((n_rows, len(raw_names)), dtype=float)
    base_value = np.empty(n_rows, dtype=float)

    # Aggregate inside the loop: the transformed matrix is 804 wide, the raw one
    # 611, and holding the full transformed contribution matrix for the whole
    # corpus is the only part of this that would be large.
    for start in range(0, n_rows, max(int(chunk_rows), 1)):
        stop = min(start + max(int(chunk_rows), 1), n_rows)
        transformed = transform_frame(pipe, X_raw.iloc[start:stop])
        contrib = booster.predict(transformed, pred_contrib=True)
        contrib = np.asarray(contrib, dtype=float)
        if contrib.shape[1] != len(booster.feature_name()) + 1:
            raise ValueError(
                f"unexpected pred_contrib width {contrib.shape[1]}; "
                "multi-output boosters are not supported here"
            )
        values[start:stop] = np.add.reduceat(contrib[:, :-1], starts, axis=1)
        base_value[start:stop] = contrib[:, -1]

    return RawContributions(
        feature_names=raw_names,
        values=values,
        base_value=base_value,
        link=link,
    )


def raw_split_counts(pipe) -> dict:
    """Split count per raw predictor. Zero means the model provably never used it."""
    booster = booster_of(pipe)
    raw_names, widths = onehot_group_map(pipe)
    starts = _group_starts(widths)
    splits = np.asarray(booster.feature_importance("split"), dtype=float)
    grouped = np.add.reduceat(splits, starts)
    return {name: int(count) for name, count in zip(raw_names, grouped)}


def align_to(contribs: RawContributions, feature_names) -> np.ndarray:
    """Reorder a contribution matrix onto a caller-supplied predictor order."""
    index = {name: i for i, name in enumerate(contribs.feature_names)}
    missing = [name for name in feature_names if name not in index]
    if missing:
        raise KeyError(f"{len(missing)} predictor(s) absent from contributions: {missing[:5]}")
    return contribs.values[:, [index[name] for name in feature_names]]
