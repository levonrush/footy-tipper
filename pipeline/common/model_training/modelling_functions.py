import os
import pathlib

import pandas as pd
import sqlite3
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import numpy as np
import re
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import dump
import dill as pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

# Import BayesSearchCV and search space objects from scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from pipeline.common import console
from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.model_training.cv import InSeasonSplit

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _make_tuning_callback(total, outcome_var):
    """Emit a live 'candidate N/total · best cv' line as the search progresses.

    skopt minimises the negative CV score, so the best score seen so far is
    ``-min(func_vals)``. Reported through the marker channel so the parent CLI
    can turn it into an in-place progress line.
    """

    def _callback(result):
        try:
            done = len(result.x_iters)
            best = -float(np.min(result.func_vals))
            console.emit_progress(
                f"tuning {outcome_var}: candidate {done}/{total} · best cv {best:.4f}"
            )
        except Exception:
            pass

    return _callback


def select_blend_weights_by_log_loss(y, baseline_mu_home, baseline_mu_away, model_mu_home, model_mu_away):
    """Grid-search (w_home, w_away) minimising log-loss of the conditional win prob.

    All inputs must already be filtered to the rows used for selection
    (typically non-draw, genuinely out-of-fold rows). Log-loss is a smooth
    proper scoring rule, so the argmin is far more stable than maximising
    0/1 tipping accuracy. Returns (w_home, w_away, log_loss, accuracy).
    """
    from sklearn.metrics import log_loss as _log_loss

    y = np.asarray(y, dtype=int)
    bh = np.asarray(baseline_mu_home, dtype=float)
    ba = np.asarray(baseline_mu_away, dtype=float)
    mh = np.asarray(model_mu_home, dtype=float)
    ma = np.asarray(model_mu_away, dtype=float)

    candidates = np.linspace(0.0, 1.0, 11)
    best_wh, best_wa, best_ll = 1.0, 1.0, np.inf
    for wh in candidates:
        blended_h = np.maximum((1.0 - wh) * bh + wh * mh, 1e-6)
        for wa in candidates:
            blended_a = np.maximum((1.0 - wa) * ba + wa * ma, 1e-6)
            win_probs = np.clip(pf.conditional_home_win_prob_vec(blended_h, blended_a), 1e-6, 1 - 1e-6)
            ll = _log_loss(y, win_probs)
            if ll < best_ll:
                best_ll, best_wh, best_wa = float(ll), float(wh), float(wa)

    best_h = np.maximum((1.0 - best_wh) * bh + best_wh * mh, 1e-6)
    best_a = np.maximum((1.0 - best_wa) * ba + best_wa * ma, 1e-6)
    best_probs = pf.conditional_home_win_prob_vec(best_h, best_a)
    best_acc = float(((best_probs > 0.5) == y.astype(bool)).mean())
    return best_wh, best_wa, best_ll, best_acc


def sanitize_feature_names(names):
    """
    Replace any non-alphanumeric or underscore characters with underscore,
    then enforce uniqueness for downstream libraries like LightGBM.
    """
    sanitized = []
    seen = {}
    for name in names:
        base = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        count = seen.get(base, 0)
        if count == 0:
            sanitized_name = base
        else:
            sanitized_name = f"{base}__dup{count}"
        seen[base] = count + 1
        sanitized.append(sanitized_name)
    return sanitized


def get_training_data(db_path, sql_file):
    con = sqlite3.connect(str(db_path))
    with open(sql_file, 'r') as f:
        query = f.read()
    df = pd.read_sql_query(query, con)
    con.close()
    return df


def create_pipeline(estimator, search_spaces, use_rfe, cv, opt_metric, cat_cols):
    """
    Create a pipeline with:
      1) One-hot encoding
      2) Wrapping array output into DataFrame with sanitized feature names
      3) Scaling for MLPRegressor
      4) Optional RFECV
      5) Bayesian hyperparameter search
    """
    steps = []
    # 1) One-hot encode categorical cols
    preprocessor = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )
    steps.append(('one_hot', preprocessor))

    # 2) After fitting preprocessor, wrap into DataFrame with sanitized names
    def to_df(X_array):
        cols = preprocessor.get_feature_names_out(preprocessor.feature_names_in_)
        df   = pd.DataFrame(X_array, columns=sanitize_feature_names(cols))
        # coerce everything to numeric, turn errors into NaN, then fill with 0
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return df

    steps.append(('to_df', FunctionTransformer(func=to_df, validate=False)))

    # 3) Scale numeric features if using MLP
    if isinstance(estimator, MLPRegressor):
        steps.append(('scaler', StandardScaler()))

    # 4) Recursive feature elimination if requested
    if use_rfe:
        steps.append(('rfe', RFECV(estimator=estimator, cv=cv, scoring=opt_metric)))

    # 5) Bayesian hyperparameter tuning
    bayes = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        cv=cv,
        scoring=opt_metric,
        n_jobs=-1,
        verbose=1,
        # Env override lets retrain-A/B cycles run fast without changing defaults.
        n_iter=int(os.getenv("FOOTY_TIPPER_TUNE_ITER", "100"))
    )
    steps.append(('hyperparamtuning', bayes))

    return Pipeline(steps)


def train_model_pipeline(data, predictors, outcome_var,
                         estimator, search_spaces,
                         use_rfe, num_folds, opt_metric):
    print(f"\nTraining {type(estimator).__name__}...")
    df_sorted = data.sort_values(['competition_year','round_id']).reset_index(drop=True)
    X = df_sorted[predictors].copy()
    y = df_sorted[outcome_var].copy()
    groups = df_sorted['competition_year'].values

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    cv = InSeasonSplit(n_splits=num_folds)
    pipeline = create_pipeline(estimator, search_spaces, use_rfe, cv, opt_metric, cat_cols)

    total_candidates = int(os.getenv("FOOTY_TIPPER_TUNE_ITER", "100"))
    tuning_callback = _make_tuning_callback(total_candidates, outcome_var)
    console.emit_progress(f"tuning {outcome_var}: starting {total_candidates} candidates")
    try:
        pipeline.fit(
            X, y,
            hyperparamtuning__groups=groups,
            hyperparamtuning__callback=tuning_callback,
        )
    except TypeError:
        # Older skopt routing that does not accept a fit-time callback.
        pipeline.fit(X, y, hyperparamtuning__groups=groups)

    if use_rfe:
        print(f"Selected features: {pipeline.named_steps['rfe'].n_features_}")
    best = pipeline.named_steps['hyperparamtuning']
    print(f"Best params: {best.best_params_}\nBest score: {best.best_score_}\n")
    return pipeline


def score_regressor():
    """Build the score estimator used inside the parallel Bayesian search."""
    # BayesSearchCV parallelises fits, so each individual LightGBM fit must
    # stay single-threaded to avoid severe nested CPU oversubscription.
    return lgb.LGBMRegressor(objective='poisson', n_jobs=1, verbose=-1)


def train_and_select_best_model(data, predictors, outcome_var,
                                use_rfe, num_folds, opt_metric):
    models_and_spaces = [
        (score_regressor(), {
            'n_estimators': Integer(20, 500),
            'learning_rate': Real(0.01, 0.9, prior='log-uniform'),
            'max_depth': Integer(2, 20),
            'num_leaves': Integer(2, 100),
            'subsample': Real(0.1, 1.0),
            'colsample_bytree': Real(0.1, 0.99),
            'reg_alpha': Real(0, 1),
            'reg_lambda': Real(0, 1)
        }),
    ]
    best_pipe, best_score = None, -np.inf
    for est, spaces in models_and_spaces:
        pipe = train_model_pipeline(data, predictors, outcome_var,
                                    est, spaces, use_rfe, num_folds, opt_metric)
        score = pipe.named_steps['hyperparamtuning'].best_score_
        if score > best_score:
            best_pipe, best_score = pipe, score
    print(f"Best overall model: {type(best_pipe.named_steps['hyperparamtuning'].best_estimator_).__name__}\nScore: {best_score}")
    return best_pipe


def generate_oof_score_predictions(data, predictors, full_pipeline, outcome_var, return_mask=False):
    """Generate out-of-fold score predictions using expanding year windows.

    For each year Y (starting from the 2nd year in data), trains a LightGBM with
    the best hyperparameters from `full_pipeline` on all years < Y, then predicts
    on year Y. The first year has no prior history so falls back to in-sample
    predictions from `full_pipeline`.

    This gives the stacker unbiased (out-of-sample) Tier-B inputs, preventing it
    from over-weighting Tier-B due to in-sample overfitting.

    With `return_mask=True`, also returns a boolean array marking rows whose
    predictions are genuinely out-of-fold (first-year/failed-fold rows are
    in-sample fallbacks and should be excluded from meta-model training).
    """
    years = sorted(
        pd.to_numeric(data["competition_year"], errors="coerce")
        .dropna().astype(int).unique()
    )

    best_params = dict(full_pipeline.named_steps["hyperparamtuning"].best_params_)
    best_estimator = full_pipeline.named_steps["hyperparamtuning"].best_estimator_
    # Slice all steps except the final estimator (one_hot + to_df, already fitted).
    preprocessor_steps = full_pipeline[:-1]

    oof_preds = pd.Series(np.nan, index=data.index, dtype=float)

    for i, test_year in enumerate(years):
        if i == 0:
            continue  # No prior history; will fall back to in-sample below.

        year_col = pd.to_numeric(data["competition_year"], errors="coerce")
        train_mask = year_col < test_year
        test_mask = year_col == test_year

        if train_mask.sum() < 10 or test_mask.sum() == 0:
            continue

        X_train = data.loc[train_mask, predictors]
        y_train = data.loc[train_mask, outcome_var].values
        X_test = data.loc[test_mask, predictors]

        try:
            X_train_t = preprocessor_steps.transform(X_train)
            X_test_t = preprocessor_steps.transform(X_test)

            fold_model = type(best_estimator)(
                objective="poisson", n_jobs=1, verbose=-1, **best_params
            )
            fold_model.fit(X_train_t, y_train)
            oof_preds.loc[test_mask] = np.maximum(fold_model.predict(X_test_t), 1e-6)
        except Exception as exc:
            print(f"OOF generation failed for year {test_year}: {exc}")

    # Fill NaN (first year + any failed folds) with in-sample predictions.
    nan_mask = oof_preds.isna()
    genuine_oof = (~nan_mask).to_numpy()
    if nan_mask.any():
        in_sample = np.maximum(full_pipeline.predict(data[predictors]), 1e-6)
        oof_preds[nan_mask] = in_sample[nan_mask]

    if return_mask:
        return oof_preds.values, genuine_oof
    return oof_preds.values


def train_binary_classifier(data, predictors, outcome_var, best_params, preprocessor_steps):
    """Train a binary LightGBM classifier (objective='binary') on win/loss outcome.

    Reuses the already-fitted preprocessor from the score models and the best
    hyperparameters found by BayesSearchCV, avoiding a second expensive tuning run.
    Returns a full sklearn Pipeline so predict_proba works on raw feature DataFrames.
    """
    df_sorted = data.sort_values(['competition_year', 'round_id']).reset_index(drop=True)
    X = df_sorted[predictors].copy()
    y = df_sorted[outcome_var].values.astype(int)

    X_t = preprocessor_steps.transform(X)

    clf = lgb.LGBMClassifier(objective='binary', n_jobs=1, verbose=-1, **best_params)
    clf.fit(X_t, y)

    # Wrap into a Pipeline so inference can call predict_proba on raw DataFrames.
    from sklearn.pipeline import Pipeline as _Pipeline
    binary_pipeline = _Pipeline(
        list(preprocessor_steps.steps) + [('binary_clf', clf)]
    )
    return binary_pipeline


def generate_oof_binary_predictions(data, non_draw_mask, predictors, preprocessor_steps, best_params, return_mask=False):
    """Generate OOF binary win/loss predictions using an expanding year window.

    Mirrors generate_oof_score_predictions but trains a binary classifier on
    non-draw games only. Returns an array aligned to data.index with P(home win).
    With `return_mask=True`, also returns a boolean genuine-OOF row mask.
    """
    years = sorted(
        pd.to_numeric(data["competition_year"], errors="coerce")
        .dropna().astype(int).unique()
    )

    nd = np.asarray(non_draw_mask, dtype=bool)
    y_col = (
        data["team_final_score_home"].to_numpy(dtype=float)
        > data["team_final_score_away"].to_numpy(dtype=float)
    ).astype(int)

    oof_preds = pd.Series(np.nan, index=data.index, dtype=float)

    for i, test_year in enumerate(years):
        if i == 0:
            continue

        year_col = pd.to_numeric(data["competition_year"], errors="coerce")
        train_mask = (year_col < test_year).values & nd
        test_mask = (year_col == test_year).values & nd

        if train_mask.sum() < 10 or test_mask.sum() == 0:
            continue

        X_train = data.loc[train_mask, predictors]
        y_train = y_col[train_mask]
        X_test = data.loc[test_mask, predictors]

        try:
            X_train_t = preprocessor_steps.transform(X_train)
            X_test_t = preprocessor_steps.transform(X_test)

            fold_clf = lgb.LGBMClassifier(objective='binary', n_jobs=1, verbose=-1, **best_params)
            fold_clf.fit(X_train_t, y_train)
            oof_preds.loc[test_mask] = fold_clf.predict_proba(X_test_t)[:, 1]
        except Exception as exc:
            print(f"OOF binary generation failed for year {test_year}: {exc}")

    # Fallback for first year and any failed folds: train on all non-draw data.
    nan_mask = oof_preds.isna() & nd
    genuine_oof = (oof_preds.notna() & nd).to_numpy()
    if nan_mask.any():
        try:
            X_all_t = preprocessor_steps.transform(data.loc[nd, predictors])
            fallback_clf = lgb.LGBMClassifier(objective='binary', n_jobs=1, verbose=-1, **best_params)
            fallback_clf.fit(X_all_t, y_col[nd])
            fallback_preds = fallback_clf.predict_proba(preprocessor_steps.transform(data.loc[nan_mask, predictors]))[:, 1]
            oof_preds.loc[nan_mask] = fallback_preds
        except Exception as exc:
            print(f"OOF binary fallback failed: {exc}")
            oof_preds = oof_preds.fillna(0.5)

    oof_preds = oof_preds.fillna(0.5)
    if return_mask:
        return oof_preds.values, genuine_oof
    return oof_preds.values


def save_models(pipeline, name, project_root, models_dir=None):
    output_dir = pathlib.Path(models_dir) if models_dir is not None else project_root / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved pipeline to {path}")
