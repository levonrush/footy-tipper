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

from pipeline.common.model_training.cv import InSeasonSplit

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
        n_iter=100
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

    pipeline.fit(X, y, hyperparamtuning__groups=groups)

    if use_rfe:
        print(f"Selected features: {pipeline.named_steps['rfe'].n_features_}")
    best = pipeline.named_steps['hyperparamtuning']
    print(f"Best params: {best.best_params_}\nBest score: {best.best_score_}\n")
    return pipeline


def train_and_select_best_model(data, predictors, outcome_var,
                                use_rfe, num_folds, opt_metric):
    models_and_spaces = [
        (lgb.LGBMRegressor(objective='poisson', n_jobs=-1, verbose=-1), {
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


def save_models(pipeline, name, project_root):
    path = project_root / 'models' / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved pipeline to {path}")
