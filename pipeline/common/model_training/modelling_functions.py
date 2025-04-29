import pandas as pd
import sqlite3
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import xgboost as xgb
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

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_training_data(db_path, sql_file):
    """
    Retrieve training data from an SQLite database.
    """
    con = sqlite3.connect(str(db_path))
    with open(sql_file, 'r') as file:
        query = file.read()
    training_data = pd.read_sql_query(query, con)
    con.close()
    return training_data

def create_pipeline(estimator, search_spaces, use_rfe, num_folds, opt_metric, cat_cols):
    """
    Create a pipeline that includes pre-processing, feature selection and Bayesian hyperparameter tuning.
    
    Args:
        estimator: The estimator to tune.
        search_spaces (dict): The hyperparameter search space (using skopt.space objects).
        use_rfe (bool): Whether to include recursive feature elimination.
        num_folds (int): Number of cross-validation folds.
        opt_metric (str): The metric to optimize.
        cat_cols (list): List of categorical column names.
        
    Returns:
        Pipeline: The constructed pipeline.
    """
    pipeline_steps = []
    
    # One-hot encode categorical columns
    preprocessor = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)], remainder='passthrough')
    pipeline_steps.append(('one_hot_encoder', preprocessor))
    
    # Add scaling for MLPRegressor
    if isinstance(estimator, MLPRegressor):
        pipeline_steps.append(('scaler', StandardScaler()))
    
    # Add RFECV if requested
    if use_rfe:
        pipeline_steps.append(('feature_elimination', RFECV(estimator=estimator, cv=num_folds, scoring=opt_metric)))
    
    # Add Bayesian hyperparameter tuning using BayesSearchCV
    pipeline_steps.append(('hyperparamtuning', BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        cv=num_folds,
        scoring=opt_metric,
        n_jobs=-1,
        verbose=1,
        n_iter=100  # Adjust the number of iterations as needed
    )))
    
    pipeline = Pipeline(pipeline_steps)
    return pipeline

def train_model_pipeline(data, predictors, outcome_var, estimator, search_spaces, use_rfe, num_folds, opt_metric):
    """
    Train the model using the pipeline with Bayesian hyperparameter tuning.
    """
    print(f"\nModel training: {type(estimator).__name__}")
    
    X = data[predictors].copy()
    y = data[outcome_var].copy()
    
    # Detect categorical columns in X
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create pipeline with BayesSearchCV
    pipeline = create_pipeline(estimator, search_spaces, use_rfe, num_folds, opt_metric, cat_cols)
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    if use_rfe:
        num_features_selected = pipeline.named_steps['feature_elimination'].n_features_
        print(f"Number of features selected: {num_features_selected}")
    
    # Report best parameters and score
    best_params = pipeline.named_steps['hyperparamtuning'].best_params_
    best_score = pipeline.named_steps['hyperparamtuning'].best_score_
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}\n")
    
    return pipeline

def train_and_select_best_model(data, predictors, outcome_var, use_rfe, num_folds, opt_metric):
    """
    Train multiple models (with Bayesian optimization) and select the best one.
    """
    # Define models and their search spaces (example using LightGBM)
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
        # Add additional models and search spaces as needed.
    ]
    
    best_pipeline = None
    best_score = -float('inf')
    
    for estimator, search_spaces in models_and_spaces:
        pipeline = train_model_pipeline(
            data, predictors, outcome_var,
            estimator, search_spaces,
            use_rfe=use_rfe, num_folds=num_folds,
            opt_metric=opt_metric
        )
        score = pipeline.named_steps['hyperparamtuning'].best_score_
        if score > best_score:
            best_pipeline = pipeline
            best_score = score
    
    print(f"Best overall model: {type(best_pipeline.named_steps['hyperparamtuning'].estimator).__name__}")
    print(f"Best overall score: {best_pipeline.named_steps['hyperparamtuning'].best_score_}")
    return best_pipeline

def save_models(pipeline, name, project_root):
    """
    Save the trained pipeline to a pickle file.
    """
    model_path = project_root / "models" / f'{name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to models/{name}.pkl")