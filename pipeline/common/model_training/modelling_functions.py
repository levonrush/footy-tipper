import pandas as pd
import sqlite3
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous, Categorical
from joblib import dump
import dill as pickle

def get_training_data(db_path, sql_file):
    """
    Retrieve training data from an SQLite database.
    
    Args:
        db_path (Path): The path to the SQLite database.
        sql_file (str): The path to the SQL file that contains the query.
        
    Returns:
        DataFrame: The DataFrame containing training data.
    """
    
    # Connect to the SQLite database
    con = sqlite3.connect(str(db_path))

    # Read SQL query from external SQL file
    with open(sql_file, 'r') as file:
        query = file.read()

    training_data = pd.read_sql_query(query, con)

    # Close the connection
    con.close()
    
    return training_data


def create_pipeline(estimator, param_grid, use_rfe, num_folds, opt_metric, cat_cols):
    """
    Create a pipeline that includes pre-processing, feature selection and hyperparameter tuning.
    
    Args:
        estimator (object): The estimator algorithm to use.
        param_grid (dict): The hyperparameters to tune for the estimator.
        use_rfe (bool): Whether to use recursive feature elimination.
        num_folds (int): The number of cross-validation folds.
        opt_metric (str): The metric to optimize.
        cat_cols (list): The categorical columns in the data.
        
    Returns:
        pipeline (Pipeline): The constructed pipeline.
    """
    # Create a list of pipeline steps
    pipeline_steps = []

    # Add one-hot encoding step
    preprocessor = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)], remainder='passthrough')
    pipeline_steps.append(('one_hot_encoder', preprocessor))

    # If use_rfe is True, add feature elimination step
    if use_rfe:
        pipeline_steps.append(('feature_elimination', RFECV(estimator=estimator, cv=num_folds, scoring=opt_metric)))

    # Add hyperparameter tuning step
    pipeline_steps.append(('hyperparamtuning', GASearchCV(
        estimator=estimator, 
        param_grid=param_grid, 
        cv=num_folds, 
        scoring=opt_metric, 
        n_jobs=-1,
        population_size=200,
        generations=100, 
        crossover_probability=0.5, 
        mutation_probability=0.2, 
        verbose=True
        )))

    pipeline = Pipeline(pipeline_steps)
    return pipeline

def train_model_pipeline(data, predictors, outcome_var, estimator, param_grid, use_rfe, num_folds, opt_metric):
    """
    Train the model using the given pipeline.
    
    Args:
        data (DataFrame): The dataset to train on.
        predictors (list): The feature columns in the data.
        outcome_var (str): The target variable.
        estimator (object): The estimator algorithm to use.
        param_grid (dict): The hyperparameters to tune for the estimator.
        use_rfe (bool): Whether to use recursive feature elimination.
        num_folds (int): The number of cross-validation folds.
        opt_metric (str): The metric to optimize.
        
    Returns:
        pipeline (Pipeline): The trained pipeline.
    """

    # Print the type of model that's been trained
    print(f"\nModel training: {type(estimator).__name__}")

    # Split the data into features (X) and target (y)
    X = data[predictors].copy()
    y = data[outcome_var].copy()

    # Detect categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Create a pipeline
    pipeline = create_pipeline(estimator, param_grid, use_rfe, num_folds, opt_metric, cat_cols)

    # Fit the pipeline
    pipeline.fit(X, y)
    
    if use_rfe:
        # Print the number of features selected
        num_features_selected = pipeline.named_steps['feature_elimination'].n_features_
        print(f"Number of features selected: {num_features_selected}")
    
    # Print the best parameters and best score
    print(f"Best parameters: {pipeline.named_steps['hyperparamtuning'].best_params_}")
    print(f"Best score: {pipeline.named_steps['hyperparamtuning'].best_score_}\n")

    return  pipeline


def train_and_select_best_model(data, predictors, outcome_var, use_rfe, num_folds, opt_metric):
    """
    Train multiple models and select the best one.
    
    Args:
        data (DataFrame): The dataset to train on.
        predictors (list): The feature columns in the data.
        outcome_var (str): The target variable.
        use_rfe (bool): Whether to use recursive feature elimination.
        num_folds (int): The number of cross-validation folds.
        opt_metric (str): The metric to optimize.
        
    Returns:
        best_pipeline (Pipeline): The best model pipeline.
    """

    # Define your models and parameter grids
    models_and_params = [
        (xgb.XGBRegressor(objective='count:poisson', n_jobs=-1), {
            'n_estimators': Integer(20, 500),
            'learning_rate': Continuous(0.01, 0.9),
            'max_depth': Integer(2, 20),
            'subsample': Continuous(0.1, 1.0),
            'colsample_bytree': Continuous(0.1, 0.99),
            'gamma': Continuous(0, 0.9),
        })
    ]
    
    best_pipeline = None
    best_score = -float('inf')
    
    # Train each model and keep track of the best one
    for estimator, param_grid in models_and_params:
        pipeline = train_model_pipeline(
            data, predictors, outcome_var,
            estimator, param_grid,
            use_rfe=use_rfe, num_folds=num_folds,
            opt_metric=opt_metric
        )

        score = pipeline.named_steps['hyperparamtuning'].best_score_

        # Update best_model, best_score, etc.
        if score > best_score:
            best_pipeline = pipeline
            best_score = score

    # Print the best model and its score at the end
    print(f"Best overall model: {type(best_pipeline.named_steps['hyperparamtuning'].estimator).__name__}")
    print(f"Best overall score: {best_pipeline.named_steps['hyperparamtuning'].best_score_}")
            
    return best_pipeline

def save_models(pipeline, name, project_root):
    """
    Save the Pipeline objects for future use.
    
    Args:
        pipeline (Pipeline): The trained Pipeline.
        name (str): The name to save the Pipeline as.
        project_root (Path): The root path of the project.
        
    Returns:
        None
    """
    
    # Save the pipeline
    with open(project_root / "models" / f'{name}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to models/{name}.pkl")
