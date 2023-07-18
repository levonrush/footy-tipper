from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous, Categorical
from joblib import dump
import dill as pickle

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
    pipeline_steps.append(('hyperparamtuning', GASearchCV(estimator=estimator, param_grid=param_grid, cv=num_folds, scoring=opt_metric, n_jobs=-1,
    population_size=50, generations=5, crossover_probability=0.5, mutation_probability=0.2, verbose=1)))

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
        le (LabelEncoder): The label encoder.
        pipeline (Pipeline): The trained pipeline.
    """

    # Split the data into features (X) and target (y)
    X = data[predictors].copy()
    y = data[outcome_var].copy()

    # Create and fit a LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Detect categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Create a pipeline
    pipeline = create_pipeline(estimator, param_grid, use_rfe, num_folds, opt_metric, cat_cols)

    # Fit the pipeline
    pipeline.fit(X, y)

    return  le, pipeline


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
        best_label_encoder (LabelEncoder): The label encoder used for the best model.
    """

    # Define your models and parameter grids
    models_and_params = [
        (xgb.XGBClassifier(n_jobs=-1), {
            'n_estimators': Integer(25, 250),
            'learning_rate': Continuous(0.01, 0.5),
            'max_depth': Integer(2, 8),
            'subsample': Continuous(0.8, 1.0),
            'colsample_bytree': Continuous(0.3, 0.7),
            'gamma': Continuous(0, 0.5),
        }),
        (RandomForestClassifier(n_jobs=-1, class_weight='balanced'), {
            'n_estimators': Integer(50, 250),
            'max_features': Categorical(['sqrt', 'log2']),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'bootstrap': Categorical([True, False]),
        }),
        (GradientBoostingClassifier(), {
            'n_estimators': Integer(50, 250),
            'learning_rate': Continuous(0.01, 0.2),
            'max_depth': Integer(3, 5),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'subsample': Continuous(0.8, 1.0),
            'max_features': Categorical(['sqrt', 'log2']),
        })
    ]
    
    best_pipeline = None
    best_score = -float('inf')
    best_label_encoder = None
    
    # Train each model and keep track of the best one
    for estimator, param_grid in models_and_params:
        label_encoder, pipeline = train_model_pipeline(
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
            best_label_encoder = label_encoder
            
    return best_pipeline, best_label_encoder

def save_models(label_encoder, pipeline, project_root):
    """
    Save the LabelEncoder and the Pipeline objects for future use.
    
    Args:
        label_encoder (LabelEncoder): The trained LabelEncoder.
        pipeline (Pipeline): The trained Pipeline.
        project_root (Path): The root path of the project.
        
    Returns:
        None
    """

    # Save the LabelEncoder
    dump(label_encoder, project_root / "models" / 'label_encoder.pkl')
    
    # Save the pipeline
    with open(project_root / "models" / 'footy_tipper.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
