# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import class_weight
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer

# Other libraries
import xgboost as xgb
import pandas as pd
import numpy as np

# This function first one-hot encodes the categorical variables in the dataset.
def one_hot_encode_and_split(data, predictors, outcome_var):

    # One hot encode categorical variables
    X = data[predictors].copy()
    object_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=object_columns)
    
    # Getting updated predictors
    updated_predictors = X.columns.tolist()

    # Encode the label if it's a categorical variable
    y = data[outcome_var].copy()
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    return X, y, updated_predictors, label_encoder

# This function performs Recursive Feature Elimination (RFE) using cross-validation on the training set.
# The goal of RFE is to select features by recursively considering smaller and smaller sets of features.
def feature_selection(estimator, X, y, num_folds, opt_metric):

    # Scoring metric
    if opt_metric == 'ROC':
        scoring = make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)
    elif opt_metric == 'F1':
        scoring = 'f1'
    else:
        scoring = 'accuracy'

    cv=StratifiedKFold(num_folds)

    # Recursive Feature Elimination with cross-validation
    rfecv = RFECV(estimator=estimator, step=1, n_jobs=-1, cv=cv, scoring=scoring)
    rfecv.fit(X, y)

    # Optimal set of predictors (one-hot encoded)
    optimal_features = np.array(X.columns)[rfecv.support_]

    # Return the one-hot encoded data and the optimal features
    return X, y, optimal_features

# This function performs hyperparameter tuning using Genetic Search and Cross Validation on the training set.
# The function takes as input the machine learning estimator, parameter grid, training set, optimal features 
# obtained from RFE, number of folds for cross-validation, the optimization metric, and a seed for random state.
def train_tune_model(estimator, param_grid, X, y, optimal_features, num_folds=5, opt_metric='ROC', seed=69):
    
    # Scoring metric
    if opt_metric == 'ROC':
        scoring = make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)
    elif opt_metric == 'F1':
        scoring = 'f1'
    else:
        scoring = 'accuracy'

    # Select only the optimal features from X
    X_optimal = X[optimal_features]

    cv = StratifiedKFold(n_splits=num_folds, shuffle=True)
   
    # Genetic Search with cross-validation    
    evolved_estimator = GASearchCV(
        estimator=estimator,
        cv=cv,
        scoring=scoring,
        population_size=5,
        generations=3,
        tournament_size=2,
        elitism=True,
        crossover_probability=0.7,
        mutation_probability=0.3,
        param_grid=param_grid,
        criteria='max',
        algorithm='eaMuPlusLambda',
        n_jobs=-1,
        verbose=True,
        keep_top_k=2
    )
    
    evolved_estimator.fit(X_optimal, y)
    
    # Print results
    print(evolved_estimator.best_params_)
    print(evolved_estimator.best_score_)

    return evolved_estimator

# This function is a pipeline for training a machine learning model. 
# It first one-hot encodes and splits the data, performs Recursive Feature Elimination (if specified), 
# and then trains and tunes the model.
def train_model_pipeline(data, predictors, outcome_var, estimator, param_grid, use_rfe=False, num_folds=5, opt_metric='ROC', seed=69):

    print(f"Training model: {type(estimator).__name__}")
    
    # Step 1: One-hot encode and split the dataset
    X_train, y_train, updated_predictors, label_encoder = one_hot_encode_and_split(data, predictors, outcome_var)
    
    # Step 2: Perform Recursive Feature Elimination (RFE) if required
    if use_rfe:
        _, _, optimal_features = feature_selection(estimator, X_train, y_train, num_folds, opt_metric)
        print(f"Number of features selected: {len(optimal_features)} out of {len(updated_predictors)}")
    else:
        optimal_features = updated_predictors
    
    # Step 3: Hyperparameter tuning
    tuned_model = train_tune_model(estimator, param_grid, X_train, y_train, optimal_features, num_folds, opt_metric, seed)
    
    # Return tuned model as well as label encoder for making predictions
    return tuned_model, label_encoder

# This function trains and tunes multiple models specified in 'models_and_params' 
# and selects the best model based on the specified optimization metric.
def train_and_select_best_model(data, predictors, outcome_var, use_rfe, num_folds, opt_metric):
    # Define your models and parameter grids
    models_and_params = [
        (xgb.XGBClassifier(n_jobs=-1), {
            'n_estimators': Integer(25, 250),
            'learning_rate': Continuous(0.01, 0.5, distribution='uniform'),
            'max_depth': Integer(2, 8),
            'subsample': Continuous(0.8, 1.0, distribution='uniform'),
            'colsample_bytree': Continuous(0.3, 0.7, distribution='uniform'),
            'gamma': Continuous(0, 0.5, distribution='uniform'),
        }),
        # (RandomForestClassifier(n_jobs=-1, class_weight='balanced'), {
        #     'n_estimators': Integer(50, 250),
        #     'max_features': Categorical(['sqrt', 'log2']),
        #     'max_depth': Integer(10, 30),
        #     'min_samples_split': Integer(2, 10),
        #     'min_samples_leaf': Integer(1, 4),
        #     'bootstrap': Categorical([True, False]),
        # }),
        # (GradientBoostingClassifier(), {
        #     'n_estimators': Integer(50, 250),
        #     'learning_rate': Continuous(0.01, 0.2, distribution='uniform'),
        #     'max_depth': Integer(3, 5),
        #     'min_samples_split': Integer(2, 10),
        #     'min_samples_leaf': Integer(1, 4),
        #     'subsample': Continuous(0.8, 1.0, distribution='uniform'),
        #     'max_features': Categorical(['sqrt', 'log2']),
        # })
    ]
    
    best_model = None
    best_score = -float('inf')
    best_label_encoder = None
    
    # Train each model and keep track of the best one
    for estimator, param_grid in models_and_params:
        tuned_model, label_encoder = train_model_pipeline(
            data, predictors, outcome_var,
            estimator, param_grid,
            use_rfe=use_rfe, num_folds=num_folds,
            opt_metric=opt_metric
        )

        score = tuned_model.best_score_

        # Update best_model, best_score, etc.
        if score > best_score:
            best_model = tuned_model
            best_score = score
            best_label_encoder = label_encoder
            
    return best_model, best_label_encoder
