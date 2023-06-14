# sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

# Other libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def one_hot_encode_features(data, predictors):

    X = data[predictors].copy()
    object_columns = X.select_dtypes(include=['object']).columns

    return pd.get_dummies(X, columns=object_columns)

def perform_rfe(estimator, data, k, opt_metric, maximise, steps=None, outcome_var=None, predictors=None):
    
    label_encoder = LabelEncoder()
    data[outcome_var] = label_encoder.fit_transform(data[outcome_var])
    
    # One-hot encode features
    X = one_hot_encode_features(data, predictors)
    y = data[outcome_var]

    # One-hot encode categorical columns
    object_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=object_columns)

    # Scoring metric
    if opt_metric == 'ROC':
        scoring = make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)
    else:
        scoring = 'accuracy'

    # Recursive Feature Elimination with cross-validation
    rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(k), scoring=scoring)
    rfecv.fit(X, y)

    # Plotting number of features vs. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    # Optimal set of predictors (one-hot encoded)
    optimal_features = np.array(X.columns)[rfecv.support_]

    return optimal_features, X.columns


def train_tune_model(estimator, param_grid, data, predictors, outcome_var, num_folds=5, opt_metric='ROC', seed=69):
    
    # One-hot encode features
    X = one_hot_encode_features(data, predictors)
    y = data[outcome_var]

    # Scoring metric
    if opt_metric == 'ROC':
        scoring = make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)
    else:
        scoring = 'accuracy'

    # Grid Search with cross-validation
    cv = GridSearchCV(estimator, param_grid, cv=StratifiedKFold(num_folds), scoring=scoring, verbose=1)
    cv.fit(X, y)
    
    # Print results
    print(cv.best_params_)
    print(cv.best_score_)

    return cv

def training_pipeline(train_df, estimator, outcome_var, predictors):

    label_encoder = LabelEncoder()
    train_df[outcome_var] = label_encoder.fit_transform(train_df[outcome_var])

    if estimator == 'rf':
        rf_estimator = RandomForestClassifier()
        rf_param_grid = {'n_estimators': [10, 50, 100], 'max_features': ['auto', 'sqrt', 'log2']}
        # Capture the returned values
        opt_rf_predictors, all_rf_features = perform_rfe(rf_estimator, data=train_df, k=5, opt_metric='ROC', maximise=True, outcome_var=outcome_var, predictors=predictors)
        # Use opt_rf_predictors as the predictors parameter
        cv = train_multiclass_model(rf_estimator, rf_param_grid, data=train_df, predictors=opt_rf_predictors, outcome_var=outcome_var, opt_metric='ROC')
    
    elif estimator == 'gbm':
        gb_estimator = GradientBoostingClassifier()
        gb_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
        # Capture the returned values
        opt_gb_predictors, all_gb_features = perform_rfe(gb_estimator, data=train_df, k=5, opt_metric='ROC', maximise=True, outcome_var=outcome_var, predictors=predictors)
        # Use opt_gb_predictors as the predictors parameter
        cv = train_multiclass_model(gb_estimator, gb_param_grid, data=train_df, predictors=opt_gb_predictors, outcome_var=outcome_var, opt_metric='ROC')

    return cv
