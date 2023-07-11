# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import class_weight
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Continuous, Categorical, Integer

# Other libraries
import xgboost as xgb
import pandas as pd
import numpy as np

# This function first one-hot encodes the categorical variables in the dataset. 
# It then splits the dataset into training and inference datasets based on certain conditions.
# The function also extracts game_id for the inference set.
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
    
    # Compute minimum competition_year
    min_competition_year = data['competition_year'].min()

    # Split the dataset into train and inference
    train_mask = ((data['game_state_name'] == 'Final') & 
                  (data['competition_year'] != min_competition_year) &
                  (data['team_head_to_head_odds_away'].notna()))
    
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Compute minimum round_id for the remaining set
    remaining_set = data[data['game_state_name'] == 'Pre Game']
    if not remaining_set.empty:
        min_round_id = remaining_set['round_id'].min()
        inference_mask = ((data['game_state_name'] == 'Pre Game') & 
                          (data['round_id'] == min_round_id))
        X_inference = X[inference_mask]
    else:
        X_inference = pd.DataFrame(columns=X.columns)  # empty DataFrame with same columns

    # Extract game_id for the inference set
    game_id_inference = data.loc[X_inference.index, 'game_id']

    # Return game_id_inference as well
    return X_train, y_train, X_inference, updated_predictors, label_encoder, game_id_inference

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

    # evolved_estimator = GAFeatureSelectionCV(
    #     estimator=estimator,
    #     cv=cv,
    #     scoring=scoring,
    #     population_size=30,
    #     generations=40,
    #     n_jobs=-1,
    #     verbose=False,
    #     keep_top_k=2,
    #     elitism=True,
    # )

    # # Fit the model to your data
    # evolved_estimator.fit(X, y)

    # Optimal set of predictors (one-hot encoded)
    # optimal_features = np.array(X.columns)[evolved_estimator.support_]

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
        population_size=10,
        generations=100,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        param_grid=param_grid,
        criteria='max',
        algorithm='eaMuPlusLambda',
        n_jobs=-1,
        verbose=False,
        keep_top_k=4
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
    X_train, y_train, X_inference, updated_predictors, label_encoder, game_id_inference = one_hot_encode_and_split(data, predictors, outcome_var)
    
    # Step 2: Perform Recursive Feature Elimination (RFE) if required
    if use_rfe:
        _, _, optimal_features = feature_selection(estimator, X_train, y_train, num_folds, opt_metric)
        print(f"Number of features selected: {len(optimal_features)} out of {len(updated_predictors)}")
    else:
        optimal_features = updated_predictors
    
    # Step 3: Hyperparameter tuning
    tuned_model = train_tune_model(estimator, param_grid, X_train, y_train, optimal_features, num_folds, opt_metric, seed)
    
    # Return tuned model as well as X_inference and label encoder for making predictions
    return tuned_model, X_inference[optimal_features], label_encoder, game_id_inference

# This function trains and tunes multiple models specified in 'models_and_params' 
# and selects the best model based on the specified optimization metric.
def train_and_select_best_model(data, predictors, outcome_var, use_rfe, num_folds, opt_metric):
    # Define your models and parameter grids
    models_and_params = [
        (xgb.XGBClassifier(n_jobs=-1), {
            'n_estimators': Integer(50, 250),
            'learning_rate': Continuous(0.01, 0.2, distribution='uniform'),
            'max_depth': Integer(3, 5),
            'subsample': Continuous(0.8, 1.0, distribution='uniform'),
            'colsample_bytree': Continuous(0.3, 0.7, distribution='uniform'),
            'gamma': Continuous(0, 0.2, distribution='uniform'),
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
            'learning_rate': Continuous(0.01, 0.2, distribution='uniform'),
            'max_depth': Integer(3, 5),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'subsample': Continuous(0.8, 1.0, distribution='uniform'),
            'max_features': Categorical(['sqrt', 'log2']),
        })
    ]
    
    best_model = None
    best_score = -float('inf')
    best_label_encoder = None
    best_X_inference = None
    best_game_id_inference = None
    
    # Train each model and keep track of the best one
    for estimator, param_grid in models_and_params:
        tuned_model, X_inference, label_encoder, game_id_inference = train_model_pipeline(
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
            best_X_inference = X_inference
            best_game_id_inference = game_id_inference
            
    return best_model, best_X_inference, best_label_encoder, best_game_id_inference

# This function takes as input a trained model, inference dataset, label encoder, and game_id for the inference set. 
# It outputs the predictions made by the model on the inference set, and returns a dataframe with game_id, predicted outcome, 
# and the probability estimates for each outcome.
def model_predictions(tuned_model, X_inference, label_encoder, game_id_inference):
    
    # Make predictions
    predictions = tuned_model.predict(X_inference)
    
    # Get probability estimates
    probability_estimates = tuned_model.predict_proba(X_inference)

    # Convert integer labels back to original class labels
    original_class_labels = label_encoder.inverse_transform(predictions)

    # Split probability estimates into separate columns
    home_team_win_prob = probability_estimates[:, 1]
    home_team_lose_prob = probability_estimates[:, 0]

    # Create a DataFrame with the results
    results = pd.DataFrame({
        'game_id': game_id_inference.values,
        'home_team_result': original_class_labels,
        'home_team_win_prob': home_team_win_prob,
        'home_team_lose_prob': home_team_lose_prob
    })
    
    # Return the DataFrame
    return results
