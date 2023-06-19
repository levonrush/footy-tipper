# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

# Other libraries
import pandas as pd
import numpy as np

def one_hot_encode_and_split(data, predictors, outcome_var):
    # # One hot encode categorical variables
    # X = data[predictors].copy()
    # object_columns = X.select_dtypes(include=['object']).columns
    # X = pd.get_dummies(X, columns=object_columns)
    
    # # Getting updated predictors
    # updated_predictors = X.columns.tolist()

    # # Encode the label if it's a categorical variable
    # y = data[outcome_var].copy()
    # if y.dtype == 'object':
    #     label_encoder = LabelEncoder()
    #     y = label_encoder.fit_transform(y)

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
        
    # return X_train, y_train, X_inference, updated_predictors, label_encoder

    # Extract game_id for the inference set
    game_id_inference = data.loc[X_inference.index, 'game_id']

    # Return game_id_inference as well
    return X_train, y_train, X_inference, updated_predictors, label_encoder, game_id_inference


def perform_rfe(estimator, X, y, num_folds, opt_metric):

    # Scoring metric
    if opt_metric == 'ROC':
        scoring = make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)
    else:
        scoring = 'accuracy'

    # Recursive Feature Elimination with cross-validation
    rfecv = RFECV(estimator=estimator, step=1, n_jobs=-1, cv=StratifiedKFold(num_folds), scoring=scoring)
    rfecv.fit(X, y)

    # Optimal set of predictors (one-hot encoded)
    optimal_features = np.array(X.columns)[rfecv.support_]

    # Return the one-hot encoded data and the optimal features
    return X, y, optimal_features

def train_tune_model(estimator, param_grid, X, y, optimal_features, num_folds=5, opt_metric='ROC', seed=69):
    
    # Scoring metric
    if opt_metric == 'ROC':
        scoring = make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)
    else:
        scoring = 'accuracy'

    # Select only the optimal features from X
    X_optimal = X[optimal_features]

    # Grid Search with cross-validation
    cv = GridSearchCV(estimator, param_grid, cv=StratifiedKFold(num_folds), scoring=scoring, verbose=1, n_jobs=-1)
    cv.fit(X_optimal, y)
    
    # Print results
    print(cv.best_params_)
    print(cv.best_score_)

    return cv

def train_model_pipeline(data, predictors, outcome_var, estimator, param_grid, use_rfe=False, num_folds=5, opt_metric='ROC', seed=69):
    
    # Step 1: One-hot encode and split the dataset
    X_train, y_train, X_inference, updated_predictors, label_encoder, game_id_inference = one_hot_encode_and_split(data, predictors, outcome_var)
    
    # Step 2: Perform Recursive Feature Elimination (RFE) if required
    if use_rfe:
        _, _, optimal_features = perform_rfe(estimator, X_train, y_train, num_folds, opt_metric)
    else:
        optimal_features = updated_predictors
    
    # Step 3: Hyperparameter tuning
    tuned_model = train_tune_model(estimator, param_grid, X_train, y_train, optimal_features, num_folds, opt_metric, seed)
    
    # Return tuned model as well as X_inference and label encoder for making predictions
    return tuned_model, X_inference[optimal_features], label_encoder, game_id_inference

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
