import pandas as pd
import numpy as np

def preprocess_inference_data(inference_data, predictors, label_encoder, optimal_features):
    # One hot encode categorical variables
    X_infer = inference_data[predictors].copy()
    object_columns = X_infer.select_dtypes(include=['object']).columns
    X_infer = pd.get_dummies(X_infer, columns=object_columns)
    
    # Make sure inference data has the same columns as the training data
    missing_cols = set(optimal_features) - set(X_infer.columns)
    for c in missing_cols:
        X_infer[c] = 0
    X_infer = X_infer[optimal_features]

    return X_infer

# This is your inference data
inference_data = ...

# You'll need to use the 'optimal_features' variable which was returned from 'train_model_pipeline' or 'train_and_select_best_model' function
optimal_features = best_model.best_estimator_.named_steps['columntransformer'].transformers_[1][1].named_steps['onehotencoder'].get_feature_names(input_features=categorical_columns)

X_inference = preprocess_inference_data(inference_data, predictors, label_encoder, optimal_features)

# Make predictions
predictions = best_model.predict(X_inference)

# # This function takes as input a trained model, inference dataset, label encoder, and game_id for the inference set. 
# # It outputs the predictions made by the model on the inference set, and returns a dataframe with game_id, predicted outcome, 
# # and the probability estimates for each outcome.
# def model_predictions(tuned_model, X_inference, label_encoder, game_id_inference):
    
#     # Make predictions
#     predictions = tuned_model.predict(X_inference)
    
#     # Get probability estimates
#     probability_estimates = tuned_model.predict_proba(X_inference)

#     # Convert integer labels back to original class labels
#     original_class_labels = label_encoder.inverse_transform(predictions)

#     # Split probability estimates into separate columns
#     home_team_win_prob = probability_estimates[:, 1]
#     home_team_lose_prob = probability_estimates[:, 0]

#     # Create a DataFrame with the results
#     results = pd.DataFrame({
#         'game_id': game_id_inference.values,
#         'home_team_result': original_class_labels,
#         'home_team_win_prob': home_team_win_prob,
#         'home_team_lose_prob': home_team_lose_prob
#     })
    
#     # Return the DataFrame
#     return results