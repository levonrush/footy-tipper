import pandas as pd

def get_feature_importance(pipeline, column_names):
    """
    Extracts feature importances from a fitted scikit-learn pipeline.
    
    Args:
        pipeline (Pipeline): The fitted pipeline.
        column_names (list): List of original column names.
        
    Returns:
        DataFrame: The DataFrame containing feature names and their corresponding importance.
    """


    # Get the final step ('hyperparamtuning' in your case) estimator from the pipeline
    final_estimator = pipeline.named_steps['hyperparamtuning'].best_estimator_

    # Check if the final_estimator has the feature_importances_ attribute
    if hasattr(final_estimator, 'feature_importances_'):
        # Extract the feature importances
        importances = final_estimator.feature_importances_
    else:
        raise AttributeError(f"The final estimator doesn't have 'feature_importances_' attribute")

    # If you've used 'RFECV' for feature selection
    if 'feature_elimination' in pipeline.named_steps:
        # Get the support mask
        support_mask = pipeline.named_steps['feature_elimination'].support_
        # Get the names of the features selected by RFE
        column_names = [col for (col, mask) in zip(column_names, support_mask) if mask]

    # If you've used OneHotEncoder
    if 'one_hot_encoder' in pipeline.named_steps:
        # Get the feature names from the one-hot encoder
        one_hot_features = pipeline.named_steps['one_hot_encoder'].get_feature_names(input_features=column_names)
        column_names = one_hot_features.tolist()

    # Create a DataFrame of feature importances
    feature_importances = pd.DataFrame(
        {
            'Feature': column_names,
            'Importance': importances
        }
    )
    # Sort the DataFrame by importance in descending order
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

    return feature_importances
