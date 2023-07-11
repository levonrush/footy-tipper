import pandas as pd

# This function takes a trained model and a list of feature names. It then returns a DataFrame
# listing the features and their corresponding importances, sorted in descending order of importance.
def get_sorted_feature_importance(model, feature_names):

    # Create a dictionary linking feature names to their importances
    feature_importance_dict = dict(zip(feature_names, model.best_estimator_.feature_importances_))

    # Convert the dictionary into a DataFrame and sort it by importance
    feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)

    # Reset the DataFrame's index
    feature_importance_df.reset_index(drop=True, inplace=True)

    return feature_importance_df
