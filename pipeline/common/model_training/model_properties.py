import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pipeline.common.model_prediciton import prediction_functions as pf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

def plot_sampling_distributions(home_model, away_model, test_data, predictors, n_simulations=100000):
    """
    Plot the sampling distributions for the home and away scores.
    
    Args:
        home_model (Pipeline): The trained model for home team scores.
        away_model (Pipeline): The trained model for away team scores.
        test_data (DataFrame): The data for which predictions are to be made.
        predictors (list): The list of predictor columns.
        n_simulations (int): The number of simulations to run.
    """
    # Predict the expected scores
    test_data['home_score_avg'] = pf.predict_scores(home_model, test_data[predictors])
    test_data['away_score_avg'] = pf.predict_scores(away_model, test_data[predictors])
    
    # Simulate the game outcomes for the first match in the test data
    match = test_data.iloc[0]
    home_score_avg = match['home_score_avg']
    away_score_avg = match['away_score_avg']
    
    home_score_sim = poisson.rvs(home_score_avg, size=n_simulations)
    away_score_sim = poisson.rvs(away_score_avg, size=n_simulations)
    
    home_wins = (home_score_sim > away_score_sim).sum() / n_simulations
    away_wins = (home_score_sim < away_score_sim).sum() / n_simulations
    draws = (home_score_sim == away_score_sim).sum() / n_simulations
    
    # Calculate the most frequent scoreline
    scorelines = list(zip(home_score_sim, away_score_sim))
    predicted_scoreline = max(set(scorelines), key=scorelines.count)
    
    # Plot the distributions of the scores
    plt.figure(figsize=(14, 6))
    
    # Histogram for home team scores
    plt.subplot(1, 2, 1)
    sns.histplot(home_score_sim, kde=False, color='blue', bins=range(0, max(home_score_sim)+1), alpha=0.7, label='Home Score')
    sns.histplot(away_score_sim, kde=False, color='red', bins=range(0, max(away_score_sim)+1), alpha=0.7, label='Away Score')
    plt.axvline(home_score_avg, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(away_score_avg, color='red', linestyle='dashed', linewidth=1)
    plt.title('Distribution of Simulated Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Add text annotation for expected scoreline
    plt.text(0.05, 0.95, f'Expected Scoreline:\nHome {predicted_scoreline[0]} - Away {predicted_scoreline[1]}',
             horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))

    # Bar plot for win probabilities
    plt.subplot(1, 2, 2)
    probabilities = [home_wins, away_wins, draws]
    labels = ['Home Win', 'Away Win', 'Draw']
    sns.barplot(x=labels, y=probabilities, palette=['blue', 'red', 'gray'])
    plt.ylim(0, 1)
    plt.title('Win Probabilities')
    for i in range(len(labels)):
        plt.text(i, probabilities[i] + 0.02, f'{probabilities[i]:.2f}', ha='center')
    
    plt.tight_layout()
    plt.show()

def evaluate_models(home_model, away_model, test_data, predictors, n_simulations=100000):
    """
    Evaluate the models on the test data and calculate accuracy and other metrics.
    
    Args:
        home_model (Pipeline): The trained model for home team scores.
        away_model (Pipeline): The trained model for away team scores.
        test_data (DataFrame): The test dataset.
        predictors (list): The list of predictor columns.
        n_simulations (int): The number of simulations to run for each game.
        
    Returns:
        DataFrame: The test data with predicted probabilities and actual outcomes.
    """
    # Predict the expected scores
    test_data['home_goals_avg'] = pf.predict_scores(home_model, test_data[predictors])
    test_data['away_goals_avg'] = pf.predict_scores(away_model, test_data[predictors])
    
    # Simulate the games and calculate probabilities
    results = []
    for index, row in test_data.iterrows():
        probabilities, predicted_scoreline = pf.simulate_game(row['home_goals_avg'], row['away_goals_avg'], n_simulations)
        result = {
            'home_win_prob': probabilities['home_win_prob'],
            'away_win_prob': probabilities['away_win_prob'],
            'draw_prob': probabilities['draw_prob'],
            'predicted_home_goals': predicted_scoreline[0],
            'predicted_away_goals': predicted_scoreline[1],
        }
        results.append(result)
    
    probabilities_df = pd.DataFrame(results)
    result_df = pd.concat([test_data.reset_index(drop=True), probabilities_df], axis=1)
    
    # Determine the predicted outcomes
    result_df['predicted_outcome'] = result_df.apply(
        lambda row: 'home_win' if row['home_win_prob'] > max(row['away_win_prob'], row['draw_prob']) else
                    ('away_win' if row['away_win_prob'] > max(row['home_win_prob'], row['draw_prob']) else 'draw'),
        axis=1
    )
    
    # Determine the actual outcomes
    result_df['actual_outcome'] = result_df.apply(
        lambda row: 'home_win' if row['team_final_score_home'] > row['team_final_score_away'] else
                    ('away_win' if row['team_final_score_home'] < row['team_final_score_away'] else 'draw'),
        axis=1
    )
    
    # Calculate accuracy
    accuracy = (result_df['predicted_outcome'] == result_df['actual_outcome']).mean()
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print classification report
    labels = ['home_win', 'away_win', 'draw']
    print("\nClassification Report:")
    print(classification_report(result_df['actual_outcome'], result_df['predicted_outcome'], labels=labels, target_names=labels, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(result_df['actual_outcome'], result_df['predicted_outcome'], labels=labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # ROC curves and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for label in labels:
        if any(result_df['actual_outcome'] == label):
            fpr[label], tpr[label], _ = roc_curve(result_df['actual_outcome'] == label, result_df['predicted_outcome'] == label)
            roc_auc[label] = auc(fpr[label], tpr[label])
    
    # Plot ROC curves
    plt.figure()
    for label in labels:
        if label in fpr and label in tpr:
            plt.plot(fpr[label], tpr[label], lw=2, label=f'ROC curve of class {label} (area = {roc_auc[label]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return result_df

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

    # If you've used OneHotEncoder inside a ColumnTransformer
    if 'one_hot_encoder' in pipeline.named_steps:
        # Get the ColumnTransformer
        column_transformer = pipeline.named_steps['one_hot_encoder']
        # Find the OneHotEncoder inside the ColumnTransformer
        for name, transformer, columns in column_transformer.transformers_:
            if isinstance(transformer, OneHotEncoder):
                one_hot_encoder = transformer
                break
        # Get the feature names from the one-hot encoder
        one_hot_features = one_hot_encoder.get_feature_names_out(input_features=column_names)
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
