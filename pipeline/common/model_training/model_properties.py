import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pipeline.common.model_prediciton import prediction_functions as pf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
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

def evaluate_score_predictions(result_df):
    """
    Evaluate the predicted scores against the actual scores and plot the results.
    
    Args:
        result_df (DataFrame): The DataFrame containing actual and predicted scores.
        
    Returns:
        dict: A dictionary with evaluation metrics.
    """
    metrics = {}

    # Mean Absolute Error
    metrics['mae_home'] = mean_absolute_error(result_df['team_final_score_home'], result_df['predicted_home_goals'])
    metrics['mae_away'] = mean_absolute_error(result_df['team_final_score_away'], result_df['predicted_away_goals'])

    # Root Mean Squared Error
    metrics['rmse_home'] = np.sqrt(mean_squared_error(result_df['team_final_score_home'], result_df['predicted_home_goals']))
    metrics['rmse_away'] = np.sqrt(mean_squared_error(result_df['team_final_score_away'], result_df['predicted_away_goals']))

    # R-squared
    metrics['r2_home'] = r2_score(result_df['team_final_score_home'], result_df['predicted_home_goals'])
    metrics['r2_away'] = r2_score(result_df['team_final_score_away'], result_df['predicted_away_goals'])

    # Plotting the differences between predicted and actual scores
    result_df['home_diff'] = result_df['team_final_score_home'] - result_df['predicted_home_goals']
    result_df['away_diff'] = result_df['team_final_score_away'] - result_df['predicted_away_goals']

    plt.figure(figsize=(14, 6))

    # Histogram of the differences
    plt.subplot(1, 2, 1)
    sns.histplot(result_df['home_diff'], kde=True, color='blue', label='Home Score Difference')
    sns.histplot(result_df['away_diff'], kde=True, color='red', label='Away Score Difference')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Score Differences')
    plt.legend()

    # Scatter plot of predicted vs actual scores
    plt.subplot(1, 2, 2)
    plt.scatter(result_df['team_final_score_home'], result_df['predicted_home_goals'], color='blue', label='Home')
    plt.scatter(result_df['team_final_score_away'], result_df['predicted_away_goals'], color='red', label='Away')
    plt.plot([0, max(result_df['team_final_score_home'].max(), result_df['team_final_score_away'].max())], 
             [0, max(result_df['predicted_home_goals'].max(), result_df['predicted_away_goals'].max())], 
             color='green', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Predicted vs Actual Scores')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return metrics

def evaluate_models_with_scores(home_model, away_model, test_data, predictors, n_simulations=100000):
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
            'predicted_home_goals_sim': predicted_scoreline[0],
            'predicted_away_goals_sim': predicted_scoreline[1],
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
    
    # Evaluate score predictions
    result_df['predicted_home_goals'] = result_df['home_goals_avg']
    result_df['predicted_away_goals'] = result_df['away_goals_avg']
    metrics_poisson = evaluate_score_predictions(result_df)
    
    result_df['predicted_home_goals'] = result_df['predicted_home_goals_sim']
    result_df['predicted_away_goals'] = result_df['predicted_away_goals_sim']
    metrics_simulation = evaluate_score_predictions(result_df)
    
    print("\nPoisson Model Metrics:")
    print(metrics_poisson)
    
    print("\nSimulation Model Metrics:")
    print(metrics_simulation)
    
    return result_df, metrics_poisson, metrics_simulation

def get_feature_importances(pipeline, feature_names):
    """
    Get feature importances from the trained pipeline.
    
    Args:
        pipeline (Pipeline): The trained pipeline.
        feature_names (list): The list of feature names.
        
    Returns:
        DataFrame: A DataFrame with the top 20 features and their importance values.
    """
    # Get the model from the pipeline
    model = pipeline.named_steps['hyperparamtuning'].best_estimator_
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    # Create a DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Get the top 20 features
    top_20_features = feature_importance_df.head(20)
    
    return top_20_features
