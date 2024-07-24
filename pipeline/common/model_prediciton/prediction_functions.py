from joblib import load
import dill as pickle
import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import poisson

def get_inference_data(db_path, sql_file):
    """
    Retrieve data for inference from an SQLite database.
    
    Args:
        db_path (Path): The path to the SQLite database.
        sql_file (str): The path to the SQL file that contains the query.
        
    Returns:
        DataFrame: The DataFrame containing inference data.
    """

    print("Getting inference data...")
    
    # Connect to the SQLite database
    con = sqlite3.connect(str(db_path))

    # Read SQL query from external SQL file
    with open(sql_file, 'r') as file:
        query = file.read()

    inference_data = pd.read_sql_query(query, con)

    # Close the connection
    con.close()
    
    return inference_data

def predict_scores(model, data):
    """
    Predict the expected scores using the trained model.
    
    Args:
        model (Pipeline): The trained model.
        data (DataFrame): The input data for predictions.
        
    Returns:
        np.array: The predicted scores.
    """
    return model.predict(data)

def simulate_game(home_score_avg, away_score_avg, n_simulations=100000):
    """
    Simulate a number of games and calculate the probabilities of each outcome.
    
    Args:
        home_score_avg (float): The expected score for the home team.
        away_score_avg (float): The expected score for the away team.
        n_simulations (int): The number of simulations to run.
        
    Returns:
        dict: The probabilities of home win, away win, and draw.
        tuple: The predicted scoreline (home_goals, away_goals).
    """
    home_goals_sim = poisson.rvs(home_score_avg, size=n_simulations)
    away_goals_sim = poisson.rvs(away_score_avg, size=n_simulations)
    
    home_wins = (home_goals_sim > away_goals_sim).sum()
    away_wins = (home_goals_sim < away_goals_sim).sum()
    draws = (home_goals_sim == away_goals_sim).sum()
    
    total_games = n_simulations
    probabilities = {
        'home_win_prob': home_wins / total_games,
        'away_win_prob': away_wins / total_games,
        'draw_prob': draws / total_games
    }
    
    # Determine the most frequent scoreline
    scorelines = list(zip(home_goals_sim, away_goals_sim))
    predicted_scoreline = max(set(scorelines), key=scorelines.count)
    
    return probabilities, predicted_scoreline

def calculate_bayes_factor(probabilities):
    """
    Calculate the Bayes factor for home win versus away win.
    
    Args:
        probabilities (dict): Dictionary containing probabilities of home win, away win, and draw.
        
    Returns:
        float: Bayes factor for home win versus away win.
    """
    home_win_prob = probabilities['home_win_prob']
    away_win_prob = probabilities['away_win_prob']
    
    # Calculate Bayes factor
    bayes_factor = home_win_prob / away_win_prob if away_win_prob != 0 else np.inf
    
    return bayes_factor

def map_bayes_factor_to_evidence(bayes_factor):
    """
    Map Bayes factor to evidence strength category.
    
    Args:
        bayes_factor (float): Bayes factor for home win versus away win.
        
    Returns:
        str: Evidence strength category.
    """
    if bayes_factor < 1:
        return "Negative evidence"
    elif 1 <= bayes_factor < 3:
        return "Anecdotal evidence"
    elif 3 <= bayes_factor < 10:
        return "Moderate evidence"
    elif 10 <= bayes_factor < 30:
        return "Strong evidence"
    elif 30 <= bayes_factor < 100:
        return "Very strong evidence"
    else:
        return "Decisive evidence"

def predict_match_outcome_and_scoreline_with_bayes(home_model, away_model, inference_data, predictors, n_simulations=100000):
    """
    Predict match outcomes and scorelines for the inference data, including Bayes factors.
    
    Args:
        home_model (Pipeline): The trained model for home team scores.
        away_model (Pipeline): The trained model for away team scores.
        inference_data (DataFrame): The data for which predictions are to be made.
        predictors (list): The list of predictor columns.
        n_simulations (int): The number of simulations to run for each game.
        
    Returns:
        DataFrame: The inference data with predicted probabilities, outcomes, scorelines, and Bayes factors.
    """
    # Predict the expected scores
    inference_data['home_goals_avg'] = predict_scores(home_model, inference_data[predictors])
    inference_data['away_goals_avg'] = predict_scores(away_model, inference_data[predictors])
    
    # Simulate the games and calculate probabilities and scorelines
    results = []
    for index, row in inference_data.iterrows():
        probabilities, predicted_scoreline = simulate_game(row['home_goals_avg'], row['away_goals_avg'], n_simulations)
        home_team_result = 'Win' if probabilities['home_win_prob'] > probabilities['away_win_prob'] else 'Loss'
        
        bayes_factor = calculate_bayes_factor(probabilities)
        evidence_strength = map_bayes_factor_to_evidence(bayes_factor)
        
        result = {
            'game_id': row['game_id'],
            'home_team_win_prob': probabilities['home_win_prob'],
            'home_team_lose_prob': probabilities['away_win_prob'],
            'draw_prob': probabilities['draw_prob'],
            'predicted_home_score': predicted_scoreline[0],
            'predicted_away_score': predicted_scoreline[1],
            'predicted_margin': predicted_scoreline[0] - predicted_scoreline[1],
            'home_team_result': home_team_result,
            'bayes_factor': bayes_factor,
            'evidence_strength': evidence_strength
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Select the required columns
    outcome_df = results_df[['game_id', 'home_team_result', 'home_team_win_prob', 'home_team_lose_prob', 'draw_prob', 'bayes_factor', 'evidence_strength']]
    margin_df = results_df[['game_id', 'predicted_home_score', 'predicted_away_score', 'predicted_margin']]

    return outcome_df, margin_df

def get_predictions(db_path, sql_file):
    """
    Retrieve predictions from an SQLite database.

    Args:
        db_path (Path): The path to the SQLite database.
        sql_file (str): The path to the SQL file that contains the query.

    Returns:
        DataFrame: The DataFrame containing predictions.
    """

    # Connect to the SQLite database
    con = sqlite3.connect(str(db_path))

    # Read SQL query from external SQL file
    with open(sql_file, 'r') as file:
        query = file.read()

    # Execute the query and fetch the results into a data frame
    predictions = pd.read_sql_query(query, con)

    # Disconnect from the SQLite database
    con.close()

    predictions

def load_models(model, project_root):
    """
    Load the Pipeline objects from files.
    
    Args:
        model (str): The name of the model to load.
        project_root (Path): The root path of the project.
        
    Returns:
        pipeline (Pipeline): The loaded Pipeline.
    """
    
    # Load the pipeline
    with open(project_root / "models" / f'{model}.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    print(f"{model} model pipeline loaded")
        
    return pipeline

def save_predictions_to_db(predictions_df, db_path, create_table_sql_file, insert_into_table_sql_file):
    """
    Save predictions to an SQLite database.
    
    Args:
        predictions_df (DataFrame): The DataFrame containing predictions.
        db_path (Path): The path to the SQLite database.
        create_table_sql_file (str): The path to the SQL file that contains the CREATE TABLE query.
        insert_into_table_sql_file (str): The path to the SQL file that contains the INSERT INTO query.
        
    Returns:
        None
    """
    
    print("Saving predictions to database...")

    # Connect to the SQLite database
    con = sqlite3.connect(str(db_path))

    # Read SQL query from external SQL file and create table
    with open(create_table_sql_file, 'r') as file:
        create_table_query = file.read()
    con.execute(create_table_query)

    # Read SQL query from external SQL file for insertion
    with open(insert_into_table_sql_file, 'r') as file:
        insert_into_table_query = file.read()

    # Write each row in the DataFrame to the database
    for index, row in predictions_df.iterrows():
        con.execute(insert_into_table_query, (
            row['game_id'], 
            row['home_team_result'],
            row['home_team_win_prob'],
            row['home_team_lose_prob']
        ))

    # Commit the transaction
    con.commit()

    # Close the connection
    con.close()
