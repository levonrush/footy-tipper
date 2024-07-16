from joblib import load
import dill as pickle
import pandas as pd
import sqlite3
import pandas as pd
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

def predict_match_outcome_and_scoreline(home_model, away_model, inference_data, predictors, n_simulations=100000):
    """
    Predict match outcomes and scorelines for the inference data.
    
    Args:
        home_model (Pipeline): The trained model for home team scores.
        away_model (Pipeline): The trained model for away team scores.
        inference_data (DataFrame): The data for which predictions are to be made.
        predictors (list): The list of predictor columns.
        n_simulations (int): The number of simulations to run for each game.
        
    Returns:
        DataFrame: The inference data with predicted probabilities, outcomes, and scorelines.
    """
    # Predict the expected scores
    inference_data['home_goals_avg'] = predict_scores(home_model, inference_data[predictors])
    inference_data['away_goals_avg'] = predict_scores(away_model, inference_data[predictors])
    
    # Simulate the games and calculate probabilities and scorelines
    results = []
    for index, row in inference_data.iterrows():
        probabilities, predicted_scoreline = simulate_game(row['home_goals_avg'], row['away_goals_avg'], n_simulations)
        home_team_result = 'Win' if (probabilities['home_win_prob'] + probabilities['draw_prob']) > probabilities['away_win_prob'] else 'Loss' if (probabilities['away_win_prob'] + probabilities['draw_prob']) > probabilities['home_win_prob'] else 'Draw'
        
        result = {
            'game_id': row['game_id'],
            'home_team_win_prob': probabilities['home_win_prob'],
            'home_team_lose_prob': probabilities['away_win_prob'],
            'draw_prob': probabilities['draw_prob'],
            'predicted_home_score': predicted_scoreline[0],
            'predicted_away_score': predicted_scoreline[1],
            'predicted_margin': (predicted_scoreline[0] - predicted_scoreline[1]),
            'home_team_result': home_team_result
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Select the required columns
    outcome_df = results_df[['game_id', 'home_team_result', 'home_team_win_prob', 'home_team_lose_prob', 'draw_prob']]
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
