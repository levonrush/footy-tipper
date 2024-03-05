from joblib import load
import dill as pickle
import pandas as pd
import sqlite3
import pandas as pd

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

def load_models(project_root):
    """
    Load the LabelEncoder and Pipeline objects from files.
    
    Args:
        project_root (Path): The root path of the project.
        
    Returns:
        label_encoder (LabelEncoder): The loaded LabelEncoder.
        pipeline (Pipeline): The loaded Pipeline.
    """

    # Load the LabelEncoder
    print("Loading models...")
    label_encoder = load(project_root / "models" / 'label_encoder.pkl')
    print("Label encoder loaded")
    
    # Load the pipeline
    with open(project_root / "models" / 'footy_tipper.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    print("Model pipeline loaded")
        
    return label_encoder, pipeline

def model_predictions(pipeline, inference_data, label_encoder):
    """
    Make predictions using the trained model.

    Args:
        tuned_model (Pipeline): The trained model.
        inference_data (DataFrame): The data to make predictions on.
        label_encoder (LabelEncoder): The LabelEncoder object.

    Returns:
        results (DataFrame): The predictions and probability estimates.
    """
    
    print("Making predictions...")

    # Make predictions
    encoded_predictions = pipeline.predict(inference_data)

    # Get probability estimates
    probability_estimates = pipeline.predict_proba(inference_data)

    # Reverse transform the predictions to get the original labels
    predictions = label_encoder.inverse_transform(encoded_predictions)

    # Put everything into a DataFrame
    results = pd.DataFrame({
        'game_id': inference_data["game_id"],
        'home_team_result': predictions,
        'home_team_win_prob': probability_estimates[:, 1],
        'home_team_lose_prob': probability_estimates[:, 0]
    })
    
    # Return the DataFrame
    return results

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
