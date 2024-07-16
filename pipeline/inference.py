# Description: This script is used to make predictions using the model and save the predictions to the database.
print("Running the inference.py script...")

# import libraries
import os
import sys
import pathlib

script_dir = os.path.dirname(os.path.abspath(__file__))

# get the parent directory
parent_dir = os.path.dirname(script_dir)

# add the parent directory to the system path
sys.path.insert(0, parent_dir)

# import functions from common like this:
from pipeline.common.model_prediciton import prediction_functions as pf
from pipeline.common.model_training import training_config as tc

# Get to the root directory
project_root = pathlib.Path().absolute()

# Now construct the relative path to your SQLite database
db_path = project_root / "data" / "footy-tipper-db.sqlite"

# Load the model
home_model = pf.load_models('home_model', project_root)
away_model = pf.load_models('away_model', project_root)

# Get the inference data
inference_data = pf.get_inference_data(db_path, project_root / 'pipeline/common/sql/inference_data.sql')

# Make predictions
# Predict match outcomes and scorelines for the inference data
outcomes, margins = pf.predict_match_outcome_and_scoreline(home_model, away_model, inference_data, tc.predictors)

# Save the predictions
pf.save_predictions_to_db(
    outcomes, 
    db_path, 
    project_root / 'pipeline/common/sql/create_table.sql', 
    project_root / 'pipeline/common/sql/insert_into_table.sql'
    )

print("Predictions saved to the database!")
