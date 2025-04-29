# Description: This script is used to make predictions using the model and save the predictions to the database.
print("Running the inference.py script...")

# import libraries
import os
import sys
import pathlib
import pandas as pd
# from deap import creator
# # creator.__dict__.pop('_created', None)
# import importlib
# import deap.creator
# importlib.reload(deap.creator)

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

# get the correct predictors
predictors = tc.filter_predictors(include_performance=tc.include_performance, predictor_list=tc.predictors)

# Load the model
home_model = pf.load_models('home_model', project_root)
away_model = pf.load_models('away_model', project_root)

# Get the inference data
inference_data = pf.get_inference_data(db_path, project_root / 'pipeline/common/sql/inference_data.sql')

# Make predictions
# Predict match outcomes and scorelines for the inference data
outcomes, margins = pf.predict_match_outcome_and_scoreline_with_bayes(home_model, away_model, inference_data, predictors)
outcome_df = pd.merge(outcomes, margins, on='game_id')

# Save the predictions
pf.save_predictions_to_db(
    outcome_df, 
    db_path, 
    project_root / 'pipeline/common/sql/create_table.sql', 
    project_root / 'pipeline/common/sql/insert_into_table.sql'
    )

print("Predictions saved to the database!")
