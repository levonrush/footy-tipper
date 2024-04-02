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

# Get to the root directory
project_root = pathlib.Path().absolute()

# Now construct the relative path to your SQLite database
db_path = project_root / "data" / "footy-tipper-db.sqlite"

# Load the model
label_encoder, footy_tipper = pf.load_models(project_root)

# Get the inference data
inference_data = pf.get_inference_data(db_path, project_root / 'pipeline/common/sql/inference_data.sql')

# Make predictions
predictions_df = pf.model_predictions(footy_tipper, inference_data, label_encoder)

# Save the predictions
pf.save_predictions_to_db(
    predictions_df, 
    db_path, 
    project_root / 'pipeline/common/sql/create_table.sql', 
    project_root / 'pipeline/common/sql/insert_into_table.sql'
    )
