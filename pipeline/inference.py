# import libraries
import os
import sys
import pathlib
from dotenv import load_dotenv

# this takes us down a level to the root of the project
sys.path.insert(0, os.path.abspath('..'))

# import functions from common like this:
from pipeline.common.model_training import (
    modelling_functions as mf,
    model_properties as mp,
    training_config as tc,
    prediction_functions as pf,
)
from pipeline.common.use_predictions import sending_functions as sf

# Get to the root directory
project_root = pathlib.Path().absolute().parent

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

# send predictions to users

