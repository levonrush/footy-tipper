# Description: This script is used to train the model and save it to the models directory
print("Running the train.py script...")

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
from pipeline.common.model_training import (
    training_config as tc,
    modelling_functions as mf
)
# Get to the root directory
project_root = pathlib.Path().absolute()

# Now construct the relative path to your SQLite database
db_path = project_root / "data" / "footy-tipper-db.sqlite"

# Get the training data
print("Get Training Data")
training_data = mf.get_training_data(
    db_path = project_root / "data" / "footy-tipper-db.sqlite", 
    sql_file = project_root / 'pipeline/common/sql/training_data.sql'
    )

# Train the model
print("Run Model Training Module")
footy_tipper, label_encoder = mf.train_and_select_best_model(
    training_data, tc.predictors, tc.outcome_var,
    tc.use_rfe, tc.num_folds, tc.opt_metric
)
print("Model training complete!")

# # Save the model
# print("Save the model")
# mf.save_models(label_encoder, footy_tipper, project_root)
# print("Model saved!")

print("Model training complete!")   

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
# label_encoder, footy_tipper = pf.load_models(project_root)

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
