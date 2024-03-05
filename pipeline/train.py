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
    modelling_functions as mf,
    model_properties as mp
)
# Get to the root directory
project_root = pathlib.Path().absolute()

# Now construct the relative path to your SQLite database
db_path = project_root / "data" / "footy-tipper-db.sqlite"

# Get the training data
training_data = mf.get_training_data(
    db_path = project_root / "data" / "footy-tipper-db.sqlite", 
    sql_file = project_root / 'pipeline/common/sql/training_data.sql'
    )

# Train the model
footy_tipper, label_encoder = mf.train_and_select_best_model(
    training_data, tc.predictors, tc.outcome_var,
    tc.use_rfe, tc.num_folds, tc.opt_metric
)

# Save the model
mf.save_models(label_encoder, footy_tipper, project_root)
