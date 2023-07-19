import os
import subprocess

# Define paths to data-preparation script and model prediction notebooks
data_prep_r = os.path.join(os.getcwd(), "pipeline/data-prep/data-prep.R")
model_prediction_ipynb = os.path.join(os.getcwd(), "pipeline/model-prediction/model-prediction.ipynb")
use_predicitona_ipynb = os.path.join(os.getcwd(), "pipeline/use_predictions/send-predictiona.ipynb")

# Run data-preparation script
subprocess.run(["Rscript", data_prep_r], check=True)

# Convert and run prediction notebooks
subprocess.run(["python", "-m", "nbconvert", "--to", "python", "--execute", model_prediction_ipynb], check=True)
subprocess.run(["python", "-m", "nbconvert", "--to", "python", "--execute", use_predicitona_ipynb], check=True)
