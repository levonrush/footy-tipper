import os
import subprocess

# Define paths to data-preparation script and model-training notebook
data_prep_r = os.path.join(os.getcwd(), "pipeline/data-prep/data-prep.R")
model_training_ipynb = os.path.join(os.getcwd(), "pipeline/model-training/model-training.ipynb")

# Run data-preparation script
subprocess.run(["Rscript", data_prep_r], check=True)

# Convert and run model-training notebook
subprocess.run(["python", "-m", "nbconvert", "--to", "python", "--execute", model_training_ipynb], check=True)
