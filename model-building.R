# Load 'here' package and set project directory
library(here)
i_am("model_building.R")

# Define paths to data-preparation script and model-training notebook
data_prep_r <- paste0(here(), "/pipeline/data-prep/data-prep.R")
model_training_ipynb <- paste0(here(), "/pipeline/model-training/model-training.ipynb")

# Run data-preparation script
source(data_prep_r)

# Convert and run model-training notebook
system(paste("python -m nbconvert --to python --execute", model_training_ipynb))
