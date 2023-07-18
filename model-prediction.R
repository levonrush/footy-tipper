# Load 'here' package and set project directory
library(here)
i_am("model_prediction.R")

# Define paths to data-preparation script, model-prediction and send-predictions notebooks
data_prep_r <- paste0(here(), "/pipeline/data-prep/data-prep.R")
model_prediction_ipynb <- paste0(here(), "/pipeline/model-prediction/model-prediction.ipynb")
send_predictions_ipynb <- paste0(here(), "/pipeline/use-predictions/send_predictions.ipynb")

# Run data-preparation script
source(data_prep_r)

# Convert and run model-prediction notebook
system(paste("python -m nbconvert --to python --execute", model_prediction_ipynb))

# Convert and run send-predictions notebook
system(paste("python -m nbconvert --to python --execute", send_predictions_ipynb))
