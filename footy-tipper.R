### The Footy Tipper Pipeline ###

# first, decide what type of run we are doing
prod_run = T

# define where the pipeline lives
library(here)
i_am("footy-tipper.R")

# find and load all the helper functions for the project and load them
pipeline_functions <- list.files(
    paste0(here(), "/pipeline"), pattern = "*.R$",
    full.names = TRUE, ignore.case = TRUE
)

sapply(pipeline_functions, source, .GlobalEnv)

# load the project secrets
load_dot_env(file = "secrets.env")

# define what type of run we're doing and set env variables
detect_set_environment(prod_run = prod_run)

# define the paths to the notebooks
data_prep_rmd <- paste0(here(), "/pipeline/data-prep/data-prep.Rmd")
model_training_ipynb <- paste0(here(), "/pipeline/model-training/model-training.ipynb")
# use_predictions_rmd <- paste0(here(), "/pipeline/use-predictions/use-predictions.Rmd")
use_predictions_r <- paste0(here(), "/pipeline/use-predictions/use-predictions.R")

# Execute the data-prep.Rmd notebook
rmarkdown::render(
    data_prep_rmd,
    output_format = "github_document"
)

# Execute the model-training.ipynb notebook
system(paste("jupyter nbconvert --to notebook --execute", model_training_ipynb))

# Execute the use-predictions.Rmd notebook
# rmarkdown::render(
#     use_predictions_rmd,
#     output_format = "github_document"
# )
source(use_predictions_r)
