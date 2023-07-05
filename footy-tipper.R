### The Footy Tipper Pipeline ###

# define where the pipeline lives
library(here)
i_am("footy-tipper.R")

# find and load all the helper functions for the project and load them
pipeline_functions <- list.files(
    paste0(here(), "/pipeline"), pattern = "*.R$",
    full.names = TRUE, ignore.case = TRUE
)

sapply(pipeline_functions, source, .GlobalEnv)

# define the paths to the notebooks
data_prep_rmd <- paste0(here(), "/pipeline/data-prep/data-prep.Rmd")
model_training_ipynb <- paste0(here(), "/pipeline/model-training/model-training.ipynb")
send_predictions_ipynb <- paste0(here(), "/pipeline/use-predictions/send_predictions.ipynb")

# Execute the data-prep.Rmd notebook
rmarkdown::render(
    data_prep_rmd,
    output_format = "github_document"
)

# Execute the model-training.ipynb notebook
system(paste("jupyter nbconvert --to notebook --execute", model_training_ipynb))

# Execute the use-predictions.Rmd notebook
system(paste("jupyter nbconvert --to notebook --execute", send_predictions_ipynb))
