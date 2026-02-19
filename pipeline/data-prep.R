# Description: This script is responsible for running the data preparation pipeline.
print("Running the data-prep.R script...")

# Resolve the project root from the script location so this runs reliably in
# Docker, CI, and local environments.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", script_arg[1]))
  project_root <- normalizePath(file.path(dirname(script_path), ".."))
  setwd(project_root)
}

library(here)

print("Loading environment variables...")
suppressMessages(library(dotenv))
load_dot_env(here("secrets.env"))

# Find and load all helper functions located in 'pipeline/data-prep/functions' directory
print("Finding helper functions...")
data_prep_functions <- list.files(
    paste0(here(), "/pipeline/common/data-prep"),
    pattern = "\\.R$", # Search for R scripts
    full.names = TRUE, # Return the full path
    ignore.case = TRUE # Case-insensitive
)

# Source each function into the Global Environment for use
print("Sourcing helper functions...")
sapply(data_prep_functions, source, .GlobalEnv)

print(paste0("Prep mode: ", prep_mode))
print(paste0("Season scope: ", min(year_span), " to ", max(year_span)))

# Run the data pipeline function (defined in one of the helper files) with specified parameters
print("Running the data pipeline...")
pipeline_data <- data_pipeline(
    year_span, pipeline = "binomial",
    form_period, carry_over, k_val,
    elo_init, use_odds, include_performance, prep_mode
)

# Separate the datasets from the pipeline
print("Separating the datasets...")
footy_tipping_data <- pipeline_data[["footy_tipping_data"]]
training_data <- pipeline_data[["training_data"]]
inference_data <- pipeline_data[["inference_data"]]

# Connect to the SQLite database located in '/data/footy-tipper-db.sqlite'
print("Connecting to the SQLite database...")
con <- dbConnect(SQLite(), paste0(here(), "/data/footy-tipper-db.sqlite"))

# Write processed data to SQLite.
# - full/train: overwrite all prepared tables.
# - infer: upsert historical tables by game_id and overwrite only current inference batch.
print("Writing the processed data to the SQLite database...")
write_prepared_tables(
  con,
  prep_mode,
  footy_tipping_data,
  training_data,
  inference_data
)

# Disconnect from the SQLite database to ensure no other operations are unintentionally performed
print("Disconnecting from the SQLite database...")
dbDisconnect(con)

print("Data preparation complete!")
