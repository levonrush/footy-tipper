# Set the project directory to 'pipeline/data-prep/data-prep.R'
i_am("pipeline/data-prep/data-prep.R")

# Find and load all helper functions located in 'pipeline/data-prep/functions' directory
data_prep_functions <- list.files(
    paste0(here(), "/pipeline/data-prep/functions"),
    pattern = "*.R$",  # Search for R scripts
    full.names = TRUE, # Return the full path
    ignore.case = TRUE # Case-insensitive
)

# Source each function into the Global Environment for use
sapply(data_prep_functions, source, .GlobalEnv)

load_dot_env(paste0(here(), "/secrets.env"))

# Run the data pipeline function (defined in one of the helper files) with specified parameters
pipeline_data <- data_pipeline(
    year_span, pipeline = "binomial",
    form_period, carry_over, k_val, elo_init
)

# Separate the datasets from the pipeline
footy_tipping_data <- pipeline_data[["footy_tipping_data"]]
training_data <- pipeline_data[["training_data"]]
inference_data <- pipeline_data[["inference_data"]]

# Connect to the SQLite database located in '/data/footy-tipper-db.sqlite'
con <- dbConnect(SQLite(), paste0(here(), "/data/footy-tipper-db.sqlite"))

# Write the processed data into the SQLite database, 
# overwrite the table if it already exists
dbWriteTable(con, "footy_tipping_data", footy_tipping_data, overwrite = T)
dbWriteTable(con, "training_data", training_data, overwrite = T)
dbWriteTable(con, "inference_data", inference_data, overwrite = T)

# Disconnect from the SQLite database to ensure no other operations are unintentionally performed
dbDisconnect(con)
