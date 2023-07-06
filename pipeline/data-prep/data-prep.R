i_am("pipeline/data-prep/data-prep.R")

# find and load all the helper functions for the project
data_prep_functions <- list.files(
    paste0(here(), "/pipeline/data-prep/functions"),
    pattern = "*.R$",
    full.names = TRUE, ignore.case = TRUE
)

sapply(data_prep_functions, source, .GlobalEnv)

footy_tipping_data <- data_pipeline(
    year_span, pipeline = "binomial",
    form_period, carry_over, k_val, elo_init
)

# Connect to the SQLite database
con <- dbConnect(SQLite(), paste0(here(), "/data/footy-tipper-db.sqlite"))

# Write your dataframes
dbWriteTable(con, "footy_tipping_data", footy_tipping_data, overwrite = T)

# Disconnect from the SQLite database
dbDisconnect(con)
