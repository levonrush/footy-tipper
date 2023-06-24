# List of packages
pkg <- c(
  "dplyr"
  , "tidyr"
  # , "ggplot2"
  , "lubridate"
  , "elo"
  , "randomForest"
  # , "caret"
  # , "OptimalCutpoints"
  # , "parallel"
  # , "doParallel"
  , "here"
  , "skimr"
  , "googledrive"
  , "scales"
  , "xml2"
  , "janitor"
  , "zoo"
  , "rmarkdown"
  , "purrr"
  , "readr"
  , "stringr"
  , "forcats"
  # , "MLmetrics"
  , "tibble"
  , "dotenv"
  , "RSQLite"
  , "DBI"
  , "reticulate"
)

# Function to load packages
load_packages <- function(packages) {
  for (package in packages) {
    suppressMessages(library(package, character.only = TRUE))
  }
}

# Use the function
load_packages(pkg)
