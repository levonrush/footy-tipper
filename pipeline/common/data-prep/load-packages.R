# List of packages
pkg <- c(
  "dplyr"
  , "tidyr"
  , "lubridate"
  , "elo"
  , "here"
  , "xml2"
  , "janitor"
  , "zoo"
  , "purrr"
  , "readr"
  , "stringr"
  , "forcats"
  , "tibble"
  , "RSQLite"
  , "DBI"
  , "dotenv"
)

# Function to load packages
load_packages <- function(packages) {
  for (package in packages) {
    suppressMessages(library(package, character.only = TRUE))
  }
}

# Use the function
load_packages(pkg)
