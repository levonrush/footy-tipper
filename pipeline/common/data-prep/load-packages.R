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

# Set CRAN mirror for runtime install fallback.
local({
  r <- getOption("repos")
  r["CRAN"] <- "https://cloud.r-project.org/"
  options(repos = r)
})

# Function to load packages
load_packages <- function(packages) {
  user_lib <- Sys.getenv("R_LIBS_USER")
  if (nzchar(user_lib)) {
    dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(user_lib, .libPaths()))
  }

  for (package in packages) {
    if (!requireNamespace(package, quietly = TRUE)) {
      message(paste0("Package '", package, "' not found. Installing..."))
      tryCatch(
        {
          install.packages(package, dependencies = NA, lib = .libPaths()[1])
        },
        error = function(e) {
          stop(paste0("Failed to install required R package '", package, "': ", conditionMessage(e)))
        }
      )
    }

    suppressMessages(library(package, character.only = TRUE))
  }
}

# Use the function
load_packages(pkg)
