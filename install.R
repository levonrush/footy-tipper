# Setting CRAN mirror to use
local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.ms.unimelb.edu.au/"
       options(repos = r)
})

packages <- c(
  "dplyr"
  , "tidyr"
  , "lubridate"
  , "elo"
  , "randomForest"
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

install_if_missing <- function(package){
  print(paste("Installing", package))
  if(!package %in% installed.packages()){
    tryCatch({
      install.packages(package, dependencies = NA)
    }, error = function(e) {
      print(paste("Failed to install", package))
      print(e)
      stop("Stopping due to failure in package installation.")
    })
  }
}


invisible(lapply(packages, install_if_missing))
