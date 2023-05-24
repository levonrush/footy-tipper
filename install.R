# Setting CRAN mirror to use
local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.r-project.org" 
       options(repos = r)
})

packages <- c(
  "dplyr"
  , "tidyr"
  , "ggplot2"
  , "lubridate"
  , "elo"
  , "randomForest"
  , "caret"
  , "OptimalCutpoints"
  , "parallel"
  , "doParallel"
  , "here"
  , "skimr"
  # , "Epi"
  # , "pROC"
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
  , "MLmetrics"
)

install_if_missing <- function(package){
  print(paste("Installing", package))
  if(!package %in% installed.packages()){
    tryCatch({
      install.packages(package, dependencies = NA)
    }, error = function(e) {
      print(paste("Failed to install", package))
      print(e)
    })
  }
}

invisible(lapply(packages, install_if_missing))
