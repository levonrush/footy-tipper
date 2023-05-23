# List of packages
packages <- c(
    "tidyverse"
    , "lubridate"
    , "elo"
    , "randomForest"
    , "caret"
    , "OptimalCutpoints"
    , "parallel"
    , "doParallel"
    , "here"
    , "skimr"
    , "Epi"
    , "pROC"
    , "googledrive"
    , "scales"
)

install_if_missing <- function(package){
  if(!package %in% installed.packages()){
    install.packages(package, dependencies = TRUE)
  }
}

invisible(lapply(packages, install_if_missing))
