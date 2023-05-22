# List of packages
pkg <- c(
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
  , "xml2"
  , "janitor"
  , "zoo"
)

# Function to load packages
load_or_install <- function(packages){
  
  for(package in packages){
    
    suppressMessages(library(package, character.only = TRUE))
  
  }
  
}

# Use the function
load_packages(pkg)
