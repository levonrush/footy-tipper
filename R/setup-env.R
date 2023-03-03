setup_env <- function(){
  
  library(here)
  library(skimr)
  library(Epi)
  library(pROC)
  library(googledrive)
  
  i_am("pipeline/footy-tipper.Rmd")
  
  drive_auth()
  
  helper_functions <- list.files(paste0(here(), "/R"), pattern = "*.R$", 
                                 full.names = TRUE, ignore.case = TRUE)
  
  sapply(helper_functions, source, .GlobalEnv)
  
}