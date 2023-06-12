detect_set_environment <- function(prod_run) {
  # Perform the authentication first
  drive_auth(path = "/footy-tipper/footy-tipper-c5bcb9639ee2.json")
  
  # Check if we are in Docker
  if (Sys.getenv("DOCKER") == "true") {
    prod_run <- Sys.getenv("PROD_RUN") == "T"
  }
  
  # Check if we are in RStudio or VSCode, if not already in Docker
  else if (Sys.getenv("RSTUDIO") == "1" || Sys.getenv("VSCODE_R_SESSION") == "true") {
    # prod_run remains unchanged
  }
  
  # Default case: assume we're in a production environment
  else {
    prod_run <- Sys.getenv("PROD_RUN") == "T"
  }
  
  # Return the prod_run flag
  return(prod_run)
}
