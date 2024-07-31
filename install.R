# Setting CRAN mirror to use
local({r <- getOption("repos")
       r["CRAN"] <- "https://cloud.r-project.org/"
       options(repos = r)
})

# List of packages to install
packages <- c(
  "dplyr", "tidyr", "lubridate", "elo", "here",
  "xml2", "janitor", "zoo", "purrr", "readr",
  "stringr", "forcats", "tibble", "RSQLite",
  "DBI", "dotenv"
)

# Function to install a specific version of a package using 'remotes'
install_specific_version <- function(package, version) {
  message(paste("Installing", package, "version", version))
  tryCatch({
    remotes::install_version(package, version = version)
  }, error = function(e) {
    message(paste("Failed to install", package, "version", version))
    print(e)
    stop("Stopping due to failure in package installation.")
  })
}

# Function to install a package if it's missing
install_if_missing <- function(package) {
  if (!package %in% installed.packages()[,"Package"]) {
    message(paste("Installing package:", package))
    tryCatch({
      install.packages(package, dependencies = NA)
    }, error = function(e) {
      message(paste("Failed to install package:", package))
      print(e)
      stop("Stopping due to failure in package installation.")
    })
  } else {
    message(paste("Package already installed:", package))
  }
}
