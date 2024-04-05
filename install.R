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
  # Note: 'randomForest' is intentionally omitted here and handled separately
)

# Ensure 'remotes' package is installed first for installing specific versions
install.packages("remotes")

# Dictionary mapping packages to specific versions for installation
specific_package_versions <- list(
  randomForest = "4.6-14" # Example, specify the version you need
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

# Install packages with specific versions
lapply(names(specific_package_versions), function(pkg) {
  version <- specific_package_versions[[pkg]]
  install_specific_version(pkg, version)
})

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

# Install remaining packages that are not listed in 'specific_package_versions'
invisible(lapply(packages, install_if_missing))
