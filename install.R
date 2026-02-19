# Setting CRAN mirror to use
local({r <- getOption("repos")
       r["CRAN"] <- "https://cloud.r-project.org/"
       options(repos = r)
})

# Ensure installs use a writable user library, not system R library.
user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(user_lib, .libPaths()))
}

# List of packages to install
packages <- c(
  "dplyr", "tidyr", "lubridate", "elo", "here",
  "xml2", "janitor", "zoo", "purrr", "readr",
  "stringr", "forcats", "tibble", "RSQLite",
  "DBI", "dotenv"
)

# Function to install a package if it's missing
install_if_missing <- function(package) {
  if (!package %in% installed.packages()[,"Package"]) {
    message(paste("Installing package:", package))
    tryCatch({
      install.packages(package, dependencies = NA, lib = .libPaths()[1])
    }, error = function(e) {
      message(paste("Failed to install package:", package))
      print(e)
      stop("Stopping due to failure in package installation.")
    })
  } else {
    message(paste("Package already installed:", package))
  }
}

for (pkg in packages) {
  install_if_missing(pkg)
}
