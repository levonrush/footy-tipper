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

# Whether a package can actually be loaded, which is the property that matters.
# Presence in installed.packages() is not enough: a half-finished upgrade leaves
# the directory in place with unloadable shared objects.
can_load <- function(package) {
  suppressWarnings(suppressMessages(
    requireNamespace(package, quietly = TRUE)
  ))
}

# Only touch packages that cannot be loaded. Installing over a working package
# pulls its dependencies too, and a failed dependency build removes and restores
# the original -- which is how a run that needed no installs at all can still
# take down dplyr on the way through.
needed <- Filter(function(p) !can_load(p), packages)

if (length(needed) == 0) {
  message("All R packages present and loadable; nothing to install.")
} else {
  message(paste("Installing missing R packages:", paste(needed, collapse = ", ")))
  for (pkg in needed) {
    tryCatch({
      install.packages(pkg, dependencies = NA, lib = .libPaths()[1])
    }, error = function(e) {
      message(paste("Install attempt failed for:", pkg))
      print(e)
    })
  }

  # Re-check rather than trusting the installer's exit status. A package that
  # loads is fine even if its install printed warnings; one that still will not
  # load is fatal, and the message says which and how to fix it by hand.
  still_broken <- Filter(function(p) !can_load(p), needed)
  if (length(still_broken) > 0) {
    stop(paste0(
      "These R packages cannot be loaded: ",
      paste(still_broken, collapse = ", "),
      "\nInstall them manually in R with:\n",
      "  install.packages(c(",
      paste(sprintf('"%s"', still_broken), collapse = ", "),
      "))"
    ))
  }
  message("All R packages present and loadable.")
}
