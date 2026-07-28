# data params
start_year <- as.integer(Sys.getenv("FOOTY_TIPPER_START_YEAR", unset = "2008"))
end_year_env <- Sys.getenv("FOOTY_TIPPER_END_YEAR", unset = "")
if (end_year_env == "") {
  end_year <- as.integer(format(Sys.Date(), "%Y"))
} else {
  end_year <- as.integer(end_year_env)
}

if (is.na(start_year) || is.na(end_year) || end_year < start_year) {
  stop("Invalid year range. Check FOOTY_TIPPER_START_YEAR and FOOTY_TIPPER_END_YEAR.")
}

prep_mode <- tolower(Sys.getenv("FOOTY_TIPPER_PREP_MODE", unset = "full"))
if (!prep_mode %in% c("full", "train", "infer")) {
  stop("Invalid FOOTY_TIPPER_PREP_MODE. Use one of: full, train, infer.")
}

infer_context_years <- as.integer(Sys.getenv("FOOTY_TIPPER_INFER_CONTEXT_YEARS", unset = "1"))
if (is.na(infer_context_years) || infer_context_years < 0) {
  stop("Invalid FOOTY_TIPPER_INFER_CONTEXT_YEARS. Use an integer >= 0.")
}

if (prep_mode == "infer") {
  infer_start_year <- max(start_year, end_year - infer_context_years)
  year_span <- infer_start_year:end_year
} else {
  year_span <- start_year:end_year
}

form_period <- 5
pipeline <- "binomial"
require_odds_env <- tolower(Sys.getenv("FOOTY_TIPPER_REQUIRE_ODDS", unset = "false"))
if (!require_odds_env %in% c("1", "true", "yes", "y", "0", "false", "no", "n")) {
  stop("Invalid FOOTY_TIPPER_REQUIRE_ODDS. Use true/false.")
}
use_odds <- require_odds_env %in% c("1", "true", "yes", "y")

# elo params
elo_init <- 1500
k_val <- 70
carry_over <- 0.5

include_performance <- TRUE
include_performance_env <- tolower(Sys.getenv("FOOTY_TIPPER_INCLUDE_PERFORMANCE", unset = "true"))
include_performance <- include_performance_env %in% c("1", "true", "yes", "y")

# Data source for the feed_cache_* tables:
# - "python": caches are written upstream by the nrl.com ingestion
#   (pipeline/common/nrl_data); R reads them without fetching.
# - "feed": legacy XML feed fetch inside get_data() (requires PASSWORD/BASE_URL).
feed_source <- tolower(Sys.getenv("FOOTY_TIPPER_FEED_SOURCE", unset = "python"))
if (!feed_source %in% c("python", "feed")) {
  stop("Invalid FOOTY_TIPPER_FEED_SOURCE. Use one of: python, feed.")
}
