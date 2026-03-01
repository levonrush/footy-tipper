normalize_year_vector <- function(years) {
  normalized <- suppressWarnings(as.integer(years))
  normalized <- normalized[!is.na(normalized)]
  sort(unique(normalized))
}

format_year_vector <- function(years) {
  normalized <- normalize_year_vector(years)
  if (length(normalized) == 0) {
    return("none")
  }
  paste(normalized, collapse = ", ")
}

feed_cache_table_name <- function(feed_name) {
  paste0("feed_cache_", gsub("[^a-z0-9_]+", "_", tolower(feed_name)))
}

load_cached_feed <- function(con, table_name, years = NULL) {
  if (!dbExistsTable(con, table_name)) {
    return(tibble())
  }

  data <- dbReadTable(con, table_name) %>% as_tibble()
  if (is.null(years) || !("competition_year" %in% names(data))) {
    return(data)
  }

  data %>%
    mutate(competition_year = suppressWarnings(as.integer(competition_year))) %>%
    filter(competition_year %in% normalize_year_vector(years))
}

align_cache_columns_to_table <- function(con, table_name, data) {
  if (!dbExistsTable(con, table_name)) {
    return(data)
  }

  existing_cols <- get_table_fields(con, table_name)
  incoming_cols <- names(data)

  cols_missing_in_table <- setdiff(incoming_cols, existing_cols)
  add_missing_table_columns(con, table_name, data, cols_missing_in_table)

  existing_cols <- get_table_fields(con, table_name)
  cols_missing_in_data <- setdiff(existing_cols, incoming_cols)
  if (length(cols_missing_in_data) > 0) {
    for (col_name in cols_missing_in_data) {
      data[[col_name]] <- NA
    }
  }

  data %>% select(all_of(existing_cols))
}

replace_cached_feed_year <- function(con, table_name, year, data) {
  year <- suppressWarnings(as.integer(year))
  if (is.na(year)) {
    stop("Feed cache replacement requires a valid competition_year.")
  }
  if (!("competition_year" %in% names(data))) {
    stop(paste0("Feed cache rows for table '", table_name, "' must include competition_year."))
  }

  normalized <- data %>%
    mutate(competition_year = suppressWarnings(as.integer(competition_year))) %>%
    filter(competition_year == year)

  if (nrow(normalized) == 0) {
    return(invisible(FALSE))
  }

  if (!dbExistsTable(con, table_name)) {
    dbWriteTable(con, table_name, normalized, overwrite = TRUE)
    return(invisible(TRUE))
  }

  normalized <- align_cache_columns_to_table(con, table_name, normalized)
  table_sql <- dbQuoteIdentifier(con, table_name)
  dbExecute(
    con,
    paste0("DELETE FROM ", table_sql, " WHERE competition_year = ?"),
    params = list(year)
  )
  dbWriteTable(con, table_name, normalized, append = TRUE)
  invisible(TRUE)
}

refresh_feed_cache_years <- function(con, table_name, years, fetcher, feed_label) {
  years <- normalize_year_vector(years)
  if (length(years) == 0) {
    return(invisible(NULL))
  }

  for (year in years) {
    had_cache <- nrow(load_cached_feed(con, table_name, years = year)) > 0
    fetched <- tryCatch(
      fetcher(year),
      error = function(e) {
        message(
          paste0(
            "Get Data: Refresh failed for ", feed_label, " ", year,
            ". ", conditionMessage(e)
          )
        )
        NULL
      }
    )

    if (is.null(fetched) || nrow(fetched) == 0) {
      if (had_cache) {
        message(
          paste0(
            "Get Data: No fresh ", feed_label, " rows returned for ", year,
            ". Keeping cached copy."
          )
        )
      } else {
        message(
          paste0(
            "Get Data: No ", feed_label, " data available for ", year,
            "."
          )
        )
      }
      next
    }

    replace_cached_feed_year(con, table_name, year, fetched)
    message(
      paste0(
        "Get Data: Cached ", nrow(fetched), " ", feed_label,
        " rows for ", year, "."
      )
    )
  }

  invisible(NULL)
}

year_max_round_lookup <- function(data, state_name = NULL) {
  if (nrow(data) == 0 || !all(c("competition_year", "round_id") %in% names(data))) {
    return(setNames(integer(0), character(0)))
  }

  working <- data %>%
    mutate(
      competition_year = suppressWarnings(as.integer(competition_year)),
      round_id = suppressWarnings(as.integer(round_id))
    ) %>%
    filter(!is.na(competition_year), !is.na(round_id))

  if (!is.null(state_name) && "game_state_name" %in% names(working)) {
    working <- working %>% filter(game_state_name == state_name)
  }

  if (nrow(working) == 0) {
    return(setNames(integer(0), character(0)))
  }

  summary <- working %>%
    group_by(competition_year) %>%
    summarise(max_round = max(round_id, na.rm = TRUE), .groups = "drop") %>%
    arrange(competition_year)

  stats::setNames(as.integer(summary$max_round), as.character(summary$competition_year))
}

lookup_year_value <- function(values, year) {
  year_key <- as.character(suppressWarnings(as.integer(year)))
  if (!nzchar(year_key) || !(year_key %in% names(values))) {
    return(NA_integer_)
  }
  suppressWarnings(as.integer(values[[year_key]]))
}

resolve_fixture_refresh_years <- function(requested_years, cached_years, current_year, refresh_mode = "smart") {
  requested_years <- normalize_year_vector(requested_years)
  cached_years <- normalize_year_vector(cached_years)
  current_year <- suppressWarnings(as.integer(current_year))

  if (identical(refresh_mode, "full")) {
    return(requested_years)
  }

  refresh_years <- setdiff(requested_years, cached_years)
  if (!is.na(current_year) && current_year %in% requested_years) {
    refresh_years <- c(refresh_years, current_year)
  }

  normalize_year_vector(refresh_years)
}

resolve_ladder_refresh_years <- function(
  requested_years,
  cached_years,
  fixture_rounds,
  cached_rounds,
  current_year,
  refresh_mode = "smart"
) {
  requested_years <- normalize_year_vector(requested_years)
  cached_years <- normalize_year_vector(cached_years)
  current_year <- suppressWarnings(as.integer(current_year))

  if (identical(refresh_mode, "full")) {
    return(requested_years)
  }

  refresh_years <- setdiff(requested_years, cached_years)
  covered_years <- intersect(requested_years, cached_years)
  for (year in covered_years) {
    fixture_round <- lookup_year_value(fixture_rounds, year)
    cached_round <- lookup_year_value(cached_rounds, year)
    if (!is.na(fixture_round) && (is.na(cached_round) || cached_round < fixture_round)) {
      refresh_years <- c(refresh_years, year)
    }
  }

  if (!is.na(current_year) && current_year %in% requested_years) {
    refresh_years <- c(refresh_years, current_year)
  }

  normalize_year_vector(refresh_years)
}

resolve_performance_refresh_years <- function(
  requested_years,
  cached_years,
  final_fixture_rounds,
  cached_rounds,
  current_year,
  refresh_mode = "smart"
) {
  requested_years <- normalize_year_vector(requested_years)
  cached_years <- normalize_year_vector(cached_years)
  current_year <- suppressWarnings(as.integer(current_year))

  if (identical(refresh_mode, "full")) {
    return(requested_years)
  }

  refresh_years <- integer(0)
  for (year in requested_years) {
    final_round <- lookup_year_value(final_fixture_rounds, year)
    cached_round <- lookup_year_value(cached_rounds, year)
    year_cached <- year %in% cached_years

    if (!year_cached && !is.na(final_round) && final_round > 0) {
      refresh_years <- c(refresh_years, year)
      next
    }

    if (!is.na(final_round) && final_round > 0 && (is.na(cached_round) || cached_round < final_round)) {
      refresh_years <- c(refresh_years, year)
    }
  }

  if (!is.na(current_year) && current_year %in% requested_years) {
    refresh_years <- c(refresh_years, current_year)
  }

  normalize_year_vector(refresh_years)
}
