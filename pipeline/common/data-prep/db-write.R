get_table_fields <- function(con, table_name) {
  dbListFields(con, table_name)
}

sqlite_type_from_vector <- function(x) {
  if (inherits(x, c("POSIXct", "POSIXt", "Date"))) {
    return("TEXT")
  }
  if (is.character(x)) {
    return("TEXT")
  }
  if (is.integer(x) || is.logical(x)) {
    return("INTEGER")
  }
  if (is.numeric(x)) {
    return("REAL")
  }
  return("TEXT")
}

add_missing_table_columns <- function(con, table_name, data, columns) {
  if (length(columns) == 0) {
    return(invisible(NULL))
  }

  table_sql <- dbQuoteIdentifier(con, table_name)
  for (col_name in columns) {
    col_sql <- dbQuoteIdentifier(con, col_name)
    col_type <- sqlite_type_from_vector(data[[col_name]])
    dbExecute(
      con,
      paste0(
        "ALTER TABLE ", table_sql,
        " ADD COLUMN ", col_sql, " ", col_type
      )
    )
  }
  invisible(NULL)
}

align_incoming_to_table <- function(con, table_name, data, key_col = "game_id") {
  if (!dbExistsTable(con, table_name)) {
    return(data)
  }

  existing_cols <- get_table_fields(con, table_name)
  incoming_cols <- names(data)

  if (!(key_col %in% existing_cols) || !(key_col %in% incoming_cols)) {
    stop(paste0(
      "Incremental write for table '", table_name,
      "' requires key column '", key_col,
      "' in both existing and incoming data."
    ))
  }

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

upsert_by_game_id <- function(con, table_name, data, key_col = "game_id") {
  if (!dbExistsTable(con, table_name)) {
    dbWriteTable(con, table_name, data, overwrite = TRUE)
    return(invisible(NULL))
  }

  if (nrow(data) == 0) {
    return(invisible(NULL))
  }

  ordered_data <- align_incoming_to_table(con, table_name, data, key_col = key_col)

  temp_table <- paste0(table_name, "_incoming")
  if (dbExistsTable(con, temp_table)) {
    dbRemoveTable(con, temp_table)
  }

  dbWriteTable(con, temp_table, ordered_data, overwrite = TRUE)

  table_sql <- dbQuoteIdentifier(con, table_name)
  temp_sql <- dbQuoteIdentifier(con, temp_table)
  key_sql <- dbQuoteIdentifier(con, key_col)

  dbExecute(
    con,
    paste0(
      "DELETE FROM ", table_sql,
      " WHERE ", key_sql,
      " IN (SELECT ", key_sql,
      " FROM ", temp_sql,
      ")"
    )
  )

  dbExecute(
    con,
    paste0("INSERT INTO ", table_sql, " SELECT * FROM ", temp_sql)
  )

  dbRemoveTable(con, temp_table)
  invisible(NULL)
}

append_odds_snapshots <- function(con, footy_tipping_data) {
  snapshot_ts <- format(Sys.time(), tz = "UTC", usetz = TRUE)

  odds_snapshots <- footy_tipping_data %>%
    filter(game_state_name == "Pre Game") %>%
    select(
      any_of(c(
        "game_id", "competition_year", "round_id", "round_name", "start_time",
        "team_home", "team_away",
        "team_head_to_head_odds_home", "team_line_odds_home", "team_line_amount_home",
        "team_head_to_head_odds_away", "team_line_odds_away", "team_line_amount_away",
        "total_line", "total_over_odds", "total_under_odds"
      ))
    ) %>%
    mutate(snapshot_time_utc = snapshot_ts)

  if (nrow(odds_snapshots) == 0) {
    return(invisible(NULL))
  }

  # Align both ways so new market columns (totals) auto-ALTER onto the
  # pre-existing snapshot table instead of failing the append.
  if (dbExistsTable(con, "odds_snapshots")) {
    odds_snapshots <- align_cache_columns_to_table(con, "odds_snapshots", odds_snapshots)
  }

  dbWriteTable(con, "odds_snapshots", odds_snapshots, append = dbExistsTable(con, "odds_snapshots"))
  invisible(NULL)
}

write_prepared_tables <- function(con, prep_mode, footy_tipping_data, training_data, inference_data) {
  if (prep_mode %in% c("full", "train")) {
    dbWriteTable(con, "footy_tipping_data", footy_tipping_data, overwrite = TRUE)
    dbWriteTable(con, "training_data", training_data, overwrite = TRUE)
    dbWriteTable(con, "inference_data", inference_data, overwrite = TRUE)
    append_odds_snapshots(con, footy_tipping_data)
    return(invisible(NULL))
  }

  if (prep_mode == "infer") {
    upsert_by_game_id(con, "footy_tipping_data", footy_tipping_data)
    upsert_by_game_id(con, "training_data", training_data)
    dbWriteTable(con, "inference_data", inference_data, overwrite = TRUE)
    append_odds_snapshots(con, footy_tipping_data)
    return(invisible(NULL))
  }

  stop("Unsupported prep mode.")
}
