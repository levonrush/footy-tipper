from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys

from dotenv import load_dotenv

from pipeline.common import console


def _log(message: str) -> None:
    console.emit_progress(message)


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline/nrl_data.py",
        description=(
            "nrl.com data ingestion: fixtures/match stats into SQLite and the "
            "feed_cache_* tables (feed-independent data source)."
        ),
    )
    parser.add_argument(
        "action",
        choices=["refresh", "backfill", "validate"],
        help=(
            "refresh=current season draw + match centres + derived caches; "
            "backfill=historical match centres (2012+); "
            "validate=parity report vs cached feed history (no writes)."
        ),
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--season", type=int, default=None, help="refresh: season override.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Cap on match centre pages fetched this run.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="validate: CSV report destination (default reports/nrl_data_parity_<date>.csv).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero on ingestion errors. Default is fail-soft.",
    )
    parser.add_argument("--db-path", default=None, help="SQLite path override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / "secrets.env")

    enabled = _to_bool(os.getenv("FOOTY_TIPPER_NRL_DATA_ENABLED"), True)
    if not enabled and args.action == "refresh":
        _log("nrl.com ingestion disabled via FOOTY_TIPPER_NRL_DATA_ENABLED=false. Skipping.")
        return 0

    db_path = (
        pathlib.Path(args.db_path)
        if args.db_path
        else project_root / "data" / "footy-tipper-db.sqlite"
    )
    venue_csv = project_root / "data" / "reference" / "venue_locations.csv"
    current_year = dt.datetime.now(dt.timezone.utc).year

    max_pages = args.max_pages
    if max_pages is None:
        env_pages = os.getenv("FOOTY_TIPPER_NRL_DATA_MAX_PAGES", "").strip()
        max_pages = int(env_pages) if env_pages.isdigit() else None

    try:
        from pipeline.common.nrl_data import refresh as refresh_module
        from pipeline.common.nrl_data import validate as validate_module
    except ModuleNotFoundError as exc:
        _log(f"nrl.com ingestion dependencies missing ({exc}).")
        return 1 if args.strict else 0

    try:
        if args.action == "refresh":
            result = refresh_module.refresh_season(
                db_path=db_path,
                season=args.season or current_year,
                venue_csv=venue_csv,
                max_pages=max_pages,
            )
        elif args.action == "backfill":
            result = refresh_module.backfill_seasons(
                db_path=db_path,
                start_year=args.start_year or refresh_module.BACKFILL_FIRST_SEASON,
                end_year=args.end_year or current_year,
                venue_csv=venue_csv,
                max_pages=max_pages,
            )
        else:
            result = validate_module.validate_seasons(
                db_path=db_path,
                start_year=args.start_year or 2013,
                end_year=args.end_year or (current_year - 1),
                report_path=args.report_path,
                venue_csv=venue_csv,
            )
    except Exception as exc:
        _log(f"nrl.com ingestion {args.action} failed: {exc}")
        if args.strict:
            return 1
        _log("Fail-soft mode enabled. Continuing on cached data.")
        return 0

    if args.action in {"refresh", "backfill"} and _to_bool(
        os.getenv("FOOTY_TIPPER_WEATHER_ENABLED"), True
    ):
        try:
            from pipeline.common.nrl_data import weather

            weather.fetch_weather_for_games(
                db_path, refresh_upcoming=(args.action == "refresh")
            )
        except Exception as exc:
            _log(f"Weather fetch skipped ({exc}).")

    errors = result.get("errors") or []
    if errors:
        _log(f"nrl.com ingestion captured {len(errors)} errors (first: {errors[0]}).")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
