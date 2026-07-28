from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys

from dotenv import load_dotenv

from pipeline.common import console


def _log(message: str) -> None:
    # Doubles as the ingestion progress callback; routed through the marker
    # channel so the parent CLI shows it on the live status line.
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


def _default_year(value: str | None, fallback: int) -> int:
    try:
        parsed = int(value) if value is not None and value != "" else fallback
    except Exception:
        parsed = fallback
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline/lineups.py",
        description="Fetch NRL team-list and late-mail lineup snapshots into SQLite.",
    )
    parser.add_argument(
        "--mode",
        choices=["recent", "backfill"],
        default=None,
        help="recent=topic hub focused pull; backfill=include sitemap crawl.",
    )
    parser.add_argument("--start-year", type=int, default=None, help="Lower year bound for sitemap filtering.")
    parser.add_argument("--end-year", type=int, default=None, help="Upper year bound for sitemap filtering.")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Max article URLs to fetch in this run.",
    )
    parser.add_argument(
        "--include-sitemap-in-recent",
        action="store_true",
        help="In recent mode, also include sitemap URLs (slower but broader).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero on ingestion errors. Default is fail-soft.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Optional SQLite path override (defaults to data/footy-tipper-db.sqlite).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / "secrets.env")

    enabled = _to_bool(os.getenv("FOOTY_TIPPER_LINEUPS_ENABLED"), True)
    if not enabled:
        _log("Lineup ingestion disabled via FOOTY_TIPPER_LINEUPS_ENABLED=false. Skipping.")
        return 0

    mode = args.mode or os.getenv("FOOTY_TIPPER_LINEUPS_MODE", "recent").strip().lower()
    if mode not in {"recent", "backfill"}:
        _log(f"Invalid lineup mode '{mode}'. Falling back to 'recent'.")
        mode = "recent"

    start_year = args.start_year
    end_year = args.end_year
    if start_year is None:
        start_year = _default_year(os.getenv("FOOTY_TIPPER_START_YEAR"), 2008)
    if end_year is None:
        end_year_env = os.getenv("FOOTY_TIPPER_END_YEAR")
        end_year = _default_year(end_year_env, dt.datetime.now(dt.timezone.utc).year)
    if end_year < start_year:
        end_year = start_year

    max_articles = args.max_articles
    if max_articles is None:
        default_max = "80" if mode == "recent" else "500"
        max_articles = _default_year(os.getenv("FOOTY_TIPPER_LINEUPS_MAX_ARTICLES"), int(default_max))
    max_articles = max(1, int(max_articles))

    include_sitemap_in_recent = args.include_sitemap_in_recent or _to_bool(
        os.getenv("FOOTY_TIPPER_LINEUPS_INCLUDE_SITEMAP_IN_RECENT"),
        False,
    )
    strict_mode = bool(args.strict) or _to_bool(os.getenv("FOOTY_TIPPER_LINEUPS_STRICT"), False)

    try:
        from pipeline.common.lineups.ingest import IngestionConfig, run_lineup_ingestion
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or "").strip()
        optional_missing = {"bs4", "lxml"}
        if missing and missing not in optional_missing:
            raise

        _log(
            "Lineup ingestion dependencies are missing "
            f"(missing module: {missing or 'unknown'}). "
            "Install BeautifulSoup + lxml for scraping support."
        )
        if strict_mode:
            return 1
        _log("Fail-soft mode enabled. Continuing without lineup refresh.")
        return 0

    db_path = pathlib.Path(args.db_path) if args.db_path else (project_root / "data" / "footy-tipper-db.sqlite")
    cfg = IngestionConfig(
        mode=mode,
        start_year=start_year,
        end_year=end_year,
        max_articles=max_articles,
        include_sitemap_in_recent=include_sitemap_in_recent,
    )

    _log(
        "Running lineup ingestion with "
        f"mode={cfg.mode}, years={cfg.start_year}-{cfg.end_year}, max_articles={cfg.max_articles}, "
        f"include_sitemap_in_recent={cfg.include_sitemap_in_recent}"
    )

    try:
        stats = run_lineup_ingestion(db_path=db_path, cfg=cfg, progress_callback=_log)
    except Exception as exc:
        _log(f"Lineup ingestion failed: {exc}")
        if strict_mode:
            return 1
        _log("Fail-soft mode enabled. Continuing without lineup refresh.")
        return 0

    _log(
        "Lineup ingestion summary: "
        f"candidates={stats.get('url_candidates', 0)}, "
        f"processed={stats.get('urls_processed', 0)}, "
        f"snapshots_inserted={stats.get('snapshots_inserted', 0)}, "
        f"hash_skips={stats.get('snapshots_skipped_existing_hash', 0)}, "
        f"nrl_skips={stats.get('articles_skipped_not_nrl', 0)}, "
        f"entries_inserted={stats.get('entries_inserted', 0)}, "
        f"parse_failures={stats.get('parse_failures', 0)}"
    )

    if stats.get("errors"):
        _log(f"Lineup ingestion captured {len(stats['errors'])} errors.")
        if strict_mode:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
