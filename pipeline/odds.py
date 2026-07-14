from __future__ import annotations

import argparse
import pathlib
import sys

from dotenv import load_dotenv


def _log(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline/odds.py",
        description=(
            "Odds ingestion: aussportsbetting historical backfill and Betfair "
            "live pre-game snapshots into odds_history + the fixture cache."
        ),
    )
    parser.add_argument(
        "action",
        choices=["live", "backfill"],
        help="live=Betfair snapshot for upcoming games; backfill=historical xlsx.",
    )
    parser.add_argument("--xlsx-path", default=None, help="backfill: local workbook override.")
    parser.add_argument("--url", default=None, help="backfill: workbook URL override.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero on errors. Default is fail-soft.",
    )
    parser.add_argument("--db-path", default=None, help="SQLite path override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / "secrets.env")

    db_path = (
        pathlib.Path(args.db_path)
        if args.db_path
        else project_root / "data" / "footy-tipper-db.sqlite"
    )

    try:
        if args.action == "backfill":
            from pipeline.common.odds import aussportsbetting

            result = aussportsbetting.backfill(
                db_path=db_path, url=args.url, xlsx_path=args.xlsx_path
            )
        else:
            from pipeline.common.odds import betfair

            result = betfair.snapshot_live_odds(db_path=db_path)
    except Exception as exc:
        _log(f"Odds {args.action} failed: {exc}")
        if args.strict:
            return 1
        _log("Fail-soft mode enabled. Continuing without odds update.")
        return 0

    if args.strict and result.get("status") in {"failed"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
