from __future__ import annotations

import argparse
import os
import pathlib
import sys

from dotenv import load_dotenv

from pipeline.common import console


def _log(message: str) -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline/odds.py",
        description=(
            "Odds ingestion: aussportsbetting historical backfill and live "
            "pre-game snapshots into odds_history + the fixture cache."
        ),
    )
    parser.add_argument(
        "action",
        choices=["live", "backfill"],
        help="live=current provider snapshot; backfill=historical xlsx.",
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


def _snapshot_live_odds(db_path: pathlib.Path) -> dict:
    """Run the configured live provider with the production-safe fallback."""
    configured = (
        os.environ.get("FOOTY_TIPPER_LIVE_ODDS_PROVIDER") or "the_odds_api"
    ).strip().lower()
    aliases = {
        "auto": "the_odds_api",
        "odds_api": "the_odds_api",
        "the-odds-api": "the_odds_api",
        "the_odds_api": "the_odds_api",
        "betfair": "betfair",
    }
    provider = aliases.get(configured)
    if provider is None:
        raise ValueError(
            "FOOTY_TIPPER_LIVE_ODDS_PROVIDER must be 'the_odds_api' or 'betfair'"
        )

    from pipeline.common.odds import betfair, the_odds_api
    from pipeline.ops.odds_gate import current_round_odds_coverage

    runners = (
        [
            ("the_odds_api", the_odds_api.snapshot_live_odds),
            ("betfair", betfair.snapshot_live_odds),
        ]
        if provider == "the_odds_api"
        else [("betfair", betfair.snapshot_live_odds)]
    )
    attempts: list[dict] = []
    successes: list[dict] = []
    primary_game_ids: set[int] = set()
    for index, (name, runner) in enumerate(runners):
        if index:
            _log(f"[odds] Falling back to {name}.")
        kwargs = {"db_path": db_path}
        if name == "betfair" and primary_game_ids:
            # Retain the selected primary bookmaker on covered fixtures; the
            # fallback is only for gaps in the primary response.
            kwargs["exclude_game_ids"] = set(primary_game_ids)
        result = runner(**kwargs)
        attempts.append(result)
        games_updated = int(result.get("games_updated", 0))
        if result.get("status") != "completed" or games_updated <= 0:
            continue

        successes.append(result)
        primary_game_ids.update(int(value) for value in result.get("game_ids_updated", ()))
        if name != "the_odds_api":
            break
        coverage = current_round_odds_coverage(db_path)
        if coverage.complete:
            break
        if coverage.error:
            detail = f"unavailable ({coverage.error})"
        else:
            detail = f"partial ({coverage.covered_games}/{coverage.total_games})"
        _log(
            f"[odds] Primary current-round H2H coverage is {detail}; "
            "trying Betfair for uncovered fixtures."
        )

    if successes:
        combined = dict(successes[0])
        if len(successes) > 1:
            numeric_fields = (
                "snapshots_inserted",
                "h2h_games",
                "line_games",
                "totals_games",
            )
            for field in numeric_fields:
                combined[field] = sum(int(result.get(field, 0)) for result in successes)
            combined_ids = {
                int(value)
                for result in successes
                for value in result.get("game_ids_updated", ())
            }
            combined["games_updated"] = (
                len(combined_ids)
                if combined_ids
                else sum(int(result.get("games_updated", 0)) for result in successes)
            )
            combined["game_ids_updated"] = tuple(sorted(combined_ids))
            combined["fixture_count"] = max(
                int(result.get("fixture_count", 0)) for result in successes
            )
            combined["fallback_provider"] = successes[-1].get("provider")
        return {**combined, "attempts": attempts}

    failed = next(
        (attempt for attempt in attempts if attempt.get("status") == "failed"),
        None,
    )
    if failed:
        return {**failed, "attempts": attempts}
    if any(attempt.get("status") in {"no_markets", "no_matches"} for attempt in attempts):
        return {
            "status": "no_markets",
            "reason": "no provider returned matchable live NRL markets",
            "attempts": attempts,
        }
    return {
        "status": "skipped",
        "reason": "no live odds provider is configured",
        "attempts": attempts,
    }


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
            result = _snapshot_live_odds(db_path)
    except Exception as exc:
        _log(f"Odds {args.action} failed: {exc}")
        if args.strict:
            return 1
        _log("Fail-soft mode enabled. Continuing without odds update.")
        return 0

    if args.action != "backfill":
        try:
            from pipeline.ops.odds_gate import current_round_odds_coverage

            coverage = current_round_odds_coverage(db_path)
            if coverage.error:
                detail = f"unavailable ({coverage.error})"
            else:
                detail = f"{coverage.covered_games}/{coverage.total_games} current-round games priced"
            console.emit_result("freshness", source="odds", detail=detail)
        except Exception:
            pass

    if args.strict and result.get("status") in {"failed"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
