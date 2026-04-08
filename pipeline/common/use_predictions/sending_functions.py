import html
import json
import os
import pandas as pd
import re
import sqlite3
import urllib.request
from pathlib import Path

# for google
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    import gspread
except Exception:
    service_account = None
    build = None
    MediaFileUpload = None
    gspread = None

# for emails
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# For direct Anthropic API calls
try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

# For DALL-E image generation
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

# The 'get_predictions' function reads the predictions from the SQLite database and returns them as a pandas DataFrame.
def get_predictions(db_path, project_root):
    con = sqlite3.connect(str(db_path))
    with open(project_root / 'pipeline/common' / 'sql/create_table.sql', 'r') as file:
        create_table_query = file.read()
    con.execute(create_table_query)
    _ensure_predictions_table_columns(con)
    with open(project_root / 'pipeline/common' / 'sql/prediction_table.sql', 'r') as file:
        query = file.read()
    predictions = pd.read_sql_query(query, con)
    con.close()
    return _sort_predictions_for_display(predictions)


def _load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_joker_policy(project_root):
    project_root = Path(project_root)
    configured = os.getenv("FOOTY_TIPPER_JOKER_POLICY_PATH", "").strip()
    candidates = []

    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        candidates.append(configured_path)

    manifest_path = project_root / "models" / "model_manifest.json"
    manifest_payload = _load_json_file(manifest_path)
    if isinstance(manifest_payload, dict):
        manifest_policy = str(manifest_payload.get("joker_policy_file", "")).strip()
        if manifest_policy:
            candidates.append(project_root / "models" / manifest_policy)

    candidates.append(project_root / "models" / "joker_policy.json")

    for path in candidates:
        if path.exists() and path.is_file():
            payload = _load_json_file(path)
            if isinstance(payload, dict):
                payload["_policy_path"] = str(path)
                return payload
    return None


def _coerce_env_float(name, default, minimum=None):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _coerce_env_int(name, default, minimum=None):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(float(raw))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _resolve_joker_strategy():
    strategy_raw = os.getenv("FOOTY_TIPPER_JOKER_STRATEGY", "points").strip().lower()
    return _resolve_joker_strategy_value(strategy_raw)


def _resolve_joker_strategy_value(strategy_raw):
    strategy_raw = str(strategy_raw or "").strip().lower()
    aliases = {
        "points": "points",
        "expected": "points",
        "ev": "points",
        "protect": "protect",
        "lead": "protect",
        "conservative": "protect",
        "chase": "chase",
        "aggressive": "chase",
        "swing": "chase",
    }
    return aliases.get(strategy_raw, "points")


def _resolve_joker_strategy_context(project_root):
    requested = os.getenv("FOOTY_TIPPER_JOKER_STRATEGY", "auto").strip().lower()
    points_gap = _coerce_env_float("FOOTY_TIPPER_JOKER_POINTS_GAP", 0.0)

    if requested and requested != "auto":
        return {
            "strategy": _resolve_joker_strategy_value(requested),
            "source": "explicit_env",
            "requested": requested,
            "points_gap": points_gap,
            "scenario": "manual",
            "policy_used": False,
            "policy_path": None,
        }

    policy = _load_joker_policy(project_root)
    if not isinstance(policy, dict):
        return {
            "strategy": "points",
            "source": "fallback_default",
            "requested": requested or "auto",
            "points_gap": points_gap,
            "scenario": "neutral",
            "policy_used": False,
            "policy_path": None,
        }

    thresholds = policy.get("state_thresholds", {}) if isinstance(policy.get("state_thresholds"), dict) else {}
    lead_max_gap = pd.to_numeric(pd.Series([thresholds.get("lead_max_gap", -3.0)]), errors="coerce").fillna(-3.0).iloc[0]
    chase_min_gap = pd.to_numeric(pd.Series([thresholds.get("chase_min_gap", 3.0)]), errors="coerce").fillna(3.0).iloc[0]

    scenario = "neutral"
    if points_gap <= float(lead_max_gap):
        scenario = "lead"
    elif points_gap >= float(chase_min_gap):
        scenario = "chase"

    recommended = (
        policy.get("recommended_strategy_by_scenario", {})
        if isinstance(policy.get("recommended_strategy_by_scenario"), dict)
        else {}
    )
    preferred = recommended.get(scenario, policy.get("default_strategy", "points"))
    strategy = _resolve_joker_strategy_value(preferred)

    return {
        "strategy": strategy,
        "source": "policy_auto",
        "requested": requested or "auto",
        "points_gap": points_gap,
        "scenario": scenario,
        "policy_used": True,
        "policy_path": policy.get("_policy_path"),
        "lead_max_gap": float(lead_max_gap),
        "chase_min_gap": float(chase_min_gap),
    }


def _joker_objective_meta(strategy, risk_lambda):
    if strategy == "protect":
        return {
            "objective_column": "score_protect",
            "objective_label": f"mean-variance (mu - {risk_lambda:.2f}*sigma)",
            "strategy_label": "Protect the lead",
        }
    if strategy == "chase":
        return {
            "objective_column": "score_chase",
            "objective_label": "swing potential (variance)",
            "strategy_label": "Chase upside",
        }
    return {
        "objective_column": "score_points",
        "objective_label": "expected correct tips (mu)",
        "strategy_label": "Max expected points",
    }


def _coerce_competition_year(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def _round_label(round_id, round_name):
    if pd.notna(round_name) and str(round_name).strip():
        return str(round_name).strip()
    if pd.isna(round_id):
        return "Unknown round"
    return f"Round {int(round_id)}"


def _unavailable_joker_recommendation(reason, strategy_context=None):
    strategy_context = strategy_context or {}
    strategy = _resolve_joker_strategy_value(strategy_context.get("strategy", _resolve_joker_strategy()))
    risk_lambda = _coerce_env_float("FOOTY_TIPPER_JOKER_RISK_LAMBDA", 1.0, minimum=0.0)
    meta = _joker_objective_meta(strategy, risk_lambda)
    return {
        "available": False,
        "status": "unavailable",
        "headline": "Joker call unavailable",
        "detail": reason,
        "should_use_this_round": False,
        "strategy": strategy,
        "strategy_label": meta["strategy_label"],
        "objective_label": meta["objective_label"],
        "objective_column": meta["objective_column"],
        "strategy_source": strategy_context.get("source", "unknown"),
        "strategy_scenario": strategy_context.get("scenario", "unknown"),
        "points_gap": strategy_context.get("points_gap"),
        "policy_path": strategy_context.get("policy_path"),
        "risk_lambda": risk_lambda,
        "competition_year": None,
        "current_round_id": None,
        "current_round_name": None,
        "current_rank": None,
        "current_score": None,
        "current_mu": None,
        "current_sigma": None,
        "recommended_round_id": None,
        "recommended_round_name": None,
        "recommended_score": None,
        "recommended_mu": None,
        "recommended_sigma": None,
        "score_gap_to_best": None,
        "rounds_evaluated": 0,
        "ranked_rounds": pd.DataFrame(),
    }


def _ensure_joker_usage_table(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS joker_usage (
                competition_year INTEGER PRIMARY KEY,
                round_id INTEGER NOT NULL,
                round_name TEXT,
                played_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL DEFAULT 'unknown'
            )
            """
        )
        con.commit()
    finally:
        con.close()


def get_joker_usage_for_year(db_path, competition_year):
    year = _coerce_competition_year(competition_year)
    if year is None:
        return None

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            SELECT competition_year, round_id, round_name, played_at_utc, source
            FROM joker_usage
            WHERE competition_year = ?
            LIMIT 1
            """,
            (int(year),),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            row = None
        else:
            print(f"Joker usage lookup failed ({exc}).")
            row = None
    except Exception as exc:
        print(f"Joker usage lookup failed ({exc}).")
        row = None
    finally:
        con.close()

    if row is None:
        return None

    return {
        "competition_year": _coerce_competition_year(row["competition_year"]),
        "round_id": _coerce_competition_year(row["round_id"]),
        "round_name": row["round_name"],
        "played_at_utc": row["played_at_utc"],
        "source": row["source"],
    }


def _infer_joker_competition_year(predictions=None, fixtures=None, recommendation=None):
    if isinstance(recommendation, dict):
        year = _coerce_competition_year(recommendation.get("competition_year"))
        if year is not None:
            return year

    if predictions is not None and not predictions.empty and "competition_year" in predictions.columns:
        year = _coerce_competition_year(predictions.iloc[0].get("competition_year"))
        if year is not None:
            return year

    if fixtures is not None and not fixtures.empty and "competition_year" in fixtures.columns:
        numeric = pd.to_numeric(fixtures["competition_year"], errors="coerce").dropna()
        if not numeric.empty:
            return int(numeric.iloc[0])

    return None


def _apply_joker_usage_state(recommendation, usage_record):
    if not isinstance(recommendation, dict):
        return recommendation

    result = dict(recommendation)
    result.setdefault("joker_already_used", False)
    result.setdefault("joker_used_round_id", None)
    result.setdefault("joker_used_round_name", None)
    result.setdefault("joker_used_at_utc", None)
    result.setdefault("joker_used_source", None)

    if not isinstance(usage_record, dict):
        return result

    used_round_label = _round_label(usage_record.get("round_id"), usage_record.get("round_name"))
    used_round_id = _coerce_competition_year(usage_record.get("round_id"))
    current_round_id = _coerce_competition_year(result.get("current_round_id"))
    is_current_round_usage = (
        used_round_id is not None
        and current_round_id is not None
        and int(used_round_id) == int(current_round_id)
    )
    used_at = str(usage_record.get("played_at_utc", "") or "").strip()
    used_at_suffix = f" (recorded {used_at} UTC)" if used_at else ""
    detail = f"Joker already played in {used_round_label}{used_at_suffix}."
    if is_current_round_usage:
        detail = f"Joker already locked for this round in {used_round_label}{used_at_suffix}."

    result.update(
        {
            "available": False,
            "status": "already_used_current_round" if is_current_round_usage else "already_used",
            "headline": "PLAY JOKER THIS ROUND (ALREADY LOCKED)" if is_current_round_usage else "JOKER ALREADY USED THIS SEASON",
            "detail": detail,
            "should_use_this_round": bool(is_current_round_usage),
            "joker_already_used": True,
            "joker_usage_applies_to_current_round": bool(is_current_round_usage),
            "joker_used_round_id": usage_record.get("round_id"),
            "joker_used_round_name": used_round_label,
            "joker_used_at_utc": usage_record.get("played_at_utc"),
            "joker_used_source": usage_record.get("source"),
            "competition_year": _coerce_competition_year(
                usage_record.get("competition_year", result.get("competition_year"))
            ),
        }
    )
    return result


def persist_joker_usage_if_applicable(db_path, joker_recommendation, allow_write=False, source="send"):
    outcome = {
        "recorded": False,
        "reason": "no_recommendation",
        "competition_year": None,
        "round_id": None,
        "round_name": None,
    }
    if not isinstance(joker_recommendation, dict):
        return outcome

    competition_year = _coerce_competition_year(joker_recommendation.get("competition_year"))
    current_round_id = _coerce_competition_year(joker_recommendation.get("current_round_id"))
    recommended_round_id = _coerce_competition_year(joker_recommendation.get("recommended_round_id"))
    round_id = current_round_id if current_round_id is not None else recommended_round_id
    round_name = str(
        joker_recommendation.get("current_round_name")
        or joker_recommendation.get("recommended_round_name")
        or ""
    ).strip() or None

    outcome.update(
        {
            "reason": "pending",
            "competition_year": competition_year,
            "round_id": round_id,
            "round_name": round_name,
        }
    )

    if joker_recommendation.get("joker_already_used"):
        outcome["reason"] = "already_used"
        return outcome
    if not joker_recommendation.get("should_use_this_round", False):
        outcome["reason"] = "not_play_signal"
        return outcome
    if not allow_write:
        outcome["reason"] = "write_disabled"
        return outcome
    if competition_year is None or round_id is None:
        outcome["reason"] = "missing_round_context"
        return outcome

    _ensure_joker_usage_table(db_path)
    con = None
    try:
        con = sqlite3.connect(str(db_path))
        cursor = con.execute(
            """
            INSERT OR IGNORE INTO joker_usage (competition_year, round_id, round_name, source)
            VALUES (?, ?, ?, ?)
            """,
            (int(competition_year), int(round_id), round_name, str(source or "send")),
        )
        con.commit()
        inserted = cursor.rowcount > 0
    except Exception as exc:
        outcome["reason"] = "db_error"
        outcome["error"] = str(exc)
        return outcome
    finally:
        if con is not None:
            con.close()

    if inserted:
        outcome["recorded"] = True
        outcome["reason"] = "recorded"
    else:
        outcome["reason"] = "already_recorded"

    usage_record = get_joker_usage_for_year(db_path, competition_year)
    if isinstance(usage_record, dict):
        outcome["usage_record"] = usage_record
    return outcome


def get_joker_round_candidates(db_path, project_root):
    output_columns = [
        "game_id",
        "round_id",
        "competition_year",
        "round_name",
        "team_home",
        "team_away",
        "team_head_to_head_odds_home",
        "team_head_to_head_odds_away",
    ]
    con = sqlite3.connect(str(db_path))
    try:
        with open(project_root / "pipeline/common/sql/joker_round_candidates.sql", "r") as file:
            query = file.read()
        candidates = pd.read_sql_query(query, con)
    except Exception as exc:
        print(f"Joker round candidate query failed ({exc}).")
        candidates = pd.DataFrame(columns=output_columns)
    finally:
        con.close()

    if candidates.empty:
        return pd.DataFrame(columns=output_columns)
    return candidates


def compute_joker_round_metrics(fixtures):
    output_columns = [
        "round_id",
        "competition_year",
        "round_name",
        "matches_considered",
        "matches_total",
        "odds_coverage",
        "mu",
        "variance",
        "sigma",
        "mean_tip_probability",
        "perfect_round_probability",
        "max_matches_in_scope",
        "is_reduced_round",
        "score_points",
        "score_protect",
        "score_chase",
    ]
    if fixtures.empty:
        return pd.DataFrame(columns=output_columns)

    required_cols = {
        "game_id",
        "round_id",
        "competition_year",
        "round_name",
        "team_head_to_head_odds_home",
        "team_head_to_head_odds_away",
    }
    if not required_cols.issubset(set(fixtures.columns)):
        return pd.DataFrame(columns=output_columns)

    risk_lambda = _coerce_env_float("FOOTY_TIPPER_JOKER_RISK_LAMBDA", 1.0, minimum=0.0)
    min_round_coverage = _coerce_env_float("FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE", 0.95, minimum=0.0)

    data = fixtures.copy()
    round_totals = (
        data.groupby(["round_id", "competition_year", "round_name"], dropna=False, as_index=False)
        .agg(matches_total=("game_id", "count"))
    )

    data["odds_home"] = pd.to_numeric(data["team_head_to_head_odds_home"], errors="coerce")
    data["odds_away"] = pd.to_numeric(data["team_head_to_head_odds_away"], errors="coerce")
    data = data[(data["odds_home"] > 1.0) & (data["odds_away"] > 1.0)].copy()
    if data.empty:
        return pd.DataFrame(columns=output_columns)

    data["q_home"] = 1.0 / data["odds_home"]
    data["q_away"] = 1.0 / data["odds_away"]
    data["overround"] = data["q_home"] + data["q_away"]
    data = data[data["overround"] > 0].copy()
    if data.empty:
        return pd.DataFrame(columns=output_columns)

    data["p_home"] = data["q_home"] / data["overround"]
    data["p_away"] = data["q_away"] / data["overround"]
    data["p_tip_correct"] = data[["p_home", "p_away"]].max(axis=1)
    data["match_variance"] = data["p_tip_correct"] * (1.0 - data["p_tip_correct"])

    round_metrics = (
        data.groupby(["round_id", "competition_year", "round_name"], dropna=False, as_index=False)
        .agg(
            matches_considered=("game_id", "count"),
            mu=("p_tip_correct", "sum"),
            variance=("match_variance", "sum"),
            mean_tip_probability=("p_tip_correct", "mean"),
            perfect_round_probability=("p_tip_correct", "prod"),
        )
    )
    if round_metrics.empty:
        return pd.DataFrame(columns=output_columns)

    round_metrics = round_metrics.merge(
        round_totals,
        on=["round_id", "competition_year", "round_name"],
        how="left",
    )
    round_metrics["matches_total"] = pd.to_numeric(round_metrics["matches_total"], errors="coerce").fillna(0).astype(int)
    round_metrics["odds_coverage"] = round_metrics["matches_considered"] / round_metrics["matches_total"].replace(0, pd.NA)
    round_metrics["odds_coverage"] = pd.to_numeric(round_metrics["odds_coverage"], errors="coerce").fillna(0.0)
    round_metrics = round_metrics[round_metrics["odds_coverage"] >= min_round_coverage].copy()
    if round_metrics.empty:
        return pd.DataFrame(columns=output_columns)

    round_metrics["sigma"] = round_metrics["variance"].pow(0.5)
    round_metrics["score_points"] = round_metrics["mu"]
    round_metrics["score_protect"] = round_metrics["mu"] - (risk_lambda * round_metrics["sigma"])
    round_metrics["score_chase"] = round_metrics["variance"]

    max_matches = int(round_metrics["matches_considered"].max())
    round_metrics["max_matches_in_scope"] = max_matches
    round_metrics["is_reduced_round"] = round_metrics["matches_considered"] < max_matches

    round_metrics = round_metrics.sort_values("round_id").reset_index(drop=True)
    return round_metrics[output_columns]


def recommend_joker_round(
    fixtures,
    current_round_id=None,
    current_round_name=None,
    strategy=None,
    strategy_context=None,
):
    strategy_context = strategy_context or {}
    strategy = _resolve_joker_strategy_value(strategy or strategy_context.get("strategy") or _resolve_joker_strategy())
    risk_lambda = _coerce_env_float("FOOTY_TIPPER_JOKER_RISK_LAMBDA", 1.0, minimum=0.0)
    min_rounds_with_odds = _coerce_env_int("FOOTY_TIPPER_JOKER_MIN_ROUNDS_WITH_ODDS", 2, minimum=1)
    min_margin_ratio = _coerce_env_float("FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO", 0.05, minimum=0.0)
    min_round_coverage = _coerce_env_float("FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE", 0.95, minimum=0.0)
    meta = _joker_objective_meta(strategy, risk_lambda)

    round_metrics = compute_joker_round_metrics(fixtures)
    if round_metrics.empty:
        return _unavailable_joker_recommendation(
            "No upcoming rounds had valid head-to-head odds, so the joker model could not score rounds.",
            strategy_context=strategy_context,
        )

    objective_column = meta["objective_column"]
    ranked = (
        round_metrics.sort_values(
            [objective_column, "mu", "matches_considered"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
        .copy()
    )
    ranked["rank"] = ranked.index + 1

    if current_round_id is None:
        current_round_id = pd.to_numeric(ranked["round_id"], errors="coerce").min()
    current_row = ranked[ranked["round_id"] == current_round_id]

    if current_row.empty and current_round_name is not None:
        lookup = str(current_round_name).strip().lower()
        current_row = ranked[
            ranked["round_name"].astype(str).str.strip().str.lower() == lookup
        ]

    if current_row.empty:
        current_row = ranked.iloc[[0]]

    recommended = ranked.iloc[0]
    current = current_row.iloc[0]

    should_use_pre_gate = int(current["round_id"]) == int(recommended["round_id"])
    should_use = should_use_pre_gate
    score_gap_to_best = float(recommended[objective_column]) - float(current[objective_column])
    margin_to_next = None
    margin_to_next_ratio = None

    if len(ranked) > 1:
        second_best_score = float(ranked.iloc[1][objective_column])
        margin_to_next = float(recommended[objective_column]) - second_best_score
        denom = max(abs(float(recommended[objective_column])), 1e-9)
        margin_to_next_ratio = margin_to_next / denom

    gate_reasons = []
    if should_use_pre_gate and len(ranked) < min_rounds_with_odds:
        gate_reasons.append(
            f"only {len(ranked)} round(s) meet odds coverage >= {min_round_coverage:.0%} (min {min_rounds_with_odds})"
        )
    if should_use_pre_gate and margin_to_next_ratio is not None and margin_to_next_ratio < min_margin_ratio:
        gate_reasons.append(
            f"lead over next-best round is {margin_to_next_ratio:.1%} (min {min_margin_ratio:.1%})"
        )
    if gate_reasons:
        should_use = False

    recommended_round_label = _round_label(recommended["round_id"], recommended["round_name"])
    current_round_label = _round_label(current["round_id"], current["round_name"])

    if should_use:
        detail = (
            f"{current_round_label} is ranked #1/{len(ranked)} on {meta['objective_label']} "
            f"(mu {float(current['mu']):.2f}, sigma {float(current['sigma']):.2f})."
        )
        headline = "PLAY JOKER THIS ROUND"
    elif should_use_pre_gate and gate_reasons:
        detail = (
            f"{current_round_label} rates #1 on {meta['objective_label']}, but hold for now: "
            f"{'; '.join(gate_reasons)}."
        )
        headline = "HOLD JOKER THIS ROUND"
    else:
        detail = (
            f"{current_round_label} is ranked #{int(current['rank'])}/{len(ranked)} on "
            f"{meta['objective_label']} (score {float(current[objective_column]):.2f}); "
            f"best is {recommended_round_label} (score {float(recommended[objective_column]):.2f})."
        )
        headline = "HOLD JOKER THIS ROUND"

    return {
        "available": True,
        "status": "ok",
        "headline": headline,
        "detail": detail,
        "should_use_this_round": bool(should_use),
        "strategy": strategy,
        "strategy_label": meta["strategy_label"],
        "strategy_source": strategy_context.get("source", "env"),
        "strategy_scenario": strategy_context.get("scenario"),
        "points_gap": strategy_context.get("points_gap"),
        "policy_path": strategy_context.get("policy_path"),
        "objective_label": meta["objective_label"],
        "objective_column": objective_column,
        "risk_lambda": risk_lambda,
        "competition_year": _coerce_competition_year(current.get("competition_year")),
        "current_round_id": int(current["round_id"]),
        "current_round_name": current_round_label,
        "current_rank": int(current["rank"]),
        "current_score": float(current[objective_column]),
        "current_mu": float(current["mu"]),
        "current_sigma": float(current["sigma"]),
        "recommended_round_id": int(recommended["round_id"]),
        "recommended_round_name": recommended_round_label,
        "recommended_score": float(recommended[objective_column]),
        "recommended_mu": float(recommended["mu"]),
        "recommended_sigma": float(recommended["sigma"]),
        "score_gap_to_best": score_gap_to_best,
        "margin_to_next": margin_to_next,
        "margin_to_next_ratio": margin_to_next_ratio,
        "min_rounds_with_odds": int(min_rounds_with_odds),
        "min_margin_ratio": float(min_margin_ratio),
        "min_round_coverage": float(min_round_coverage),
        "rounds_evaluated": int(len(ranked)),
        "ranked_rounds": ranked,
    }


def get_joker_round_recommendation(db_path, project_root, predictions=None):
    strategy_context = _resolve_joker_strategy_context(project_root)
    current_round_id = None
    current_round_name = None
    if predictions is not None and not predictions.empty:
        current_round_id_val = pd.to_numeric(
            pd.Series([predictions.iloc[0].get("round_id")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(current_round_id_val):
            current_round_id = int(current_round_id_val)

        round_name_val = predictions.iloc[0].get("round_name")
        if pd.notna(round_name_val):
            current_round_name = str(round_name_val)

    fixtures = get_joker_round_candidates(db_path, project_root)
    recommendation = recommend_joker_round(
        fixtures,
        current_round_id=current_round_id,
        current_round_name=current_round_name,
        strategy=strategy_context.get("strategy"),
        strategy_context=strategy_context,
    )
    competition_year = _infer_joker_competition_year(
        predictions=predictions,
        fixtures=fixtures,
        recommendation=recommendation,
    )
    recommendation["competition_year"] = competition_year
    usage_record = get_joker_usage_for_year(db_path, competition_year)
    return _apply_joker_usage_state(recommendation, usage_record)

# The 'get_tipper_picks' function calculates the odds thresholds and returns a DataFrame of tipper picks.
def get_tipper_picks(predictions, prod_run=False):
    output_columns = [
        "game_id",
        "team",
        "opponent",
        "side",
        "price",
        "price_min",
        "model_prob",
        "edge",
        "kelly_full",
        "kelly_fraction",
        "kelly_capped_fraction",
        "stake_fraction",
        "stake_amount",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=output_columns)

    min_edge_default = 0.03 if prod_run else 0.02
    min_edge = float(os.getenv("FOOTY_TIPPER_MIN_VALUE_EDGE", str(min_edge_default)))
    kelly_multiplier = float(os.getenv("FOOTY_TIPPER_KELLY_FRACTION", "0.5"))
    max_stake_fraction = float(os.getenv("FOOTY_TIPPER_MAX_STAKE_FRACTION", "0.05"))
    min_stake_fraction = float(os.getenv("FOOTY_TIPPER_MIN_STAKE_FRACTION", "0.0"))
    stake_mode = os.getenv("FOOTY_TIPPER_STAKE_MODE", "normalized").strip().lower()
    if stake_mode not in {"normalized", "bankroll"}:
        stake_mode = "normalized"
    bankroll_env = os.getenv("FOOTY_TIPPER_BANKROLL", "")

    bankroll = None
    if bankroll_env.strip():
        try:
            bankroll_value = float(bankroll_env)
            if bankroll_value > 0:
                bankroll = bankroll_value
        except ValueError:
            bankroll = None

    predictions = predictions.copy()

    # Use expected value (p * odds - 1) for the model's predicted winner only.
    # Only tips the model expects to win are eligible as value picks.
    records = []
    for _, row in predictions.iterrows():
        game_id = row.get("game_id")
        home_team = row.get("team_home")
        away_team = row.get("team_away")
        home_prob = pd.to_numeric(pd.Series([row.get("home_team_win_prob")]), errors="coerce").iloc[0]
        away_prob = pd.to_numeric(pd.Series([row.get("home_team_lose_prob")]), errors="coerce").iloc[0]
        home_odds = pd.to_numeric(pd.Series([row.get("team_head_to_head_odds_home")]), errors="coerce").iloc[0]
        away_odds = pd.to_numeric(pd.Series([row.get("team_head_to_head_odds_away")]), errors="coerce").iloc[0]

        side_candidates = []
        predicted_result = row.get("home_team_result")
        for side, team, opp, prob, odds in [
            ("home", home_team, away_team, home_prob, home_odds),
            ("away", away_team, home_team, away_prob, away_odds),
        ]:
            # Only evaluate sides the model tips to win
            if side == "home" and predicted_result != "Win":
                continue
            if side == "away" and predicted_result != "Loss":
                continue

            if pd.isna(prob) or pd.isna(odds) or odds <= 1 or prob <= 0 or prob >= 1:
                continue

            fair_odds = 1 / prob
            edge = (prob * odds) - 1.0
            denominator = odds - 1.0
            kelly_full = edge / denominator if denominator > 0 else 0.0
            kelly_full = max(0.0, kelly_full)
            kelly_fractional = max(0.0, kelly_full * kelly_multiplier)
            kelly_capped = min(max_stake_fraction, kelly_fractional)
            if kelly_capped < min_stake_fraction:
                kelly_capped = 0.0

            side_candidates.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "opponent": opp,
                    "side": side,
                    "price": odds,
                    "price_min": fair_odds,
                    "model_prob": prob,
                    "edge": edge,
                    "kelly_full": kelly_full,
                    "kelly_fraction": kelly_fractional,
                    "kelly_capped_fraction": kelly_capped,
                }
            )

        if not side_candidates:
            continue

        best = max(side_candidates, key=lambda x: x["edge"])
        if best["edge"] >= min_edge and best["kelly_capped_fraction"] > 0:
            records.append(best)

    if not records:
        return pd.DataFrame(columns=output_columns)

    tipper_picks = pd.DataFrame.from_records(records)
    if stake_mode == "normalized":
        total_weight = float(tipper_picks["kelly_capped_fraction"].sum())
        if total_weight > 0:
            tipper_picks["stake_fraction"] = tipper_picks["kelly_capped_fraction"] / total_weight
        else:
            tipper_picks["stake_fraction"] = 0.0
    else:
        tipper_picks["stake_fraction"] = tipper_picks["kelly_capped_fraction"]

    if bankroll is not None:
        tipper_picks["stake_amount"] = tipper_picks["stake_fraction"] * bankroll
    else:
        tipper_picks["stake_amount"] = pd.NA

    tipper_picks = tipper_picks.sort_values(["stake_fraction", "edge"], ascending=False).reset_index(drop=True)
    return tipper_picks[output_columns]

# The 'upload_df_to_drive' function uploads a pandas DataFrame as a CSV file to Google Drive.
def upload_df_to_drive(df, json_path, parent_folder_id, filename):
    if service_account is None or build is None or MediaFileUpload is None:
        print("Upload skipped: Google Drive dependencies are not installed.")
        return
    if df.empty:
        print("Upload skipped: no predictions to upload.")
        return
    if not parent_folder_id:
        print("Upload skipped: FOLDER_ID is not configured.")
        return
    if not os.path.exists(json_path):
        print(f"Upload skipped: missing Google service account token at {json_path}.")
        return

    creds = service_account.Credentials.from_service_account_file(json_path)
    drive_service = build('drive', 'v3', credentials=creds)
    competition_year = str(df['competition_year'].unique()[0])
    
    def get_or_create_folder(service, folder_name, parent_folder_id):
        query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        if not items:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            return folder.get('id')
        else:
            return items[0]['id']
    
    competition_year_folder_id = get_or_create_folder(drive_service, competition_year, parent_folder_id)
    df.to_csv(filename, index=False)
    file_name = f"round{df['round_id'].unique()[0]}_{df['competition_year'].unique()[0]}.csv"
    
    def get_existing_file_id(service, folder_id, file_name):
        query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        if items:
            return items[0]['id']
        return None
    
    existing_file_id = get_existing_file_id(drive_service, competition_year_folder_id, file_name)
    if existing_file_id:
        drive_service.files().delete(fileId=existing_file_id).execute()
    
    file_metadata = {
        'name': file_name,
        'parents': [competition_year_folder_id]
    }
    media = MediaFileUpload(filename, mimetype='text/csv')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print('File ID:', file.get('id'))
    os.remove(filename)

def _default_subject(predictions):
    if predictions.empty:
        return "Footy Tipper Predictions Update"
    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    return f"Footy Tipper Predictions for {round_name} {competition_year}"


def _format_probability(value):
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.0%}"


def _format_price(value):
    if pd.isna(value):
        return "n/a"
    return f"${float(value):.2f}"


def _format_percent(value):
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def _format_number(value, decimals=2):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):.{decimals}f}"


def _ensure_predictions_table_columns(con):
    expected_columns = {
        "draw_prob": "REAL",
        "bayes_factor": "REAL",
        "evidence_strength": "TEXT",
        "predicted_home_score": "INTEGER",
        "predicted_away_score": "INTEGER",
        "predicted_margin": "INTEGER",
    }
    existing_columns = {row[1] for row in con.execute("PRAGMA table_info(predictions_table)").fetchall()}
    for column_name, column_ddl in expected_columns.items():
        if column_name not in existing_columns:
            con.execute(f"ALTER TABLE predictions_table ADD COLUMN {column_name} {column_ddl}")


def _coerce_int(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(round(float(numeric)))


def _sort_predictions_for_display(predictions):
    if predictions.empty:
        return predictions.copy()

    sort_columns = [column for column in ("start_time", "game_number", "game_id") if column in predictions.columns]
    if not sort_columns:
        return predictions.reset_index(drop=True)

    ordered = predictions.copy()
    helper_columns = []
    for column in sort_columns:
        helper_column = f"__sort_{column}"
        helper_columns.append(helper_column)
        ordered[helper_column] = pd.to_numeric(ordered[column], errors="coerce")

    ordered = ordered.sort_values(helper_columns, kind="stable", na_position="last")
    return ordered.drop(columns=helper_columns, errors="ignore").reset_index(drop=True)


def _prediction_winner(row):
    return row["team_home"] if row.get("home_team_result") == "Win" else row["team_away"]


def _format_predicted_score_numbers(row):
    home_score = _coerce_int(row.get("predicted_home_score"))
    away_score = _coerce_int(row.get("predicted_away_score"))
    if home_score is None or away_score is None:
        return "n/a"
    return f"{home_score}-{away_score}"


def _format_predicted_scoreline(row):
    score_numbers = _format_predicted_score_numbers(row)
    if score_numbers == "n/a":
        return "Score tip unavailable"
    return f"{row['team_home']} {score_numbers} {row['team_away']}"


def _format_predicted_margin(row):
    margin = _coerce_int(row.get("predicted_margin"))
    if margin is None:
        return "n/a"
    if margin == 0:
        return "Draw"
    return f"{_prediction_winner(row)} by {abs(margin)}"


def _first_game_callout(predictions):
    if predictions.empty:
        return None

    first_game = predictions.iloc[0]
    is_home_tip = first_game.get("home_team_result") == "Win"
    tip_probability = (
        first_game.get("home_team_win_prob")
        if is_home_tip
        else first_game.get("home_team_lose_prob")
    )
    return {
        "fixture": f"{first_game['team_home']} vs {first_game['team_away']}",
        "tip": _prediction_winner(first_game),
        "tip_probability": _format_probability(tip_probability),
        "scoreline": _format_predicted_scoreline(first_game),
        "margin": _format_predicted_margin(first_game),
    }


def _joker_summary_lines(joker_recommendation):
    if not isinstance(joker_recommendation, dict):
        return ["Joker call: unavailable (no recommendation data provided)."]

    headline = str(joker_recommendation.get("headline", "Joker call unavailable")).strip()
    detail = str(joker_recommendation.get("detail", "")).strip()
    strategy_label = str(joker_recommendation.get("strategy_label", "")).strip()
    objective_label = str(joker_recommendation.get("objective_label", "")).strip()
    joker_already_used = bool(joker_recommendation.get("joker_already_used", False))
    used_round_label = _round_label(
        joker_recommendation.get("joker_used_round_id"),
        joker_recommendation.get("joker_used_round_name"),
    )
    used_at = str(joker_recommendation.get("joker_used_at_utc", "") or "").strip()

    lines = [f"Joker call: {headline}"]
    if joker_already_used:
        usage_line = f"Season status: already played in {used_round_label}."
        if used_at:
            usage_line = f"Season status: already played in {used_round_label} (recorded {used_at} UTC)."
        lines.append(usage_line)

    if strategy_label:
        if objective_label:
            lines.append(f"Strategy: {strategy_label} using {objective_label}.")
        else:
            lines.append(f"Strategy: {strategy_label}.")
    strategy_source = str(joker_recommendation.get("strategy_source", "")).strip()
    strategy_scenario = str(joker_recommendation.get("strategy_scenario", "")).strip()
    if strategy_source == "policy_auto":
        scenario_suffix = f", scenario {strategy_scenario}" if strategy_scenario else ""
        lines.append(f"Strategy source: learned training policy{scenario_suffix}.")
    elif strategy_source == "explicit_env":
        lines.append("Strategy source: explicit environment setting.")
    if detail:
        lines.append(detail)

    if joker_recommendation.get("available"):
        lines.append(
            "Current round metrics: "
            f"mu {_format_number(joker_recommendation.get('current_mu'))}, "
            f"sigma {_format_number(joker_recommendation.get('current_sigma'))}."
        )
        if not joker_recommendation.get("should_use_this_round", False):
            lines.append(
                "Recommended hold target: "
                f"{joker_recommendation.get('recommended_round_name', 'Unknown round')} "
                f"(mu {_format_number(joker_recommendation.get('recommended_mu'))}, "
                f"sigma {_format_number(joker_recommendation.get('recommended_sigma'))})."
            )

    return lines


def _joker_prompt_block(joker_recommendation):
    return "\n".join(f"- {line}" for line in _joker_summary_lines(joker_recommendation))


def _resolve_banner_path():
    project_root = Path(__file__).resolve().parents[3]
    configured = os.getenv("FOOTY_TIPPER_EMAIL_BANNER")

    candidates = []
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = project_root / configured_path
        candidates.append(configured_path)
    candidates.append(project_root / "images" / "email-banner.png")

    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)
    return None


def _build_fallback_copy(predictions, folder_url, joker_recommendation=None):
    if predictions.empty:
        return {
            "subject": "Footy Tipper Update",
            "opening": "No pre-game NRL fixtures were found for the current run, so there are no tips to send this week.",
            "closing": "Bring back the biff.",
        }

    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    special_event_context = _special_event_context(round_name, competition_year)
    opening = (
        f"Welcome to {round_name} {competition_year}. {special_event_context['opening_line']}\n"
        "The model has done the hard yakka and lined up this week's tips.\n"
        "If these picks torch your tipping comp, remember this was all done with science and zero accountability."
    )
    if isinstance(joker_recommendation, dict):
        opening += (
            "\nJoker watch: "
            f"{joker_recommendation.get('headline', 'Joker call unavailable')}."
        )
    if folder_url:
        opening += f"\nFull details are in the tips folder: {folder_url}"
    closing = (
        "If you're in tipping comps at the Seven Seas Hotel in Carrington or the work comp at Hunter Water, "
        "you did not get this from us.\nBring back the biff."
    )
    return {
        "subject": _default_subject(predictions),
        "opening": opening,
        "closing": closing,
    }


def _special_event_context(round_name, competition_year):
    normalized = str(round_name).strip().lower()
    if "grand final" in normalized:
        return {
            "event_name": "Grand Final",
            "opening_line": "It's the grand final decider, so every tip is legacy-defining and stress-inducing.",
            "prompt_angle": "Treat this as the premiership decider. Big stakes, one-shot narrative, no generic weekly intro.",
        }
    if "preliminary final" in normalized:
        return {
            "event_name": "Preliminary Final",
            "opening_line": "It's preliminary final weekend, where reputations get made and seasons get buried.",
            "prompt_angle": "Frame as knockout footy on the edge of the grand final.",
        }
    if "qualifying final" in normalized or "elimination final" in normalized or "semi final" in normalized:
        return {
            "event_name": "Finals",
            "opening_line": "Finals footy is here, so the margin for error is basically non-existent.",
            "prompt_angle": "Write as finals football: pressure, knockout stakes, and tactical edges.",
        }
    if re.search(r"\bround\s*1\b", normalized):
        return {
            "event_name": "Round 1",
            "opening_line": f"It's Round 1, the season opener for {competition_year}, so optimism is irrationally high.",
            "prompt_angle": "Treat as season opener energy: fresh starts, overreactions, and new-year storylines.",
        }
    return {
        "event_name": "Regular Round",
        "opening_line": "Another week, another chance to make objectively questionable tipping decisions.",
        "prompt_angle": "Treat as a regular season round with concise but lively banter.",
    }


def _build_prompt_input(predictions, tipper_picks, joker_recommendation=None):
    fixture_lines = []
    for _, row in predictions.iterrows():
        winner = _prediction_winner(row)
        fixture_lines.append(
            f"- {row['team_home']} vs {row['team_away']}: tip {winner} "
            f"(home win {_format_probability(row['home_team_win_prob'])}, "
            f"away win {_format_probability(row['home_team_lose_prob'])}, "
            f"score tip {_format_predicted_score_numbers(row)}, "
            f"margin {_format_predicted_margin(row)}, "
            f"market {row['team_home']} {_format_price(row['team_head_to_head_odds_home'])}, "
            f"{row['team_away']} {_format_price(row['team_head_to_head_odds_away'])})"
        )

    pick_lines = []
    if tipper_picks.empty:
        pick_lines.append("- None flagged by the model.")
    else:
        for _, row in tipper_picks.iterrows():
            pick_lines.append(
                f"- {row['team']} vs {row['opponent']}: market {_format_price(row['price'])}, "
                f"fair {_format_price(row['price_min'])}, edge {_format_percent(row['edge'])}, "
                f"stake share {_format_percent(row['stake_fraction'])}"
            )

    return "\n".join(fixture_lines), "\n".join(pick_lines), _joker_prompt_block(joker_recommendation)


def _sanitize_json_newlines(text):
    """Replace literal newlines inside JSON string values with escaped \\n."""
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\":
            result.append(ch)
            escape_next = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            pass  # strip CR
        else:
            result.append(ch)
    return "".join(result)


def _parse_json_object(text):
    if not text:
        return None
    # Strip markdown code fences if present
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        stripped = stripped[start:end + 1]

    sanitized = _sanitize_json_newlines(stripped)
    candidates = [stripped, sanitized]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _generate_claude_copy(predictions, tipper_picks, api_key, folder_url, temperature, joker_recommendation=None, news_context=None):
    if predictions.empty:
        return None
    if not api_key:
        print("ANTHROPIC_API_KEY is not configured. Using fallback email content.")
        return None
    if Anthropic is None:
        print("Anthropic SDK is unavailable. Using fallback email content.")
        return None

    fixtures_text, picks_text, joker_text = _build_prompt_input(
        predictions,
        tipper_picks,
        joker_recommendation=joker_recommendation,
    )
    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    special_event_context = _special_event_context(round_name, competition_year)
    folder_line = folder_url if folder_url else "No public folder URL is configured this run."
    prompt = f"""
Write Reg Reagan's weekly NRL tipping email. Reg is loud, passionate, and deeply invested — he doesn't hedge, he doesn't whisper, and he definitely doesn't forgive bad footy. Write like he's been awake since 5am thinking about this round.

Round: {round_name} {competition_year}
Special event context: {special_event_context['event_name']}
Special event writing angle: {special_event_context['prompt_angle']}
Tips folder: {folder_line}

Fixtures and model picks:
{fixtures_text}

Value picks:
{picks_text}

Joker recommendation:
{joker_text}

Current NRL news this week (use if something is funny or worth a dig — otherwise ignore):
{news_context if news_context else "Nothing notable found this week."}

Return JSON only with this exact schema:
{{
  "subject": "short email subject line, max 75 chars",
  "news_hit": "1 punchy paragraph where Reg calls out the biggest scandal or story from the news this week — opinionated, direct, sets the tone before the tips. If news is provided above, you MUST write this. Only use null if no news was provided.",
  "opening": "2-3 paragraphs — Reg's take on the round with some personality and genuine opinions on the key games",
  "closing": "1-2 short paragraphs. Must end with: Bring back the biff."
}}

Rules:
- If news is provided in "Current NRL news", you MUST write news_hit — do not bury it in the opening and do not set it to null.
- Mention the Newcastle Knights positively.
- Take a dig at Manly.
- Include this disclaimer naturally: if people are in tipping comps at Seven Seas Hotel in Carrington or the Hunter Water work comp, they should not use these tips.
- Include one explicit sentence that starts with "Joker call:" and states PLAY or HOLD for this round.
- Keep it punchy and readable — a touch of colour, not a wall of slang.
- Output raw JSON only. No markdown fences, no preamble, no text before {{ or after }}.
- Do not include markdown, HTML, or extra keys.
"""

    client = Anthropic(api_key=api_key)
    configured_model = os.getenv("CLAUDE_MODEL")
    model_candidates = (
        [configured_model]
        if configured_model
        else ["claude-sonnet-4-6"]
    )
    last_exception = None

    for model_name in model_candidates:
        if not model_name:
            continue
        try:
            response = client.messages.create(
                model=model_name,
                system="You are Reg Reagan — an opinionated Australian NRL tragic who writes weekly tipping emails. You're enthusiastic and direct, use occasional Australian slang, and have genuine strong opinions on footy. You're entertaining but not over the top — think passionate pub regular, not raving lunatic.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=temperature,
            )
            raw_text = response.content[0].text or ""
            payload = _parse_json_object(raw_text)
            if not payload:
                print(f"Claude email generation returned non-JSON payload for model '{model_name}'.")
                continue
            subject = str(payload.get("subject", "")).strip()
            opening = str(payload.get("opening", "")).strip()
            closing = str(payload.get("closing", "")).strip()
            if not subject or not opening or not closing:
                print(f"Claude email generation returned incomplete JSON keys for model '{model_name}'.")
                continue
            news_hit_raw = payload.get("news_hit")
            news_hit = str(news_hit_raw).strip() if news_hit_raw and str(news_hit_raw).lower() != "null" else None
            print(f"Claude email generation model: {model_name}")
            return {
                "subject": subject,
                "news_hit": news_hit,
                "opening": opening,
                "closing": closing,
            }
        except Exception as exc:
            last_exception = exc
            print(f"Claude email generation failed for model '{model_name}' ({exc}).")
            if configured_model:
                break

    if last_exception is not None:
        print(f"Claude email generation failed ({last_exception}). Using fallback email content.")
    return None


def _to_html_paragraphs(text):
    blocks = []
    for paragraph in [p.strip() for p in text.split("\n\n") if p.strip()]:
        safe = html.escape(paragraph).replace("\n", "<br>")
        blocks.append(
            "<p style=\"margin:0 0 16px; color:#111827; font-family:'Trebuchet MS', Arial, sans-serif; "
            "font-size:17px; line-height:1.65;\">"
            f"{safe}"
            "</p>"
        )
    return "".join(blocks)


def _render_plain_email(predictions, tipper_picks, folder_url, subject, opening, closing, joker_recommendation=None, news_hit=None):
    first_game = _first_game_callout(predictions)
    lines = [subject, ""]
    if news_hit:
        lines.extend(["--- THIS WEEK IN LEAGUE ---", news_hit, "---------------------------", ""])
    lines.append(opening)
    if first_game is not None:
        lines.extend(
            [
                "",
                "First game spotlight:",
                f"- {first_game['fixture']}",
                f"- Tip: {first_game['tip']} ({first_game['tip_probability']})",
                f"- Score tip: {first_game['scoreline']}",
                f"- Margin: {first_game['margin']}",
            ]
        )

    lines.extend(["", "Predicted winners:"])
    for _, row in predictions.iterrows():
        winner = _prediction_winner(row)
        lines.append(
            f"- {row['team_home']} vs {row['team_away']}: {winner} "
            f"(home {_format_probability(row['home_team_win_prob'])}, "
            f"away {_format_probability(row['home_team_lose_prob'])}, "
            f"score {_format_predicted_score_numbers(row)}, "
            f"margin {_format_predicted_margin(row)})"
        )

    lines.append("")
    if tipper_picks.empty:
        lines.append("Value picks: none flagged this round.")
    else:
        lines.append("Value picks:")
        for _, row in tipper_picks.iterrows():
            stake_suffix = ""
            if not pd.isna(row.get("stake_amount", pd.NA)):
                stake_suffix = f", stake {_format_price(row['stake_amount'])}"
            lines.append(
                f"- {row['team']} vs {row['opponent']} at {_format_price(row['price'])} "
                f"(fair {_format_price(row['price_min'])}, edge {_format_percent(row['edge'])}, "
                f"stake share {_format_percent(row['stake_fraction'])}{stake_suffix})"
            )

    if folder_url:
        lines.extend(["", f"Tips folder: {folder_url}"])

    lines.extend(["", "Joker round call:"])
    for line in _joker_summary_lines(joker_recommendation):
        lines.append(f"- {line}")

    lines.extend(["", closing])
    return "\n".join(lines)


def _render_html_email(
    predictions,
    tipper_picks,
    folder_url,
    opening,
    closing,
    banner_available,
    joker_recommendation=None,
    news_hit=None,
):
    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    first_game = _first_game_callout(predictions)

    match_rows = []
    for i, (_, row) in enumerate(predictions.iterrows()):
        winner = _prediction_winner(row)
        row_bg = "#f9fafb" if i % 2 == 0 else "#ffffff"
        home_prob = row['home_team_win_prob']
        if home_prob >= 0.65:
            badge_bg, badge_color = "#dcfce7", "#15803d"
        elif home_prob >= 0.45:
            badge_bg, badge_color = "#fef9c3", "#854d0e"
        else:
            badge_bg, badge_color = "#fee2e2", "#b91c1c"
        match_rows.append(
            f"<tr style=\"background:{row_bg};\">"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px; width:36%;\">"
            f"{html.escape(str(row['team_home']))} vs {html.escape(str(row['team_away']))}"
            "</td>"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; color:#0f766e; "
            "font-family:Arial, sans-serif; font-size:15px; font-weight:700; width:32%;\">"
            f"<div>{html.escape(str(winner))}</div>"
            "</td>"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; width:16%;\">"
            f"<span style=\"display:inline-block; padding:3px 7px; border-radius:12px; "
            f"background:{badge_bg}; color:{badge_color}; font-family:Arial, sans-serif; font-size:12px; font-weight:700;\">"
            f"H {_format_probability(row['home_team_win_prob'])}</span>"
            f"<span style=\"display:block; margin-top:3px; color:#6b7280; font-family:Arial, sans-serif; font-size:12px;\">"
            f"A {_format_probability(row['home_team_lose_prob'])}</span>"
            "</td>"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; color:#374151; "
            "font-family:Arial, sans-serif; font-size:13px; width:16%;\">"
            f"H {_format_price(row['team_head_to_head_odds_home'])}<br>A {_format_price(row['team_head_to_head_odds_away'])}"
            "</td>"
            "</tr>"
        )

    pick_rows = []
    for _, row in tipper_picks.iterrows():
        stake_text = _format_percent(row["stake_fraction"])
        if not pd.isna(row.get("stake_amount", pd.NA)):
            stake_text = f"{stake_text} ({_format_price(row['stake_amount'])})"
        pick_rows.append(
            "<tr>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{html.escape(str(row['team']))} vs {html.escape(str(row['opponent']))}"
            "</td>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{_format_price(row['price'])}"
            "</td>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{_format_price(row['price_min'])}"
            "</td>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{_format_percent(row['edge'])}"
            "</td>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{stake_text}"
            "</td>"
            "</tr>"
        )

    value_section = ""
    if tipper_picks.empty:
        value_section = (
            "<p style=\"margin:0; color:#4b5563; font-family:Arial, sans-serif; font-size:14px; line-height:1.5;\">"
            "No value picks were flagged this round."
            "</p>"
        )
    else:
        value_section = (
            "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" "
            "style=\"border-collapse:collapse; border:1px solid #bbf7d0; border-radius:8px; overflow:hidden;\">"
            "<thead>"
            "<tr style=\"background:#dcfce7;\">"
            "<th align=\"left\" style=\"padding:10px; color:#15803d; font-family:Arial, sans-serif; font-size:12px;\">Team</th>"
            "<th align=\"left\" style=\"padding:10px; color:#15803d; font-family:Arial, sans-serif; font-size:12px;\">Market</th>"
            "<th align=\"left\" style=\"padding:10px; color:#15803d; font-family:Arial, sans-serif; font-size:12px;\">Fair</th>"
            "<th align=\"left\" style=\"padding:10px; color:#15803d; font-family:Arial, sans-serif; font-size:12px;\">Edge</th>"
            "<th align=\"left\" style=\"padding:10px; color:#15803d; font-family:Arial, sans-serif; font-size:12px;\">Stake Share</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            f"{''.join(pick_rows)}"
            "</tbody>"
            "</table>"
        )

    first_game_section = ""
    if first_game is not None:
        first_game_section = (
            "<tr><td style=\"padding:10px 24px 6px;\">"
            "<div style=\"padding:16px 18px; border-radius:12px; background:#ecfeff; border:1px solid #67e8f9;\">"
            "<p style=\"margin:0 0 8px; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; "
            "font-size:16px; font-weight:700;\">First game spotlight</p>"
            "<p style=\"margin:0 0 6px; color:#0f172a; font-family:Arial, sans-serif; font-size:14px;\">"
            f"{html.escape(first_game['fixture'])}"
            "</p>"
            "<p style=\"margin:0 0 4px; color:#0f172a; font-family:Arial, sans-serif; font-size:14px;\">"
            f"Tip: {html.escape(first_game['tip'])} ({html.escape(first_game['tip_probability'])})"
            "</p>"
            "<p style=\"margin:0 0 4px; color:#0f172a; font-family:Arial, sans-serif; font-size:14px;\">"
            f"Score tip: {html.escape(first_game['scoreline'])}"
            "</p>"
            "<p style=\"margin:0; color:#0f172a; font-family:Arial, sans-serif; font-size:14px;\">"
            f"Margin: {html.escape(first_game['margin'])}"
            "</p>"
            "</div>"
            "</td></tr>"
        )

    joker_lines = _joker_summary_lines(joker_recommendation)
    joker_list_html = "".join(
        [
            "<li style=\"margin:0 0 6px; color:#1f2937; font-family:Arial, sans-serif; font-size:14px; line-height:1.5;\">"
            f"{html.escape(line)}"
            "</li>"
            for line in joker_lines
        ]
    )
    joker_headline = html.escape(str(joker_recommendation.get("headline", "Joker call unavailable"))) if isinstance(joker_recommendation, dict) else "Joker call unavailable"
    joker_bg = "#fff7ed"
    joker_border = "#f59e0b"
    if isinstance(joker_recommendation, dict) and joker_recommendation.get("joker_already_used"):
        joker_bg = "#f3f4f6"
        joker_border = "#6b7280"
    elif isinstance(joker_recommendation, dict) and joker_recommendation.get("should_use_this_round"):
        joker_bg = "#ecfdf5"
        joker_border = "#10b981"
    joker_section = (
        "<div style=\"padding:14px; border-radius:10px; "
        f"background:{joker_bg}; border:1px solid {joker_border};\">"
        "<p style=\"margin:0 0 10px; color:#111827; font-family:'Trebuchet MS', Arial, sans-serif; font-size:16px; font-weight:700;\">"
        f"{joker_headline}"
        "</p>"
        "<ul style=\"margin:0; padding-left:18px;\">"
        f"{joker_list_html}"
        "</ul>"
        "</div>"
    )

    banner_html = ""
    if banner_available:
        banner_html = (
            "<img src=\"cid:footy_tipper_email_banner\" alt=\"Footy Tipper\" "
            "style=\"display:block; width:100%; max-width:680px; height:auto; border:0; border-radius:12px 12px 0 0;\">"
        )
    else:
        banner_html = (
            "<div style=\"padding:26px 24px; background:linear-gradient(135deg, #115e59 0%, #0369a1 100%); border-radius:12px 12px 0 0;\">"
            "<h1 style=\"margin:0; color:#ffffff; font-family:'Trebuchet MS', Arial, sans-serif; font-size:30px; letter-spacing:0.5px;\">"
            "Footy Tipper"
            "</h1>"
            "</div>"
        )

    folder_button = ""
    if folder_url:
        safe_url = html.escape(folder_url, quote=True)
        folder_button = (
            "<tr><td style=\"padding:8px 24px 24px;\">"
            f"<a href=\"{safe_url}\" "
            "style=\"display:inline-block; background:#0f766e; color:#ffffff; text-decoration:none; "
            "font-family:Arial, sans-serif; font-size:14px; font-weight:700; padding:12px 18px; border-radius:8px;\">"
            "Open Tips Folder"
            "</a>"
            "</td></tr>"
        )

    return (
        "<html><body style=\"margin:0; padding:20px; background:#eef2f7;\">"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" style=\"border-collapse:collapse;\">"
        "<tr><td align=\"center\">"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"680\" "
        "style=\"max-width:680px; width:100%; border-collapse:collapse; background:#ffffff; border-radius:12px;\">"
        f"<tr><td>{banner_html}</td></tr>"
        "<tr><td style=\"padding:24px 24px 10px;\">"
        "<h2 style=\"margin:0; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:26px;\">"
        f"{html.escape(str(round_name))} {html.escape(str(competition_year))} Tips"
        "</h2>"
        "</td></tr>"
        + (
            "<tr><td style=\"padding:6px 24px 10px;\">"
            "<div style=\"border-radius:8px; overflow:hidden; border:1px solid #fca5a5;\">"
            "<div style=\"background:#dc2626; padding:8px 14px;\">"
            "<p style=\"margin:0; color:#ffffff; font-family:'Trebuchet MS', Arial, sans-serif; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase;\">This Week In League</p>"
            "</div>"
            "<div style=\"padding:14px 16px; background:#fff7f7;\">"
            f"<p style=\"margin:0; color:#1f2937; font-family:'Trebuchet MS', Arial, sans-serif; font-size:15px; line-height:1.65;\">{html.escape(news_hit)}</p>"
            "</div>"
            "</div>"
            "</td></tr>"
            if news_hit else ""
        ) +
        "<tr><td style=\"padding:6px 24px 6px;\">"
        f"{_to_html_paragraphs(opening)}"
        "</td></tr>"
        f"{first_game_section}"
        "<tr><td style=\"padding:10px 24px 8px;\">"
        "<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid #0f766e; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:18px;\">Predicted winners</h3>"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" "
        "style=\"border-collapse:collapse; border:1px solid #e5e7eb; border-radius:8px; overflow:hidden;\">"
        "<thead><tr style=\"background:#f9fafb;\">"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Fixture</th>"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Tip</th>"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Win Prob</th>"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">H2H Odds</th>"
        "</tr></thead>"
        "<tbody>"
        f"{''.join(match_rows)}"
        "</tbody></table>"
        "</td></tr>"
        "<tr><td style=\"padding:14px 24px 8px;\">"
        "<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid #16a34a; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:18px;\">Value picks</h3>"
        f"{value_section}"
        "</td></tr>"
        "<tr><td style=\"padding:14px 24px 8px;\">"
        "<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid #f59e0b; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:18px;\">Joker round call</h3>"
        f"{joker_section}"
        "</td></tr>"
        f"{folder_button}"
        "<tr><td style=\"padding:6px 24px 22px;\">"
        f"{_to_html_paragraphs(closing)}"
        "</td></tr>"
        "<tr><td style=\"padding:16px 24px 24px; border-top:1px solid #e5e7eb;\">"
        "<p style=\"margin:0 0 4px; color:#0f766e; font-family:'Trebuchet MS', Arial, sans-serif; font-size:13px; font-weight:700;\">Bring back the biff.</p>"
        "<p style=\"margin:0; color:#9ca3af; font-family:Arial, sans-serif; font-size:11px;\">Generated by Footy Tipper.</p>"
        "</td></tr>"
        "</table>"
        "</td></tr>"
        "</table>"
        "</body></html>"
    )


_NRL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=NRL+rugby+league&hl=en-AU&gl=AU&ceid=AU:en",
    "https://news.google.com/rss/search?q=NRL+rugby+league+scandal+drama&hl=en-AU&gl=AU&ceid=AU:en",
]


def _fetch_rss_headlines(max_items=20):
    """Fetch recent NRL headlines from Google News RSS. Returns plain text list or empty string."""
    import xml.etree.ElementTree as ET
    headlines = []
    for url in _NRL_NEWS_FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
            for item in root.findall(".//item")[:max_items]:
                title = (item.findtext("title") or "").strip()
                desc = (item.findtext("description") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                if title:
                    headlines.append(f"- {title} ({pub}): {desc[:120]}")
        except Exception:
            continue
    return "\n".join(headlines[:max_items])


def _fetch_nrl_news_context(anthropic_client):
    """Fetch NRL headlines then ask Claude to pick the top story. Always returns something."""
    try:
        headlines = _fetch_rss_headlines()
        if not headlines:
            print("NRL news: RSS fetch returned nothing.")
            return None

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            system=(
                "You are a news editor. Given a list of NRL rugby league headlines, "
                "pick the single most interesting, scandalous, or dramatic story from the past 7 days and summarise it in 2-3 sentences. "
                "Be specific — name the player, club, or incident. "
                "It could be anything: a scandal, a big signing, a code switch, a surprise result, a feud, a sacking, a comeback — whatever people in NRL circles are talking about most this week. "
                "Return only the summary. No preamble."
            ),
            messages=[{"role": "user", "content": f"Headlines:\n{headlines}"}],
            max_tokens=300,
        )
        text = response.content[0].text.strip() if response.content else None
        if text:
            print(f"NRL news: {text[:100]}...")
            return text
        return None
    except Exception as exc:
        print(f"NRL news fetch failed ({exc}). Skipping.")
        return None


def _build_banner_edit_instruction(copy, anthropic_client, news_context=None, news_hit=None):
    """Ask Claude for a fun, topical scenario for the two banner characters this week."""
    subject = copy.get("subject", "")
    opening = copy.get("opening", "")[:300]
    # news_hit is the primary source — it's already the most interesting story distilled
    if news_hit:
        inspiration = f"This week's big story (PRIMARY inspiration for the banner):\n{news_hit}"
    elif news_context:
        inspiration = f"NRL news this week:\n{news_context}"
    else:
        inspiration = f"Email subject: {subject}\nEmail opening: {opening}"
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        system="You write short, vivid image editing instructions for a fun weekly sports email banner.",
        messages=[{"role": "user", "content": (
            f"A weekly NRL tipping email banner features two cartoon characters: Reg Reagan (a bloke in a shirt that says 'Bring Back the Biff') and a dingo. "
            f"Come up with a funny or energetic scenario for this week's banner inspired by the content below. "
            f"Put Reg and the dingo in a situation that directly references the story or themes — they can be doing anything: celebrating, arguing, cowering, riding something, holding a sign, dressed up, etc. "
            f"Be creative and specific.\n\n"
            f"{inspiration}\n\n"
            "Return 2-3 sentences describing the scene. Be visual and specific. No preamble."
        )}],
        max_tokens=150,
        temperature=1.0,
    )
    topical = response.content[0].text.strip()
    return (
        f"Reimagine this image as a wide landscape email banner (roughly 3:1 aspect ratio — broad and horizontal). "
        f"The banner must include the 'Reg's Footy Tips' logo/title text prominently, matching the style of the original. "
        f"The two characters are Reg Reagan (a bloke whose shirt reads 'Bring Back the Biff') and a dingo — keep both present. "
        f"Maintain the same overall visual style, colour palette, and brand aesthetic as the original. "
        f"Scene: {topical} "
        f"Composition: logo/title on one side, characters and scene filling the rest of the banner. "
        f"Fun, punchy sports editorial illustration style."
    )


def _generate_dynamic_banner(copy, anthropic_api_key, openai_api_key, news_context=None, news_hit=None):
    """Edit the existing email banner with topical elements via Claude + gpt-image-1."""
    if not anthropic_api_key or not openai_api_key:
        return None
    if Anthropic is None or OpenAIClient is None:
        print("Dynamic banner skipped: Anthropic or OpenAI SDK unavailable.")
        return None
    try:
        import base64
        import io
        from PIL import Image
    except ImportError:
        print("Dynamic banner skipped: Pillow not installed.")
        return None

    try:
        project_root = Path(__file__).resolve().parents[3]
        banner_path = project_root / "images" / "email-banner.png"
        if not banner_path.exists():
            print("Dynamic banner skipped: base banner not found.")
            return None

        anthropic_client = Anthropic(api_key=anthropic_api_key)
        edit_instruction = _build_banner_edit_instruction(copy, anthropic_client, news_context=news_context, news_hit=news_hit)
        print(f"Banner edit: {edit_instruction[:120]}...")

        img = Image.open(banner_path).convert("RGBA")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        openai_client = OpenAIClient(api_key=openai_api_key)
        response = openai_client.images.edit(
            model="gpt-image-1",
            image=("email-banner.png", img_bytes, "image/png"),
            prompt=edit_instruction,
        )
        image_data = base64.b64decode(response.data[0].b64_json)
        out_path = project_root / "images" / "email-banner-generated.png"
        out_path.write_bytes(image_data)
        print(f"Dynamic banner saved: {out_path.name}")
        return str(out_path)
    except Exception as exc:
        import traceback
        print(f"Dynamic banner generation failed: {exc}")
        traceback.print_exc()
        return None


def generate_reg_regan_email_payload(
    predictions,
    tipper_picks,
    api_key,
    folder_url,
    temperature,
    use_openai=True,
    joker_recommendation=None,
    openai_api_key=None,
):
    predictions = _sort_predictions_for_display(predictions)
    fallback_copy = _build_fallback_copy(
        predictions,
        folder_url,
        joker_recommendation=joker_recommendation,
    )
    news_context = None
    if use_openai and api_key and Anthropic is not None:
        news_context = _fetch_nrl_news_context(Anthropic(api_key=api_key))

    if not use_openai:
        print("Claude generation disabled. Using fallback email content.")
    openai_copy = (
        _generate_claude_copy(
            predictions,
            tipper_picks,
            api_key,
            folder_url,
            temperature,
            joker_recommendation=joker_recommendation,
            news_context=news_context,
        )
        if use_openai
        else None
    )
    copy = openai_copy or fallback_copy

    if predictions.empty:
        plain = copy["opening"]
        html_email = (
            "<html><body style=\"font-family:Arial,sans-serif; background:#eef2f7; padding:20px;\">"
            "<div style=\"max-width:680px; margin:0 auto; background:#fff; border-radius:12px; padding:24px;\">"
            f"<p style=\"margin:0; color:#111827; font-size:16px; line-height:1.5;\">{html.escape(copy['opening'])}</p>"
            "</div></body></html>"
        )
        return {
            "subject": copy["subject"],
            "plain_text": plain,
            "html_text": html_email,
            "inline_images": [],
        }

    news_hit = copy.get("news_hit")
    banner_path = (
        _generate_dynamic_banner(copy, api_key, openai_api_key, news_context=news_context, news_hit=news_hit)
        or _resolve_banner_path()
    )
    plain_email = _render_plain_email(
        predictions,
        tipper_picks,
        folder_url,
        copy["subject"],
        copy["opening"],
        copy["closing"],
        joker_recommendation=joker_recommendation,
        news_hit=news_hit,
    )
    html_email = _render_html_email(
        predictions,
        tipper_picks,
        folder_url,
        copy["opening"],
        copy["closing"],
        banner_available=bool(banner_path),
        joker_recommendation=joker_recommendation,
        news_hit=news_hit,
    )

    inline_images = []
    if banner_path:
        inline_images.append({"cid": "footy_tipper_email_banner", "path": banner_path})

    return {
        "subject": copy["subject"],
        "plain_text": plain_email,
        "html_text": html_email,
        "inline_images": inline_images,
    }


# Backward-compatible wrapper: returns plain text body only.
def generate_reg_regan_email(
    predictions,
    tipper_picks,
    api_key,
    folder_url,
    temperature,
    joker_recommendation=None,
):
    payload = generate_reg_regan_email_payload(
        predictions,
        tipper_picks,
        api_key,
        folder_url,
        temperature,
        use_openai=bool(api_key),
        joker_recommendation=joker_recommendation,
    )
    return payload["plain_text"]


def _attach_inline_images(msg, inline_images):
    if not inline_images:
        return
    for image in inline_images:
        cid = image.get("cid") if isinstance(image, dict) else None
        path = image.get("path") if isinstance(image, dict) else None
        if not cid or not path:
            continue
        if not os.path.exists(path):
            print(f"Inline image skipped: file not found at {path}.")
            continue
        try:
            with open(path, "rb") as img_file:
                img = MIMEImage(img_file.read())
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
            msg.attach(img)
        except Exception as exc:
            print(f"Inline image skipped ({path}): {exc}")


def _build_mime_message(subject, sender_email, recipients, plain_message, html_message=None, inline_images=None):
    has_html = bool(html_message)
    msg = MIMEMultipart("related") if has_html else MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    if has_html:
        alternatives = MIMEMultipart("alternative")
        alternatives.attach(MIMEText(plain_message, "plain", "utf-8"))
        alternatives.attach(MIMEText(html_message, "html", "utf-8"))
        msg.attach(alternatives)
        _attach_inline_images(msg, inline_images)
    else:
        msg.attach(MIMEText(plain_message, "plain", "utf-8"))

    return msg


# The 'send_emails' function sends an email with the generated content.
def send_emails(doc_name, subject, message, sender_email, sender_password, json_path, html_message=None, inline_images=None):
    if service_account is None or gspread is None:
        print("Email send skipped: Google Sheets dependencies are not installed.")
        return False
    if not sender_email or not sender_password:
        print("Email send skipped: missing MY_EMAIL or EMAIL_PASSWORD.")
        return False
    if not os.path.exists(json_path):
        print(f"Email send skipped: missing Google service account token at {json_path}.")
        return False

    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_file(json_path, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open(doc_name).sheet1
    email_data = sheet.get_all_records()
    recipient_emails = [row['Email'] for row in email_data if row.get('Email')]
    if not recipient_emails:
        print("Email send skipped: no recipients found in the email list.")
        return False
    
    msg = _build_mime_message(
        subject=subject,
        sender_email=sender_email,
        recipients=recipient_emails,
        plain_message=message,
        html_message=html_message,
        inline_images=inline_images,
    )
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    text = msg.as_string()
    server.sendmail(sender_email, recipient_emails, text)
    server.quit()
    return True


def send_test_email(subject, message, sender_email, sender_password, recipient_email, html_message=None, inline_images=None):
    if not sender_email or not sender_password:
        print("Test email skipped: missing MY_EMAIL or EMAIL_PASSWORD.")
        return False
    if not recipient_email:
        print("Test email skipped: missing recipient email.")
        return False

    msg = _build_mime_message(
        subject=subject,
        sender_email=sender_email,
        recipients=[recipient_email],
        plain_message=message,
        html_message=html_message,
        inline_images=inline_images,
    )

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    text = msg.as_string()
    server.sendmail(sender_email, [recipient_email], text)
    server.quit()
    return True



# import os
# import pandas as pd
# import sqlite3

# # for google
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaFileUpload
# import gspread
# from google.oauth2 import service_account

# # for reg
# # from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI

# # for emails
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText

# # The 'get_predictions' function reads the predictions from the SQLite database and returns them as a pandas DataFrame.
# def get_predictions(db_path, project_root):

#     # Connect to the SQLite database
#     con = sqlite3.connect(str(db_path))

#     # Read SQL query from external SQL file
#     with open(project_root / 'pipeline/common' / 'sql/prediction_table.sql', 'r') as file:
#         query = file.read()

#     # Execute the query and fetch the results into a data frame
#     predictions = pd.read_sql_query(query, con)

#     # Disconnect from the SQLite database
#     con.close()

#     return predictions

# # The 'get_tipper_picks' function calculates the odds threshold for both home and away teams and then selects the home and away teams based on their predicted results.
# def get_tipper_picks(predictions, prod_run=False):
    
#     # Calculate odds thresholds for home and away teams
#     predictions['home_odds_thresh'] = 1 / predictions['home_team_win_prob']
#     predictions['away_odds_thresh'] = 1 / predictions['home_team_lose_prob'] 
    
#     # Select home teams that are predicted to win and rename the columns accordingly.
#     home_picks = predictions[predictions['home_team_result'] == 'Win'][['team_home', 'team_head_to_head_odds_home', 'home_odds_thresh']].copy()
#     home_picks.rename(columns={'team_home': 'team', 'team_head_to_head_odds_home': 'price', 'home_odds_thresh': 'price_min'}, inplace=True)
    
#     # Select away teams that are predicted to lose and rename the columns accordingly.
#     away_picks = predictions[predictions['home_team_result'] == 'Loss'][['team_away', 'team_head_to_head_odds_away', 'away_odds_thresh']].copy()
#     away_picks.rename(columns={'team_away': 'team', 'team_head_to_head_odds_away': 'price', 'away_odds_thresh': 'price_min'}, inplace=True)
    
#     # Concatenate the home and away picks and filter rows where 'price' is more than 15% of 'price_min'.
#     tipper_picks = pd.concat([home_picks, away_picks])
#     tipper_picks = tipper_picks[tipper_picks['price'] > (tipper_picks['price_min'] * 1.05)]

#     return tipper_picks

# # The 'upload_df_to_drive' function uploads a pandas DataFrame to Google Drive as a CSV file.
# def upload_df_to_drive(df, json_path, parent_folder_id, filename):

#     # Load the credentials from the service_account.json
#     creds = service_account.Credentials.from_service_account_file(json_path)

#     # Build the Google Drive service
#     drive_service = build('drive', 'v3', credentials=creds)

#     # Extract competition year
#     competition_year = str(df['competition_year'].unique()[0])

#     # Check if the folder for the competition year exists, if not, create it
#     def get_or_create_folder(service, folder_name, parent_folder_id):
#         query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
#         results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
#         items = results.get('files', [])
        
#         if not items:
#             # Folder does not exist, create it
#             file_metadata = {
#                 'name': folder_name,
#                 'mimeType': 'application/vnd.google-apps.folder',
#                 'parents': [parent_folder_id]
#             }
#             folder = service.files().create(body=file_metadata, fields='id').execute()
#             return folder.get('id')
#         else:
#             # Folder exists, return the id
#             return items[0]['id']
    
#     competition_year_folder_id = get_or_create_folder(drive_service, competition_year, parent_folder_id)

#     # Save your dataframe to CSV
#     df.to_csv(filename, index=False)

#     # Prepare file metadata
#     file_name = f"round{df['round_id'].unique()[0]}_{df['competition_year'].unique()[0]}.csv"

#     # Check if a file with the same name exists in the target folder
#     def get_existing_file_id(service, folder_id, file_name):
#         query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
#         results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
#         items = results.get('files', [])
        
#         if items:
#             return items[0]['id']
#         return None
    
#     existing_file_id = get_existing_file_id(drive_service, competition_year_folder_id, file_name)

#     # If the file exists, delete it
#     if existing_file_id:
#         drive_service.files().delete(fileId=existing_file_id).execute()

#     # Upload the file
#     file_metadata = {
#         'name': file_name,
#         'parents': [competition_year_folder_id]
#     }
#     media = MediaFileUpload(filename, mimetype='text/csv')
#     file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

#     print('File ID:', file.get('id'))

#     # Delete the local file after upload
#     os.remove(filename)

# # The 'generate_reg_regan_email' function generates an email content with the help of an AI language model (OpenAI). The email contains a synopsis of NRL games and some value tips.
# def generate_reg_regan_email(predictions, tipper_picks, api_key, folder_url, temperature):

#     # Set up the OpenAI model using provided API key and model parameters
#     llm = ChatOpenAI(openai_api_key=api_key,
#                      model_name="gpt-4",
#                      max_tokens=7000,
#                      temperature=temperature)

#     # Generate input_predictions string by iterating over 'predictions' dataframe and formatting data into string
#     input_predictions = ""
#     for index, row in predictions.iterrows():
#         input_predictions += f"""
#             Round Name: {row['round_name']},
#             Home Team Result: {row['home_team_result']},
#             Home Team: {row['team_home']}, 
#             Home Team Position: {row['position_home']},
#             Home Team Head to Head Price: {row['team_head_to_head_odds_home']}
#             Away Team: {row['team_away']},
#             Away Team Position: {row['position_away']},
#             Away Team Head to Head Price: {row['team_head_to_head_odds_away']}
#             """
    
#     # Generate input_picks string by iterating over 'tipper_picks' dataframe and formatting data into string
#     input_picks = ""
#     for index, row in tipper_picks.iterrows():
#         input_picks += f"""
#             Team: {row['team']},
#             Price: {row['price']}
#             """

#     # Generate the prompt string to be used with the AI model
#     prompt = f"""
#         I have a set of predictions for NRL games in {predictions['round_name'].unique()[0]} {predictions['competition_year'].unique()[0]} made by a machine learning pipeline called the Footy Tipper: \n{input_predictions}\n 
#         The description of the columns of interest is:
        
#         * Home Team Result: the predicted result of the home team
#         * Home Team: the home team
#         * Home Team Position: the home team's position on the NRL ladder
#         * Home Team Head to Head Price: the price bookies are offering for a home win
#         * Away Team: the away team
#         * Away Team Position: the away team's position on the NRL ladder
#         * Away Team Head to Head Price: the price bookies are offering for an away win
        
#         It also comes up with some good value tips for those interested in a punt in \n{input_picks}\n. If it is empty there isn't much value for punting in the round. The description of the columns of interest is:
        
#         * Team = Team that is a good value pick
#         * Price = what the bookies are offering them at
        
#         Could you write up an email to my mates from Reg Reagan, giving them a synopsis of the round along with the tips? 
#         Accompany the tips with some smart arsed comments about the teams playing.
#         Remember to link everyone to the tips folder: {folder_url}
#         Also, tell everyone to bring back the biff at the end of the email.
#         Also also your favorite team is the Newcastle Knights and you hate Manly.
#         Also also also, tell them that if they are in tipping comps at either the Seven Seas Hotel in Carrington or the Ship Inn on Hunter St then they aren't allowed to use the tips.
#         """

#     # Use the AI model to generate the email content based on the prompt
#     reg_regan = llm.predict(prompt)

#     return reg_regan

# # The 'send_emails' function sends an email to a list of recipients. The email details are prepared and the SMTP server is used to send the emails.
# def send_emails(doc_name, subject, message, sender_email, sender_password, json_path):

#     # 
#     scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
#              "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

#     # Authorize Google client using service account credentials to access Google Sheets
#     creds = service_account.Credentials.from_service_account_file(json_path, scopes=scope)
#     client = gspread.authorize(creds)

#     # Open the spreadsheet and get the data
#     sheet = client.open(doc_name).sheet1 # this is the spreadsheet with the emails
#     email_data = sheet.get_all_records()  # gets all the data inside your Google Sheet

#     # Extract the recipient emails from the Google Sheet data
#     recipient_emails = [row['Email'] for row in email_data]  # replace 'Email' with your actual column name

#     # Prepare the email message using MIMEText
#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = ', '.join(recipient_emails)
#     msg['Subject'] = subject
#     msg.attach(MIMEText(message, 'plain'))

#     # Setup the SMTP server for sending the email
#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     server.starttls()

#     # Login to the SMTP server using sender's email and password
#     server.login(sender_email, sender_password)

#     # Send the email to the list of recipients
#     text = msg.as_string()
#     server.sendmail(sender_email, recipient_emails, text)

#     # Close the SMTP server connection
#     server.quit()
