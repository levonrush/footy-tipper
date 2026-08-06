"""Joker round policy resolution, recommendation, and season usage state."""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

from pipeline.common.use_predictions.probabilities import two_way_home_probability


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
        "used_model_probs": 0,
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


def compute_joker_round_metrics(fixtures, model_probs=None):
    output_columns = [
        "round_id",
        "competition_year",
        "round_name",
        "matches_considered",
        "matches_total",
        "odds_coverage",
        "model_prob_matches",
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
    valid_odds = (data["odds_home"] > 1.0) & (data["odds_away"] > 1.0)
    q_home = (1.0 / data["odds_home"]).where(valid_odds)
    q_away = (1.0 / data["odds_away"]).where(valid_odds)
    overround = q_home + q_away
    p_home_market = (q_home / overround).where(overround > 0)
    p_tip_market = pd.concat([p_home_market, 1.0 - p_home_market], axis=1).max(axis=1)

    # Prefer the model's calibrated probability where a prediction exists
    # (typically the current round); market-implied fills the rest.
    p_tip_model = pd.Series(float("nan"), index=data.index, dtype="float64")
    if (
        model_probs is not None
        and not model_probs.empty
        and {"game_id", "p_home"}.issubset(model_probs.columns)
    ):
        mp = model_probs.copy()
        mp["game_id"] = pd.to_numeric(mp["game_id"], errors="coerce")
        mp["p_home"] = pd.to_numeric(mp["p_home"], errors="coerce").clip(1e-6, 1 - 1e-6)
        mp = mp.dropna().drop_duplicates(subset=["game_id"])
        merged = pd.to_numeric(data["game_id"], errors="coerce").map(
            mp.set_index("game_id")["p_home"]
        )
        p_tip_model = pd.concat([merged, 1.0 - merged], axis=1).max(axis=1)

    data["from_model"] = p_tip_model.notna()
    data["p_tip_correct"] = p_tip_model.fillna(p_tip_market)
    data = data[data["p_tip_correct"].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=output_columns)

    data["match_variance"] = data["p_tip_correct"] * (1.0 - data["p_tip_correct"])

    round_metrics = (
        data.groupby(["round_id", "competition_year", "round_name"], dropna=False, as_index=False)
        .agg(
            matches_considered=("game_id", "count"),
            model_prob_matches=("from_model", "sum"),
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
    round_metrics["model_prob_matches"] = pd.to_numeric(round_metrics["model_prob_matches"], errors="coerce").fillna(0).astype(int)
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
    model_probs=None,
):
    strategy_context = strategy_context or {}
    strategy = _resolve_joker_strategy_value(strategy or strategy_context.get("strategy") or _resolve_joker_strategy())
    risk_lambda = _coerce_env_float("FOOTY_TIPPER_JOKER_RISK_LAMBDA", 1.0, minimum=0.0)
    min_rounds_with_odds = _coerce_env_int("FOOTY_TIPPER_JOKER_MIN_ROUNDS_WITH_ODDS", 2, minimum=1)
    min_margin_ratio = _coerce_env_float("FOOTY_TIPPER_JOKER_MIN_MARGIN_RATIO", 0.05, minimum=0.0)
    min_round_coverage = _coerce_env_float("FOOTY_TIPPER_JOKER_MIN_ROUND_COVERAGE", 0.95, minimum=0.0)
    meta = _joker_objective_meta(strategy, risk_lambda)

    round_metrics = compute_joker_round_metrics(fixtures, model_probs=model_probs)
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
        "used_model_probs": int(current.get("model_prob_matches", 0) or 0),
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

    # Score the current round with the model's own calibrated probabilities
    # where predictions exist; future rounds remain market-implied.
    model_probs = None
    if (
        predictions is not None
        and not predictions.empty
        and {"game_id", "home_team_win_prob", "home_team_lose_prob"}.issubset(predictions.columns)
    ):
        p_home = two_way_home_probability(
            predictions["home_team_win_prob"], predictions["home_team_lose_prob"]
        )
        model_probs = pd.DataFrame(
            {
                "game_id": pd.to_numeric(predictions["game_id"], errors="coerce"),
                "p_home": p_home,
            }
        ).dropna()

    fixtures = get_joker_round_candidates(db_path, project_root)
    recommendation = recommend_joker_round(
        fixtures,
        current_round_id=current_round_id,
        current_round_name=current_round_name,
        strategy=strategy_context.get("strategy"),
        strategy_context=strategy_context,
        model_probs=model_probs,
    )
    competition_year = _infer_joker_competition_year(
        predictions=predictions,
        fixtures=fixtures,
        recommendation=recommendation,
    )
    recommendation["competition_year"] = competition_year
    usage_record = get_joker_usage_for_year(db_path, competition_year)
    return _apply_joker_usage_state(recommendation, usage_record)
