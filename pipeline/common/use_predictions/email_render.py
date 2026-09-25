"""Plain-text and HTML rendering of the weekly tipping email."""

import html
import math
import os

import pandas as pd

from pipeline.common.odds.validity import valid_decimal_odds
from pipeline.common.club_context.product import (
    SHADOW_DISCLAIMER,
    category_label as _context_category_label,
    normalize_context_cards,
)
from pipeline.common.use_predictions.finals import theme as _stage_theme
from pipeline.common.use_predictions.joker import _round_label
from pipeline.common.use_predictions.probabilities import (  # re-exported for site/email_copy
    tip_probability,
    two_way_home_probability,
)

# Every finals extra arrives in one `finals` payload rather than as a dozen new
# arguments. When it is absent or says this is a regular round, the renderers
# take exactly the paths they took before finals support existed.


def _finals_on(finals):
    return bool(isinstance(finals, dict) and finals.get("is_finals"))


def _theme(finals):
    if _finals_on(finals) and isinstance(finals.get("theme"), dict):
        return finals["theme"]
    return _stage_theme(None)


def _finals_field(finals, key, default=None):
    if not _finals_on(finals):
        return default
    value = finals.get(key)
    return default if value is None else value


def _stakes_for(finals, row):
    return (_finals_field(finals, "stakes", {}) or {}).get(_game_key(row))


def _head_to_head_for(finals, row):
    return (_finals_field(finals, "head_to_head", {}) or {}).get(_game_key(row))


def _distribution_for(finals, row):
    return (_finals_field(finals, "distributions", {}) or {}).get(_game_key(row))


def _game_key(row):
    value = row.get("game_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _margin_band_text(distribution):
    """One line on how close the model expects the game to be."""
    if not isinstance(distribution, dict):
        return None
    one_score = distribution.get("p_one_score_game")
    blowout = None
    home_big = distribution.get("p_home_by_13_plus")
    away_big = distribution.get("p_away_by_13_plus")
    if home_big is not None and away_big is not None:
        blowout = float(home_big) + float(away_big)
    parts = []
    if one_score is not None:
        parts.append(f"{_format_probability(float(one_score))} it is a one-score game")
    if blowout is not None:
        parts.append(f"{_format_probability(blowout)} it is 13 or more")
    total = distribution.get("median_total")
    if total is not None:
        parts.append(f"median total {int(round(float(total)))}")
    return "; ".join(parts) if parts else None


def _default_subject(predictions, finals=None):
    if predictions.empty:
        return "Footy Tipper Predictions Update"
    if _finals_on(finals):
        from pipeline.common.use_predictions.finals import finals_subject

        branded = finals_subject(
            finals.get("stage"), predictions['competition_year'].iloc[0]
        )
        if branded:
            return branded
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


def _format_market_price(value, fresh=True):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    try:
        is_fresh = False if pd.isna(fresh) else bool(fresh)
    except (TypeError, ValueError):
        is_fresh = False
    if not is_fresh or not valid_decimal_odds(numeric):
        return "n/a"
    return f"${float(numeric):.2f}"


def _format_percent(value):
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def _format_number(value, decimals=2):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):.{decimals}f}"



def _coerce_int(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(round(float(numeric)))


def _prediction_winner(row):
    return row["team_home"] if row.get("home_team_result") == "Win" else row["team_away"]


def _market_coverage_notice(predictions):
    """Return reader-facing copy when one or more fixtures lack real H2H prices."""
    if predictions.empty:
        return None
    required = {
        "team_head_to_head_odds_home",
        "team_head_to_head_odds_away",
    }
    if not required.issubset(predictions.columns):
        missing = len(predictions)
    else:
        home = pd.to_numeric(
            predictions["team_head_to_head_odds_home"], errors="coerce"
        )
        away = pd.to_numeric(
            predictions["team_head_to_head_odds_away"], errors="coerce"
        )
        finite_home = home.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
        finite_away = away.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
        valid = finite_home & finite_away & (home > 1.0) & (away > 1.0)
        if "market_odds_fresh" in predictions.columns:
            valid &= predictions["market_odds_fresh"].fillna(False).astype(bool)
        missing = int((~valid).sum())
    if missing == 0:
        return None
    fixture_word = "fixture" if missing == 1 else "fixtures"
    return (
        f"Market odds are unavailable for {missing} {fixture_word}. "
        "Those tips are model-only; no market edge or staking claim is being made."
    )


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
    home_score = _coerce_int(row.get("predicted_home_score"))
    away_score = _coerce_int(row.get("predicted_away_score"))
    if home_score is not None and away_score is not None:
        margin = home_score - away_score
        if margin == 0:
            return "Draw"
        winner = row["team_home"] if margin > 0 else row["team_away"]
        return f"{winner} by {abs(margin)}"

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
    return {
        "fixture": f"{first_game['team_home']} vs {first_game['team_away']}",
        "tip": _prediction_winner(first_game),
        "tip_probability": _format_probability(tip_probability(first_game)),
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


def _joker_reader_lines(joker_recommendation):
    """Short, reader-facing joker summary: the call plus one sentence.

    The full diagnostics (_joker_summary_lines) stay in CLI logs and the
    plain-text email; readers just need PLAY/HOLD and the reason.
    """
    if not isinstance(joker_recommendation, dict):
        return ["No joker call available this round."]

    lines = []
    detail = str(joker_recommendation.get("detail", "")).strip()
    if detail:
        lines.append(detail)

    if joker_recommendation.get("joker_already_used"):
        used_round_label = _round_label(
            joker_recommendation.get("joker_used_round_id"),
            joker_recommendation.get("joker_used_round_name"),
        )
        lines.append(f"This season's joker has already been played in {used_round_label}.")
    elif joker_recommendation.get("available") and not joker_recommendation.get("should_use_this_round", False):
        target = str(joker_recommendation.get("recommended_round_name", "")).strip()
        if target:
            lines.append(f"Best round left for it looks like {target}.")

    return lines or ["No joker call available this round."]


def _joker_footnote(joker_recommendation):
    """One small grey line of diagnostics for the curious."""
    if not isinstance(joker_recommendation, dict) or not joker_recommendation.get("available"):
        return None
    parts = []
    strategy_label = str(joker_recommendation.get("strategy_label", "")).strip()
    if strategy_label:
        parts.append(f"Strategy: {strategy_label}")
    mu = _format_number(joker_recommendation.get("current_mu"))
    sigma = _format_number(joker_recommendation.get("current_sigma"))
    if mu != "n/a":
        parts.append(f"this round mu {mu}, sigma {sigma}")
    return " · ".join(parts) if parts else None


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


def _scoreboard_text_line(scoreboard):
    if not isinstance(scoreboard, dict):
        return None
    line = (
        f"The ledger — {scoreboard['last_round_name']}: "
        f"{scoreboard['last_round_correct']}/{scoreboard['last_round_games']} tips. "
        f"Season: {scoreboard['season_correct']}/{scoreboard['season_games']} "
        f"({scoreboard['season_accuracy']:.0%})"
    )
    if scoreboard.get("market_accuracy") is not None:
        line += f" vs market favourite {scoreboard['market_accuracy']:.0%}"
    return line + "."


def _why_text(row) -> str:
    """The stored one-line reason for a tip, or "" when there is not one.

    Absent whenever inference ran before explanations existed, or the
    explanation write failed. Both render exactly as the email did before.
    """
    value = row.get("why_line") if hasattr(row, "get") else None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _market_picks_lines(finals):
    picks = _finals_field(finals, "market_picks")
    if picks is None or getattr(picks, "empty", True):
        return []
    lines = ["", "Line and totals:"]
    for _, pick in picks.iterrows():
        stake_suffix = ""
        if not pd.isna(pick.get("stake_amount", pd.NA)):
            stake_suffix = f", stake {_format_price(pick['stake_amount'])}"
        lines.append(
            f"- {pick['market']}: {pick['selection']} at {_format_price(pick['price'])} "
            f"(fair {_format_price(pick['price_min'])}, edge {_format_percent(pick['edge'])}, "
            f"stake share {_format_percent(pick['stake_fraction'])}{stake_suffix})"
        )
    return lines


def _premiership_lines(finals):
    race = _finals_field(finals, "premiership")
    lines = ["", "Road to the big dance:"]
    if not isinstance(race, dict) or not race.get("available"):
        lines.append("- Premiership probabilities are unavailable this week.")
        return lines
    for team in race["teams"]:
        if not team["alive"]:
            continue
        lines.append(
            f"- {team['seed']}. {team['team']}: "
            f"premiership {_format_probability(team['p_premiership'])}, "
            f"grand final {_format_probability(team['p_grand_final'])}"
        )
    lines.append(f"  {race.get('method', '')}".rstrip())
    return lines


def _context_watch_plain_lines(context_cards):
    """Render verified, locked Club Context facts without an LLM rewrite."""
    cards = normalize_context_cards(context_cards)
    if not cards:
        return []
    lines = ["--- CONTEXT WATCH ---", SHADOW_DISCLAIMER]
    for card in cards:
        owner = f"{card['team_name']} · " if card.get("team_name") else ""
        label = _context_category_label(card["category"])
        phase = str(card.get("phase") or "").replace("_", " ").strip()
        detail = f" · {phase.title()}" if phase else ""
        lines.extend(
            [
                f"- {owner}{label}{detail}",
                f"  {card['display_copy']}",
                f"  Source: {card['source_title']} — {card['source_url']}",
            ]
        )
    lines.extend(["---------------------", ""])
    return lines


def _context_watch_html(context_cards):
    """Email-safe Context Watch card; absent input returns byte-for-byte empty."""
    cards = normalize_context_cards(context_cards)
    if not cards:
        return ""
    items = []
    for card in cards:
        owner = f"{card['team_name']} · " if card.get("team_name") else ""
        label = _context_category_label(card["category"])
        phase = str(card.get("phase") or "").replace("_", " ").strip()
        title = f"{owner}{label}" + (f" · {phase.title()}" if phase else "")
        url = html.escape(card["source_url"], quote=True)
        source = html.escape(card["source_title"])
        items.append(
            "<div style=\"margin:0 0 12px;\">"
            "<p style=\"margin:0 0 4px; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; "
            f"font-size:14px; font-weight:700;\">{html.escape(title)}</p>"
            "<p style=\"margin:0 0 4px; color:#334155; font-family:Arial, sans-serif; "
            f"font-size:14px; line-height:1.5;\">{html.escape(card['display_copy'])}</p>"
            "<p style=\"margin:0; color:#64748b; font-family:Arial, sans-serif; font-size:12px;\">"
            f"Source: <a href=\"{url}\" style=\"color:#0369a1;\">{source}</a></p>"
            "</div>"
        )
    return (
        "<tr><td style=\"padding:6px 24px 10px;\">"
        "<div style=\"border-radius:8px; overflow:hidden; border:1px solid #93c5fd;\">"
        "<div style=\"background:#0369a1; padding:8px 14px;\">"
        "<p style=\"margin:0; color:#ffffff; font-family:'Trebuchet MS', Arial, sans-serif; "
        "font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase;\">"
        "Context Watch</p></div>"
        "<div style=\"padding:14px 16px 4px; background:#eff6ff;\">"
        f"{''.join(items)}"
        "<p style=\"margin:0 0 10px; color:#475569; font-family:Arial, sans-serif; "
        f"font-size:11px; line-height:1.4;\">{html.escape(SHADOW_DISCLAIMER)}</p>"
        "</div></div></td></tr>"
    )


def _render_plain_email(predictions, tipper_picks, folder_url, subject, opening, closing, joker_recommendation=None, news_hit=None, scoreboard=None, finals=None, context_cards=None):
    first_game = _first_game_callout(predictions)
    market_notice = _market_coverage_notice(predictions)
    is_finals = _finals_on(finals)
    lines = [subject, ""]
    scoreboard_line = _scoreboard_text_line(scoreboard)
    if scoreboard_line:
        lines.extend([scoreboard_line, ""])
    if news_hit and not is_finals:
        lines.extend(["--- THIS WEEK IN LEAGUE ---", news_hit, "---------------------------", ""])
    lines.extend(_context_watch_plain_lines(context_cards))
    lines.append(opening)
    if market_notice:
        lines.extend(["", f"MARKET DATA NOTICE: {market_notice}"])
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
        tip_prob = tip_probability(row)
        lines.append(
            f"- {row['team_home']} vs {row['team_away']}: {winner} "
            f"({_format_probability(tip_prob)}, "
            f"score {_format_predicted_score_numbers(row)}, "
            f"margin {_format_predicted_margin(row)})"
        )
        why = _why_text(row)
        if why:
            lines.append(f"  why: {why}")
        if is_finals:
            for extra in (
                _stakes_for(finals, row),
                _margin_band_text(_distribution_for(finals, row)),
                _head_to_head_for(finals, row),
            ):
                if extra:
                    lines.append(f"  {extra}")

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

    if is_finals:
        lines.extend(_market_picks_lines(finals))

    if folder_url:
        lines.extend(["", f"Tips folder: {folder_url}"])

    if is_finals:
        lines.extend(_premiership_lines(finals))
    else:
        lines.extend(["", "Joker round call:"])
        for line in _joker_summary_lines(joker_recommendation):
            lines.append(f"- {line}")

    lines.extend(["", closing])
    lines.extend(["", "Reply 'unsubscribe' to stop getting these."])
    return "\n".join(lines)


def _scoreboard_section_html(scoreboard):
    if not isinstance(scoreboard, dict):
        return ""

    def _stat(label, value):
        return (
            "<td align=\"center\" style=\"padding:10px 6px;\">"
            "<p style=\"margin:0; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; "
            f"font-size:20px; font-weight:700;\">{html.escape(value)}</p>"
            "<p style=\"margin:2px 0 0; color:#64748b; font-family:Arial, sans-serif; "
            f"font-size:11px; text-transform:uppercase; letter-spacing:0.5px;\">{html.escape(label)}</p>"
            "</td>"
        )

    cells = [
        _stat(
            f"{scoreboard['last_round_name']}",
            f"{scoreboard['last_round_correct']}/{scoreboard['last_round_games']}",
        ),
        _stat(
            "Season",
            f"{scoreboard['season_correct']}/{scoreboard['season_games']} ({scoreboard['season_accuracy']:.0%})",
        ),
    ]
    if scoreboard.get("market_accuracy") is not None:
        cells.append(_stat("Market fav", f"{scoreboard['market_accuracy']:.0%}"))

    return (
        "<tr><td style=\"padding:6px 24px 4px;\">"
        "<div style=\"border-radius:10px; background:#f0fdfa; border:1px solid #99f6e4;\">"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" style=\"border-collapse:collapse;\">"
        f"<tr>{''.join(cells)}</tr>"
        "</table>"
        "</div>"
        "</td></tr>"
    )


def _ribbon_html(theme):
    """The SPECIAL EDITION strip that marks a finals week."""
    label = theme.get("ribbon_label")
    if not label:
        return ""
    return (
        "<tr><td style=\"padding:0;\">"
        f"<div style=\"background:{theme['ribbon_background']}; padding:9px 24px;\">"
        f"<p style=\"margin:0; color:{theme['ribbon_text']}; font-family:'Trebuchet MS', Arial, sans-serif; "
        "font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase;\">"
        f"{html.escape(str(label))}"
        "</p></div></td></tr>"
    )


def _fixture_detail_html(finals, row, row_bg, theme):
    """A full-width row under a finals fixture: stakes, shape, history.

    Spans the existing four columns so the table widths a regular round renders
    with are untouched.
    """
    stakes = _stakes_for(finals, row)
    bands = _margin_band_text(_distribution_for(finals, row))
    history = _head_to_head_for(finals, row)
    if not any((stakes, bands, history)):
        return ""

    blocks = []
    if stakes:
        # A stakes line wraps to two or three lines on a phone, so it needs a
        # line-height of its own rather than the browser default.
        blocks.append(
            f"<div style=\"color:{theme['accent']}; font-weight:700; font-size:12px; "
            "letter-spacing:0.4px; line-height:1.4; text-transform:uppercase; "
            "margin-bottom:5px;\">"
            f"{html.escape(str(stakes))}</div>"
        )
    for text in (bands, history):
        if text:
            blocks.append(
                "<div style=\"color:#4b5563; font-size:12px; line-height:1.5;\">"
                f"{html.escape(str(text))}</div>"
            )
    return (
        f"<tr style=\"background:{row_bg};\">"
        "<td colspan=\"4\" style=\"padding:4px 10px 14px; border-bottom:1px solid #e5e7eb; "
        "font-family:Arial, sans-serif;\">"
        + "".join(blocks)
        + "</td></tr>"
    )


def _market_picks_html(finals, theme):
    """Line and totals picks. Finals only, and only when the model found an edge."""
    picks = _finals_field(finals, "market_picks")
    if picks is None or getattr(picks, "empty", True):
        return ""
    rows = []
    for _, pick in picks.iterrows():
        stake_text = _format_percent(pick["stake_fraction"])
        if not pd.isna(pick.get("stake_amount", pd.NA)):
            stake_text = f"{stake_text} ({_format_price(pick['stake_amount'])})"
        cells = (
            str(pick["market"]),
            str(pick["selection"]),
            _format_price(pick["price"]),
            _format_price(pick["price_min"]),
            _format_percent(pick["edge"]),
            stake_text,
        )
        rows.append(
            "<tr>"
            + "".join(
                "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
                f"font-family:Arial, sans-serif; font-size:14px;\">{html.escape(cell)}</td>"
                for cell in cells
            )
            + "</tr>"
        )
    headers = ("Market", "Selection", "Price", "Fair", "Edge", "Stake Share")
    return (
        "<tr><td style=\"padding:14px 24px 8px;\">"
        "<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid "
        f"{theme['value_accent']}; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; "
        "font-size:18px;\">Line and totals</h3>"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" "
        "style=\"border-collapse:collapse; border:1px solid #bbf7d0; border-radius:8px; overflow:hidden;\">"
        "<thead><tr style=\"background:#dcfce7;\">"
        + "".join(
            "<th align=\"left\" style=\"padding:10px; color:#15803d; "
            f"font-family:Arial, sans-serif; font-size:12px;\">{header}</th>"
            for header in headers
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></td></tr>"
    )


def _premiership_html(finals, theme):
    """The premiership race table, or an honest note when it is unavailable."""
    race = _finals_field(finals, "premiership")
    heading = (
        "<tr><td style=\"padding:14px 24px 8px;\">"
        "<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid "
        f"{theme['feature_accent']}; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; "
        "font-size:18px;\">Road to the big dance</h3>"
    )
    if not isinstance(race, dict) or not race.get("available"):
        return (
            heading
            + "<p style=\"margin:0; color:#4b5563; font-family:Arial, sans-serif; "
            "font-size:14px; line-height:1.5;\">"
            "Premiership probabilities are unavailable this week."
            "</p></td></tr>"
        )

    rows = []
    for team in race["teams"]:
        if not team["alive"]:
            continue
        emphasis = "font-weight:700;" if team["p_premiership"] >= 0.25 else ""
        cells = (
            str(team["seed"]),
            str(team["team"]),
            _format_probability(team["p_grand_final"]),
            _format_probability(team["p_premiership"]),
        )
        rows.append(
            "<tr>"
            + "".join(
                "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
                f"font-family:Arial, sans-serif; font-size:14px; {emphasis}\">{html.escape(cell)}</td>"
                for cell in cells
            )
            + "</tr>"
        )
    headers = ("Seed", "Team", "Grand Final", "Premiership")
    return (
        heading
        + "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" "
        f"style=\"border-collapse:collapse; border:1px solid {theme['feature_accent']}; "
        "border-radius:8px; overflow:hidden;\">"
        "<thead><tr style=\"background:#f1f5f9;\">"
        + "".join(
            "<th align=\"left\" style=\"padding:10px; color:#334155; "
            f"font-family:Arial, sans-serif; font-size:12px;\">{header}</th>"
            for header in headers
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "<p style=\"margin:8px 0 0; color:#9ca3af; font-family:Arial, sans-serif; font-size:11px;\">"
        + html.escape(str(race.get("method", "")))
        + f" Based on {race['simulations']:,} simulated finals series."
        + "</p></td></tr>"
    )


def _render_html_email(
    predictions,
    tipper_picks,
    folder_url,
    opening,
    closing,
    banner_available,
    joker_recommendation=None,
    news_hit=None,
    scoreboard=None,
    finals=None,
    context_cards=None,
):
    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    first_game = _first_game_callout(predictions)
    market_notice = _market_coverage_notice(predictions)
    is_finals = _finals_on(finals)
    theme = _theme(finals)
    if is_finals:
        news_hit = None
    heading = (
        f"{_finals_field(finals, 'display_name') or round_name} {competition_year}"
        if is_finals
        else f"{round_name} {competition_year} Tips"
    )

    match_rows = []
    for i, (_, row) in enumerate(predictions.iterrows()):
        winner = _prediction_winner(row)
        row_bg = "#f9fafb" if i % 2 == 0 else "#ffffff"
        # Colour by confidence in the tipped team, not by home-win probability:
        # a strong away favourite must read green, not red.
        tip_prob = tip_probability(row)
        if pd.isna(tip_prob):
            tip_prob = 0.5
        if tip_prob >= 0.70:
            badge_bg, badge_color = "#dcfce7", "#15803d"
        elif tip_prob >= 0.55:
            badge_bg, badge_color = "#fef9c3", "#854d0e"
        else:
            badge_bg, badge_color = "#fee2e2", "#b91c1c"
        # The why line rides inside the existing Tip cell rather than becoming a
        # fifth column: the header row and the table widths stay exactly as they
        # were, so a round without explanations renders identically.
        why = _why_text(row)
        why_html = (
            "<div style=\"margin-top:3px; font-weight:400; font-size:12px; "
            f"line-height:1.4; color:#6b7280;\">{html.escape(why)}</div>"
            if why
            else ""
        )
        # A fixture that carries a stakes block below it is one unit: the rule
        # belongs under the block, not through the middle of it.
        detail = _fixture_detail_html(finals, row, row_bg, theme) if is_finals else ""
        cell_border = "" if detail else "border-bottom:1px solid #e5e7eb; "
        match_rows.append(
            f"<tr style=\"background:{row_bg};\">"
            f"<td style=\"padding:12px 10px; {cell_border}color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px; width:36%;\">"
            f"{html.escape(str(row['team_home']))} vs {html.escape(str(row['team_away']))}"
            "</td>"
            f"<td style=\"padding:12px 10px; {cell_border}color:#0f766e; "
            "font-family:Arial, sans-serif; font-size:15px; font-weight:700; width:32%;\">"
            f"<div>{html.escape(str(winner))}</div>{why_html}"
            "</td>"
            f"<td style=\"padding:12px 10px; {cell_border}width:16%;\">"
            f"<span style=\"display:inline-block; padding:3px 7px; border-radius:12px; "
            f"background:{badge_bg}; color:{badge_color}; font-family:Arial, sans-serif; font-size:12px; font-weight:700;\">"
            f"{_format_probability(tip_prob)}</span>"
            "</td>"
            f"<td style=\"padding:12px 10px; {cell_border}color:#374151; "
            "font-family:Arial, sans-serif; font-size:13px; width:16%;\">"
            f"H {_format_market_price(row['team_head_to_head_odds_home'], row.get('market_odds_fresh', True))}"
            f"<br>A {_format_market_price(row['team_head_to_head_odds_away'], row.get('market_odds_fresh', True))}"
            "</td>"
            "</tr>"
        )
        if detail:
            match_rows.append(detail)

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

    market_notice_section = ""
    if market_notice:
        market_notice_section = (
            "<tr><td style=\"padding:8px 24px 6px;\">"
            "<div style=\"padding:12px 14px; border-radius:8px; "
            "background:#fff7ed; border:1px solid #fdba74;\">"
            "<p style=\"margin:0; color:#9a3412; font-family:Arial, sans-serif; "
            "font-size:13px; line-height:1.5;\">"
            f"<strong>Market data notice:</strong> {html.escape(market_notice)}"
            "</p></div></td></tr>"
        )

    joker_lines = _joker_reader_lines(joker_recommendation)
    joker_body_html = "".join(
        [
            "<p style=\"margin:0 0 6px; color:#1f2937; font-family:Arial, sans-serif; font-size:14px; line-height:1.5;\">"
            f"{html.escape(line)}"
            "</p>"
            for line in joker_lines
        ]
    )
    joker_footnote = _joker_footnote(joker_recommendation)
    joker_footnote_html = ""
    if joker_footnote:
        joker_footnote_html = (
            "<p style=\"margin:8px 0 0; color:#9ca3af; font-family:Arial, sans-serif; font-size:11px;\">"
            f"{html.escape(joker_footnote)}"
            "</p>"
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
        f"{joker_body_html}"
        f"{joker_footnote_html}"
        "</div>"
    )

    if is_finals:
        # The comp the joker belongs to has finished. The slot goes to the only
        # question left in September.
        feature_section = _premiership_html(finals, theme)
    else:
        feature_section = (
            "<tr><td style=\"padding:14px 24px 8px;\">"
            "<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid "
            f"{theme['feature_accent']}; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; "
            "font-size:18px;\">Joker round call</h3>"
            f"{joker_section}"
            "</td></tr>"
        )

    banner_html = ""
    if banner_available:
        banner_html = (
            "<img src=\"cid:footy_tipper_email_banner\" alt=\"Footy Tipper\" "
            "style=\"display:block; width:100%; max-width:680px; height:auto; border:0; border-radius:12px 12px 0 0;\">"
        )
    else:
        banner_html = (
            f"<div style=\"padding:26px 24px; background:{theme['header_gradient']}; border-radius:12px 12px 0 0;\">"
            "<h1 style=\"margin:0; color:#ffffff; font-family:'Trebuchet MS', Arial, sans-serif; font-size:30px; letter-spacing:0.5px;\">"
            "Footy Tipper"
            "</h1>"
            "</div>"
        )

    buttons = []
    site_url = os.getenv("FOOTY_TIPPER_SITE_URL", "").strip()
    if site_url:
        safe_site_url = html.escape(site_url, quote=True)
        buttons.append(
            f"<a href=\"{safe_site_url}\" "
            "style=\"display:inline-block; background:#0369a1; color:#ffffff; text-decoration:none; "
            "font-family:Arial, sans-serif; font-size:14px; font-weight:700; padding:12px 18px; "
            "border-radius:8px; margin-right:10px;\">"
            "View This Round Online"
            "</a>"
        )
    if folder_url:
        safe_url = html.escape(folder_url, quote=True)
        buttons.append(
            f"<a href=\"{safe_url}\" "
            "style=\"display:inline-block; background:#0f766e; color:#ffffff; text-decoration:none; "
            "font-family:Arial, sans-serif; font-size:14px; font-weight:700; padding:12px 18px; border-radius:8px;\">"
            "Open Tips Folder"
            "</a>"
        )
    folder_button = ""
    if buttons:
        folder_button = (
            "<tr><td style=\"padding:8px 24px 24px;\">"
            f"{''.join(buttons)}"
            "</td></tr>"
        )

    return (
        "<html><body style=\"margin:0; padding:20px; background:#eef2f7;\">"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" style=\"border-collapse:collapse;\">"
        "<tr><td align=\"center\">"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"680\" "
        "style=\"max-width:680px; width:100%; border-collapse:collapse; background:#ffffff; border-radius:12px;\">"
        f"<tr><td>{banner_html}</td></tr>"
        f"{_ribbon_html(theme)}"
        "<tr><td style=\"padding:24px 24px 10px;\">"
        "<h2 style=\"margin:0; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:26px;\">"
        f"{html.escape(str(heading))}"
        "</h2>"
        "</td></tr>"
        f"{_scoreboard_section_html(scoreboard)}"
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
        f"{_context_watch_html(context_cards)}"
        "<tr><td style=\"padding:6px 24px 6px;\">"
        f"{_to_html_paragraphs(opening)}"
        "</td></tr>"
        f"{market_notice_section}"
        f"{first_game_section}"
        "<tr><td style=\"padding:10px 24px 8px;\">"
        f"<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid {theme['accent']}; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:18px;\">Predicted winners</h3>"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\" "
        "style=\"border-collapse:collapse; border:1px solid #e5e7eb; border-radius:8px; overflow:hidden;\">"
        "<thead><tr style=\"background:#f9fafb;\">"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Fixture</th>"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Tip</th>"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Confidence</th>"
        "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">H2H Odds</th>"
        "</tr></thead>"
        "<tbody>"
        f"{''.join(match_rows)}"
        "</tbody></table>"
        "</td></tr>"
        "<tr><td style=\"padding:14px 24px 8px;\">"
        f"<h3 style=\"margin:0 0 10px; padding-left:10px; border-left:4px solid {theme['value_accent']}; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:18px;\">Value picks</h3>"
        f"{value_section}"
        "</td></tr>"
        f"{_market_picks_html(finals, theme)}"
        f"{feature_section}"
        f"{folder_button}"
        "<tr><td style=\"padding:6px 24px 22px;\">"
        f"{_to_html_paragraphs(closing)}"
        "</td></tr>"
        "<tr><td style=\"padding:16px 24px 24px; border-top:1px solid #e5e7eb;\">"
        "<p style=\"margin:0 0 4px; color:#0f766e; font-family:'Trebuchet MS', Arial, sans-serif; font-size:13px; font-weight:700;\">Bring back the biff.</p>"
        "<p style=\"margin:0 0 4px; color:#9ca3af; font-family:Arial, sans-serif; font-size:11px;\">Generated by Footy Tipper.</p>"
        "<p style=\"margin:0; color:#9ca3af; font-family:Arial, sans-serif; font-size:11px;\">Reply &quot;unsubscribe&quot; to stop getting these.</p>"
        "</td></tr>"
        "</table>"
        "</td></tr>"
        "</table>"
        "</body></html>"
    )
