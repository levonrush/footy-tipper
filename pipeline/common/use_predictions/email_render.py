"""Plain-text and HTML rendering of the weekly tipping email."""

import html

import pandas as pd

from pipeline.common.use_predictions.joker import _round_label


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



def _coerce_int(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(round(float(numeric)))


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
