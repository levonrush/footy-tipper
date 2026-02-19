import html
import json
import os
import pandas as pd
import re
import sqlite3
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

# For direct OpenAI API calls (removing langchain)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# The 'get_predictions' function reads the predictions from the SQLite database and returns them as a pandas DataFrame.
def get_predictions(db_path, project_root):
    con = sqlite3.connect(str(db_path))
    with open(project_root / 'pipeline/common' / 'sql/prediction_table.sql', 'r') as file:
        query = file.read()
    predictions = pd.read_sql_query(query, con)
    con.close()
    return predictions

# The 'get_tipper_picks' function calculates the odds thresholds and returns a DataFrame of tipper picks.
def get_tipper_picks(predictions, prod_run=False):
    if predictions.empty:
        return pd.DataFrame(columns=['team', 'price', 'price_min'])

    predictions = predictions.copy()
    predictions['home_odds_thresh'] = 1 / predictions['home_team_win_prob'].replace(0, pd.NA)
    predictions['away_odds_thresh'] = 1 / predictions['home_team_lose_prob'].replace(0, pd.NA)

    home_picks = predictions[predictions['home_team_result'] == 'Win'][['team_home', 'team_head_to_head_odds_home', 'home_odds_thresh']].copy()
    home_picks.rename(columns={'team_home': 'team', 'team_head_to_head_odds_home': 'price', 'home_odds_thresh': 'price_min'}, inplace=True)
    away_picks = predictions[predictions['home_team_result'] == 'Loss'][['team_away', 'team_head_to_head_odds_away', 'away_odds_thresh']].copy()
    away_picks.rename(columns={'team_away': 'team', 'team_head_to_head_odds_away': 'price', 'away_odds_thresh': 'price_min'}, inplace=True)
    tipper_picks = pd.concat([home_picks, away_picks], ignore_index=True)
    tipper_picks = tipper_picks.dropna(subset=['price', 'price_min'])
    tipper_picks = tipper_picks[tipper_picks['price'] > (tipper_picks['price_min'] * 1.05)]
    return tipper_picks

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
    return f"{float(value):.2f}"


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


def _build_fallback_copy(predictions, folder_url):
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


def _build_prompt_input(predictions, tipper_picks):
    fixture_lines = []
    for _, row in predictions.iterrows():
        winner = row['team_home'] if row['home_team_result'] == 'Win' else row['team_away']
        fixture_lines.append(
            f"- {row['team_home']} vs {row['team_away']}: tip {winner} "
            f"(home win {_format_probability(row['home_team_win_prob'])}, "
            f"away win {_format_probability(row['home_team_lose_prob'])}, "
            f"market {row['team_home']} {_format_price(row['team_head_to_head_odds_home'])}, "
            f"{row['team_away']} {_format_price(row['team_head_to_head_odds_away'])})"
        )

    pick_lines = []
    if tipper_picks.empty:
        pick_lines.append("- None flagged by the model.")
    else:
        for _, row in tipper_picks.iterrows():
            pick_lines.append(
                f"- {row['team']}: market {_format_price(row['price'])}, model threshold {_format_price(row['price_min'])}"
            )

    return "\n".join(fixture_lines), "\n".join(pick_lines)


def _parse_json_object(text):
    if not text:
        return None
    candidates = [text.strip()]
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1].strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _generate_openai_copy(predictions, tipper_picks, api_key, folder_url, temperature):
    if predictions.empty:
        return None
    if not api_key:
        print("OPENAI_KEY is not configured. Using fallback email content.")
        return None
    if OpenAI is None:
        print("OpenAI SDK is unavailable. Using fallback email content.")
        return None

    fixtures_text, picks_text = _build_prompt_input(predictions, tipper_picks)
    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    special_event_context = _special_event_context(round_name, competition_year)
    folder_line = folder_url if folder_url else "No public folder URL is configured this run."
    prompt = f"""
You are writing copy for an NRL tipping email from Reg Reagan.

Round: {round_name} {competition_year}
Special event context: {special_event_context['event_name']}
Special event writing angle: {special_event_context['prompt_angle']}
Tips folder: {folder_line}

Fixtures and model picks:
{fixtures_text}

Value picks:
{picks_text}

Return JSON only with this exact schema:
{{
  "subject": "short email subject line, max 75 chars",
  "opening": "2-4 short paragraphs in plain text, with cheeky NRL banter",
  "closing": "1-2 short paragraphs and must end with Bring back the biff."
}}

Rules:
- Mention the Newcastle Knights positively.
- Take a light dig at Manly.
- Include this disclaimer naturally: if people are in tipping comps at Seven Seas Hotel in Carrington or the Hunter Water work comp, they should not use these tips.
- Keep it punchy and readable.
- Do not include markdown, HTML, or extra keys.
"""

    client = OpenAI(api_key=api_key)
    configured_model = os.getenv("OPENAI_MODEL")
    model_candidates = (
        [configured_model]
        if configured_model
        else ["gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1"]
    )
    last_exception = None

    for model_name in model_candidates:
        if not model_name:
            continue
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a witty Australian NRL writer with concise, readable style."},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": 600,
                "response_format": {"type": "json_object"},
            }
            # Keep GPT-5 calls minimal for widest SDK compatibility.
            if not str(model_name).lower().startswith("gpt-5"):
                request_kwargs["temperature"] = temperature

            response = client.chat.completions.create(**request_kwargs)
            payload = _parse_json_object(response.choices[0].message.content or "")
            if not payload:
                print(f"OpenAI email generation returned non-JSON payload for model '{model_name}'.")
                continue
            subject = str(payload.get("subject", "")).strip()
            opening = str(payload.get("opening", "")).strip()
            closing = str(payload.get("closing", "")).strip()
            if not subject or not opening or not closing:
                print(f"OpenAI email generation returned incomplete JSON keys for model '{model_name}'.")
                continue
            print(f"OpenAI email generation model: {model_name}")
            return {
                "subject": subject,
                "opening": opening,
                "closing": closing,
            }
        except Exception as exc:
            last_exception = exc
            print(f"OpenAI email generation failed for model '{model_name}' ({exc}).")
            if configured_model:
                break

    if last_exception is not None:
        print(f"OpenAI email generation failed ({last_exception}). Using fallback email content.")
    return None


def _to_html_paragraphs(text):
    blocks = []
    for paragraph in [p.strip() for p in text.split("\n\n") if p.strip()]:
        safe = html.escape(paragraph).replace("\n", "<br>")
        blocks.append(
            "<p style=\"margin:0 0 14px; color:#1f2937; font-family:'Trebuchet MS', Arial, sans-serif; "
            "font-size:16px; line-height:1.55;\">"
            f"{safe}"
            "</p>"
        )
    return "".join(blocks)


def _render_plain_email(predictions, tipper_picks, folder_url, subject, opening, closing):
    lines = [subject, "", opening, "", "Predicted winners:"]
    for _, row in predictions.iterrows():
        winner = row['team_home'] if row['home_team_result'] == 'Win' else row['team_away']
        lines.append(
            f"- {row['team_home']} vs {row['team_away']}: {winner} "
            f"(home {_format_probability(row['home_team_win_prob'])}, "
            f"away {_format_probability(row['home_team_lose_prob'])})"
        )

    lines.append("")
    if tipper_picks.empty:
        lines.append("Value picks: none flagged this round.")
    else:
        lines.append("Value picks:")
        for _, row in tipper_picks.iterrows():
            lines.append(
                f"- {row['team']} at {_format_price(row['price'])} "
                f"(model threshold {_format_price(row['price_min'])})"
            )

    if folder_url:
        lines.extend(["", f"Tips folder: {folder_url}"])

    lines.extend(["", closing])
    return "\n".join(lines)


def _render_html_email(predictions, tipper_picks, folder_url, opening, closing, banner_available):
    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]

    match_rows = []
    for _, row in predictions.iterrows():
        winner = row['team_home'] if row['home_team_result'] == 'Win' else row['team_away']
        match_rows.append(
            "<tr>"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{html.escape(str(row['team_home']))} vs {html.escape(str(row['team_away']))}"
            "</td>"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; color:#0f766e; "
            "font-family:Arial, sans-serif; font-size:14px; font-weight:700;\">"
            f"{html.escape(str(winner))}"
            "</td>"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; color:#374151; "
            "font-family:Arial, sans-serif; font-size:13px;\">"
            f"H {_format_probability(row['home_team_win_prob'])} | A {_format_probability(row['home_team_lose_prob'])}"
            "</td>"
            "<td style=\"padding:12px 10px; border-bottom:1px solid #e5e7eb; color:#374151; "
            "font-family:Arial, sans-serif; font-size:13px;\">"
            f"H {_format_price(row['team_head_to_head_odds_home'])} | A {_format_price(row['team_head_to_head_odds_away'])}"
            "</td>"
            "</tr>"
        )

    pick_rows = []
    for _, row in tipper_picks.iterrows():
        pick_rows.append(
            "<tr>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{html.escape(str(row['team']))}"
            "</td>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{_format_price(row['price'])}"
            "</td>"
            "<td style=\"padding:10px; border-bottom:1px solid #f3f4f6; color:#111827; "
            "font-family:Arial, sans-serif; font-size:14px;\">"
            f"{_format_price(row['price_min'])}"
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
            "style=\"border-collapse:collapse; border:1px solid #e5e7eb; border-radius:8px; overflow:hidden;\">"
            "<thead>"
            "<tr style=\"background:#f9fafb;\">"
            "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Team</th>"
            "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Market</th>"
            "<th align=\"left\" style=\"padding:10px; color:#374151; font-family:Arial, sans-serif; font-size:12px;\">Model Min</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            f"{''.join(pick_rows)}"
            "</tbody>"
            "</table>"
        )

    banner_html = ""
    if banner_available:
        banner_html = (
            "<img src=\"cid:footy_tipper_email_banner\" alt=\"Footy Tipper\" "
            "style=\"display:block; width:100%; max-width:680px; height:auto; border:0;\">"
        )
    else:
        banner_html = (
            "<div style=\"padding:26px 24px; background:linear-gradient(135deg, #115e59 0%, #0369a1 100%);\">"
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
        "style=\"max-width:680px; width:100%; border-collapse:collapse; background:#ffffff; border-radius:12px; overflow:hidden;\">"
        f"<tr><td>{banner_html}</td></tr>"
        "<tr><td style=\"padding:24px 24px 10px;\">"
        "<h2 style=\"margin:0; color:#0f172a; font-family:'Trebuchet MS', Arial, sans-serif; font-size:26px;\">"
        f"{html.escape(str(round_name))} {html.escape(str(competition_year))} Tips"
        "</h2>"
        "</td></tr>"
        "<tr><td style=\"padding:6px 24px 6px;\">"
        f"{_to_html_paragraphs(opening)}"
        "</td></tr>"
        "<tr><td style=\"padding:10px 24px 8px;\">"
        "<h3 style=\"margin:0 0 10px; color:#111827; font-family:'Trebuchet MS', Arial, sans-serif; font-size:18px;\">Predicted winners</h3>"
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
        "<h3 style=\"margin:0 0 10px; color:#111827; font-family:'Trebuchet MS', Arial, sans-serif; font-size:18px;\">Value picks</h3>"
        f"{value_section}"
        "</td></tr>"
        f"{folder_button}"
        "<tr><td style=\"padding:6px 24px 22px;\">"
        f"{_to_html_paragraphs(closing)}"
        "</td></tr>"
        "<tr><td style=\"padding:16px 24px 24px; border-top:1px solid #e5e7eb;\">"
        "<p style=\"margin:0; color:#6b7280; font-family:Arial, sans-serif; font-size:12px; line-height:1.5;\">"
        "Generated by Footy Tipper. Bring back the biff."
        "</p>"
        "</td></tr>"
        "</table>"
        "</td></tr>"
        "</table>"
        "</body></html>"
    )


def generate_reg_regan_email_payload(predictions, tipper_picks, api_key, folder_url, temperature, use_openai=True):
    fallback_copy = _build_fallback_copy(predictions, folder_url)
    if not use_openai:
        print("OpenAI generation disabled. Using fallback email content.")
    openai_copy = (
        _generate_openai_copy(predictions, tipper_picks, api_key, folder_url, temperature)
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

    banner_path = _resolve_banner_path()
    plain_email = _render_plain_email(
        predictions,
        tipper_picks,
        folder_url,
        copy["subject"],
        copy["opening"],
        copy["closing"],
    )
    html_email = _render_html_email(
        predictions,
        tipper_picks,
        folder_url,
        copy["opening"],
        copy["closing"],
        banner_available=bool(banner_path),
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
def generate_reg_regan_email(predictions, tipper_picks, api_key, folder_url, temperature):
    payload = generate_reg_regan_email_payload(
        predictions,
        tipper_picks,
        api_key,
        folder_url,
        temperature,
        use_openai=bool(api_key),
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
        return
    if not sender_email or not sender_password:
        print("Email send skipped: missing MY_EMAIL or EMAIL_PASSWORD.")
        return
    if not os.path.exists(json_path):
        print(f"Email send skipped: missing Google service account token at {json_path}.")
        return

    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_file(json_path, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open(doc_name).sheet1
    email_data = sheet.get_all_records()
    recipient_emails = [row['Email'] for row in email_data if row.get('Email')]
    if not recipient_emails:
        print("Email send skipped: no recipients found in the email list.")
        return
    
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


def send_test_email(subject, message, sender_email, sender_password, recipient_email, html_message=None, inline_images=None):
    if not sender_email or not sender_password:
        print("Test email skipped: missing MY_EMAIL or EMAIL_PASSWORD.")
        return
    if not recipient_email:
        print("Test email skipped: missing recipient email.")
        return

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
