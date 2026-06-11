"""Prediction retrieval, Google Drive upload, and SMTP email distribution."""

import os
import sqlite3

import pandas as pd

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
