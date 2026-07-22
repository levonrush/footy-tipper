"""Prediction retrieval, Google Drive upload, and SMTP email distribution."""

import os
import sqlite3
from dataclasses import dataclass, field
from email.utils import parseaddr

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


EMAIL_LIST_SCOPE = (
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
)


class EmailPreparationError(RuntimeError):
    """A safe, pre-SMTP production email configuration failure."""


@dataclass(frozen=True)
class PreparedEmailDelivery:
    """Validated production envelope details resolved before delivery is claimed."""

    sender_email: str
    sender_password: str = field(repr=False)
    recipients: tuple[str, ...]


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


def _build_mime_message(
    subject,
    sender_email,
    recipients,
    plain_message,
    html_message=None,
    inline_images=None,
    bcc_recipients=False,
):
    has_html = bool(html_message)
    msg = MIMEMultipart("related") if has_html else MIMEMultipart()
    msg["From"] = sender_email
    # For list sends, keep recipient addresses out of the headers (effective BCC):
    # they only appear in the SMTP envelope. Recipients must not see each other.
    msg["To"] = sender_email if bcc_recipients else ", ".join(recipients)
    msg["Subject"] = subject
    if sender_email:
        msg["List-Unsubscribe"] = f"<mailto:{sender_email}?subject=unsubscribe>"

    if has_html:
        alternatives = MIMEMultipart("alternative")
        alternatives.attach(MIMEText(plain_message, "plain", "utf-8"))
        alternatives.attach(MIMEText(html_message, "html", "utf-8"))
        msg.attach(alternatives)
        _attach_inline_images(msg, inline_images)
    else:
        msg.attach(MIMEText(plain_message, "plain", "utf-8"))

    return msg


def _normalized_email_address(value):
    raw = str(value or "").strip()
    if not raw or "\r" in raw or "\n" in raw:
        raise ValueError("missing or unsafe email address")

    _display_name, address = parseaddr(raw)
    local, separator, domain = address.rpartition("@")
    if (
        not separator
        or not local
        or not domain
        or address.count("@") != 1
        or any(character.isspace() for character in address)
    ):
        raise ValueError("invalid email address")
    return address


def prepare_email_delivery(doc_name, sender_email, sender_password, json_path):
    """Resolve and validate the production envelope before claiming delivery.

    Google access and recipient-list failures are deterministic: no SMTP call
    could have happened yet, so callers must run this before writing a pending
    delivery marker.
    """
    if service_account is None or gspread is None:
        raise EmailPreparationError("Google Sheets dependencies are not installed.")
    if not str(sender_email or "").strip() or not str(sender_password or "").strip():
        raise EmailPreparationError("MY_EMAIL or EMAIL_PASSWORD is missing.")
    if not os.path.isfile(json_path):
        raise EmailPreparationError("The Google service-account token is missing.")

    try:
        normalized_sender = _normalized_email_address(sender_email)
    except ValueError as exc:
        raise EmailPreparationError("MY_EMAIL is not a valid email address.") from exc

    try:
        creds = service_account.Credentials.from_service_account_file(
            json_path,
            scopes=EMAIL_LIST_SCOPE,
        )
    except Exception as exc:
        raise EmailPreparationError(
            "The Google service-account token could not be loaded."
        ) from exc

    try:
        client = gspread.authorize(creds)
        sheet = client.open(doc_name).sheet1
        email_data = sheet.get_all_records()
    except Exception as exc:
        raise EmailPreparationError(
            "The production recipient list could not be read from Google Sheets."
        ) from exc

    recipient_emails = []
    seen = set()
    invalid_count = 0
    for row in email_data:
        value = row.get("Email") if isinstance(row, dict) else None
        if not str(value or "").strip():
            continue
        try:
            recipient = _normalized_email_address(value)
        except ValueError:
            invalid_count += 1
            continue
        identity = recipient.casefold()
        if identity not in seen:
            recipient_emails.append(recipient)
            seen.add(identity)

    if invalid_count:
        raise EmailPreparationError(
            f"The production recipient list contains {invalid_count} invalid email address(es)."
        )
    if not recipient_emails:
        raise EmailPreparationError("The production recipient list is empty.")

    return PreparedEmailDelivery(
        sender_email=normalized_sender,
        sender_password=str(sender_password),
        recipients=tuple(recipient_emails),
    )


# The 'send_emails' function sends to a prevalidated production envelope.
def send_emails(
    subject,
    message,
    prepared_delivery,
    html_message=None,
    inline_images=None,
):
    if not isinstance(prepared_delivery, PreparedEmailDelivery):
        raise TypeError("prepared_delivery must come from prepare_email_delivery().")

    msg = _build_mime_message(
        subject=subject,
        sender_email=prepared_delivery.sender_email,
        recipients=prepared_delivery.recipients,
        plain_message=message,
        html_message=html_message,
        inline_images=inline_images,
        bcc_recipients=True,
    )

    server = smtplib.SMTP("smtp.gmail.com", 587)
    try:
        server.starttls()
        server.login(
            prepared_delivery.sender_email,
            prepared_delivery.sender_password,
        )
        refused = server.sendmail(
            prepared_delivery.sender_email,
            list(prepared_delivery.recipients),
            msg.as_string(),
        )
    finally:
        try:
            server.quit()
        except Exception:
            pass

    if refused:
        print(
            "SMTP refused one or more production recipients. Delivery may be partial; "
            "the round must remain blocked as uncertain."
        )
        return False

    # Truthy on success; callers can use the count for the send ledger.
    return len(prepared_delivery.recipients)


def backup_db_to_drive(db_path, json_path, parent_folder_id, keep=8):
    """Snapshot the SQLite DB, gzip it, and upload to a Drive backups folder.

    The DB can hold costly-to-rebuild feed/lineup history and operational
    state. Uses the sqlite3 backup API so the snapshot is consistent even
    mid-write. Keeps the newest `keep` backups.
    Fail-soft: backup problems must never break a send.
    """
    import gzip
    import tempfile
    from datetime import datetime, timezone

    if service_account is None or build is None or MediaFileUpload is None:
        print("DB backup skipped: Google Drive dependencies are not installed.")
        return False
    if not parent_folder_id:
        print("DB backup skipped: FOLDER_ID is not configured.")
        return False
    if not os.path.exists(json_path):
        print(f"DB backup skipped: missing Google service account token at {json_path}.")
        return False
    if not os.path.exists(str(db_path)):
        print(f"DB backup skipped: database not found at {db_path}.")
        return False

    tmp_dir = tempfile.mkdtemp(prefix="footy-tipper-backup-")
    snapshot_path = os.path.join(tmp_dir, "snapshot.sqlite")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_name = f"footy-tipper-db-{stamp}.sqlite.gz"
    archive_path = os.path.join(tmp_dir, archive_name)

    try:
        source = sqlite3.connect(str(db_path))
        try:
            snapshot = sqlite3.connect(snapshot_path)
            try:
                source.backup(snapshot)
            finally:
                snapshot.close()
        finally:
            source.close()

        with open(snapshot_path, "rb") as src, gzip.open(archive_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        creds = service_account.Credentials.from_service_account_file(json_path)
        drive_service = build('drive', 'v3', credentials=creds)

        query = (
            f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' "
            "and name='backups' and trashed=false"
        )
        results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        items = results.get('files', [])
        if items:
            backups_folder_id = items[0]['id']
        else:
            folder = drive_service.files().create(
                body={
                    'name': 'backups',
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [parent_folder_id],
                },
                fields='id',
            ).execute()
            backups_folder_id = folder.get('id')

        media = MediaFileUpload(archive_path, mimetype='application/gzip')
        drive_service.files().create(
            body={'name': archive_name, 'parents': [backups_folder_id]},
            media_body=media,
            fields='id',
        ).execute()
        print(f"DB backup uploaded: {archive_name}")

        # Prune old backups beyond `keep`.
        listing = drive_service.files().list(
            q=f"'{backups_folder_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id, name)',
            orderBy='name desc',
        ).execute()
        old = listing.get('files', [])[int(keep):]
        for stale in old:
            try:
                drive_service.files().delete(fileId=stale['id']).execute()
                print(f"DB backup pruned: {stale.get('name')}")
            except Exception as exc:
                print(f"DB backup prune failed for {stale.get('name')} ({exc}).")
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"DB backup failed ({exc}).")
        return False
    finally:
        for path in (archive_path, snapshot_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass


def _ensure_email_sends_table(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS email_sends (
                competition_year INTEGER NOT NULL,
                round_id INTEGER NOT NULL,
                sent_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                recipients_count INTEGER,
                source TEXT NOT NULL DEFAULT 'unknown',
                PRIMARY KEY (competition_year, round_id)
            )
            """
        )
        con.commit()
    finally:
        con.close()


def email_send_already_recorded(db_path, competition_year, round_id):
    """Return the prior send record for (year, round) or None.

    Used to keep production sends idempotent: a re-run of the predict/send
    flow must not email the list twice for the same round.
    """
    if competition_year is None or round_id is None:
        return None

    row = None
    con = None
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT competition_year, round_id, sent_at_utc, recipients_count, source
            FROM email_sends
            WHERE competition_year = ? AND round_id = ?
            LIMIT 1
            """,
            (int(competition_year), int(round_id)),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            print(f"Email send ledger lookup failed ({exc}).")
    except Exception as exc:
        print(f"Email send ledger lookup failed ({exc}).")
    finally:
        if con is not None:
            con.close()

    if row is None:
        return None
    return dict(row)


def record_email_send(db_path, competition_year, round_id, recipients_count=None, source="send"):
    """Record a successful production send for (year, round). Fail-soft."""
    if competition_year is None or round_id is None:
        return False

    try:
        _ensure_email_sends_table(db_path)
        con = sqlite3.connect(str(db_path))
        try:
            con.execute(
                """
                INSERT OR IGNORE INTO email_sends (competition_year, round_id, recipients_count, source)
                VALUES (?, ?, ?, ?)
                """,
                (int(competition_year), int(round_id), recipients_count, str(source or "send")),
            )
            con.commit()
        finally:
            con.close()
        return True
    except Exception as exc:
        print(f"Email send ledger write failed ({exc}).")
        return False


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
    try:
        server.starttls()
        server.login(sender_email, sender_password)
        refused = server.sendmail(sender_email, [recipient_email], msg.as_string())
    finally:
        try:
            server.quit()
        except Exception:
            pass
    if refused:
        print("Test email was refused by SMTP.")
        return False
    return True
