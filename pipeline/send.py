# Description: This script is used to send the predictions to the email list
print("Running the send.py script...")

# import libraries
import os
import sys
import pathlib
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))

# get the parent directory
parent_dir = os.path.dirname(script_dir)

# add the parent directory to the system path
sys.path.insert(0, parent_dir)

from pipeline.common.use_predictions import sending_functions as sf

# Get to the root directory
project_root = pathlib.Path().absolute()

# Now construct the relative path to your SQLite database
db_path = project_root / "data" / "footy-tipper-db.sqlite"
secrets_path = project_root / "secrets.env"
json_path = project_root / "service-account-token.json"

# Load the secrets
print("Loading the secrets...")
load_dotenv(dotenv_path=secrets_path)

# Get predictions
print("Getting predictions...")
predictions = sf.get_predictions(db_path, project_root)

# Offseason-safe behavior: skip send/upload when there are no upcoming pre-game fixtures.
if predictions.empty:
    print("No pre-game predictions available. Skipping upload and email send.")
    sys.exit(0)

# Get tipper picks
print("Generating tipper picks...")
tipper_picks = sf.get_tipper_picks(predictions)

# Get joker recommendation based on market-implied round difficulty profile.
print("Scoring joker round options...")
joker_recommendation = sf.get_joker_round_recommendation(db_path, project_root, predictions)
print(joker_recommendation.get("headline", "Joker call unavailable"))
print(joker_recommendation.get("detail", ""))

# Upload the predictions
print("Uploading predictions to Google Drive...")
sf.upload_df_to_drive(
    predictions, 
    json_path, 
    os.getenv('FOLDER_ID'), 
    "predictions.csv"
)

# Generate the Reg Regan email
print("Generating the Reg Regan email...")
email_payload = sf.generate_reg_regan_email_payload(
    predictions, 
    tipper_picks, 
    os.getenv('ANTHROPIC_API_KEY'),
    os.getenv('FOLDER_URL'),
    0.9,
    joker_recommendation=joker_recommendation,
)

print(email_payload["plain_text"])

# Send the email
print("Sending the email...")
sent = sf.send_emails(
    "footy-tipper-email-list", 
    email_payload["subject"], 
    email_payload["plain_text"], 
    os.getenv('MY_EMAIL'), 
    os.getenv('EMAIL_PASSWORD'), 
    json_path,
    html_message=email_payload["html_text"],
    inline_images=email_payload["inline_images"],
)
if sent:
    usage_outcome = sf.persist_joker_usage_if_applicable(
        db_path,
        joker_recommendation,
        allow_write=True,
        source="send_py_production",
    )
    if usage_outcome.get("recorded"):
        print(
            "Recorded joker usage for "
            f"{usage_outcome.get('competition_year')} round {usage_outcome.get('round_id')}."
        )
    elif usage_outcome.get("reason") == "already_recorded":
        print("Joker usage already recorded for this season.")
    elif usage_outcome.get("reason") == "not_play_signal":
        print("Joker recommendation is HOLD. Usage state unchanged.")
    elif usage_outcome.get("reason") == "already_used":
        print("Joker already marked as used for this season. Usage state unchanged.")
    elif usage_outcome.get("reason") == "db_error":
        print(f"Joker usage write failed: {usage_outcome.get('error', 'unknown db error')}")
else:
    print("Email send skipped or failed. Joker usage state unchanged.")

print("Send step complete.")
