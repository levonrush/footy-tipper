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
reg_reagan = sf.generate_reg_regan_email(
    predictions, 
    tipper_picks, 
    os.getenv('OPENAI_KEY'), 
    os.getenv('FOLDER_URL'),
    0.9
)

print(reg_reagan)

# Send the email
print("Sending the email...")
round_name = predictions['round_name'].iloc[0]
competition_year = predictions['competition_year'].iloc[0]
sf.send_emails(
    "footy-tipper-email-list", 
    f"Footy Tipper Predictions for {round_name} {competition_year}", 
    reg_reagan, 
    os.getenv('MY_EMAIL'), 
    os.getenv('EMAIL_PASSWORD'), 
    json_path
)

print("Send step complete.")
