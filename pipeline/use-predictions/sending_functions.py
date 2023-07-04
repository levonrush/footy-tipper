import os
import pandas as pd

# for google
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import gspread
from google.oauth2 import service_account

# for reg
from langchain.llms import OpenAI

# for emails
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def get_tipper_picks(predictions, prod_run=False):
    predictions['home_odds_thresh'] = 1 / predictions['home_team_win_prob']
    predictions['away_odds_thresh'] = 1 / predictions['home_team_lose_prob']
    
    home_picks = predictions[predictions['home_team_result'] == 'Win'][['team_home', 'team_head_to_head_odds_home', 'home_odds_thresh']].copy()
    home_picks.rename(columns={'team_home': 'team', 'team_head_to_head_odds_home': 'price', 'home_odds_thresh': 'price_min'}, inplace=True)
    
    away_picks = predictions[predictions['home_team_result'] == 'Loss'][['team_away', 'team_head_to_head_odds_away', 'away_odds_thresh']].copy()
    away_picks.rename(columns={'team_away': 'team', 'team_head_to_head_odds_away': 'price', 'away_odds_thresh': 'price_min'}, inplace=True)
    
    tipper_picks = pd.concat([home_picks, away_picks])
    tipper_picks = tipper_picks[tipper_picks['price'] > (tipper_picks['price_min'] * 1.15)]

    return tipper_picks

def upload_df_to_drive(df, json_path, folder_id, filename):
    # Load the credentials from the service_account.json
    creds = service_account.Credentials.from_service_account_file(json_path)

    # Build the Google Drive service
    drive_service = build('drive', 'v3', credentials=creds)

    # Save your dataframe to CSV
    df.to_csv(filename, index=False)

    # Upload the file
    file_metadata = {
        'name': f"round{df['round_id'].unique()[0]}_{df['competition_year'].unique()[0]}.csv",
        'parents': [folder_id]
    }
    media = MediaFileUpload(filename, mimetype='text/csv')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print('File ID:', file.get('id'))

    # Delete the local file after upload
    os.remove(filename)

def generate_reg_regan_email(predictions, tipper_picks, api_key, folder_url):
    # Set up the OpenAI model
    llm = OpenAI(openai_api_key=api_key,
                 model_name="gpt-3.5-turbo-16k",
                 max_tokens=15000,
                 temperature=1.1)

    # Generate input_predictions string
    input_predictions = ""
    for index, row in predictions.iterrows():
        input_predictions += f"""
            Round Name: {row['round_name']},
            Home Team Result: {row['home_team_result']},
            Home Team: {row['team_home']}, 
            Home Team Position: {row['position_home']},
            Home Team Head to Head Price: {row['team_head_to_head_odds_home']}
            Away Team: {row['team_away']},
            Away Team Position: {row['position_away']},
            Away Team Head to Head Price: {row['team_head_to_head_odds_away']}
            """
    # Generate input_picks string
    input_picks = ""
    for index, row in tipper_picks.iterrows():
        input_picks += f"""
            Team: {row['team']},
            Price: {row['price']}
            """

    # Generate the prompt
    prompt = f"""
        I have a set of predictions for NRL games in {predictions['round_name'].unique()[0]} {predictions['competition_year'].unique()[0]} made by a machine learning pipeline called the Footy Tipper: \n{input_predictions}\n 
        The description of the columns of interest is:
        
        * Home Team Result: the predicted result of the home team
        * Home Team: the home team
        * Home Team Position: the home team's position on the NRL ladder
        * Home Team Head to Head Price: the price bookies are offering for a home win
        * Away Team: the away team
        * Away Team Position: the away team's position on the NRL ladder
        * Away Team Head to Head Price: the price bookies are offering for an away win
        
        It also comes up with some good value tips for those interested in a punt in \n{input_picks}\n. The description of the columns of interest is:
        
        * Team = Team that is a good value pick
        * Price = what the bookies are offering them at
        
        Could you write up an email to my mates from Reg Regan, giving them a synopsis of the round along with the tips? 
        Accompany the tips with some smart arsed comments to about the teams playing.
        Remember to link everyone to the tips folder: {folder_url}
        Also, tell everyone to bring back the biff at the end of the email.
        Always sign off the email as Reg Regan.
        """
    # Generate the email with the OpenAI model
    reg_regan = llm.predict(prompt)

    return reg_regan

def send_emails(doc_name, subject, message, sender_email, sender_password, json_path):
    # Use your downloaded credentials file
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    creds = service_account.Credentials.from_service_account_file(json_path, scopes=scope)
    client = gspread.authorize(creds)

    # Open the test sheet and get the data
    sheet = client.open(doc_name).sheet1  # use your actual sheet name
    email_data = sheet.get_all_records()  # gets all the data inside your Google Sheet

    recipient_emails = [row['Email'] for row in email_data]  # replace 'Email' with your actual column name

    # Setup the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipient_emails)
    msg['Subject'] = subject

    # Add your message
    msg.attach(MIMEText(message, 'plain'))

    # Setup the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    # Add your credentials
    server.login(sender_email, sender_password)

    # Send the email
    text = msg.as_string()
    server.sendmail(sender_email, recipient_emails, text)

    # Close the connection
    server.quit()
