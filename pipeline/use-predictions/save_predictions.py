import os
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def save_predictions(predictions, prod_run=False):
    if prod_run:
        # OAuth 2.0 setup
        creds = None
        scopes = ['https://www.googleapis.com/auth/drive']

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('path_to_your_credentials.json', scopes)
            creds = flow.run_local_server(port=0)
        
        # Build the Google Drive service
        drive_service = build('drive', 'v3', credentials=creds)

        predictions.to_csv('predictions.csv', index=False)
        
        # Upload the file
        file_metadata = {
            'name': f"round{predictions['round_id'].unique()[0]}_{predictions['competition_year'].unique()[0]}.csv",
            'parents': ['your_folder_id']
        }
        media = MediaFileUpload('predictions.csv', mimetype='text/csv')
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()


def tipper_picks(predictions, prod_run=False):
    predictions['home_odds_thresh'] = 1 / predictions['home_team_win_prob']
    predictions['away_odds_thresh'] = 1 / predictions['home_team_lose_prob']
    
    home_picks = predictions[predictions['home_team_result'] == 'Win'][['team_home', 'team_head_to_head_odds_home', 'home_odds_thresh']].copy()
    home_picks.rename(columns={'team_home': 'team', 'team_head_to_head_odds_home': 'price', 'home_odds_thresh': 'price_min'}, inplace=True)
    
    away_picks = predictions[predictions['home_team_result'] == 'Loss'][['team_away', 'team_head_to_head_odds_away', 'away_odds_thresh']].copy()
    away_picks.rename(columns={'team_away': 'team', 'team_head_to_head_odds_away': 'price', 'away_odds_thresh': 'price_min'}, inplace=True)
    
    tipper_picks = pd.concat([home_picks, away_picks])
    tipper_picks = tipper_picks[tipper_picks['price'] > (tipper_picks['price_min'] * 1.15)]
    
    if prod_run:
        # OAuth 2.0 setup
        creds = None
        scopes = ['https://www.googleapis.com/auth/drive']

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('path_to_your_credentials.json', scopes)
            creds = flow.run_local_server(port=0)
        
        # Build the Google Drive service
        drive_service = build('drive', 'v3', credentials=creds)

        tipper_picks.to_csv('tipper_picks.csv', index=False)
        
        # Upload the file
        file_metadata = {
            'name': f"tipper_picks_round{predictions['round_id'].unique()[0]}_{predictions['competition_year'].unique()[0]}.csv",
            'parents': ['your_folder_id']
        }
        media = MediaFileUpload('tipper_picks.csv', mimetype='text/csv')
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        return tipper_picks
