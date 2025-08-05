# src/predict_historical.py

import pandas as pd
import joblib

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/processed/final_dataset_with_labels.csv"

def load_team_features(team_name: str, season: str):
    df = pd.read_csv(DATA_PATH)
    row = df[(df["TEAM_NAME"] == team_name) & (df["season"] == season)]

    if row.empty:
        return None, "Team/season not found in dataset."

    # model_input = row.drop(columns=["TEAM_ID", "TEAM_NAME", "season", "won_championship"], errors="ignore")
    model_input = row.drop(columns=["TEAM_NAME", "season", "won_championship"], errors="ignore")
    
    return model_input, None

def predict_championship(team_name: str, season: str):
    model = joblib.load(MODEL_PATH)
    X, error = load_team_features(team_name, season)
    if error:
        return None, error

    proba = model.predict_proba(X)[:, 1][0]
    return proba, None