# src/predict_cli.py

import argparse
import pandas as pd
import joblib
from recompute_features import preprocess_input_team
from config import features_to_rank

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Predict championship probability for a team")
parser.add_argument("--input", type=str, required=True, help="Path to CSV file for input team")
args = parser.parse_args()

# --- Load user team data ---
user_df = pd.read_csv(args.input)

# --- Load trained model ---
model = joblib.load("models/best_model.pkl")  # Adjust if your best model has a different path

# --- Get season from input row ---
season = user_df["season"].values[0]

# --- Load league data for same season ---
league_df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
league_same_season = league_df[league_df["season"] == season].copy()

# --- Recompute features and rank user team in context of season ---
processed_df = preprocess_input_team(user_df, league_same_season, features_to_rank)

# --- Drop columns not used during training ---
# columns_to_drop = ["TEAM_ID", "TEAM_NAME", "season", "won_championship"]
# X = processed_df.drop(columns=[col for col in columns_to_drop if col in processed_df.columns])
# --- Drop unused columns before prediction ---
X = processed_df.drop(columns=["TEAM_NAME", "season", "won_championship"], errors="ignore")


# --- Add dummy TEAM_ID column to match training feature names ---
X["TEAM_ID"] = 0  # Placeholder, just for alignment

# --- Reorder columns to match model’s expected feature order ---
X = X[model.get_booster().feature_names]

# --- Predict ---
proba = model.predict_proba(X)[:, 1][0]
print(f"\n🏆 Predicted championship probability: {proba * 100:.2f}%")