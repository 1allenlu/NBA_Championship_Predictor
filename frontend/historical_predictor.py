# frontend/historical_predictor.py
import sys
import os

# Add the root directory (NBA_Championship_Predictor) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.historial_predictor.predict_historical import predict_championship

st.set_page_config(page_title="NBA Championship Predictor", layout="centered")

# 🔵 Blue gradient styling
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to bottom right, #1e3c72, #2a5298);
            color: white;
        }
        .stButton > button {
            background-color: #004080;
            color: white;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 Historical NBA Team Championship Predictor")

# Load teams and seasons
df = pd.read_csv("data/processed/final_dataset_with_labels.csv")
available_seasons = sorted(df["season"].unique())
# available_teams = sorted(df["TEAM_NAME"].unique(),)

season = st.selectbox("Select Season:", available_seasons)
# team = st.selectbox("Select Team:", df[df["season"] == season]["TEAM_NAME"].unique())
team = st.selectbox(
    "Select Team:", 
    sorted(df[df["season"] == season]["TEAM_NAME"].unique())
)

if st.button("Predict Championship Probability"):
    with st.spinner("Computing..."):
        proba, error = predict_championship(team, season)
        if error:
            st.error(error)
        else:
            st.success(f"🏆 **{team}** in {season} had a **{proba * 100:.2f}%** chance to win the championship.")