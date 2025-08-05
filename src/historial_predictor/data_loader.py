# src/data_loader.py
import pandas as pd

def load_nba_data():
    return pd.read_csv("data/processed/training_data.csv")
