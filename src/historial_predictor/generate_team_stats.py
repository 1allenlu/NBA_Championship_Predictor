# src/generate_team_stats.py

from data_collection import collect_all_team_stats
import pandas as pd

if __name__ == "__main__":
    df = collect_all_team_stats()
    df.to_csv("data/raw/team_stats.csv", index=False)
    print("✅ Saved team stats to data/raw/team_stats.csv")
    
