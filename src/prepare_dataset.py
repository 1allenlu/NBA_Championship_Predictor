import pandas as pd
import os

# Ensure the folder exists
os.makedirs("data/processed", exist_ok=True)

# Load team stats and MVP summary
team_stats = pd.read_csv("data/raw/team_stats.csv")
mvp_summary = pd.read_csv("data/raw/team_mvp_summary.csv")

# print("📂 Columns in team_stats.csv:", team_stats.columns.tolist())
# print("📂 Columns in team_mvp_summary.csv:", mvp_summary.columns.tolist())

# Merge on team and season
merged = pd.merge(
    team_stats,
    mvp_summary,
    left_on=["TEAM_NAME", "season"],
    right_on=["team", "season"],
    how="left"
)

# Optional cleanup: drop the duplicate team column
merged.drop(columns=["team"], inplace=True)

# Or if you prefer consistent naming:
# merged.rename(columns={"TEAM_NAME": "team"}, inplace=True)

# Fill missing MVP values
merged["mvp_vote_share"] = merged["mvp_vote_share"].fillna(0)
merged["has_mvp"] = merged["has_mvp"].fillna(0).astype(int)

# Optional: sort by season
merged = merged.sort_values(by="season", ascending=False)

# Save final merged dataset
merged.to_csv("data/processed/final_dataset.csv", index=False)
# print("✅ Merged data saved to data/processed/final_dataset.csv")