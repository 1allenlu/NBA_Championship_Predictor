# src/recompute_features.py

import pandas as pd

def recompute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Step 1: Compute derived percentage features ---
    if "W" in df.columns and "GP" in df.columns:
        df["W_PCT"] = df["W"] / df["GP"]

    # Add other derived calculations if needed here

    return df


def add_rank_features(df: pd.DataFrame, features_to_rank: list) -> pd.DataFrame:
    df = df.copy()

    for feat in features_to_rank:
        if feat in df.columns:
            rank_col = f"{feat}_RANK"
            # Rank in descending order so better values get lower rank numbers
            df[rank_col] = df[feat].rank(method="min", ascending=False).astype(int)

    return df


def preprocess_input_team(team_df: pd.DataFrame, league_df: pd.DataFrame, features_to_rank: list) -> pd.DataFrame:
    """
    Merges a user-input team row with the full league dataset (for that season),
    recomputes derived stats and ranks it relative to the league.
    """
    # Append the new team to the league to compute ranks
    combined = pd.concat([league_df, team_df], ignore_index=True)

    # Recompute derived features and rank columns
    combined = recompute_derived_features(combined)
    combined = add_rank_features(combined, features_to_rank)

    # Return just the last row (i.e., the user’s team with ranks)
    return combined.tail(1).reset_index(drop=True)