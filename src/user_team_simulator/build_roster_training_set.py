# src/user_team_simulator/build_roster_training_set.py

import os
import re
import json
import numpy as np
import pandas as pd
from glob import glob

# ---------- Config ----------
RAW_FOLDER = "data/raw/players_advanced_by_season"   # put per-season CSVs here
OUT_CSV    = "data/processed/roster_training_set.csv"

# Seasons to include (must match files you’ve downloaded)
# File naming assumed: players_advanced_stats_2011_12.csv (use your pattern)
SEASONS = [
    "2011-12","2012-13","2013-14","2014-15","2015-16",
    "2016-17","2017-18","2018-19","2019-20","2020-21",
    "2021-22","2022-23","2023-24"
]

# Champions mapping (season -> full TEAM_NAME used in your labels)
CHAMPIONS = {
    "1999-00": "Los Angeles Lakers",
    "2000-01": "Los Angeles Lakers",
    "2001-02": "Los Angeles Lakers",
    "2002-03": "San Antonio Spurs",
    "2003-04": "Detroit Pistons",
    "2004-05": "San Antonio Spurs",
    "2005-06": "Miami Heat",
    "2006-07": "San Antonio Spurs",
    "2007-08": "Boston Celtics",
    "2008-09": "Los Angeles Lakers",
    "2009-10": "Los Angeles Lakers",
    "2010-11": "Dallas Mavericks",
    "2011-12": "Miami Heat",
    "2012-13": "Miami Heat",
    "2013-14": "San Antonio Spurs",
    "2014-15": "Golden State Warriors",
    "2015-16": "Cleveland Cavaliers",
    "2016-17": "Golden State Warriors",
    "2017-18": "Golden State Warriors",
    "2018-19": "Toronto Raptors",
    "2019-20": "Los Angeles Lakers",
    "2020-21": "Milwaukee Bucks",
    "2021-22": "Golden State Warriors",
    "2022-23": "Denver Nuggets",
    "2023-24": "Boston Celtics"
}

# BR abbreviations to full TEAM_NAME used in your dataset
TEAM_ABBREV_TO_NAME = {
    # Atlantic
    "BOS": "Boston Celtics", "BRK": "Brooklyn Nets", "NJN": "New Jersey Nets",
    "NYK": "New York Knicks", "PHI": "Philadelphia 76ers", "TOR": "Toronto Raptors",
    # Central
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DET": "Detroit Pistons",
    "IND": "Indiana Pacers", "MIL": "Milwaukee Bucks",
    # Southeast
    "ATL": "Atlanta Hawks", "CHO": "Charlotte Hornets", "CHA": "Charlotte Hornets",  # BR sometimes uses CHO
    "MIA": "Miami Heat", "ORL": "Orlando Magic", "WAS": "Washington Wizards",
    # Northwest
    "DEN": "Denver Nuggets", "MIN": "Minnesota Timberwolves", "OKC": "Oklahoma City Thunder",
    "SEA": "Seattle SuperSonics", "POR": "Portland Trail Blazers", "UTA": "Utah Jazz",
    # Pacific
    "GSW": "Golden State Warriors", "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers",
    "PHO": "Phoenix Suns", "SAC": "Sacramento Kings",
    # Southwest
    "DAL": "Dallas Mavericks", "HOU": "Houston Rockets", "MEM": "Memphis Grizzlies",
    "NOP": "New Orleans Pelicans", "NOH": "New Orleans Hornets", "SAS": "San Antonio Spurs",
    # Jersey/Tot placeholders
    "TOT": None, "2TM": None, "3TM": None, "4TM": None
}

# ---------- Helpers ----------

def season_to_filename(season: str) -> str:
    # you can change this to match your actual filenames
    # e.g. "players_advanced_stats_2011_12.csv"
    y1 = int(season.split("-")[0])
    y2 = int(season.split("-")[1])
    return os.path.join(RAW_FOLDER, f"players_advanced_stats_{y1}_{str(y2).zfill(2)}.csv")

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def pos_bucket(pos: str) -> str:
    # Take the first position token, bucket to PG/SG/SF/PF/C
    if not isinstance(pos, str) or len(pos) == 0:
        return "UNK"
    primary = pos.split("-")[0].strip().upper()
    if primary in {"PG","SG","SF","PF","C"}:
        return primary
    return "UNK"

def aggregate_team(players_df: pd.DataFrame) -> dict:
    """
    Given per-team player rows (one season, one team), build team-level features.
    We weight averages by minutes (MP). Fill missing with 0 where sensible.
    """
    df = players_df.copy()

    # Coerce numeric for core stats
    for col in ["MP", "WS", "VORP", "BPM", "PER"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- FIX for TS% handling ---
    if "TS%" in df.columns:
        if pd.api.types.is_numeric_dtype(df["TS%"]):
            if df["TS%"].max() > 1:  # e.g. 60 instead of 0.60
                df["TS%"] = df["TS%"] / 100.0
        else:
            df["TS%"] = (
                pd.to_numeric(df["TS%"].str.replace("%", "", regex=False), errors="coerce") / 100.0
            )
    # ---------------------------------

    df["MP"] = df["MP"].fillna(0)
    total_mp = df["MP"].sum()
    w = (df["MP"] / total_mp).fillna(0) if total_mp > 0 else 0

    # Core aggregates
    sum_ws   = df["WS"].fillna(0).sum() if "WS" in df.columns else np.nan
    sum_vorp = df["VORP"].fillna(0).sum() if "VORP" in df.columns else np.nan
    wavg_per = (df["PER"].fillna(0) * w).sum() if "PER" in df.columns else np.nan
    wavg_ts  = (df["TS%"].fillna(0) * w).sum() if "TS%" in df.columns else np.nan
    wavg_age = (df["Age"].fillna(df["Age"].median()).astype(float) * w).sum() if "Age" in df.columns else np.nan
    wavg_bpm = (df["BPM"].fillna(0) * w).sum() if "BPM" in df.columns else np.nan

    # top2/top3 VORP/WS
    top2_vorp = df["VORP"].fillna(0).nlargest(2).sum() if "VORP" in df.columns else np.nan
    top3_vorp = df["VORP"].fillna(0).nlargest(3).sum() if "VORP" in df.columns else np.nan
    top2_ws   = df["WS"].fillna(0).nlargest(2).sum() if "WS" in df.columns else np.nan
    top3_ws   = df["WS"].fillna(0).nlargest(3).sum() if "WS" in df.columns else np.nan

    # position counts (use primary position bucket)
    if "Pos" in df.columns:
        prim = df["Pos"].apply(pos_bucket)
        pos_counts = prim.value_counts().to_dict()
    else:
        pos_counts = {}

    feats = {
        "sum_ws": sum_ws,
        "sum_vorp": sum_vorp,
        "wavg_per": wavg_per,
        "wavg_ts": wavg_ts,
        "wavg_age": wavg_age,
        "wavg_bpm": wavg_bpm,
        "top2_vorp": top2_vorp,
        "top3_vorp": top3_vorp,
        "top2_ws": top2_ws,
        "top3_ws": top3_ws,
        "num_players": len(df),
        "total_minutes": total_mp
    }

    # add pos counts
    for p in ["PG", "SG", "SF", "PF", "C", "UNK"]:
        feats[f"count_{p}"] = pos_counts.get(p, 0)

    return feats

def load_and_clean_players_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names we rely on
    if "Team" in df.columns and "Tm" not in df.columns:
        df = df.rename(columns={"Team": "Tm"})
    if "Player-additional" in df.columns and "bbref_id" not in df.columns:
        df = df.rename(columns={"Player-additional": "bbref_id"})

    # Ensure expected columns exist
    needed = ["Player","Age","Tm","Pos","G","MP","PER","TS%","WS","BPM","VORP"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # Drop multi-team aggregate rows (TOT/2TM/3TM/4TM) for team aggregation
    df = df[~df["Tm"].isin(["TOT","2TM","3TM","4TM"])].copy()
    return df

# ---------- Main build ----------

def main():
    rows = []

    for season in SEASONS:
        path = season_to_filename(season)
        if not os.path.exists(path):
            print(f"⚠️ Missing CSV for {season}: {path}")
            continue

        pdf = load_and_clean_players_csv(path)

        # group by team abbrev
        for tm, g in pdf.groupby("Tm"):
            full_name = TEAM_ABBREV_TO_NAME.get(tm)
            if full_name is None:
                # skip invalid team tokens (TOT, 2TM, etc.)
                continue

            feats = aggregate_team(g)
            feats["TEAM_NAME"] = full_name
            feats["season"] = season
            feats["won_championship"] = int(CHAMPIONS.get(season, "") == full_name)
            rows.append(feats)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved roster training set: {OUT_CSV}")
    print("📐 Shape:", out.shape)
    print("🧪 Positives:", out["won_championship"].sum(), "/ total:", len(out))

if __name__ == "__main__":
    main()