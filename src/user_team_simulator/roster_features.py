# src/user_team_simulator/roster_features.py
import numpy as np
import pandas as pd
import re

# Same position bucketing used in training
def pos_bucket(p: str) -> str:
    if not isinstance(p, str):
        return "UNK"
    p = p.upper()
    if p.startswith("PG"): return "PG"
    if p.startswith("SG"): return "SG"
    if p.startswith("SF"): return "SF"
    if p.startswith("PF"): return "PF"
    if p.startswith("C"):  return "C"
    return "UNK"

def _to_float_series(s):
    """Coerce series to float; also handle '0.585' and '58.5%' cases."""
    if s.dtype.kind in "biufc":
        return s.astype(float)
    s2 = s.astype(str)
    # strip percent signs if present
    s2 = s2.str.replace("%", "", regex=False)
    # to float
    out = pd.to_numeric(s2, errors="coerce")
    # if many values > 1, they were percents (like 58.5 for 0.585)
    if (out > 1.0).mean() > 0.5:
        out = out / 100.0
    return out

def aggregate_team(players_df: pd.DataFrame) -> dict:
    """
    Given a set of player rows (from the merged players_stats_with_salaries_2025_26.csv),
    produce the same team-level features used in training Part 2.
    """
    df = players_df.copy()

    # Ensure columns exist; fill as needed
    for col in ["MP","WS","VORP","BPM","PER","Age"]:
        if col in df.columns:
            df[col] = _to_float_series(df[col]).fillna(0.0)
        else:
            df[col] = 0.0

    if "TS%" in df.columns:
        df["TS%"] = _to_float_series(df["TS%"]).fillna(0.0)
    else:
        df["TS%"] = 0.0

    # Minutes weights
    total_mp = float(df["MP"].sum())
    w = (df["MP"] / total_mp).fillna(0.0) if total_mp > 0 else pd.Series(0.0, index=df.index)

    # Core aggregates (match training logic)
    sum_ws   = float(df["WS"].sum())
    sum_vorp = float(df["VORP"].sum())
    wavg_per = float((df["PER"] * w).sum())
    wavg_ts  = float((df["TS%"] * w).sum())
    wavg_age = float((df["Age"].fillna(df["Age"].median()) * w).sum())
    wavg_bpm = float((df["BPM"] * w).sum())

    top2_vorp = float(df["VORP"].nlargest(2).sum())
    top3_vorp = float(df["VORP"].nlargest(3).sum())
    top2_ws   = float(df["WS"].nlargest(2).sum())
    top3_ws   = float(df["WS"].nlargest(3).sum())

    # Position counts
    if "Pos" in df.columns:
        prim = df["Pos"].astype(str).apply(pos_bucket)
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
        "num_players": int(len(df)),
        "total_minutes": float(total_mp),
        "count_PG": int(pos_counts.get("PG", 0)),
        "count_SG": int(pos_counts.get("SG", 0)),
        "count_SF": int(pos_counts.get("SF", 0)),
        "count_PF": int(pos_counts.get("PF", 0)),
        "count_C":  int(pos_counts.get("C", 0)),
        "count_UNK":int(pos_counts.get("UNK", 0)),
    }
    return feats