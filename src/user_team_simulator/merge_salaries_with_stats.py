# src/user_team_simulator/merge_salaries_with_stats.py

import os
import re
import unicodedata
import numpy as np
import pandas as pd

# ---- Paths ----
STATS_CSV   = "data/raw/players_advanced_stats_2024_25.csv"
SALARIES_CSV= "data/raw/players_salaries_2025_26.csv"
OUTPUT_CSV  = "data/processed/players_stats_with_salaries_2025_26.csv"

# --- Salary constants (adjust as needed) ---
VET_MIN_2025_26 = 1_200_000
TWO_WAY_2025_26 = 600_000

def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[\.\-']", "", s)       # remove punctuation like periods/apostrophes
    s = re.sub(r"\s+", " ", s).strip()  # normalize spaces
    return s

def impute_missing_salaries(df: pd.DataFrame, strategy="heuristic") -> pd.DataFrame:
    """
    strategy: "drop" | "min" | "heuristic" | "percentile"
    Adds a boolean column 'salary_imputed' for tracking.
    """
    out = df.copy()
    out["salary_imputed"] = False
    miss = out["salary_2025_26"].isna()

    if strategy == "drop":
        return out[~miss].reset_index(drop=True)

    if strategy == "min":
        out.loc[miss, "salary_2025_26"] = VET_MIN_2025_26
        out.loc[miss, "salary_imputed"] = True
        return out.reset_index(drop=True)

    if strategy == "heuristic":
        low_usage = (out["G"].fillna(0) < 10) | (out["MP"].fillna(0) < 200)
        mask_two_way = miss & low_usage
        out.loc[mask_two_way, "salary_2025_26"] = TWO_WAY_2025_26
        out.loc[mask_two_way, "salary_imputed"] = True

        mask_vet = miss & ~low_usage
        out.loc[mask_vet, "salary_2025_26"] = VET_MIN_2025_26
        out.loc[mask_vet, "salary_imputed"] = True
        return out.reset_index(drop=True)

    if strategy == "percentile":
        p10 = np.nanpercentile(out["salary_2025_26"].dropna(), 10) if out["salary_2025_26"].notna().any() else VET_MIN_2025_26
        out.loc[miss, "salary_2025_26"] = p10
        out.loc[miss, "salary_imputed"] = True
        return out.reset_index(drop=True)

    # default -> min
    out.loc[miss, "salary_2025_26"] = VET_MIN_2025_26
    out.loc[miss, "salary_imputed"] = True
    return out.reset_index(drop=True)

def read_salaries(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Find the bbref id column (often '-9999')
    id_col = None
    for c in df.columns[::-1]:
        if re.fullmatch(r"-?\d+", str(c)) or c.lower() in ["id", "bbref_id"]:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[-1]

    df = df.rename(columns={id_col: "bbref_id"})

    # Keep what we need
    keep = ["Player", "bbref_id"]
    if "2025-26" not in df.columns:
        raise ValueError("Could not find '2025-26' column in salaries CSV.")
    keep.append("2025-26")
    if "Tm" in df.columns:
        keep.append("Tm")

    df = df[keep].copy()

    # Clean salary -> float
    df["salary_2025_26"] = (
        df["2025-26"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace({"": None})
        .astype(float)
    )
    df.drop(columns=["2025-26"], inplace=True)

    df["bbref_id"] = df["bbref_id"].astype(str).str.strip()
    df["Player_clean"] = df["Player"].apply(normalize_name)
    return df

def read_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {}
    if "Player-additional" in df.columns:
        rename_map["Player-additional"] = "bbref_id"
    if "Team" in df.columns and "Tm" not in df.columns:
        rename_map["Team"] = "Tm"
    df = df.rename(columns=rename_map)

    if "Player" not in df.columns:
        raise ValueError("Stats CSV is missing 'Player' column.")
    if "bbref_id" not in df.columns:
        df["bbref_id"] = None

    df["Player_clean"] = df["Player"].apply(normalize_name)

    # Prefer TOT row if present
    if "Tm" in df.columns:
        df["is_tot"] = (df["Tm"].astype(str) == "TOT").astype(int)
        df = (
            df.sort_values(["Player_clean", "is_tot"], ascending=[True, False])
              .drop_duplicates(subset=["Player_clean"], keep="first")
              .drop(columns=["is_tot"])
        )
    else:
        df = df.drop_duplicates(subset=["Player_clean"], keep="first")

    df["bbref_id"] = df["bbref_id"].astype(str).str.strip()
    return df

def merge_stats_and_salaries(stats_path: str, salaries_path: str, out_path: str):
    stats_df = read_stats(stats_path)
    salaries_df = read_salaries(salaries_path)

    # 1) Merge on bbref_id
    merged = stats_df.merge(
        salaries_df[["bbref_id", "salary_2025_26"]],
        on="bbref_id",
        how="left"
    )

    # 2) Fallback: name-based merge for still-missing rows
    missing_mask = merged["salary_2025_26"].isna()
    if missing_mask.any():
        left_missing = merged.loc[missing_mask, ["Player_clean"]].copy()
        fallback = left_missing.merge(
            salaries_df[["Player_clean", "salary_2025_26"]],
            on="Player_clean",
            how="left"
        )
        merged.loc[missing_mask, "salary_2025_26"] = fallback["salary_2025_26"].values

    # Diagnostics (pre-imputation)
    total = len(merged)
    matched = merged["salary_2025_26"].notna().sum()
    print(f"💾 Total players in stats: {total}")
    print(f"✅ Matched salaries:       {matched}")
    print(f"❓ Unmatched salaries:     {total - matched}")
    if total - matched > 0:
        print("\nExamples of unmatched (first 10):")
        print(merged[merged["salary_2025_26"].isna()][["Player", "bbref_id"]].head(10).to_string(index=False))

    # 3) Impute any remaining missing salaries
    merged = impute_missing_salaries(merged, strategy="heuristic")

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n✅ Saved merged file to: {out_path}")
    print("🧮 Imputed salaries:", merged["salary_imputed"].sum())

if __name__ == "__main__":
    merge_stats_and_salaries(STATS_CSV, SALARIES_CSV, OUTPUT_CSV)