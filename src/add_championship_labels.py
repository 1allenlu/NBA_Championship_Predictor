import pandas as pd

# Load your processed dataset
df = pd.read_csv("data/processed/final_dataset.csv")

# List of (team, season) that won the championship
champions = [
    ("LAL", "1999-00"), ("LAL", "2000-01"), ("LAL", "2001-02"), ("SAS", "2002-03"),
    ("DET", "2003-04"), ("SAS", "2004-05"), ("MIA", "2005-06"), ("SAS", "2006-07"),
    ("BOS", "2007-08"), ("LAL", "2008-09"), ("LAL", "2009-10"), ("DAL", "2010-11"),
    ("MIA", "2011-12"), ("MIA", "2012-13"), ("SAS", "2013-14"), ("GSW", "2014-15"),
    ("CLE", "2015-16"), ("GSW", "2016-17"), ("GSW", "2017-18"), ("TOR", "2018-19"),
    ("LAL", "2019-20"), ("MIL", "2020-21"), ("GSW", "2021-22"), ("DEN", "2022-23"),
    ("BOS", "2023-24")
]

# Turn into DataFrame for merging
champ_df = pd.DataFrame(champions, columns=["TEAM_NAME", "season"])
champ_df["won_championship"] = 1

# Merge with main dataset
df = df.merge(champ_df, on=["TEAM_NAME", "season"], how="left")

# Fill missing values with 0 (i.e. didn't win)
df["won_championship"] = df["won_championship"].fillna(0).astype(int)

# Save updated dataset
df.to_csv("data/processed/final_dataset_with_labels.csv", index=False)

print("✅ Championship labels added! File saved to data/processed/final_dataset_with_labels.csv")