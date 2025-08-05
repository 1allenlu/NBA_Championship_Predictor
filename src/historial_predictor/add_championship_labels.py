import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/final_dataset.csv")

# Championship winners dict
champions = {
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

# Initialize the column to 0
df["won_championship"] = 0

# Loop through dictionary and set correct rows to 1
for season, champion_team in champions.items():
    mask = (df["TEAM_NAME"] == champion_team) & (df["season"] == season)
    df.loc[mask, "won_championship"] = 1

# Save updated dataset
df.to_csv("data/processed/final_dataset_with_labels.csv", index=False)
print("🏆 Championship flags updated and saved to final_dataset_with_labels.csv!")