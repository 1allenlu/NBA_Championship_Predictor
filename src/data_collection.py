from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import time

#team stats scraper
def get_team_stats_for_season(season: str) -> pd.DataFrame:
    """
    Fetch team stats for a given NBA season. Season format: '2023-24'
    """
    print(f"Fetching season: {season}")
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    
    df = stats.get_data_frames()[0]
    df['season'] = season
    return df

def collect_all_team_stats(start_year=2000, end_year=2024) -> pd.DataFrame:
    all_seasons = []

    for year in range(start_year, end_year + 1):
        season_str = f"{year}-{str(year+1)[-2:]}"
        try:
            df = get_team_stats_for_season(season_str)
            all_seasons.append(df)
            time.sleep(1.2)  # avoid rate limiting
        except Exception as e:
            print(f"Error fetching {season_str}: {e}")

    full_df = pd.concat(all_seasons, ignore_index=True)
    return full_df

#MVP Scrapers
def find_mvp_table(tables):
    for table in tables:
        cols = table.columns

        # Case 1: MultiIndex (nested headers)
        if isinstance(cols, pd.MultiIndex):
            flat_cols = ['_'.join(c).strip() for c in cols]
            if any('Share' in c for c in flat_cols) and any('Tm' in c for c in flat_cols):
                table.columns = flat_cols
                return table

        # Case 2: Flat headers
        elif isinstance(cols, pd.Index):
            if 'Share' in cols and 'Tm' in cols and 'Player' in cols:
                return table

    return None

def scrape_mvp_vote_share(start_year=2000, end_year=2024):
    import pandas as pd
    import time

    all_mvps = []

    for year in range(start_year, end_year + 1):
        url = f"https://www.basketball-reference.com/awards/awards_{year}.html"
        print(f"Scraping {url}")

        try:
            tables = pd.read_html(url)
            mvp_table = find_mvp_table(tables)

            if mvp_table is None:
                print(f"⚠️ Couldn't find MVP table for {year}")
                continue

            # Identify correct column names dynamically
            team_col = None
            share_col = None
            player_col = None

            for col in mvp_table.columns:
                if 'Tm' in col:
                    team_col = col
                if 'Share' in col:
                    share_col = col
                if 'Player' in col:
                    player_col = col

            if not team_col or not share_col or not player_col:
                raise ValueError("Required columns not found")

            cleaned_table = mvp_table[[player_col, team_col, share_col]].copy()
            cleaned_table.columns = ['Player', 'Tm', 'Share']
            cleaned_table = cleaned_table[cleaned_table['Tm'].notna()]
            cleaned_table['season'] = f"{year-1}-{str(year)[-2:]}"
            all_mvps.append(cleaned_table)

            time.sleep(1.5)  # be nice to the server
        except Exception as e:
            print(f"❌ Error scraping {year}: {e}")

    if not all_mvps:
        raise ValueError("No MVP tables were found!")

    return pd.concat(all_mvps, ignore_index=True)

def build_team_vote_shares(mvp_df: pd.DataFrame) -> pd.DataFrame:
    mvp_df = mvp_df[['Player', 'Tm', 'Share', 'season']]
    mvp_df.columns = ['player', 'team', 'share', 'season']

    mvp_df['share'] = pd.to_numeric(mvp_df['share'], errors='coerce')

    team_votes = (
        mvp_df.groupby(['team', 'season'])['share']
        .sum()
        .reset_index()
        .rename(columns={'share': 'mvp_vote_share'})
    )

    return team_votes


def build_team_has_mvp(mvp_df: pd.DataFrame) -> pd.DataFrame:
    mvp_df = mvp_df[['Tm', 'season']].drop_duplicates()
    mvp_df['has_mvp'] = 1
    mvp_df.columns = ['team', 'season', 'has_mvp']
    return mvp_df

if __name__ == "__main__":
    # Scrape detailed MVP vote shares from season award pages
    mvp_votes_df = scrape_mvp_vote_share()
    team_votes = build_team_vote_shares(mvp_votes_df)
    has_mvp_df = build_team_has_mvp(mvp_votes_df)
    # Save to raw data
    combined = pd.merge(team_votes, has_mvp_df, on=["team", "season"], how="outer")
    combined['has_mvp'] = combined['has_mvp'].fillna(0).astype(int)
    combined.to_csv("data/raw/team_mvp_summary.csv", index=False)
    print("✅ Saved MVP summary to data/raw/team_mvp_summary.csv")

# --- Optional Sorting ---
df = pd.read_csv("data/raw/team_mvp_summary.csv")
df['season_start'] = df['season'].str[:4].astype(int)
df_sorted = df.sort_values(by='season_start', ascending=False).drop(columns='season_start')
df_sorted.to_csv("data/raw/team_mvp_summary_sorted.csv", index=False)
print("✅ Also saved sorted file to data/raw/team_mvp_summary_sorted.csv")