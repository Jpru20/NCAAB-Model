import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD KAGGLE DATA
# ==========================================
def load_data():
    print("Loading Kaggle Datasets...")
    try:
        reg_season = pd.read_csv("MRegularSeasonDetailedResults.csv")
        tourney = pd.read_csv("MNCAATourneyCompactResults.csv")
        return reg_season, tourney
    except FileNotFoundError:
        print("ERROR: Missing Kaggle files. Please ensure 'MRegularSeasonDetailedResults.csv'")
        print("and 'MNCAATourneyCompactResults.csv' are in the same folder.")
        return None, None

# ==========================================
# 2. FLATTEN BOX SCORES & CALCULATE STATS
# ==========================================
def process_box_scores(df):
    print("Flattening Box Scores and Calculating Advanced Metrics...")
    
    # ADDED 'LTeamID' and 'WTeamID' as 'OppID' to track who they played
    w_stats = df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTA', 'WOR', 'WDR', 'WTO', 'LDR', 'LTeamID']].copy()
    w_stats.columns = ['Season', 'DayNum', 'TeamID', 'Pts', 'OppPts', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTA', 'OR', 'DR', 'TO', 'OppDR', 'OppID']
    w_stats['Won'] = 1

    l_stats = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTA', 'LOR', 'LDR', 'LTO', 'WDR', 'WTeamID']].copy()
    l_stats.columns = ['Season', 'DayNum', 'TeamID', 'Pts', 'OppPts', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTA', 'OR', 'DR', 'TO', 'OppDR', 'OppID']
    l_stats['Won'] = 0

    games = pd.concat([w_stats, l_stats]).sort_values(by=['Season', 'DayNum']).reset_index(drop=True)

    # Prevent division by zero
    games['FGA'] = games['FGA'].replace(0, 1)
    games['FGA3'] = games['FGA3'].replace(0, 1)
    
    # Calculate Advanced Metrics (Matching your daily predictor)
    games['POSS'] = games['FGA'] + 0.44 * games['FTA'] + games['TO'] - games['OR']
    games['eFG'] = (games['FGM'] + 0.5 * games['FGM3']) / games['FGA']
    games['3P_pct'] = games['FGM3'] / games['FGA3']
    games['TOV_pct'] = games['TO'] / games['POSS']
    games['ORB_rate'] = games['OR'] / (games['OR'] + games['OppDR'])

    return games

# ==========================================
# 3. CALCULATE ELO RATINGS
# ==========================================
def calculate_historical_elo(reg_season_df):
    print("Running Historical Elo Engine (2003 - Present)...")
    
    elos = {}
    elo_records = []
    
    # We reset Elo to 1500 at the start of every season to prevent ratings from inflating over decades
    for season in reg_season_df['Season'].unique():
        season_games = reg_season_df[reg_season_df['Season'] == season].sort_values(by='DayNum')
        season_elos = {}
        
        for _, row in season_games.iterrows():
            w_id, l_id = row['WTeamID'], row['LTeamID']
            w_elo = season_elos.get(w_id, 1500)
            l_elo = season_elos.get(l_id, 1500)
            
            # Simple Elo calculation matching your predictor framework
            margin = row['WScore'] - row['LScore']
            dr = w_elo - l_elo + 100 # Add 100 for basic Home Court Advantage in regular season
            e_win = 1 / (1 + 10 ** (-dr / 400))
            
            mult = np.log(abs(margin) + 1)
            shift = 20 * mult * (1 - e_win)
            
            season_elos[w_id] = w_elo + shift
            season_elos[l_id] = l_elo - shift
            
        # Store the final pre-tournament Elo for every team in this season
        for team, elo in season_elos.items():
            elo_records.append({'Season': season, 'TeamID': team, 'elo': elo})
            
    return pd.DataFrame(elo_records)

# ==========================================
# 4. CALCULATE SELECTION SUNDAY SNAPSHOTS
# ==========================================
def build_snapshots(games, elo_df):
    print("Calculating Season Baselines, 7-Game EMAs, and Strength of Schedule...")
    
    target_cols = ['eFG', '3P_pct', 'TOV_pct', 'ORB_rate']
    
    season_avg = games.groupby(['Season', 'TeamID'])[target_cols].mean().reset_index()
    
    recent_ema = games.groupby(['Season', 'TeamID'])[target_cols].apply(
        lambda x: x.ewm(span=7, min_periods=1).mean().tail(1)
    ).reset_index(level=[0, 1])
    recent_ema.columns = ['Season', 'TeamID', 'recent_eFG', 'recent_3P_pct', 'recent_TOV_pct', 'recent_ORB_rate']
    
    # --- NEW: CALCULATE STRENGTH OF SCHEDULE ---
    # Merge every game with the opponent's final Elo rating, then average it
    games_with_opp = games.merge(elo_df.rename(columns={'TeamID': 'OppID', 'elo': 'Opp_Elo'}), on=['Season', 'OppID'])
    sos_df = games_with_opp.groupby(['Season', 'TeamID'])['Opp_Elo'].mean().reset_index()
    sos_df.rename(columns={'Opp_Elo': 'SOS'}, inplace=True)
    
    snapshots = pd.merge(season_avg, recent_ema, on=['Season', 'TeamID'])
    snapshots = pd.merge(snapshots, sos_df, on=['Season', 'TeamID'])
    return snapshots

# ==========================================
# 5. MERGE WITH TOURNAMENT RESULTS
# ==========================================
def build_training_dataset(tourney, snapshots, elo_df):
    print("Merging Snapshots with Actual Tournament Results...")
    
    # Drop play-in games (DayNum < 136) to only train on the main bracket
    tourney = tourney[tourney['DayNum'] >= 136].copy()
    
    # Randomly swap Team A and Team B so "Team_A" isn't always the winner
    # This naturally balances the 1s and 0s in the dataset
    np.random.seed(42)
    swap_mask = np.random.rand(len(tourney)) > 0.5
    
    tourney['Team_A'] = np.where(swap_mask, tourney['LTeamID'], tourney['WTeamID'])
    tourney['Team_B'] = np.where(swap_mask, tourney['WTeamID'], tourney['LTeamID'])
    tourney['Team_A_Won'] = np.where(swap_mask, 0, 1)
    
    master = tourney[['Season', 'Team_A', 'Team_B', 'Team_A_Won']].copy()
    
    # Join Team A Stats
    master = master.merge(snapshots, left_on=['Season', 'Team_A'], right_on=['Season', 'TeamID']).drop(columns=['TeamID'])
    master = master.merge(elo_df, left_on=['Season', 'Team_A'], right_on=['Season', 'TeamID']).rename(columns={'elo': 'Team_A_elo'}).drop(columns=['TeamID'])
    
    # Rename Team A Columns
    master.rename(columns={
        'eFG': 'Team_A_eFG', '3P_pct': 'Team_A_3P_pct', 'TOV_pct': 'Team_A_TOV_pct', 'ORB_rate': 'Team_A_ORB_rate',
        'recent_eFG': 'Team_A_recent_eFG', 'recent_3P_pct': 'Team_A_recent_3P_pct', 'recent_TOV_pct': 'Team_A_recent_TOV_pct', 'recent_ORB_rate': 'Team_A_recent_ORB_rate',
        'SOS': 'Team_A_SOS'
    }, inplace=True)
    
    # Join Team B Stats
    master = master.merge(snapshots, left_on=['Season', 'Team_B'], right_on=['Season', 'TeamID']).drop(columns=['TeamID'])
    master = master.merge(elo_df, left_on=['Season', 'Team_B'], right_on=['Season', 'TeamID']).rename(columns={'elo': 'Team_B_elo'}).drop(columns=['TeamID'])
    
    # Rename Team B Columns
    master.rename(columns={
        'eFG': 'Team_B_eFG', '3P_pct': 'Team_B_3P_pct', 'TOV_pct': 'Team_B_TOV_pct', 'ORB_rate': 'Team_B_ORB_rate',
        'recent_eFG': 'Team_B_recent_eFG', 'recent_3P_pct': 'Team_B_recent_3P_pct', 'recent_TOV_pct': 'Team_B_recent_TOV_pct', 'recent_ORB_rate': 'Team_B_recent_ORB_rate',
        'SOS': 'Team_B_SOS'
    }, inplace=True)
    
    master.to_csv("historical_tourney_data.csv", index=False)
    print(f"✅ Success! Saved {len(master)} historical tournament matchups to 'historical_tourney_data.csv'.")

if __name__ == "__main__":
    reg_season, tourney = load_data()
    if reg_season is not None:
        games = process_box_scores(reg_season)
        elo_df = calculate_historical_elo(reg_season)
        snapshots = build_snapshots(games, elo_df)
        build_training_dataset(tourney, snapshots, elo_df)
