import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import requests
import io

# CONFIGURATION
TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026

print("--- [1/3] DOWNLOADING DATA FROM BARTTORVIK ---")

def get_torvik_data(year):
    url = f"https://barttorvik.com/getgamestats.php?year={year}&csv=1"
    try:
        s = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)
        df['Season'] = year
        return df
    except Exception as e:
        print(f"Error fetching {year}: {e}")
        return pd.DataFrame()

# Download all data
frames = [get_torvik_data(y) for y in TRAIN_SEASONS + [CURRENT_SEASON]]
raw_df = pd.concat(frames, ignore_index=True)
print(f"   > Raw rows downloaded: {len(raw_df)}")

print("--- [1.5/3] PARSING EXTENDED STATS ---")

def parse_torvik_row(row):
    try:
        date = row[0]
        team = row[2]
        opp = row[4]
        loc = row[5]
        
        result_str = str(row[6]) 
        parts = result_str.replace("W, ", "").replace("L, ", "").split("-")
        s1, s2 = int(parts[0]), int(parts[1])
        
        if "W" in result_str:
            tm_score, opp_score = max(s1, s2), min(s1, s2)
        else:
            tm_score, opp_score = min(s1, s2), max(s1, s2)
            
        margin = tm_score - opp_score
        
        # --- PARSING CORE STATS ---
        ortg = float(row[9])
        efg = float(row[10])
        tov = float(row[11])
        orb = float(row[12])
        
        try: ftr = float(row[13]) 
        except: ftr = 0.0
            
        try: two_p = float(row[15]) 
        except: two_p = 0.0
            
        try: three_p = float(row[16]) 
        except: three_p = 0.0
            
        try: three_p_rate = float(row[17]) 
        except: three_p_rate = 35.0 

        try: drtg = float(row[14])
        except: drtg = 100.0

        pace = (tm_score / ortg * 100) if ortg > 0 else 70.0
        
        return pd.Series([date, team, opp, loc, tm_score, opp_score, margin, pace, 
                          ortg, drtg, efg, tov, orb, ftr, two_p, three_p, three_p_rate])
    except:
        return pd.Series([None] * 17)

cols = ['Date', 'Team', 'Opponent', 'Location', 'Tm_Pts', 'Opp_Pts', 'Margin', 
        'Pace', 'ORtg', 'DRtg', 'eFG', 'TOV_pct', 'ORB_pct', 
        'FTR', '2P_pct', '3P_pct', '3P_rate']

parsed_df = raw_df.apply(parse_torvik_row, axis=1)
parsed_df.columns = cols

df = parsed_df.dropna(subset=['Date', 'Tm_Pts'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Season'] = raw_df['Season'].values[df.index]
df['is_home'] = df['Location'] == 'H'
df = df.sort_values('Date')

print(f"   > Successfully extracted {len(df)} games.")

# ==========================================
# ELO CALCULATION (MOVED UP FOR SOS)
# ==========================================
print("--- [1.8/3] RUNNING ELO SYSTEM ---")
elos = {}
for idx, row in df.iterrows():
    t, o, margin = row['Team'], row['Opponent'], row['Margin']
    t_elo = elos.get(t, 1500)
    o_elo = elos.get(o, 1500)
    k = 22
    prob = 1 / (1 + 10 ** (-(t_elo - o_elo) / 400))
    result = 1 if margin > 0 else 0
    change = k * (result - prob)
    elos[t] = t_elo + change
    elos[o] = o_elo - change

joblib.dump(elos, "ncaa_elo_state.pkl")

# ==========================================
# CALCULATING DUAL-FEATURE STATS
# ==========================================
print("--- [2/3] CALCULATING DUAL-FEATURE STATS (SEASON AVG + EMA + SOS) ---")

current_df = df[df['Season'] == CURRENT_SEASON].copy()
if current_df.empty:
    print("!!! CRITICAL WARNING: No 2026 data found! !!!")

# Define the stats we want to track
off_cols = ['Pace', 'ORtg', 'DRtg', 'Tm_Pts', 'eFG', 'TOV_pct', 'ORB_pct', 'FTR', '2P_pct', '3P_pct', '3P_rate']
def_cols = ['eFG', 'TOV_pct', 'ORB_pct', 'FTR', '2P_pct', '3P_pct', '3P_rate']

# --- OFFENSIVE STATS ---
off_stats_avg = current_df.groupby('Team')[off_cols].mean().reset_index()

current_df_chrono = current_df.sort_values(by=['Team', 'Date'])
off_stats_ema = current_df_chrono.groupby('Team')[off_cols].apply(
    lambda x: x.ewm(span=7, min_periods=1).mean().tail(1)
).reset_index(level=0)
off_stats_ema.columns = ['Team'] + ['recent_' + col for col in off_cols]

off_stats = pd.merge(off_stats_avg, off_stats_ema, on='Team')

# --- DEFENSIVE STATS ---
def_stats_avg = current_df.groupby('Opponent')[def_cols].mean().reset_index()

current_df_opp = current_df.sort_values(by=['Opponent', 'Date'])
def_stats_ema = current_df_opp.groupby('Opponent')[def_cols].apply(
    lambda x: x.ewm(span=7, min_periods=1).mean().tail(1)
).reset_index(level=0)
def_stats_ema.columns = ['Opponent'] + ['recent_' + col for col in def_cols]

def_stats = pd.merge(def_stats_avg, def_stats_ema, on='Opponent')

def_cols_rename = ['Def_eFG', 'Def_TOV_pct', 'Def_ORB_pct', 'Def_FTR', 'Def_2P_pct', 'Def_3P_pct', 'Def_3P_rate']
recent_def_cols_rename = ['recent_' + c for c in def_cols_rename]
def_stats.columns = ['Team'] + def_cols_rename + recent_def_cols_rename

# 3. MERGE THEM
stats = pd.merge(off_stats, def_stats, on='Team', how='left')

# ==========================================
# NEW: CALCULATE STRENGTH OF SCHEDULE (SOS)
# ==========================================
# Map the current Elo rating for every opponent. If missing, assume 1500.
current_df['Opp_Elo'] = current_df['Opponent'].map(lambda x: elos.get(x, 1500))

# Group by team and average the opponents' Elos
sos_df = current_df.groupby('Team')['Opp_Elo'].mean().reset_index()
sos_df.rename(columns={'Opp_Elo': 'SOS'}, inplace=True)

# Merge the new SOS column into our stats dataframe
stats = pd.merge(stats, sos_df, on='Team', how='left')
stats['SOS'] = stats['SOS'].fillna(1500)

# 4. DERIVED METRICS
stats['off_rtg'] = stats['ORtg']
stats['def_rtg'] = stats['DRtg']
stats['ORB_rate'] = stats['ORB_pct']
stats['DRB_rate'] = 100 - stats['Def_ORB_pct']

stats['recent_off_rtg'] = stats['recent_ORtg']
stats['recent_def_rtg'] = stats['recent_DRtg']
stats['recent_ORB_rate'] = stats['recent_ORB_pct']
stats['recent_DRB_rate'] = 100 - stats['recent_Def_ORB_pct']

stats.to_csv("ncaa_unit_stats.csv", index=False)
print(f"   > Saved ncaa_unit_stats.csv ({len(stats)} teams with SOS & Dual-Feature stats)")

print("--- [3/3] TRAINING XGBOOST MODELS (DUAL-FEATURE EXTENDED) ---")

season_stats = df.groupby(['Season', 'Team']).agg({
    'ORtg': 'mean', 'eFG': 'mean', 'TOV_pct': 'mean', 'ORB_pct': 'mean', 'Pace': 'mean',
    '3P_pct': 'mean', '2P_pct': 'mean', 'FTR': 'mean'
}).to_dict('index')

def calc_ema(x): return x.ewm(span=7, min_periods=1).mean().iloc[-1] if not x.empty else np.nan
df_chrono = df.sort_values('Date')

season_stats_ema = df_chrono.groupby(['Season', 'Team']).agg({
    'eFG': calc_ema, '3P_pct': calc_ema, 'ORB_pct': calc_ema, 'TOV_pct': calc_ema
}).to_dict('index')

def get_stat(season, team, stat, lookup):
    try: return lookup[(season, team)][stat]
    except: return None

train_rows = []
weights = []

for idx, row in df.iterrows():
    t, o, s = row['Team'], row['Opponent'], row['Season']
    if t not in elos or o not in elos: continue
    
    t_efg = get_stat(s, t, 'eFG', season_stats)
    t_3p = get_stat(s, t, '3P_pct', season_stats)
    t_orb = get_stat(s, t, 'ORB_pct', season_stats)
    t_tov = get_stat(s, t, 'TOV_pct', season_stats)
    
    o_efg = get_stat(s, o, 'eFG', season_stats)
    o_3p = get_stat(s, o, '3P_pct', season_stats)
    o_orb = get_stat(s, o, 'ORB_pct', season_stats)
    o_tov = get_stat(s, o, 'TOV_pct', season_stats)

    t_efg_ema = get_stat(s, t, 'eFG', season_stats_ema)
    t_3p_ema = get_stat(s, t, '3P_pct', season_stats_ema)
    t_orb_ema = get_stat(s, t, 'ORB_pct', season_stats_ema)
    t_tov_ema = get_stat(s, t, 'TOV_pct', season_stats_ema)
    
    o_efg_ema = get_stat(s, o, 'eFG', season_stats_ema)
    o_3p_ema = get_stat(s, o, '3P_pct', season_stats_ema)
    o_orb_ema = get_stat(s, o, 'ORB_pct', season_stats_ema)
    o_tov_ema = get_stat(s, o, 'TOV_pct', season_stats_ema)

    if t_efg is None or o_efg is None or t_efg_ema is None or o_efg_ema is None: continue
    
    if s == CURRENT_SEASON: w = 3.0
    elif s == (CURRENT_SEASON - 1): w = 1.5
    else: w = 1.0
    weights.append(w)

    feat = {
        'elo_diff': elos[t] - elos[o],
        'home_elo': elos[t],
        'away_elo': elos[o],
        
        'efg_mismatch': t_efg - o_efg,
        '3p_mismatch': t_3p - o_3p, 
        'tov_mismatch': o_tov - t_tov, 
        'orb_mismatch': t_orb - o_orb, 
        
        'recent_efg_mismatch': t_efg_ema - o_efg_ema,
        'recent_3p_mismatch': t_3p_ema - o_3p_ema,
        'recent_tov_mismatch': o_tov_ema - t_tov_ema,
        'recent_orb_mismatch': t_orb_ema - o_orb_ema,

        'margin': row['Margin'],
        'total_pts': row['Tm_Pts'] + row['Opp_Pts']
    }
    train_rows.append(feat)

train_df = pd.DataFrame(train_rows)
features = ['elo_diff', 'home_elo', 'away_elo', 
            'efg_mismatch', '3p_mismatch', 'tov_mismatch', 'orb_mismatch',
            'recent_efg_mismatch', 'recent_3p_mismatch', 'recent_tov_mismatch', 'recent_orb_mismatch']

X = train_df[features]
y_spread = train_df['margin']
y_total = train_df['total_pts']

model_s = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
model_s.fit(X, y_spread, sample_weight=weights)
joblib.dump(model_s, "ncaa_model_spread.pkl")

model_t = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
model_t.fit(X, y_total, sample_weight=weights)
joblib.dump(model_t, "ncaa_model_total.pkl")

joblib.dump(features, "ncaa_features.pkl")
print("\nSYSTEM BUILD COMPLETE (With Dual-Feature Stats & SOS)")
