import joblib
import pandas as pd
import difflib
import psycopg2

# ==========================================
# 1. LOAD THE BRAINS & STATS
# ==========================================
DB_URL = "postgresql://neondb_owner:npg_fx3jXEOrYd4a@ep-jolly-sunset-a4ktuss0-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

print("Loading Bracket Brain & Current Season Stats...")
try:
    classifier = joblib.load('ncaa_bracket_classifier.pkl')
    feats = joblib.load('ncaa_bracket_features.pkl')
    elo = joblib.load('ncaa_elo_state.pkl')
    units = pd.read_csv('ncaa_unit_stats.csv')
    db_teams = units['Team'].dropna().unique().tolist()
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

def match_team(name):
    if name in db_teams: return name
    match = difflib.get_close_matches(name, db_teams, n=1, cutoff=0.4)
    return match[0] if match else name

# ==========================================
# 2. THE 2026 BRACKET (TEST DATA)
# Teams in exact seed order: 1 through 16.
# ==========================================
EAST = [
    "Duke", "UCONN", "Michigan St", "Kansas", 
    "St John's","Louisvile", "UCLA", "Ohio State", 
    "TCU", "UCF", "South Florida", "Northern Iowa", 
    "CA Baptist", "N Dakota St", "Furman", "Siena"
]

WEST = [
    "Arizona", "Purdue", "Gonzaga", "Arkansas", 
    "Wisconsin", "BYU", "Miami", "Villanova", 
    "Utah State", "Missouri", "NC State", "High Point", 
    "Hawaii", "Kennesaw St", "Queens", "Long Island"
]

SOUTH = [
    "Florida", "Houston", "Illinois", "Nebraska", 
    "Vanderbilt", "North Carolina", "Saint Mary's", "Clemson", 
    "Iowa", "Texas A&M", "VCU", "McNeese", 
    "Troy", "Penn", "Idaho", "Lehigh"
]

MIDWEST = [
    "Michigan", "Iowa State", "Virginia", "Alabama", 
    "Texas Tech", "Tenessee", "Kentucky", "Georgia", 
    "Saint Louis", "Santa Clara", "SMU", "Akron", 
    "Hofstra", "Wright St", "Tenessee St", "UMBC"
]

# Match the names to our database exactly
EAST = [match_team(t) for t in EAST]
WEST = [match_team(t) for t in WEST]
SOUTH = [match_team(t) for t in SOUTH]
MIDWEST = [match_team(t) for t in MIDWEST]

# Standard bracket matchup path (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
MATCHUP_ORDER = [0, 15, 7, 8, 4, 11, 3, 12, 5, 10, 2, 13, 6, 9, 1, 14]

# Database collection array
simulation_records = []

# ==========================================
# 3. SIMULATION LOGIC
# ==========================================
def simulate_matchup(team_a, team_b):
    # If a small-conference team isn't in our DB, the big team auto-advances
    if team_a not in db_teams: return team_b, 0.99
    if team_b not in db_teams: return team_a, 0.99

    tA_u = units[units['Team'] == team_a].iloc[0]
    tB_u = units[units['Team'] == team_b].iloc[0]

    row = pd.DataFrame([{
        'elo_diff': elo.get(team_a, 1500) - elo.get(team_b, 1500),
        'efg_mismatch': tA_u['eFG'] - tB_u['eFG'],
        '3p_mismatch': tA_u['3P_pct'] - tB_u['3P_pct'],
        'tov_mismatch': tB_u['TOV_pct'] - tA_u['TOV_pct'], 
        'orb_mismatch': tA_u['ORB_rate'] - tB_u['ORB_rate'],
        'sos_mismatch': tA_u['SOS'] - tB_u['SOS'],
        'recent_efg_mismatch': tA_u['recent_eFG'] - tB_u['recent_eFG'],
        'recent_3p_mismatch': tA_u['recent_3P_pct'] - tB_u['recent_3P_pct'],
        'recent_tov_mismatch': tB_u['recent_TOV_pct'] - tA_u['recent_TOV_pct'],
        'recent_orb_mismatch': tA_u['recent_ORB_rate'] - tB_u['recent_ORB_rate']
    }])

    # Ensure columns match training exactly
    row = row[feats]

    prob_A_wins = classifier.predict_proba(row)[0][1]
    
    if prob_A_wins >= 0.50:
        return team_a, prob_A_wins
    else:
        # Team B upset!
        return team_b, (1.0 - prob_A_wins)

def play_round(teams, round_name, round_order):
    print(f"\n{'='*40}")
    print(f" {round_name.upper()}")
    print(f"{'='*40}")
    
    # Extract the region for the database, or default to Final Four/Champ
    region = round_name.split(" - ")[1] if " - " in round_name else "Final Four / Championship"
    
    winners = []
    # Step through the array two teams at a time
    for i in range(0, len(teams), 2):
        t1 = teams[i]
        t2 = teams[i+1]
        
        winner, prob = simulate_matchup(t1, t2)
        winners.append(winner)
        
        # Format the printout nicely
        loser = t2 if winner == t1 else t1
        print(f"🏀 {winner:<18} over {loser:<18} ({prob*100:.1f}%)")
        
        # Append to our database collection list
        simulation_records.append((
            2026, region, round_name, round_order, 
            t1, t2, winner, float(prob * 100)
        ))
        
    return winners

def save_simulation_to_db():
    print("\nSaving Master Bracket to Neon database...")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        sql = """
        INSERT INTO bracket_simulations (
            tournament_year, region, round_name, round_order, 
            team_a, team_b, predicted_winner, win_probability
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tournament_year, round_name, team_a, team_b) 
        DO UPDATE SET 
            predicted_winner = EXCLUDED.predicted_winner,
            win_probability = EXCLUDED.win_probability,
            created_at = CURRENT_TIMESTAMP;
        """
        cur.executemany(sql, simulation_records)
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Successfully saved {len(simulation_records)} simulated matchups to the DB.")
    except Exception as e:
        print(f"❌ Database Error: {e}")

# ==========================================
# 4. RUN THE TOURNAMENT
# ==========================================
def run_tournament():
    # Setup initial matchups based on standard seeding paths
    def arrange_region(region):
        return [region[i] for i in MATCHUP_ORDER]

    r64_east = arrange_region(EAST)
    r64_west = arrange_region(WEST)
    r64_south = arrange_region(SOUTH)
    r64_midwest = arrange_region(MIDWEST)

    # --- ROUND OF 64 ---
    r32_east = play_round(r64_east, "Round of 64 - East", 1)
    r32_west = play_round(r64_west, "Round of 64 - West", 1)
    r32_south = play_round(r64_south, "Round of 64 - South", 1)
    r32_midwest = play_round(r64_midwest, "Round of 64 - Midwest", 1)

    # --- ROUND OF 32 ---
    s16_east = play_round(r32_east, "Round of 32 - East", 2)
    s16_west = play_round(r32_west, "Round of 32 - West", 2)
    s16_south = play_round(r32_south, "Round of 32 - South", 2)
    s16_midwest = play_round(r32_midwest, "Round of 32 - Midwest", 2)

    # --- SWEET 16 ---
    e8_east = play_round(s16_east, "Sweet 16 - East", 3)
    e8_west = play_round(s16_west, "Sweet 16 - West", 3)
    e8_south = play_round(s16_south, "Sweet 16 - South", 3)
    e8_midwest = play_round(s16_midwest, "Sweet 16 - Midwest", 3)

    # --- ELITE 8 ---
    f4_east = play_round(e8_east, "Elite 8 - East", 4)
    f4_west = play_round(e8_west, "Elite 8 - West", 4)
    f4_south = play_round(e8_south, "Elite 8 - South", 4)
    f4_midwest = play_round(e8_midwest, "Elite 8 - Midwest", 4)

    # --- FINAL FOUR ---
    # Standard Final Four pairing: East vs West, South vs Midwest
    final_four_matchups = f4_east + f4_west + f4_south + f4_midwest
    championship_matchups = play_round(final_four_matchups, "Final Four", 5)

    # --- NATIONAL CHAMPIONSHIP ---
    champion = play_round(championship_matchups, "National Championship", 6)
    
    print(f"\n🏆 2026 NATIONAL CHAMPION: {champion[0].upper()} 🏆\n")
    
    # Save everything to the database
    save_simulation_to_db()

if __name__ == "__main__":
    run_tournament()
