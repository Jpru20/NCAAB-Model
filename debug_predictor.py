import joblib
import pandas as pd
import requests
import difflib
from datetime import datetime, timedelta, timezone

# CONFIGURATION
ODDS_API_KEY = "01d8be5a8046e9fd1b16a19c5f5823ae"
SPORT_KEY = "basketball_ncaab"

def get_live_odds():
    print("   [1/3] Fetching Live Odds...")
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds'
    now = datetime.now(timezone.utc)
    # Get 24h window
    upcoming_window = now + timedelta(hours=24)
    
    params = {
        'api_key': ODDS_API_KEY, 
        'regions': 'us', 
        'markets': 'h2h', # Simple Moneyline for matching test
        'bookmakers': 'betmgm',
        'commenceTimeFrom': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'commenceTimeTo': upcoming_window.strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    
    res = requests.get(url, params=params).json()
    if isinstance(res, dict) and 'message' in res:
        print(f"API Error: {res['message']}")
        return []
    return res

def run_debug():
    print("--- DIAGNOSTIC MODE ---")
    
    # 1. LOAD YOUR DATABASE TEAMS
    try:
        units = pd.read_csv('ncaa_unit_stats.csv')
        db_teams = units['Team'].unique().tolist()
        print(f"   [2/3] Loaded Database. You have stats for {len(db_teams)} teams.")
        print(f"   Sample DB Teams: {db_teams[:5]}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read ncaa_unit_stats.csv. {e}")
        return

    # 2. GET API TEAMS
    games = get_live_odds()
    print(f"   [3/3] Found {len(games)} games from API.")
    
    if len(games) == 0:
        print("   No games found. Check your API key or time window.")
        return

    print("\n" + "="*60)
    print("   MATCHING DIAGNOSTICS")
    print("="*60)
    print(f"{'API NAME':<30} | {'BEST DB MATCH':<30} | {'SCORE':<5} | {'STATUS'}")
    print("-" * 85)

    matches_found = 0

    for g in games:
        # Check both Home and Away
        for api_name in [g['home_team'], g['away_team']]:
            
            # CLEANING: Remove common suffixes that confuse matchers
            clean_api = api_name.replace("State", "St").replace(" University", "").strip()
            
            # FIND BEST MATCH
            # difflib.get_close_matches returns a list, we take top 1
            matches = difflib.get_close_matches(clean_api, db_teams, n=1, cutoff=0.0)
            
            best_match = matches[0] if matches else "NONE"
            
            # CALCULATE SCORE (0.0 to 1.0)
            score = difflib.SequenceMatcher(None, clean_api, best_match).ratio()
            
            status = "✅ OK" if score > 0.6 else "❌ FAIL"
            if score > 0.6: matches_found += 1
            
            print(f"{api_name:<30} | {best_match:<30} | {score:.2f}  | {status}")

    print("-" * 85)
    print(f"SUMMARY: Matched {matches_found} / {len(games)*2} teams.")
    print("If Status is FAIL, your cutoff (0.6) is too strict or names are different.")
    print("Example: If API says 'UConn' and DB says 'Connecticut', Score will be low.")

if __name__ == "__main__":
    run_debug()
