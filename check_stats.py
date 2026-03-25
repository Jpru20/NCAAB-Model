import pandas as pd
import sys

# Load the stats file
try:
    df = pd.read_csv('ncaa_unit_stats.csv')
    print(f"\nSUCCESS: Loaded stats for {len(df)} teams from 2026 season.")
except FileNotFoundError:
    print("Error: ncaa_unit_stats.csv not found. Run build_system.py first.")
    sys.exit()

print("Type a team name to see their raw stats used by the model.")

while True:
    search = input("\nEnter team name (or 'q' to quit): ").strip().lower()
    if search == 'q': break
    
    # Search for the team
    match = df[df['Team'].str.lower().str.contains(search)]
    
    if match.empty:
        print("No team found. Try a shorter name (e.g., 'Duke' instead of 'Duke Blue Devils').")
    else:
        print("\n--- MODEL DATA (2026 Averages) ---")
        # Display the key stats
        print(match[['Team', 'Pace', 'off_rtg', 'def_rtg', 'eFG', 'TOV_pct', 'ORB_rate']].to_string(index=False))
        print("\nKey:")
        print("Pace    = Possessions per game")
        print("off_rtg = Points scored per 100 possessions")
        print("eFG     = Effective Field Goal % (Includes 3-point value)")
        print("-------------------------------------------------")
