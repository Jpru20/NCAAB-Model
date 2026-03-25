import pandas as pd

print("Checking ncaa_unit_stats.csv for Strength of Schedule (SOS) data...\n")

try:
    df = pd.read_csv('ncaa_unit_stats.csv')
    
    if 'SOS' in df.columns:
        print("✅ SUCCESS: 'SOS' column is officially in the file!\n")
        
        # Sort and print the top 5 and bottom 5 to verify the math makes sense
        print("--- TOUGHEST SCHEDULES (Highest SOS) ---")
        top_sos = df[['Team', 'SOS']].sort_values(by='SOS', ascending=False).head(5)
        print(top_sos.to_string(index=False))
        
        print("\n--- EASIEST SCHEDULES (Lowest SOS) ---")
        bot_sos = df[['Team', 'SOS']].sort_values(by='SOS', ascending=True).head(5)
        print(bot_sos.to_string(index=False))
        
    else:
        print("❌ ERROR: The 'SOS' column is MISSING. The scraper didn't save it.")
        print("Columns found:", df.columns.tolist())

except Exception as e:
    print(f"Error reading the file: {e}")
