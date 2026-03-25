import pandas as pd
import requests
import io

# Test only the current season
YEAR = 2026
URL = f"https://barttorvik.com/getgamestats.php?year={YEAR}&csv=1"

print(f"--- FETCHING DATA FOR {YEAR} ---")
try:
    # 1. Download the raw text
    response = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    raw_text = response.text
    
    print("Download successful.")
    print(f"Raw data length: {len(raw_text)} characters.")
    
    # 2. Peek at the first 5 lines of raw text
    # This helps you visually verify if there's a header row
    print("\n--- RAW FILE PREVIEW (First 5 lines) ---")
    print("\n".join(raw_text.splitlines()[:5]))
    
    # 3. Test Parsing (First 50 rows only)
    print("\n--- PANDAS PARSE TEST (First 50 rows) ---")
    
    # Define the columns we expect
    cols = ['Date', 'Team', 'Conf', 'Opponent', 'Result', 'Tm_Pts', 'Opp_Pts', 
            'Margin', 'Location', 'OT', 'PF', 'Opp_PF', 'Pace', 'ORtg', 'DRtg']
    
    # Use io.StringIO to treat the string like a file
    # nrows=50 limits it to just 50 rows
    df = pd.read_csv(io.StringIO(raw_text), header=None, names=cols, nrows=50)
    
    # 4. Check for the "HHAA..." Error
    # We look for rows where 'Tm_Pts' is NOT a number
    print("\nChecking for bad rows in sample...")
    
    # Force convert to numeric, turning errors into NaN
    df['Tm_Pts_Check'] = pd.to_numeric(df['Tm_Pts'], errors='coerce')
    
    # Filter for rows where conversion failed (NaN)
    bad_rows = df[df['Tm_Pts_Check'].isna()]
    
    if not bad_rows.empty:
        print(f"⚠️ FOUND {len(bad_rows)} BAD ROWS!")
        print("Here is what they look like (likely headers or bad alignment):")
        print(bad_rows[['Date', 'Team', 'Tm_Pts']].head())
    else:
        print("✅ No parsing errors found in the first 50 rows.")
        
    print("\n--- FINAL DATAFRAME PREVIEW ---")
    print(df.head())
    print(f"\nColumns Detected: {df.columns.tolist()}")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
