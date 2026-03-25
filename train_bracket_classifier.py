import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# ==========================================
# 1. LOAD HISTORICAL TOURNAMENT DATA
# ==========================================
def load_historical_data():
    print("Loading Historical Tournament Snapshots...")
    # NOTE: You will need a CSV of historical tournament games merged with 
    # their Selection Sunday stats. 
    # Columns needed: Team_A, Team_B, Team_A_Won (1 or 0), plus all your unit stats
    try:
        # Example filename - you will need to compile this from Barttorvik or Kaggle
        df = pd.read_csv("historical_tourney_data.csv") 
        print(f"Loaded {len(df)} historical tournament matchups.")
        return df
    except FileNotFoundError:
        print("ERROR: 'historical_tourney_data.csv' not found.")
        print("Please compile a dataset of past tournament games and their stats.")
        return None

# ==========================================
# 2. FEATURE ENGINEERING (THE MIRROR METHOD)
# ==========================================
def prepare_training_data(df):
    print("Engineering Neutral-Court Features (Mirroring Data)...")
    
    features_list = []
    targets = []

    for _, row in df.iterrows():
        # --- PASS 1: Winner on the Left (Target = 1) ---
        # Assume Team A is the winner for this pass, or use the actual outcome
        team_a_won = row['Team_A_Won'] 
        
        features_list.append({
            'elo_diff': row['Team_A_elo'] - row['Team_B_elo'],
            'efg_mismatch': row['Team_A_eFG'] - row['Team_B_eFG'],
            '3p_mismatch': row['Team_A_3P_pct'] - row['Team_B_3P_pct'],
            'tov_mismatch': row['Team_B_TOV_pct'] - row['Team_A_TOV_pct'], # Note: inverted because less TOV is better
            'orb_mismatch': row['Team_A_ORB_rate'] - row['Team_B_ORB_rate'],
            'sos_mismatch': row['Team_A_SOS'] - row['Team_B_SOS'],
            'recent_efg_mismatch': row['Team_A_recent_eFG'] - row['Team_B_recent_eFG'],
            'recent_3p_mismatch': row['Team_A_recent_3P_pct'] - row['Team_B_recent_3P_pct'],
            'recent_tov_mismatch': row['Team_B_recent_TOV_pct'] - row['Team_A_recent_TOV_pct'],
            'recent_orb_mismatch': row['Team_A_recent_ORB_rate'] - row['Team_B_recent_ORB_rate']
        })
        targets.append(team_a_won)

        # --- PASS 2: Winner on the Right (Target = 0) ---
        features_list.append({
            'elo_diff': row['Team_B_elo'] - row['Team_A_elo'],
            'efg_mismatch': row['Team_B_eFG'] - row['Team_A_eFG'],
            '3p_mismatch': row['Team_B_3P_pct'] - row['Team_A_3P_pct'],
            'tov_mismatch': row['Team_A_TOV_pct'] - row['Team_B_TOV_pct'],
            'orb_mismatch': row['Team_B_ORB_rate'] - row['Team_A_ORB_rate'],
            'sos_mismatch': row['Team_B_SOS'] - row['Team_A_SOS'],
            'recent_efg_mismatch': row['Team_B_recent_eFG'] - row['Team_A_recent_eFG'],
            'recent_3p_mismatch': row['Team_B_recent_3P_pct'] - row['Team_A_recent_3P_pct'],
            'recent_tov_mismatch': row['Team_A_recent_TOV_pct'] - row['Team_B_recent_TOV_pct'],
            'recent_orb_mismatch': row['Team_B_recent_ORB_rate'] - row['Team_A_recent_ORB_rate']
        })
        targets.append(1 - team_a_won) # Flips 1 to 0, and 0 to 1

    X = pd.DataFrame(features_list)
    y = np.array(targets)
    
    print(f"Generated {len(X)} perfectly balanced training rows.")
    return X, y

# ==========================================
# 3. TRAIN THE XGBOOST CLASSIFIER
# ==========================================
def train_model(X, y):
    print("Training Win Probability Classifier...")
    
    # Split into training and testing sets to verify accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Notice: We are using XGBClassifier instead of XGBRegressor!
    # objective='binary:logistic' tells it to output a percentage probability (0.0 to 1.0)
    classifier = xgb.XGBClassifier(
        n_estimators=1000, 
        max_depth=4, 
        learning_rate=0.01,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    classifier.fit(X_train, y_train)
    
    # Test the Brain
    preds = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {acc*100:.2f}% (Predicting the straight-up winner)")
    print(f"Log Loss: {ll:.4f} (Lower is better)")
    
    # Save the new Brain and its required features
    joblib.dump(classifier, 'ncaa_bracket_classifier.pkl')
    joblib.dump(X.columns.tolist(), 'ncaa_bracket_features.pkl')
    print("\n✅ Saved 'ncaa_bracket_classifier.pkl'. Ready for March Madness.")

    # Print Feature Importance
    print("\n--- WHAT MATTERS IN MARCH? (Brain Priority) ---")
    imp = classifier.get_booster().get_score(importance_type='gain')
    total_imp = sum(imp.values())
    for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<20}: {(v/total_imp)*100:.1f}%")

if __name__ == "__main__":
    df = load_historical_data()
    if df is not None:
        X, y = prepare_training_data(df)
        train_model(X, y)
