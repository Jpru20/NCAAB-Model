# 🏀 XGpicks NCAAB Predictive Model & Bracket Engine

An advanced, full-stack machine learning pipeline designed to predict NCAA Men's Basketball point spreads, totals, and tournament bracket outcomes. 

The system utilizes daily scraped data, dynamic Power Ratings (Elo), Strength of Schedule (SOS), and recent momentum metrics to identify high-value betting edges against live Vegas odds.



## 🧠 System Architecture

The XGpicks engine operates on three core pillars:
1. **The Data Engine:** Scrapes historical and daily game data (via Barttorvik), parsing advanced metrics (eFG%, TOV%, ORB%, Pace).
2. **The Machine Learning Brain:** Utilizes XGBoost Regressors to project game margins and totals, and an XGBoost Classifier to predict binary straight-up winners for tournament elimination games.
3. **The Simulation & Consensus Pipeline:** Merges ML predictions with Monte Carlo simulations, compares them to live lines from The Odds API, and logs the highest-confidence picks into a cloud PostgreSQL database.

## 🗂️ Core Files

* **`predictor_reg_season.py`**
  The daily regular season workhorse. Fetches live odds, neutralizes home-court bias, runs ML models and Monte Carlo simulations, calculates betting edges, and upserts predictions into the database. Also checks for completed games from the previous 24 hours to update final scores.
* **`predictor.py`**
  The March Madness specific daily predictor. Identical to the regular season script, but injects a "Tournament Override" utilizing the Bracket Classifier to ensure daily straight-up survivor picks perfectly align with the Master Bracket logic.
* **`simulate_bracket.py`**
  The Master Bracket generator. Takes a 64-team field and simulates all 63 games of the NCAA Tournament chronologically. Uses a strictly deterministic mathematical approach (>50% win probability advances) to kill the "Mid-Major Trap" and save the Master Bracket to the database.
* **`[Your Daily Training Script].py`**
  The data compiler. Downloads up to 5 years of historical data, calculates base Elo ratings, generates dynamic 7-game Exponential Moving Averages (EMA) for momentum tracking, calculates live Strength of Schedule (SOS), and exports `ncaa_unit_stats.csv` alongside the trained `.pkl` models.

## 📊 Key Predictive Metrics

To prevent the models from falling for inflated statistics (especially from small-conference teams), the engine relies on a "Dual-Feature" statistical approach combined with rigorous schedule adjusting:

* **Season Class (Baseline):** The team's season-long averages for core metrics (`eFG`, `3P_pct`, `TOV_pct`, `ORB_rate`).
* **Recent Form (Momentum):** A 7-game Exponential Moving Average (`recent_eFG`, etc.) to weight how the team is currently playing.
* **Elo Differential (`elo_diff`):** The primary indicator of overall team strength and historical performance.
* **Strength of Schedule (`SOS`):** The average Elo rating of a team's opponents. Used explicitly by the Bracket Classifier to sniff out fraudulent shooting metrics built against weak defenses.

## 🛠️ Tech Stack & Integrations

* **Machine Learning:** `xgboost`, `scikit-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Live Odds & Scores:** [The Odds API](https://the-odds-api.com/)
* **Database:** Neon Serverless PostgreSQL (`psycopg2`)

## ⚙️ Database Schema

Predictions and bracket paths are stored in a Neon PostgreSQL database with the following key tables:
* `predictions_2.0`: Stores daily matchups, predicted spreads/totals, Vegas consensus lines, calculated betting edges, and final game scores.
* `bracket_simulations`: Stores the chronological step-by-step results of `simulate_bracket.py`, mapping out a team's path to the championship. Used by the AI Co-Pilot to generate narrative-driven betting analyses.
