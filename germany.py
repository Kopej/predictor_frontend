from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
bundesliga_df = pd.read_csv("Bundesliga/Bundesliga_season_2010_2025.csv")

# Map API team names to CSV equivalents
TEAM_NAME_MAP = {
    "eintracht frankfurt": "ein frankfurt",
    "borussia monchengladbach": "m'gladbach",
    "bayer leverkusen": "leverkusen",
    "1. fc heidenheim": "heidenheim",
    "fc st. pauli": "st pauli",
    "borussia dortmund": "dortmund",
    "vfl wolfsburg": "wolfsburg",
    "tsg hoffenheim": "hoffenheim",
    "vfb stuttgart": "stuttgart",
    "sc freiburg": "freiburg",
    "hamburger sv": "hamburg",
    "1. fc köln": "fc koln",
    "fsv mainz 05": "mainz"
}

def normalize_team_name(name):
    return TEAM_NAME_MAP.get(name.strip().lower(), name.strip().lower())

# Train model
def train_model():
    df = bundesliga_df.dropna()
    features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
    X = df[features]
    y = df['FTR']
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Prediction endpoint
@app.post("/predict-bundesliga")
async def predict_bundesliga_match(request: Request):
    data = await request.json()
    home_raw = data.get("home_team", "")
    away_raw = data.get("away_team", "")
    home = normalize_team_name(home_raw)
    away = normalize_team_name(away_raw)

    df = bundesliga_df.copy()

    # Get recent matches
    home_recent = df[(df['HomeTeam'].str.lower() == home) | (df['AwayTeam'].str.lower() == home)].sort_values(by='Date', ascending=False).head(60)
    away_recent = df[(df['HomeTeam'].str.lower() == away) | (df['AwayTeam'].str.lower() == away)].sort_values(by='Date', ascending=False).head(60)

    if home_recent.empty or away_recent.empty:
        return {
            "prediction": "No prediction",
            "stats": [],
            "discussion": "Insufficient recent match data for one or both teams."
        }

    # Feature averages
    features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
    avg_home = home_recent[features].mean()
    avg_away = away_recent[features].mean()
    input_features = np.mean([avg_home.values, avg_away.values], axis=0).reshape(1, -1)
    raw_prediction = model.predict(input_features)[0]

    # Convert to full text
    prediction_map = {
        'H': 'Home Win',
        'D': 'Draw',
        'A': 'Away Win'
    }
    prediction = prediction_map.get(raw_prediction, "Unknown")


    # Stats section
    stats = [
        f"{home_raw.title()} Last 40: Avg Goals Scored: {avg_home['FTHG']:.2f}, Avg Shots: {avg_home['HS']:.1f}, Avg Possession (proxy): {avg_home['HST']:.1f}",
        f"{away_raw.title()} Last 40: Avg Goals Scored: {avg_away['FTAG']:.2f}, Avg Shots: {avg_away['AS']:.1f}, Avg Possession (proxy): {avg_away['AST']:.1f}",
    ]

    # Simple discussion based on metrics
    if avg_home['FTHG'] > avg_away['FTAG']:
        reason = f"{home_raw} has shown slightly better scoring form recently."
    elif avg_away['FTAG'] > avg_home['FTHG']:
        reason = f"{away_raw} has been scoring more consistently over the last few matches."
    else:
        reason = "Both teams have similar recent scoring performance."

    discussion = f"Based on recent stats and overall form, the model predicts a **'{prediction}'**. {reason}"

    return {
        "prediction": prediction,
        "stats": stats,
        "discussion": discussion
    }
