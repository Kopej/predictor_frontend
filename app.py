from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ------------------ INIT APP -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ EPL (RULE-BASED) -------------------
epl_df = pd.read_csv("epl_matches_2017_2023.csv")

PROMOTED_TEAMS_NOTE = """
Note:
- Sunderland, Leeds United, and Burnley are in the Premier League for the 2025/26 season.
"""

def summarize_form(matches, team_name):
    wins = (matches['Result'] == 'W').sum()
    draws = (matches['Result'] == 'D').sum()
    losses = (matches['Result'] == 'L').sum()
    avg_gf = matches['GF'].mean()
    avg_ga = matches['GA'].mean()
    return f"""{team_name} - Last {len(matches)} Matches:
- Wins: {wins}, Draws: {draws}, Losses: {losses}
- Avg Goals Scored: {avg_gf:.2f}, Avg Goals Conceded: {avg_ga:.2f}"""

def summarize_h2h(matches, home, away):
    total = len(matches)
    home_wins = ((matches['Team'] == home) & (matches['Result'] == 'W')).sum()
    away_wins = ((matches['Team'] == away) & (matches['Result'] == 'W')).sum()
    draws = (matches['Result'] == 'D').sum()
    return f"Head-to-Head ({home} vs {away}) - Last {total} Matches:\n- {home} Wins: {home_wins}, {away} Wins: {away_wins}, Draws: {draws}"

def get_team_form(df, team, n=5):
    recent = df[df['Team'] == team].sort_values(by='date', ascending=False).head(n)
    return {
        "wins": (recent['Result'] == 'W').sum(),
        "draws": (recent['Result'] == 'D').sum(),
        "losses": (recent['Result'] == 'L').sum(),
        "avg_gf": recent['GF'].mean(),
        "avg_ga": recent['GA'].mean(),
        "avg_xg": recent['xG'].mean() if 'xG' in recent else 0,
    }

def get_head_to_head(df, home, away, n=5):
    h2h = df[((df['Team'] == home) & (df['Opponent'] == away)) |
             ((df['Team'] == away) & (df['Opponent'] == home))]\
             .sort_values(by='date', ascending=False).head(n)
    return {
        "home_wins": ((h2h['Team'] == home) & (h2h['Result'] == 'W')).sum(),
        "away_wins": ((h2h['Team'] == away) & (h2h['Result'] == 'W')).sum(),
        "draws": (h2h['Result'] == 'D').sum()
    }

def rule_based_prediction(home_team, away_team):
    home = get_team_form(epl_df, home_team)
    away = get_team_form(epl_df, away_team)
    h2h = get_head_to_head(epl_df, home_team, away_team)

    home_score = home['wins'] + int(home['avg_gf'] > away['avg_ga']) + int(home['avg_xg'] > away['avg_xg']) + h2h['home_wins']
    away_score = away['wins'] + int(away['avg_gf'] > home['avg_ga']) + int(away['avg_xg'] > home['avg_xg']) + h2h['away_wins']

    diff = home_score - away_score
    if diff >= 2:
        return f"{home_team} to win"
    elif diff <= -2:
        return f"{away_team} to win"
    else:
        return "Draw"

@app.post("/predict-epl")
async def predict_epl(request: Request):
    data = await request.json()
    home = data.get("home_team")
    away = data.get("away_team")
    if not home or not away:
        return {"error": "Missing team names."}

    prediction = rule_based_prediction(home, away)
    form_home = summarize_form(epl_df[epl_df['Team'] == home].sort_values(by='date', ascending=False).head(10), home)
    form_away = summarize_form(epl_df[epl_df['Team'] == away].sort_values(by='date', ascending=False).head(10), away)
    h2h_data = epl_df[((epl_df['Team'] == home) & (epl_df['Opponent'] == away)) |
                      ((epl_df['Team'] == away) & (epl_df['Opponent'] == home))].sort_values(by='date', ascending=False).head(10)
    h2h = summarize_h2h(h2h_data, home, away)

    return {
        "prediction": prediction,
        "stats": [form_home, form_away, h2h],
        "discussion": f"{home} has {form_home.splitlines()[1]}. {away} shows {form_away.splitlines()[1]}. {h2h.splitlines()[-1]}. Based on this, predicted result: {prediction.lower()}."
    }

# ------------------ BUNDESLIGA & SERIE A (ML MODELS) -------------------
bundesliga_df = pd.read_csv("Bundesliga/Bundesliga_season_2010_2025.csv")
seriea_df = pd.read_csv("Serie A/serie_a_season_2013_2025.csv")

features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']

def train_model(df):
    return RandomForestClassifier(n_estimators=150, random_state=42).fit(df[features].dropna(), df['FTR'].dropna())

bundesliga_model = train_model(bundesliga_df.dropna())
seriea_model = train_model(seriea_df.dropna())

bundesliga_map = {
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

seriea_map = {
    "inter milan": "inter",
    "ac milan": "milan",
    "as roma": "roma",
    "atalanta bc": "atalanta",
    "hellas verona": "verona"
}

def normalize(name, name_map):
    return name_map.get(name.strip().lower(), name.strip().lower())

def predict_match(df, model, home_raw, away_raw, team_map):
    home = normalize(home_raw, team_map)
    away = normalize(away_raw, team_map)

    home_recent = df[(df['HomeTeam'].str.lower() == home) | (df['AwayTeam'].str.lower() == home)].sort_values(by='Date', ascending=False).head(60)
    away_recent = df[(df['HomeTeam'].str.lower() == away) | (df['AwayTeam'].str.lower() == away)].sort_values(by='Date', ascending=False).head(60)

    if home_recent.empty or away_recent.empty:
        return {"prediction": "No prediction", "stats": [], "discussion": "Insufficient recent match data."}

    avg_home = home_recent[features].mean()
    avg_away = away_recent[features].mean()
    input_data = np.mean([avg_home.values, avg_away.values], axis=0).reshape(1, -1)
    result = model.predict(input_data)[0]

    mapping = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    prediction = mapping.get(result, "Unknown")

    stats = [
        f"{home_raw.title()} Avg Goals: {avg_home['FTHG']:.2f}, Shots: {avg_home['HS']:.1f}, On Target: {avg_home['HST']:.1f}",
        f"{away_raw.title()} Avg Goals: {avg_away['FTAG']:.2f}, Shots: {avg_away['AS']:.1f}, On Target: {avg_away['AST']:.1f}"
    ]
    discussion = f"Prediction based on recent form: {prediction}."

    return {"prediction": prediction, "stats": stats, "discussion": discussion}

@app.post("/predict-bundesliga")
async def predict_bundesliga(request: Request):
    data = await request.json()
    return predict_match(bundesliga_df, bundesliga_model, data.get("home_team", ""), data.get("away_team", ""), bundesliga_map)

@app.post("/predict-seriea")
async def predict_seriea(request: Request):
    data = await request.json()
    return predict_match(seriea_df, seriea_model, data.get("home_team", ""), data.get("away_team", ""), seriea_map)
