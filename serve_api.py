import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import difflib
import os

# === File paths ===
MODEL_PKL = "shots_model.pkl"
ENCODERS_PKL = "shots_encoders.pkl"
PLAYER_AVG_PKL = "player_shot_averages.pkl"

# === Flask app setup ===
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

print("📂 Loading model and encoders...")
model = joblib.load(MODEL_PKL)
enc = joblib.load(ENCODERS_PKL)
player_avgs = joblib.load(PLAYER_AVG_PKL)

SUPPORTED_PLAYERS = list(player_avgs.keys())

# === Full NBA teams & key players ===
ALL_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]

TEAM_TO_PLAYERS = {
    "Atlanta Hawks": ["Trae Young", "Dejounte Murray", "Clint Capela"],
    "Boston Celtics": ["Jayson Tatum", "Jaylen Brown", "Jrue Holiday"],
    "Brooklyn Nets": ["Mikal Bridges", "Cam Thomas", "Nic Claxton"],
    "Charlotte Hornets": ["LaMelo Ball", "Miles Bridges", "Brandon Miller"],
    "Chicago Bulls": ["Zach LaVine", "DeMar DeRozan", "Nikola Vucevic"],
    "Cleveland Cavaliers": ["Donovan Mitchell", "Darius Garland", "Evan Mobley"],
    "Dallas Mavericks": ["Luka Doncic", "Kyrie Irving", "Derrick Jones Jr."],
    "Denver Nuggets": ["Nikola Jokic", "Jamal Murray", "Michael Porter Jr."],
    "Detroit Pistons": ["Cade Cunningham", "Jaden Ivey", "Ausar Thompson"],
    "Golden State Warriors": ["Stephen Curry", "Klay Thompson", "Draymond Green"],
    "Houston Rockets": ["Jalen Green", "Fred VanVleet", "Alperen Sengun"],
    "Indiana Pacers": ["Tyrese Haliburton", "Myles Turner", "Buddy Hield"],
    "Los Angeles Clippers": ["Kawhi Leonard", "Paul George", "James Harden"],
    "Los Angeles Lakers": ["LeBron James", "Anthony Davis", "D'Angelo Russell"],
    "Memphis Grizzlies": ["Ja Morant", "Desmond Bane", "Jaren Jackson Jr."],
    "Miami Heat": ["Jimmy Butler", "Bam Adebayo", "Tyler Herro"],
    "Milwaukee Bucks": ["Giannis Antetokounmpo", "Damian Lillard", "Khris Middleton"],
    "Minnesota Timberwolves": ["Anthony Edwards", "Karl-Anthony Towns", "Rudy Gobert"],
    "New Orleans Pelicans": ["Zion Williamson", "Brandon Ingram", "CJ McCollum"],
    "New York Knicks": ["Jalen Brunson", "Julius Randle", "RJ Barrett"],
    "Oklahoma City Thunder": ["Shai Gilgeous-Alexander", "Chet Holmgren", "Jalen Williams"],
    "Orlando Magic": ["Paolo Banchero", "Franz Wagner", "Jalen Suggs"],
    "Philadelphia 76ers": ["Joel Embiid", "Tyrese Maxey", "Paul George"],
    "Phoenix Suns": ["Kevin Durant", "Devin Booker", "Bradley Beal"],
    "Portland Trail Blazers": ["Anfernee Simons", "Scoot Henderson", "Deandre Ayton"],
    "Sacramento Kings": ["De'Aaron Fox", "Domantas Sabonis", "Keegan Murray"],
    "San Antonio Spurs": ["Victor Wembanyama", "Devin Vassell", "Jeremy Sochan"],
    "Toronto Raptors": ["Scottie Barnes", "RJ Barrett", "Immanuel Quickley"],
    "Utah Jazz": ["Lauri Markkanen", "Jordan Clarkson", "Collin Sexton"],
    "Washington Wizards": ["Kyle Kuzma", "Jordan Poole", "Tyus Jones"]
}

# === Utility functions ===
def safe_encode(label, encoder, label_type=""):
    classes = encoder.classes_.tolist()
    if label in classes:
        return encoder.transform([label])[0]
    match = difflib.get_close_matches(label, classes, n=1, cutoff=0.6)
    if match:
        return encoder.transform([match[0]])[0]
    return encoder.transform([classes[0]])[0]

def to_seconds(pctimestring: str) -> int:
    try:
        mm, ss = pctimestring.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 120

def preprocess(payload):
    try:
        player_id = safe_encode(payload["player_name"], enc["player"], "player")
        team_id   = safe_encode(payload["team_id"], enc["team"], "team")
        shot_id   = safe_encode(payload["shot_type"], enc["shot"], "shot")
    except Exception as e:
        return None, f"Encoding error: {e}"

    period = int(payload.get("period", 1))
    time_remaining = to_seconds(payload.get("pctimestring", "2:00"))
    x = float(payload.get("x", 25.0))
    y = float(payload.get("y", 25.0))

    row = np.array([[player_id, team_id, period, time_remaining, shot_id, x, y]], dtype=float)
    return row, None

# === Prediction route ===
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    player = payload.get("player_name", "").strip()
    X, err = preprocess(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    try:
        if player in player_avgs:
            fg_pct = float(player_avgs[player])
            label = "MADE" if np.random.rand() < fg_pct else "MISSED"
            return jsonify({
                "ok": True,
                "prediction_label": label,
                "made_probability": fg_pct,
                "method": "season_average",
                "player": player
            })
        else:
            prob = float(model.predict_proba(X)[0, 1])
            label = "MADE" if prob >= 0.5 else "MISSED"
            return jsonify({
                "ok": True,
                "prediction_label": label,
                "made_probability": prob,
                "method": "model_fallback",
                "player": player
            })
    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction error: {str(e)}"}), 500

# === Routes for frontend dropdowns ===
@app.route("/teams", methods=["GET"])
def get_teams():
    """Return all NBA teams."""
    return jsonify(sorted(ALL_TEAMS))

@app.route("/players/<team>", methods=["GET"])
def get_players(team):
    """Return players for the selected team."""
    players = TEAM_TO_PLAYERS.get(team, [])
    return jsonify(sorted(players))

# === Serve frontend ===
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

# === Run app ===
if __name__ == "__main__":
    print("🚀 Serving Basketball Play Predictor at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
