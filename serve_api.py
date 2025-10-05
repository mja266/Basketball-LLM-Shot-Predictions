# serve_api.py
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import difflib

MODEL_PKL = "shots_model.pkl"
ENCODERS_PKL = "shots_encoders.pkl"

app = Flask(__name__)
CORS(app)

print("📂 Loading model + encoders...")
model = joblib.load(MODEL_PKL)
enc = joblib.load(ENCODERS_PKL)

SUPPORTED_PLAYERS = [
    "Stephen Curry", "LeBron James", "Jayson Tatum", "Giannis Antetokounmpo", "Luka Doncic"
]

# Full 30-team NBA dictionary
TEAM_PLAYERS = {
    "Atlanta Hawks": ["Trae Young", "Dejounte Murray", "Clint Capela"],
    "Boston Celtics": ["Jayson Tatum", "Jaylen Brown", "Jrue Holiday"],
    "Brooklyn Nets": ["Mikal Bridges", "Cameron Johnson", "Ben Simmons"],
    "Charlotte Hornets": ["LaMelo Ball", "Brandon Miller"],
    "Chicago Bulls": ["Zach LaVine", "DeMar DeRozan", "Nikola Vucevic"],
    "Cleveland Cavaliers": ["Donovan Mitchell", "Darius Garland", "Evan Mobley"],
    "Dallas Mavericks": ["Luka Doncic", "Kyrie Irving"],
    "Denver Nuggets": ["Nikola Jokic", "Jamal Murray", "Aaron Gordon"],
    "Detroit Pistons": ["Cade Cunningham", "Jaden Ivey"],
    "Golden State Warriors": ["Stephen Curry", "Klay Thompson", "Draymond Green"],
    "Houston Rockets": ["Jalen Green", "Alperen Sengun", "Fred VanVleet"],
    "Indiana Pacers": ["Tyrese Haliburton", "Myles Turner", "Buddy Hield"],
    "LA Clippers": ["Kawhi Leonard", "James Harden", "Paul George"],
    "Los Angeles Lakers": ["LeBron James", "Anthony Davis", "Austin Reaves"],
    "Memphis Grizzlies": ["Ja Morant", "Desmond Bane", "Jaren Jackson Jr."],
    "Miami Heat": ["Jimmy Butler", "Bam Adebayo", "Tyler Herro"],
    "Milwaukee Bucks": ["Giannis Antetokounmpo", "Damian Lillard", "Khris Middleton"],
    "Minnesota Timberwolves": ["Anthony Edwards", "Karl-Anthony Towns"],
    "New Orleans Pelicans": ["Zion Williamson", "Brandon Ingram"],
    "New York Knicks": ["Jalen Brunson", "Julius Randle", "RJ Barrett"],
    "Oklahoma City Thunder": ["Shai Gilgeous-Alexander", "Chet Holmgren", "Josh Giddey"],
    "Orlando Magic": ["Paolo Banchero", "Franz Wagner"],
    "Philadelphia 76ers": ["Joel Embiid", "Tyrese Maxey", "Paul George"],
    "Phoenix Suns": ["Kevin Durant", "Devin Booker", "Bradley Beal"],
    "Portland Trail Blazers": ["Anfernee Simons", "Scoot Henderson"],
    "Sacramento Kings": ["De’Aaron Fox", "Domantas Sabonis"],
    "San Antonio Spurs": ["Victor Wembanyama", "Keldon Johnson"],
    "Toronto Raptors": ["Scottie Barnes", "RJ Barrett"],
    "Utah Jazz": ["Lauri Markkanen", "Jordan Clarkson"],
    "Washington Wizards": ["Kyle Kuzma", "Jordan Poole"]
}

def safe_encode(label, encoder):
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    match = difflib.get_close_matches(label, encoder.classes_.tolist(), n=1, cutoff=0.6)
    if match:
        return encoder.transform([match[0]])[0]
    return encoder.transform([encoder.classes_[0]])[0]

def to_seconds(pctimestring):
    try:
        mm, ss = pctimestring.split(":")
        return int(mm) * 60 + int(ss)
    except:
        return 120

@app.route("/teams", methods=["GET"])
def get_teams():
    return jsonify({"teams": list(TEAM_PLAYERS.keys())})

@app.route("/players", methods=["GET"])
def get_players():
    team = request.args.get("team")
    return jsonify({"players": TEAM_PLAYERS.get(team, [])})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    player = payload.get("player_name", "")
    team = payload.get("team_id", "")
    shot = payload.get("shot_type", "")

    try:
        player_id = safe_encode(player, enc["player"])
        team_id = safe_encode(team, enc["team"])
        shot_id = safe_encode(shot, enc["shot"])
    except Exception as e:
        return jsonify({"ok": False, "error": f"Encoding error: {str(e)}"}), 400

    period = int(payload.get("period", 1))
    time_remaining = to_seconds(payload.get("pctimestring", "2:00"))
    x = float(payload.get("x", 25.0))
    y = float(payload.get("y", 25.0))
    X = np.array([[player_id, team_id, period, time_remaining, shot_id, x, y]])

    if player in SUPPORTED_PLAYERS:
        prob = float(model.predict_proba(X)[0, 1])
        label = "MADE" if prob >= 0.5 else "MISSED"
        return jsonify({"ok": True, "prediction_label": label, "made_probability": prob})
    else:
        return jsonify({"ok": True, "prediction_label": "N/A", "made_probability": 0})

if __name__ == "__main__":
    print("🚀 Serving Basketball Predictor on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
