import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import difflib
import pandas as pd
from pathlib import Path

MODEL_PKL = "shots_model.pkl"
ENCODERS_PKL = "shots_encoders.pkl"
PLAYER_AVG_PKL = "player_shot_averages.pkl"
DATA_CSV = "season_2024_25_shots.csv"

app = Flask(__name__)
# Allow all origins (so https://mja266.github.io can access it)
CORS(app, resources={r"/*": {"origins": "*"}})

print("📂 Loading models and encoders...")
model = joblib.load(MODEL_PKL)
enc = joblib.load(ENCODERS_PKL)
player_avgs = joblib.load(PLAYER_AVG_PKL)

# Load team → players mapping from CSV
if not Path(DATA_CSV).exists():
    raise FileNotFoundError(f"{DATA_CSV} not found!")

df = pd.read_csv(DATA_CSV)
df = df.rename(columns={
    "PLAYER_NAME": "player_name",
    "TEAM_NAME": "team_name"
})
df = df.dropna(subset=["player_name", "team_name"])

team_to_players = (
    df.groupby("team_name")["player_name"]
      .unique()
      .apply(list)
      .to_dict()
)
all_teams = sorted(team_to_players.keys())
print(f"✅ Loaded {len(all_teams)} teams with player mappings.")

SUPPORTED_PLAYERS = list(player_avgs.keys())

# ---------------- Helper Functions ----------------

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

# ---------------- API Endpoints ----------------

@app.route("/teams", methods=["GET"])
def get_teams():
    """Return all teams and their players for dropdown population."""
    return jsonify({
        "ok": True,
        "teams": team_to_players
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Hybrid model: combines player realism with spatial/contextual variability."""
    payload = request.get_json(force=True)
    player = payload.get("player_name", "").strip()
    X, err = preprocess(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    try:
        # --- Player FG% baseline ---
        base_fg = float(player_avgs.get(player, 0.45))  # Default league avg if missing

        # --- Model contextual/spatial adjustment ---
        model_prob = float(model.predict_proba(X)[0, 1])

        # --- Weighted blend ---
        # 70% player realism + 30% model spatial/context features
        final_prob = 0.7 * base_fg + 0.3 * model_prob

        # Determine result
        label = "MADE" if np.random.rand() < final_prob else "MISSED"

        return jsonify({
            "ok": True,
            "prediction_label": label,
            "made_probability": final_prob,
            "method": "hybrid_model",
            "player": player
        })

    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction error: {str(e)}"}), 500

# ---------------- Server Launch ----------------

if __name__ == "__main__":
    print("Serving 🏀 Basketball Shot Predictor on https://mja266.github.io")
    app.run(host="0.0.0.0", port=5000, debug=True)
