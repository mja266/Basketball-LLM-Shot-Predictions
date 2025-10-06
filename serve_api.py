import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import difflib

MODEL_PKL = "shots_model.pkl"
ENCODERS_PKL = "shots_encoders.pkl"
PLAYER_AVG_PKL = "player_shot_averages.pkl"

app = Flask(__name__)
CORS(app)

print("📂 Loading models and encoders...")
model = joblib.load(MODEL_PKL)
enc = joblib.load(ENCODERS_PKL)
player_avgs = joblib.load(PLAYER_AVG_PKL)

# ✅ No longer use enc["supported_players"]; instead, use all players in averages
SUPPORTED_PLAYERS = list(player_avgs.keys())

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

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    player = payload.get("player_name", "").strip()
    X, err = preprocess(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    try:
        # ✅ Use season average if available for this player
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

if __name__ == "__main__":
    print("🚀 Serving Basketball Play Predictor on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
