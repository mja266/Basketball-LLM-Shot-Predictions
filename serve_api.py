# serve_api.py
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PKL = "shots_model.pkl"
ENCODERS_PKL = "shots_encoders.pkl"

app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_PKL)
enc    = joblib.load(ENCODERS_PKL)

def to_seconds(pctimestring: str) -> int:
    mm, ss = pctimestring.split(":")
    return int(mm) * 60 + int(ss)

def preprocess(payload):
    """
    Expected JSON:
    {
      "player_name": "Stephen Curry",
      "team_id": "GSW",
      "period": 4,
      "pctimestring": "2:00",
      "shot_type": "3PT",
      "x": 28.5,
      "y": 22.0
    }
    """
    try:
        player_id = enc["player"].transform([payload["player_name"]])[0]
    except Exception:
        return None, f"Unknown player_name: {payload.get('player_name')}. Known: {enc['player_classes']}"
    try:
        team_id   = enc["team"].transform([payload["team_id"]])[0]
    except Exception:
        return None, f"Unknown team_id: {payload.get('team_id')}. Known: {enc['team_classes']}"
    try:
        shot_id   = enc["shot"].transform([payload["shot_type"]])[0]
    except Exception:
        return None, f"Unknown shot_type: {payload.get('shot_type')}. Known: {enc['shot_classes']}"

    period = int(payload.get("period", 1))
    time_remaining = to_seconds(payload.get("pctimestring", "2:00"))
    x = float(payload.get("x", 25.0))
    y = float(payload.get("y", 25.0))

    row = np.array([[player_id, team_id, period, time_remaining, shot_id, x, y]], dtype=float)
    return row, None

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    X, err = preprocess(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    prob = float(model.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)
    return jsonify({
        "ok": True,
        "made_probability": prob,
        "prediction_label": "MADE" if label == 1 else "MISSED",
        "debug": payload
    })

if __name__ == "__main__":
    print("🚀 Serving on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
