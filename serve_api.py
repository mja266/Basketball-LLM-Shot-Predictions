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

print("📂 Loading dataset...")
model = joblib.load(MODEL_PKL)
enc = joblib.load(ENCODERS_PKL)

# Players we support for real predictions
SUPPORTED_PLAYERS = [
    "Giannis Antetokounmpo",
    "Jayson Tatum",
    "LeBron James",
    "Stephen Curry",
    "Luka Doncic"
]

# The fallback team if a team isn’t encoded
FALLBACK_TEAM = "Los Angeles Lakers"
FALLBACK_PLAYER = "LeBron James"
FALLBACK_SHOT = "2PT Field Goal"  # default safe value

def safe_encode(label, encoder, label_type=""):
    """Safely encode label or fall back to a close match or default."""
    classes = encoder.classes_.tolist()
    if label in classes:
        return encoder.transform([label])[0]

    # handle fallback logic
    if label_type == "player":
        if label == "Luka Doncic" and "Luka Doncic" not in classes:
            print("[INFO] Luka Doncic not in encoder – falling back to LeBron James")
            return encoder.transform([FALLBACK_PLAYER])[0]

    if label_type == "team":
        if label == "Dallas Mavericks" and "Dallas Mavericks" not in classes:
            print("[INFO] Dallas Mavericks not in encoder – falling back to Los Angeles Lakers")
            return encoder.transform([FALLBACK_TEAM])[0]

    # fuzzy match (closest)
    match = difflib.get_close_matches(label, classes, n=1, cutoff=0.6)
    if match:
        print(f"[INFO] Using fuzzy match for {label_type}: {match[0]}")
        return encoder.transform([match[0]])[0]

    # fallback to first class if nothing matches
    print(f"[WARN] Unknown {label_type}: {label}. Using first known class: {classes[0]}")
    return encoder.transform([classes[0]])[0]


def to_seconds(pctimestring: str) -> int:
    try:
        mm, ss = pctimestring.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 120


def preprocess(payload):
    player_name = payload.get("player_name")
    if player_name not in SUPPORTED_PLAYERS:
        return None, f"Predictions currently supported only for: {', '.join(SUPPORTED_PLAYERS)}."

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
    X, err = preprocess(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    try:
        prob = float(model.predict_proba(X)[0, 1])
        label = "MADE" if prob >= 0.5 else "MISSED"
        return jsonify({
            "ok": True,
            "made_probability": prob,
            "prediction_label": label,
            "debug": payload
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    print("🚀 Serving Basketball Play Predictor on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
