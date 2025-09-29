# build_episodes.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

DATA_CSV = "shots_dataset.csv"
MODEL_PKL = "shots_model.pkl"
ENCODERS_PKL = "shots_encoders.pkl"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PLAYERS = ["Stephen Curry", "LeBron James", "Jayson Tatum", "Giannis Antetokounmpo", "Luka Doncic"]
TEAMS   = ["GSW", "LAL", "BOS", "MIL", "DAL"]
PLAYER_TO_TEAM = {
    "Stephen Curry": "GSW",
    "LeBron James": "LAL",
    "Jayson Tatum": "BOS",
    "Giannis Antetokounmpo": "MIL",
    "Luka Doncic": "DAL",
}
SHOT_TYPES = ["3PT", "MID", "PAINT", "FT"]

def _base_skill(player, shot_type):
    # crude priors: higher prob in player's comfort zones
    base = 0.45
    if player == "Stephen Curry":
        if shot_type == "3PT": base += 0.25
        if shot_type == "MID": base += 0.05
    if player == "LeBron James":
        if shot_type == "PAINT": base += 0.15
        if shot_type == "MID":   base += 0.05
    if player == "Jayson Tatum":
        if shot_type in ("3PT", "MID"): base += 0.10
    if player == "Giannis Antetokounmpo":
        if shot_type == "PAINT": base += 0.30
        if shot_type == "FT":    base -= 0.10
    if player == "Luka Doncic":
        if shot_type in ("3PT", "MID"): base += 0.12
    return np.clip(base, 0.05, 0.95)

def _time_pressure_boost(seconds_remaining):
    # under 10s: a small negative pressure effect
    if seconds_remaining <= 10: return -0.05
    if seconds_remaining <= 60: return -0.02
    return 0.0

def _distance_modifier(shot_type, x, y):
    # crude distance approximation: half-court is (50, 25); hoop near (5, 25)
    hoop_x, hoop_y = 5.0, 25.0
    d = np.sqrt((x - hoop_x)**2 + (y - hoop_y)**2)
    # treat 3PT as better far, paint better near, mid in between
    if shot_type == "PAINT":
        return np.interp(d, [0, 10, 30], [0.15, 0.05, -0.10])
    if shot_type == "MID":
        return np.interp(d, [5, 20, 35], [-0.10, 0.08, -0.10])
    if shot_type == "3PT":
        return np.interp(d, [15, 25, 45], [-0.10, 0.10, -0.10])
    if shot_type == "FT":
        return 0.10  # assume freebies
    return 0.0

def make_synthetic(n_rows=100):
    rows = []
    for _ in range(n_rows):
        player = np.random.choice(PLAYERS)
        team   = PLAYER_TO_TEAM[player]
        period = int(np.random.choice([1, 2, 3, 4]))
        # time remaining in quarter (seconds)
        seconds_remaining = int(np.random.randint(0, 720))
        mm = seconds_remaining // 60
        ss = seconds_remaining % 60
        pctimestring = f"{mm}:{ss:02d}"
        shot_type = np.random.choice(SHOT_TYPES, p=[0.35, 0.25, 0.35, 0.05])

        # court coords ~ full court (94x50) — we use 0..50 for x to keep around half-court
        # focus offense near hoop_x=5..35 range to be plausible
        x = float(np.random.uniform(0, 50))
        y = float(np.random.uniform(0, 50))

        base = _base_skill(player, shot_type)
        prob = base + _time_pressure_boost(seconds_remaining) + _distance_modifier(shot_type, x, y)
        prob = np.clip(prob + np.random.normal(0, 0.03), 0.02, 0.98)
        made = np.random.rand() < prob

        rows.append({
            "player_name": player,
            "team_id": team,
            "period": period,
            "pctimestring": pctimestring,
            "shot_type": shot_type,
            "x": round(x, 2),
            "y": round(y, 2),
            "made": int(made),
        })
    df = pd.DataFrame(rows)

    # ensure we don't accidentally create all-one-class
    if df["made"].nunique() < 2:
        # flip 10% to guarantee both classes
        flip_idx = df.sample(frac=0.1, random_state=RANDOM_SEED).index
        df.loc[flip_idx, "made"] = 1 - df.loc[flip_idx, "made"]

    return df

def generate_or_load(csv_path):
    if Path(csv_path).exists():
        print("📁 Found existing dataset. Loading…")
        return pd.read_csv(csv_path)
    print("🛠️  Generating synthetic dataset…")
    df = make_synthetic(n_rows=100)
    df.to_csv(csv_path, index=False)
    return df

def to_seconds(pctimestring: str) -> int:
    mm, ss = pctimestring.split(":")
    return int(mm) * 60 + int(ss)

def main():
    df = generate_or_load(DATA_CSV)
    print(f"✅ Rows: {len(df)}  |  Columns: {list(df.columns)}")

    # --- Encode features ---
    enc_player = LabelEncoder().fit(df["player_name"])
    enc_team   = LabelEncoder().fit(df["team_id"])
    enc_shot   = LabelEncoder().fit(df["shot_type"])

    X = pd.DataFrame({
        "player_id": enc_player.transform(df["player_name"]),
        "team_id":   enc_team.transform(df["team_id"]),
        "period":    df["period"].astype(int),
        "time_remaining": df["pctimestring"].apply(to_seconds).astype(int),
        "shot_id":   enc_shot.transform(df["shot_type"]),
        "x":         df["x"].astype(float),
        "y":         df["y"].astype(float),
    })
    y = df["made"].astype(int)

    # simple split (no stratify to avoid tiny-class errors with toy data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )

    model = LGBMClassifier(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")

    print(f"🏁 Accuracy: {acc:.3f} | AUC: {auc:.3f}")

    joblib.dump(model, MODEL_PKL)
    joblib.dump(
        {
            "player": enc_player,
            "team": enc_team,
            "shot": enc_shot,
            "player_classes": enc_player.classes_.tolist(),
            "team_classes": enc_team.classes_.tolist(),
            "shot_classes": enc_shot.classes_.tolist(),
        },
        ENCODERS_PKL,
    )
    print(f"💾 Saved: {MODEL_PKL}, {ENCODERS_PKL}, {DATA_CSV}")

if __name__ == "__main__":
    main()
