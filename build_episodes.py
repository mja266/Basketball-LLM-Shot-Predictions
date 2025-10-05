# build_episodes.py
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

DATA_CSV = "season_2024_25_shots.csv"
MODEL_PKL = "shots_model.pkl"
ENCODERS_PKL = "shots_encoders.pkl"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SUPPORTED_PLAYERS = [
    "Stephen Curry", "LeBron James", "Jayson Tatum", "Giannis Antetokounmpo", "Luka Doncic"
]

def main():
    df = pd.read_csv(DATA_CSV)
    df = df.rename(columns={
        "PLAYER_NAME": "player_name",
        "TEAM_NAME": "team_name",
        "SHOT_TYPE": "shot_type",
        "PERIOD": "period",
        "SECONDS_REMAINING": "seconds_remaining",
        "LOC_X": "x",
        "LOC_Y": "y",
        "SHOT_MADE_FLAG": "made"
    })
    df = df.dropna(subset=["player_name", "team_name", "shot_type", "x", "y", "made"])
    df["made"] = df["made"].astype(int)
    df["time_remaining"] = df["seconds_remaining"].astype(int)

    # Train only on supported players
    df = df[df["player_name"].isin(SUPPORTED_PLAYERS)]

    enc_player = LabelEncoder().fit(df["player_name"])
    enc_team = LabelEncoder().fit(df["team_name"])
    enc_shot = LabelEncoder().fit(df["shot_type"])

    X = pd.DataFrame({
        "player_id": enc_player.transform(df["player_name"]),
        "team_id": enc_team.transform(df["team_name"]),
        "period": df["period"].astype(int),
        "time_remaining": df["time_remaining"].astype(int),
        "shot_id": enc_shot.transform(df["shot_type"]),
        "x": df["x"].astype(float),
        "y": df["y"].astype(float),
    })
    y = df["made"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
    model = LGBMClassifier(n_estimators=200, learning_rate=0.08, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"✅ Model trained | Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    joblib.dump(model, MODEL_PKL)
    joblib.dump({
        "player": enc_player,
        "team": enc_team,
        "shot": enc_shot,
        "player_classes": enc_player.classes_.tolist(),
        "team_classes": enc_team.classes_.tolist(),
        "shot_classes": enc_shot.classes_.tolist(),
    }, ENCODERS_PKL)
    print(f"💾 Saved: {MODEL_PKL}, {ENCODERS_PKL}")

if __name__ == "__main__":
    main()
