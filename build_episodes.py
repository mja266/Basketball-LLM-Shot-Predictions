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
PLAYER_AVG_PKL = "player_shot_averages.pkl"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Keep the same 5-player model training for fallback (unchanged behavior)
SUPPORTED_PLAYERS = [
    "Stephen Curry", "LeBron James", "Jayson Tatum",
    "Giannis Antetokounmpo", "Luka Doncic"
]

def _read_csv(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found!")

    df = pd.read_csv(path)

    # Normalize column names from NBA stats CSV -> our pipeline
    rename_map = {
        "PLAYER_NAME": "player_name",
        "TEAM_NAME": "team_name",
        "SHOT_TYPE": "shot_type",
        "PERIOD": "period",
        "SECONDS_REMAINING": "seconds_remaining",
        "LOC_X": "x",
        "LOC_Y": "y",
        "SHOT_MADE_FLAG": "made",
    }
    # Only rename columns that actually exist
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def _compute_and_save_player_averages(df: pd.DataFrame):
    """
    Compute each player's FG% from the full season CSV and save as {name: pct}.
    This runs BEFORE any filtering so it includes *everyone* in the dataset.
    """
    needed = {"player_name", "made"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise ValueError(f"CSV missing columns required for averages: {missing}")

    # Each row is a shot attempt; average of made (0/1) = FG%
    avgs = (
        df.dropna(subset=["player_name", "made"])
          .groupby("player_name")["made"]
          .mean()
          .to_dict()
    )

    joblib.dump(avgs, PLAYER_AVG_PKL)
    print(f"💾 Saved per-player season averages for {len(avgs):,} players -> {PLAYER_AVG_PKL}")

def main():
    print("📥 Loading dataset...")
    df_full = _read_csv(DATA_CSV)

    # 1) Always (re)build the holistic per-player averages for the whole league.
    _compute_and_save_player_averages(df_full)

    # 2) Train the same fallback model you already use (unchanged behavior):
    #    filter to 5 supported players for a simple, light model.
    print("🧹 Preparing data for the small fallback model (5 players)...")
    df = df_full.copy()

    # Ensure required columns exist for model training
    required = ["player_name", "team_name", "shot_type", "period",
                "seconds_remaining", "x", "y", "made"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required model columns: {missing}")

    # Coerce types and clean NaNs
    df = df.dropna(subset=required).copy()
    df["made"] = df["made"].astype(int)
    df["period"] = df["period"].astype(int)
    df["seconds_remaining"] = df["seconds_remaining"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)

    # Keep only the 5 supported players for the light fallback model
    df_small = df[df["player_name"].isin(SUPPORTED_PLAYERS)].copy()
    if df_small.empty:
        raise ValueError(
            "No rows found for the 5 supported players in the CSV. "
            "Please confirm names match the dataset."
        )

    # Encoders for the model (fit on the small training subset, unchanged)
    enc_player = LabelEncoder().fit(df_small["player_name"])
    enc_team   = LabelEncoder().fit(df_small["team_name"])
    enc_shot   = LabelEncoder().fit(df_small["shot_type"])

    X = pd.DataFrame({
        "player_id": enc_player.transform(df_small["player_name"]),
        "team_id":   enc_team.transform(df_small["team_name"]),
        "period":    df_small["period"].astype(int),
        "time_remaining": df_small["seconds_remaining"].astype(int),
        "shot_id":   enc_shot.transform(df_small["shot_type"]),
        "x":         df_small["x"].astype(float),
        "y":         df_small["y"].astype(float),
    })
    y = df_small["made"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )

    model = LGBMClassifier(
        n_estimators=200,
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

    print(f"✅ Fallback model trained | Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    joblib.dump(model, MODEL_PKL)
    joblib.dump(
        {
            "player": enc_player,
            "team": enc_team,
            "shot": enc_shot,
            # keeping class lists if you need to inspect:
            "player_classes": enc_player.classes_.tolist(),
            "team_classes": enc_team.classes_.tolist(),
            "shot_classes": enc_shot.classes_.tolist(),
        },
        ENCODERS_PKL,
    )
    print(f"💾 Saved: {MODEL_PKL}, {ENCODERS_PKL}")

if __name__ == "__main__":
    main()
