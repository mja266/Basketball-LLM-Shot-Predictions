🏀 Basketball Shot Predictor

A full-stack NBA shot prediction system powered by Python (Flask), machine learning, and a sleek web UI.
It predicts whether a player’s shot will be made or missed based on shot type, game context, and shot coordinates.

🚀 Features

🧠 Machine Learning Model trained on real NBA shot data (2024–25 season).

🏀 All 30 NBA Teams selectable, each with dynamically loaded players from the dataset.

🔢 Prediction Engine: Returns “MADE” or “MISSED” with real-time probability.

🎯 Player-Based Season Averages: Players’ 2024–25 shooting percentages drive accuracy.

🌐 Interactive Frontend: HTML/CSS/JS interface for team, player, and shot context selection.

🔧 Flask API Backend: Handles shot prediction, team/player data, and model inference.

📂 Project Structure
File	Description
build_episodes.py	Prepares and trains the LightGBM model on the season_2024_25_shots.csv dataset.
serve_api.py	Flask backend serving team/player data and prediction endpoints.
index.html	Frontend web interface for selecting teams, players, and making predictions.
season_2024_25_shots.csv	NBA shot dataset containing all 2024–25 player shot data.
🖥️ How It Works

Launch the Flask API using serve_api.py.

The frontend (index.html) automatically loads all NBA teams via /teams.

When you select a team, its players populate dynamically from the backend.

Input shot details (period, time remaining, shot type, and coordinates).

Press Predict — the backend evaluates the shot and returns:

“MADE” (green) or “MISSED” (red)

Along with the player’s probability of success.

🧩 Tech Stack

Python: Flask, LightGBM, Pandas, NumPy

Frontend: HTML, CSS, JavaScript

Data Source: 2024–25 NBA shot data (season_2024_25_shots.csv)

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Basketball-Shot-Predictor.git
cd Basketball-Shot-Predictor

2️⃣ Install Dependencies
pip install flask flask-cors joblib lightgbm pandas numpy scikit-learn

3️⃣ Train the Model
python build_episodes.py


This will generate:

shots_model.pkl — Trained ML model

shots_encoders.pkl — Encoders for teams, players, and shot types

player_shot_averages.pkl — Computed player season averages

4️⃣ Start the Flask API
python serve_api.py


Expected output:

📂 Loading models and encoders...
✅ Loaded 30 teams with player mappings
🚀 Serving Basketball Play Predictor on http://127.0.0.1:5000

5️⃣ Open the Frontend

Open index.html directly in your browser (e.g. via Live Server in VS Code).

The web app should automatically load all NBA teams and their rosters.

📂 Data Access

The 2024–25 NBA shot dataset used for training and predictions is hosted externally due to GitHub’s file size limits.
You can download it directly from Dropbox using the link below:

🔗 Download the 2024–25 NBA Shot Dataset (season_2024_25_shots.csv)

Once downloaded, place the file in the root directory of this project (alongside build_episodes.py, serve_api.py, and index.html) before running the model training step:

python build_episodes.py


If you prefer, you can also modify build_episodes.py to automatically download the dataset by adding:

DROPBOX_CSV_URL = "https://www.dropbox.com/scl/fi/2p2ym6akwvqbu0d8nb648/season_2024_25_shots.csv?rlkey=80ua85sxeijwgnu8gjmxs1ida&st=luv4qhix&dl=1"

🧠 Prediction Logic

If the selected player has recorded season stats, predictions use their 2024–25 FG%.

Otherwise, the trained LightGBM model provides a context-based probability.

Results appear instantly with color-coded feedback:

🟢 MADE — Successful shot

🔴 MISSED — Missed shot

📊 Example Usage

Select Team: “Golden State Warriors”

Select Player: “Stephen Curry”

Select Shot Type: “3PT”

Coordinates: x = 25, y = 25

Click Predict →
Output:

Stephen Curry MADE (42.8% chance)

🏁 Notes

Ensure the Flask API runs before loading the web interface.

Predictions rely on both season averages and the trained model.

The project is designed for expansion — new players or seasons can be added by updating the dataset and rerunning build_episodes.py.
