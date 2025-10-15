🏀 Basketball Shot Predictor

A full-stack NBA shot prediction web app powered by machine learning and an interactive browser-based interface.
Built with a Flask backend (hosted on Render) and a sleek HTML/CSS/JS frontend (hosted via GitHub Pages), it predicts whether an NBA player’s shot will be MADE or MISSED based on shot type, game context, and shot coordinates.

🌐 Live Demo:
👉 https://mja266.github.io/NBA-Shot-Predictor/

🚀 Features

🧠 Machine Learning Model trained on real 2024–25 NBA shot data.

🏀 All 30 NBA Teams selectable — players dynamically load from the live backend.

🎯 Prediction Engine: Instantly returns “MADE” or “MISSED” with probability.

📊 Player-Based Season Averages: Predictions weighted by individual FG%.

🌐 Frontend (GitHub Pages): Responsive and fully client-side.

🔧 Backend (Render): Flask API for player/team data and model inference.

🖥️ How It Works

Visit the Live Web App:
Go to 👉 https://mja266.github.io/NBA-Shot-Predictor/

Select Team & Player:
The frontend automatically fetches all NBA teams and player rosters from the live backend.

Enter Game Context:
Choose:

Period (1–4)

Time remaining (mm:ss)

Shot type (2PT / 3PT / Free Throw)

X/Y coordinates (court location)

Click Predict:
Instantly get:

Kyrie Irving MADE (47.7% chance)


or

Jayson Tatum MISSED (35.2% chance)


Enjoy!
The interface and backend communicate seamlessly — no installation required.

🧩 Tech Stack
Layer	Technology
Frontend	HTML, CSS, JavaScript
Backend	Python (Flask, Flask-CORS)
Machine Learning	LightGBM, scikit-learn, Pandas, NumPy
Hosting	GitHub Pages (Frontend), Render (Backend API)
Data	2024–25 NBA Shot Dataset (privately hosted)
⚙️ Project Structure
NBA-Shot-Predictor/
│
├── index.html              # Frontend Web UI (GitHub Pages)
├── build_episodes.py       # Model training script
├── serve_api.py            # Flask backend API (Render)
├── season_2024_25_shots.csv # Training dataset (stored privately)
├── requirements.txt        # Python dependencies for backend
├── Dockerfile              # Render deployment file
└── README.md               # Project documentation

🌍 Architecture Overview
+-------------------------+
|   GitHub Pages (UI)     |
|  index.html + JS Logic  |
+-----------+-------------+
            |
            |  REST API calls (HTTPS)
            v
+--------------------------+
| Flask Backend (Render)   |
| serve_api.py + Model.pkl |
|  -> /teams               |
|  -> /players/<team>      |
|  -> /predict             |
+--------------------------+


Frontend → Backend Flow Example:

fetch("https://nba-shot-predictor.onrender.com/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    team: "Dallas Mavericks",
    player: "Kyrie Irving",
    period: 4,
    time_remaining: "2:00",
    shot_type: "2PT Field Goal",
    x: 25,
    y: 25
  })
})


Backend responds with:

{
  "result": "MADE",
  "probability": 0.477
}

🧠 Prediction Logic

If the selected player has season data, predictions blend their personal FG% with the model output.

If not, the LightGBM model provides a context-based probability.

Results appear instantly with color-coded feedback:

🟢 MADE — Successful shot

🔴 MISSED — Missed shot

🧾 Example Usage

Scenario:

Team: Golden State Warriors

Player: Stephen Curry

Shot Type: 3PT Field Goal

Coordinates: (25, 25)

Time: 2:00 left in 4th quarter

Prediction Output:

Stephen Curry MADE (42.8% chance)

📡 Deployment Info
Frontend (GitHub Pages)

Publicly accessible at
https://mja266.github.io/NBA-Shot-Predictor/

Backend (Render)

Flask API hosted at
https://nba-shot-predictor.onrender.com

Endpoints:

/teams — returns list of NBA teams

/players/<team> — returns roster for given team

/predict — returns prediction result and probability

🏁 Notes

The web app is fully functional and requires no setup or downloads.

The dataset and model are pre-trained and hosted remotely for fast API inference.

If the Render backend sleeps (free plan), the first request may take ~50 seconds to wake up.

After waking, performance is instantaneous.

💡 Future Enhancements

🧩 Add shot chart visualization (interactive court map)

📈 Display team shooting analytics

🔊 Add commentary/voice feedback for predictions

🏗️ Expand to WNBA / EuroLeague datasets

👨‍💻 Author

Mohamed Abdalla
🎓 Cornell University — B.S. Computer Science, Minor in ECE
🔗 LinkedIn

💻 GitHub

🏀 Live Project Links
Component	URL
Frontend (GitHub Pages)	https://mja266.github.io/NBA-Shot-Predictor/

Backend (Render API)	https://nba-shot-predictor.onrender.com
✅ TL;DR

Just visit:
👉 https://mja266.github.io/NBA-Shot-Predictor/

No installation. No setup. Instant NBA shot predictions.
