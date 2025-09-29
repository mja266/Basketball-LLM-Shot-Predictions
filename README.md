🏀 Basketball Shot Predictor

This project is a basketball shot prediction system built with Python (Flask), machine learning, and a simple web UI.
It predicts whether a player will make or miss a shot based on game context (quarter, time remaining, shot type, and court coordinates).

Currently, the predictor supports five star players:

Stephen Curry (GSW)

LeBron James (LAL)

Luka Dončić (DAL)

Jayson Tatum (BOS)

Giannis Antetokounmpo (MIL)

🚀 Features

Machine learning model trained on historical shot data.

Predicts HIT (1) or MISS (0) with an associated probability.

Interactive web interface with dropdowns for players, teams, shot type, and game context.

Visual progress bar showing prediction confidence.

Backend API built with Flask; frontend built with HTML, CSS, JavaScript.

📂 Project Structure

build_episodes.py → Prepares and trains the model.

serve_api.py → Runs the Flask backend and exposes the /predict endpoint.

index.html → Frontend interface for making predictions.

🖥️ How It Works

Select a player, shot type, and game context (period, time left, coordinates).

The app sends the input to the Flask API.

The trained ML model returns a probability of a make or miss.

The result is shown in the UI with a ✅ HIT or ❌ MISS indicator and a colored probability bar.

🔧 Tech Stack

Python (Flask, scikit-learn, pandas, numpy)

Frontend: HTML, CSS, JavaScript

Data: NBA shot data (restricted to the 5 players above)
