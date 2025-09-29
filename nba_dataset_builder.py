import requests
import csv
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
OUT_FILE = "nba_play_dataset.csv"
START_DATE = datetime(2024, 6, 17).date()  # last known NBA game
MAX_RETRIES = 3

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]


def get_headers():
    return {
        "Host": "stats.nba.com",
        "Connection": "keep-alive",
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Accept-Encoding": "gzip, deflate, br"
    }


# ----------------------------
# Fetch game IDs for a date
# ----------------------------
def get_games_for_date(date):
    url = "https://stats.nba.com/stats/scoreboardv2"
    params = {"GameDate": date.strftime("%m/%d/%Y"), "LeagueID": "00", "DayOffset": "0"}

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"📅 Checking {date} ...")
        try:
            resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            games = data["resultSets"][0]["rowSet"]
            return [g[2] for g in games]  # game_id column
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            time.sleep(5)
    return []


# ----------------------------
# Fetch play-by-play for one game
# ----------------------------
def get_play_by_play(game_id):
    url = "https://stats.nba.com/stats/playbyplayv2"
    params = {"GameID": game_id, "StartPeriod": 1, "EndPeriod": 10}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            rows = data["resultSets"][0]["rowSet"]
            return rows
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed for game {game_id}: {e}")
            time.sleep(5)
    print(f"❌ Skipping game {game_id}")
    return []


# ----------------------------
# Main build logic
# ----------------------------
def build_dataset():
    date = START_DATE
    all_rows = []

    while not all_rows:
        game_ids = get_games_for_date(date)
        if game_ids:
            print(f"✅ Found {len(game_ids)} games on {date}")
            for gid in game_ids:
                plays = get_play_by_play(gid)
                all_rows.extend(plays)
                time.sleep(3)
        else:
            print(f"⚠️ No games found on {date}. Trying previous day...")
            date -= timedelta(days=1)

    print(f"💾 Saving {len(all_rows)} rows to {OUT_FILE}")
    Path(OUT_FILE).write_text("")  # clear file if exists
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["game_id", "event_num", "period", "pctimestring",
                         "team_id", "player1_name", "event_msg_type", "description"])
        for row in all_rows:
            writer.writerow([row[0], row[1], row[4], row[6], row[7], row[17], row[2], row[9]])

    print("✅ Done - Dataset ready!")


if __name__ == "__main__":
    build_dataset()
