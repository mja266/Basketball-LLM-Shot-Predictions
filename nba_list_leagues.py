import requests

API_KEY = "YOUR_REAL_API_KEY_HERE"
HEADERS = {"x-apisports-key": API_KEY}

resp = requests.get("https://v1.basketball.api-sports.io/leagues", headers=HEADERS)
data = resp.json()

print("Status:", resp.status_code)
for league in data.get("response", []):
    print(league.get("id"), "-", league.get("name"))
