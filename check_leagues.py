import requests, json

API_KEY = "YOUR_REAL_API_KEY_HERE"
HEADERS = {"x-apisports-key": API_KEY}

url = "https://v1.basketball.api-sports.io/leagues"
resp = requests.get(url, headers=HEADERS)
print("Status:", resp.status_code)

data = resp.json()
print(json.dumps(data, indent=2))
