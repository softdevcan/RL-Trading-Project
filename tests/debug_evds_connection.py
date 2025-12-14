
import requests
import json

API_KEY = "tV4qq6RzPr"
TEST_SERIES = "TP.DK.USD.A.YTL" # USD/TRY is usually open

def test_connection_variations():
    url = f"https://evds2.tcmb.gov.tr/service/evds/series={TEST_SERIES}"
    params = {
        'startDate': '01-01-2024',
        'endDate': '10-01-2024',
        'type': 'json'
    }
    
    print(f"Testing API Key: {API_KEY}")
    
    # 1. Standard Header
    print("\n1. Standard Header {'key': API_KEY}")
    try:
        resp = requests.get(url, params=params, headers={'key': API_KEY})
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200: print("Success!")
    except Exception as e: print(e)

    # 2. With User-Agent
    print("\n2. With User-Agent")
    try:
        headers = {
            'key': API_KEY,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, params=params, headers=headers)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200: print("Success!")
    except Exception as e: print(e)

    # 3. Check if key is empty or placeholder
    if API_KEY == "tV4qq6RzPr":
        print("\nNOTE: Using the API key found in the code. Is this a real key or a placeholder?")

if __name__ == "__main__":
    test_connection_variations()
