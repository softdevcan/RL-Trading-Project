
import requests
import pandas as pd
import json

API_KEY = "tV4qq6RzPr"
SERIES_CODES = {
    'policy_rate': 'TP.YSSK.A01',
    'bist100_index': 'TP.HISSE.XUTUM',
    'cpi_inflation': 'TP.FG.J0'
}

def test_evds_series(name, code):
    print(f"\nTesting {name} ({code})...")
    url = f"https://evds2.tcmb.gov.tr/service/evds/series={code}"
    params = {
        'startDate': '01-01-2024',
        'endDate': '01-02-2024',
        'type': 'json'
    }
    headers = {'key': API_KEY}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and data['items']:
                print(f"Success! First item: {data['items'][0]}")
            else:
                print("Response 200 but no items found.")
                print(f"Full response: {data}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    for name, code in SERIES_CODES.items():
        test_evds_series(name, code)
