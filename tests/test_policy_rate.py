
import requests
import json

API_KEY = "tV4qq6RzPr"
# TP.KTF10: TCMB Weighted Average Funding Cost (often close to policy rate)
# TP.YSSK.A01: Overnight Borrowing (The one that failed)
CODES = ['TP.KTF10', 'TP.YSSK.A01']

def test_evds_code(code):
    print(f"\nTesting {code}...")
    url = f"https://evds2.tcmb.gov.tr/service/evds/series={code}"
    params = {
        'startDate': '01-01-2024',
        'endDate': '10-01-2024',
        'type': 'json'
    }
    headers = {'key': API_KEY}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and data['items']:
                print(f"Success! First item: {data['items'][0]}")
            else:
                print("Response 200 but no items.")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    for code in CODES:
        test_evds_code(code)
