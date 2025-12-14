
import yfinance as yf
import requests
import pandas as pd

API_KEY = "tV4qq6RzPr"

def test_yfinance():
    print("\nTesting yfinance...")
    symbols = ['XU100.IS', 'TRY=X', 'EURTRY=X']
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="5d")
            if not hist.empty:
                print(f"Success {sym}: Last close {hist['Close'].iloc[-1]}")
            else:
                print(f"Failed {sym}: No data")
        except Exception as e:
            print(f"Error {sym}: {e}")

def test_evds_formulas():
    print("\nTesting EVDS Formulas...")
    # TP.FG.J0 is CPI Index. Try to get YoY change.
    # transformation: 5 (Percentage change compared to same period of previous year)
    code = 'TP.FG.J0'
    url = f"https://evds2.tcmb.gov.tr/service/evds/series={code}"
    params = {
        'startDate': '01-01-2024',
        'endDate': '01-02-2024',
        'type': 'json',
        'formulas': '5' 
    }
    headers = {'key': API_KEY}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and data['items']:
                print(f"Success Formula! First item: {data['items'][0]}")
            else:
                print("Response 200 but no items.")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_yfinance()
    test_evds_formulas()
