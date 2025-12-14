
import requests

API_KEY = "tV4qq6RzPr"
SERIES_CODE = "TP.YSSK.A01" # Policy Rate

def test_evds_url_param():
    print(f"Testing EVDS Web Service with Key in URL...")
    # Construct URL with key parameter
    url = f"https://evds2.tcmb.gov.tr/service/evds/series={SERIES_CODE}&startDate=01-01-2024&endDate=10-01-2024&type=json&key={API_KEY}"
    
    print(f"URL: {url}")
    
    try:
        # Note: Verify SSL is sometimes an issue with corporate proxies, but usually not for TCMB
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_evds_url_param()
