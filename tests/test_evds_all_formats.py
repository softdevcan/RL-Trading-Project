
import requests

API_KEY = "tV4qq6RzPr"
SERIES_CODE = "TP.DK.USD.A.YTL"  # USD/TRY - genellikle erişilebilir

def test_all_formats():
    base_url = f"https://evds2.tcmb.gov.tr/service/evds/series={SERIES_CODE}"
    params = {
        'startDate': '01-01-2024',
        'endDate': '10-01-2024',
        'type': 'json'
    }
    
    # Format 1: header'da 'key'
    print("1. Header: {'key': API_KEY}")
    try:
        resp = requests.get(base_url, params=params, headers={'key': API_KEY})
        print(f"   Status: {resp.status_code}")
    except Exception as e: 
        print(f"   Error: {e}")
    
    # Format 2: header'da 'Authorization'
    print("\n2. Header: {'Authorization': API_KEY}")
    try:
        resp = requests.get(base_url, params=params, headers={'Authorization': API_KEY})
        print(f"   Status: {resp.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Format 3: header'da 'Authorization: Bearer'
    print("\n3. Header: {'Authorization': 'Bearer ' + API_KEY}")
    try:
        resp = requests.get(base_url, params=params, headers={'Authorization': f'Bearer {API_KEY}'})
        print(f"   Status: {resp.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Format 4: URL parametresi olarak 'key'
    print("\n4. URL Parameter: ?key=...")
    try:
        params_with_key = params.copy()
        params_with_key['key'] = API_KEY
        resp = requests.get(base_url, params=params_with_key)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   SUCCESS! Bu format çalıştı!")
            print(f"   Data: {resp.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Format 5: header'da 'X-API-Key'
    print("\n5. Header: {'X-API-Key': API_KEY}")
    try:
        resp = requests.get(base_url, params=params, headers={'X-API-Key': API_KEY})
        print(f"   Status: {resp.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Format 6: Direkt URL constructor (evds2 formatı)
    print("\n6. Direct URL with aggregationTypes")
    try:
        url = f"https://evds2.tcmb.gov.tr/service/evds/series={SERIES_CODE}&startDate=01-01-2024&endDate=10-01-2024&type=json&key={API_KEY}&aggregationTypes=avg"
        resp = requests.get(url)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   SUCCESS! Bu format çalıştı!")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_all_formats()
