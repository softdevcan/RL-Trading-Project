
import requests

API_KEY = "tV4qq6RzPr"
SERIES_CODE = "TP.FE.OKTG01"  # Dokümanda kullanılan seri

def test_exact_php_format():
    # PHP örneğiyle birebir aynı URL
    url = f"https://evds2.tcmb.gov.tr/service/evds/series={SERIES_CODE}&startDate=31-12-2023&endDate=31-12-2024&type=xml&aggregationTypes=avg&frequency=1"
    
    print(f"Testing with API Key: {API_KEY}")
    print(f"URL: {url}\n")
    
    # Format 1: Dict formatında (requests otomatik dönüştürür)
    print("1. Dict format: {'key': API_KEY}")
    try:
        headers = {'key': API_KEY}
        resp = requests.get(url, headers=headers)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   SUCCESS!")
            print(f"   Response: {resp.text[:200]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Format 2: Tuple list formatında (daha kesin kontrol)
    print("\n2. Tuple list format: [('key', API_KEY)]")
    try:
        resp = requests.get(url, headers=[('key', API_KEY)])
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   SUCCESS!")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Format 3: Manuel string header
    print("\n3. Manual prepared header")
    try:
        session = requests.Session()
        session.headers.update({'key': API_KEY})
        resp = session.get(url)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   SUCCESS!")
    except Exception as e:
        print(f"   Error: {e}")

    # Debug: Gerçek request header'ını göster
    print("\n--- DEBUG: Actual request headers ---")
    import requests
    req = requests.Request('GET', url, headers={'key': API_KEY})
    prepared = req.prepare()
    print(f"Headers being sent: {prepared.headers}")

if __name__ == "__main__":
    test_exact_php_format()
