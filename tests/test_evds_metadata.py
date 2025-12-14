
import requests
from evds import evdsAPI

API_KEY = "tV4qq6RzPr"

def test_metadata():
    print(f"Testing Metadata Access with Key: {API_KEY}")
    
    # 1. Try using the library to get main categories
    try:
        evds = evdsAPI(API_KEY)
        print("\nAttempting to fetch main categories...")
        # The library usually has a method like main_categories or similar, 
        # but let's try a direct request to be sure about the raw response
        url = "https://evds2.tcmb.gov.tr/service/evds/categories"
        headers = {'key': API_KEY}
        response = requests.get(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Key is valid for metadata.")
            print(response.text[:200])
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_metadata()
