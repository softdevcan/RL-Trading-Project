"""
Test Daily Trading Backend
Quick test script for daily trading endpoints
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/api/trading"


def test_daily_decision():
    """Test daily decision endpoint"""
    print("=" * 60)
    print("Testing Daily Decision Endpoint")
    print("=" * 60)

    # Test with a past date (where we have data)
    test_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    payload = {
        "model_name": "ppo_phase1_20241113_143022",  # Update with your actual model name
        "balance": 1000000,
        "shares": {
            "ASELS.IS": 0,
            "THYAO.IS": 0,
            "EREGL.IS": 0,
            "KCHOL.IS": 0,
            "SAHOL.IS": 0
        },
        "risk_mode": "moderate",
        "max_shares_per_trade": 100,
        "date": test_date
    }

    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(
            f"{BASE_URL}/daily-decision",
            json=payload,
            timeout=120  # 2 minutes timeout
        )

        print(f"\nResponse status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"\nDate: {result['date']}")
            print(f"Total Trades: {result['summary']['total_trades']}")
            print(f"Daily Return: {result['summary']['daily_return_pct']:.2f}%")
            print(f"\nPortfolio Before: ₺{result['portfolio_before']['portfolio_value']:,.2f}")
            print(f"Portfolio After:  ₺{result['portfolio_after']['portfolio_value']:,.2f}")

            print("\n--- Decisions ---")
            for decision in result['decisions']:
                print(f"\n{decision['symbol']}: {decision['action']}")
                print(f"  Signal: {decision['raw_signal']:.2f}")
                print(f"  Shares: {decision['shares']}")
                print(f"  Price: ₺{decision['price']:.2f}")
                print(f"  Reason: {decision['reason']}")
                print(f"  Executed: {decision['executed']}")

        else:
            print(f"\n❌ FAILED: {response.status_code}")
            print(response.text)

    except requests.exceptions.Timeout:
        print("\n❌ Request timed out (> 2 minutes)")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def test_latest_portfolio():
    """Test latest portfolio endpoint"""
    print("\n" + "=" * 60)
    print("Testing Latest Portfolio Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/latest-portfolio")

        print(f"\nResponse status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"\nBalance: ₺{result['balance']:,.2f}")
            print(f"Portfolio Value: ₺{result['portfolio_value']:,.2f}")
            print(f"Date: {result['date']}")
            print(f"\nShares:")
            for symbol, shares in result['shares'].items():
                print(f"  {symbol}: {shares}")
        else:
            print(f"\n❌ FAILED: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def test_portfolio_history():
    """Test portfolio history endpoint"""
    print("\n" + "=" * 60)
    print("Testing Portfolio History Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/portfolio-history?days=7")

        print(f"\nResponse status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"\nHistory entries: {len(result['dates'])}")

            if result['dates']:
                print("\nLast 3 entries:")
                for i in range(min(3, len(result['dates']))):
                    idx = -(i + 1)
                    print(f"\n{result['dates'][idx]}:")
                    print(f"  Portfolio: ₺{result['portfolio_values'][idx]:,.2f}")
                    print(f"  Return: {result['daily_returns'][idx]:.2f}%")
                    print(f"  Balance: ₺{result['balances'][idx]:,.2f}")
            else:
                print("\nNo history entries found")

        else:
            print(f"\n❌ FAILED: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def list_available_models():
    """List available models"""
    print("\n" + "=" * 60)
    print("Listing Available Models")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/models")

        if response.status_code == 200:
            models = response.json()
            print(f"\n✅ Found {len(models)} models:")

            for i, model in enumerate(models, 1):
                print(f"\n{i}. {model['name']}")
                if 'metrics' in model and 'algorithm' in model['metrics']:
                    print(f"   Algorithm: {model['metrics']['algorithm']}")
                    print(f"   Sharpe: {model['metrics'].get('sharpe_ratio', 'N/A')}")
        else:
            print(f"\n❌ FAILED: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" DAILY TRADING BACKEND TEST")
    print("=" * 60)
    print("\nMake sure FastAPI server is running on http://localhost:8000")
    input("\nPress Enter to continue...")

    # List available models first
    list_available_models()

    # Test latest portfolio
    test_latest_portfolio()

    # Test portfolio history
    test_portfolio_history()

    # Test daily decision (main test)
    print("\n\n⚠️  IMPORTANT: Daily decision test will:")
    print("  1. Fetch real market data from yfinance")
    print("  2. Load the model and run inference")
    print("  3. May take 30-60 seconds")
    input("\nPress Enter to proceed with daily decision test...")

    test_daily_decision()

    print("\n" + "=" * 60)
    print(" TEST COMPLETED")
    print("=" * 60)
