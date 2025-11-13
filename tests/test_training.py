"""
Quick Training Test Script
Test all algorithms with optimized parameters
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api/trading"

def train_model(algorithm: str, timesteps: int = 10000):
    """Start training a model"""
    print(f"\n{'='*60}")
    print(f"Starting {algorithm} Training ({timesteps} steps)")
    print(f"{'='*60}\n")

    payload = {
        "algorithm": algorithm,
        "phase": 1,
        "total_timesteps": timesteps,
        "learning_rate": 0.0007 if algorithm == "A2C" else 0.0003,
        "initial_balance": 1000000,
        "commission_rate": 0.001,
        "max_shares_per_trade": 100
    }

    response = requests.post(f"{BASE_URL}/train", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Training started: {result['message']}")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def monitor_training():
    """Monitor training progress"""
    print("\nMonitoring training progress...\n")

    while True:
        try:
            response = requests.get(f"{BASE_URL}/train/status")
            status = response.json()

            if status["is_training"]:
                progress = status["progress"] * 100
                print(f"\rProgress: {progress:.1f}% ({status['current_step']}/{status['total_steps']})", end="")
                time.sleep(2)  # Check every 2 seconds
            else:
                print("\n\n✅ Training completed!")

                # Show metrics if available
                if status.get("metrics"):
                    print("\n📊 Results:")
                    metrics = status["metrics"]
                    print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")
                    print(f"  Cumulative Return: {metrics.get('cumulative_return', 'N/A'):.2%}")
                    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
                    print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}")
                    print(f"  Final Portfolio: ${metrics.get('final_portfolio_value', 'N/A'):,.2f}")

                # Check for errors
                if status.get("error"):
                    print(f"\n❌ Error occurred: {status['error']}")

                break

        except KeyboardInterrupt:
            print("\n\n⚠️ Monitoring interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

def list_models():
    """List all trained models"""
    print(f"\n{'='*60}")
    print("Trained Models")
    print(f"{'='*60}\n")

    response = requests.get(f"{BASE_URL}/models")

    if response.status_code == 200:
        models = response.json()

        if not models:
            print("No models found.")
            return

        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Created: {model['created_at']}")

            if model.get('metrics'):
                m = model['metrics']
                print(f"   Trades: {m.get('total_trades', 'N/A')}")
                print(f"   Return: {m.get('cumulative_return', 'N/A'):.2%}")
                print(f"   Sharpe: {m.get('sharpe_ratio', 'N/A')}")
            print()
    else:
        print(f"❌ Error: {response.text}")

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API is running!")
            return True
    except:
        pass

    print("❌ API is not running!")
    print("\nPlease start the API first:")
    print("  python -m app.main")
    return False

def main():
    """Main test menu"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║         RL Trading Bot - Training Test Menu              ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Check API health
    if not check_api_health():
        return

    while True:
        print(f"\n{'='*60}")
        print("Select an option:")
        print(f"{'='*60}")
        print("1. Quick Test (A2C - 10K steps) - ~2-3 mins")
        print("2. Standard Test (PPO - 50K steps) - ~10-15 mins")
        print("3. Best Performance (SAC - 50K steps) - ~10-15 mins")
        print("4. Compare All (A2C, PPO, SAC - 10K each)")
        print("5. List Trained Models")
        print("6. Check Training Status")
        print("0. Exit")
        print(f"{'='*60}")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            # Quick A2C test
            if train_model("A2C", 10000):
                monitor_training()

        elif choice == "2":
            # Standard PPO test
            if train_model("PPO", 50000):
                monitor_training()

        elif choice == "3":
            # Best SAC test
            if train_model("SAC", 50000):
                monitor_training()

        elif choice == "4":
            # Compare all algorithms
            print("\n🔬 Running comparison test...")
            print("This will train 3 models sequentially (A2C, PPO, SAC)")
            confirm = input("Continue? (y/n): ").strip().lower()

            if confirm == 'y':
                for algo in ["A2C", "PPO", "SAC"]:
                    if train_model(algo, 10000):
                        monitor_training()
                        time.sleep(2)  # Brief pause between models

                print("\n📊 All models trained! Listing results...\n")
                time.sleep(1)
                list_models()

        elif choice == "5":
            # List models
            list_models()

        elif choice == "6":
            # Check status
            response = requests.get(f"{BASE_URL}/train/status")
            status = response.json()

            if status["is_training"]:
                print("\n⏳ Training in progress!")
                monitor_training()
            else:
                print("\n✅ No training in progress")
                if status.get("metrics"):
                    print("\nLast training results:")
                    print(json.dumps(status["metrics"], indent=2))

        elif choice == "0":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
