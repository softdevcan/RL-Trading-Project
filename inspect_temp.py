import json


file_path = 'results/ppo_phase2_20251214_224452_metrics.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    print("Keys:", data.keys())
    
    if 'trades' in data and len(data['trades']) > 0:
        print("First trade:", data['trades'][0])
        print("Last trade:", data['trades'][-1])
    elif 'portfolio_history' in data:
        print(f"Portfolio history length: {len(data['portfolio_history'])}")

    
except Exception as e:
    print(f"Error: {e}")
