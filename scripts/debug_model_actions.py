"""
Debug Model Actions
Test what actions the model is actually producing
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, TD3, SAC
from env.trading_env import make_env
import pickle

def debug_model_actions(model_name: str):
    """
    Debug what actions a model produces
    """
    print(f"\n{'='*80}")
    print(f"DEBUGGING MODEL: {model_name}")
    print(f"{'='*80}")

    # Load test data
    data_file = 'data/preprocessed/train_val_test_data.pkl'
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    test_df = data['test']

    # Load model
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model not found: {model_path}")
        return

    # Determine model type
    model_name_lower = model_name.lower()
    if "ppo" in model_name_lower:
        model = PPO.load(model_path)
    elif "a2c" in model_name_lower:
        model = A2C.load(model_path)
    elif "td3" in model_name_lower:
        model = TD3.load(model_path)
    elif "sac" in model_name_lower:
        model = SAC.load(model_path)
    else:
        print(f"❌ Unknown model type: {model_name}")
        return

    print(f"✅ Model loaded: {type(model).__name__}")

    # Create environment
    env = make_env(test_df, initial_balance=1_000_000, max_shares_per_trade=100)

    # Reset environment
    obs, _ = env.reset()

    print(f"\n📊 Environment Info:")
    print(f"  Stocks: {env.symbols}")
    print(f"  Max shares per trade: {env.max_shares_per_trade}")

    # Sample actions for first 50 steps
    print(f"\n🎯 Sampling Actions (first 50 steps):")
    print(f"{'Step':<6} {'Raw Actions':<60} {'Scaled Actions (shares)':<40}")
    print("-" * 110)

    actions_raw = []
    actions_scaled = []

    for step in range(50):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        actions_raw.append(action.copy())

        # Scale action (same as environment does)
        scaled = np.round(action * env.max_shares_per_trade).astype(int)
        scaled = np.clip(scaled, -env.max_shares_per_trade, env.max_shares_per_trade)
        actions_scaled.append(scaled.copy())

        # Print every 5 steps
        if step % 5 == 0:
            raw_str = ', '.join([f'{a:+.3f}' for a in action])
            scaled_str = ', '.join([f'{s:+4d}' for s in scaled])
            print(f"{step:<6} [{raw_str}] [{scaled_str}]")

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    # Statistical analysis
    actions_raw = np.array(actions_raw)
    actions_scaled = np.array(actions_scaled)

    print(f"\n📈 Statistical Analysis:")
    print(f"{'Metric':<30} {'Stock 0':<12} {'Stock 1':<12} {'Stock 2':<12} {'Stock 3':<12} {'Stock 4':<12}")
    print("-" * 90)

    # Raw actions
    print(f"{'Raw Action Mean:':<30} " + ' '.join([f"{actions_raw[:, i].mean():+11.4f}" for i in range(5)]))
    print(f"{'Raw Action Std:':<30} " + ' '.join([f"{actions_raw[:, i].std():11.4f}" for i in range(5)]))
    print(f"{'Raw Action Min:':<30} " + ' '.join([f"{actions_raw[:, i].min():+11.4f}" for i in range(5)]))
    print(f"{'Raw Action Max:':<30} " + ' '.join([f"{actions_raw[:, i].max():+11.4f}" for i in range(5)]))

    print()

    # Scaled actions
    print(f"{'Scaled Action Mean:':<30} " + ' '.join([f"{actions_scaled[:, i].mean():+11.1f}" for i in range(5)]))
    print(f"{'Scaled Action Std:':<30} " + ' '.join([f"{actions_scaled[:, i].std():11.1f}" for i in range(5)]))
    print(f"{'Scaled Action Min:':<30} " + ' '.join([f"{actions_scaled[:, i].min():+11d}" for i in range(5)]))
    print(f"{'Scaled Action Max:':<30} " + ' '.join([f"{actions_scaled[:, i].max():+11d}" for i in range(5)]))

    print()

    # Action distribution
    print(f"{'Positive actions (BUY):':<30} " + ' '.join([f"{(actions_scaled[:, i] > 0).sum():11d}" for i in range(5)]))
    print(f"{'Negative actions (SELL):':<30} " + ' '.join([f"{(actions_scaled[:, i] < 0).sum():11d}" for i in range(5)]))
    print(f"{'Zero actions (HOLD):':<30} " + ' '.join([f"{(actions_scaled[:, i] == 0).sum():11d}" for i in range(5)]))

    print(f"\n💡 Interpretation:")
    if actions_raw.mean() > 0.5:
        print("  ⚠️  Model is HEAVILY BIASED towards BUY (mean > 0.5)")
        print("  → This explains why all trades are BUY with max shares")
    elif actions_raw.std() < 0.1:
        print("  ⚠️  Model has LOW VARIANCE (std < 0.1)")
        print("  → Model is not exploring different action magnitudes")
    else:
        print("  ✅ Model seems to have reasonable action distribution")

    # Check if model always outputs same action
    unique_actions = len(np.unique(actions_scaled, axis=0))
    print(f"\n  Unique action combinations: {unique_actions} / {len(actions_scaled)}")
    if unique_actions == 1:
        print("  ⚠️  MODEL IS STUCK! Always producing the same action")
    elif unique_actions < len(actions_scaled) / 10:
        print("  ⚠️  Model has very low action diversity")

if __name__ == "__main__":
    import sys

    # Find available models
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found")
        sys.exit(1)

    model_files = [f.replace('.zip', '') for f in os.listdir(models_dir) if f.endswith('.zip')]

    if not model_files:
        print("❌ No trained models found")
        sys.exit(1)

    print(f"\n🤖 Found {len(model_files)} models:")
    for i, model in enumerate(model_files, 1):
        print(f"  {i}. {model}")

    # Debug each model
    for model_name in model_files:
        debug_model_actions(model_name)
        print("\n" + "="*80 + "\n")
