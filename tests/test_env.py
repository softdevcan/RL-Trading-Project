"""
Quick test to see if the environment accepts actions and executes trades
"""
# -*- coding: utf-8 -*-

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from data.data_fetcher import DataFetcher
from data.bist30_symbols import get_symbols
from data.technical_indicators import add_indicators_to_multi_symbol_df
from env.trading_env import make_env

# Load data
symbols = get_symbols(phase=1)
fetcher = DataFetcher()

try:
    df = fetcher.load_data('stock_data_with_indicators.csv')
except FileNotFoundError:
    print("Data file not found. Please run data fetching first.")
    exit(1)

train_df, val_df, test_df = fetcher.split_data(df)

# Create environment
env = make_env(
    df=train_df,
    initial_balance=100000,
    commission_rate=0.001,
    max_shares_per_trade=100
)

print("=" * 60)
print("ENVIRONMENT TEST")
print("=" * 60)

# Reset environment
obs, info = env.reset()
print(f"\nInitial state:")
print(f"  Balance: ₺{info['balance']:,.2f}")
print(f"  Portfolio Value: ₺{info['portfolio_value']:,.2f}")
print(f"  Observation shape: {obs.shape}")
print(f"  Observation range: [{obs.min():.4f}, {obs.max():.4f}]")

# Test with manual actions
print("\n" + "=" * 60)
print("TESTING MANUAL ACTIONS")
print("=" * 60)

# Test 1: Try to buy stock 0 (max amount)
print("\nTest 1: Action = [1.0, 0, 0, 0, 0] (buy max shares of stock 0)")
action = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
print(f"  Balance after: ₺{info['balance']:,.2f}")
print(f"  Portfolio Value: ₺{info['portfolio_value']:,.2f}")
print(f"  Shares owned: {info['shares_owned']}")
print(f"  Reward: {reward:.6f}")
print(f"  Total trades: {len(env.trades_history)}")
if env.trades_history:
    print(f"  Last trade: {env.trades_history[-1]}")

# Test 2: Try to buy multiple stocks
print("\nTest 2: Action = [0.5, 0.5, 0.3, 0, 0] (buy different amounts)")
action = np.array([0.5, 0.5, 0.3, 0.0, 0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
print(f"  Balance after: ₺{info['balance']:,.2f}")
print(f"  Portfolio Value: ₺{info['portfolio_value']:,.2f}")
print(f"  Shares owned: {info['shares_owned']}")
print(f"  Reward: {reward:.6f}")
print(f"  Total trades: {len(env.trades_history)}")
if len(env.trades_history) > 1:
    print(f"  Last trade: {env.trades_history[-1]}")

# Test 3: Run a few random steps
print("\n" + "=" * 60)
print("TESTING RANDOM ACTIONS (10 steps)")
print("=" * 60)

env.reset()
for i in range(10):
    # Random action in [-1, 1]
    action = np.random.uniform(-1, 1, size=5).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if i < 3 or len(env.trades_history) > 0:  # Show first 3 or if trades happen
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Balance: ₺{info['balance']:,.2f}")
        print(f"  Total trades so far: {len(env.trades_history)}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total trades executed: {len(env.trades_history)}")
if env.trades_history:
    print("\nAll trades:")
    for trade in env.trades_history:
        print(f"  {trade}")
else:
    print("\n⚠️ WARNING: No trades were executed!")
    print("This means the environment is not accepting actions properly.")

print("\nEnvironment symbols:", env.symbols)
print("Max shares per trade:", env.max_shares_per_trade)
print("Action space:", env.action_space)
