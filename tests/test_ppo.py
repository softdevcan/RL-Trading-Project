"""
Quick test to verify PPO algorithm learns to trade
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
env = DummyVecEnv([lambda: env])

print("=" * 60)
print("PPO ALGORITHM TEST")
print("=" * 60)
print(f"\nTraining with optimized PPO hyperparameters...")
print(f"Timesteps: 50,000")
print(f"Initial Balance: ₺100,000")
print(f"Learning Rate: 0.0007")
print(f"n_steps: 2048")
print(f"batch_size: 64")
print(f"n_epochs: 10")
print(f"ent_coef: 0.01 (for exploration)")

# Create PPO model with optimized hyperparameters
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=0.0007,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Entropy coefficient for exploration
    verbose=1
)

print("\n" + "=" * 60)
print("TRAINING...")
print("=" * 60)

# Train model
model.learn(total_timesteps=50000)

print("\n" + "=" * 60)
print("EVALUATING ON TEST SET...")
print("=" * 60)

# Evaluate on test set
test_env = make_env(
    df=test_df,
    initial_balance=100000,
    commission_rate=0.001,
    max_shares_per_trade=100
)

obs, info = test_env.reset()
done = False
step = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    step += 1

print(f"\nEvaluation completed in {step} steps")

# Get metrics
metrics = test_env.get_metrics()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total Trades: {metrics['total_trades']}")
print(f"Final Portfolio Value: ₺{metrics['final_portfolio_value']:,.2f}")
print(f"Cumulative Return: {metrics['cumulative_return']:.4%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}" if metrics['sharpe_ratio'] else "Sharpe Ratio: N/A")
print(f"Max Drawdown: {metrics['max_drawdown']:.4%}")

if metrics['total_trades'] > 0:
    print("\n✅ SUCCESS: PPO learned to trade!")
    print(f"\nAll trades executed:")
    for trade in test_env.trades_history[:10]:  # Show first 10 trades
        print(f"  {trade}")
    if len(test_env.trades_history) > 10:
        print(f"  ... and {len(test_env.trades_history) - 10} more trades")
else:
    print("\n❌ FAILURE: PPO still not trading")
