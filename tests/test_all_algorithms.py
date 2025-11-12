"""
Test all three RL algorithms (A2C, PPO, TD3) with optimized hyperparameters
"""
# -*- coding: utf-8 -*-

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from data.data_fetcher import DataFetcher
from data.bist30_symbols import get_symbols
from env.trading_env import make_env
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

# Load data
symbols = get_symbols(phase=1)
fetcher = DataFetcher()

try:
    df = fetcher.load_data('stock_data_with_indicators.csv')
except FileNotFoundError:
    print("Data file not found. Please run data fetching first.")
    exit(1)

train_df, val_df, test_df = fetcher.split_data(df)

print("=" * 80)
print("TESTING ALL THREE ALGORITHMS WITH OPTIMIZED HYPERPARAMETERS")
print("=" * 80)
print(f"\nDataset: {len(train_df)} training samples")
print(f"Initial Balance: ₺100,000")
print(f"Timesteps: 50,000")
print(f"Learning Rate: 0.0007")
print("\n" + "=" * 80)

results = {}

# ============================================================================
# TEST 1: PPO (Recommended)
# ============================================================================
print("\n1️⃣  TESTING PPO (Proximal Policy Optimization)")
print("-" * 80)
print("Status: ⭐ Recommended - Most stable for single-threaded training")
print("\nHyperparameters:")
print("  - n_steps: 2048")
print("  - batch_size: 64")
print("  - n_epochs: 10")
print("  - ent_coef: 0.01 (exploration bonus)")
print("\nTraining...")

env = make_env(df=train_df, initial_balance=100000, commission_rate=0.001, max_shares_per_trade=100)
env = DummyVecEnv([lambda: env])

model_ppo = PPO(
    'MlpPolicy', env, learning_rate=0.0007,
    n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.01, verbose=0
)
model_ppo.learn(total_timesteps=50000)

# Evaluate
test_env = make_env(df=test_df, initial_balance=100000, commission_rate=0.001, max_shares_per_trade=100)
obs, info = test_env.reset()
done = False
while not done:
    action, _ = model_ppo.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

metrics_ppo = test_env.get_metrics()
results['PPO'] = metrics_ppo

print(f"\n✅ PPO Results:")
print(f"   Trades: {metrics_ppo['total_trades']}")
print(f"   Return: {metrics_ppo['cumulative_return']:.2%}")
print(f"   Sharpe: {metrics_ppo['sharpe_ratio']:.4f}" if metrics_ppo['sharpe_ratio'] else "   Sharpe: N/A")
print(f"   Final Portfolio: ₺{metrics_ppo['final_portfolio_value']:,.2f}")

# ============================================================================
# TEST 2: A2C (Synchronous Actor-Critic)
# ============================================================================
print("\n" + "=" * 80)
print("\n2️⃣  TESTING A2C (Advantage Actor-Critic)")
print("-" * 80)
print("Status: ⚠️  Requires careful tuning - less stable than PPO")
print("\nHyperparameters:")
print("  - n_steps: 512 (more frequent updates)")
print("  - normalize_advantage: True")
print("  - use_rms_prop: True (as per original paper)")
print("  - ent_coef: 0.01 (exploration bonus)")
print("\nTraining...")

env = make_env(df=train_df, initial_balance=100000, commission_rate=0.001, max_shares_per_trade=100)
env = DummyVecEnv([lambda: env])

model_a2c = A2C(
    'MlpPolicy', env, learning_rate=0.0007,
    n_steps=512, gamma=0.99, gae_lambda=0.95,
    ent_coef=0.01, vf_coef=0.5,
    normalize_advantage=True, use_rms_prop=True,
    verbose=0
)
model_a2c.learn(total_timesteps=50000)

# Evaluate
test_env = make_env(df=test_df, initial_balance=100000, commission_rate=0.001, max_shares_per_trade=100)
obs, info = test_env.reset()
done = False
while not done:
    action, _ = model_a2c.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

metrics_a2c = test_env.get_metrics()
results['A2C'] = metrics_a2c

print(f"\n✅ A2C Results:")
print(f"   Trades: {metrics_a2c['total_trades']}")
print(f"   Return: {metrics_a2c['cumulative_return']:.2%}")
print(f"   Sharpe: {metrics_a2c['sharpe_ratio']:.4f}" if metrics_a2c['sharpe_ratio'] else "   Sharpe: N/A")
print(f"   Final Portfolio: ₺{metrics_a2c['final_portfolio_value']:,.2f}")

# ============================================================================
# TEST 3: TD3 (Twin Delayed DDPG)
# ============================================================================
print("\n" + "=" * 80)
print("\n3️⃣  TESTING TD3 (Twin Delayed Deep Deterministic Policy Gradient)")
print("-" * 80)
print("Status: 🔬 Advanced - Uses experience replay and delayed updates")
print("\nHyperparameters:")
print("  - buffer_size: 100,000 (experience replay)")
print("  - batch_size: 256")
print("  - action_noise: NormalActionNoise(sigma=0.1)")
print("  - policy_delay: 2 (delayed policy updates)")
print("\nTraining...")

env = make_env(df=train_df, initial_balance=100000, commission_rate=0.001, max_shares_per_trade=100)
env = DummyVecEnv([lambda: env])

# Add action noise for exploration
action_noise = NormalActionNoise(
    mean=np.zeros(env.action_space.shape[0]),
    sigma=0.1 * np.ones(env.action_space.shape[0])
)

model_td3 = TD3(
    'MlpPolicy', env, learning_rate=0.0007,
    buffer_size=100000, learning_starts=1000,
    batch_size=256, tau=0.005, gamma=0.99,
    train_freq=1, gradient_steps=1,
    action_noise=action_noise, policy_delay=2,
    target_policy_noise=0.2, target_noise_clip=0.5,
    verbose=0
)
model_td3.learn(total_timesteps=50000)

# Evaluate
test_env = make_env(df=test_df, initial_balance=100000, commission_rate=0.001, max_shares_per_trade=100)
obs, info = test_env.reset()
done = False
while not done:
    action, _ = model_td3.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

metrics_td3 = test_env.get_metrics()
results['TD3'] = metrics_td3

print(f"\n✅ TD3 Results:")
print(f"   Trades: {metrics_td3['total_trades']}")
print(f"   Return: {metrics_td3['cumulative_return']:.2%}")
print(f"   Sharpe: {metrics_td3['sharpe_ratio']:.4f}" if metrics_td3['sharpe_ratio'] else "   Sharpe: N/A")
print(f"   Final Portfolio: ₺{metrics_td3['final_portfolio_value']:,.2f}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 ALGORITHM COMPARISON SUMMARY")
print("=" * 80)
print(f"\n{'Algorithm':<10} {'Trades':<10} {'Return':<15} {'Sharpe':<12} {'Final Value':<15}")
print("-" * 80)

for algo, metrics in results.items():
    trades = metrics['total_trades']
    ret = f"{metrics['cumulative_return']:.2%}"
    sharpe = f"{metrics['sharpe_ratio']:.4f}" if metrics['sharpe_ratio'] else "N/A"
    final = f"₺{metrics['final_portfolio_value']:,.2f}"
    print(f"{algo:<10} {trades:<10} {ret:<15} {sharpe:<12} {final:<15}")

# Find best algorithm
best_algo = max(results.items(), key=lambda x: x[1]['cumulative_return'])
print("\n" + "=" * 80)
print(f"🏆 WINNER: {best_algo[0]} with {best_algo[1]['cumulative_return']:.2%} return")
print("=" * 80)

# Check if all algorithms are trading
trading_algos = [algo for algo, metrics in results.items() if metrics['total_trades'] > 0]
print(f"\n✅ {len(trading_algos)}/3 algorithms successfully learned to trade")

if len(trading_algos) < 3:
    non_trading = [algo for algo, metrics in results.items() if metrics['total_trades'] == 0]
    print(f"⚠️  Algorithms not trading: {', '.join(non_trading)}")
else:
    print("🎉 All algorithms are working correctly!")
