"""
A/B Testing: Simple Reward vs PSR Reward
Faz 2 - Sprint 2

Bu script, iki reward fonksiyonunu karsilastirir:
1. Baseline (Simple Reward - Faz 1)
2. PSR Reward (Faz 2)

Ayrica Faz 1 (56 features) vs Faz 2 (97 features) karsilastirmasi yapar.

Metrikler: Sharpe Ratio, MDD, Total Return, Total Trades

Referans: Ansari et al. (2024)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from data.data_fetcher import DataFetcher
from data.bist30_symbols import get_symbols
from data.technical_indicators import add_indicators_to_multi_symbol_df
from data.fundamental_fetcher import FundamentalDataFetcher
from data.macro_fetcher import MacroDataFetcher
from env.trading_env import make_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ab_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_phase2_data():
    """Load all Phase 2 data"""
    logger.info("Loading Phase 2 data...")

    # Market data
    symbols = get_symbols(phase=2)
    fetcher = DataFetcher()

    try:
        df = fetcher.load_data('stock_data_with_indicators.csv')
        logger.info(f"Loaded market data: {len(df)} rows")
    except:
        logger.info("Fetching market data...")
        df = fetcher.fetch_stock_data(symbols)
        df = fetcher.clean_data(df)
        df = add_indicators_to_multi_symbol_df(df)
        fetcher.save_data(df, 'stock_data_with_indicators.csv')

    # Fundamental data
    fund_fetcher = FundamentalDataFetcher()
    try:
        fundamental_df = fund_fetcher.load_data('fundamental_data.csv')
        logger.info(f"Loaded fundamental data: {len(fundamental_df)} symbols")
    except:
        logger.info("Fetching fundamental data...")
        fundamental_df = fund_fetcher.fetch_fundamental_data(symbols)
        fund_fetcher.save_data(fundamental_df, 'fundamental_data.csv')

    # Macro data
    TCMB_API_KEY = os.getenv('TCMB_API_KEY', 'tV4qq6RzPr')
    macro_fetcher = MacroDataFetcher(api_key=TCMB_API_KEY)
    try:
        macro_df = macro_fetcher.load_data('macro_data.csv')
        logger.info(f"Loaded macro data: {len(macro_df)} rows")
    except:
        logger.info("Fetching macro data...")
        macro_df = macro_fetcher.fetch_macro_data()
        macro_fetcher.save_data(macro_df, 'macro_data.csv')

    # Split data
    train_df, val_df, test_df = fetcher.split_data(df)

    return train_df, val_df, test_df, fundamental_df, macro_df


def train_and_evaluate(
    train_df,
    test_df,
    fundamental_df,
    macro_df,
    phase: int,
    reward_type: str,
    reward_weights: Dict = None,
    timesteps: int = 50000
) -> Dict:
    """
    Train and evaluate a model

    Args:
        train_df: Training data
        test_df: Test data
        fundamental_df: Fundamental data (Phase 2 only)
        macro_df: Macro data (Phase 2 only)
        phase: 1 or 2
        reward_type: 'simple' or 'psr'
        reward_weights: PSR weights (optional)
        timesteps: Training timesteps

    Returns:
        Dict with metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: Phase {phase}, Reward: {reward_type.upper()}")
    logger.info(f"{'='*80}")

    # Create training environment
    def make_train_env():
        env_kwargs = {
            'initial_balance': 1_000_000,
            'phase': phase,
            'reward_type': reward_type
        }

        if phase == 2:
            env_kwargs['fundamental_df'] = fundamental_df
            env_kwargs['macro_df'] = macro_df

        if reward_weights:
            env_kwargs['reward_weights'] = reward_weights

        return make_env(train_df, **env_kwargs)

    train_env = DummyVecEnv([make_train_env])

    # Train model
    logger.info(f"Training PPO for {timesteps} timesteps...")

    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model.learn(total_timesteps=timesteps)

    # Evaluate on test set
    logger.info("Evaluating on test set...")

    def make_test_env():
        env_kwargs = {
            'initial_balance': 1_000_000,
            'phase': phase,
            'reward_type': reward_type
        }

        if phase == 2:
            env_kwargs['fundamental_df'] = fundamental_df
            env_kwargs['macro_df'] = macro_df

        if reward_weights:
            env_kwargs['reward_weights'] = reward_weights

        return make_env(test_df, **env_kwargs)

    test_env = DummyVecEnv([make_test_env])

    obs = test_env.reset()
    done = [False]
    portfolio_values = []
    trades_count = 0

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)

        # VecEnv returns lists/arrays
        current_done = done[0] if isinstance(done, (list, np.ndarray)) else done
        current_info = info[0] if isinstance(info, list) else info

        portfolio_values.append(current_info['portfolio_value'])
        trades_count = current_info.get('total_trades', trades_count)
        done = [current_done]  # Update for next iteration

    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)

    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax
    mdd = np.min(drawdown)

    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    metrics = {
        'phase': phase,
        'reward_type': reward_type,
        'sharpe_ratio': float(sharpe_ratio),
        'mdd': float(mdd),
        'total_return': float(total_return),
        'final_value': float(portfolio_values[-1]),
        'total_trades': trades_count,
        'portfolio_values': portfolio_values
    }

    logger.info(f"\nResults:")
    logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    logger.info(f"  MDD: {mdd:.4f} ({mdd*100:.2f}%)")
    logger.info(f"  Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
    logger.info(f"  Final Value: TRY {portfolio_values[-1]:,.0f}")
    logger.info(f"  Total Trades: {trades_count}")

    return metrics


def plot_comparison(results: Dict[str, Dict]):
    """
    Plot comparison charts

    Args:
        results: Dict of experiment results
    """
    logger.info("\nGenerating comparison plots...")

    results_dir = project_root / 'results' / 'ab_test'
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Portfolio value over time
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, metrics in results.items():
        portfolio_values = metrics['portfolio_values']
        ax.plot(portfolio_values, label=name, linewidth=2)

    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value (TRY)', fontsize=12)
    ax.set_title('Portfolio Value Over Time - A/B Test Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'portfolio_comparison.png', dpi=300)
    logger.info(f"  - Portfolio comparison saved")

    # 2. Metrics comparison (bar chart)
    metrics_names = ['Sharpe Ratio', 'Max Drawdown (%)', 'Total Return (%)', 'Total Trades']
    n_metrics = len(metrics_names)
    n_experiments = len(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric_name in enumerate(metrics_names):
        ax = axes[i]

        if metric_name == 'Sharpe Ratio':
            values = [m['sharpe_ratio'] for m in results.values()]
        elif metric_name == 'Max Drawdown (%)':
            values = [m['mdd'] * 100 for m in results.values()]
        elif metric_name == 'Total Return (%)':
            values = [m['total_return'] * 100 for m in results.values()]
        else:  # Total Trades
            values = [m['total_trades'] for m in results.values()]

        x_pos = np.arange(n_experiments)
        bars = ax.bar(x_pos, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:n_experiments])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(results.keys(), rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / 'metrics_comparison.png', dpi=300)
    logger.info(f"  - Metrics comparison saved")

    # 3. Results table
    results_df = pd.DataFrame({
        'Experiment': list(results.keys()),
        'Phase': [m['phase'] for m in results.values()],
        'Reward': [m['reward_type'] for m in results.values()],
        'Sharpe': [m['sharpe_ratio'] for m in results.values()],
        'MDD (%)': [m['mdd'] * 100 for m in results.values()],
        'Return (%)': [m['total_return'] * 100 for m in results.values()],
        'Final Value (TRY)': [m['final_value'] for m in results.values()],
        'Trades': [m['total_trades'] for m in results.values()]
    })

    results_df.to_csv(results_dir / 'ab_test_results.csv', index=False)
    logger.info(f"  - Results table saved")

    # Print table to console
    logger.info("\n" + "="*80)
    logger.info("A/B TEST RESULTS SUMMARY")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))
    logger.info("\n" + "="*80)


def run_ab_test(timesteps: int = 50000):
    """
    Run A/B test

    Args:
        timesteps: Training timesteps per experiment
    """
    logger.info("="*80)
    logger.info("A/B TESTING: SIMPLE vs PSR REWARD")
    logger.info("="*80)

    # Load data
    train_df, val_df, test_df, fundamental_df, macro_df = load_phase2_data()

    results = {}

    # Experiment 1: Phase 1 + Simple Reward (Baseline)
    logger.info("\n[1/4] Experiment 1: Phase 1 + Simple Reward (Baseline)")
    results['Phase1-Simple'] = train_and_evaluate(
        train_df, test_df, None, None,
        phase=1,
        reward_type='simple',
        timesteps=timesteps
    )

    # Experiment 2: Phase 2 + Simple Reward
    logger.info("\n[2/4] Experiment 2: Phase 2 + Simple Reward")
    results['Phase2-Simple'] = train_and_evaluate(
        train_df, test_df, fundamental_df, macro_df,
        phase=2,
        reward_type='simple',
        timesteps=timesteps
    )

    # Experiment 3: Phase 1 + PSR Reward
    logger.info("\n[3/4] Experiment 3: Phase 1 + PSR Reward")
    results['Phase1-PSR'] = train_and_evaluate(
        train_df, test_df, None, None,
        phase=1,
        reward_type='psr',
        timesteps=timesteps
    )

    # Experiment 4: Phase 2 + PSR Reward (Best configuration)
    logger.info("\n[4/4] Experiment 4: Phase 2 + PSR Reward (Best)")
    results['Phase2-PSR'] = train_and_evaluate(
        train_df, test_df, fundamental_df, macro_df,
        phase=2,
        reward_type='psr',
        timesteps=timesteps
    )

    # Plot comparison
    plot_comparison(results)

    logger.info("\n" + "="*80)
    logger.info("[SUCCESS] A/B Test complete! Check results/ab_test/ for outputs.")
    logger.info("="*80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='A/B test Simple vs PSR reward')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps per experiment (default: 50000)')
    args = parser.parse_args()

    results = run_ab_test(timesteps=args.timesteps)
