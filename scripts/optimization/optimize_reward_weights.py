"""
Optuna Hyperparameter Optimization for PSR Reward Weights
Faz 2 - Sprint 2

Bu script, PSR reward function'daki w1-w5 agirliklarini optimize eder.
Amaç: Sharpe Ratio maksimize, MDD minimize et.

Referans: Ansari et al. (2024)
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour
)
import plotly.io as pio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
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
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QuickTrainingCallback(BaseCallback):
    """
    Callback for quick training (early stopping after N steps for optimization)
    """
    def __init__(self, max_steps: int = 10000, verbose=0):
        super().__init__(verbose)
        self.max_steps = max_steps
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        return self.step_count < self.max_steps


def load_phase2_data():
    """
    Load all Phase 2 data (market, fundamental, macro)
    """
    logger.info("Loading Phase 2 data...")

    # 1. Market data
    symbols = get_symbols(phase=2)
    fetcher = DataFetcher()

    try:
        df = fetcher.load_data('stock_data_with_indicators.csv')
        logger.info(f"Loaded market data from cache: {len(df)} rows")
    except:
        logger.info("Fetching market data from Yahoo Finance...")
        df = fetcher.fetch_stock_data(symbols)
        df = fetcher.clean_data(df)
        df = add_indicators_to_multi_symbol_df(df)
        fetcher.save_data(df, 'stock_data_with_indicators.csv')

    # 2. Fundamental data
    fund_fetcher = FundamentalDataFetcher()
    try:
        fundamental_df = fund_fetcher.load_data('fundamental_data.csv')
        logger.info(f"Loaded fundamental data from cache: {len(fundamental_df)} symbols")
    except:
        logger.info("Fetching fundamental data...")
        fundamental_df = fund_fetcher.fetch_fundamental_data(symbols)
        fund_fetcher.save_data(fundamental_df, 'fundamental_data.csv')

    # 3. Macro data
    TCMB_API_KEY = os.getenv('TCMB_API_KEY', 'tV4qq6RzPr')
    macro_fetcher = MacroDataFetcher(api_key=TCMB_API_KEY)
    try:
        macro_df = macro_fetcher.load_data('macro_data.csv')
        logger.info(f"Loaded macro data from cache: {len(macro_df)} rows")
    except:
        logger.info("Fetching macro data from TCMB EVDS...")
        macro_df = macro_fetcher.fetch_macro_data()
        macro_fetcher.save_data(macro_df, 'macro_data.csv')

    # Split data
    train_df, val_df, test_df = fetcher.split_data(df)

    logger.info(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df, fundamental_df, macro_df


def create_env_with_weights(df, fundamental_df, macro_df, weights: dict):
    """
    Create trading environment with specific PSR weights
    """
    def _init():
        return make_env(
            df,
            initial_balance=1_000_000,
            phase=2,
            reward_type='psr',
            reward_weights=weights,
            fundamental_df=fundamental_df,
            macro_df=macro_df
        )

    return DummyVecEnv([_init])


def objective(trial: optuna.Trial, train_df, val_df, fundamental_df, macro_df):
    """
    Optuna objective function

    Goal: Maximize Sharpe Ratio while minimizing MDD

    Returns:
        float: Combined score (higher is better)
    """
    # Suggest hyperparameters
    weights = {
        'w1': trial.suggest_float('w1', 0.3, 0.7),    # Portfolio return
        'w2': trial.suggest_float('w2', 0.2, 0.5),    # Sharpe ratio
        'w3': trial.suggest_float('w3', 0.05, 0.2),   # MDD penalty
        'w4': trial.suggest_float('w4', 0.01, 0.1),   # Volatility penalty
        'w5': trial.suggest_float('w5', 0.01, 0.1),   # Trade frequency
        'rolling_window': trial.suggest_int('rolling_window', 20, 60),
        'target_trades_per_100': trial.suggest_int('target_trades_per_100', 30, 80)
    }

    # Normalize weights (w1-w5 should sum to ~1.0)
    weight_sum = sum([weights['w1'], weights['w2'], weights['w3'], weights['w4'], weights['w5']])
    for key in ['w1', 'w2', 'w3', 'w4', 'w5']:
        weights[key] /= weight_sum

    logger.info(f"\nTrial {trial.number}:")
    logger.info(f"  Weights: w1={weights['w1']:.3f}, w2={weights['w2']:.3f}, "
                f"w3={weights['w3']:.3f}, w4={weights['w4']:.3f}, w5={weights['w5']:.3f}")

    try:
        # Create environment with these weights
        env = create_env_with_weights(train_df, fundamental_df, macro_df, weights)

        # Quick training (10k steps for fast iteration)
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        callback = QuickTrainingCallback(max_steps=10000)
        model.learn(total_timesteps=10000, callback=callback)

        # Evaluate on validation set
        eval_env = create_env_with_weights(val_df, fundamental_df, macro_df, weights)
        obs = eval_env.reset()
        done = [False]
        portfolio_values = []

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

            # VecEnv returns lists/arrays
            current_done = done[0] if isinstance(done, (list, np.ndarray)) else done
            current_info = info[0] if isinstance(info, list) else info

            portfolio_values.append(current_info['portfolio_value'])
            done = [current_done]  # Update for next iteration

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)

        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        mdd = np.min(drawdown)

        cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Combined score: Maximize Sharpe + minimize MDD
        # MDD is negative, so we negate it (lower MDD = higher score)
        score = sharpe_ratio - (mdd * 10)  # Weight MDD penalty 10x

        logger.info(f"  Results: Sharpe={sharpe_ratio:.3f}, MDD={mdd:.3f}, "
                    f"Return={cumulative_return:.3f}, Score={score:.3f}")

        # Store metrics in trial
        trial.set_user_attr('sharpe_ratio', float(sharpe_ratio))
        trial.set_user_attr('mdd', float(mdd))
        trial.set_user_attr('return', float(cumulative_return))

        return score

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -999999  # Very low score for failed trials


def run_optimization(n_trials: int = 50):
    """
    Run Optuna optimization

    Args:
        n_trials: Number of trials to run
    """
    logger.info("="*80)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION - PSR REWARD WEIGHTS")
    logger.info("="*80)

    # Load data
    train_df, val_df, test_df, fundamental_df, macro_df = load_phase2_data()

    # Create Optuna study
    study = optuna.create_study(
        study_name='psr_reward_weights',
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    # Run optimization
    logger.info(f"\nStarting optimization with {n_trials} trials...")
    logger.info("This may take 1-2 hours depending on your hardware.\n")

    study.optimize(
        lambda trial: objective(trial, train_df, val_df, fundamental_df, macro_df),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Results
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)

    best_trial = study.best_trial
    logger.info(f"\nBest trial: #{best_trial.number}")
    logger.info(f"  Score: {best_trial.value:.4f}")
    logger.info(f"  Sharpe Ratio: {best_trial.user_attrs['sharpe_ratio']:.4f}")
    logger.info(f"  MDD: {best_trial.user_attrs['mdd']:.4f}")
    logger.info(f"  Return: {best_trial.user_attrs['return']:.4f}")
    logger.info("\nBest weights:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save results
    results_dir = project_root / 'results' / 'optimization'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save best params to JSON
    import json
    best_params_path = results_dir / 'best_psr_weights.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=2)
    logger.info(f"\nBest parameters saved to: {best_params_path}")

    # Save study results to CSV
    trials_df = study.trials_dataframe()
    trials_path = results_dir / 'optimization_trials.csv'
    trials_df.to_csv(trials_path, index=False)
    logger.info(f"All trials saved to: {trials_path}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    try:
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html(results_dir / 'optimization_history.html')
        logger.info("  - Optimization history saved")

        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_html(results_dir / 'param_importances.html')
        logger.info("  - Parameter importances saved")

        # Contour plot (w1 vs w2)
        fig3 = plot_contour(study, params=['w1', 'w2'])
        fig3.write_html(results_dir / 'contour_w1_w2.html')
        logger.info("  - Contour plot (w1 vs w2) saved")

    except Exception as e:
        logger.warning(f"Could not generate all visualizations: {e}")

    logger.info("\n" + "="*80)
    logger.info("[SUCCESS] Optimization complete! Check results/ directory for outputs.")
    logger.info("="*80)

    return study


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Optimize PSR reward weights using Optuna')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    args = parser.parse_args()

    study = run_optimization(n_trials=args.trials)
