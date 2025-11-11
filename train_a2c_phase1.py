"""
A2C Model Training Script (Faz 1)
Basit proof-of-concept: BIST-30'dan 5 hisse ile A2C eğitimi
"""

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import logging
from datetime import datetime

# Local imports
from data.data_fetcher import DataFetcher
from data.bist30_symbols import get_symbols
from data.technical_indicators import add_indicators_to_multi_symbol_df
from env.trading_env import make_env

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """Custom callback to log training progress"""

    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Log at the end of each rollout"""
        if self.num_timesteps % self.eval_freq == 0:
            logger.info(f"Timesteps: {self.num_timesteps}")


def prepare_data():
    """Veri hazırlığı"""
    logger.info("Preparing data...")

    symbols = get_symbols(phase=1)
    logger.info(f"Trading symbols: {symbols}")

    fetcher = DataFetcher()

    # Veriyi yükle veya çek
    try:
        df = fetcher.load_data('stock_data_with_indicators.csv')
        logger.info("Data loaded from cache")
    except FileNotFoundError:
        logger.info("Fetching fresh data...")
        df = fetcher.fetch_stock_data(symbols, save=True)
        df = fetcher.clean_data(df)
        df = add_indicators_to_multi_symbol_df(df)
        fetcher.save_data(df, 'stock_data_with_indicators.csv')

    # Train/Val/Test split
    train_df, val_df, test_df = fetcher.split_data(df)

    logger.info(f"Train: {len(train_df)} rows")
    logger.info(f"Val: {len(val_df)} rows")
    logger.info(f"Test: {len(test_df)} rows")

    return train_df, val_df, test_df


def create_env(df: pd.DataFrame):
    """Training environment oluştur"""
    env = make_env(
        df=df,
        initial_balance=1_000_000,
        commission_rate=0.001,
        max_shares_per_trade=100
    )

    # Vectorize (SB3 requirement)
    env = DummyVecEnv([lambda: env])

    return env


def train_a2c(
    train_df: pd.DataFrame,
    total_timesteps: int = 50_000,
    learning_rate: float = 0.0007,
    n_steps: int = 5
):
    """
    A2C modelini eğit

    Args:
        train_df: Training data
        total_timesteps: Total training steps
        learning_rate: Learning rate
        n_steps: Number of steps to run for each environment per update
    """
    logger.info("="*60)
    logger.info("TRAINING A2C MODEL (FAZ 1)")
    logger.info("="*60)

    # Environment oluştur
    env = create_env(train_df)

    # A2C model
    model = A2C(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )

    logger.info(f"Model parameters:")
    logger.info(f"  Policy: MlpPolicy")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  N steps: {n_steps}")
    logger.info(f"  Total timesteps: {total_timesteps}")

    # Callback
    callback = TradingCallback(eval_freq=5000)

    # Training
    logger.info("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Model kaydet
    os.makedirs('models', exist_ok=True)
    model_path = 'models/a2c_bist30_phase1'
    model.save(model_path)
    logger.info(f"\nModel saved to {model_path}")

    return model


def evaluate_model(model, test_df: pd.DataFrame, n_episodes=1):
    """
    Modeli test datasında değerlendir

    Args:
        model: Trained A2C model
        test_df: Test data
        n_episodes: Number of test episodes
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATING MODEL")
    logger.info("="*60)

    env = create_env(test_df)

    all_metrics = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

        # Metrics al
        metrics = env.envs[0].get_metrics()
        all_metrics.append(metrics)

        logger.info(f"\nEpisode {episode + 1} Results:")
        logger.info(f"  Cumulative Return: {metrics['cumulative_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Final Portfolio: ${metrics['final_portfolio_value']:,.2f}")
        logger.info(f"  Total Trades: {metrics['total_trades']}")

    # Ortalama metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    logger.info("\n" + "="*60)
    logger.info("AVERAGE METRICS")
    logger.info("="*60)
    for key, value in avg_metrics.items():
        logger.info(f"{key}: {value}")

    return avg_metrics


def main():
    """Main training pipeline"""
    start_time = datetime.now()

    # 1. Veri hazırlığı
    train_df, val_df, test_df = prepare_data()

    # 2. Model eğitimi
    model = train_a2c(
        train_df,
        total_timesteps=50_000,
        learning_rate=0.0007,
        n_steps=5
    )

    # 3. Değerlendirme
    test_metrics = evaluate_model(model, test_df, n_episodes=1)

    # 4. Sonuçları kaydet
    results = {
        'timestamp': start_time.isoformat(),
        'model': 'A2C',
        'phase': 1,
        'symbols': get_symbols(phase=1),
        'training_timesteps': 50_000,
        'test_metrics': test_metrics
    }

    # Save results
    os.makedirs('results', exist_ok=True)
    results_file = f'results/a2c_phase1_{start_time.strftime("%Y%m%d_%H%M%S")}.txt'

    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FAZ 1: A2C TRAINING RESULTS\n")
        f.write("="*60 + "\n\n")

        for key, value in results.items():
            if key == 'test_metrics':
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

    logger.info(f"\nResults saved to {results_file}")

    elapsed_time = datetime.now() - start_time
    logger.info(f"\nTotal execution time: {elapsed_time}")

    logger.info("\n" + "="*60)
    logger.info("FAZ 1 COMPLETED!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
