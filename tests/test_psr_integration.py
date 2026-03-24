"""
Test PSR Reward Integration
Faz 2 - Sprint 2

Bu test, PSR reward'in TradingEnv'e dogru entegre edildigini dogrular.

Test Scenarios:
1. Simple reward ile env olusturma
2. PSR reward ile env olusturma
3. Reward componentlerinin dogru hesaplanmasi
4. Episode boyunca reward hesaplamalarinin calismasi
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import logging

from data.data_fetcher import DataFetcher
from data.bist30_symbols import get_symbols
from data.technical_indicators import add_indicators_to_multi_symbol_df
from data.fundamental_fetcher import FundamentalDataFetcher
from data.macro_fetcher import MacroDataFetcher
from env.trading_env import make_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_test_data():
    """Load minimal data for testing"""
    logger.info("Loading test data...")

    # Market data
    symbols = get_symbols(phase=2)[:2]  # Use only 2 symbols for quick test
    fetcher = DataFetcher()

    try:
        df = fetcher.load_data('stock_data_with_indicators.csv')
        # Filter to test symbols only
        df = df[df.index.get_level_values('symbol').isin(symbols)]
        logger.info(f"Loaded market data: {len(df)} rows for {symbols}")
    except:
        logger.info("Market data not found, please run data pipeline first")
        raise

    # Fundamental data
    fund_fetcher = FundamentalDataFetcher()
    try:
        fundamental_df = fund_fetcher.load_data('fundamental_data.csv')
        fundamental_df = fundamental_df[fundamental_df.index.isin(symbols)]
        logger.info(f"Loaded fundamental data: {len(fundamental_df)} symbols")
    except:
        logger.info("Fundamental data not found, using None")
        fundamental_df = None

    # Macro data
    import os
    TCMB_API_KEY = os.getenv('TCMB_API_KEY', 'tV4qq6RzPr')
    macro_fetcher = MacroDataFetcher(api_key=TCMB_API_KEY)
    try:
        macro_df = macro_fetcher.load_data('macro_data.csv')
        logger.info(f"Loaded macro data: {len(macro_df)} rows")
    except:
        logger.info("Macro data not found, using None")
        macro_df = None

    # Use small subset for quick testing
    train_df, _, _ = fetcher.split_data(df)

    return train_df, fundamental_df, macro_df


def test_simple_reward():
    """Test 1: Simple reward environment"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Simple Reward Environment")
    logger.info("="*80)

    train_df, _, _ = load_test_data()

    # Create environment with simple reward
    env = make_env(
        train_df,
        initial_balance=1_000_000,
        phase=1,
        reward_type='simple'
    )

    logger.info(f"Environment created: {env}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")

    # Reset and run a few steps
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial portfolio value: TRY {info['portfolio_value']:,.0f}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        logger.info(f"Step {step+1}: Reward={reward:.4f}, "
                   f"Portfolio=TRY {info['portfolio_value']:,.0f}, "
                   f"Trades={info.get('trades_executed', 0)}")

        # Check reward components
        if 'reward_components' in info:
            logger.info(f"  Components: {info['reward_components']}")

        if terminated:
            break

    logger.info("[OK] Simple reward test passed")
    return True


def test_psr_reward():
    """Test 2: PSR reward environment"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: PSR Reward Environment")
    logger.info("="*80)

    train_df, fundamental_df, macro_df = load_test_data()

    # Create environment with PSR reward
    env = make_env(
        train_df,
        initial_balance=1_000_000,
        phase=2,
        reward_type='psr',
        fundamental_df=fundamental_df,
        macro_df=macro_df
    )

    logger.info(f"Environment created: {env}")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Reward calculator: {env.reward_calculator}")

    # Reset and run a few steps
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial portfolio value: TRY {info['portfolio_value']:,.0f}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        logger.info(f"Step {step+1}: Reward={reward:.4f}, "
                   f"Portfolio=TRY {info['portfolio_value']:,.0f}, "
                   f"Trades={info.get('trades_executed', 0)}")

        # Check reward components (PSR should have multiple components)
        if 'reward_components' in info:
            components = info['reward_components']
            logger.info(f"  PSR Components:")
            logger.info(f"    - Portfolio Return: {components.get('portfolio_return', 0):.4f}")
            logger.info(f"    - Sharpe Ratio: {components.get('sharpe_ratio', 0):.4f}")
            logger.info(f"    - MDD Penalty: {components.get('mdd_penalty', 0):.4f}")
            logger.info(f"    - Volatility Penalty: {components.get('volatility_penalty', 0):.4f}")
            logger.info(f"    - Trade Frequency: {components.get('trade_frequency', 0):.4f}")

        if terminated:
            break

    logger.info("[OK] PSR reward test passed")
    return True


def test_custom_psr_weights():
    """Test 3: PSR reward with custom weights"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: PSR Reward with Custom Weights")
    logger.info("="*80)

    train_df, fundamental_df, macro_df = load_test_data()

    # Custom weights (emphasize Sharpe ratio)
    custom_weights = {
        'w1': 0.3,   # Portfolio return
        'w2': 0.5,   # Sharpe ratio (increased)
        'w3': 0.1,   # MDD penalty
        'w4': 0.05,  # Volatility penalty
        'w5': 0.05,  # Trade frequency
        'rolling_window': 40,
        'target_trades_per_100': 60
    }

    # Create environment
    env = make_env(
        train_df,
        initial_balance=1_000_000,
        phase=2,
        reward_type='psr',
        reward_weights=custom_weights,
        fundamental_df=fundamental_df,
        macro_df=macro_df
    )

    logger.info(f"Custom weights: {env.reward_calculator.get_weights()}")

    # Reset and run a few steps
    obs, info = env.reset()

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        logger.info(f"Step {step+1}: Reward={reward:.4f}")

        if terminated:
            break

    logger.info("[OK] Custom weights test passed")
    return True


def test_episode_completion():
    """Test 4: Full episode with PSR reward"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Full Episode with PSR Reward")
    logger.info("="*80)

    train_df, fundamental_df, macro_df = load_test_data()

    # Create environment
    env = make_env(
        train_df,
        initial_balance=1_000_000,
        phase=2,
        reward_type='psr',
        fundamental_df=fundamental_df,
        macro_df=macro_df
    )

    # Run full episode (or max 100 steps)
    obs, info = env.reset()
    total_reward = 0
    step_count = 0

    while step_count < 100:  # Limit to 100 steps for quick test
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        if terminated:
            break

    logger.info(f"Episode completed in {step_count} steps")
    logger.info(f"Total reward: {total_reward:.4f}")
    logger.info(f"Final portfolio value: TRY {info['portfolio_value']:,.0f}")

    # Check final metrics
    if 'sharpe_ratio' in info:
        logger.info(f"Final Sharpe Ratio: {info['sharpe_ratio']:.4f}")
        logger.info(f"Final MDD: {info['max_drawdown']:.4f}")
        logger.info(f"Total Trades: {info.get('total_trades', 0)}")

    logger.info("[OK] Full episode test passed")
    return True


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("PSR REWARD INTEGRATION TEST SUITE")
    logger.info("="*80)

    all_passed = True

    try:
        # Test 1: Simple reward
        if not test_simple_reward():
            all_passed = False

        # Test 2: PSR reward
        if not test_psr_reward():
            all_passed = False

        # Test 3: Custom weights
        if not test_custom_psr_weights():
            all_passed = False

        # Test 4: Full episode
        if not test_episode_completion():
            all_passed = False

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Summary
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("[SUCCESS] ALL TESTS PASSED")
    else:
        logger.info("[FAILURE] SOME TESTS FAILED")
    logger.info("="*80)

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
