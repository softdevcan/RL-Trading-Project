"""
(#58-61) Pytest-based environment tests using mock fixtures.
Run with: pytest tests/test_env_pytest.py -v
No network calls — uses synthetic data from tests/fixtures.py.
"""

import numpy as np
import pytest

from tests.fixtures import make_mock_df_with_indicators, PHASE1_SYMBOLS
from env.trading_env import make_env


@pytest.fixture
def mock_df():
    """300-day synthetic OHLCV + indicator DataFrame."""
    return make_mock_df_with_indicators(n_days=300)


@pytest.fixture
def env(mock_df):
    fetcher_mock = __import__('data.data_fetcher', fromlist=['DataFetcher']).DataFetcher
    _, _, test_df = fetcher_mock().split_data(mock_df)
    return make_env(df=test_df, initial_balance=1_000_000, commission_rate=0.001, max_shares_per_trade=100)


# ─── Observation Space ────────────────────────────────────────────────────────

def test_obs_shape(env):
    """(#58) Observation must be shape (56,)."""
    obs, _ = env.reset()
    assert obs.shape == (56,), f"Expected (56,), got {obs.shape}"


def test_obs_in_bounds(env):
    """(#1) Observation values must be within [-10, 10]."""
    obs, _ = env.reset()
    assert obs.min() >= -10.0, f"obs.min()={obs.min()} < -10"
    assert obs.max() <= 10.0, f"obs.max()={obs.max()} > 10"


def test_obs_in_observation_space(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


# ─── Step ─────────────────────────────────────────────────────────────────────

def test_step_returns_correct_types(env):
    env.reset()
    action = np.zeros(len(PHASE1_SYMBOLS), dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (56,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_random_steps_dont_crash(env):
    """100 random steps should not raise."""
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.uniform(-1, 1, len(PHASE1_SYMBOLS)).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        assert obs.shape == (56,)
        if terminated or truncated:
            env.reset()


# ─── Financials ───────────────────────────────────────────────────────────────

def test_initial_balance(env):
    _, info = env.reset()
    assert info['balance'] == pytest.approx(1_000_000.0)


def test_episode_metrics(env):
    """(#59) get_metrics() must return required keys with sane types."""
    obs, _ = env.reset()
    rng = np.random.default_rng(1)
    done = False
    while not done:
        action = rng.uniform(-1, 1, len(PHASE1_SYMBOLS)).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    metrics = env.get_metrics()
    required_keys = ['cumulative_return', 'sharpe_ratio', 'max_drawdown',
                     'final_portfolio_value', 'total_trades']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
    assert isinstance(metrics['total_trades'], (int, float))
    assert metrics['final_portfolio_value'] > 0
