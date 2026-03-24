"""
(#60) Mock data fixtures for offline testing.
Generates synthetic OHLCV + indicator data without any network calls.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


PHASE1_SYMBOLS = ['AKBNK.IS', 'THYAO.IS', 'TUPRS.IS', 'BIMAS.IS', 'ASELS.IS']


def make_mock_ohlcv(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic multi-index (symbol, date) OHLCV DataFrame.
    Uses random walk prices starting at 50 so z-score normalization works.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2022, 1, 3)
    dates = [start + timedelta(days=i) for i in range(n_days) if (start + timedelta(days=i)).weekday() < 5][:n_days]

    records = []
    for symbol in PHASE1_SYMBOLS:
        price = 50.0
        for date in dates:
            pct = rng.normal(0.0005, 0.01)
            price = max(price * (1 + pct), 0.1)
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            open_ = price * (1 + rng.normal(0, 0.003))
            volume = rng.integers(100_000, 10_000_000)
            records.append({
                'symbol': symbol,
                'date': date,
                'open': open_,
                'high': high,
                'low': low,
                'close': price,
                'volume': float(volume),
            })

    df = pd.DataFrame(records)
    df = df.set_index(['symbol', 'date'])
    return df


def make_mock_df_with_indicators(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Returns mock OHLCV DataFrame with all technical indicator columns expected
    by TradingEnv (rsi, macd, macd_signal, macd_hist, bb_upper, bb_lower,
    bb_mid, adx, atr, obv, turbulence).
    Fills indicator columns with plausible constant or random values.
    """
    df = make_mock_ohlcv(n_days=n_days, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = len(df)

    df['rsi'] = rng.uniform(30, 70, n)
    df['macd'] = rng.normal(0, 0.5, n)
    df['macd_signal'] = rng.normal(0, 0.4, n)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['bb_upper'] = df['close'] * 1.02
    df['bb_lower'] = df['close'] * 0.98
    df['bb_mid'] = df['close']
    df['adx'] = rng.uniform(15, 40, n)
    df['atr'] = df['close'] * rng.uniform(0.005, 0.02, n)
    df['obv'] = rng.integers(0, 10_000_000, n).astype(float)
    df['turbulence'] = rng.uniform(0, 50, n)

    return df
