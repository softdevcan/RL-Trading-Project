"""
Teknik İndikatör Hesaplayıcı
Ansari et al. (2024) makalesindeki 5 teknik indikatörü hesaplar:
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- CCI (Commodity Channel Index)
- ADX (Average Directional Index)
- Turbulence
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Ansari et al. teknik indikatörleri"""

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
        """
        MACD (Moving Average Convergence Divergence)

        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period=14) -> pd.Series:
        """
        RSI (Relative Strength Index)

        Args:
            df: DataFrame with 'close' column
            period: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Wilder's EMA (Wilder 1978)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_cci(df: pd.DataFrame, period=20) -> pd.Series:
        """
        CCI (Commodity Channel Index)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: CCI period (default 20)

        Returns:
            CCI values
        """
        tp = (df['high'] + df['low'] + df['close']) / 3  # Typical Price
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

        cci = (tp - sma_tp) / (0.015 * mad)
        return cci

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period=14) -> pd.Series:
        """
        ADX (Average Directional Index)

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ADX period (default 14)

        Returns:
            ADX values
        """
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Wilder's EMA smoothing (Wilder 1978)
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr

        # DX and ADX — guard against division by zero (#8)
        denominator = plus_di + minus_di
        dx = pd.Series(np.where(denominator > 0,
                                100 * np.abs(plus_di - minus_di) / denominator,
                                0))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return adx

    @staticmethod
    def calculate_turbulence(df: pd.DataFrame, window=252) -> pd.Series:
        """
        Turbulence Index — symbol-level fallback (rolling volatility).
        For the proper cross-sectional Mahalanobis turbulence, use
        add_indicators_to_multi_symbol_df() which operates on all symbols.

        Args:
            df: DataFrame with 'close' column
            window: Rolling window (default 252 = 1 year)

        Returns:
            Turbulence values (normalised rolling volatility * 100)
        """
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window, min_periods=20).std()
        return (volatility * 100).fillna(0)

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm teknik indikatörleri DataFrame'e ekle

        Args:
            df: Input DataFrame (symbol level)

        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()

        # Her indikatörü hesapla
        macd, macd_signal, macd_hist = self.calculate_macd(df)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['rsi'] = self.calculate_rsi(df)
        df['cci'] = self.calculate_cci(df)
        df['adx'] = self.calculate_adx(df)
        df['turbulence'] = self.calculate_turbulence(df)

        # NaN değerleri forward fill
        indicator_cols = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'adx', 'turbulence']
        df[indicator_cols] = df[indicator_cols].ffill()  # Forward fill
        df[indicator_cols] = df[indicator_cols].bfill()  # Backward fill
        df[indicator_cols] = df[indicator_cols].fillna(0)

        return df


def _calculate_mahalanobis_turbulence(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Cross-sectional Mahalanobis turbulence index (Liu et al. 2020, Kritzman & Li 2010).

    Formula: turb_t = (r_t - mu)^T * Sigma^{-1} * (r_t - mu)
    where r_t is the vector of daily returns across all symbols on date t,
    mu and Sigma are rolling mean and covariance over `window` days.

    Args:
        df: Multi-index DataFrame (symbol, date) with 'close' column
        window: Rolling lookback period (default 252)

    Returns:
        Series indexed by date with turbulence values
    """
    # Build a (date x symbol) returns matrix
    close_pivot = df['close'].unstack(level='symbol')
    returns = close_pivot.pct_change()

    turbulence_values = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        hist = returns.iloc[i - window:i].dropna(axis=1)
        if hist.shape[1] < 2:
            turbulence_values.iloc[i] = 0.0
            continue

        current = returns.iloc[i][hist.columns].values
        if np.any(np.isnan(current)):
            turbulence_values.iloc[i] = 0.0
            continue

        mu = hist.mean().values
        try:
            cov = hist.cov().values
            cov_inv = np.linalg.pinv(cov)  # pseudo-inverse handles near-singular matrices
            diff = current - mu
            turb = float(diff @ cov_inv @ diff)
            turbulence_values.iloc[i] = max(turb, 0.0)
        except np.linalg.LinAlgError:
            turbulence_values.iloc[i] = 0.0

    return turbulence_values.fillna(0)


def add_indicators_to_multi_symbol_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-symbol DataFrame'e teknik indikatörleri ekle

    Args:
        df: Multi-index DataFrame (symbol, date)

    Returns:
        DataFrame with technical indicators for all symbols
    """
    calculator = TechnicalIndicators()
    result_dfs = []

    for symbol in df.index.get_level_values('symbol').unique():
        logger.info(f"Calculating indicators for {symbol}...")

        # Symbol için veriyi al
        symbol_df = df.xs(symbol, level='symbol').copy()

        # İndikatörleri ekle
        symbol_df = calculator.add_all_indicators(symbol_df)

        result_dfs.append((symbol, symbol_df))

    # Birleştir
    result_df = pd.concat([df for _, df in result_dfs], keys=[s for s, _ in result_dfs])
    result_df.index.names = ['symbol', 'date']

    # Cross-sectional Mahalanobis turbulence — overwrite symbol-level fallback
    logger.info("Calculating cross-sectional Mahalanobis turbulence...")
    turb_by_date = _calculate_mahalanobis_turbulence(result_df)

    # Broadcast back to every symbol row
    for symbol in result_df.index.get_level_values('symbol').unique():
        symbol_dates = result_df.loc[symbol].index
        result_df.loc[symbol, 'turbulence'] = turb_by_date.reindex(symbol_dates).values

    logger.info("Technical indicators added for all symbols")

    return result_df


def main():
    """Test script"""
    from data_fetcher import DataFetcher
    from bist30_symbols import get_symbols

    # Veri yükle
    fetcher = DataFetcher()

    try:
        df = fetcher.load_data('raw_stock_data.csv')
    except FileNotFoundError:
        logger.info("Data not found, fetching...")
        symbols = get_symbols(phase=1)
        df = fetcher.fetch_stock_data(symbols, save=True)
        df = fetcher.clean_data(df)

    # Teknik indikatörleri ekle
    df_with_indicators = add_indicators_to_multi_symbol_df(df)

    # Kaydet
    fetcher.save_data(df_with_indicators, 'stock_data_with_indicators.csv')

    print("\n" + "="*60)
    print("TECHNICAL INDICATORS SUMMARY")
    print("="*60)
    print(f"\nColumns: {list(df_with_indicators.columns)}")
    print(f"\nSample data (first symbol, first 5 rows):")
    first_symbol = df_with_indicators.index.get_level_values('symbol').unique()[0]
    print(df_with_indicators.xs(first_symbol, level='symbol').head())

    print("\nIndicator statistics:")
    indicator_cols = ['macd', 'rsi', 'cci', 'adx', 'turbulence']
    print(df_with_indicators[indicator_cols].describe())


if __name__ == '__main__':
    main()
