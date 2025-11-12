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
    def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
        """
        MACD (Moving Average Convergence Divergence)

        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            MACD line (fast EMA - slow EMA)
        """
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd

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

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
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

        # Smoothed values
        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def calculate_turbulence(df: pd.DataFrame, window=252) -> pd.Series:
        """
        Turbulence Index (Ansari et al.)
        Measures market volatility/turbulence

        Args:
            df: DataFrame with 'close' column
            window: Rolling window (default 252 = 1 year)

        Returns:
            Turbulence values
        """
        # Basit volatilite metriği olarak rolling std kullanıyoruz
        # Normalize edilmiş volatilite
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window, min_periods=20).std()

        # Turbulence = volatility * 100
        turbulence = volatility * 100

        return turbulence.fillna(0)

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
        df['macd'] = self.calculate_macd(df)
        df['rsi'] = self.calculate_rsi(df)
        df['cci'] = self.calculate_cci(df)
        df['adx'] = self.calculate_adx(df)
        df['turbulence'] = self.calculate_turbulence(df)

        # NaN değerleri forward fill
        indicator_cols = ['macd', 'rsi', 'cci', 'adx', 'turbulence']
        df[indicator_cols] = df[indicator_cols].fillna(method='ffill')
        df[indicator_cols] = df[indicator_cols].fillna(method='bfill')
        df[indicator_cols] = df[indicator_cols].fillna(0)

        return df


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

        # Symbol bilgisini geri ekleme - concat keys parametresi ile eklenecek
        # symbol_df['symbol'] = symbol  # REMOVED - causes duplicate

        result_dfs.append((symbol, symbol_df))

    # Birleştir
    result_df = pd.concat([df for _, df in result_dfs], keys=[s for s, _ in result_dfs])
    result_df.index.names = ['symbol', 'date']

    logger.info(f"Technical indicators added for all symbols")

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
