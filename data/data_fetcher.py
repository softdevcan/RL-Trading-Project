"""
Veri Çekme Modülü
BIST-30 hisseleri için Yahoo Finance'den günlük OHLCV verisi çeker
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """BIST-30 hisse verileri için veri çekici"""

    def __init__(self, start_date: str = "2018-01-01", end_date: str = None):
        """
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD), None ise bugün
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = "data"

        # Veri dizinini oluştur
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_stock_data(self, symbols: List[str], save=True) -> pd.DataFrame:
        """
        Birden fazla hisse için OHLCV verisi çek

        Args:
            symbols: Hisse sembolleri listesi
            save: CSV olarak kaydet

        Returns:
            Multi-index DataFrame (Date, Symbol)
        """
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")

        all_data = {}

        for symbol in symbols:
            try:
                logger.info(f"Downloading {symbol}...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)

                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                # Sütun isimlerini küçük harfe çevir
                df.columns = df.columns.str.lower()

                # Sadece ihtiyacımız olan sütunları al
                df = df[['open', 'high', 'low', 'close', 'volume']]

                # DON'T add symbol column here - it will be added by concat with keys
                # df['symbol'] = symbol  # REMOVED - causes duplicate column!

                all_data[symbol] = df

                logger.info(f"  ✓ {symbol}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data was fetched for any symbol")

        # Tüm verileri birleştir
        combined_df = pd.concat(all_data.values(), keys=all_data.keys())
        combined_df.index.names = ['symbol', 'date']

        logger.info(f"\nTotal data points: {len(combined_df)}")
        logger.info(f"Date range: {combined_df.index.get_level_values('date').min().date()} "
                   f"to {combined_df.index.get_level_values('date').max().date()}")

        if save:
            self.save_data(combined_df, 'raw_stock_data.csv')

        return combined_df

    def save_data(self, df: pd.DataFrame, filename: str):
        """DataFrame'i CSV olarak kaydet"""
        filepath = os.path.join(self.data_dir, filename)
        # Reset index to avoid duplicate symbol column in CSV
        df.to_csv(filepath, index=True)
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename: str = 'raw_stock_data.csv') -> pd.DataFrame:
        """Kaydedilmiş veriyi yükle"""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Read CSV and handle potential duplicate columns
        df = pd.read_csv(filepath, parse_dates=['date'])

        # Remove duplicate symbol column if it exists
        if 'symbol' in df.columns and df.columns.tolist().count('symbol') > 1:
            logger.warning("Found duplicate 'symbol' column in CSV, removing duplicates")
            # Keep only first occurrence of each column
            df = df.loc[:, ~df.columns.duplicated()]

        # Set multi-index
        if 'symbol' in df.columns and 'date' in df.columns:
            df = df.set_index(['symbol', 'date'])
        else:
            # Already has multi-index from CSV
            df.index.names = ['symbol', 'date']

        logger.info(f"Data loaded from {filepath}: {len(df)} rows")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"  Index names: {df.index.names}")
        logger.info(f"  Unique symbols: {df.index.get_level_values('symbol').unique().tolist()}")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Veri temizleme

        - NaN değerleri forward fill
        - Negatif fiyatları kaldır
        - Duplicate kayıtları kaldır
        """
        logger.info("Cleaning data...")

        # Duplicate index kaldır
        df = df[~df.index.duplicated(keep='first')]

        # Her sembol için ayrı ayrı işle
        cleaned_dfs = []

        for symbol in df.index.get_level_values('symbol').unique():
            symbol_df = df.xs(symbol, level='symbol').copy()

            # Forward fill NaN değerler
            symbol_df = symbol_df.fillna(method='ffill')

            # Geriye kalan NaN'ları backward fill
            symbol_df = symbol_df.fillna(method='bfill')

            # Hala NaN varsa 0 ile doldur (volume için)
            symbol_df = symbol_df.fillna(0)

            # Negatif veya 0 fiyatları temizle
            for col in ['open', 'high', 'low', 'close']:
                symbol_df = symbol_df[symbol_df[col] > 0]

            # Volume negatif olamaz
            symbol_df['volume'] = symbol_df['volume'].clip(lower=0)

            # Symbol bilgisini geri ekleme - concat keys parametresi ile eklenecek
            # symbol_df['symbol'] = symbol  # REMOVED - causes duplicate

            cleaned_dfs.append((symbol, symbol_df))

        # Yeniden birleştir
        cleaned_df = pd.concat([df for _, df in cleaned_dfs], keys=[s for s, _ in cleaned_dfs])
        cleaned_df.index.names = ['symbol', 'date']

        removed_rows = len(df) - len(cleaned_df)
        logger.info(f"Cleaned: {removed_rows} rows removed ({len(cleaned_df)} rows remaining)")

        return cleaned_df

    def split_data(self, df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Veriyi train/validation/test olarak böl

        Args:
            df: Input DataFrame
            train_ratio: Training set oranı (0.7 = %70)
            val_ratio: Validation set oranı (0.15 = %15)

        Returns:
            (train_df, val_df, test_df)
        """
        # Tarihleri al
        dates = df.index.get_level_values('date').unique().sort_values()
        n_dates = len(dates)

        # Split indices
        train_end = int(n_dates * train_ratio)
        val_end = int(n_dates * (train_ratio + val_ratio))

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]

        # Split dataframes
        train_df = df[df.index.get_level_values('date').isin(train_dates)]
        val_df = df[df.index.get_level_values('date').isin(val_dates)]
        test_df = df[df.index.get_level_values('date').isin(test_dates)]

        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(train_df)} rows ({train_dates[0].date()} to {train_dates[-1].date()})")
        logger.info(f"  Val:   {len(val_df)} rows ({val_dates[0].date()} to {val_dates[-1].date()})")
        logger.info(f"  Test:  {len(test_df)} rows ({test_dates[0].date()} to {test_dates[-1].date()})")

        return train_df, val_df, test_df


def main():
    """Test script"""
    from bist30_symbols import get_symbols

    # Faz 1 için 5 hisse ile test
    symbols = get_symbols(phase=1)

    # Data fetcher oluştur
    fetcher = DataFetcher(start_date="2018-01-01")

    # Veri çek
    df = fetcher.fetch_stock_data(symbols, save=True)

    # Veriyi temizle
    df_clean = fetcher.clean_data(df)

    # Train/Val/Test split
    train_df, val_df, test_df = fetcher.split_data(df_clean)

    # Özet istatistikler
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"\nTotal rows: {len(df_clean)}")
    print(f"Date range: {df_clean.index.get_level_values('date').min().date()} "
          f"to {df_clean.index.get_level_values('date').max().date()}")

    print("\nSample data (first 5 rows):")
    print(df_clean.head())

    print("\nData types:")
    print(df_clean.dtypes)

    print("\nBasic statistics:")
    print(df_clean.describe())


if __name__ == '__main__':
    main()
