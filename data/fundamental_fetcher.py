"""
Fundamental Veri Çekme Modülü (Faz 2)
yfinance kullanarak şirketlerin finansal oranlarını çeker

Fundamental Faktörler (7 oran):
1. ROE (Return on Equity) - Özkaynak karlılığı
2. ROA (Return on Assets) - Aktif karlılığı
3. Debt/Equity - Kaldıraç oranı
4. Current Ratio - Cari oran
5. P/E Ratio - Fiyat/Kazanç oranı
6. P/B Ratio - Piyasa Değeri/Defter Değeri oranı
7. Profit Margin - Kar marjı
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalDataFetcher:
    """yfinance ile fundamental veri çekici"""

    # Fundamental faktörler ve yfinance key'leri
    FUNDAMENTAL_KEYS = {
        'roe': 'returnOnEquity',              # ROE (%)
        'roa': 'returnOnAssets',              # ROA (%)
        'debt_to_equity': 'debtToEquity',     # Kaldıraç oranı
        'current_ratio': 'currentRatio',      # Cari oran
        'pe_ratio': 'trailingPE',             # F/K oranı
        'pb_ratio': 'priceToBook',            # PD/DD oranı
        'profit_margin': 'profitMargins',     # Kar marjı (%)
    }

    def __init__(self):
        """Initialize fundamental data fetcher"""
        self.data_dir = "data/fundamental"
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info("FundamentalDataFetcher initialized")

    def fetch_fundamental_data(self, symbols: List[str], save: bool = True) -> pd.DataFrame:
        """
        Birden fazla hisse için fundamental veri çek

        Args:
            symbols: Hisse sembolleri listesi (e.g., ['AKBNK.IS', 'THYAO.IS'])
            save: CSV olarak kaydet

        Returns:
            DataFrame with multi-index (symbol) and columns (7 ratios)
        """
        logger.info(f"Fetching fundamental data for {len(symbols)} symbols...")

        all_data = {}

        for symbol in symbols:
            try:
                logger.info(f"Fetching {symbol}...")

                # yfinance ticker oluştur
                ticker = yf.Ticker(symbol)

                # info sözlüğünü al
                info = ticker.info

                if not info or 'symbol' not in info:
                    logger.warning(f"No info found for {symbol}")
                    continue

                # Fundamental ratios çek
                ratios = {}
                for ratio_name, yf_key in self.FUNDAMENTAL_KEYS.items():
                    value = info.get(yf_key, np.nan)

                    # yfinance bazen None döndürür
                    if value is None:
                        value = np.nan

                    # Yüzde değerleri ondalık formatında gelebilir (0.15 = %15)
                    # ROE, ROA, profit_margin 0-1 arasında gelirse *100 yap
                    if ratio_name in ['roe', 'roa', 'profit_margin']:
                        if not np.isnan(value) and 0 <= value <= 1:
                            value *= 100  # Yüzdeye çevir

                    ratios[ratio_name] = value

                all_data[symbol] = ratios

                # Başarılı çekilen değerleri logla
                available = [k for k, v in ratios.items() if not np.isnan(v)]
                logger.info(f"  ✓ {symbol}: {len(available)}/7 ratios available "
                           f"({', '.join(available)})")

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                # Hata durumunda boş seri ekle
                all_data[symbol] = {k: np.nan for k in self.FUNDAMENTAL_KEYS.keys()}

        if not all_data:
            raise ValueError("No fundamental data was fetched")

        # DataFrame oluştur
        df = pd.DataFrame.from_dict(all_data, orient='index')
        df.index.name = 'symbol'

        logger.info(f"\n✓ Fundamental data fetched for {len(df)} symbols")
        logger.info(f"Columns: {df.columns.tolist()}")

        # NaN istatistikleri
        nan_stats = df.isna().sum()
        logger.info(f"\nMissing data per ratio:")
        for col, nan_count in nan_stats.items():
            logger.info(f"  {col:20s}: {nan_count}/{len(df)} NaN")

        # Forward fill ve median ile doldur
        df = self._handle_missing_data(df)

        if save:
            self.save_data(df, 'fundamental_data.csv')

        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Eksik fundamental verileri işle

        Strateji:
        1. Sektör ortalaması ile doldur (gelecekte)
        2. Tüm sembollerin median'ı ile doldur (şimdilik)
        3. Hala NaN varsa 0 kullan

        Args:
            df: Input DataFrame

        Returns:
            Temizlenmiş DataFrame
        """
        logger.info("Handling missing fundamental data...")

        before_nan = df.isna().sum().sum()

        for col in df.columns:
            if df[col].isna().any():
                # Median ile doldur
                median_val = df[col].median()

                if np.isnan(median_val):
                    # Tüm değerler NaN ise makul bir değer kullan
                    default_values = {
                        'roe': 15.0,          # Ortalama ROE %15
                        'roa': 5.0,           # Ortalama ROA %5
                        'debt_to_equity': 1.0,# Ortalama kaldıraç 1.0
                        'current_ratio': 1.5, # Ortalama cari oran 1.5
                        'pe_ratio': 10.0,     # Ortalama F/K 10
                        'pb_ratio': 1.5,      # Ortalama PD/DD 1.5
                        'profit_margin': 10.0,# Ortalama kar marjı %10
                    }
                    fill_value = default_values.get(col, 0.0)
                    df[col] = df[col].fillna(fill_value)
                    logger.warning(f"{col}: All NaN, filled with default ({fill_value:.2f})")
                else:
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"{col}: Filled NaN with median ({median_val:.2f})")

        after_nan = df.isna().sum().sum()
        logger.info(f"Missing data handled: {before_nan} NaNs -> {after_nan} NaNs")

        return df

    def get_fundamental_features_for_symbol(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Belirli bir sembol için fundamental ratios al

        Args:
            df: Fundamental DataFrame (fetch_fundamental_data() çıktısı)
            symbol: Hisse sembolü

        Returns:
            Dict: {ratio_name: value}
        """
        if symbol not in df.index:
            logger.warning(f"Symbol {symbol} not found in fundamental data")
            # Varsayılan değerler döndür
            return {k: 0.0 for k in self.FUNDAMENTAL_KEYS.keys()}

        result = df.loc[symbol].to_dict()
        # Ensure all values are floats
        return {str(k): float(v) if not pd.isna(v) else 0.0 for k, v in result.items()}

    def save_data(self, df: pd.DataFrame, filename: str):
        """DataFrame'i CSV olarak kaydet"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=True)
        logger.info(f"Fundamental data saved to {filepath}")

    def load_data(self, filename: str = 'fundamental_data.csv') -> pd.DataFrame:
        """Kaydedilmiş fundamental veriyi yükle"""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fundamental data file not found: {filepath}")

        df = pd.read_csv(filepath, index_col='symbol')
        logger.info(f"Fundamental data loaded from {filepath}: {len(df)} symbols")
        logger.info(f"Columns: {df.columns.tolist()}")

        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fundamental ratios'ları normalize et (RL için önemli)

        Strateji:
        - Z-score normalization (mean=0, std=1)

        Args:
            df: Fundamental DataFrame

        Returns:
            Normalized DataFrame
        """
        logger.info("Normalizing fundamental features...")

        normalized_df = df.copy()

        for col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()

            if std_val > 0:
                normalized_df[col] = (df[col] - mean_val) / std_val
            else:
                logger.warning(f"{col} has zero std, skipping normalization")

        logger.info("Normalization complete")
        return normalized_df

    def update_fundamental_data(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Mevcut fundamental verileri güncelle (yeni veriler çek)

        Args:
            df: Mevcut DataFrame
            symbols: Güncellenecek semboller

        Returns:
            Güncellenmiş DataFrame
        """
        logger.info(f"Updating fundamental data for {len(symbols)} symbols...")

        # Yeni veriyi çek
        new_df = self.fetch_fundamental_data(symbols, save=False)

        # Mevcut veriyi güncelle
        df.update(new_df)

        # Yeni sembolleri ekle
        for symbol in new_df.index:
            if symbol not in df.index:
                df = pd.concat([df, new_df.loc[[symbol]]])

        logger.info("Fundamental data updated")
        return df


def main():
    """Test script"""
    from bist30_symbols import get_symbols

    # Faz 1 için 5 hisse ile test
    symbols = get_symbols(phase=1)

    # Fetcher oluştur
    fetcher = FundamentalDataFetcher()

    # Veri çek
    fundamental_df = fetcher.fetch_fundamental_data(symbols, save=True)

    # Özet istatistikler
    print("\n" + "="*60)
    print("FUNDAMENTAL DATA SUMMARY")
    print("="*60)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Total rows: {len(fundamental_df)}")
    print(f"\nColumns: {fundamental_df.columns.tolist()}")

    print("\nFundamental ratios:")
    print(fundamental_df)

    print("\nBasic statistics:")
    print(fundamental_df.describe())

    # Normalizasyon testi
    normalized_df = fetcher.normalize_features(fundamental_df)
    print("\n" + "="*60)
    print("NORMALIZED DATA")
    print("="*60)
    print(normalized_df)

    # Belirli bir sembol için ratios
    test_symbol = symbols[0]
    ratios = fetcher.get_fundamental_features_for_symbol(fundamental_df, test_symbol)
    print(f"\n" + "="*60)
    print(f"FUNDAMENTAL RATIOS FOR {test_symbol}")
    print("="*60)
    for key, value in ratios.items():
        print(f"{key:20s}: {value:12.2f}")


if __name__ == '__main__':
    main()
