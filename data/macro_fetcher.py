"""
Makroekonomik Veri Çekme Modülü (Faz 2)
TCMB EVDS API ve yfinance kullanarak makroekonomik göstergeleri çeker

Göstergeler:
1. TCMB Politika Faizi (%) - EVDS (Fallback: 50.0)
2. TÜFE Enflasyon (YoY %) - EVDS (Endeksten hesaplanır)
3. ÜFE Enflasyon (YoY %) - EVDS (Endeksten hesaplanır)
4. USD/TRY Döviz Kuru - yfinance
5. EUR/TRY Döviz Kuru - yfinance
6. BIST-100 Endeks - yfinance
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os
import yfinance as yf

# EVDS API için
try:
    from evds import evdsAPI
    EVDS_AVAILABLE = True
except ImportError:
    EVDS_AVAILABLE = False
    logging.warning("evds package not installed. Install with: pip install evds")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacroDataFetcher:
    """TCMB EVDS API ve yfinance ile makroekonomik veri çekici"""

    # EVDS Seri Kodları
    EVDS_CODES = {
        'policy_rate': 'TP.YSSK.A01',          # Gecelik Faiz (günlük)
        'cpi_index': 'TP.FG.J0',               # TÜFE Endeks (2003=100) - Aylık
        'ppi_index': 'TP.FG.G0',               # ÜFE Endeks (2003=100) - Aylık
    }

    # yfinance Sembolleri
    YF_SYMBOLS = {
        'usd_try': 'TRY=X',
        'eur_try': 'EURTRY=X',
        'bist100_index': 'XU100.IS'
    }

    def __init__(self, api_key: str, start_date: str = "2018-01-01", end_date: Optional[str] = None):
        """
        Args:
            api_key: TCMB EVDS API anahtarı
            start_date: Başlangıç tarihi (DD-MM-YYYY veya YYYY-MM-DD)
            end_date: Bitiş tarihi (DD-MM-YYYY veya YYYY-MM-DD), None ise bugün
        """
        self.api_key = api_key
        self.start_date_evds = self._convert_date_format(start_date, to_evds=True)
        self.end_date_evds = self._convert_date_format(end_date, to_evds=True) if end_date else datetime.now().strftime("%d-%m-%Y")
        
        # yfinance için YYYY-MM-DD formatı
        self.start_date_yf = self._convert_date_format(start_date, to_evds=False)
        self.end_date_yf = self._convert_date_format(end_date, to_evds=False) if end_date else datetime.now().strftime("%Y-%m-%d")

        self.data_dir = "data"
        
        # EVDS API client oluştur
        if EVDS_AVAILABLE:
            self.evds = evdsAPI(self.api_key)
        else:
            self.evds = None

        # Veri dizini oluştur
        os.makedirs(self.data_dir, exist_ok=True)

        logger.info(f"MacroDataFetcher initialized")
        logger.info(f"Date range: {self.start_date_evds} to {self.end_date_evds}")

    def _convert_date_format(self, date_str: Optional[str], to_evds: bool = True) -> Optional[str]:
        """Tarih formatını çevir"""
        if not date_str:
            return None
        
        try:
            # Input formatı tahmin et
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts[0]) == 4: # YYYY-MM-DD
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                else: # DD-MM-YYYY
                    dt = datetime.strptime(date_str, "%d-%m-%Y")
            else:
                return date_str

            if to_evds:
                return dt.strftime("%d-%m-%Y")
            else:
                return dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Date conversion error: {e}")
            return date_str

    def fetch_macro_data(self, save=True) -> pd.DataFrame:
        """Tüm makro verileri çek ve birleştir"""
        data_dict = {}

        # 1. yfinance verilerini çek (Döviz + BIST100)
        logger.info("Fetching market data from yfinance...")
        for name, symbol in self.YF_SYMBOLS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=self.start_date_yf, end=self.end_date_yf)
                
                if not hist.empty:
                    # Timezone kaldır
                    from pandas import DatetimeIndex
                    if isinstance(hist.index, DatetimeIndex) and hist.index.tz is not None:
                        hist.index = hist.index.tz_localize(None)
                    data_dict[name] = hist['Close']
                    logger.info(f"  ✓ {name}: {len(hist)} rows")
                else:
                    logger.warning(f"  ✗ {name}: No data found")
            except Exception as e:
                logger.error(f"Error fetching {name} from yfinance: {e}")

        # 2. EVDS verilerini çek (Enflasyon + Faiz)
        if self.evds:
            logger.info("Fetching economic data from EVDS...")
            
            # 2a. Enflasyon (CPI/PPI)
            try:
                # Endeksleri çek
                series = [self.EVDS_CODES['cpi_index'], self.EVDS_CODES['ppi_index']]
                # Ensure end_date_evds is not None
                end_date = self.end_date_evds or datetime.now().strftime("%d-%m-%Y")
                df_evds = self.evds.get_data(series, startdate=self.start_date_evds, enddate=end_date)
                
                if df_evds is not None and not df_evds.empty:
                    # Tarih indeksini ayarla
                    df_evds['Tarih'] = pd.to_datetime(df_evds['Tarih'], format="%d-%m-%Y")
                    df_evds.set_index('Tarih', inplace=True)
                    
                    # CPI Inflation Calculation (YoY)
                    cpi_col = self.EVDS_CODES['cpi_index']
                    if cpi_col in df_evds.columns:
                        # Yıllık değişim hesapla (12 ay öncesine göre)
                        cpi_inflation = df_evds[cpi_col].pct_change(12) * 100
                        # Günlük frekansa genişlet (ffill)
                        cpi_daily = cpi_inflation.resample('D').ffill()
                        data_dict['cpi_inflation'] = cpi_daily
                        logger.info(f"  ✓ cpi_inflation: Calculated from index")

                    # PPI Inflation Calculation (YoY)
                    ppi_col = self.EVDS_CODES['ppi_index']
                    if ppi_col in df_evds.columns:
                        ppi_inflation = df_evds[ppi_col].pct_change(12) * 100
                        ppi_daily = ppi_inflation.resample('D').ffill()
                        data_dict['ppi_inflation'] = ppi_daily
                        logger.info(f"  ✓ ppi_inflation: Calculated from index")
                        
            except Exception as e:
                logger.error(f"Error fetching inflation data: {e}")

            # 2b. Politika Faizi (Fallback mekanizmalı)
            try:
                # Önce API'yi dene
                pr_col = self.EVDS_CODES['policy_rate']
                # Ensure end_date_evds is not None
                end_date = self.end_date_evds or datetime.now().strftime("%d-%m-%Y")
                df_pr = self.evds.get_data([pr_col], startdate=self.start_date_evds, enddate=end_date)
                
                if df_pr is not None and not df_pr.empty and pr_col in df_pr.columns:
                    df_pr['Tarih'] = pd.to_datetime(df_pr['Tarih'], format="%d-%m-%Y")
                    df_pr.set_index('Tarih', inplace=True)
                    data_dict['policy_rate'] = df_pr[pr_col]
                    logger.info(f"  ✓ policy_rate: Fetched from EVDS")
                else:
                    raise ValueError("Empty response for Policy Rate")
                    
            except Exception as e:
                logger.warning(f"Could not fetch Policy Rate from EVDS ({e}). Using fallback value (50.0).")
                # Fallback: 50.0 sabit değer
                # Ensure dates are not None
                start = self.start_date_yf or "2018-01-01"
                end = self.end_date_yf or datetime.now().strftime("%Y-%m-%d")
                dates = pd.date_range(start=start, end=end, freq='D')
                data_dict['policy_rate'] = pd.Series(50.0, index=dates)

        # 3. Verileri Birleştir
        if not data_dict:
            raise ValueError("No macro data could be fetched!")

        # DataFrame oluştur
        macro_df = pd.DataFrame(data_dict)
        
        # Eksik verileri doldur (Forward fill sonra Backward fill)
        macro_df = macro_df.ffill().bfill()
        
        # Son kontroller
        required_cols = ['policy_rate', 'cpi_inflation', 'ppi_inflation', 'usd_try', 'eur_try', 'bist100_index']
        for col in required_cols:
            if col not in macro_df.columns:
                logger.warning(f"Missing column {col}, filling with 0")
                macro_df[col] = 0.0

        logger.info(f"\nMacro Data Summary:")
        logger.info(f"Rows: {len(macro_df)}")
        logger.info(f"Columns: {macro_df.columns.tolist()}")
        
        if save:
            self.save_data(macro_df, 'macro_data.csv')

        return macro_df

    def save_data(self, df: pd.DataFrame, filename: str):
        """Veriyi kaydet"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        logger.info(f"Macro data saved to {filepath}")

    def load_data(self, filename: str = 'macro_data.csv') -> pd.DataFrame:
        """Veriyi yükle"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded macro data: {len(df)} rows")
        return df

def main():
    """Test script"""
    API_KEY = "tV4qq6RzPr"  # User's key
    
    fetcher = MacroDataFetcher(api_key=API_KEY, start_date="2018-01-01")
    try:
        df = fetcher.fetch_macro_data(save=True)
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print("\nStats:")
        print(df.describe())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
