"""
Altın Veri Çekme Modülü
Ons altın (USD) ve gram altın (TRY) için veri sağlar.
Birincil kaynak: yfinance, opsiyonel: borsapy
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)

TROY_OZ_TO_GRAM = 31.1035

GOLD_OZ_SYMBOL = 'GC=F'
USDTRY_SYMBOL = 'USDTRY=X'
GOLD_GRAM_SYMBOL = 'GOLD_GRAM_TRY'

GOLD_DATA_FILE = os.path.join('data', 'gold_data.csv')


class GoldFetcher:
    """Altın ve döviz verileri için veri çekici.

    DataFetcher ile aynı retry/cache/multi-index pattern'ini kullanır.
    """

    _cache: Dict[str, pd.DataFrame] = {}

    def __init__(self, start_date: str = "2018-01-01", end_date: Optional[str] = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Yardımcı metotlar
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """yfinance üzerinden tek sembol çek (exponential backoff)."""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    back_adjust=False,
                )

                if df is None or df.empty:
                    logger.warning(f"yfinance: {symbol} için veri bulunamadı")
                    return None

                df.columns = df.columns.str.lower()
                available = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
                df = df[available]
                for col in ['open', 'high', 'low', 'close']:
                    if col not in df.columns:
                        df[col] = np.nan
                if 'volume' not in df.columns:
                    df['volume'] = 0
                df['volume'] = df['volume'].fillna(0)
                # Strip timezone from index
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df[['open', 'high', 'low', 'close', 'volume']]

            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"  yfinance {symbol} deneme {attempt + 1}/{max_retries} başarısız: {exc}."
                        f" {wait}s bekleniyor..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"yfinance {symbol} {max_retries} denemede başarısız: {exc}")
                    return None

    def _fetch_borsapy_gold(self) -> Optional[pd.DataFrame]:
        """borsapy ile ons altın (USD) verisi çek."""
        try:
            from borsapy import FX  # type: ignore
            fx = FX("GOLD")
            df = fx.get_historical(self.start_date, self.end_date)

            if df is None or df.empty:
                logger.warning("borsapy: GOLD verisi boş döndü")
                return None

            df.columns = df.columns.str.lower()
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    df[col] = df.get('close', df.iloc[:, 0])
            if 'volume' not in df.columns:
                df['volume'] = 0

            df.index = pd.to_datetime(df.index)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except ImportError:
            logger.info("borsapy kurulu değil, yfinance fallback kullanılıyor")
            return None
        except Exception as exc:
            logger.warning(f"borsapy GOLD hatası: {exc}")
            return None

    def _fetch_borsapy_usd(self) -> Optional[pd.DataFrame]:
        """borsapy ile USD/TRY verisi çek."""
        try:
            from borsapy import FX  # type: ignore
            fx = FX("USD")
            df = fx.get_historical(self.start_date, self.end_date)

            if df is None or df.empty:
                logger.warning("borsapy: USD verisi boş döndü")
                return None

            df.columns = df.columns.str.lower()
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    df[col] = df.get('close', df.iloc[:, 0])
            if 'volume' not in df.columns:
                df['volume'] = 0

            df.index = pd.to_datetime(df.index)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except ImportError:
            return None
        except Exception as exc:
            logger.warning(f"borsapy USD hatası: {exc}")
            return None

    # ------------------------------------------------------------------
    # Ana çekme metotları
    # ------------------------------------------------------------------

    def fetch_gold_oz(self) -> Optional[pd.DataFrame]:
        """Ons altın (USD) verisi çek. Önce borsapy, yoksa yfinance."""
        logger.info("Ons altın verisi çekiliyor...")

        df = self._fetch_borsapy_gold()
        if df is not None and not df.empty:
            logger.info(f"  borsapy GOLD: {len(df)} satır")
            return df

        logger.info("  borsapy başarısız, yfinance fallback...")
        df = self._fetch_yfinance(GOLD_OZ_SYMBOL)
        if df is not None:
            logger.info(f"  yfinance {GOLD_OZ_SYMBOL}: {len(df)} satır")
        return df

    def fetch_usdtry(self) -> Optional[pd.DataFrame]:
        """USD/TRY kur verisi çek. Önce borsapy, yoksa yfinance."""
        logger.info("USD/TRY verisi çekiliyor...")

        df = self._fetch_borsapy_usd()
        if df is not None and not df.empty:
            logger.info(f"  borsapy USD: {len(df)} satır")
            return df

        logger.info("  borsapy başarısız, yfinance fallback...")
        df = self._fetch_yfinance(USDTRY_SYMBOL)
        if df is not None:
            logger.info(f"  yfinance {USDTRY_SYMBOL}: {len(df)} satır")
        return df

    def compute_gold_gram_try(
        self, gold_oz_df: pd.DataFrame, usdtry_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Gram altın (TRY) = (ons_usd * usd_try) / 31.1035 olarak hesapla."""
        logger.info("Gram altın (TRY) hesaplanıyor...")

        # Ortak tarihlere hizala
        gold_close = gold_oz_df['close'].rename('gold_oz_close')
        usd_close = usdtry_df['close'].rename('usdtry_close')

        # İndeks timezone'u normalize et
        def _strip_idx(s):
            idx = pd.to_datetime(s.index)
            s.index = idx.tz_localize(None) if idx.tz is not None else idx
            return s
        gold_close = _strip_idx(gold_close)
        usd_close = _strip_idx(usd_close)

        merged = pd.concat([gold_close, usd_close], axis=1).dropna()

        if merged.empty:
            raise ValueError("Gram altın hesaplanamadı: ortak tarih yok")

        gram_close = (merged['gold_oz_close'] * merged['usdtry_close']) / TROY_OZ_TO_GRAM

        # OHLCV formatı oluştur
        def _aligned(df):
            df = df.copy()
            idx = pd.to_datetime(df.index)
            df.index = idx.tz_localize(None) if idx.tz is not None else idx
            return df
        gold_oz_df_aligned = _aligned(gold_oz_df)
        usdtry_df_aligned = _aligned(usdtry_df)

        gram_df = pd.DataFrame(index=merged.index)
        gram_df['close'] = gram_close

        # Open/High/Low için de dönüşüm uygula (tutarlı format)
        for col in ['open', 'high', 'low']:
            gold_col = gold_oz_df_aligned[col].reindex(merged.index)
            usd_col = usdtry_df_aligned[col].reindex(merged.index)
            gram_df[col] = (gold_col * usd_col) / TROY_OZ_TO_GRAM

        gram_df['volume'] = 0
        gram_df = gram_df[['open', 'high', 'low', 'close', 'volume']].dropna()

        logger.info(f"  Gram altın: {len(gram_df)} satır, "
                    f"son fiyat: {gram_df['close'].iloc[-1]:.2f} TRY")
        return gram_df

    def fetch_all_gold_data(self, save: bool = False) -> pd.DataFrame:
        """3 sembol için multi-index DataFrame döndür: GC=F, USDTRY=X, GOLD_GRAM_TRY.

        Returns:
            Multi-index DataFrame ('symbol', 'date') index ile
        """
        cache_key = f"gold_{self.start_date}_{self.end_date}"
        if cache_key in GoldFetcher._cache:
            logger.info("Altın verisi cache'den alınıyor")
            return GoldFetcher._cache[cache_key].copy()

        all_data: dict = {}

        def _strip(df: pd.DataFrame) -> pd.DataFrame:
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            df.index = idx
            return df

        # 1. Ons altın
        gold_oz = self.fetch_gold_oz()
        if gold_oz is not None:
            gold_oz = _strip(gold_oz)
            all_data[GOLD_OZ_SYMBOL] = gold_oz
        else:
            logger.error("Ons altın verisi çekilemedi!")

        # 2. USD/TRY
        usdtry = self.fetch_usdtry()
        if usdtry is not None:
            usdtry = _strip(usdtry)
            all_data[USDTRY_SYMBOL] = usdtry
        else:
            logger.error("USD/TRY verisi çekilemedi!")

        # 3. Gram altın (hesaplanmış)
        if gold_oz is not None and usdtry is not None:
            try:
                gram_gold = self.compute_gold_gram_try(gold_oz, usdtry)
                all_data[GOLD_GRAM_SYMBOL] = gram_gold
            except Exception as exc:
                logger.error(f"Gram altın hesaplanamadı: {exc}")
        else:
            logger.warning("Gram altın hesaplanamadı: eksik kaynak veri")

        if not all_data:
            raise ValueError("Hiçbir altın/döviz verisi çekilemedi")

        combined = pd.concat(list(all_data.values()), keys=list(all_data.keys()))
        combined.index.names = ['symbol', 'date']

        logger.info(f"\nToplam altın/döviz satırı: {len(combined)}")
        for sym in combined.index.get_level_values('symbol').unique():
            sub = combined.xs(sym, level='symbol')
            logger.info(f"  {sym}: {len(sub)} satır "
                        f"({sub.index.min().date()} - {sub.index.max().date()})")

        GoldFetcher._cache[cache_key] = combined.copy()
        return combined

    # ------------------------------------------------------------------
    # Persistence ve incremental metodları
    # ------------------------------------------------------------------

    def save_gold_data(self, df: pd.DataFrame, filepath: str = GOLD_DATA_FILE) -> None:
        """Multi-index DataFrame'i CSV olarak kaydet."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df.to_csv(filepath, index=True)
        logger.info(f"Altın verisi kaydedildi: {filepath} ({len(df)} satır)")

    @staticmethod
    def load_gold_data(filepath: str = GOLD_DATA_FILE) -> Optional[pd.DataFrame]:
        """Kaydedilmiş altın verisini yükle. Multi-index (symbol, date) döner."""
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            dates = pd.to_datetime(df['date'])
            if getattr(dates.dt, 'tz', None) is not None:
                dates = dates.dt.tz_localize(None)
            df['date'] = dates
            df = df.set_index(['symbol', 'date'])
            logger.info(f"Altın verisi yüklendi: {filepath} ({len(df)} satır)")
            return df
        except Exception as exc:
            logger.error(f"Altın verisi yüklenemedi: {exc}")
            return None

    @staticmethod
    def _normalize_df_index(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-index'teki tarih timezone bilgisini kaldır."""
        df = df.reset_index()
        dates = pd.to_datetime(df['date'])
        if getattr(dates.dt, 'tz', None) is not None:
            dates = dates.dt.tz_localize(None)
        df['date'] = dates
        return df.set_index(['symbol', 'date'])

    def get_source_status(self, filepath: str = GOLD_DATA_FILE) -> dict:
        """Dosyadaki mevcut veri durumunu raporla."""
        today = datetime.now().date()
        df = GoldFetcher.load_gold_data(filepath)
        if df is None or df.empty:
            return {
                'exists': False,
                'last_date': None,
                'missing_days': None,
                'symbols': [],
            }
        all_dates = pd.to_datetime(df.index.get_level_values('date'))
        last_date = all_dates.max().date()
        missing = sum(
            1 for i in range(1, (today - last_date).days + 1)
            if (last_date + timedelta(days=i)).weekday() < 5
        )
        symbols = df.index.get_level_values('symbol').unique().tolist()
        return {
            'exists': True,
            'last_date': str(last_date),
            'missing_days': missing,
            'symbols': symbols,
        }

    def fetch_incremental(self, filepath: str = GOLD_DATA_FILE) -> dict:
        """Sadece eksik günleri çek ve mevcut dosyaya ekle."""
        today_str = datetime.now().strftime('%Y-%m-%d')
        existing = GoldFetcher.load_gold_data(filepath)

        if existing is None or existing.empty:
            logger.info("Altın verisi bulunamadı, tam çekme yapılıyor...")
            df = self.fetch_all_gold_data(save=False)
            df = GoldFetcher._normalize_df_index(df)
            self.save_gold_data(df, filepath)
            return {'mode': 'full', 'new_rows': len(df), 'total_rows': len(df)}

        all_dates = pd.to_datetime(existing.index.get_level_values('date'))
        min_last_date = all_dates.max().date()
        fetch_from = (datetime.combine(min_last_date, datetime.min.time()) + timedelta(days=1)).strftime('%Y-%m-%d')

        if fetch_from > today_str:
            logger.info("Altın verisi güncel, çekme gerekmez")
            return {'mode': 'skip', 'new_rows': 0, 'total_rows': len(existing)}

        logger.info(f"Eksik altın verisi çekiliyor: {fetch_from} → {today_str}")
        delta_fetcher = GoldFetcher(start_date=fetch_from, end_date=today_str)
        new_data = delta_fetcher.fetch_all_gold_data(save=False)
        new_data = GoldFetcher._normalize_df_index(new_data)

        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()

        self.save_gold_data(combined, filepath)
        return {
            'mode': 'incremental',
            'new_rows': len(new_data),
            'total_rows': len(combined),
            'fetch_from': fetch_from,
            'fetch_to': today_str,
        }

    def get_latest_prices(self) -> dict:
        """Güncel ons altın, gram altın ve USD/TRY fiyatlarını döndür."""
        try:
            df = self.fetch_all_gold_data()
        except Exception as exc:
            logger.error(f"Güncel altın fiyatı çekilemedi: {exc}")
            return {}

        prices = {}
        for sym in [GOLD_OZ_SYMBOL, USDTRY_SYMBOL, GOLD_GRAM_SYMBOL]:
            try:
                sub = df.xs(sym, level='symbol')
                latest = sub.sort_index().iloc[-1]
                prev = sub.sort_index().iloc[-2] if len(sub) > 1 else latest
                change_pct = ((latest['close'] - prev['close']) / prev['close'] * 100
                              if prev['close'] != 0 else 0.0)
                prices[sym] = {
                    'close': float(latest['close']),
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'change_pct': float(change_pct),
                    'date': str(sub.index[-1].date()),
                }
            except Exception as exc:
                logger.warning(f"  {sym} son fiyat alınamadı: {exc}")

        return prices
