"""
Veri Çekme Modülü
BIST-30 hisseleri için Yahoo Finance'den günlük OHLCV verisi çeker
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional
import os
import logging
import warnings

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Borsa saat dilimi. yfinance tz-aware timestamp dondurur; tarihleri bu
# dilimde naive'e indiriyoruz ki takvim gunu yazildigi gibi kalsin.
MARKET_TZ = 'Europe/Istanbul'

# BIST icin en erken kullanilabilir tarih. Daha eskisi istenirse buraya cekilir.
#
# Gerekce: 2005 oncesi yfinance BIST verisi bozuk. 2005'teki TL/YTL
# denominasyonunun (1 YTL = 1.000.000 TL) geriye-duzeltmesi negatif fiyat
# uretiyor — olculdu: 2000-2004 arasi 8.155 satir (%4,5), 2003'te satirlarin
# %42'si negatif. clean_data() bunlari zaten dusuruyor, yani daha eskiye gitmek
# veri kazandirmiyor; sadece indirme suresi ve gurultu ekliyor.
#
# Yalnizca BIST'i baglar: altin/makro/fundamental ayri fetcher siniflaridir.
BIST_MIN_START_DATE = '2005-01-01'


def to_naive_market_dates(values, tz: str = MARKET_TZ) -> pd.DatetimeIndex:
    """Tarihleri borsa yerel saatinde tz'siz (naive) datetime64'e normalize et.

    Neden gerekli: yfinance tz-aware timestamp dondurur ve CSV'ye UTC ofsetiyle
    yazilir. Turkiye 2016 Eylul'e kadar yaz saati uyguladigi icin uzun gecmisli
    dosyalarda +02:00 (kis) ve +03:00 (yaz) satirlari yan yana bulunur. pandas
    2.x boyle bir sutunu datetime64'e cevirmez -> object dtype + FutureWarning,
    ardindan pd.to_datetime "Tz-aware datetime.datetime cannot be converted to
    datetime64 unless utc=True" hatasi verir.

    Cozum: once utc=True ile tek tipe indir, sonra borsa saat dilimine geri
    cevirip tz'yi at. Dogrudan UTC'de birakmak kis satirlarini bir gun geriye
    kaydirirdi (2000-10-30 00:00+02:00 -> 2000-10-29 22:00). Bu yol takvim
    gununu korur.

    Zaten tz'siz gelen degerler oldugu gibi dondurulur — UTC varsayip yerel
    saate cevirmek her gidis-donuste tarihi +3 saat kaydirirdi.
    """
    idx = pd.Index(values)
    if len(idx) == 0:
        return pd.DatetimeIndex([])

    # Tz'siz girdiyi kaydirmadan gec. Karisik ofsetli girdi burada ValueError
    # firlatir; asagidaki utc=True yolu onu toparlar.
    try:
        with warnings.catch_warnings():
            # Karisik ofsette pandas 2.x once uyarir, ileride hata firlatacak;
            # iki durumu da asagidaki utc=True yolu karsiliyor.
            warnings.simplefilter('ignore', FutureWarning)
            parsed = pd.to_datetime(idx)
        if getattr(parsed, 'tz', None) is None and parsed.dtype.kind == 'M':
            return pd.DatetimeIndex(parsed)
    except (ValueError, TypeError):
        pass

    utc = pd.DatetimeIndex(pd.to_datetime(idx, utc=True))
    return utc.tz_convert(tz).tz_localize(None)


def _default_cache_maxsize() -> int:
    """Cache tavanini config'ten oku (env override); config yoksa 64."""
    try:
        from app.core.config import get_settings
        return int(get_settings().DATA_CACHE_MAXSIZE)
    except Exception:
        return 64


class DataFetcher:
    """BIST-30 hisse verileri için veri çekici"""

    # Class-level LRU cache to share data across instances (Faz 6 · B8).
    # OrderedDict = insertion/erisim sirasi -> en eski atilir. Sinirsiz dict
    # batch egitimde RAM'i sisiriyordu. maxsize=0 -> sinirsiz (eski davranis).
    _cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    _cache_maxsize: int = _default_cache_maxsize()

    @classmethod
    def _cache_get(cls, key: str) -> Optional[pd.DataFrame]:
        """LRU okuma: hit ise anahtari 'en yeni' konumuna tasi."""
        if key not in cls._cache:
            return None
        cls._cache.move_to_end(key)
        return cls._cache[key]

    @classmethod
    def _cache_put(cls, key: str, value: pd.DataFrame) -> None:
        """LRU yazma: ekle, tavani asarsa en eski (LRU) girdiyi at."""
        cls._cache[key] = value
        cls._cache.move_to_end(key)
        if cls._cache_maxsize and len(cls._cache) > cls._cache_maxsize:
            evicted_key, _ = cls._cache.popitem(last=False)  # en eski
            logger.debug("Cache tavani (%d) asildi, atilan: %s",
                         cls._cache_maxsize, evicted_key)

    @staticmethod
    def _clamp_start_date(start_date: Optional[str]) -> Tuple[str, bool]:
        """BIST_MIN_START_DATE'ten eski istekleri o tarihe cek.

        (kullanilacak_tarih, cekildi_mi) doner. Tarih cozulemezse dokunmadan
        gecer — bicim dogrulamasi bu metodun isi degil, cagiran taraf/yfinance
        kendi hatasini versin.
        """
        if not start_date:
            return BIST_MIN_START_DATE, False
        try:
            requested = pd.to_datetime(start_date).date()
            floor = pd.to_datetime(BIST_MIN_START_DATE).date()
        except (ValueError, TypeError):
            return start_date, False

        if requested >= floor:
            return start_date, False

        logger.warning(
            "BIST baslangic tarihi %s -> %s olarak cekildi. 2005 oncesi yfinance "
            "verisi TL/YTL denominasyonu yuzunden negatif fiyat iceriyor.",
            start_date, BIST_MIN_START_DATE,
        )
        return BIST_MIN_START_DATE, True

    def __init__(self, start_date: str = "2018-01-01", end_date: Optional[str] = None,
                 max_workers: int = 8):
        """
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD), None ise bugün
            max_workers: Paralel sembol çekme için thread sayısı (Faz 6, 1.1).
                         yfinance rate-limit'ine saygılı; 1 = seri (eski davranış).
        """
        self.start_date, self.start_date_clamped = self._clamp_start_date(start_date)
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = "data/bist"
        self.max_workers = max(1, int(max_workers))

        # Veri dizinini oluştur
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        (#13) Single symbol fetch with exponential backoff retry.
        Returns DataFrame or None if all attempts fail.
        """
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)

                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    return None

                df.columns = df.columns.str.lower()
                df = df[['open', 'high', 'low', 'close', 'volume']]
                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Error fetching {symbol} after {max_retries} attempts: {e}")
                    return None

    def fetch_stock_data(self, symbols: List[str], save=True) -> pd.DataFrame:
        """
        Birden fazla hisse için OHLCV verisi çek

        Args:
            symbols: Hisse sembolleri listesi
            save: CSV olarak kaydet

        Returns:
            Multi-index DataFrame (Date, Symbol)
        """
        # Create cache key
        cache_key = f"{'-'.join(sorted(symbols))}_{self.start_date}_{self.end_date}"

        # Check cache first (LRU — hit anahtari en yeni konuma tasir)
        cached = DataFetcher._cache_get(cache_key)
        if cached is not None:
            logger.info(f"✓ Using cached data for {len(symbols)} symbols ({self.start_date} to {self.end_date})")
            return cached.copy()

        logger.info(f"Fetching data for {len(symbols)} symbols...")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")

        # Calculate expected trading days for coverage check (#14)
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        calendar_days = (end_dt - start_dt).days
        # ~252 trading days/year; approximate expected rows
        expected_rows = int(calendar_days * 252 / 365)

        def _fetch_one(symbol):
            """Tek sembol çek + coverage kontrolü. (symbol, df|None) döndürür."""
            df = self._fetch_with_retry(symbol)  # (#13) retry logic
            if df is None:
                return symbol, None
            # (#14) Minimum coverage check: require ≥80% of expected trading days
            if expected_rows > 0 and len(df) < expected_rows * 0.8:
                coverage_pct = len(df) / expected_rows * 100
                logger.warning(
                    f"  ⚠ {symbol}: low coverage {coverage_pct:.0f}% "
                    f"({len(df)} rows, expected ~{expected_rows})"
                )
            return symbol, df

        # Faz 6 (1.1): Sembolleri paralel çek (I/O-bound → ThreadPool, GIL salınır).
        # Sonuçları symbols sırasına göre topla — concat deterministik kalsın.
        results = {}
        if self.max_workers == 1 or len(symbols) == 1:
            for symbol in symbols:
                sym, df = _fetch_one(symbol)
                results[sym] = df
        else:
            from concurrent.futures import ThreadPoolExecutor
            workers = min(self.max_workers, len(symbols))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for sym, df in pool.map(_fetch_one, symbols):
                    results[sym] = df

        all_data = {}
        for symbol in symbols:  # deterministik sıra
            df = results.get(symbol)
            if df is None:
                continue
            all_data[symbol] = df
            logger.info(f"  ✓ {symbol}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

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

        # Cache the result (LRU — tavan asilirsa en eski girdi atilir)
        DataFetcher._cache_put(cache_key, combined_df.copy())
        logger.info(f"✓ Data cached for future use")

        return combined_df

    def save_data(self, df: pd.DataFrame, filename: str):
        """DataFrame'i CSV olarak kaydet"""
        filepath = os.path.join(self.data_dir, filename)
        # Tarihleri ofsetsiz yaz: aksi halde yaz/kis saati gecisleri yuzunden
        # dosyada +02:00 ve +03:00 birlikte olusur ve okuma tarafi patlar.
        if 'date' in (df.index.names or []):
            dates = to_naive_market_dates(df.index.get_level_values('date'))
            df = df.copy()
            df.index = pd.MultiIndex.from_arrays(
                [df.index.get_level_values(name) if name != 'date' else dates
                 for name in df.index.names],
                names=df.index.names,
            )
        # Reset index to avoid duplicate symbol column in CSV
        df.to_csv(filepath, index=True)
        logger.info(f"Data saved to {filepath}")

    def load_data(self, filename: str = 'raw_stock_data.csv') -> pd.DataFrame:
        """Kaydedilmiş veriyi yükle"""
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # date'i pandas'a parse ettirmiyoruz: karisik UTC ofsetli dosyalarda
        # object dtype + FutureWarning uretiyor. Ham okuyup normalize ediyoruz.
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df['date'] = to_naive_market_dates(df['date'])

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

            # Forward fill NaN değerler (#12)
            symbol_df = symbol_df.ffill()

            # Geriye kalan NaN'ları backward fill
            symbol_df = symbol_df.bfill()

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

    def get_source_status(self, symbols: List[str], filename: str = 'raw_stock_data.csv') -> dict:
        """Dosyadaki mevcut veri durumunu raporla."""
        today = datetime.now().date()
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return {
                'exists': False,
                'last_date': None,
                'missing_days': None,
                'symbols': [],
            }

        try:
            df = self.load_data(filename)
            all_dates = to_naive_market_dates(df.index.get_level_values('date'))
            last_date = all_dates.max().date()
            missing = sum(
                1 for i in range(1, (today - last_date).days + 1)
                if (last_date + timedelta(days=i)).weekday() < 5
            )
            existing_symbols = df.index.get_level_values('symbol').unique().tolist()
            return {
                'exists': True,
                'last_date': str(last_date),
                'missing_days': missing,
                'symbols': existing_symbols,
            }
        except Exception as exc:
            return {
                'exists': True,
                'last_date': None,
                'missing_days': None,
                'error': str(exc),
                'symbols': [],
            }

    def fetch_incremental(self, symbols: List[str], filename: str = 'raw_stock_data.csv') -> dict:
        """Sadece eksik günleri çek ve mevcut dosyaya ekle.

        Mevcut dosyadan son tarihi okur, sadece eksik delta'yı çeker ve append eder.
        """
        today_str = datetime.now().strftime('%Y-%m-%d')
        filepath = os.path.join(self.data_dir, filename)

        existing = None
        if os.path.exists(filepath):
            try:
                existing = self.load_data(filename)
            except Exception as exc:
                logger.warning(f"Mevcut veri yüklenemedi, tam çekme yapılıyor: {exc}")

        if existing is None or existing.empty:
            logger.info(f"Mevcut {filename} bulunamadı, tam çekme yapılıyor...")
            df = self.fetch_stock_data(symbols, save=True)
            return {'mode': 'full', 'new_rows': len(df), 'total_rows': len(df)}

        all_dates = to_naive_market_dates(existing.index.get_level_values('date'))
        min_last_date = all_dates.max().date()
        fetch_from = (datetime.combine(min_last_date, datetime.min.time()) + timedelta(days=1)).strftime('%Y-%m-%d')

        if fetch_from > today_str:
            logger.info(f"{filename} güncel, çekme gerekmez")
            return {'mode': 'skip', 'new_rows': 0, 'total_rows': len(existing)}

        logger.info(f"Eksik BIST verisi çekiliyor: {fetch_from} → {today_str}")
        delta_fetcher = DataFetcher(start_date=fetch_from, end_date=today_str)
        new_data = delta_fetcher.fetch_stock_data(symbols, save=False)

        # Normalize timezone
        new_data = new_data.reset_index()
        new_data['date'] = to_naive_market_dates(new_data['date'])
        new_data = new_data.set_index(['symbol', 'date'])

        existing = existing.reset_index()
        dates_ex = pd.to_datetime(existing['date'])
        if getattr(dates_ex.dt, 'tz', None) is not None:
            dates_ex = dates_ex.dt.tz_localize(None)
        existing['date'] = dates_ex
        existing = existing.set_index(['symbol', 'date'])

        # yfinance seans kapanmadan once OHLC'si NaN, volume'u DOLU bir taslak
        # satir dondurur. Diske yazilirsa iki zarar birden:
        #   1) panelde tamamen bos bir gun kalir (28.08.2026'da 30 sembolun
        #      tamami boyle gelmisti),
        #   2) `min_last_date` o gune kayar, `fetch_from` ertesi gun olur ve o
        #      seansin GERCEK verisi bir daha HIC cekilmez — kalici bosluk.
        # Filtre YALNIZCA yeni veriye uygulanir: `existing` ham panel ve icinde
        # 2005 oncesi duzeltme artefaktlarindan gelen negatif fiyatlar var
        # (bkz. CLAUDE.md "Veri butunlugu"); onlari burada silmek gecmisi
        # sessizce degistirirdi.
        stub_rows = int((~(new_data['close'].notna() & (new_data['close'] > 0))).sum())
        if stub_rows:
            new_data = new_data[new_data['close'].notna() & (new_data['close'] > 0)]
            logger.warning(
                f"{stub_rows} taslak satir (gecersiz kapanis) diske yazilmadi; "
                f"o seans bir sonraki cekimde yeniden denenecek"
            )

        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()

        self.save_data(combined, filename)
        return {
            'mode': 'incremental',
            'new_rows': len(new_data),
            'skipped_stub_rows': stub_rows,
            'total_rows': len(combined),
            'fetch_from': fetch_from,
            'fetch_to': today_str,
        }

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
