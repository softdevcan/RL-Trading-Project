"""
Altın ve Döviz Veri Çekme Modülü

Desteklenen metrikler:
  gram_altin_try  : Gram Altın (TRY)   — borsapy: FX("gram-altin") | yfinance: hesaplanmış
  ons_altin_try   : Ons Altın (TRY)    — borsapy: FX("ons-altin")  | yfinance: hesaplanmış
  ons_altin_usd   : Ons Altın (USD)    — yfinance only: GC=F
  gram_altin_usd  : Gram Altın (USD)   — yfinance only: GC=F / 31.1035
  usd_try         : USD/TRY            — borsapy: FX("USD")  | yfinance: TRY=X
  eur_try         : EUR/TRY            — borsapy: FX("EUR")  | yfinance: EURTRY=X

Kaynak seçimi:
  source="borsapy"  → borsapy kullanılır (ons/gram USD metrikler desteklenmez)
  source="yfinance" → yfinance kullanılır (TRY metrikler GC=F × USDTRY'den hesaplanır)

Kullanım:
  fetcher = GoldFetcher(source="borsapy", metrics=["gram_altin_try", "usd_try", "eur_try"])
  df = fetcher.fetch_all()

  fetcher = GoldFetcher(source="yfinance", metrics=["ons_altin_usd", "gram_altin_usd", "usd_try"])
  df = fetcher.fetch_all()
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

TROY_OZ_TO_GRAM = 31.1035

# Kaynak → sembol eşlemeleri
BORSAPY_SYMBOLS: Dict[str, str] = {
    "gram_altin_try": "gram-altin",
    "ons_altin_try":  "ons-altin",
    "usd_try":        "USD",
    "eur_try":        "EUR",
}

YFINANCE_SYMBOLS: Dict[str, str] = {
    "ons_altin_usd": "GC=F",
    "usd_try":       "TRY=X",
    "eur_try":       "EURTRY=X",
}

# Multi-index çıktısında kullanılacak sembol adları
OUTPUT_SYMBOL: Dict[str, str] = {
    "gram_altin_try": "GOLD_GRAM_TRY",
    "ons_altin_try":  "GOLD_ONS_TRY",
    "ons_altin_usd":  "GC=F",
    "gram_altin_usd": "GOLD_GRAM_USD",
    "usd_try":        "USDTRY=X",
    "eur_try":        "EURTRY=X",
}

METRIC_INFO: Dict[str, Dict] = {
    "gram_altin_try": {
        "desc": "Gram Altın (TRY)",
        "borsapy": True,
        "yfinance": "computed",   # GC=F × USDTRY / 31.1035
    },
    "ons_altin_try": {
        "desc": "Ons Altın (TRY)",
        "borsapy": True,
        "yfinance": "computed",   # GC=F × USDTRY
    },
    "ons_altin_usd": {
        "desc": "Ons Altın (USD)",
        "borsapy": False,         # borsapy'de yok
        "yfinance": "direct",     # GC=F
    },
    "gram_altin_usd": {
        "desc": "Gram Altın (USD)",
        "borsapy": False,         # borsapy'de yok
        "yfinance": "computed",   # GC=F / 31.1035
    },
    "usd_try": {
        "desc": "USD/TRY",
        "borsapy": True,
        "yfinance": "direct",
    },
    "eur_try": {
        "desc": "EUR/TRY",
        "borsapy": True,
        "yfinance": "direct",
    },
}

DEFAULT_METRICS = ["gram_altin_try", "ons_altin_usd", "usd_try", "eur_try"]
GOLD_DIR        = os.path.join("data", "gold")


def gold_filepath(source: str, name: Optional[str] = None) -> str:
    """Kaynak ve opsiyonel özel isme göre dosya yolu üret."""
    filename = f"{name}.csv" if name else f"gold_{source}.csv"
    return os.path.join(GOLD_DIR, filename)


class GoldFetcher:
    """Altın ve döviz veri çekici.

    Kaynak (borsapy / yfinance) ve metrik seçimi desteklenir.
    Fallback yoktur: kaynak açıkça belirtilmelidir.
    """

    _cache: Dict[str, pd.DataFrame] = {}

    def __init__(
        self,
        source: str = "borsapy",
        metrics: Optional[List[str]] = None,
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
    ):
        """
        Args:
            source    : "borsapy" veya "yfinance"
            metrics   : Çekilecek metrik listesi. None ise DEFAULT_METRICS kullanılır.
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date  : Bitiş tarihi (YYYY-MM-DD), None ise bugün
        """
        if source not in ("borsapy", "yfinance"):
            raise ValueError(f"Geçersiz kaynak: {source!r}. 'borsapy' veya 'yfinance' olmalı.")

        self.source     = source
        self.metrics    = metrics if metrics is not None else DEFAULT_METRICS
        self.start_date = start_date
        self.end_date   = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir   = GOLD_DIR
        os.makedirs(self.data_dir, exist_ok=True)

        self._validate_metrics()

        logger.info(f"GoldFetcher | source={source} | metrics={self.metrics}")
        logger.info(f"Tarih: {self.start_date} → {self.end_date}")

    # ------------------------------------------------------------------ #
    # Doğrulama
    # ------------------------------------------------------------------ #

    def _validate_metrics(self) -> None:
        """Seçilen metriklerin kaynak ile uyumlu olduğunu kontrol et."""
        for metric in self.metrics:
            if metric not in METRIC_INFO:
                raise ValueError(
                    f"Bilinmeyen metrik: {metric!r}\n"
                    f"Geçerli metrikler: {list(METRIC_INFO)}"
                )
            info = METRIC_INFO[metric]
            if self.source == "borsapy" and not info["borsapy"]:
                raise ValueError(
                    f"'{metric}' ({info['desc']}) borsapy ile kullanılamaz. "
                    f"Bu metrik için source='yfinance' seçin."
                )

    # ------------------------------------------------------------------ #
    # borsapy yardımcıları
    # ------------------------------------------------------------------ #

    def _borsapy_fetch(self, metric: str) -> Optional[pd.DataFrame]:
        """borsapy FX ile tek metrik çek."""
        symbol = BORSAPY_SYMBOLS[metric]
        try:
            from borsapy import FX
            fx = FX(symbol)
            df  = fx.history(start=self.start_date, end=self.end_date)
            if df is None or df.empty:
                logger.warning(f"borsapy {symbol}: boş veri döndü")
                return None
            df = self._normalize(df)
            logger.info(f"  ✓ borsapy {symbol}: {len(df)} satır")
            return df
        except Exception as exc:
            logger.error(f"borsapy {symbol} hatası: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # yfinance yardımcıları
    # ------------------------------------------------------------------ #

    def _yfinance_fetch(self, symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """yfinance ile tek sembol çek (exponential backoff)."""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                )
                if df is None or df.empty:
                    logger.warning(f"yfinance {symbol}: veri bulunamadı")
                    return None
                df = self._normalize(df)
                logger.info(f"  ✓ yfinance {symbol}: {len(df)} satır")
                return df
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"yfinance {symbol} deneme {attempt+1}/{max_retries}: {exc}. {wait}s bekleniyor...")
                    time.sleep(wait)
                else:
                    logger.error(f"yfinance {symbol} başarısız: {exc}")
                    return None

    # ------------------------------------------------------------------ #
    # Normalize (ortak çıktı formatı)
    # ------------------------------------------------------------------ #

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV formatına getir, timezone kaldır."""
        df = df.copy()
        df.columns = df.columns.str.lower()
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                df[col] = np.nan
        if "volume" not in df.columns:
            df["volume"] = 0
        df["volume"] = df["volume"].fillna(0)
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize(None) if idx.tz is not None else idx
        return df[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------ #
    # Hesaplama yardımcıları (yfinance computed metrikler için)
    # ------------------------------------------------------------------ #

    def _compute_divide(self, df: pd.DataFrame, divisor: float) -> pd.DataFrame:
        """Her OHLC sütununu divisor'a böl (gram = ons / 31.1035)."""
        result = df.copy()
        for col in ["open", "high", "low", "close"]:
            result[col] = result[col] / divisor
        return result

    def _compute_multiply_usdtry(
        self,
        usd_df: pd.DataFrame,
        usdtry_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """USD cinsinden veriyi TRY'ye çevir: usd_df × usdtry_df (OHLC × OHLC)."""
        idx = usd_df.index.intersection(usdtry_df.index)
        if idx.empty:
            raise ValueError("USD ve USDTRY verilerinde ortak tarih bulunamadı")
        result = pd.DataFrame(index=idx)
        for col in ["open", "high", "low", "close"]:
            result[col] = usd_df.loc[idx, col].values * usdtry_df.loc[idx, col].values
        result["volume"] = 0
        return result[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------ #
    # Metrik dispatch
    # ------------------------------------------------------------------ #

    def _fetch_metric(self, metric: str, pool: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Tek metrik çek. pool: bu sefer zaten çekilmiş veriler."""

        # ---- borsapy path -------------------------------------------- #
        if self.source == "borsapy":
            return self._borsapy_fetch(metric)

        # ---- yfinance path -------------------------------------------- #
        if metric == "ons_altin_usd":
            return self._yfinance_fetch(YFINANCE_SYMBOLS["ons_altin_usd"])

        if metric == "gram_altin_usd":
            ons = pool.get("ons_altin_usd")
            if ons is None:
                ons = self._yfinance_fetch(YFINANCE_SYMBOLS["ons_altin_usd"])
            if ons is None:
                return None
            return self._compute_divide(ons, TROY_OZ_TO_GRAM)

        if metric in ("gram_altin_try", "ons_altin_try"):
            ons = pool.get("ons_altin_usd")
            if ons is None:
                ons = self._yfinance_fetch(YFINANCE_SYMBOLS["ons_altin_usd"])
            usdtry = pool.get("usd_try")
            if usdtry is None:
                usdtry = self._yfinance_fetch(YFINANCE_SYMBOLS["usd_try"])
            if ons is None or usdtry is None:
                return None
            ons_try = self._compute_multiply_usdtry(ons, usdtry)
            if metric == "gram_altin_try":
                return self._compute_divide(ons_try, TROY_OZ_TO_GRAM)
            return ons_try  # ons_altin_try

        if metric == "usd_try":
            return self._yfinance_fetch(YFINANCE_SYMBOLS["usd_try"])

        if metric == "eur_try":
            return self._yfinance_fetch(YFINANCE_SYMBOLS["eur_try"])

        return None

    # ------------------------------------------------------------------ #
    # Ana metot
    # ------------------------------------------------------------------ #

    def default_filepath(self, name: Optional[str] = None) -> str:
        """Bu fetcher için varsayılan dosya yolu."""
        return gold_filepath(self.source, name)

    def _try_load_local(self) -> Optional[pd.DataFrame]:
        """Yerel CSV'den yukle; gerekli sembolleri ve tarih araligini kapsiyorsa dondur.

        Kapsama yetersizse None doner (cagiran taraf internetten ceker).
        """
        filepath = self.default_filepath()
        if not os.path.exists(filepath):
            return None

        df = GoldFetcher.load_data(filepath)
        if df is None or df.empty:
            return None

        # Gereken sembollerin hepsi var mi?
        needed_symbols = {OUTPUT_SYMBOL[m] for m in self.metrics if m in OUTPUT_SYMBOL}
        available = set(df.index.get_level_values("symbol").unique())
        if not needed_symbols.issubset(available):
            logger.debug(f"Yerel CSV eksik sembol(ler): {needed_symbols - available}")
            return None

        # Tarih araligini kontrol et
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates = pd.to_datetime(df.index.get_level_values("date"))
        if dates.min() > start or dates.max() < (end - pd.Timedelta(days=7)):
            # 7 gunluk tolerans — hafta sonu/tatil bosluklarini kabul et
            logger.debug(
                f"Yerel CSV tarih araligi yetersiz: "
                f"istenen {start.date()}..{end.date()}, mevcut {dates.min().date()}..{dates.max().date()}"
            )
            return None

        # Gereken sembolleri ve tarih araligini filtrele
        mask = np.asarray((dates >= start) & (dates <= end))
        filtered = df[mask]
        filtered = filtered[filtered.index.get_level_values("symbol").isin(needed_symbols)]
        return filtered

    def fetch_all(self, save: bool = False) -> pd.DataFrame:
        """Seçili metriklerin tamamını çek.

        Oncelik sirasi:
          1. Memory cache (ayni sorgu tekrar)
          2. Yerel CSV (source bazli dosya) — gerekli metrikleri ve tarih
             araligini kapsiyorsa kullanilir, internet istegi yapilmaz.
          3. Uzak kaynak (borsapy/yfinance) — eksik ise buradan cekilir.

        Returns:
            Multi-index DataFrame (symbol, date) — DataFetcher ile aynı format
        """
        cache_key = (
            f"{self.source}_{'_'.join(sorted(self.metrics))}"
            f"_{self.start_date}_{self.end_date}"
        )
        if cache_key in GoldFetcher._cache:
            logger.info("Altın verisi cache'den alınıyor")
            return GoldFetcher._cache[cache_key].copy()

        # Yerel CSV'den yuklemeyi dene — internet olmadiginda veya offline
        # calismak icin kritik. Gereken metrikler + tarih araligi kapsaniyorsa kullan.
        local = self._try_load_local()
        if local is not None:
            logger.info(f"Yerel CSV kullaniliyor: {self.default_filepath()} ({len(local)} satır)")
            GoldFetcher._cache[cache_key] = local.copy()
            return local

        pool: Dict[str, pd.DataFrame] = {}   # çekilen veriler (computed metrikler için)
        all_data: Dict[str, pd.DataFrame] = {}

        # yfinance computed metrikler ortak bağımlılıkları önceden çek
        if self.source == "yfinance":
            needs_ons    = any(m in self.metrics for m in ["gram_altin_usd", "gram_altin_try", "ons_altin_try"])
            needs_usdtry = any(m in self.metrics for m in ["gram_altin_try", "ons_altin_try"])
            if needs_ons and "ons_altin_usd" not in self.metrics:
                logger.info("Ara veri çekiliyor: ons_altin_usd (GC=F)")
                df = self._yfinance_fetch("GC=F")
                if df is not None:
                    pool["ons_altin_usd"] = df
            if needs_usdtry and "usd_try" not in self.metrics:
                logger.info("Ara veri çekiliyor: usd_try (TRY=X)")
                df = self._yfinance_fetch("TRY=X")
                if df is not None:
                    pool["usd_try"] = df

        for metric in self.metrics:
            info = METRIC_INFO[metric]
            logger.info(f"Çekiliyor: {metric} — {info['desc']}")
            df = self._fetch_metric(metric, pool)
            if df is not None and not df.empty:
                pool[metric] = df
                all_data[OUTPUT_SYMBOL[metric]] = df
                logger.info(
                    f"  ✓ {metric}: {len(df)} satır "
                    f"({df.index.min().date()} - {df.index.max().date()})"
                )
            else:
                logger.error(f"  ✗ {metric}: veri çekilemedi")

        if not all_data:
            raise ValueError("Hiçbir veri çekilemedi")

        combined = pd.concat(list(all_data.values()), keys=list(all_data.keys()))
        combined.index.names = ["symbol", "date"]

        logger.info(f"\nToplam satır: {len(combined)} | {len(all_data)} metrik")

        GoldFetcher._cache[cache_key] = combined.copy()

        if save:
            self.save_data(combined)

        return combined

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_data(self, df: pd.DataFrame, filepath: Optional[str] = None) -> None:
        """Multi-index DataFrame'i CSV olarak kaydet."""
        if filepath is None:
            filepath = self.default_filepath()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df.to_csv(filepath, index=True)
        logger.info(f"Kaydedildi: {filepath} ({len(df)} satır)")

    @staticmethod
    def load_data(filepath: str) -> Optional[pd.DataFrame]:
        """Kaydedilmiş veriyi yükle. Multi-index (symbol, date) döner."""
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.set_index(["symbol", "date"])
            logger.info(f"Yüklendi: {filepath} ({len(df)} satır)")
            return df
        except Exception as exc:
            logger.error(f"Yüklenemedi: {exc}")
            return None

    def get_source_status(self, filepath: Optional[str] = None) -> dict:
        """Mevcut dosyadaki veri durumunu raporla."""
        if filepath is None:
            filepath = self.default_filepath()
        today = datetime.now().date()
        df    = GoldFetcher.load_data(filepath)
        if df is None or df.empty:
            return {"exists": False, "last_date": None, "missing_days": None, "symbols": []}
        all_dates = pd.to_datetime(df.index.get_level_values("date"))
        last_date = all_dates.max().date()
        missing   = sum(
            1 for i in range(1, (today - last_date).days + 1)
            if (last_date + timedelta(days=i)).weekday() < 5
        )
        return {
            "exists":       True,
            "last_date":    str(last_date),
            "missing_days": missing,
            "symbols":      df.index.get_level_values("symbol").unique().tolist(),
        }

    def fetch_incremental(self, filepath: Optional[str] = None) -> dict:
        """Sadece eksik günleri çek ve mevcut dosyaya ekle."""
        if filepath is None:
            filepath = self.default_filepath()
        today_str = datetime.now().strftime("%Y-%m-%d")
        existing  = GoldFetcher.load_data(filepath)

        if existing is None or existing.empty:
            logger.info("Mevcut dosya yok, tam çekme yapılıyor...")
            df = self.fetch_all(save=False)
            self.save_data(df, filepath)
            return {"mode": "full", "new_rows": len(df), "total_rows": len(df)}

        all_dates   = pd.to_datetime(existing.index.get_level_values("date"))
        last_date   = all_dates.max().date()
        fetch_from  = (datetime.combine(last_date, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")

        if fetch_from > today_str:
            logger.info("Veri güncel, çekme gerekmez")
            return {"mode": "skip", "new_rows": 0, "total_rows": len(existing)}

        logger.info(f"Eksik veri çekiliyor: {fetch_from} → {today_str}")
        delta = GoldFetcher(
            source=self.source,
            metrics=self.metrics,
            start_date=fetch_from,
            end_date=today_str,
        )
        new_data = delta.fetch_all(save=False)

        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self.save_data(combined, filepath)

        return {
            "mode":       "incremental",
            "new_rows":   len(new_data),
            "total_rows": len(combined),
            "fetch_from": fetch_from,
            "fetch_to":   today_str,
        }

    # ------------------------------------------------------------------ #
    # Yardımcı
    # ------------------------------------------------------------------ #

    @staticmethod
    def available_metrics() -> None:
        """Kullanılabilir metrikleri ve kaynak desteğini göster."""
        print(f"{'Metrik':<20} {'Açıklama':<22} {'borsapy':<12} {'yfinance'}")
        print("-" * 70)
        for metric, info in METRIC_INFO.items():
            bp = "OK direkt"  if info["borsapy"]          else "-"
            yf = "OK direkt"  if info["yfinance"] == "direct" \
            else "~ hesaplanmis" if info["yfinance"] == "computed" \
            else "-"
            print(f"{metric:<20} {info['desc']:<22} {bp:<12} {yf}")


def main():
    """Hızlı test."""
    logging.basicConfig(level=logging.INFO)

    print("\n=== borsapy testi ===")
    fetcher_bp = GoldFetcher(
        source="borsapy",
        metrics=["gram_altin_try", "ons_altin_try", "usd_try", "eur_try"],
        start_date="2024-01-01",
        end_date="2024-03-01",
    )
    df_bp = fetcher_bp.fetch_all()
    print(df_bp.groupby(level="symbol").tail(2))

    print("\n=== yfinance testi ===")
    fetcher_yf = GoldFetcher(
        source="yfinance",
        metrics=["ons_altin_usd", "gram_altin_usd", "gram_altin_try", "usd_try", "eur_try"],
        start_date="2024-01-01",
        end_date="2024-03-01",
    )
    df_yf = fetcher_yf.fetch_all()
    print(df_yf.groupby(level="symbol").tail(2))

    print("\n=== Kullanılabilir metrikler ===")
    GoldFetcher.available_metrics()


if __name__ == "__main__":
    main()
