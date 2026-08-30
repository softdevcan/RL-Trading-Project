"""
Gunluk karar — panel tazeligi, gecersiz seans ve onbellek

Uc kusur birden, hepsi `fetch_latest_market_data()` icinde:

1. TAZELIK OLCUTU TAKVIM GUNUNE BAGLIYDI. `cached_last < end_date.date()`
   sorusu 30 Agustos 2026 (Pazar) hedefinde CSV 28 Agustos Cuma'ya kadar DOLU
   olsa bile dogru cikiyordu; her karar isteginde 30 sembolun tamami yeniden
   indiriliyordu. Olculdu: tek istek = 30 CSV okumasi + 30 yfinance indirmesi,
   sonuc hic degismeden. Hafta sonu ve tatil boyunca her tikmada tekrar.

2. CSV YOLUNDA GECERSIZ-CLOSE TEMIZLIGI YOKTU. yfinance seans kapanmadan once
   OHLC'si NaN, volume'u DOLU bir taslak satir dondurur ve bu satir CSV'ye
   yazilabiliyor (28.08.2026: 30 sembolun tamami boyle). yfinance yolunda ayni
   temizlik vardi, CSV yolunda yoktu.

3. `actual_date` YALNIZCA "o tarihte satir var mi" diye soruyordu. Panelin bir
   kismi tazelenip kalani tazelenmeyince (gercek olayda 30 sembolden 3'u) en
   son tarihte yalnizca o 3 sembolun gecerli kapanisi oluyor, geri kalan 27
   sembol tek tek bir onceki gune dusuyordu. Durum vektoru tek bir gunu degil,
   iki gunun karisimini tasiyordu.

Sunucu gerektirmez: yfinance ve CSV okuma sahte nesnelerle degistirilir.

    python tests/test_panel_freshness.py
"""

import asyncio
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import daily_trading as dt

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")


SYMBOLS = ["AKBNK.IS", "ARCLK.IS", "ASELS.IS", "THYAO.IS", "TUPRS.IS"]

# 2026-08-30 Pazar. Son seans 2026-08-28 Cuma.
SUNDAY = "2026-08-30"
FRIDAY = pd.Timestamp("2026-08-28")
THURSDAY = pd.Timestamp("2026-08-27")


def _ohlcv(index, base=100.0):
    n = len(index)
    close = base + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )


def _csv_panel(stub_last_day=False, stub_symbols=None):
    """CSV onbellegi. `stub_last_day` ile son gun yfinance taslagina cevrilir:
    OHLC NaN, volume DOLU — 28.08.2026'da diskte duran satirin ta kendisi."""
    dates = pd.bdate_range("2026-06-01", "2026-08-28")
    frames, keys = [], []
    for i, sym in enumerate(SYMBOLS):
        d = _ohlcv(dates, base=100.0 + 10 * i)
        if stub_last_day and (stub_symbols is None or sym in stub_symbols):
            d.loc[FRIDAY, ["open", "high", "low", "close"]] = np.nan
        frames.append(d)
        keys.append(sym)
    df = pd.concat(frames, keys=keys)
    df.index.names = ["symbol", "date"]
    return df


class _Recorder:
    """Indirilen sembolleri sayan sahte yfinance."""

    downloads = []
    serve = SYMBOLS  # bunlar disindaki semboller icin bos cerceve doner

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None):
        _Recorder.downloads.append(self.symbol)
        if self.symbol not in _Recorder.serve:
            return pd.DataFrame()
        dates = pd.bdate_range("2026-08-03", "2026-08-28")
        df = _ohlcv(dates, base=500.0)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df


def _fetcher_returning(panel):
    class _F:
        def load_data(self, filename="raw_stock_data.csv"):
            return panel
    return _F


def _run(panel, target=SUNDAY):
    import data.data_fetcher as df_mod
    df_mod.DataFetcher = _fetcher_returning(panel)
    dt.clear_panel_cache()
    _Recorder.downloads = []
    return asyncio.run(dt.fetch_latest_market_data(SYMBOLS, target, lookback_days=30))


def _incremental_checks():
    """Yazma tarafi: taslak satir DISKE hic yazilmamali.

    Yazilirsa `min_last_date` o gune kayar, `fetch_from` ertesi gun olur ve o
    seansin gercek verisi bir daha hic cekilmez.
    """
    import shutil
    import tempfile

    import data.data_fetcher as df_mod

    print("\n[8] fetch_incremental taslak satiri diske yazmiyor")

    tmp = tempfile.mkdtemp(prefix="rlt_panel_")
    try:
        dates = pd.bdate_range("2026-08-03", "2026-08-27")
        frames, keys = [], []
        for i, sym in enumerate(SYMBOLS[:2]):
            frames.append(_ohlcv(dates, base=100.0 + 10 * i))
            keys.append(sym)
        existing = pd.concat(frames, keys=keys)
        existing.index.names = ["symbol", "date"]

        # Delta: 28 Agustos icin OHLC'si NaN, volume'u dolu taslak satir.
        stub_idx = pd.MultiIndex.from_product(
            [SYMBOLS[:2], [FRIDAY]], names=["symbol", "date"]
        )
        stub = pd.DataFrame(
            {"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
             "volume": 1_000_000.0},
            index=stub_idx,
        )

        orig_ctor = df_mod.DataFetcher
        fetcher = orig_ctor.__new__(orig_ctor)
        fetcher.data_dir = tmp
        fetcher.start_date = "2026-08-03"
        fetcher.end_date = "2026-08-28"
        fetcher.max_workers = 1

        saved = {}
        fetcher.save_data = lambda df, filename: saved.update(df=df.copy())
        fetcher.load_data = lambda filename="raw_stock_data.csv": existing

        class _DeltaFetcher:
            def __init__(self, *a, **kw):
                pass

            def fetch_stock_data(self, symbols, save=False):
                return stub

        # fetch_incremental icinde `os.path.exists(filepath)` True olmali
        open(os.path.join(tmp, "raw_stock_data.csv"), "w").close()
        df_mod.DataFetcher = _DeltaFetcher
        try:
            result = orig_ctor.fetch_incremental(fetcher, SYMBOLS[:2])
        finally:
            df_mod.DataFetcher = orig_ctor

        out = saved.get("df")
        check("kayit yapildi", out is not None)
        if out is not None:
            closes = out["close"]
            check("diske yazilanda NaN kapanis yok", not closes.isna().any(),
                  f"{int(closes.isna().sum())} NaN")
            last_day = out.index.get_level_values("date").max()
            check("son tarih taslak gune KAYMADI", last_day == THURSDAY,
                  str(last_day.date()))
            check("atlanan satir sayisi raporlandi",
                  result.get("skipped_stub_rows") == 2,
                  str(result.get("skipped_stub_rows")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run():
    print("=" * 62)
    print("Gunluk karar — panel tazeligi, gecersiz seans, onbellek")
    print("=" * 62)

    import data.data_fetcher as df_mod
    orig_ticker = dt.yf.Ticker
    orig_fetcher = df_mod.DataFetcher
    orig_mtime = dt._raw_csv_mtime
    dt.yf.Ticker = _Recorder
    dt._raw_csv_mtime = lambda: 1234.0

    try:
        print("\n[1] last_expected_session: seans olmayan gun geriye sarilir")
        check("Pazar -> Cuma",
              dt.last_expected_session(datetime(2026, 8, 30)) == FRIDAY.date(),
              str(dt.last_expected_session(datetime(2026, 8, 30))))
        check("Cumartesi -> Cuma",
              dt.last_expected_session(datetime(2026, 8, 29)) == FRIDAY.date())
        check("Cuma -> kendisi",
              dt.last_expected_session(datetime(2026, 8, 28)) == FRIDAY.date())
        check("Sali -> kendisi",
              dt.last_expected_session(datetime(2026, 8, 25))
              == pd.Timestamp("2026-08-25").date())

        print("\n[2] CSV son seansa kadar dolu -> HIC indirme yok")
        combined = _run(_csv_panel())
        check("yfinance hic cagrilmadi", _Recorder.downloads == [],
              f"{len(_Recorder.downloads)} indirme: {_Recorder.downloads[:4]}")
        check("panel yine de tam",
              combined.index.get_level_values("symbol").nunique() == len(SYMBOLS))
        check("secilen gun son seans",
              combined.attrs["actual_date"] == "2026-08-28",
              combined.attrs.get("actual_date"))

        print("\n[3] CSV'deki taslak satir (OHLC NaN, volume dolu) panele girmiyor")
        combined = _run(_csv_panel(stub_last_day=True))
        closes = combined["close"]
        check("panelde NaN kapanis yok", not closes.isna().any(),
              f"{int(closes.isna().sum())} NaN")
        check("panelde sifir/negatif kapanis yok", bool((closes > 0).all()))
        # CSV'nin son gunu dusunce tazelik olcutu tutmaz -> yfinance devreye girer
        check("gecersiz gun indirmeyi tetikledi",
              sorted(set(_Recorder.downloads)) == sorted(SYMBOLS),
              str(sorted(set(_Recorder.downloads))))

        print("\n[4] Kismi tazeleme karma tarihli panel uretmiyor")
        # Gercek olayin birebir kurulumu: CSV'de son gun 5 sembolde de taslak,
        # yfinance yalnizca 2 sembole cevap veriyor. Eskiden actual_date 28
        # secilir, kalan 3 sembol tek tek 27'ye duserdi.
        _Recorder.serve = SYMBOLS[:2]
        combined = _run(_csv_panel(stub_last_day=True))
        chosen = pd.Timestamp(combined.attrs["actual_date"])
        rows = combined.xs(chosen, level="date")
        covered = int((rows["close"].notna() & (rows["close"] > 0)).sum())
        check("secilen gun sembollerin cogunu kapsiyor",
              covered >= max(1, int(round(len(SYMBOLS) * dt.MIN_SESSION_COVERAGE))),
              f"{covered}/{len(SYMBOLS)} sembol, gun={chosen.date()}")
        check("kapsamsiz gun yerine bir onceki seans secildi",
              chosen == THURSDAY, str(chosen.date()))
        check("secilen gunde her sembol var",
              len(rows) == len(SYMBOLS), f"{len(rows)} satir")
        _Recorder.serve = SYMBOLS

        print("\n[5] TTL onbellegi tekrarli istekte indirmeyi kesiyor")
        panel = _csv_panel(stub_last_day=True)
        _run(panel)                       # 1. istek: indirir
        first = len(_Recorder.downloads)
        _Recorder.downloads = []
        df_mod.DataFetcher = _fetcher_returning(panel)
        asyncio.run(dt.fetch_latest_market_data(SYMBOLS, SUNDAY, lookback_days=30))
        check("1. istek indirdi", first > 0, str(first))
        check("2. istek hic indirmedi", _Recorder.downloads == [],
              f"{len(_Recorder.downloads)} indirme")

        print("\n[6] CSV degisince onbellek dusuyor (mtime anahtarda)")
        _Recorder.downloads = []
        dt._raw_csv_mtime = lambda: 9999.0   # Veri sayfasi yeni veri indirdi
        df_mod.DataFetcher = _fetcher_returning(panel)
        asyncio.run(dt.fetch_latest_market_data(SYMBOLS, SUNDAY, lookback_days=30))
        check("yeni CSV surumu TTL beklemeden tazeledi",
              len(_Recorder.downloads) > 0,
              f"{len(_Recorder.downloads)} indirme")
        dt._raw_csv_mtime = lambda: 1234.0

        print("\n[7] Hafta ici hedefte taze veri hala isteniyor")
        _Recorder.downloads = []
        # CSV Cuma'ya kadar dolu, hedef Pazartesi 31 -> yeni seans var, indir
        combined = _run(_csv_panel(), target="2026-08-31")
        check("yeni seans indirmeyi tetikledi", len(_Recorder.downloads) > 0,
              f"{len(_Recorder.downloads)} indirme")

    finally:
        dt.yf.Ticker = orig_ticker
        df_mod.DataFetcher = orig_fetcher
        dt._raw_csv_mtime = orig_mtime
        dt.clear_panel_cache()

    _incremental_checks()

    print("\n" + "=" * 62)
    print(f"Toplam: {PASSED + FAILED} | Gecti: {PASSED} | Kaldi: {FAILED}")
    print("=" * 62)
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
