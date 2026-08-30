"""
Gunluk karar — saat dilimi regresyonu

Kusur: `fetch_latest_market_data()` CSV onbellegini tz'siz, yfinance'i ise
tz'li (Europe/Istanbul) indeksle okuyordu. Ikisi birlestirilince indeks object
dtype'a dusuyor ve `sort_index()` "Cannot compare tz-naive and tz-aware
timestamps" firlatiyordu. Hata sembol dongusunun icinde yakalandigi icin her
sembol sessizce dusuyor, gunluk karar hicbir taze fiyat goremiyordu.

Sunucu gerektirmez: yfinance ve CSV okuma sahte nesnelerle degistirilir.

    python tests/test_daily_trading_tz.py
"""

import asyncio
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import daily_trading as dt
from data.data_fetcher import MARKET_TZ, to_naive_market_dates

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


TARGET = "2026-08-31"
SYMBOLS = ["AKBNK.IS", "THYAO.IS"]


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


def _fake_csv():
    """Onbellek: tz'siz indeks, hedef tarihten once biten (bayat) veri."""
    dates = pd.bdate_range("2026-06-01", "2026-08-28")
    frames, keys = [], []
    for i, sym in enumerate(SYMBOLS):
        frames.append(_ohlcv(dates, base=100.0 + 10 * i))
        keys.append(sym)
    df = pd.concat(frames, keys=keys)
    df.index.names = ["symbol", "date"]
    return df


class _FakeTicker:
    """yfinance gibi TZ'LI (Europe/Istanbul) indeks dondurur."""

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None):
        dates = pd.bdate_range("2026-08-03", "2026-08-31", tz=MARKET_TZ)
        df = _ohlcv(dates, base=200.0)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        # yfinance'in kapanis oncesi bastigi tas satir
        df.iloc[-1, df.columns.get_loc("Close")] = np.nan
        return df


class _ErrorCatcher(logging.Handler):
    """Sembol dongusu hatalari yutar (except/continue) — sadece log'a dusuyor."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _FakeFetcher:
    def load_data(self, filename="raw_stock_data.csv"):
        return _fake_csv()


def run():
    print("=" * 62)
    print("Gunluk karar — saat dilimi regresyonu")
    print("=" * 62)

    orig_ticker = dt.yf.Ticker
    import data.data_fetcher as df_mod
    orig_fetcher = df_mod.DataFetcher
    dt.yf.Ticker = _FakeTicker
    df_mod.DataFetcher = _FakeFetcher

    catcher = _ErrorCatcher()
    dt.logger.addHandler(catcher)

    try:
        print("\n[1] CSV (tz'siz) + yfinance (tz'li) birlesimi")
        dt.clear_panel_cache()  # panel TTL onbellegi bu testi maskelemesin
        try:
            combined = asyncio.run(
                dt.fetch_latest_market_data(SYMBOLS, TARGET, lookback_days=30)
            )
            err = None
        except Exception as exc:  # noqa: BLE001 — kusurun ta kendisi
            combined, err = None, exc
        check("birlestirme hata firlatmiyor", err is None, repr(err))
        # Asil kusur burada: sembol dongusundeki `except Exception: continue`
        # her sembolu sessizce dusuruyor, cagri disaridan basarili gorunuyor.
        # Tek kanit ERROR log satiri.
        check("hicbir sembol hatayla dusmedi", not catcher.messages,
              " | ".join(catcher.messages))
        check("tz karsilastirma hatasi yok",
              not any("tz-naive and tz-aware" in m for m in catcher.messages),
              " | ".join(catcher.messages))
        if combined is None:
            return

        print("\n[2] Indeks tipi")
        dates = combined.index.get_level_values("date")
        check("date seviyesi DatetimeIndex", isinstance(dates, pd.DatetimeIndex),
              f"{type(dates).__name__}")
        check("date seviyesi tz'siz", getattr(dates, "tz", None) is None,
              f"tz={getattr(dates, 'tz', None)}")
        check("indeks siralanabiliyor", combined.sort_index() is not None)

        print("\n[3] Her sembol hayatta kaldi")
        got = sorted(combined.index.get_level_values("symbol").unique())
        check("iki sembol de var", got == sorted(SYMBOLS), str(got))

        print("\n[4] Taze veri gercekten geldi")
        for sym in SYMBOLS:
            sub = combined.xs(sym, level="symbol")
            last = sub.index.max()
            # NaN kapanisli son satir dusuruldugu icin son is gunu 2026-08-28
            check(f"{sym}: son tarih onbellekten yeni", last >= pd.Timestamp("2026-08-28"),
                  str(last))
            check(f"{sym}: yfinance fiyatlari kazandi",
                  bool(sub.loc[pd.Timestamp("2026-08-28"), "close"] >= 200.0),
                  str(sub.loc[pd.Timestamp("2026-08-28"), "close"]))
            check(f"{sym}: NaN kapanis yok", not sub["close"].isna().any())
            check(f"{sym}: yinelenen tarih yok", not sub.index.duplicated().any())

        print("\n[5] Takvim gunu kaymadi (tz atarken +3 saat tuzagi)")
        aware = pd.bdate_range("2026-08-03", "2026-08-07", tz=MARKET_TZ)
        naive = to_naive_market_dates(aware)
        check("tz atma gunu kaydirmiyor",
              list(naive.date) == list(aware.date),
              f"{list(naive.date)[:2]} vs {list(aware.date)[:2]}")

        print("\n[6] Onbellek yokken de tz'siz cikiyor")
        df_mod.DataFetcher = None  # load_data cagrisi patlar -> yalniz yfinance yolu
        dt.clear_panel_cache()  # yoksa [1]'in sonucu onbellekten doner
        try:
            only_yf = asyncio.run(
                dt.fetch_latest_market_data(SYMBOLS, TARGET, lookback_days=30)
            )
            err2 = None
        except Exception as exc:  # noqa: BLE001
            only_yf, err2 = None, exc
        check("yalniz yfinance yolu calisiyor", err2 is None, repr(err2))
        if only_yf is not None:
            check("yalniz yfinance yolunda da tz yok",
                  only_yf.index.get_level_values("date").tz is None)
    finally:
        dt.logger.removeHandler(catcher)
        dt.yf.Ticker = orig_ticker
        df_mod.DataFetcher = orig_fetcher

    print("\n" + "=" * 62)
    print(f"Toplam: {PASSED + FAILED} | Gecti: {PASSED} | Kaldi: {FAILED}")
    print("=" * 62)
    return FAILED == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
