"""A8 — Tek kaynak risksiz faiz.

Sorun (roadmap D3; Faz 10 §1.1): risksiz faiz kodda dort yerde dort deger
tasiyor — 0,15 (env/reward_functions.py), 0,0 (prediction/evaluator.py),
0,02 (app/services/model_analysis.py, scripts/benchmarking/benchmark_strategies.py).
Bu modul tum yeni degerlendirme kodunun okudugu TEK kaynaktir. Seri
yoksa ya da yedek degerle doldurulmussa sessizce sabite DUSMEZ:
`RiskFreeUnavailable` yukseltir.

Kaynak serisi: TCMB politika faizi (EVDS, `data/macro/macro_data.csv`
`policy_rate`, yillik %). Su an kullanilamiyor: son cekimin kalite bayragi
`fallback_constant_50.0` (bkz. Faz 10 K2 / Faz 9 B2). B2 kapanana kadar bu
modulu cagiran her akis acik hata verir — bu bilincli: yanlis faizle
hesaplanmis Sharpe raporlanmamali.

Gunluk donusum (tasarim varsayimi, belgelenir): t seansinin risksiz getirisi,
t-1'de bilinen yillik oranin iki seans arasindaki TAKVIM gunu kadar bilesik
uygulanmasidir:  rf_t = (1 + R_{t-1}/100) ** (gun_t / 365) - 1.
TL para piyasasi ACT/365 kuralini izler; hafta sonu ve tatil getirisi bir
sonraki seansa yazilir.

ESKI CAGIRANLAR BU TURDA BAGLANMADI: reward_functions'in 0,15'i egitilmis RL
modellerinin oduludur (davranis dondurma); model_analysis ve evaluator
panoda gorunen sayilari degistirir. Baglama B2 (EVDS) sonrasi, Milestone 0'
ile birlikte yapilir.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_MACRO_CSV = os.path.join('data', 'macro', 'macro_data.csv')


class RiskFreeUnavailable(RuntimeError):
    """Gecerli bir risksiz faiz serisi yok; sabite dusulmez."""


def _quality_flags(csv_path: str) -> dict:
    side = os.path.splitext(csv_path)[0] + '_quality.json'
    if not os.path.exists(side):
        return {}
    try:
        with open(side, encoding='utf-8') as f:
            return json.load(f).get('data_quality', {}) or {}
    except Exception:
        return {'_unreadable': side}


def load_annual_rate(csv_path: str = DEFAULT_MACRO_CSV, column: str = 'policy_rate') -> pd.Series:
    """Yillik % oran serisi; kalite bayragi ya da sabit seri varsa hata."""
    if not os.path.exists(csv_path):
        raise RiskFreeUnavailable(f"{csv_path} yok — makro veri cekilmemis")
    flags = _quality_flags(csv_path)
    if column in flags:
        raise RiskFreeUnavailable(
            f"{column} kalite bayragi '{flags[column]}' — seri gercek veri degil. "
            f"EVDS erisimi duzelmeden (Faz 9 B2 / Faz 10 K2) risksiz faiz kurulamaz."
        )
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if column not in df.columns:
        raise RiskFreeUnavailable(f"{csv_path} icinde '{column}' sutunu yok")
    s = df[column].astype(float).dropna()
    if s.empty or s.nunique() <= 1:
        raise RiskFreeUnavailable(
            f"{column} serisi sabit ({s.unique()[:3].tolist()}) — yedek degerle "
            f"doldurulmus olabilir; risksiz faiz olarak kullanilmaz."
        )
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, 'tz', None) else pd.to_datetime(s.index)
    return s.sort_index()


def per_period_rate(session_dates, annual_pct: pd.Series) -> pd.Series:
    """Seans tarihleri icin donem basi risksiz getiri (ilk seans NaN).

    rf_t = (1 + R_{t-1}/100) ** ((d_t - d_{t-1}).days / 365) - 1, R_{t-1} =
    d_{t-1} itibariyla bilinen son oran (ileriye bakma yok).
    """
    dates = pd.DatetimeIndex(pd.to_datetime(session_dates)).sort_values()
    known = annual_pct.reindex(annual_pct.index.union(dates)).ffill().reindex(dates)
    prev_rate = known.shift(1)
    gap_days = pd.Series(dates, index=dates).diff().dt.days
    rf = (1.0 + prev_rate / 100.0) ** (gap_days / 365.0) - 1.0
    return pd.Series(rf.to_numpy(), index=dates, name='rf')


def risk_free_series(session_dates, csv_path: Optional[str] = None, column: str = 'policy_rate') -> pd.Series:
    """Tek giris noktasi: seans tarihleri -> donem basi risksiz getiri."""
    rate = load_annual_rate(csv_path or DEFAULT_MACRO_CSV, column)
    out = per_period_rate(session_dates, rate)
    first = pd.Timestamp(pd.to_datetime(session_dates).min())
    if rate.index.min() > first:
        raise RiskFreeUnavailable(
            f"Faiz serisi {rate.index.min().date()} tarihinde basliyor, istenen "
            f"donem {first.date()}'den — basi doldurulmaz (bfill sizinti olurdu)."
        )
    return out
