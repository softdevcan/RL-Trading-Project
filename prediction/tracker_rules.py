"""Canli tahmin takibinin kurallari (Faz 9 Y28 + Y29) — ensemble ve tracker ortak kullanir.

- Ufuk seans sayisi: gunluk 1, haftalik 5 (hedef `close.shift(-h)`, feature_engineer._build_targets).
- Gecerli canli kayit (`invalid_reason` None): `feature_date` tasir (Y28 sonrasi), `stale_features`
  degil, degerlendirilmisse Y29 kuraliyla (`actual_date` var) ve tahmin hedef seanstan ONCE
  uretilmis (ex-ante; `made_at` gunu <= `actual_date`).
- Gerceklesen kapanis: `feature_date`'ten SONRAKI h. gercek seans (tatil/hafta sonu takvimden degil
  veriden). Kapanisi NaN satir (seans kapanmadan donen taslak) ve duz satir (volume = 0, O = H = L = C;
  data/integrity.py "hayalet gun / bayat hucre" tanimi) seans sayilmaz. Tahmini `prediction_date`
  yalniz gosterim ve on-filtre.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd


def horizon_sessions(horizon: str) -> int:
    return 1 if horizon == 'daily' else 5


def invalid_reason(entry: Dict[str, Any]) -> Optional[str]:
    """Canli dogruluga GIRMEYECEK kaydin nedeni; gecerliyse None."""
    if not entry.get('feature_date'):
        return 'pre_y28_no_feature_date'
    if entry.get('stale_features') is True:
        return 'stale_features'
    if entry.get('actual_close') is not None:
        if not entry.get('actual_date'):
            return 'pre_y29_evaluation'          # v1.10.0'da duvar saati tarihinden degerlendirilmis
        if str(entry.get('made_at', ''))[:10] > entry['actual_date']:
            return 'not_ex_ante'                 # tahmin hedef seanstan SONRA uretilmis (veri gerideydi)
    return None


def _session_rows(bars: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    df = bars.to_frame('Close') if isinstance(bars, pd.Series) else bars.copy()
    df = df[df['Close'].notna()]
    if {'Open', 'High', 'Low', 'Volume'} <= set(df.columns):
        flat = ((df['Volume'] == 0) & (df['Open'] == df['High']) & (df['High'] == df['Low'])
                & (df['Low'] == df['Close']))
        df = df[~flat]
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


def actual_after_sessions(bars: Union[pd.Series, pd.DataFrame], feature_date: str,
                          h: int) -> Optional[Tuple[str, float]]:
    """feature_date'ten sonraki h. seansin (tarih, kapanis); henuz yoksa None.

    `bars`: gun indeksli yfinance cubugu (Close; varsa Open/High/Low/Volume) ya da kapanis serisi.
    """
    s = _session_rows(bars)
    after = s[s.index > pd.Timestamp(feature_date)]
    if len(after) < h:
        return None
    return after.index[h - 1].strftime('%Y-%m-%d'), float(after['Close'].iloc[h - 1])
