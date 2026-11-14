"""Canli tahmin takibinin kurallari (Faz 9 Y28 + Y29) — ensemble ve tracker ortak kullanir.

- Ufuk seans sayisi: gunluk 1, haftalik 5 (hedef `close.shift(-h)`, feature_engineer._build_targets).
- Gecerli canli kayit: `feature_date` tasir (Y28 sonrasi) ve `stale_features` degil.
- Gerceklesen kapanis: `feature_date`'ten SONRAKI h. gercek seans (tatil/hafta sonu takvimden degil
  veriden); tahmini `prediction_date` yalniz gosterim ve on-filtre.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd


def horizon_sessions(horizon: str) -> int:
    return 1 if horizon == 'daily' else 5


def invalid_reason(entry: Dict[str, Any]) -> Optional[str]:
    """Canli dogruluga GIRMEYECEK kaydin nedeni; gecerliyse None."""
    if not entry.get('feature_date'):
        return 'pre_y28_no_feature_date'
    if entry.get('stale_features') is True:
        return 'stale_features'
    return None


def actual_after_sessions(closes: pd.Series, feature_date: str, h: int) -> Optional[Tuple[str, float]]:
    """feature_date'ten sonraki h. seansin (tarih, kapanis); henuz yoksa None.

    `closes` gun indeksli kapanis serisi (tz'li olabilir). NaN kapanisli satir (yfinance seans
    kapanmadan donen taslak satir) seans sayilmaz.
    """
    s = closes.dropna()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    days = idx.normalize()
    after = s[days > pd.Timestamp(feature_date)]
    if len(after) < h:
        return None
    d = days[days > pd.Timestamp(feature_date)][h - 1]
    return d.strftime('%Y-%m-%d'), float(after.iloc[h - 1])
