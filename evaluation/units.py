"""
Faz 9 / D3 — getiri birimi: nominal TL, TUFE-reel TL, USD (vision §4.1; lit. taramasi §7.6).

Yalniz RAPORLAMA (ex post): karar girdisi degil, gozleme ya da odule girmez. Bu yuzden TUFE'nin
aciklanma gecikmesi uygulanmaz (deger zaten gerceklesmis; B3 kurali karar girdileri icin).

Tasarim varsayimlari:
  * Reel TL: V_reel_t = V_t / TUFE(ay(t)) — basamak deflator (ayin TUFE duzeyi ayin her gunune).
    Uctan uca reel getiri tam; gunluk dagilimda ay basinda aylik enflasyon kadar tek gunluk
    dusus olur (interpolasyon yok: son ayin ortasindan sonrasi icin ekstrapolasyon gerekirdi).
    Kaynak: TP.TUKFIY2025.GENEL (`data/macro/cpi_index.csv`, `fetch_cpi_index`).
  * USD: V_usd_t = V_t / USDTRY_{t-1} (t'den kesin once biten son kur; B6 USD etiketiyle ayni
    kural, `evaluation.regimes.usd_index`). Bir gunluk kur kaymasi beyan edilir (hakem-finans I2).
  * Birincil metrik nominal TL'de A8 risksiz faiz uzeri Sharpe (plan §5 M8). Reel ve USD
    birimlerinde birim-tutarli risksiz faiz serisi yok: Sharpe rf = 0 ile, AYLIK getiriden
    (`evaluation.baselines.monthly_metrics`): gunluk reel getiri ay basinda yapay sicrama, gunluk USD
    getirisi bir gunluk kur kaymasi tasir (hakem-finans I3 #4-#5).
Deflator ya da kur istenen gunlerden birinde yoksa `UnitUnavailable` (sessiz atlama yok).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

UNITS = ('nominal_try', 'real_try', 'usd')
DEFAULT_CPI_CSV = os.path.join('data', 'macro', 'cpi_index.csv')


class UnitUnavailable(ValueError):
    """Istenen gunlerde deflator ya da kur yok."""


def load_cpi(path: str = DEFAULT_CPI_CSV) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} yok: MacroDataFetcher().fetch_cpi_index()")
    s = pd.read_csv(path, parse_dates=['month'], index_col='month')['cpi_index'].astype(float)
    if (s <= 0).any() or s.index.has_duplicates:
        raise UnitUnavailable("TUFE endeksi gecersiz")
    return s.sort_index()


def load_usdtry() -> pd.Series:
    from data.index_fetcher import DEFAULT_USDTRY_CSV, FX_JUMP_LIMIT, load_index_close
    return load_index_close(DEFAULT_USDTRY_CSV, jump_limit=FX_JUMP_LIMIT)


def cpi_on(dates, cpi: pd.Series) -> pd.Series:
    """Basamak deflator: her gun icin kendi ayinin TUFE duzeyi."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    m = d.to_period('M').to_timestamp()
    return pd.Series(cpi.reindex(m).to_numpy(), index=d, name='cpi')


def fx_before(dates, fx: pd.Series) -> pd.Series:
    """t'den kesin once biten son kur gozlemi."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    f = fx.dropna().sort_index()
    pos = f.index.searchsorted(d, side='left') - 1
    vals = np.where(pos >= 0, f.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=d, name='usdtry_prev')


def convert_values(values: pd.Series, unit: str, cpi: Optional[pd.Series] = None,
                   fx: Optional[pd.Series] = None) -> pd.Series:
    """Nominal TL portfoy degeri -> istenen birim."""
    if unit not in UNITS:
        raise ValueError(f"birim {unit!r}: {UNITS}")
    v = pd.Series(values, dtype=float)
    if unit == 'nominal_try':
        return v
    if unit == 'real_try':
        defl = cpi_on(v.index, cpi if cpi is not None else load_cpi())
    else:
        defl = fx_before(v.index, fx if fx is not None else load_usdtry())
    if defl.isna().any():
        bad = defl.index[defl.isna().to_numpy()]
        raise UnitUnavailable(f"{unit}: {len(bad)} gunde deflator/kur yok (ilk {bad[0].date()}, "
                              f"son {bad[-1].date()})")
    return pd.Series(v.to_numpy() / defl.to_numpy(), index=v.index, name=f'value_{unit}')


def values_from_returns(returns: pd.Series, v0: float = 1.0) -> pd.Series:
    r = pd.Series(returns, dtype=float).fillna(0.0)
    return v0 * (1.0 + r).cumprod()
