"""
Faz 9 / D1 + D4 — baseline stratejileri ve maliyet duyarliligi (plan §4 D; DeMiguel vd. 2009).

D1:
  * `ew_monthly`: aylik yeniden dengelenen 1/N. Her ayin ilk seansinda kapanista, o gun fiyati
    olan (listelenmis) sembollere esit agirlik; ay icinde agirliklar fiyatla kayar.
  * `buy_hold`: ilk seansta 1/N alinir, hic yeniden dengelenmez (sonradan listelenen eklenmez).
  * `xu030`: gercek BIST 30 FIYAT endeksi (`data/macro/xu030.csv`, yfinance 'XU030.IS'); sentetik
    yedek YOK — seri yoksa hata. Fiyat endeksi temettu icermez, panel kapanislari yfinance
    auto_adjust (temettu duzeltmeli): endeks kolu sistematik olarak asagi yanli (beyan).
    Endeks islem goremez: maliyetsiz referans.
D4: maliyet = gidis-donus bps (`COST_GRID_BPS`, ana 20); tek yon bps/2, islem goren nominal
    (devir = sum |dw|) uzerinden: net_t = brut_t - (bps / 2 / 1e4) * devir_t. Ilk alim devri 1.
    RL ajanlarinda politika maliyete gore egitildigi icin ex post izgara bir yaklasiklik;
    Milestone 0' ortami `commission` ile yeniden degerlendirir (beyan).

Girdi: veri kapisindan gecmis RL paneli (`prepare_rl_panel`); eksik hucre (bayat/atilan) sembolun
son kapanisiyla tasinir (o gun getiri 0; islem gormeyen gun ayrica isaretlenmez), listelenmeden once
NaN (yatirilamaz).
Metrikler: nominal TL'de A8 fazla getiri Sharpe'i (birincil, gunluk); reel TL ve USD'de AYLIK getiriden
`sharpe_rf0_monthly` (`monthly_metrics`; hakem-finans I3 #4-#5).

Beyan (hakem-finans I3 #3, #7, #8):
  * EVREN HAYATTA KALAN, DONDURULMUS panel 30'u (E0); tarihsel BIST 30 uyeligi degil (E1 acik). 1/N ve
    al-tut bu evrende; XU030 ayni evren degil ve temettusuz -> baseline'lar arasi ya da XU030'a karsi
    USTUNLUK cikarimi yapilmaz; XU030 ayri referans satiridir.
  * Maliyet yalniz komisyon (spread, lot, fiyat limiti yok). Al-tutta cikis maliyeti yok (RL ile
    tutarli: pencere sonunda pozisyon kapatilmaz).
  * 1/N ile al-tut arasindaki fark istatistiksel olarak test EDILMEDI (gerekirse Ledoit-Wolf, A3).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

COST_GRID_BPS = (0, 10, 20, 25, 50)
MAIN_COST_BPS = 20
DEFAULT_XU030_CSV = os.path.join('data', 'macro', 'xu030.csv')
PERIODS = 252


def close_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex (date, symbol) panel -> tarih x sembol kapanis; ilk gecerli gunden sonra ffill."""
    c = panel['close'].unstack('symbol').sort_index()
    c.index = pd.DatetimeIndex(pd.to_datetime(c.index))
    if c.index.tz is not None:
        c.index = c.index.tz_localize(None)
    return c.ffill()   # listelenmeden onceki NaN ffill'le dolmaz (onunde deger yok)


def _window(close: pd.DataFrame, start, end) -> pd.DataFrame:
    c = close[close.index >= pd.Timestamp(start)]
    return c[c.index < pd.Timestamp(end)] if end is not None else c


def equal_weight(close: pd.DataFrame, start, end=None, rebalance: str = 'monthly') -> pd.DataFrame:
    """1/N portfoyunun brut gunluk getirisi ve devri. rebalance: 'monthly' | 'none'."""
    if rebalance not in ('monthly', 'none'):
        raise ValueError("rebalance 'monthly' | 'none'")
    c = _window(close, start, end)
    if len(c) < 2:
        raise ValueError("pencerede en az 2 seans gerekir")
    px = c.to_numpy(dtype=np.float64)
    n_days, n_sym = px.shape
    month = c.index.to_period('M')
    gross = np.zeros(n_days)
    turn = np.zeros(n_days)
    w = np.zeros(n_sym)

    def target(i):
        ok = np.isfinite(px[i])
        t = np.zeros(n_sym)
        t[ok] = 1.0 / ok.sum()
        return t

    w = target(0)
    turn[0] = 1.0            # nakitten ilk alim
    for i in range(1, n_days):
        r = np.where(np.isfinite(px[i]) & np.isfinite(px[i - 1]), px[i] / px[i - 1] - 1.0, 0.0)
        rp = float(np.dot(w, r))
        gross[i] = rp
        w = w * (1.0 + r) / (1.0 + rp) if (1.0 + rp) != 0 else w
        if rebalance == 'monthly' and month[i] != month[i - 1]:
            t = target(i)
            turn[i] = float(np.abs(t - w).sum())
            w = t
    return pd.DataFrame({'gross': gross, 'turnover': turn}, index=c.index)


def net_returns(gross: pd.Series, turnover: pd.Series, bps_roundtrip: float) -> pd.Series:
    return gross - (float(bps_roundtrip) / 2.0 / 1e4) * turnover


def load_xu030(path: str = DEFAULT_XU030_CSV) -> pd.Series:
    from data.index_fetcher import load_index_close
    return load_index_close(path)


def index_returns(index_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Endeksin `dates` seanslarindaki basit getirisi (as-of: o gune kadarki son endeks kapanisi)."""
    s = index_close.sort_index()
    pos = s.index.searchsorted(dates, side='right') - 1
    if (pos < 0).any():
        raise ValueError("endeks serisi pencereden sonra basliyor")
    lvl = pd.Series(s.to_numpy()[pos], index=dates)
    r = lvl.pct_change().fillna(0.0)
    return r


def metrics(returns: pd.Series, risk_free: Optional[pd.Series] = None) -> Dict[str, Optional[float]]:
    """Yillik geometrik getiri, oynaklik, MDD; rf verilirse fazla getiri Sharpe'i, yoksa sharpe_rf0."""
    from evaluation.rl_folds import excess_sharpe
    r = pd.Series(returns, dtype=float)
    v = (1.0 + r).cumprod()
    n = len(r)
    # Ilk seans: brut 0, net = -giris maliyeti. Sharpe'a DAHIL (al-tutun tek maliyeti budur);
    # o gun icin tahakkuk eden risksiz faiz yok (rf_0 = 0; per_period_rate ilk seansi NaN verir).
    out = {'n': int(n),
           'ann_return': float(v.iloc[-1] ** (PERIODS / n) - 1.0),
           'ann_vol': float(r.std(ddof=1) * np.sqrt(PERIODS)) if n > 2 else None,
           'max_drawdown': float(np.minimum((v / v.cummax() - 1.0).min(), v.iloc[0] - 1.0)),
           'total_return': float(v.iloc[-1] - 1.0)}
    if risk_free is not None:
        rf = pd.Series(risk_free).reindex(r.index)
        if pd.isna(rf.iloc[0]):
            rf.iloc[0] = 0.0
        out['sharpe_excess'] = excess_sharpe(r.to_numpy(), r.index, rf)
    else:
        out['sharpe_rf0'] = excess_sharpe(r.to_numpy())
    return out


def monthly_metrics(values: pd.Series) -> Dict[str, Optional[float]]:
    """Reel TL ve USD birimi icin AYLIK getiriden metrik (hakem-finans I3 #4-#5).

    Gunluk reel getiri basamak deflatorle ay basinda yapay sicrama, gunluk USD getirisi bir gunluk kur
    kaymasi tasir; ikisi de gunluk Sharpe'in paydasini bozar. `values`: islem ONCESI sermaye = 1 olacak
    bicimde olceklenmis birim degeri (ilk gun giris maliyeti dahil). Aylik getiri: [1 (baslangic),
    ay sonu degerleri]; Sharpe rf = 0, yillik x sqrt(12). MDD ve toplam getiri gunluk degerden.
    """
    from evaluation.rl_folds import excess_sharpe
    v = pd.Series(values, dtype=float)
    start = pd.Series([1.0], index=[v.index[0] - pd.Timedelta(days=1)])
    rm = pd.concat([start, v.resample('ME').last()]).pct_change().dropna()
    n_days = len(v)
    return {'n': int(n_days), 'n_months': int(len(rm)),
            'ann_return': float(v.iloc[-1] ** (PERIODS / n_days) - 1.0),
            'ann_vol_monthly': float(rm.std(ddof=1) * np.sqrt(12)) if len(rm) > 2 else None,
            'max_drawdown': float(np.minimum((v / v.cummax() - 1.0).min(), v.iloc[0] - 1.0)),
            'total_return': float(v.iloc[-1] - 1.0),
            'sharpe_rf0_monthly': excess_sharpe(rm.to_numpy(), periods=12) if len(rm) >= 3 else None}


def baseline_table(close: pd.DataFrame, start, end=None, xu030: Optional[pd.Series] = None,
                   grid: Sequence[float] = COST_GRID_BPS, cpi: Optional[pd.Series] = None,
                   fx: Optional[pd.Series] = None, risk_free: Optional[pd.Series] = None) -> dict:
    """strateji -> maliyet (bps) -> birim -> metrik. Nominal TL'de A8 fazla getiri Sharpe'i."""
    from evaluation import units as U
    from evaluation.risk_free import risk_free_series

    out: Dict[str, dict] = {}
    strat = {'ew_monthly': equal_weight(close, start, end, 'monthly'),
             'buy_hold': equal_weight(close, start, end, 'none')}
    dates = next(iter(strat.values())).index
    rf = risk_free if risk_free is not None else risk_free_series(dates)
    cpi = cpi if cpi is not None else U.load_cpi()
    fx = fx if fx is not None else U.load_usdtry()

    def by_unit(r):
        vals = U.values_from_returns(r)
        res = {'nominal_try': metrics(r, rf)}
        for unit in ('real_try', 'usd'):
            vu = U.convert_values(vals, unit, cpi=cpi, fx=fx)
            # islem oncesi sermaye 1: ilk gun degeri nominaldeki gibi (1 - giris maliyeti)
            res[unit] = monthly_metrics(vu * (vals.iloc[0] / vu.iloc[0]))
        return res

    for name, df in strat.items():
        out[name] = {'mean_daily_turnover': float(df['turnover'].iloc[1:].mean()),
                     'costs': {str(int(b)): by_unit(net_returns(df['gross'], df['turnover'], b))
                               for b in grid}}
    if xu030 is not None:
        out['xu030'] = {'note': 'fiyat endeksi (temettu yok), maliyetsiz referans',
                        'costs': {'0': by_unit(index_returns(xu030, dates))}}
    return {'window': [str(dates[0].date()), str(dates[-1].date())], 'n_sessions': int(len(dates)),
            'strategies': out}
