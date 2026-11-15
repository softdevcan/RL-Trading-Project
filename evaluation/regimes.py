"""
Onceden tanimli, nedensel piyasa rejimi etiketi (Faz 9 B6; Y3).

Eski `PredictionEvaluator.compute_regime_metrics` "boga"yi GERCEKLESEN getirisi pozitif
ornekler diye tanimliyordu: hedef degiskene gore alt kumeleme, rejim degil (Y3). Bu modul
etiketi yalniz piyasa endeksinin <= t verisinden kurar (lit. taramasi §7.5; plan §4 B6):

  trend_t  : 'bull' eger kapanis_t > SMA_200(kapanis)_t, aksi halde 'bear'
  vol_t    : 60 gunluk log getiri std'si x sqrt(252)
  vol_level: vol_t'nin GENISLEYEN pencere tercilleri (<= t, en az 252 vol gozlemi):
             'low' (<= q1/3), 'mid', 'high' (> q2/3)
  regime   : '<trend>_<vol_level>' (6 hucre); isinma doluncaya kadar NaN (etiket YOK)

Tasarim varsayimlari: 200/60 gun (plan B6), genisleyen tercil (sabit tam-orneklem tercili
ileriye bakar; kayan tercil ayri bir karar olurdu), esitlik durumunda 'bear' ve alt kova.
Etiket t gunu KAPANIS sonrasi bilinir; t'de verilen bir tahmin t'nin etiketini kullanir.

Kabul: kesme degismezligi (B0) — etiketler tam seride ve t'de kesilmis seride t'ye kadar
bit-es (`tests/test_regimes.py`).
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

TREND_WINDOW = 200
VOL_WINDOW = 60
MIN_VOL_HISTORY = 252
VOL_LEVELS = ('low', 'mid', 'high')


def regime_labels(close: pd.Series, trend_window: int = TREND_WINDOW,
                  vol_window: int = VOL_WINDOW,
                  min_vol_history: int = MIN_VOL_HISTORY) -> pd.DataFrame:
    """Gun basina rejim etiketi. Girdi: tarih indeksli endeks kapanisi."""
    c = pd.Series(close, dtype=float).sort_index()
    if c.index.has_duplicates:
        raise ValueError("endeks serisinde tekrarlanan tarih")
    if (c.dropna() <= 0).any():
        raise ValueError("pozitif olmayan kapanis")
    sma = c.rolling(trend_window, min_periods=trend_window).mean()
    ratio = c / sma - 1.0
    trend = pd.Series(np.where(ratio > 0, 'bull', 'bear'), index=c.index, dtype=object)
    trend[ratio.isna()] = np.nan

    logr = np.log(c).diff()
    vol = logr.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    q1 = vol.expanding(min_periods=min_vol_history).quantile(1 / 3)
    q2 = vol.expanding(min_periods=min_vol_history).quantile(2 / 3)
    level = pd.Series(np.where(vol <= q1, 'low', np.where(vol <= q2, 'mid', 'high')),
                      index=c.index, dtype=object)
    level[q1.isna() | vol.isna()] = np.nan

    regime = trend.str.cat(level, sep='_')
    regime[trend.isna() | level.isna()] = np.nan
    return pd.DataFrame({'close': c, 'sma_ratio': ratio, 'trend': trend, 'vol': vol,
                         'vol_q1': q1, 'vol_q2': q2, 'vol_level': level, 'regime': regime})


def regime_summary(labels: pd.DataFrame, column: str = 'regime',
                   start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, dict]:
    """Rejim basina gun sayisi ve bagimsiz donem (ardisik blok) sayisi (M6: raporlanir)."""
    lab = labels[column]
    if start is not None:
        lab = lab[lab.index >= pd.Timestamp(start)]
    if end is not None:
        lab = lab[lab.index < pd.Timestamp(end)]
    out: Dict[str, dict] = {}
    known = lab.dropna()
    run_id = (known != known.shift()).cumsum()
    for name, grp in known.groupby(known):
        out[str(name)] = {'days': int(len(grp)), 'episodes': int(run_id[grp.index].nunique())}
    out['_unlabeled_days'] = int(lab.isna().sum())
    out['_total_days'] = int(len(lab))
    return out


def asof_labels(labels: pd.DataFrame, dates, column: str = 'regime') -> pd.Series:
    """`dates` icin etiket: o gun ya da oncesindeki son endeks gununun etiketi (<= t)."""
    d = pd.DatetimeIndex(dates)
    src = labels[column]
    pos = src.index.searchsorted(d, side='right') - 1
    vals = np.where(pos >= 0, src.to_numpy(dtype=object)[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=d, dtype=object).where(pd.notna(vals), np.nan)


def by_regime(dates, labels: pd.DataFrame, metric: Callable[[np.ndarray], float],
              *arrays, column: str = 'regime', min_samples: int = 5) -> Dict[str, dict]:
    """Ornek basina diziler (tarihleriyle) -> rejim basina metrik.

    `dates[i]` ornegin KARAR tarihi (tahminin verildigi gun); etiket o gunun ya da endeksin
    o gune kadarki SON islem gununun etiketi (as-of, nedensel: panelde olup endekste olmayan
    tatil satirlari icin). Etiketi olmayan (isinma / seri oncesi) ornekler `_unlabeled` sayilir.
    """
    d = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    if d.tz is not None:
        d = d.tz_localize(None)
    lab = asof_labels(labels, d, column=column)
    arrs = [np.asarray(a) for a in arrays]
    out: Dict[str, dict] = {}
    for name in sorted(set(lab.dropna())):
        m = (lab == name).to_numpy()
        if m.sum() < min_samples:
            out[name] = {'n_samples': int(m.sum()), 'value': None}
            continue
        out[name] = {'n_samples': int(m.sum()), 'value': float(metric(*[a[m] for a in arrs]))}
    out['_unlabeled'] = int(lab.isna().sum())
    return out


def load_xu100_labels(path: Optional[str] = None, **kw) -> pd.DataFrame:
    """Kaydedilmis XU100 serisinden etiket (`data.index_fetcher`)."""
    from data.index_fetcher import DEFAULT_INDEX_CSV, load_index_close
    return regime_labels(load_index_close(path or DEFAULT_INDEX_CSV), **kw)
