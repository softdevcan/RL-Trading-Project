"""
Faz 9 / B6 — onceden tanimli, nedensel rejim etiketi (`evaluation/regimes.py`; Y3).

Kontroller:
  1. B0 kesme degismezligi: etiketler (trend, vol, tercil esikleri, vol_level, regime) tam
     seride ve t'de kesilmis seride t'ye kadar bit-es — sentetik seri, 6 kesim noktasi.
  2. Mutasyon: merkezlenmis SMA ve tam-orneklem tercili B0'i KIRAR.
  3. Anlam: belirgin yukselis -> 'bull', yuksek oynaklik bolumu -> 'high'; isinmada etiket yok.
  4. Y3: eski `compute_regime_metrics` kaldirildi; `compute_comprehensive_metrics` rejimi
     yalniz tarih + etiketle hesaplar.
  5. as-of: endekste olmayan (tatil) gun onceki islem gununun etiketini alir; seri oncesi NaN.
  6. Ozet: gun ve bagimsiz donem (ardisik blok) sayisi.
  7. Gercek XU100 (dosya varsa): SHA-256 yan dosyayla eslesiyor, B0 gercek seride de bit-es.

Ag yok. Calistirma:
    python tests/test_regimes.py
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from evaluation import regimes as rg  # noqa: E402

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def synthetic(n=1400, seed=7):
    rng = np.random.default_rng(seed)
    mu = np.r_[np.full(400, 0.0002), np.full(300, 0.0025), np.full(300, -0.002), np.full(n - 1000, 0.0005)]
    sd = np.r_[np.full(600, 0.008), np.full(200, 0.035), np.full(n - 800, 0.01)]
    r = mu + sd * rng.standard_normal(n)
    return pd.Series(1000 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-01", periods=n))


COLS = ['sma_ratio', 'trend', 'vol', 'vol_q1', 'vol_q2', 'vol_level', 'regime']


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    for c in COLS:
        x, y = a[c], b[c]
        if not x.index.equals(y.index):
            return False
        if x.dtype == object or y.dtype == object:
            if not ((x.isna() & y.isna()) | (x == y)).all():
                return False
        elif not np.array_equal(x.to_numpy(), y.to_numpy(), equal_nan=True):
            return False
    return True


def cutoff_invariant(fn, close, ts):
    full = fn(close)
    bad = []
    for t in ts:
        d = close.index[t]
        if not same(full.loc[:d], fn(close.loc[:d]).loc[:d]):
            bad.append(t)
    return bad


close = synthetic()
TS = (199, 260, 455, 700, 1000, len(close) - 1)

print("\n[1] B0 kesme degismezligi (sentetik)")
bad = cutoff_invariant(rg.regime_labels, close, TS)
check(f"t in {TS}: tum etiket sutunlari t'ye kadar bit-es", not bad, f"kirilan t: {bad}")

print("\n[2] Mutasyonlar B0'i kirmali")


def mut_center(c):
    out = rg.regime_labels(c)
    sma = c.rolling(rg.TREND_WINDOW, min_periods=1, center=True).mean()
    out['sma_ratio'] = c / sma - 1
    out['trend'] = np.where(out['sma_ratio'] > 0, 'bull', 'bear')
    return out


def mut_fullsample(c):
    out = rg.regime_labels(c)
    q1, q2 = out['vol'].quantile([1 / 3, 2 / 3])
    out['vol_q1'] = q1
    out['vol_q2'] = q2
    lvl = pd.Series(np.where(out['vol'] <= q1, 'low', np.where(out['vol'] <= q2, 'mid', 'high')),
                    index=out.index, dtype=object).where(out['vol'].notna())
    out['vol_level'] = lvl
    out['regime'] = out['trend'].str.cat(lvl, sep='_').where(out['trend'].notna() & lvl.notna())
    return out


check("merkezlenmis SMA (ileriye bakar) -> B0 kirilir", bool(cutoff_invariant(mut_center, close, TS)))
check("tam-orneklem tercili (ileriye bakar) -> B0 kirilir", bool(cutoff_invariant(mut_fullsample, close, TS)))

print("\n[3] Anlam")
lab = rg.regime_labels(close)
up = lab['trend'].iloc[650:700]
hv = lab['vol_level'].iloc[700:800]
check("yukselis bolumunun sonunda (650..700) trend 'bull'", (up == 'bull').mean() > 0.95, str(up.value_counts().to_dict()))
check("yuksek oynaklik bolumunde (700..800) vol_level 'high'", (hv == 'high').mean() > 0.95, str(hv.value_counts().to_dict()))
first = lab['regime'].first_valid_index()
check("isinmada etiket yok: ilk etiket >= max(200, 60 + 252) gozlem sonra",
      lab.index.get_loc(first) >= max(rg.TREND_WINDOW - 1, rg.VOL_WINDOW + rg.MIN_VOL_HISTORY - 1),
      str(lab.index.get_loc(first)))
check("6 hucre; trend x vol_level", set(lab['regime'].dropna()) <= {f"{a}_{b}" for a in ('bull', 'bear') for b in rg.VOL_LEVELS})

print("\n[4] Y3: eski fonksiyon kaldirildi, yeni yol")
from prediction.evaluator import PredictionEvaluator as PE  # noqa: E402

check("compute_regime_metrics yok (gerceklesen getiriye kosullu 'rejim')", not hasattr(PE, 'compute_regime_metrics'))
n = 300
dates = close.index[-n:]
rng = np.random.default_rng(1)
cp = close.iloc[-n - 1:-1].to_numpy()
yt = close.iloc[-n:].to_numpy()
yp = yt * (1 + rng.normal(0, 0.01, n))
m_none = PE.compute_comprehensive_metrics(yt, yp, cp)
m_lab = PE.compute_comprehensive_metrics(yt, yp, cp, dates=dates, regime_labels=lab)
da = m_lab['regime_metrics']['direction_accuracy']
check("etiketsiz -> regime_metrics None; etiketle rejim basina n_samples toplami = n",
      m_none['regime_metrics'] is None and
      sum(v['n_samples'] for k, v in da.items() if not k.startswith('_')) + da['_unlabeled'] == n)

print("\n[5] as-of")
idx = lab.index
probe = pd.DatetimeIndex([idx[600], idx[600] + pd.Timedelta(days=1), idx[0] - pd.Timedelta(days=3)])
lab2 = lab.drop(index=idx[601]) if idx[600] + pd.Timedelta(days=1) == idx[601] else lab
a = rg.asof_labels(lab2, probe)
check("tatil (endekste yok) -> onceki islem gununun etiketi; seri oncesi -> NaN",
      a.iloc[0] == lab['regime'].iloc[600] and a.iloc[1] == lab['regime'].iloc[600] and pd.isna(a.iloc[2]),
      str(list(a)))

print("\n[6] Ozet")
toy = pd.DataFrame({'regime': ['a', 'a', 'b', 'a', np.nan, 'b', 'b']},
                   index=pd.bdate_range("2020-01-01", periods=7))
sm = rg.regime_summary(toy)
check("gun ve donem: a 3 gun / 2 donem, b 3 gun / 2 donem, etiketsiz 1",
      sm['a'] == {'days': 3, 'episodes': 2} and sm['b'] == {'days': 3, 'episodes': 2}
      and sm['_unlabeled_days'] == 1, str(sm))

print("\n[7] Gercek XU100")
path = os.path.join('data', 'macro', 'xu100.csv')
if os.path.exists(path):
    with open(path, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join('data', 'macro', 'xu100_source.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    check("xu100.csv SHA-256 = xu100_source.json", sha == meta['sha256'])
    from data.index_fetcher import load_index_close  # noqa: E402
    real = load_index_close(path)
    rts = [real.index.get_indexer([pd.Timestamp(d)], method='pad')[0]
           for d in ('2013-06-03', '2018-08-13', '2020-07-27', '2023-02-15', '2025-06-02')]
    bad = cutoff_invariant(rg.regime_labels, real, rts + [len(real) - 1])
    check("gercek seride B0 bit-es (2013..2026, 6 kesim)", not bad, f"kirilan: {bad}")
else:
    print("    [ATLANDI] data/macro/xu100.csv yok (`python -m data.index_fetcher`)")

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
