"""
Faz 10 M8 A4: kural portfoyleri `marl/rule_baselines.py` (on kayit §6.2 a-f; hakem-finans #6 ve 2. okuma (e)).
ANA venv (sklearn/scipy yalniz parite icin).

Kontroller:
  1. Ledoit-Wolf: kovaryans ve daraltma `sklearn.covariance.ledoit_wolf` ile ayni.
  2. Uzun-yalniz min-varyans: scipy SLSQP ile ayni amac degeri; agirlik >= 0, toplam 1.
  3. HRP: kosegen kovaryansta ters varyans agirligina esit; rastgele kovaryansta uzun-yalniz, toplam 1.
  4. (e) ileriye bakmama: `monthly_target` t'den sonrasi kesilmis panelle ayni sonucu verir (sentetik + gercek
     panel_m8'de 2021 katinin ilk karar satiri 2020-12-31; panel yoksa atlanir).
  5. (a)/(c) seri: yeniden dengeleme yalniz ilk karar satiri + ayin ilk seansi; ay icinde devir 0; gec listelenen
     hisse 252 getirisi dolana kadar evrene girmez; uzun-yalniz (nakit >= 0); ew/bh/cash eski davranis (cash X = 0).

Ag yok. ~10 sn.
    python tests/test_marl_baselines.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import accounting as A  # noqa: E402
from marl import rule_baselines as RB  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


rng = np.random.default_rng(7)

# 1. Ledoit-Wolf
from sklearn.covariance import ledoit_wolf as sk_lw  # noqa: E402
x = rng.normal(0, 0.02, (252, 12)) @ rng.normal(0, 1, (12, 12)) * 0.3
cov, sh = RB.ledoit_wolf(x)
cov_sk, sh_sk = sk_lw(x)
check("1. Ledoit-Wolf = sklearn (kovaryans ve daraltma)", np.allclose(cov, cov_sk, rtol=1e-10, atol=1e-14)
      and abs(sh - sh_sk) < 1e-12)

# 2. min-varyans
from scipy.optimize import minimize  # noqa: E402
ok2 = True
for _ in range(5):
    p = int(rng.integers(3, 25))
    m = rng.normal(0, 1, (p, p))
    c = m @ m.T / p + np.diag(rng.uniform(0.05, 1.0, p))
    w = RB.minvar_long_only(c)
    ref = minimize(lambda v: v @ c @ v, np.full(p, 1 / p), method='SLSQP', bounds=[(0, 1)] * p,
                   constraints=[{'type': 'eq', 'fun': lambda v: v.sum() - 1}], options={'ftol': 1e-15, 'maxiter': 500})
    ok2 &= bool(w.min() >= 0 and abs(w.sum() - 1) < 1e-9 and w @ c @ w <= ref.fun * (1 + 1e-6) + 1e-14)
check("2. min-varyans: uzun-yalniz, toplam 1, SLSQP amaciyla ayni (5 rastgele kovaryans)", ok2)

# 3. HRP
d = rng.uniform(0.01, 0.2, 9)
check("3a. HRP kosegen kovaryansta = ters varyans", np.allclose(RB.hrp(np.diag(d)), (1 / d) / (1 / d).sum()))
w = RB.hrp(c)
check("3b. HRP rastgele kovaryansta uzun-yalniz, toplam 1", w.min() > 0 and abs(w.sum() - 1) < 1e-12)


# sentetik panel: 3 yil, 8 hisse; hisse 7 300. satirda listelenir; ay basi tarihleri gercek takvim
def synth(T=760, N=8):
    import pandas as pd
    dates = np.array([str(x.date()) for x in pd.bdate_range('2016-01-04', periods=T)])
    lr = rng.normal(0.0004, 0.015, (T, N))
    close = 10 * np.exp(np.cumsum(lr, axis=0))
    opn = close * np.exp(rng.normal(0, 0.003, (T, N)))
    close[:300, 7] = opn[:300, 7] = np.nan
    return {'dates': dates, 'open': opn, 'close': close, 'rf': np.full(T, 0.0004)}


arr = synth()
px = A.fill_prices(arr['open'], arr['close'], A.LISTING_LOCKOUT, 0.095)

# 4. (e) ileriye bakmama: kesilmis panel
ok4 = True
for t in (400, 520, 640):
    cut = {k: (v[:t + 1] if k != 'rf' else v[:t + 1]) for k, v in arr.items()}
    pc = A.fill_prices(cut['open'], cut['close'], A.LISTING_LOCKOUT, 0.095)
    for kind in RB.MONTHLY:
        ok4 &= bool(np.array_equal(RB.monthly_target(px, t, kind)[0], RB.monthly_target(pc, t, kind)[0]))
check("4a. (e) monthly_target t sonrasi kesilmis panelle ayni (minvar, hrp; 3 tarih)", ok4)

PANEL = os.path.join('data', 'marl', 'panel_m8.npz')
if os.path.exists(PANEL):
    from marl import panel as PN
    real = PN.load_npz(PANEL)
    t0 = int(np.flatnonzero(real['dates'] < '2021-01-01')[-1])        # eval_fold_m8.decision_rows kurali
    full = A.fill_prices(real['open'], real['close'], A.LISTING_LOCKOUT, A.limit_down_gaps(real['dates']))
    trunc = A.fill_prices(real['open'][:t0 + 1], real['close'][:t0 + 1], A.LISTING_LOCKOUT,
                          A.limit_down_gaps(real['dates'][:t0 + 1]))
    same = all(np.array_equal(RB.monthly_target(full, t0, k)[0], RB.monthly_target(trunc, t0, k)[0]) for k in RB.MONTHLY)
    u0, info = RB.monthly_target(full, t0, 'minvar')
    check(f"4b. (e) gercek panel: 2021 katinin ilk agirligi yalniz <= {real['dates'][t0]} verisinden "
          f"(evren {info['n_universe']})", str(real['dates'][t0]) == '2020-12-31' and same and info['n_universe'] > 0)
else:
    print(f"[ATLANDI] 4b: {PANEL} yok")

# 5. seri
s0 = 300
ser = {k: RB.rule_series(arr, s0, 700, k, 20.0, limit_down_gap=0.095) for k in RB.KINDS}
reb = RB.rebalance_rows(arr['dates'], s0, 700)
months = {str(d)[:7] for d in arr['dates'][s0:700]}
off = [i for i, t in enumerate(range(s0, 700)) if t not in set(reb)]
ok5 = all(len(ser[k]['rebalances']) == len(reb) == len(months) for k in RB.MONTHLY)
ok5 &= all(np.abs(ser[k]['turnover'][off]).max() < 1e-12 for k in RB.MONTHLY)
ok5 &= all(ser[k]['cash_weight'].min() > -1e-12 for k in RB.KINDS)
uni = {k: [(r['date'], r['n_universe']) for r in ser[k]['rebalances']] for k in RB.MONTHLY}
late_in = [n for d, n in uni['minvar'] if d < arr['dates'][300 + 252]]
late_after = [n for d, n in uni['minvar'] if d > arr['dates'][300 + 253 + 20]]
ok5 &= max(late_in) == 7 and min(late_after) == 8
ok5 &= np.abs(ser['cash']['X']).max() < 1e-15
check("5. seri: dengeleme ilk satir + ay basi; ay icinde devir 0; gec listelenen hisse 252 getiri dolmadan evrende "
      "yok; nakit >= 0; cash X = 0", ok5)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
