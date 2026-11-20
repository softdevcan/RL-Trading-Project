"""
Faz 10 M6c — rejime kosullu kol karsilastirmasi (`evaluation/regime_contrast.py`; H4 okuma araci).

Kontroller:
  1. Elle hesap: hucre basina D = tohum ortalamasi [SR(a|c) - SR(b|c)] (yillik) ve kontrast = D_bear - D_bull, elle
     hesapla esit; etiketsiz gun disarida; az gunlu hucre ve kontrast None; boyut uyusmazligi hata.
  2. Guc: etki yalniz 'bear' hucresinde (gunluk +%0,15 fazla getiri) -> kontrast GA'si 0'i disliyor, p < 0,05; 'bull'
     hucresindeki D, bear'dekinin yarisindan kucuk (GA'nin 0'i icermesi sansa bagli oldugu icin kosul degil).
  3. Boyut: etkisiz 100 sentetik seride (rejime bagli oynaklik, ortak piyasa sokuyla esli tohumlar) kontrastin %5
     duzeyde red orani <= 0,12 (yuzdelik bootstrap; beyan edilen tolerans).

Ag yok, ana venv. Calistirma:
    python tests/test_regime_contrast.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from evaluation import regime_contrast as RC  # noqa: E402

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


def labels_blocks(n, length=60):
    return np.array(['bear' if (i // length) % 2 == 0 else 'bull' for i in range(n)], dtype=object)


def make(n, k, rng, effect=0.0):
    lab = labels_blocks(n)
    vol = np.where(lab == 'bear', 0.02, 0.01)
    market = rng.normal(0, 1, n) * vol
    pairs = []
    for _ in range(k):
        b = market + rng.normal(0, 0.005, n)
        a = market + rng.normal(0, 0.005, n) + np.where(lab == 'bear', effect, 0.0)
        pairs.append((a, b))
    return pairs, lab


def sr(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252)


print("\n[1] Elle hesap")
rng = np.random.default_rng(1)
pairs, lab = make(600, 3, rng)
lab2 = lab.copy()
lab2[:10] = None
res = RC.regime_contrast(pairs, lab2, n_boot=50)
m_bear = (lab2 == 'bear')
m_bull = (lab2 == 'bull')
d_bear = np.mean([sr(a[m_bear]) - sr(b[m_bear]) for a, b in pairs])
d_bull = np.mean([sr(a[m_bull]) - sr(b[m_bull]) for a, b in pairs])
small = RC.regime_contrast(pairs, lab, n_boot=20, min_days=400)
try:
    RC.regime_contrast(pairs, lab[:-1], n_boot=5)
    dim_err = False
except ValueError:
    dim_err = True
check("hucre D ve kontrast elle hesapla esit; etiketsiz gun disarida; az gunlu hucre None; boyut hatasi",
      np.isclose(res['cells']['bear']['D'], d_bear) and np.isclose(res['cells']['bull']['D'], d_bull)
      and np.isclose(res['contrast']['value'], d_bear - d_bull) and res['n_days'] == 590
      and small['cells']['bear']['D'] is None and small['contrast'] is None and dim_err,
      f"{res['cells']} {d_bear} {d_bull}")

print("\n[2] Guc")
rng = np.random.default_rng(2)
pairs, lab = make(1500, 3, rng, effect=0.0015)
res = RC.regime_contrast(pairs, lab, n_boot=1000)
c = res['contrast']
d_bear, d_bull = res['cells']['bear']['D'], res['cells']['bull']['D']
check("etki yalniz bear'de: kontrast GA'si 0'i disliyor ve p < 0,05; |D_bull| < D_bear / 2",
      c['ci'][0] > 0 and c['p_two_sided_pct'] < 0.05 and abs(d_bull) < d_bear / 2, f"{c} bear {d_bear} bull {d_bull}")

print("\n[3] Boyut (etkisiz, 100 seri)")
rng = np.random.default_rng(3)
rej = 0
for i in range(100):
    pairs, lab = make(1000, 3, rng)
    r = RC.regime_contrast(pairs, lab, n_boot=300, seed=i)
    rej += r['contrast']['p_two_sided_pct'] < 0.05
check(f"etkisiz seride %5 duzeyde red orani <= 0,12 (olculen {rej / 100:.2f})", rej / 100 <= 0.12, f"red orani {rej / 100}")

print("\n" + "=" * 60)
print(f"Sonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
