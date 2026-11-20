"""
Faz 10 M6a — ikinci rejim tanimi: HMM'in yalniz filtrelenmis olasiliklari (`evaluation/regimes_hmm.py`; K10, K-M6-2).

Kontroller (sentetik iki durumlu seri; bilinen durum dizisi):
  1. B0 kesme degismezligi: tam seride ve t'de kesilmis seride t'ye kadarki olasiliklar bit-es (3 kesim noktasi,
     ayni capa takvimi). Mutasyon: filtrelenmis yerine DUZELTILMIS olasilik kesme degismezligini KIRAR.
  2. Parametre nedenselligi: her capanin uyumu yalniz capadan onceki getirilerle (fit_log n_fit = capa konumu);
     ilk capadan once yeterli gecmis yoksa hata. Uyum deterministik: ayni veride iki uyum bit-es (search_reps
     kuresel NumPy ureticisini kullanir; tohumsuz uyum 1. kontrolu ~1e-5 ile kiriyordu) ve kuresel durum geri yuklenir.
  3. BIC: iki durumlu veride {2, 3} icinden 2 secilir.
  4. Kanonik sira ve anlam: p0 en sakin durum (sigma artan), olasiliklar 1'e toplanir; yuksek oynaklik olasiligi
     bilinen durumu yakalar (isabet > 0,8); 'two_cell' etiketleri ve mean_desc_with_sd olcumu yazilir.

Ag yok, ana venv. Calistirma:
    python tests/test_regimes_hmm.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from evaluation import regimes_hmm as H  # noqa: E402

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


def synthetic(n=1400, seed=0):
    """Iki durumlu Markov zinciri: sakin (mu 0,1, sd 1) ve calkantili (mu -0,4, sd 3); kalicilik 0,98."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < 0.98 else 1 - s[t - 1]
    r = np.where(s == 0, rng.normal(0.1, 1.0, n), rng.normal(-0.4, 3.0, n))
    idx = pd.bdate_range('2000-01-03', periods=n)
    return pd.Series(r, index=idx), pd.Series(s, index=idx)


r, s_true = synthetic()
FIRST = str(r.index[600].date())
EVERY = 200

print("\n[1] B0 kesme degismezligi")
full = H.filtered_regimes(r, FIRST, refit_every=EVERY, k=2)
cuts = [r.index[700], r.index[1001], r.index[1250]]
diffs = []
for c in cuts:
    part = H.filtered_regimes(r, FIRST, refit_every=EVERY, k=2, end=str((c + pd.Timedelta(days=1)).date()))
    common = part.probs.index
    diffs.append(float(np.nanmax(np.abs(full.probs.loc[common].values - part.probs.values))))
orig = H._filtered


def smoothed(x, k, params):
    m = H._model(np.asarray(x, dtype=float), k)
    p = np.asarray(m.smooth(params).smoothed_marginal_probabilities)
    return p if p.shape[1] == k else p.T


H._filtered = smoothed
try:
    full_m = H.filtered_regimes(r, FIRST, refit_every=EVERY, k=2)
    part_m = H.filtered_regimes(r, FIRST, refit_every=EVERY, k=2, end=str((cuts[1] + pd.Timedelta(days=1)).date()))
    d_m = float(np.nanmax(np.abs(full_m.probs.loc[part_m.probs.index].values - part_m.probs.values)))
finally:
    H._filtered = orig
check("filtrelenmis olasiliklar kesimde bit-es; duzeltilmis olasilik (mutasyon) B0'i kirar",
      max(diffs) == 0.0 and d_m > 1e-6, f"diffs={diffs} mutasyon={d_m}")

print("\n[2] Parametre nedenselligi")
pos = [int(r.index.searchsorted(pd.Timestamp(f['anchor']))) for f in full.fit_log]
try:
    H.filtered_regimes(r, str(r.index[100].date()), k=2)
    short_err = False
except ValueError:
    short_err = True
np.random.seed(123)
before = np.random.random()
np.random.seed(123)
p1 = H.fit_params(r.values[:600], 2)[0]
p2 = H.fit_params(r.values[:600], 2)[0]
after = np.random.random()
check("her capanin uyumu yalniz capa oncesiyle (n_fit = capa konumu); kisa gecmis hata verir; uyum deterministik ve "
      "kuresel NumPy durumunu degistirmez",
      all(f['n_fit'] == p for f, p in zip(full.fit_log, pos)) and pos[0] == 600
      and np.diff(pos).tolist() == [EVERY] * (len(pos) - 1) and short_err and np.array_equal(p1, p2)
      and before == after, f"{pos}")

print("\n[3] BIC")
k_sel, bics = H.select_k(r.values[:600])
check("iki durumlu veride BIC 2 secer", k_sel == 2, f"{bics}")

print("\n[4] Kanonik sira ve anlam")
lab = H.hmm_labels(full)
sd_ok = all(f['sd_sorted'][0] < f['sd_sorted'][1] for f in full.fit_log)
sums = full.probs.sum(axis=1)
truth = s_true.loc[lab.index]
acc = float(((lab['p_high_vol'] >= 0.5).astype(int) == truth).mean())
check("p0 en sakin durum, olasiliklar 1'e toplanir, yuksek oynaklik olasiligi bilinen durumu yakalar (> 0,8), "
      "iki hucreli etiket ve ortalama sirasi olcumu yazilir",
      sd_ok and np.allclose(sums, 1.0) and acc > 0.8 and set(lab['two_cell']) <= {'bull_like', 'bear_like'}
      and all('mean_desc_with_sd' in f for f in full.fit_log), f"acc={acc}")

print("\n" + "=" * 60)
print(f"Sonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
