"""Faz 10 M8 ön kaydı (tasarım §6.4) istatistik çekirdeği; saf numpy. Okuma (`analyze_m8.py`) ve güç benzetimi
(`power_m8.py`) AYNI fonksiyonları kullanır; kural yalnız burada tanımlıdır.

Girdi: d (F x k) eşli fark matrisi; satır = kat, sütun = tohum, d[f, s] = SR_a(s, f) - SR_b(s, f). NaN = eksik hücre
(eşleşmeyen hücre düşer).
  delta_bar      : eşit ağırlıklı ortalama (mevcut hücreler üzerinden)
  p_signflip     : eşli işaret-çevirme permütasyonu; her eşin işareti bağımsız; iki yönlü,
                   p = (1 + #{|D0| >= |D|}) / (R + 1)  (hakem-rl M8 #B1, #K3)
  p_bootstrap    : kat içinde tabakalı bootstrap (her katta eşler yerine koyarak); iki yönlü,
                   p = min(1, 2 min((1 + #{D* <= 0}), (1 + #{D* >= 0})) / (B + 1)); %95 yüzdelik GA
  label          : p_signflip <= alpha VE p_bootstrap <= alpha -> FARK_VAR_<a> / FARK_VAR_<b>; aksi FARK_GOZLENMEDI
  holm           : Holm-Bonferroni düzeltilmiş p (girdi sırasında)
  fold_signflip  : betimsel; 6 kat ortalamasının işaret-çevirmesi (tam sayım 2^F), iki yönlü
"""
from __future__ import annotations

import itertools

import numpy as np

ALPHA = 0.05
R_SIGNFLIP = 10_000
B_BOOT = 10_000
SEED_SIGNFLIP = 1
SEED_BOOT = 0


def _cells(d):
    d = np.asarray(d, dtype=float)
    if d.ndim != 2:
        raise ValueError('d (kat x tohum) olmali')
    return d


def delta_bar(d) -> float:
    d = _cells(d)
    return float(np.nanmean(d))


def p_signflip(d, R: int = R_SIGNFLIP, seed: int = SEED_SIGNFLIP) -> float:
    x = _cells(d)
    x = x[np.isfinite(x)]
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for lo in range(0, R, 2000):
        n = min(2000, R - lo)
        s = rng.integers(0, 2, size=(n, x.size)) * 2 - 1
        hits += int((np.abs((s * x).mean(axis=1)) >= obs - 1e-15).sum())
    return (1 + hits) / (R + 1)


def bootstrap(d, B: int = B_BOOT, seed: int = SEED_BOOT) -> dict:
    d = _cells(d)
    rng = np.random.default_rng(seed)
    rows = [r[np.isfinite(r)] for r in d]
    rows = [r for r in rows if r.size]
    n_tot = sum(r.size for r in rows)
    sums = np.zeros(B)
    for r in rows:
        idx = rng.integers(0, r.size, size=(B, r.size))
        sums += r[idx].sum(axis=1)
    star = sums / n_tot
    lo_p = (1 + int((star <= 0).sum())) / (B + 1)
    hi_p = (1 + int((star >= 0).sum())) / (B + 1)
    return {'p': float(min(1.0, 2 * min(lo_p, hi_p))),
            'ci95': [float(np.quantile(star, 0.025)), float(np.quantile(star, 0.975))]}


def label(delta: float, p_sf: float, p_bt: float, a: str, b: str, alpha: float = ALPHA) -> str:
    if p_sf <= alpha and p_bt <= alpha and delta != 0:
        return f'FARK_VAR_{a if delta > 0 else b}'
    return 'FARK_GOZLENMEDI'


def contrast(d, a: str, b: str, R: int = R_SIGNFLIP, B: int = B_BOOT) -> dict:
    dl = delta_bar(d)
    psf = p_signflip(d, R=R)
    bt = bootstrap(d, B=B)
    return {'delta_bar': dl, 'p_signflip': psf, 'p_bootstrap': bt['p'], 'ci95': bt['ci95'],
            'p_max': max(psf, bt['p']), 'label': label(dl, psf, bt['p'], a, b),
            'n_pairs': int(np.isfinite(_cells(d)).sum())}


def holm(pvals) -> list:
    p = np.asarray(pvals, dtype=float)
    m, order, out, running = len(p), np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def fold_signflip(d) -> dict:
    """Betimsel (hakem-finans M8 #1): kat ortalamaları üzerinde tam işaret-çevirme; F = 6'da en küçük p 2/64."""
    m = np.nanmean(_cells(d), axis=1)
    m = m[np.isfinite(m)]
    obs = abs(m.mean())
    null = [abs(np.mean(np.array(s) * m)) for s in itertools.product((-1, 1), repeat=m.size)]
    return {'fold_means': m.tolist(), 'n_same_sign': int(max((m > 0).sum(), (m < 0).sum())),
            'p': float(np.mean(np.array(null) >= obs - 1e-15))}
