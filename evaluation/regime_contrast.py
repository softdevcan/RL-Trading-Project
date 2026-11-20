"""Rejime kosullu kol karsilastirmasi: hucre basina Sharpe farki ve hucreler arasi fark-farki (Faz 10 M6c; H4).

H4 (Faz 9 M8): "iletisimin marjinal katkisi onceden tanimli zamansal rejimlere gore farkli". Istatistik:
  D_c = (1/K) sum_k [SR(a_k | c) - SR(b_k | c)]      (yillik; c hucresindeki gunler; K esli tohum)
  kontrast = D_{c1} - D_{c2}                           (iki hucreli birincil okuma, K-M6-1: c1 = 'bear', c2 = 'bull')
Belirsizlik: dairesel blok bootstrap (Politis & Romano 1994 ailesi). Ayni yeniden ornekleme gun indeksleri BUTUN tohumlara
ve etikete birlikte uygulanir: gunler arasi bagimlilik ve tohumlar arasi ortak piyasa gunu korunur. Ardisik rejim bloklari
bagimsiz donem sayilmaz (Faz 9 M8 maddesi); blok uzunlugu cagiran tarafindan verilir (varsayilan 20 seans). Katlar
birlestirilerek kullanilir (seriler tarih sirasiyla ard arda verilir; kat sinirini asan blok beyan edilen bir yaklasimdir).

Tasarim varsayimlari: Sharpe donem basi ortalama / std x sqrt(252) (fazla getiri girdisi cagirandan), hucrede en az
`min_days` gun yoksa o hucre ve kontrast None. Iki yonlu yuzdelik p = 2 min(P(b <= 0), P(b >= 0)) (bootstrap dagilimi
gozlenen degerin cevresinde; H0'a kaydirilmis degil, beyan). Bu bir okuma ARACIDIR; esik ve karar M8 on kaydinda.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

PERIODS = 252
MIN_DAYS = 60


def _sr(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(s > 0, m / s * np.sqrt(PERIODS), np.nan)


def _cell_d(a: np.ndarray, b: np.ndarray, lab: np.ndarray, cell: str, min_days: int) -> float:
    m = lab == cell
    if m.sum() < min_days:
        return np.nan
    return float(np.mean(_sr(a[:, m]) - _sr(b[:, m])))


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n


def regime_contrast(pairs: Sequence[Tuple[np.ndarray, np.ndarray]], labels: Sequence, cells: Sequence[str] = ('bear', 'bull'),
                    n_boot: int = 2000, block: int = 20, seed: int = 0, min_days: int = MIN_DAYS,
                    alpha: float = 0.05) -> Dict[str, object]:
    """pairs: K esli tohum icin (a, b) gunluk fazla getiri dizileri, hepsi ayni tarihlerde ve `labels` ile hizali.
    labels: gun basina hucre adi (etiketsiz gun None/NaN -> disarida). Donus: hucre basina D, gun sayisi, GA; kontrast
    (cells[0] - cells[1]) GA ve iki yonlu yuzdelik p."""
    a = np.asarray([np.asarray(p[0], dtype=float) for p in pairs])
    b = np.asarray([np.asarray(p[1], dtype=float) for p in pairs])
    lab = np.asarray([str(x) if x is not None and x == x else '' for x in labels], dtype=object)
    if a.shape != b.shape or a.shape[1] != len(lab):
        raise ValueError(f"boyut uyusmuyor: a {a.shape}, b {b.shape}, etiket {len(lab)}")
    keep = lab != ''
    a, b, lab = a[:, keep], b[:, keep], lab[keep]
    n = lab.size
    obs = {c: _cell_d(a, b, lab, c, min_days) for c in cells}
    contrast = obs[cells[0]] - obs[cells[1]] if len(cells) == 2 else np.nan
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, len(cells)), np.nan)
    for i in range(n_boot):
        idx = circular_block_indices(n, block, rng)
        ai, bi, li = a[:, idx], b[:, idx], lab[idx]
        boots[i] = [_cell_d(ai, bi, li, c, min_days) for c in cells]
    q = (100 * alpha / 2, 100 * (1 - alpha / 2))
    out: Dict[str, object] = {'cells': {}, 'n_days': int(n), 'k_pairs': int(a.shape[0]), 'block': block,
                              'n_boot': n_boot, 'seed': seed}
    for j, c in enumerate(cells):
        col = boots[:, j][np.isfinite(boots[:, j])]
        out['cells'][c] = {'n_days': int((lab == c).sum()), 'D': None if np.isnan(obs[c]) else obs[c],
                           'ci': None if np.isnan(obs[c]) or col.size == 0 else [float(np.percentile(col, q[0])),
                                                                               float(np.percentile(col, q[1]))]}
    if len(cells) == 2 and np.isfinite(contrast):
        diff = boots[:, 0] - boots[:, 1]
        diff = diff[np.isfinite(diff)]
        p = 2 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0))) if diff.size else None
        out['contrast'] = {'name': f'{cells[0]} - {cells[1]}', 'value': float(contrast),
                           'ci': [float(np.percentile(diff, q[0])), float(np.percentile(diff, q[1]))],
                           'p_two_sided_pct': None if p is None else min(1.0, p), 'n_valid_boot': int(diff.size)}
    else:
        out['contrast'] = None
    return out
