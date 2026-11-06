"""A6 — Backtest asiri uyumu olasiligi (PBO) / Kombinatoryal Simetrik CV (CSCV).

Kaynak: Bailey, D. H., Borwein, J., Lopez de Prado, M. & Zhu, Q. J. (2017).
The Probability of Backtest Overfitting. Journal of Computational Finance
20(4), 39-69. (arsiv: G_evaluation/2017_Bailey_PBO.pdf)

Algoritma (makale §2.2):
  1. M (T x N): N yapilandirmanin ayni T donemdeki getirileri (deneme kaydi C2).
  2. Satirlari S esit, ardisik bloga bol (S cift).
  3. S/2 blogun tum C(S, S/2) birlesimi icin: J = egitim, J^c = test.
  4. J'de en iyi yapilandirma n* secilir; J^c'de n*'in goreli sirasi
     w = rank / (N + 1), lambda = ln(w / (1 - w)).
  5. PBO = P(lambda <= 0) — secilen yapilandirmanin test medyaninin altinda kalma
     olasiligi.
Performans olcutu: donem basi Sharpe (varsayilan) ya da cagiranin fonksiyonu.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Optional

import numpy as np
from scipy import stats


def _sharpe_cols(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sd > 0, x.mean(axis=0) / sd, 0.0)


def probability_of_backtest_overfitting(
    returns,
    n_blocks: int = 16,
    metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_combinations: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """PBO ve lambda dagilimi.

    Args:
        returns: (T, N) getiri matrisi (DataFrame ya da dizi). NaN satirlar atilir.
        n_blocks: S (cift). C(16, 8) = 12.870 birlesim.
        metric: (t, N) -> (N,) performans; varsayilan donem basi Sharpe.
        max_combinations: verilirse birlesimler rastgele bu sayiya orneklenir.
    """
    M = np.asarray(returns, dtype=float)
    M = M[np.all(np.isfinite(M), axis=1)]
    T, N = M.shape
    if N < 2:
        raise ValueError("En az iki yapilandirma gerekli")
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks cift ve >= 2 olmali")
    if T < 2 * n_blocks:
        raise ValueError(f"T={T} gozlem {n_blocks} blok icin az")
    metric = metric or _sharpe_cols

    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    if max_combinations and len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in sorted(pick)]

    lambdas = np.empty(len(combos))
    is_best = np.empty(len(combos))
    oos_of_best = np.empty(len(combos))
    for c, train_blocks in enumerate(combos):
        test_blocks = [b for b in range(n_blocks) if b not in train_blocks]
        tr = np.concatenate([blocks[b] for b in train_blocks])
        te = np.concatenate([blocks[b] for b in test_blocks])
        perf_tr = metric(M[tr])
        perf_te = metric(M[te])
        n_star = int(np.argmax(perf_tr))
        rank = stats.rankdata(perf_te)[n_star]          # 1 = en kotu, N = en iyi
        w = rank / (N + 1.0)
        lambdas[c] = np.log(w / (1.0 - w))
        is_best[c] = perf_tr[n_star]
        oos_of_best[c] = perf_te[n_star]

    slope = float(np.polyfit(is_best, oos_of_best, 1)[0]) if len(combos) > 2 else float('nan')
    return {
        'pbo': float(np.mean(lambdas <= 0)),
        'lambdas': lambdas,
        'n_combinations': len(combos),
        'n_strategies': N,
        'n_obs': T,
        'n_blocks': n_blocks,
        'degradation_slope': slope,       # IS performansina karsi OOS regresyon egimi
        'prob_oos_loss': float(np.mean(oos_of_best < 0)),
    }
