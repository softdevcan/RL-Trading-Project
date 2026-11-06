"""A7 — Coklu karsilastirma: Benjamini-Hochberg FDR (statsmodels sarmalayicisi).

Kaynak: Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery
Rate. JRSS-B 57:289-300. DOI 10.1111/j.2517-6161.1995.tb02031.x

Aile tanimi (hangi p-degerleri birlikte duzeltilir) M8 on kaydinda sabitlenir:
aile = bir hipotezin tum karsilastirmalari. Bu fonksiyon yalnizca duzeltmeyi
yapar; aileyi cagiran kurar.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np


def benjamini_hochberg(
    pvalues: Union[Sequence[float], Mapping[str, float]], alpha: float = 0.05
) -> dict:
    from statsmodels.stats.multitest import multipletests

    if isinstance(pvalues, Mapping):
        names = list(pvalues.keys())
        p = np.asarray([pvalues[k] for k in names], dtype=float)
    else:
        p = np.asarray(pvalues, dtype=float)
        names = [str(i) for i in range(len(p))]
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("p-degerleri [0, 1] araliginda ve sonlu olmali")
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method='fdr_bh')
    return {
        'alpha': alpha,
        'method': 'fdr_bh',
        'family_size': int(len(p)),
        'results': {n: {'p': float(pi), 'p_adj': float(pa), 'reject': bool(r)}
                    for n, pi, pa, r in zip(names, p, p_adj, reject)},
    }
