"""A4 — Model Confidence Set (Hansen, Lunde & Nason 2011) — `arch` sarmalayicisi.

Kaynak: Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The Model Confidence
Set. Econometrica 79(2), 453-497. Uygulama: `arch.bootstrap.MCS` (denetlenmis
kutuphane; Faz 9 §1 "hazir ve denetlenmis kutuphane" ilkesi).

Girdi KAYIPTIR (dusuk = iyi): ornegin her modelin gunluk kare hata serisi ya da
stratejiler icin negatif getiri. Satir = zaman, sutun = model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def model_confidence_set(
    losses: pd.DataFrame,
    size: float = 0.10,
    reps: int = 2000,
    block_size: Optional[int] = None,
    method: str = 'R',
    bootstrap: str = 'stationary',
    seed=None,
) -> dict:
    """MCS'yi hesapla.

    Args:
        losses: T x M kayip matrisi (sutun adlari model adlari).
        size: 1 - guven duzeyi (0.10 -> %90 MCS).
        block_size: None ise `arch`in varsayilani (sqrt(T)).

    Returns:
        included / excluded model listeleri ve her modelin MCS p-degeri.
    """
    from arch.bootstrap import MCS

    if not isinstance(losses, pd.DataFrame):
        losses = pd.DataFrame(np.asarray(losses, dtype=float))
    losses = losses.dropna()
    if losses.shape[1] < 2:
        raise ValueError("En az iki model gerekli")
    mcs = MCS(losses, size=size, reps=reps, block_size=block_size,
              method=method, bootstrap=bootstrap, seed=seed)
    mcs.compute()
    pvals = mcs.pvalues.iloc[:, 0].to_dict()
    return {
        'included': list(mcs.included),
        'excluded': list(mcs.excluded),
        'pvalues': {str(k): float(v) for k, v in pvals.items()},
        'size': size,
        'reps': reps,
        'method': method,
        'bootstrap': bootstrap,
        'n': int(len(losses)),
    }
