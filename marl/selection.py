"""Faz 10 M8 (tasarim A1): kontrol noktasi secimi; `evaluation/rl_folds.py`'nin (Faz 9 R1) saf numpy esi.

venv-marl `evaluation` paketini acamaz (pandas yok). Kural burada AYNI tanimla yazilir ve ana venv testi
(`test_marl_core` [8]) iki uygulamanin ayni sonucu verdigini sinar; tanim degisirse ikisi birlikte degisir.
  excess_sharpe     : gunluk basit getiri R ve ayni gunlerin risksiz faizi rf -> yillik fazla getiri Sharpe'i.
                      None: ucten az gecerli gozlem ya da BRUT getiri sabit (hic islem yok) -> secilemez.
  select_checkpoint : dogrulama metrigi en yuksek nokta, esitlikte en erken adim; hicbiri tanimli degilse SON
                      nokta (`degenerate=True`, sessiz degil).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

N_CHECKPOINTS = 10          # rl_folds.N_CHECKPOINTS ile ayni (tasarim varsayimi, Faz 9 §9.11)
PERIODS_PER_YEAR = 252


def excess_sharpe(returns, rf=None, periods: int = PERIODS_PER_YEAR) -> Optional[float]:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return None
    gross_ok = np.isfinite(r)
    if gross_ok.sum() < 3 or np.nanstd(r[gross_ok], ddof=1) == 0:
        return None
    if rf is not None:
        r = r - np.asarray(rf, dtype=float)
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return None
    sd = r[ok].std(ddof=1)
    if sd == 0:
        return None
    return float(r[ok].mean() / sd * np.sqrt(periods))


def select_checkpoint(records: List[dict]) -> dict:
    """records: {'step', 'metric' (None olabilir), 'params_sha'?}. Donus: secilen kayit + degenerate, n_valid,
    n_distinct (ayni agirlikli noktalar ayri aday degildir), last_metric."""
    if not records:
        raise ValueError("Kontrol noktasi kaydi yok")
    valid = [r for r in records if r.get('metric') is not None and np.isfinite(r['metric'])]
    n_distinct = len({r['params_sha'] for r in records if r.get('params_sha')}) or None
    last = max(records, key=lambda r: r['step'])
    diag = {'n_valid': len(valid), 'n_distinct': n_distinct, 'last_metric': last.get('metric')}
    if not valid:
        return {**last, 'degenerate': True, **diag}
    return {**sorted(valid, key=lambda r: (-r['metric'], r['step']))[0], 'degenerate': False, **diag}
