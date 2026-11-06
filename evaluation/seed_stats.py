"""A5 — Tohumlar arasi raporlama (Agarwal vd. 2021) — `rliable` sarmalayicisi.

Kaynak: Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. & Bellemare,
M. G. (2021). Deep Reinforcement Learning at the Edge of the Statistical
Precipice. NeurIPS 2021. (arsiv: G_evaluation/2021_Agarwal_StatisticalPrecipice.pdf)

Skor matrisi: (n_kosum, n_gorev). Bu projede tipik olarak kosum = tohum,
gorev = walk-forward kati (tek kat varsa n_gorev = 1).

CIKARIM BIRIMI (hakem-2 #6): tek pencerede tohum araliklari yalnizca EGITIM
rastlantisalligini olcer, piyasa orneklem belirsizligini olcmez. Anlamlilik
kaniti olarak sunulmaz; zaman serisi testleriyle (sharpe_tests, sharpe.DSR)
AYRI raporlanir. Gorevler walk-forward katlari olmali (R1).

Ciktilar: IQM, medyan, ortalama, iyimserlik acigi (optimality gap) icin
katmanli bootstrap guven araliklari; iki yontem arasi iyilesme olasiligi
P(X > Y); performans profilleri.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

AGGREGATES = ('median', 'iqm', 'mean', 'optimality_gap')
_GLOBAL_RNG_LOCK = threading.Lock()


@contextmanager
def _seeded_global_rng(seed: Optional[int]):
    """rliable 1.2.0 `random_state`'i KULLANMIYOR: `StratifiedBootstrap.update_indices`
    yeniden orneklemeyi kuresel `np.random.choice` ile yapiyor (olculdu: ayni
    random_state ile iki cagri farkli GA verdi). Tekrar uretilebilirlik icin
    kuresel durum bu blok boyunca tohumlanir ve cikista GERI YUKLENIR.
    """
    if seed is None:
        yield
        return
    # Kilit: ayni surecte baska bir is parcacigi (or. SB3 egitimi) kuresel
    # durumu ayni anda kullanirsa tohumlama yarisir (hakem-2 #13). Kilit yalnizca
    # bu modulun cagrilarini siralar; kuresel np.random kullanan ve kilidi
    # bilmeyen kodla ayni anda calistirmayin.
    with _GLOBAL_RNG_LOCK:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)


def _as_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError("Skorlar (n_kosum, n_gorev) bicimde olmali")
    return a


def _aggregate_func(gap_threshold: float):
    from rliable import metrics

    def f(x):
        return np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x, gap_threshold),
        ])
    return f


def aggregate_with_ci(
    score_dict: Mapping[str, np.ndarray],
    reps: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
    gap_threshold: float = 1.0,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Her yontem icin {metrik: (nokta, alt, ust)} — katmanli bootstrap.

    `gap_threshold`: iyimserlik acigi icin hedef skor (skor birimine bagli;
    Sharpe icin anlamli bir esik cagiran tarafindan secilmeli).
    """
    from rliable import library as rly

    scores = {k: _as_matrix(v) for k, v in score_dict.items()}
    with _seeded_global_rng(seed):
        point, ci = rly.get_interval_estimates(
            scores, _aggregate_func(gap_threshold), reps=reps,
            confidence_interval_size=confidence,
        )
    out = {}
    for k in scores:
        out[k] = {name: (float(point[k][i]), float(ci[k][0, i]), float(ci[k][1, i]))
                  for i, name in enumerate(AGGREGATES)}
    return out


def probability_of_improvement(
    pairs: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    reps: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = 0,
) -> Dict[str, Tuple[float, float, float]]:
    """{'X,Y': (P(X > Y), alt, ust)} — Mann-Whitney U tabanli, gorev ortalamasi."""
    from rliable import library as rly, metrics

    data = {k: (_as_matrix(x), _as_matrix(y)) for k, (x, y) in pairs.items()}
    # IndependentSamplesBootstrap arch'in kendi uretecini kullanir; kuresel
    # tohum + arch seed'i birlikte verilir.
    with _seeded_global_rng(seed):
        point, ci = rly.get_interval_estimates(
            data, metrics.probability_of_improvement, reps=reps,
            confidence_interval_size=confidence,
            random_state=np.random.RandomState(seed) if seed is not None else None,
        )
    return {k: (float(np.squeeze(point[k])), float(np.squeeze(ci[k][0])),
                float(np.squeeze(ci[k][1]))) for k in data}


def performance_profiles(
    score_dict: Mapping[str, np.ndarray],
    taus: Sequence[float],
    reps: int = 2000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """P(skor > tau) egrisi ve guven bandi (rliable.create_performance_profile)."""
    from rliable import library as rly

    scores = {k: _as_matrix(v) for k, v in score_dict.items()}
    with _seeded_global_rng(0):
        return rly.create_performance_profile(scores, np.asarray(taus, dtype=float), reps=reps)
