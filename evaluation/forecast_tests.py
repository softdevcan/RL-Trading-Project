"""A2 — Diebold-Mariano tahmin isabeti esitligi testi.

Kaynaklar:
  Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy.
    JBES 13:253-263. DOI 10.1080/07350015.1995.10524599
  Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of
    prediction mean squared errors. IJF 13:281-291. DOI 10.1016/S0169-2070(96)00719-4
  Newey, W. K. & West, K. D. (1987). Econometrica 55:703. DOI 10.2307/1913610
  Diebold, F. X. (2015). Comparing Predictive Accuracy, Twenty Years Later.
    JBES 33:1. DOI 10.1080/07350015.2014.983236 — DM TAHMINLERI karsilastirir,
    modelleri degil; test, tahmin hatalarinin uretildigi bolmeye kosulludur.

Eski `prediction/evaluator.py::diebold_mariano_test` d_t'nin otokorelasyonunu
yok sayiyordu (iid varyans) ve normal dagilim kullaniyordu; h-adim tahminlerde
(d_t MA(h-1)) ve otokorelasyonlu kayip farkinda boyutu belirgin bicimde asar
(olcum: tests/test_eval_stats.py).

Uzun donem varyans secenekleri (`lag`):
  'h-1'  : DM/HLN'nin kendi tanimi — dikdortgen cekirdek, h-1 gecikme.
           Kestirim negatif cikarsa Bartlett (Newey-West) agirligina dusulur
           ve sonucta `variance_fallback=True` isaretlenir.
  'auto' : Newey-West Bartlett, gecikme floor(4 (T/100)^(2/9)) — d_t'nin
           genel otokorelasyonu icin (h=1'de bile). HLN duzeltmesi yine
           h ile uygulanir.
  int    : Bartlett, verilen gecikme.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy import stats

LossArg = Union[str, Callable[[np.ndarray], np.ndarray]]


def _loss(errors: np.ndarray, loss: LossArg) -> np.ndarray:
    if callable(loss):
        return np.asarray(loss(errors), dtype=float)
    if loss == 'squared':
        return errors ** 2
    if loss == 'absolute':
        return np.abs(errors)
    raise ValueError(f"Bilinmeyen kayip: {loss}")


def _autocov(d: np.ndarray, k: int) -> float:
    n = len(d)
    dc = d - d.mean()
    return float(np.dot(dc[k:], dc[:n - k]) / n)


def long_run_variance(d: np.ndarray, lags: int, kernel: str = 'bartlett') -> float:
    """sum_{|k|<=lags} w_k gamma_k; gamma_k 1/T normalize."""
    v = _autocov(d, 0)
    for k in range(1, lags + 1):
        w = 1.0 if kernel == 'rectangular' else 1.0 - k / (lags + 1.0)
        v += 2.0 * w * _autocov(d, k)
    return v


def newey_west_lags(n: int) -> int:
    """Newey & West (1994) kural-tabanli gecikme: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(
    e1,
    e2,
    h: int = 1,
    loss: LossArg = 'squared',
    lag: Union[str, int] = 'h-1',
    alternative: str = 'two-sided',
) -> dict:
    """H0: E[L(e1) - L(e2)] = 0.

    Args:
        e1, e2: ayni hedeflere ait iki tahminin HATALARI (gercek - tahmin).
        h: tahmin ufku (adim).
        loss: 'squared' | 'absolute' | callable.
        lag: uzun donem varyans secimi (modul docstring'i).
        alternative: 'two-sided' | 'less' (model 1 daha iyi: E[d] < 0) | 'greater'.

    Returns:
        dm_stat (HLN duzeltilmis, t_{T-1} ile kiyaslanir), p_value, mean_loss_diff,
        lags, kernel, variance_fallback, n.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError("e1 ve e2 ayni uzunlukta olmali")
    mask = np.isfinite(e1) & np.isfinite(e2)
    d = _loss(e1[mask], loss) - _loss(e2[mask], loss)
    n = len(d)
    if n < 10:
        raise ValueError(f"En az 10 eslesmis gozlem gerekli, {n} var")
    if h < 1:
        raise ValueError("h >= 1 olmali")

    fallback = False
    if lag == 'h-1':
        lags, kernel = h - 1, 'rectangular'
        lrv = long_run_variance(d, lags, kernel)
        if lrv <= 0:
            kernel, fallback = 'bartlett', True
            lrv = long_run_variance(d, lags, kernel)
    elif lag == 'auto':
        lags, kernel = max(newey_west_lags(n), h - 1), 'bartlett'
        lrv = long_run_variance(d, lags, kernel)
    else:
        lags, kernel = int(lag), 'bartlett'
        lrv = long_run_variance(d, lags, kernel)

    d_mean = float(d.mean())
    if lrv <= 0:
        return {
            'dm_stat': 0.0, 'p_value': 1.0, 'mean_loss_diff': d_mean, 'n': n,
            'lags': lags, 'kernel': kernel, 'variance_fallback': fallback,
            'note': 'kayip farki sabit; test tanimsiz',
        }

    dm = d_mean / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold (1997) kucuk orneklem duzeltmesi
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    df = n - 1
    if alternative == 'two-sided':
        p = 2.0 * stats.t.sf(abs(stat), df)
    elif alternative == 'less':
        p = stats.t.cdf(stat, df)
    elif alternative == 'greater':
        p = stats.t.sf(stat, df)
    else:
        raise ValueError(f"Bilinmeyen alternatif: {alternative}")

    return {
        'dm_stat': stat,
        'p_value': float(p),
        'mean_loss_diff': d_mean,
        'n': n,
        'h': h,
        'lags': lags,
        'kernel': kernel,
        'variance_fallback': fallback,
        'distribution': f't({df})',
        'alternative': alternative,
    }
