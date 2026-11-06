"""A9 — Iki stratejinin AYNI GUNLERDEKI getiri farki icin eslestirilmis testler.

Eski `ModelAnalyzer.statistical_significance_test` `scipy.stats.ttest_ind`
kullaniyordu: ayni piyasa gunlerinde olculen iki getiri serisini BAGIMSIZ
orneklem sayiyordu. Iki strateji ayni piyasaya maruz kaldigi icin getirileri
guclu korelasyonludur; bagimsiz test farkin varyansini sisirir (gucu bosa
harcar) ve otokorelasyonu yok sayar (Faz 9 A9; dogrulama §6 #8).

Burada:
  paired_t        : d_t = r1_t - r2_t uzerinde tek orneklem t-testi (iid d_t
                    varsayimi — yalnizca karsilastirma/geri uyum icin).
  mean_diff_hac   : d_t ortalamasi, Newey-West HAC standart hatasi.
  mean_diff_bootstrap : d_t ortalamasi, durağan bootstrap (Politis & Romano
                    1994; `arch.bootstrap.StationaryBootstrap`).
Sharpe farki icin `evaluation.sharpe_tests` kullanilir (ortalama farki ile
Sharpe farki ayni hipotez degildir).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from evaluation.forecast_tests import long_run_variance, newey_west_lags


def _diff(r1, r2) -> np.ndarray:
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if r1.shape != r2.shape:
        raise ValueError(
            "Eslestirilmis test ayni tarihlerdeki getirileri ister; seriler "
            f"farkli uzunlukta ({r1.shape} vs {r2.shape}). Once tarihe gore hizalayin."
        )
    d = r1 - r2
    d = d[np.isfinite(d)]
    if len(d) < 10:
        raise ValueError(f"En az 10 eslesmis gozlem gerekli, {len(d)} var")
    return d


def paired_t(r1, r2) -> dict:
    d = _diff(r1, r2)
    res = stats.ttest_1samp(d, 0.0)
    return {'test': 'Paired t-test', 'statistic': float(res.statistic),
            'p_value': float(res.pvalue), 'mean_diff': float(d.mean()), 'n': len(d)}


def mean_diff_hac(r1, r2, lags=None) -> dict:
    d = _diff(r1, r2)
    n = len(d)
    L = newey_west_lags(n) if lags is None else int(lags)
    lrv = long_run_variance(d, L, 'bartlett')
    stat = float(d.mean() / np.sqrt(lrv / n)) if lrv > 0 else 0.0
    return {'test': f'Mean difference, Newey-West HAC ({L} lags)', 'statistic': stat,
            'p_value': float(2 * stats.t.sf(abs(stat), n - 1)),
            'mean_diff': float(d.mean()), 'n': n, 'lags': L}


def mean_diff_bootstrap(r1, r2, n_boot: int = 4999, block_size=None, seed=0) -> dict:
    """Iki yonlu p-degeri: ortalamasi sifira kaydirilmis d uzerinde durağan bootstrap."""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    d = _diff(r1, r2)
    b = float(optimal_block_length(d)['stationary'].iloc[0]) if block_size is None else float(block_size)
    b = max(b, 1.0)
    obs = d.mean()
    centered = d - obs
    bs = StationaryBootstrap(b, centered, seed=seed)
    means = np.array([x[0][0].mean() for x in bs.bootstrap(n_boot)])
    p = (np.sum(np.abs(means) >= abs(obs)) + 1.0) / (n_boot + 1.0)
    return {'test': 'Mean difference, stationary bootstrap', 'statistic': float(obs),
            'p_value': float(p), 'mean_diff': float(obs), 'n': len(d),
            'block_size': b, 'n_boot': n_boot}
