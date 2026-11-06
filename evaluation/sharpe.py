"""A1 — Probabilistic ve Deflated Sharpe Ratio.

Kaynak: Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
Journal of Portfolio Management 40(5), 94-107. (arsiv:
docs/reference/papers/G_evaluation/2014_Bailey_DeflatedSharpe.pdf, s.9-10)

Eski `prediction/evaluator.py::compute_deflated_sharpe_ratio` uc yerden
yanlisti (Faz 9 Y1):
  (a) esik SR0'da denemeler arasi Sharpe varyansinin karekoku sqrt(V) yoktu,
      esik standart normal biriminde kalip YILLIKLANDIRILMIS SR ile
      kiyaslaniyordu;
  (b) sigma(SR) formulune yilliklandirilmis SR giriyordu (formul DONEM BASI SR
      ister);
  (c) (gamma4 - 1)/4 yerine FAZLA basiklik/4 kullaniliyordu.
Bu modulde tum buyukluker DONEM BASINA (orn. gunluk) tanimlidir; yilliklandirma
yalnizca raporlama icin `annualize_sharpe` ile yapilir.

Kabul olcutu (tests/test_eval_stats.py): makalenin s.10 ornegi
(N=100, V=1/2 yillik, T=1250, gamma3=-3, gamma4=10, SR=2.5 yillik, yilda 250
gozlem) -> SR0 = 0.1132 (gunluk), DSR = 0.9004.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class ReturnMoments:
    """Bir getiri serisinin DSR icin gereken momentleri (donem basina)."""

    sr: float          # donem basi Sharpe (ortalama / std, ddof=1)
    n_obs: int         # T
    skew: float        # gamma3
    kurt: float        # gamma4 — HAM basiklik (normal dagilimda 3)


def return_moments(returns, risk_free=None) -> ReturnMoments:
    """Getiri serisinden donem basi SR, carpiklik ve HAM basiklik.

    Args:
        returns: donem getirileri (basit getiri).
        risk_free: ayni uzunlukta donem basi risksiz getiri (A8) ya da skaler.
            None ise 0 (fazla getiri = getiri).
    """
    r = np.asarray(returns, dtype=float)
    if risk_free is not None:
        r = r - np.asarray(risk_free, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        raise ValueError(f"En az 3 gozlem gerekli, {len(r)} var")
    sd = r.std(ddof=1)
    if sd <= 0:
        raise ValueError("Getiri serisi sabit; Sharpe tanimsiz")
    return ReturnMoments(
        sr=float(r.mean() / sd),
        n_obs=int(len(r)),
        skew=float(stats.skew(r, bias=False)),
        # fisher=False -> ham (Pearson) basiklik; normal = 3. Eski kodun hatasi (c).
        kurt=float(stats.kurtosis(r, fisher=False, bias=False)),
    )


def annualize_sharpe(sr_per_period: float, periods_per_year: int = 252) -> float:
    """Raporlama icin: donem basi SR -> yillik SR (iid varsayimi)."""
    return float(sr_per_period * np.sqrt(periods_per_year))


def sharpe_std(sr: float, n_obs: int, skew: float, kurt: float) -> float:
    """SR tahmincisinin standart hatasi (Bailey & LdP 2014, denklem 2 paydasi).

    sigma(SR) = sqrt((1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1)); SR donem basi,
    g4 HAM basiklik.
    """
    inner = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if inner <= 0 or n_obs < 2:
        raise ValueError(f"sigma(SR) tanimsiz (ic terim={inner:.4g}, T={n_obs})")
    return float(np.sqrt(inner / (n_obs - 1)))


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float
) -> float:
    """PSR(SR*) = Phi[(SR - SR*) / sigma(SR)] — tum SR'ler donem basi."""
    return float(stats.norm.cdf((sr - sr_benchmark) / sharpe_std(sr, n_obs, skew, kurt)))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """N bagimsiz denemenin en buyuk SR'sinin beklenen degeri (H0: gercek SR=0).

    SR0 = sqrt(V) * ((1 - gamma) * Z^-1[1 - 1/N] + gamma * Z^-1[1 - 1/(N e)])

    Args:
        n_trials: N — denenen (bagimsiz) yapilandirma sayisi; deneme kaydindan
            (C2) gelir.
        var_trials: V — denemelerin DONEM BASI SR'lerinin varyansi.
    """
    if n_trials < 1:
        raise ValueError("n_trials >= 1 olmali")
    if n_trials == 1:
        return 0.0
    if var_trials is None or not np.isfinite(var_trials) or var_trials < 0:
        raise ValueError("var_trials (denemeler arasi SR varyansi) gerekli ve >= 0 olmali")
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trials) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    *,
    n_trials: int,
    var_trials: float,
    returns=None,
    moments: Optional[ReturnMoments] = None,
    risk_free=None,
) -> dict:
    """DSR = PSR(SR0), SR0 = expected_max_sharpe(N, V).

    Ya `returns` ya `moments` verilir. N=1 iken SR0=0 ve DSR = PSR(0) olur;
    bu "anlamli" demek DEGILDIR — tek denemede secim yanliligi duzeltmesi
    yoktur. Sonuc sozlugu bunu `n_trials` ile birlikte tasir.
    """
    if moments is None:
        if returns is None:
            raise ValueError("returns ya da moments verilmeli")
        moments = return_moments(returns, risk_free=risk_free)
    sr0 = expected_max_sharpe(n_trials, var_trials if n_trials > 1 else 0.0)
    dsr = probabilistic_sharpe_ratio(moments.sr, sr0, moments.n_obs, moments.skew, moments.kurt)
    return {
        'dsr': dsr,
        'sr': moments.sr,
        'sr_benchmark': sr0,
        'sigma_sr': sharpe_std(moments.sr, moments.n_obs, moments.skew, moments.kurt),
        'n_obs': moments.n_obs,
        'skew': moments.skew,
        'kurtosis': moments.kurt,
        'n_trials': int(n_trials),
        'var_trials': float(var_trials) if var_trials is not None else None,
        'units': 'per_period',
    }
