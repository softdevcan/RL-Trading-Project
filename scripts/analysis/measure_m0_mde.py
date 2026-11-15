"""
Milestone 0' MDE guc simulasyonu (hakem-2 kapanis #4; M8 "dogrulayici kanit K20 guc analiziyle"). EGITIM YOK.

Eslesik iki stratejinin gunluk fazla getirisi: iki degiskenli normal, gunluk oynaklik SIGMA, korelasyon rho.
H0: SR1 = SR2. Test: A3 `evaluation.sharpe_tests.sharpe_diff_hac` (prewhitened QS HAC), iki tarafli alfa 0,05.
MDE = guc >= 0,80 olan en kucuk yillik Sharpe farki (izgarada). Tohumlar piyasa gurultusune karsi bagimsiz
gozlem EKLEMEZ (ayni gunler): cikarim birimi tek eslesik seri; tohum ortalamasi yalniz politika gurultusunu
azaltir ve rho'yu yukseltir. Analitik yaklasiklik: (z_{1-a/2} + z_g) * sqrt(2(1-rho)/T) * sqrt(252).

    python scripts/analysis/measure_m0_mde.py
"""

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join('docs', 'development', 'measurements', 'phase9-m0', 'mde.json')
T_GRID = {'2021 (tek kat, dogrulayici)': 248, '2023-2026 (on eleme, kesifsel)': 1000, 'tum 6 kat': 1460}
RHO = [0.5, 0.8, 0.9, 0.95]
SIGMA = 0.015          # tasarim varsayimi (BIST hisse portfoyu gunluk oynaklik mertebesi)
BASE_SR = 0.5          # yillik, ikinci stratejinin Sharpe'i
DELTAS = np.round(np.arange(0.1, 3.01, 0.1), 2)   # yillik Sharpe farki izgarasi
N_REP = 400
ALPHA, POWER = 0.05, 0.80


def power_at(T, rho, d_ann, rng):
    from evaluation.sharpe_tests import sharpe_diff_hac
    mu2 = BASE_SR / np.sqrt(252) * SIGMA
    mu1 = (BASE_SR + d_ann) / np.sqrt(252) * SIGMA
    cov = SIGMA ** 2 * np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov)
    rej = 0
    for _ in range(N_REP):
        x = rng.standard_normal((T, 2)) @ L.T + np.array([mu1, mu2])
        rej += sharpe_diff_hac(x[:, 0], x[:, 1])['p_value'] < ALPHA
    return rej / N_REP


def main():
    from scipy.stats import norm
    rng = np.random.default_rng(20261115)
    res = {}
    for label, T in T_GRID.items():
        res[label] = {'T': T, 'rho': {}}
        for rho in RHO:
            analytic = float((norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)) * np.sqrt(2 * (1 - rho) / T) * np.sqrt(252))
            mde, curve = None, {}
            for d in DELTAS:
                if d < 0.6 * analytic:
                    continue
                p = power_at(T, rho, float(d), rng)
                curve[str(d)] = p
                if p >= POWER:
                    mde = float(d)
                    break
            res[label]['rho'][str(rho)] = {'mde_annual_sharpe': mde, 'analytic': round(analytic, 3),
                                           'power_curve': curve}
            print(label, 'rho', rho, 'MDE', mde, 'analitik', round(analytic, 3), flush=True)
    out = {'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
           'test': 'sharpe_diff_hac (prewhitened QS), iki tarafli', 'alpha': ALPHA, 'power': POWER,
           'n_rep': N_REP, 'sigma_daily': SIGMA, 'base_sharpe_annual': BASE_SR,
           'assumptions': 'iki degiskenli normal, iid (otokorelasyon yok); rho = eslesik kollarin gunluk getiri '
                          'korelasyonu (olculmedi; Milestone 0 sonuclarindan raporlanir)',
           'grid_step': 0.1, 'results': res}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(out, fh, indent=1)
    print('->', OUT)


if __name__ == '__main__':
    main()
