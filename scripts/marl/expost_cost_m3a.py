"""Faz 10 M3b tasarim girdisi (kesifsel, kapi degil): M3a'nin MALIYETSIZ egitilmis politikalarina 20 bps'nin sonradan etkisi.

Maliyet agirlik yolunu degistirmez (`accounting.step_fractional`: w_next maliyetten bagimsiz), yani ayni politika ayni
eylemleri uretir ve net seri kesin hesaplanir: R_net = (1 + R)(1 - c * devir) - 1, X_net = R_net - rf, c = 0,001.
Bu, maliyetle YENIDEN EGITILMIS politikanin sonucu DEGILDIR; M3b asama 1 onu olcer.

  python scripts/marl/expost_cost_m3a.py --out docs/development/measurements/phase10-m3/m3a_expost_cost.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
ANN = np.sqrt(252.0)
COST_ONE_WAY = 20 / 2 / 1e4
SEEDS = (0, 1, 2)


def sr(x) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def net_x(d) -> np.ndarray:
    R, X, tau = d['policy_R'], d['policy_X'], d['policy_turnover']
    return (1.0 + R) * (1.0 - COST_ONE_WAY * tau) - 1.0 - (R - X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m3'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    out = {'note': 'kesifsel; sabit politika, maliyetle yeniden egitim degil; kapi degil', 'cost_bps_roundtrip': 20,
           'provenance': {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
                          'script_sha256': sha}}
    for window in ('train', 'val'):
        rec = {}
        for arm in ('A1', 'S1'):
            runs = [np.load(os.path.join(a.root, arm, f's{s}', f'eval_{window}.npz'), allow_pickle=False) for s in SEEDS]
            rec[arm] = {'sharpe_gross': [sr(d['policy_X']) for d in runs], 'sharpe_net': [sr(net_x(d)) for d in runs],
                        'turnover_mean': float(np.mean([d['policy_turnover'].mean() for d in runs]))}
        delta = [x - y for x, y in zip(rec['A1']['sharpe_net'], rec['S1']['sharpe_net'])]
        out[window] = {**rec, 'delta_net_per_seed': delta, 'D_net': float(np.mean(delta))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps({w: {'D_net': out[w]['D_net'], 'delta_net_per_seed': out[w]['delta_net_per_seed']}
                      for w in ('train', 'val')}, indent=1))


if __name__ == '__main__':
    main()
