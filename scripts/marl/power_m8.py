"""Faz 10 M8 güç benzetimi (hakem-rl M8 #B2, hakem-finans #3): ön kayıt karar kuralının (m8_stats.label:
p_signflip <= 0,05 VE p_bootstrap <= 0,05) GERÇEK gücü. Ana venv, saf numpy. Performans okunmaz; girdi yalnız
boyut ölçümünün sigma'sı (`measurements/phase10-m8/noise_size.json`).

Model: SR_a(s, f) = mu_f + delta_f + e_a, SR_b(s, f) = mu_f + e_b; e ~ N(0, sigma_kol^2), eşler arası korelasyon rho.
Kat etkisi mu_f iki kolda ortak olduğu için farkta düşer. delta_f ~ N(delta, tau^2): katlar arası etki heterojenliği.
Senaryolar: sigma 0,155 (ölçülen) ve 0,226 (%80 üst sınır); SARL kolunda sigma x 1,5; rho 0 (tutucu) ve 0,5; tau 0 / 0,1.
MDE = gücün %80'i geçtiği en küçük delta (ızgarada doğrusal ara değer).

  python scripts/marl/power_m8.py --out docs/development/measurements/phase10-m8/power_m8.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_stats as ST  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
NOISE = os.path.join(ROOT, 'docs', 'development', 'measurements', 'phase10-m8', 'noise_size.json')
F, K = 6, 12
GRID = np.round(np.arange(0.0, 0.2001, 0.02), 3)


def power(delta, sa, sb, rho, tau, nsim, R, B, rng) -> float:
    cov = np.array([[sa * sa, rho * sa * sb], [rho * sa * sb, sb * sb]])
    hits = 0
    for _ in range(nsim):
        e = rng.multivariate_normal([0, 0], cov, size=(F, K))
        df = rng.normal(delta, tau, size=(F, 1)) if tau > 0 else np.full((F, 1), delta)
        d = df + e[..., 0] - e[..., 1]
        hits += ST.label(ST.delta_bar(d), ST.p_signflip(d, R=R, seed=int(rng.integers(1 << 30))),
                         ST.bootstrap(d, B=B, seed=int(rng.integers(1 << 30)))['p'], 'a', 'b').startswith('FARK_VAR')
    return hits / nsim


def mde(grid, pw):
    for i in range(1, len(grid)):
        if pw[i] >= 0.8 > pw[i - 1]:
            return float(grid[i - 1] + (0.8 - pw[i - 1]) / (pw[i] - pw[i - 1]) * (grid[i] - grid[i - 1]))
    return None if pw[-1] < 0.8 else float(grid[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsim', type=int, default=400)
    ap.add_argument('--R', type=int, default=2000, help='benzetimde işaret-çevirme sayısı (okumada 10.000)')
    ap.add_argument('--B', type=int, default=2000, help='benzetimde bootstrap sayısı (okumada 10.000)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(NOISE, encoding='utf-8') as fh:
        s = float(json.load(fh)['sigma_seed'])
    s_up = s * np.sqrt(5 / 2.342541)        # chi2(5) %20 kantili; tek yönlü %80 üst sınır
    scen = {'olculen': (s, s, 0.0, 0.0), 'olculen_rho05': (s, s, 0.5, 0.0), 'ust_sinir': (s_up, s_up, 0.0, 0.0),
            'sarl_1.5x': (s, 1.5 * s, 0.0, 0.0), 'ust_sinir_sarl_1.5x': (s_up, 1.5 * s_up, 0.0, 0.0),
            'olculen_tau0.1': (s, s, 0.0, 0.1)}
    rng = np.random.default_rng(20261121)
    res = {'model': __doc__.split('\n\n')[1], 'F': F, 'k': K, 'nsim': a.nsim, 'R': a.R, 'B': a.B,
           'sigma_seed': s, 'sigma_upper80': float(s_up), 'grid': GRID.tolist(), 'scenarios': {}}
    for name, (sa, sb, rho, tau) in scen.items():
        pw = [power(dl, sa, sb, rho, tau, a.nsim, a.R, a.B, rng) for dl in GRID]
        res['scenarios'][name] = {'sigma_a': sa, 'sigma_b': sb, 'rho': rho, 'tau': tau, 'power': pw,
                                  'size_at_0': pw[0], 'mde80': mde(GRID, pw)}
        print(f"{name:22s} boyut(0) {pw[0]:.3f}  MDE80 {res['scenarios'][name]['mde80']}", flush=True)
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        res['provenance'] = {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
                             'script_sha256': hashlib.sha256(fh.read()).hexdigest()}
    if a.out:
        with open(os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
