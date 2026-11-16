r"""Faz 10 M2 sentetik olcumleri (gercek panel DEGIL): maliyet yaklasimi farki + DSR isinma payi.

venv-marl\Scripts\python scripts/marl/measure_m2_synthetic.py --json docs/development/measurements/phase10-m2/synthetic_checks.json
"""
import argparse, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
from _marl_fixtures import synth_arrays  # noqa: E402
from marl.env import MarlPortfolioEnv  # noqa: E402
from marl import obs_spec as OS  # noqa: E402


def _cost_gap(bps):
    arr = synth_arrays(T=260, N=4, seed=0, with_rf=True)
    kw = dict(start_row=180, end_row=259, norm_stats=None, mode='eval', cost_bps_roundtrip=bps)
    tr = MarlPortfolioEnv(arr, OS.WARMUP_ROWS, 180, 'fit')
    kw['norm_stats'] = tr.norm_stats
    ef = MarlPortfolioEnv(arr, **kw)
    ei = MarlPortfolioEnv(arr, integer_shares=True, initial_capital=1e9, **kw)
    rng = np.random.default_rng(7)
    ef.reset(seed=0); ei.reset(seed=0); diffs = []
    while ef.agents:
        a = {ag: rng.uniform(-ef.bound, ef.bound, 1).astype(np.float32) for ag in ef.agents}
        ef.step(a); ei.step(a)
        diffs.append(abs(ef.last_info['R'] - ei.last_info['R']))
    return {'bps_roundtrip': bps, 'steps': len(diffs), 'max_abs_R_diff': float(max(diffs)),
            'mean_abs_R_diff': float(np.mean(diffs))}


def _dsr_nonzero(ep, eta):
    arr = synth_arrays(T=420, N=4, seed=0, with_rf=True)
    try:
        e = MarlPortfolioEnv(arr, 60, 419, 'fit', reward_term='differential', episode_len=ep, dsr_eta=eta)
    except ValueError as exc:
        return {'episode_len': ep, 'dsr_eta': eta, 'error': str(exc)}
    e.reset(seed=1); rng = np.random.default_rng(0); d = []
    while e.agents:
        e.step({ag: rng.uniform(-2, 2, 1) for ag in e.agents}); d.append(e.last_info['dsr_raw'])
    return {'episode_len': ep, 'dsr_eta': eta, 'nonzero_frac': float(np.mean(np.array(d) != 0))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--json')
    a = ap.parse_args()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {'note': 'SENTETIK panel (tests/_marl_fixtures.synth_arrays); gercek panel olcumu degil',
           'commit': commit,
           'cost_gap_integer_vs_fractional': [_cost_gap(0.0), _cost_gap(20.0)],
           'dsr_warmup': [_dsr_nonzero(ep, eta) for ep, eta in
                          ((100, 1 / 252), (252, 1 / 252), (300, 1 / 252), (252, 1 / 63), (100, 1 / 20))]}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
