r"""Faz 10 M2 sentetik olcumleri (gercek panel DEGIL): maliyet yaklasimi farki, DSR isinma payi,
odul terimlerinin olcegi (hakem-rl M2 #1).

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
        e = MarlPortfolioEnv(arr, 60, 419, 'fit', reward_term='differential', episode_len=ep, dsr_eta=eta,
                             alpha=0.0)
    except ValueError as exc:
        return {'episode_len': ep, 'dsr_eta': eta, 'error': str(exc)}
    e.reset(seed=1); rng = np.random.default_rng(0); d = []
    while e.agents:
        e.step({ag: rng.uniform(-2, 2, 1) for ag in e.agents}); d.append(e.last_info['dsr_raw'])
    return {'episode_len': ep, 'dsr_eta': eta, 'nonzero_frac': float(np.mean(np.array(d) != 0))}


def _term_scale(term, eta):
    """std(100(1-a)G) / std(100 a k_i), a = 0,25 (G != 0 adimlarda). Odulun KENDISI degil, iki terimi."""
    import marl.env as ME
    arr = synth_arrays(T=420, N=4, seed=0, with_rf=True)
    # alpha 0 ile kurulur (differential yalniz alpha=0); terimler yakalanip a = 0,25 ile tartilir
    e = MarlPortfolioEnv(arr, 60, 419, 'fit', reward_term=term, episode_len=252, dsr_eta=eta, alpha=0.0)
    K, G = [], []
    orig = ME.agent_rewards
    ME.agent_rewards = lambda k, g, al, sc: (K.append(np.array(k)), G.append(g), orig(k, g, al, sc))[2]
    try:
        e.reset(seed=1); rng = np.random.default_rng(0)
        while e.agents:
            e.step({ag: rng.uniform(-2, 2, 1) for ag in e.agents})
    finally:
        ME.agent_rewards = orig
    K, G = np.array(K), np.array(G)
    nz = G != 0
    a = 0.25
    sk, sg = float(np.std(100 * a * K[nz])), float(np.std(100 * (1 - a) * G[nz]))
    return {'term': term, 'dsr_eta': eta, 'alpha': a, 'N': 4, 'steps_used': int(nz.sum()),
            'std_individual_term': sk, 'std_portfolio_term': sg, 'ratio_portfolio_to_individual': sg / sk}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--json')
    a = ap.parse_args()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {'note': 'SENTETIK panel (tests/_marl_fixtures.synth_arrays); gercek panel olcumu degil',
           'commit': commit,
           'cost_gap_integer_vs_fractional': [_cost_gap(0.0), _cost_gap(20.0)],
           'dsr_warmup': [_dsr_nonzero(ep, eta) for ep, eta in
                          ((100, 1 / 252), (252, 1 / 252), (300, 1 / 252), (252, 1 / 63), (100, 1 / 20))],
           'reward_term_scale': [_term_scale('excess', 1 / 252), _term_scale('differential', 1 / 63)]}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
