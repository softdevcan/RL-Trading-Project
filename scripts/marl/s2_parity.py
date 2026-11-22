"""Faz 11 S2: politika aktarimi parite olcumu (ANA venv). Kabul: ayni gozlemde logit farki <= 1e-6, 72/72 politika
(plan phase-11-integration-plan.md §3 S2).

Her politika icin `marl.frozen_policy.FrozenPolicy` (BenchMARL'siz) ile venv-marl referansi
(`scripts/marl/s2_parity_ref.py` -> results/marl_s2/ref/<kol>/<tohum>.npz) karsilastirilir:
  - loc (tanh oncesi) ve eylem (logit, [-B, B]) en buyuk mutlak farki; 2026 test penceresi + rastgele gozlem
  - referans penceresinin gunluk R'si M8 test kaydiyla (results/marl_m8/2026/<kol>/<tohum>/eval/test_argmax.npz)
    birebir ayni mi (referansin M8 degerlendirmesinin kendisi oldugunun kaniti)
  - arsiv dogrulamasi (verify_archive, on kayit listesi) once kosar.

    python scripts/marl/s2_parity.py [--arm M-none --seed s201] [--out docs/development/measurements/phase11-s2/s2_parity.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from marl import frozen as FZ  # noqa: E402
from marl.frozen_policy import FrozenPolicy  # noqa: E402

ARCHIVE = os.path.join(FZ.FROZEN_ROOT, 'm8-2026')
REF = os.path.join('results', 'marl_s2', 'ref')
M8 = os.path.join('results', 'marl_m8', '2026')
READING = os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'm8_reading.json')
TOL = 1e-6


def compare(p: dict) -> dict:
    d = os.path.join(ARCHIVE, p['arm'], p['seed'])
    ref = np.load(os.path.join(REF, p['arm'], f"{p['seed']}.npz"))
    if str(ref['params_sha']) != p['params_sha']:
        raise ValueError(f"{p['arm']}/{p['seed']}: referans baska politikadan")
    fp = FrozenPolicy(d, ckpt=p['ckpt'])
    N, dd = fp.n_agents, fp.obs_dim
    out = {'arm': p['arm'], 'seed': p['seed'], 'kind': fp.kind}
    for part in ('win', 'rand'):
        obs = ref[f'{part}_obs'].reshape(-1, N, dd)                  # SARL: [T, 1, N*d] -> [T, N, d]
        loc_ref = ref[f'{part}_loc'].reshape(obs.shape[0], N)
        act_ref = ref[f'{part}_action'].reshape(obs.shape[0], N)
        # Referansla AYNI cagri bicimi: pencere gun gun (tek gozlem; canli gunluk is de boyle cagirir), rastgele toplu.
        # Parti boyutu float32 toplama sirasini degistirir (S-agr N*d = 1320 girdi); capraz bicim farki bilgi olarak.
        if part == 'win':
            loc = np.stack([fp.loc(o) for o in obs])
            act = np.stack([fp.act(o) for o in obs])
            other = fp.act(obs)
        else:
            loc, act = fp.loc(obs), fp.act(obs)
            other = np.stack([fp.act(o) for o in obs])
        out[part] = {'n_obs': int(obs.shape[0]), 'call': 'tek' if part == 'win' else 'toplu',
                     'max_abs_loc': float(np.max(np.abs(loc - loc_ref))),
                     'max_abs_action': float(np.max(np.abs(act - act_ref))),
                     'action_bitwise_equal': bool(np.array_equal(act, act_ref)),
                     'cross_call_max_abs_action': float(np.max(np.abs(other - act_ref))),
                     'saturated_frac': float(np.mean(np.abs(act_ref) >= 0.999 * float(fp.bound)))}
    m8 = np.load(os.path.join(M8, p['arm'], p['seed'], 'eval', 'test_argmax.npz'))
    out['ref_R_equals_m8_test'] = bool(np.array_equal(ref['win_R'], m8['policy_R']))
    out['pass'] = (out['ref_R_equals_m8_test'] and
                   max(out[q][k] for q in ('win', 'rand') for k in ('max_abs_loc', 'max_abs_action')) <= TOL)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default=None)
    ap.add_argument('--seed', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(READING, encoding='utf-8') as fh:
        fwd = json.load(fh)['forward_policies_2026']
    ver = FZ.verify_archive(ARCHIVE, forward=fwd)
    with open(os.path.join(ARCHIVE, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    todo = [p for p in man['policies'] if (a.arm in (None, p['arm'])) and (a.seed in (None, p['seed']))]
    rows = []
    for p in todo:
        r = compare(p)
        rows.append(r)
        print(f"{r['arm']:7s} {r['seed']} {r['kind']:9s} pencere loc {r['win']['max_abs_loc']:.2e} eylem "
              f"{r['win']['max_abs_action']:.2e} | rastgele loc {r['rand']['max_abs_loc']:.2e} eylem "
              f"{r['rand']['max_abs_action']:.2e} | R=M8 {r['ref_R_equals_m8_test']} | {'GECTI' if r['pass'] else 'KALDI'}")
    worst = lambda q, k: max(r[q][k] for r in rows)   # noqa: E731
    summary = {'n_policies': len(rows), 'n_pass': sum(r['pass'] for r in rows), 'tolerance': TOL,
               'max_abs_loc': max(worst('win', 'max_abs_loc'), worst('rand', 'max_abs_loc')),
               'max_abs_action': max(worst('win', 'max_abs_action'), worst('rand', 'max_abs_action')),
               'n_action_bitwise_equal_both': sum(r['win']['action_bitwise_equal'] and r['rand']['action_bitwise_equal']
                                                  for r in rows),
               'n_ref_R_equals_m8_test': sum(r['ref_R_equals_m8_test'] for r in rows),
               'cross_call_max_abs_action': max(worst('win', 'cross_call_max_abs_action'),
                                                worst('rand', 'cross_call_max_abs_action')),
               'by_kind': {k: sum(r['kind'] == k for r in rows) for k in sorted({r['kind'] for r in rows})}}
    print('ozet:', summary)
    if a.out:
        import subprocess
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'plan': 'phase-11-integration-plan.md §3 S2', 'archive': 'm8-2026', 'written':
                       datetime.now().isoformat(timespec='seconds'), 'commit': head, 'archive_verify': ver,
                       'reference': 'scripts/marl/s2_parity_ref.py (venv-marl, BenchMARL load_policy, DETERMINISTIC)',
                       'random_obs': 'N(0, 3^2), tohum 20261122, 256 gozlem', 'summary': summary, 'policies': rows},
                      fh, indent=1, ensure_ascii=False)
    sys.exit(0 if summary['n_pass'] == summary['n_policies'] else 1)


if __name__ == '__main__':
    main()
