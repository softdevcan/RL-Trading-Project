"""Faz 10 M7a: karsi-olgusal mesaj maskeleme (venv-marl; yalniz degerlendirme, egitim yok). BETIMSEL tani, karar yok.

Kayitli iletisimli politikalarda (gat/attn) degerlendirme zamaninda gondericilerin mesaji kapatilir
(`CommPolicy.set_sender_mask`) ve deterministik tam pencere degerlendirmesindeki portfoy Sharpe'i (fazla getiri, yillik)
maskesiz haliyle karsilastirilir:
  - base     : maske yok. Kayitli `eval_<pencere>.npz` serisiyle bit-es olmalidir (dogru politika + dogru kurulum kaniti).
  - all      : butun gondericiler kapali (alici yalniz kendi kodunu gorur).
  - sector:<g>: yalniz o sektor grubunun hisseleri kapali (E2 KAP gruplari; `load_sector_groups`).
Tez iddiasi degildir: dikkat ve maskeleme nedensel bir aciklama sayilmaz (Jain & Wallace 2019, M6 tasarimi §4).

    venv-marl\\Scripts\\python scripts/marl/mask_m7.py --runs results/marl_m4b/G-X/s3 --window train --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
os.chdir(ROOT)

import numpy as np  # noqa: E402

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def _git(*args) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_one(run_dir: str, window: str, arrays, groups_by_symbol) -> dict:
    from eval_m3 import window_rows
    from marl import obs_spec as OS
    from marl.comm_models import CommPolicy
    from marl.m3_runner import deterministic_rollout, load_policy

    s, e = window_rows(arrays['dates'], window, OS.WARMUP_ROWS)
    exp, meta = load_policy(run_dir)
    cfg = {k: v for k, v in meta['task_cfg'].items() if k != 'use_state'}
    cfg.update(mode='eval', start_row=s, end_row=e, norm_stats_path=os.path.join(run_dir, 'norm_stats.json'))
    comm = [m for m in exp.policy.modules() if isinstance(m, CommPolicy)]
    if len(comm) != 1:
        raise ValueError(f"{run_dir}: iletisimli politika degil (CommPolicy {len(comm)})")
    comm = comm[0]
    symbols = [str(x) for x in arrays['symbols']]
    groups: dict = {}
    for i, sym in enumerate(symbols):
        groups.setdefault(groups_by_symbol[sym], []).append(i)
    stored = np.load(os.path.join(run_dir, f'eval_{window}.npz'), allow_pickle=True)
    res = {'run': run_dir.replace('\\', '/'), 'kind': comm.kind, 'conf_mode': comm.conf_mode, 'window': window,
           'rows': [int(s), int(e)], 'policy_sha256': meta.get('policy_sha256'), 'variants': {}}
    t0 = time.time()
    for name, senders in [('base', None), ('all', list(range(len(symbols))))] + \
            [(f'sector:{g}', idx) for g, idx in sorted(groups.items())]:
        comm.set_sender_mask(senders)
        ser = deterministic_rollout(exp.policy, cfg)
        comm.set_sender_mask(None)
        v = {'sharpe_X': sharpe(ser['X']), 'turnover_mean': float(np.mean(ser['turnover'])),
             'cash_weight_mean': float(np.mean(ser['cash_weight'])), 'n_senders': 0 if senders is None else len(senders)}
        if name == 'base':
            v['bit_equal_stored_eval'] = bool(np.array_equal(ser['X'], stored['policy_X']))
            base_sr = v['sharpe_X']
        else:
            v['delta_sharpe_X'] = v['sharpe_X'] - base_sr
        res['variants'][name] = v
    res['seconds'] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--window', choices=('train', 'val'), default='train')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from data.bist30_symbols import load_sector_groups
    from marl import panel as PN

    arrays = PN.load_npz(PANEL)
    groups = load_sector_groups()
    out = {'note': 'M7a karsi-olgusal mesaj maskeleme (BETIMSEL tani; egitim yok, karar yok)', 'window': a.window,
           'runs': []}
    for r in a.runs:
        res = run_one(r, a.window, arrays, groups)
        out['runs'].append(res)
        v = res['variants']
        print(res['run'], 'base', round(v['base']['sharpe_X'], 3), 'bit-es', v['base']['bit_equal_stored_eval'],
              'all', round(v['all']['delta_sharpe_X'], 3), 'sn', res['seconds'])
    out['provenance'] = {'commit': _git('rev-parse', 'HEAD'), 'git_status_porcelain': _git('status', '--porcelain'),
                         'script_sha256': _sha(os.path.abspath(__file__)),
                         'comm_models_sha256': _sha(os.path.join(ROOT, 'marl', 'comm_models.py')),
                         'panel_sha256': arrays['sha256']}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
