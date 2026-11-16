"""Faz 10 M3: kayitli politikanin deterministik degerlendirmesi + ayni muhasebeyle baseline serileri (venv-marl).

Yalniz GUNLUK SERI yazar (.npz: tarih, R, X, devir, nakit agirligi); Sharpe/test/okuma M3 on kaydindaki kuralla
`scripts/marl/analyze_m3.py` (ana venv). Pencereler: train = WARMUP..2018-12-31, val = 2019..2020-12-31.
2021+ yok (panel 2020-12-31'de biter; marj takvimi sonrasi hata verir).

  venv-marl\\Scripts\\python scripts/marl/eval_m3.py --run-dir results/marl_m3/<kol>/s1 --window train --out <npz>
  venv-marl\\Scripts\\python scripts/marl/eval_m3.py --baselines --window train --cost-bps 20 --out <npz>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
TRAIN_END, VAL_END = '2018-12-31', '2020-12-31'


def window_rows(dates, window: str, warmup: int) -> tuple[int, int]:
    import numpy as np
    te = int(np.flatnonzero(dates <= TRAIN_END)[-1])
    if window == 'train':
        return warmup, te
    if window == 'val':
        return te, int(np.flatnonzero(dates <= VAL_END)[-1])
    raise ValueError(window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--baselines', action='store_true')
    ap.add_argument('--window', choices=('train', 'val'), required=True)
    ap.add_argument('--cost-bps', type=float, default=20.0, help='--baselines icin')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import numpy as np

    from marl import obs_spec as OS
    from marl import panel as PN
    from marl.m3_runner import deterministic_rollout, load_policy, rule_series

    arrays = PN.load_npz(PANEL)
    s, e = window_rows(arrays['dates'], a.window, OS.WARMUP_ROWS)
    out = {'window': a.window, 'rows': [s, e], 'panel_sha256': arrays['sha256']}
    if a.baselines:
        for kind in ('ew', 'bh', 'cash'):
            ser = rule_series(arrays, s, e, kind, a.cost_bps)
            out.update({f'{kind}_{k}': np.asarray(v) for k, v in ser.items()})
        out['cost_bps'] = a.cost_bps
    else:
        exp, meta = load_policy(a.run_dir)
        cfg = {k: v for k, v in meta['task_cfg'].items() if k != 'use_state'}
        cfg.update(mode='eval', start_row=s, end_row=e,
                   norm_stats_path=os.path.join(a.run_dir, 'norm_stats.json'))
        ser = deterministic_rollout(exp.policy, cfg)
        out.update({f'policy_{k}': np.asarray(v) for k, v in ser.items()})
        out['run_meta'] = json.dumps({k: meta[k] for k in ('build', 'policy_sha256', 'task_cfg')}, default=str)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v)) for k, v in out.items()})
    print(f"{a.out}: pencere {a.window} satir {s}..{e}")


if __name__ == '__main__':
    main()
