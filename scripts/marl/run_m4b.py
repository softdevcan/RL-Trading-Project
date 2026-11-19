"""Faz 10 M4b duzenleyicisi (M4 tasarimi §6 ON KAYDI, PILOT): gat <-> self (mimari-es iletisimsiz), komsu sinyali
(X) ve karistirilmis komsu sinyali (S), maliyetsiz, YENI tohumlar 3, 4, 5 (uretec = tohum), F = 600.000, lr 5e-4.

Kollar: G-X, Z-X, G-S, Z-S (Z = `--comm self`). Bloklar `pred_m4nb_g{3,4,5}[_shuf].npz` (build_pred_m4.py,
neighbor_map_m4b.json). Her egitimden once GPU bos mu bakar. Biten kosular atlanir. Okuma: analyze_m4b.py.

  venv-marl\\Scripts\\python scripts/marl/run_m4b.py --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m4b.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_m3a import ROOT  # noqa: E402
from run_m4a import COST_BPS, FRAMES, PY, run, wait_gpu  # noqa: E402

os.chdir(ROOT)
SEEDS = (3, 4, 5)
GEN = {3: 3, 4: 4, 5: 5}
ARMS = {'G-X': ('gat', ''), 'Z-X': ('self', ''), 'G-S': ('gat', '_shuf'), 'Z-S': ('self', '_shuf')}


def train_cmd(arm: str, seed: int, run_dir: str, device: str) -> list:
    comm, suffix = ARMS[arm]
    pred = os.path.join('data', 'marl', f'pred_m4nb_g{GEN[seed]}{suffix}.npz')
    return [PY, 'scripts/marl/train_m3.py', '--frames', str(FRAMES), '--seed', str(seed), '--cost-bps', str(COST_BPS),
            '--comm', comm, '--lr', '5e-4', '--train-device', device, '--parallel', '--save-dir', run_dir,
            '--out', os.path.join(run_dir, 'train.json'), '--pred-path', pred]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m4b'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--gpu-busy-pct', type=int, default=50)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    for s in SEEDS:
        for arm in ARMS:
            d = os.path.join(a.root, arm, f's{s}')
            if os.path.exists(os.path.join(d, 'eval_val.npz')):
                continue
            if not a.dry_run:
                wait_gpu(a)
            if not os.path.exists(os.path.join(d, 'policy.pt')):
                run(train_cmd(arm, s, d, a.train_device), a.dry_run)
            for window in ('train', 'val'):
                run([PY, 'scripts/marl/eval_m3.py', '--run-dir', d, '--window', window,
                     '--out', os.path.join(d, f'eval_{window}.npz')], a.dry_run)


if __name__ == '__main__':
    main()
