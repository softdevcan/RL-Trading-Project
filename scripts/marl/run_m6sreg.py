"""Faz 10 M6b kisa pilot duzenleyicisi (M6 tasarimi §6 ON KAYDI): S-reg acikken ogrenme bozuluyor mu?

Kollar: R0-N (none, tahminsiz, S-reg yok) ve SR-N (none, tahminsiz, S-reg acik: --extras-path data/marl/sreg_m6.npz).
Maliyetsiz, YENI tohumlar 12, 13, 14, F = 600.000, lr 5e-4. Okuma: analyze_m6sreg.py.

  venv-marl\\Scripts\\python scripts/marl/run_m6sreg.py --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m6sreg.py --jobs 2
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_m3a import ROOT  # noqa: E402
from run_m4a import FRAMES, PY, wait_gpu  # noqa: E402
from run_m4b import run_logged  # noqa: E402

os.chdir(ROOT)
SEEDS = (12, 13, 14)
COST_BPS = 0
SREG = os.path.join('data', 'marl', 'sreg_m6.npz')
ARMS = {'R0-N': [], 'SR-N': ['--extras-path', SREG]}


def train_cmd(arm: str, seed: int, run_dir: str, device: str, frames: int = FRAMES) -> list:
    return [PY, 'scripts/marl/train_m3.py', '--frames', str(frames), '--seed', str(seed), '--cost-bps', str(COST_BPS),
            '--comm', 'none', '--lr', '5e-4', '--train-device', device, '--parallel', '--save-dir', run_dir,
            '--out', os.path.join(run_dir, 'train.json'), *ARMS[arm]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m6sreg'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--frames', type=int, default=FRAMES, help='yalniz duman testi icin degistir (on kayit 600.000)')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='yalniz duman testi icin degistir')
    ap.add_argument('--gpu-busy-pct', type=int, default=50)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    if a.jobs > 1:
        a.gpu_busy_pct = max(a.gpu_busy_pct, 80)
    seeds = [int(s) for s in a.seeds.split(',')]
    if a.frames != FRAMES or tuple(seeds) != SEEDS:
        print(f'UYARI: on kayit disi yapilandirma (frames {a.frames}, tohum {seeds}): yalniz duman testi')
    if not os.path.exists(SREG):
        raise SystemExit(f'blok yok: {SREG} (python scripts/marl/build_sreg_m6.py --json ...)')
    tasks = [(s, arm) for s in seeds for arm in ARMS
             if not os.path.exists(os.path.join(a.root, arm, f's{s}', 'eval_val.npz'))]

    def one(task):
        s, arm = task
        d = os.path.join(a.root, arm, f's{s}')
        log = None if a.dry_run else os.path.join(d, 'run.log')
        if not a.dry_run:
            os.makedirs(d, exist_ok=True)
            wait_gpu(a)
        if not os.path.exists(os.path.join(d, 'policy.pt')):
            run_logged(train_cmd(arm, s, d, a.train_device, a.frames), a.dry_run, log)
        for window in ('train', 'val'):
            run_logged([PY, 'scripts/marl/eval_m3.py', '--run-dir', d, '--window', window,
                        '--out', os.path.join(d, f'eval_{window}.npz')], a.dry_run, log)

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for f in [ex.submit(one, t) for t in tasks]:
            f.result()


if __name__ == '__main__':
    main()
