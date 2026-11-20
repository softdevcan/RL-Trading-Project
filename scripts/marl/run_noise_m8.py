"""Faz 10 M8 gurultu tabani BOYUT OLCUMU duzenleyicisi (M8 tasarimi §3.1 ON KAYIT v1.0). venv-marl.

Referans MAPPO (M8 tasarimi §1), panel_m8, sozde kat '2020' (egitim ..2018-12-31, dogrulama 2019, orneklem disi 2020;
2021+ OKUNMAZ), F = 600.000, 10 kontrol noktasi, toplu toplayici (n_envs 10), 20 bps, tohum 101-106. Her tohum:
train_m3.py (--checkpoints 10) -> eval_fold_m8.py (dogrulamada secim, orneklem disi yil secilenle BIR KEZ).
Okuma: analyze_noise_m8.py (ana venv). Cikti dizininde degerlendirme varsa tohum atlanir (yeniden baslatilabilir).
GPU dolu ise (oyun vb.) her koşudan once bosalana kadar bekler (run_m4a.wait_gpu).

  venv-marl\\Scripts\\python scripts/marl/run_noise_m8.py --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_noise_m8.py --jobs 3
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_m3a import ROOT  # noqa: E402
from run_m4a import PY, wait_gpu  # noqa: E402
from run_m4b import run_logged  # noqa: E402

os.chdir(ROOT)
SEEDS = (101, 102, 103, 104, 105, 106)
FRAMES = 600_000
CHECKPOINTS = 10
PANEL = os.path.join('data', 'marl', 'panel_m8.npz')
FOLD = {'name': 'psv-2020', 'train_end': '2018-12-31', 'val_start': '2019-01-01', 'test_start': '2020-01-01',
        'test_end': '2021-01-01'}


def train_cmd(seed: int, run_dir: str, device: str, frames: int) -> list:
    return [PY, 'scripts/marl/train_m3.py', '--panel', PANEL, '--train-end', FOLD['train_end'],
            '--checkpoints', str(CHECKPOINTS), '--frames', str(frames), '--seed', str(seed), '--cost-bps', '20',
            '--lr', '5e-4', '--batched-env', '--train-device', device, '--save-dir', run_dir,
            '--out', os.path.join(run_dir, 'train.json')]


def eval_cmd(run_dir: str) -> list:
    return [PY, 'scripts/marl/eval_fold_m8.py', '--run-dir', run_dir, '--fold', FOLD['name'],
            '--val-start', FOLD['val_start'], '--test-start', FOLD['test_start'], '--test-end', FOLD['test_end'],
            '--out-dir', os.path.join(run_dir, 'eval')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m8_noise'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--frames', type=int, default=FRAMES, help='yalniz duman testi icin degistir (on kayit 600.000)')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='yalniz duman testi icin degistir')
    ap.add_argument('--gpu-busy-pct', type=int, default=50)
    ap.add_argument('--max-wait', type=int, default=12 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--jobs', type=int, default=3, help='es zamanli kosu (plan §3.2.8: GPU 8 GB -> 3-4)')
    a = ap.parse_args()
    if a.jobs > 1:
        a.gpu_busy_pct = max(a.gpu_busy_pct, 80)
    seeds = [int(s) for s in a.seeds.split(',')]
    if a.frames != FRAMES or tuple(seeds) != SEEDS:
        print(f'UYARI: on kayit disi yapilandirma (frames {a.frames}, tohum {seeds}): yalniz duman testi')
    if not os.path.exists(PANEL):
        raise SystemExit(f'panel yok: {PANEL} (prepare_panel_m2.py --data-fix m8-v1)')
    tasks = [s for s in seeds if not os.path.exists(os.path.join(a.root, f's{s}', 'eval', 'fold_eval.json'))]

    def one(s):
        d = os.path.join(a.root, f's{s}')
        log = None if a.dry_run else os.path.join(d, 'run.log')
        if not a.dry_run:
            os.makedirs(d, exist_ok=True)
            wait_gpu(a)
        if not os.path.exists(os.path.join(d, 'policy.pt')):
            run_logged(train_cmd(s, d, a.train_device, a.frames), a.dry_run, log)
        run_logged(eval_cmd(d), a.dry_run, log)

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for f in [ex.submit(one, s) for s in tasks]:
            f.result()


if __name__ == '__main__':
    main()
