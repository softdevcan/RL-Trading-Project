"""Faz 10 M3a duzenleyicisi (M3 tasarim §5): 4 kol x 5 tohum egitim + deterministik degerlendirme + baseline.

Her egitimden once GPU bos mu bakar (`nvidia-smi` kullanim %); doluysa (or. oyun) bekler, `--max-wait` dolunca
durur. Biten koslar (eval_val.npz var) atlanir; yarida kalirsa yeniden calistirmak kaldigi yerden surer.
Okuma ayri: `python scripts/marl/analyze_m3.py` (ana venv), on kayittaki kural.

  venv-marl\\Scripts\\python scripts/marl/run_m3a.py --frames <F> --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m3a.py --frames <F>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PY = sys.executable
GEN = {0: 20261115, 1: 1, 2: 2, 3: 3, 4: 4}                     # tohum s -> uretec tohumu g_s (§5)
ARMS = [('A0', 0.0, False), ('A1', 0.0, True), ('B0', 20.0, False), ('B1', 20.0, True)]


def gpu_busy(threshold: int) -> bool:
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0]) >= threshold
    except Exception:  # noqa: BLE001
        return False


def wait_gpu(a):
    t0 = time.time()
    while gpu_busy(a.gpu_busy_pct):
        if time.time() - t0 > a.max_wait:
            raise SystemExit(f'GPU {a.max_wait} sn boyunca dolu (>= %{a.gpu_busy_pct}); duruldu')
        print(f'GPU dolu (>= %{a.gpu_busy_pct}), 60 sn bekleniyor...', flush=True)
        time.sleep(60)


def run(cmd, dry):
    print('>', ' '.join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, required=True, help='F (M3 tasarim §5.1)')
    ap.add_argument('--root', default=os.path.join('results', 'marl_m3'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--gpu-busy-pct', type=int, default=30)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    for window in ('train', 'val'):
        for bps in (0, 20):
            out = os.path.join(a.root, f'baselines_{window}_{bps}.npz')
            if not os.path.exists(out):
                run([PY, 'scripts/marl/eval_m3.py', '--baselines', '--window', window, '--cost-bps', str(bps),
                     '--out', out], a.dry_run)
    for arm, bps, pred in ARMS:
        for s in (0, 1, 2, 3, 4):
            d = os.path.join(a.root, arm, f's{s}')
            if os.path.exists(os.path.join(d, 'eval_val.npz')):
                continue
            cmd = [PY, 'scripts/marl/train_m3.py', '--frames', str(a.frames), '--seed', str(s), '--cost-bps', str(bps),
                   '--train-device', a.train_device, '--parallel', '--save-dir', d, '--out', os.path.join(d, 'train.json')]
            if pred:
                cmd += ['--pred-path', os.path.join('data', 'marl', f'pred_pc060_k12_g{GEN[s]}.npz')]
            if not a.dry_run:
                wait_gpu(a)
            if not os.path.exists(os.path.join(d, 'policy.pt')):
                run(cmd, a.dry_run)
            for window in ('train', 'val'):
                run([PY, 'scripts/marl/eval_m3.py', '--run-dir', d, '--window', window,
                     '--out', os.path.join(d, f'eval_{window}.npz')], a.dry_run)


if __name__ == '__main__':
    main()
