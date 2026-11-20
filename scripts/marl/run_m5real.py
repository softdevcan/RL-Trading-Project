"""Faz 10 M5 gercek sinyal kollari duzenleyicisi (M5 tasarimi §5 ON KAYDI, TASLAK): `pit_v2` gercek sinyali (komsu
sinyali YOK), 20 bps, YENI tohumlar 9, 10, 11, F = 600.000, lr 5e-4. Saglamlik sorusu (AQ2-gercek, §2.3).

Kollar (yol x blok):
  T0-N  none, tahminsiz          SO  none + gercek      SO-S  none + karistirilmis      SO-S2  none + karistirilmis-2
  T0-G  gat,  tahminsiz          SB  gat + GL + gercek  SB-S  gat + GL + karistirilmis  SB-S2  gat + GL + karistirilmis-2
GL = zlogodds guven terimi, beta ogrenilen (M5 mekanizma kollariyla ayni). K-g: S-msg ⊂ ikisi birden (guven gozlemden).
Bloklar `build_pred_m5real.py` (pred_pitv2.npz, pred_pitv2_g{s}_shuf[2].npz). Okuma: analyze_m5real.py.

  venv-marl\\Scripts\\python scripts/marl/run_m5real.py --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m5real.py --jobs 2
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
from run_m5 import GL  # noqa: E402

os.chdir(ROOT)
SEEDS = (9, 10, 11)
COST_BPS = 20
# kol -> (iletisim, blok soneki ya da None = tahminsiz, ek bayraklar)
ARMS = {'T0-N': ('none', None, []), 'SO': ('none', '', []), 'SO-S': ('none', '_shuf', []), 'SO-S2': ('none', '_shuf2', []),
        'T0-G': ('gat', None, []), 'SB': ('gat', '', GL), 'SB-S': ('gat', '_shuf', GL), 'SB-S2': ('gat', '_shuf2', GL)}


def pred_path(suffix, seed: int):
    if suffix is None:
        return None
    name = 'pred_pitv2.npz' if suffix == '' else f'pred_pitv2_g{seed}{suffix}.npz'
    return os.path.join('data', 'marl', name)


def train_cmd(arm: str, seed: int, run_dir: str, device: str, frames: int = FRAMES) -> list:
    comm, suffix, extra = ARMS[arm]
    cmd = [PY, 'scripts/marl/train_m3.py', '--frames', str(frames), '--seed', str(seed), '--cost-bps', str(COST_BPS),
           '--comm', comm, '--lr', '5e-4', '--train-device', device, '--parallel', '--save-dir', run_dir,
           '--out', os.path.join(run_dir, 'train.json'), *extra]
    p = pred_path(suffix, seed)
    return cmd + (['--pred-path', p] if p else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m5real'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--frames', type=int, default=FRAMES, help='yalniz duman testi icin degistir (on kayit 600.000)')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='yalniz duman testi icin degistir')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--gpu-busy-pct', type=int, default=50)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--jobs', type=int, default=1, help='es zamanli kosu (M4b olcumu: 2 -> ~1,6x verim; 3 -> ~1,3x)')
    a = ap.parse_args()
    if a.jobs > 1:
        a.gpu_busy_pct = max(a.gpu_busy_pct, 80)
    seeds, arms = [int(s) for s in a.seeds.split(',')], a.arms.split(',')
    if a.frames != FRAMES or tuple(seeds) != SEEDS or set(arms) != set(ARMS):
        print(f'UYARI: on kayit disi yapilandirma (frames {a.frames}, tohum {seeds}, kollar {arms}): yalniz duman testi')
    for s in seeds:
        for arm in arms:
            p = pred_path(ARMS[arm][1], s)
            if p and not os.path.exists(p):
                raise SystemExit(f'blok yok: {p} (python scripts/marl/build_pred_m5real.py)')
    tasks = [(s, arm) for s in seeds for arm in arms
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
