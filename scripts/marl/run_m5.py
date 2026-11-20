"""Faz 10 M5 mekanizma kollari duzenleyicisi (M5 tasarimi §4 ON KAYDI, PILOT): guvenle agirlikli gat, komsu sinyali
(M4a duzeni, sigma dondurulmus), maliyetsiz, YENI tohumlar 6, 7, 8 (uretec = tohum), F = 600.000, lr 5e-4.

Kollar (taban gat: M4b PILOT_ISARETI'ne dayali KULLANICI KARARI, GECTI kuralinin sonucu degil):
  G0  : gat, guven terimi yok (= M4a/M4b G-X yapilandirmasi; yeni tohumlarla yeniden kosulur)
  GL  : gat + zlogodds, beta ogrenilen (0'dan)                   -> H2
  GLS : GL + karistirilmis guven (derangement, tohum = egitim tohumu) -> H2 kontrolu
  GLM : GL + gonderici basina sicaklik yalniz guven teriminde (T_j log-tekduze [0,25; 4], uretec = egitim tohumu) -> H3
  GD  : gat + prob, beta = 1 sabit (DebUnc'a sadik betimsel kol)
GLG (alici gecidi) bu turda YOK: komsu sinyali duzeninde ajanin "kendi guveni" baska bir hisseye ait (tasarim §4).
Bloklar `pred_m4nb_g{6,7,8}.npz` (build_pred_m4.py, neighbor_map_m5.json). Her egitimden once GPU bos mu bakar.
Biten kosular atlanir. Okuma: analyze_m5.py.

  venv-marl\\Scripts\\python scripts/marl/run_m5.py --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m5.py --jobs 2
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_m3a import ROOT  # noqa: E402
from run_m4a import COST_BPS, FRAMES, PY, wait_gpu  # noqa: E402
from run_m4b import run_logged  # noqa: E402

os.chdir(ROOT)
SEEDS = (6, 7, 8)
GL = ['--conf-mode', 'zlogodds', '--beta-mode', 'learned', '--beta-init', '0']
ARMS = {'G0': [], 'GL': GL, 'GLS': GL + ['--shuffle-conf'], 'GLM': GL + ['--conf-temp-seed', '{seed}'],
        'GD': ['--conf-mode', 'prob', '--beta-mode', 'fixed', '--beta-init', '1']}


def train_cmd(arm: str, seed: int, run_dir: str, device: str, frames: int = FRAMES) -> list:
    pred = os.path.join('data', 'marl', f'pred_m4nb_g{seed}.npz')
    extra = [x.format(seed=seed) for x in ARMS[arm]]
    return [PY, 'scripts/marl/train_m3.py', '--frames', str(frames), '--seed', str(seed), '--cost-bps',
            str(COST_BPS), '--comm', 'gat', '--lr', '5e-4', '--train-device', device, '--parallel', '--save-dir',
            run_dir, '--out', os.path.join(run_dir, 'train.json'), '--pred-path', pred, *extra]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m5'))
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
        a.gpu_busy_pct = max(a.gpu_busy_pct, 80)   # kendi kosularimiz ~%15-30 kullanir; esik yalniz dis yuk icin
    seeds, arms = [int(s) for s in a.seeds.split(',')], a.arms.split(',')
    if a.frames != FRAMES or tuple(seeds) != SEEDS or set(arms) != set(ARMS):
        print(f'UYARI: on kayit disi yapilandirma (frames {a.frames}, tohum {seeds}, kollar {arms}): yalniz duman testi')
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
