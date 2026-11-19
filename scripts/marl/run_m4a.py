"""Faz 10 M4a duzenleyicisi (M4 tasarimi §3 ON KAYDI): komsu sinyali, maliyetsiz, 3 tohum, kol basina F = 600.000.

Asama 1 : N-X (none), A-X (attn), G-X (gat); tahmin blogu `pred_m4nb_g{g}.npz` (build_pred_m4.py), lr 5e-4.
Asama 1b: asama 1'de gecmeyen iletisim kollari + N-X, lr 2e-4 (`<root>/lr2e-4/`); liste `m4a_stage1.json` ->
          `stage1b_arms` (hakem-rl M4a #1: durma kararindan once lr duyarliligi).
Asama 2 : gecen kollarin S'si + N-S (`pred_m4nb_g{g}_shuf.npz`), kolun gectigi lr ile; liste asama 1 ve 1b
          JSON'larindaki `stage2_arms` birlesimi. Liste bossa kosulmaz (on kayit §3.3).

Her egitimden once GPU bos mu bakar; doluysa bekler. Biten kosular (eval_val.npz var) atlanir.
Okuma ayri: `python scripts/marl/analyze_m4a.py --stage <1|1b|2>` (ana venv).

  venv-marl\\Scripts\\python scripts/marl/run_m4a.py --stage 1 --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m4a.py --stage 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_m3a import GEN, ROOT, SEEDS, gpu_busy  # noqa: E402

os.chdir(ROOT)
PY = sys.executable
FRAMES = 600_000
COST_BPS = 0
PREREG = 'phase-10-m4-design.md §3 v0.3'
MEAS = os.path.join('docs', 'development', 'measurements', 'phase10-m4')
STAGE_JSON = {1: os.path.join(MEAS, 'm4a_stage1.json'), '1b': os.path.join(MEAS, 'm4a_stage1b.json')}
COMM = {'N': 'none', 'A': 'attn', 'G': 'gat'}
LR = {'': '5e-4', 'lr2e-4': '2e-4'}


def arm_cfg(arm: str) -> dict:
    """'A-X' -> attn, lr 5e-4; 'lr2e-4/G-S' -> gat, lr 2e-4, karistirilmis blok."""
    prefix, _, name = arm.rpartition('/')
    k, v = name.split('-')
    if k not in COMM or v not in ('X', 'S') or prefix not in LR:
        raise ValueError(f'bilinmeyen kol {arm}')
    return {'comm': COMM[k], 'suffix': '_shuf' if v == 'S' else '', 'lr': LR[prefix]}


def train_cmd(arm: str, seed: int, run_dir: str, device: str) -> list:
    c = arm_cfg(arm)
    pred = os.path.join('data', 'marl', f"pred_m4nb_g{GEN[seed]}{c['suffix']}.npz")
    return [PY, 'scripts/marl/train_m3.py', '--frames', str(FRAMES), '--seed', str(seed), '--cost-bps', str(COST_BPS),
            '--comm', c['comm'], '--lr', c['lr'], '--train-device', device, '--parallel', '--save-dir', run_dir,
            '--out', os.path.join(run_dir, 'train.json'), '--pred-path', pred]


def _load(stage):
    path = STAGE_JSON[stage]
    if not os.path.exists(path):
        raise SystemExit(f'kosulmaz: {path} yok (once asama {stage} okumasi)')
    with open(path, encoding='utf-8') as fh:
        rec = json.load(fh)
    if rec.get('preregistration') != PREREG or rec.get('stage') != stage:
        raise SystemExit(f"kosulmaz: {path} bu on kayda ait degil ({rec.get('preregistration')})")
    return rec


def stage_arms(stage: str) -> list:
    if stage == '1':
        return ['N-X', 'A-X', 'G-X']
    s1 = _load(1)
    if stage == '1b':
        arms = s1.get('stage1b_arms') or []
        if not arms:
            raise SystemExit('asama 1b kosulmaz: asama 1de tum iletisim kollari gecti')
        return arms
    arms = list(s1.get('stage2_arms') or [])
    if os.path.exists(STAGE_JSON['1b']):
        arms += _load('1b').get('stage2_arms') or []
    if not arms:
        raise SystemExit('asama 2 kosulmaz: (i)-(iii)u saglayan iletisim kolu yok (on kayit §3.3)')
    return arms


def run(cmd, dry):
    print('>', ' '.join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def wait_gpu(a):
    t0 = time.time()
    while gpu_busy(a.gpu_busy_pct):
        if time.time() - t0 > a.max_wait:
            raise SystemExit(f'GPU {a.max_wait} sn boyunca dolu (>= %{a.gpu_busy_pct}); duruldu')
        print(f'GPU dolu (>= %{a.gpu_busy_pct}), 60 sn bekleniyor...', flush=True)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('1', '1b', '2'), required=True)
    ap.add_argument('--root', default=os.path.join('results', 'marl_m4a'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--gpu-busy-pct', type=int, default=50)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    arms = stage_arms(a.stage)
    for window in ('train', 'val'):
        out = os.path.join(a.root, f'baselines_{window}_{COST_BPS}.npz')
        if not os.path.exists(out):
            run([PY, 'scripts/marl/eval_m3.py', '--baselines', '--window', window, '--cost-bps', str(COST_BPS),
                 '--out', out], a.dry_run)
    for s in SEEDS:                    # tohum dis dongude: kesilirse her kolun ayni tohumlari biter
        for arm in arms:
            d = os.path.join(a.root, *arm.split('/'), f's{s}')
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
