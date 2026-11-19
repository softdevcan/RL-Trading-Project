"""Faz 10 M3b duzenleyicisi (M3 tasarim §6 ON KAYDI v1.1): 20 bps, 3 tohum, kol basina F = 600.000 kare.

Asama 1: A1c (PC060-K12) <-> S1c (karistirilmis PC), referans yapilandirma.
Asama 2: referansa (A1c) karsi tek yapilandirma secimi: IPPO, STATE (elestirmen state()), A05 (alfa 0,5), A00 (alfa 0, ablasyon).
Asama 2 yalniz asama 1 okumasi GECTI ise baslar (`m3b_stage1.json`).

Her egitimden once GPU bos mu bakar; doluysa bekler. Biten kosular (eval_val.npz var) atlanir.
Okuma ayri: `python scripts/marl/analyze_m3b.py --stage <1|2>` (ana venv).

  venv-marl\\Scripts\\python scripts/marl/run_m3b.py --stage 1 --dry-run
  venv-marl\\Scripts\\python scripts/marl/run_m3b.py --stage 1
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
FRAMES = 600_000                                                 # M3a ile ayni butce (§6.2)
COST_BPS = 20
PREREG = 'phase-10-m3-design.md §6 v1.1'
STAGE1_JSON = os.path.join('docs', 'development', 'measurements', 'phase10-m3', 'm3b_stage1.json')
ARMS = {
    1: {'A1c': {}, 'S1c': {'suffix': '_shuf'}},
    2: {'IPPO': {'algo': 'ippo'}, 'STATE': {'use_state': True}, 'A05': {'alpha': 0.5}, 'A00': {'alpha': 0.0}},
}


def train_cmd(arm_cfg: dict, seed: int, run_dir: str, device: str) -> list:
    pred = os.path.join('data', 'marl', f"pred_pc060_k12_g{GEN[seed]}{arm_cfg.get('suffix', '')}.npz")
    cmd = [PY, 'scripts/marl/train_m3.py', '--frames', str(FRAMES), '--seed', str(seed), '--cost-bps', str(COST_BPS),
           '--algo', arm_cfg.get('algo', 'mappo'), '--alpha', str(arm_cfg.get('alpha', 0.25)),
           '--train-device', device, '--parallel', '--save-dir', run_dir,
           '--out', os.path.join(run_dir, 'train.json'), '--pred-path', pred]
    return cmd + (['--use-state'] if arm_cfg.get('use_state') else [])


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


def require_stage1_pass():
    if not os.path.exists(STAGE1_JSON):
        raise SystemExit(f'asama 2 kosulmaz: {STAGE1_JSON} yok (once asama 1 okumasi)')
    with open(STAGE1_JSON, encoding='utf-8') as fh:
        rec = json.load(fh)
    if rec.get('preregistration') != PREREG or rec.get('stage') != 1:
        raise SystemExit(f"asama 2 kosulmaz: {STAGE1_JSON} bu on kayda ait degil ({rec.get('preregistration')}, "
                         f"asama {rec.get('stage')})")
    if rec.get('label') != 'GECTI':
        raise SystemExit(f"asama 2 kosulmaz: asama 1 etiketi {rec.get('label')} (on kayit §6.3)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=(1, 2), required=True)
    ap.add_argument('--root', default=os.path.join('results', 'marl_m3b'))
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--gpu-busy-pct', type=int, default=30)
    ap.add_argument('--max-wait', type=int, default=4 * 3600)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.stage == 2:
        require_stage1_pass()
    for window in ('train', 'val'):
        out = os.path.join(a.root, f'baselines_{window}_{COST_BPS}.npz')
        if not os.path.exists(out):
            run([PY, 'scripts/marl/eval_m3.py', '--baselines', '--window', window, '--cost-bps', str(COST_BPS),
                 '--out', out], a.dry_run)
    for arm, cfg in ARMS[a.stage].items():
        for s in SEEDS:
            d = os.path.join(a.root, arm, f's{s}')
            if os.path.exists(os.path.join(d, 'eval_val.npz')):
                continue
            if not a.dry_run:
                wait_gpu(a)
            if not os.path.exists(os.path.join(d, 'policy.pt')):
                run(train_cmd(cfg, s, d, a.train_device), a.dry_run)
            for window in ('train', 'val'):
                run([PY, 'scripts/marl/eval_m3.py', '--run-dir', d, '--window', window,
                     '--out', os.path.join(d, f'eval_{window}.npz')], a.dry_run)


if __name__ == '__main__':
    main()
