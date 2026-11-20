"""Faz 10 M3a duzenleyicisi (M3 tasarim §5 v0.2): 2 kol (A1 PC060-K12, S1 karistirilmis PC) x 3 tohum, maliyetsiz;
egitim + deterministik degerlendirme + baseline (toplam hedef ~1 saat).

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
GEN = {0: 20261115, 1: 1, 2: 2}                                  # tohum s -> uretec tohumu g_s (§5 v0.2)
ARMS = [('A1', 0.0, ''), ('S1', 0.0, '_shuf')]                    # PC060-K12 | karistirilmis PC (maliyetsiz)
SEEDS = (0, 1, 2)
F_MIN = 600_000                                                  # §5.1 alt sinir (hakem-rl ikinci okuma #3)


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
    ap.add_argument('--batched-env', action='store_true',
                    help='Faz 10 §3.2.8 toplu yerel ortam (--parallel yerine); M3 tasarimi §5.3 tekrari')
    ap.add_argument('--jobs', type=int, default=1, help='es zamanli kosu (toplu ortamla 3; GPU bellegi siniri)')
    a = ap.parse_args()
    if a.jobs > 1:
        a.gpu_busy_pct = max(a.gpu_busy_pct, 80)   # kendi kosularimiz GPU'yu kullanir; esik yalniz dis yuk icin
    if a.frames < F_MIN:
        raise SystemExit(f'F = {a.frames} < F_min = {F_MIN}: M3a kosulmaz, sure karari kullaniciya doner (§5.1)')
    for window in ('train', 'val'):
        for bps in (0,):
            out = os.path.join(a.root, f'baselines_{window}_{bps}.npz')
            if not os.path.exists(out):
                run([PY, 'scripts/marl/eval_m3.py', '--baselines', '--window', window, '--cost-bps', str(bps),
                     '--out', out], a.dry_run)
    collect = ['--batched-env'] if a.batched_env else ['--parallel']
    tasks = [(arm, bps, suffix, s) for arm, bps, suffix in ARMS for s in SEEDS
             if not os.path.exists(os.path.join(a.root, arm, f's{s}', 'eval_val.npz'))]

    def one(task):
        arm, bps, suffix, s = task
        d = os.path.join(a.root, arm, f's{s}')
        cmd = [PY, 'scripts/marl/train_m3.py', '--frames', str(a.frames), '--seed', str(s), '--cost-bps', str(bps),
               '--train-device', a.train_device, *collect, '--save-dir', d, '--out', os.path.join(d, 'train.json')]
        cmd += ['--pred-path', os.path.join('data', 'marl', f'pred_pc060_k12_g{GEN[s]}{suffix}.npz')]
        if not a.dry_run:
            wait_gpu(a)
        if not os.path.exists(os.path.join(d, 'policy.pt')):
            run(cmd, a.dry_run)
        for window in ('train', 'val'):
            run([PY, 'scripts/marl/eval_m3.py', '--run-dir', d, '--window', window,
                 '--out', os.path.join(d, f'eval_{window}.npz')], a.dry_run)

    if a.jobs <= 1:
        for t in tasks:
            one(t)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for f in [ex.submit(one, t) for t in tasks]:
                f.result()


if __name__ == '__main__':
    main()
