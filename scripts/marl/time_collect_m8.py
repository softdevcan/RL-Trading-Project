"""Faz 10 §3.2.8 adim 2: toplu yerel ortamin sure olcumu (venv-marl). Basarim OKUNMAZ; yalniz duvar suresi.

Yapilandirmalar (train_m3.py --frames 30000, gat, 20 bps, tahminsiz; ayni tohum):
  parallel_n10  : bugunku duzen (PettingZooWrapper x 10, ParallelEnv)
  batched_n10/30/60 : `--batched-env --n-envs n` (marl/batched_env.py)
  concurrent3_batched_n10 : uc toplu kosu ayni anda (toplam duvar = surec baslatma dahil)
Not: alti eszamanli toplu kosu GPU bellegini (8 GB) doldurup durdu (2026-11-20 elle denendi, kayit disi); burada kosulmaz.

  venv-marl\\Scripts\\python scripts/marl/time_collect_m8.py --out docs/development/measurements/phase10-m8/collect_timing.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)


def _git(*a) -> str:
    return subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()


def cmd(frames, seed, extra, out):
    return [sys.executable, 'scripts/marl/train_m3.py', '--frames', str(frames), '--seed', str(seed), '--cost-bps', '20',
            '--comm', 'gat', '--lr', '5e-4', '--train-device', 'cuda', '--out', out, *extra]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=30000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='collect_m8_')
    rec = {'note': 'Faz 10 §3.2.8 adim 2 sure olcumu; basarim okunmaz', 'commit': _git('rev-parse', 'HEAD'),
           'git_status_porcelain': _git('status', '--porcelain'), 'frames': a.frames,
           'script_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(), 'runs': {}}
    for name, extra in (('parallel_n10', ['--parallel']), ('batched_n10', ['--batched-env']),
                        ('batched_n30', ['--batched-env', '--n-envs', '30']), ('batched_n60', ['--batched-env', '--n-envs', '60'])):
        js = os.path.join(tmp, name + '.json')
        t0 = time.perf_counter()
        subprocess.run(cmd(a.frames, 0, extra, js), check=True, capture_output=True)
        r = json.load(open(js, encoding='utf-8'))
        rec['runs'][name] = {'train_wall_s': r['wall_seconds'], 'process_wall_s': round(time.perf_counter() - t0, 1),
                             'policy_sha256': r['policy_sha256']}
        print(name, rec['runs'][name], flush=True)
    t0 = time.perf_counter()
    procs = [subprocess.Popen(cmd(a.frames, s, ['--batched-env'], os.path.join(tmp, f'c3_{s}.json')),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for s in (1, 2, 3)]
    for p in procs:
        p.wait()
    walls = [json.load(open(os.path.join(tmp, f'c3_{s}.json'), encoding='utf-8'))['wall_seconds'] for s in (1, 2, 3)]
    rec['runs']['concurrent3_batched_n10'] = {'total_wall_s': round(time.perf_counter() - t0, 1), 'train_wall_s': walls}
    base = rec['runs']['parallel_n10']
    rec['speedup_train_wall_vs_parallel_n10'] = {k: base['train_wall_s'] / v['train_wall_s']
                                                 for k, v in rec['runs'].items() if k.startswith('batched')}
    rec['throughput_runs_per_hour'] = {
        'parallel_n10_one_at_a_time': 3600 / base['process_wall_s'],
        'batched_n10_one_at_a_time': 3600 / rec['runs']['batched_n10']['process_wall_s'],
        'batched_n10_three_concurrent': 3 * 3600 / rec['runs']['concurrent3_batched_n10']['total_wall_s']}
    print(json.dumps({k: rec[k] for k in ('speedup_train_wall_vs_parallel_n10', 'throughput_runs_per_hour')}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
