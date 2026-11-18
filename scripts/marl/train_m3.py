"""Faz 10 M3: MarlPortfolioEnv uzerinde BenchMARL MAPPO / IPPO egitimi (venv-marl).

Performans sayisi YAZMAZ: degerlendirme `scripts/marl/eval_m3.py` (deterministik, tam pencere), okuma M3 on
kaydindaki kuralla `scripts/marl/analyze_m3.py`. Kayit: sure, kare, parametre sayisi, politika ozeti, ortam
model_meta'si (K1), is parcacigi, git izi; `--save-dir` verilirse politika + norm_stats + meta (eval icin).

  # sure olcumu (M3 butcesi icin; GPU bos olmali)
  venv-marl\\Scripts\\python scripts/marl/train_m3.py --frames 30000 --train-device cuda --parallel --out <json>
  # M3 kosusu (on kayittaki kol/tohum)
  venv-marl\\Scripts\\python scripts/marl/train_m3.py --frames <butce> --pred-path data/marl/pred_pc060_k12_g1.npz \\
      --seed 1 --train-device cuda --parallel --save-dir results/marl_m3/<kol>/s1 --out <json>

Pencere: egitim WARMUP_ROWS .. 2018-12-31 (M2 §9.1 bolmesi; 2021+ okunmaz).
Tohum bit-esligi: OMP/MKL_NUM_THREADS=1 torch import'undan once (test_marl_m3 [3]).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
TRAIN_END = '2018-12-31'


def policy_digest(exp) -> str:
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def critic_summary(exp) -> dict:
    """Elestirmen kapasitesi (hakem-rl M3b #2): MAPPO, IPPO ve state() kollarinda boyut esit DEGIL, kayda gecer."""
    params = dict(exp.losses['agent'].critic_network_params.flatten_keys('.').items())
    first = next(v for k, v in params.items() if k.endswith('weight'))
    return {'params': int(sum(v.numel() for v in params.values())), 'input_dim': int(first.shape[-1])}


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def train_task_cfg(a, end_row: int, warmup: int) -> dict:
    return dict(panel_path=PANEL, start_row=warmup, end_row=end_row, mode='train', alpha=a.alpha,
                cost_bps_roundtrip=a.cost_bps, reward_term='excess', reward_scale=100.0, episode_len=a.episode_len,
                use_state=a.use_state, pred_path=a.pred_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', choices=('mappo', 'ippo'), default='mappo')
    ap.add_argument('--frames', type=int, default=30_000)
    ap.add_argument('--frames-per-batch', type=int, default=6000)
    ap.add_argument('--epochs', type=int, default=10, help='MAPPO onerisi <= 10; mini-parti bolunmez')
    ap.add_argument('--n-envs', type=int, default=10)
    ap.add_argument('--parallel', action='store_true')
    ap.add_argument('--episode-len', type=int, default=252)
    ap.add_argument('--alpha', type=float, default=0.25)
    ap.add_argument('--cost-bps', type=float, default=20.0, help='gidis-donus bps (D4)')
    ap.add_argument('--init-std', type=float, default=0.12, help='tanh oncesi baslangic std (hakem-rl R7)')
    ap.add_argument('--lr', type=float, default=5e-4, help='M3a on kaydi v0.2 (hakem-rl M3a #4); BenchMARL varsayilani 5e-5')
    ap.add_argument('--use-state', action='store_true', help='elestirmen girdisi ortam state() (varsayilan: tum gozlemler)')
    ap.add_argument('--pred-path', default=None)
    ap.add_argument('--train-device', default='cpu')
    ap.add_argument('--sampling-device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1, help='OMP/MKL/torch is parcacigi (bit-eslik icin 1)')
    ap.add_argument('--save-dir', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', str(a.threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(a.threads))

    import numpy as np
    import torch

    from marl import obs_spec as OS
    from marl import panel as PN
    from marl.m3_runner import build_experiment, save_policy

    torch.set_num_threads(a.threads)
    arrays = PN.load_npz(PANEL)
    dates = arrays['dates']
    end_row = int(np.flatnonzero(dates <= TRAIN_END)[-1])
    tcfg = train_task_cfg(a, end_row, OS.WARMUP_ROWS)
    build = dict(algo=a.algo, seed=a.seed, frames=a.frames, frames_per_batch=a.frames_per_batch, epochs=a.epochs,
                 n_envs=a.n_envs, parallel=a.parallel, init_std=a.init_std, lr=a.lr)
    exp = build_experiment(tcfg, train_device=a.train_device, sampling_device=a.sampling_device, **build)
    t0 = time.perf_counter()
    exp.run()
    wall = time.perf_counter() - t0
    rec = {
        'note': 'egitim kaydi; performans sayisi YOK (degerlendirme eval_m3, okuma M3 on kaydi)',
        **provenance(), 'args': vars(a), 'build': build, 'panel_sha256': arrays['sha256'],
        'pred_sha256': (hashlib.sha256(open(a.pred_path, 'rb').read()).hexdigest() if a.pred_path else None),
        'train_window': [str(dates[OS.WARMUP_ROWS]), str(dates[end_row])],
        'wall_seconds': round(wall, 1), 'frames': exp.total_frames,
        'seconds_per_200k': round(wall / max(exp.total_frames, 1) * 200_000, 1),
        'policy_params': sum(p.numel() for p in exp.policy.parameters()), 'critic': critic_summary(exp),
        'policy_sha256': policy_digest(exp),
        'env': {'python': platform.python_version(), 'os': platform.platform(), 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'threads': torch.get_num_threads(), 'omp': os.environ.get('OMP_NUM_THREADS')},
    }
    if a.save_dir:
        rec['model_meta'] = save_policy(exp, tcfg, {'build': build, 'provenance': provenance(),
                                                    'policy_sha256': rec['policy_sha256']}, a.save_dir)['model_meta']
    print(json.dumps({k: rec[k] for k in ('wall_seconds', 'frames', 'seconds_per_200k', 'policy_params',
                                          'policy_sha256')}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
