"""Faz 10 M3: MarlPortfolioEnv uzerinde BenchMARL MAPPO / IPPO egitimi (venv-marl).

Bu surum yalniz SURE ve TEKRAR URETIM olcer (M3 on kaydindan once butce icin; experiment-sizing):
performans sayisi YAZMAZ (Faz 10 §4 S1 ruhu; secim/degerlendirme protokolu M3 on kaydinda).
Kayit: sure, kare, parametre sayisi, politika ozeti, ortam model_meta'si (K1), is parcacigi, git izi.

  venv-marl\\Scripts\\python scripts/marl/train_m3.py --frames 30000 --train-device cuda --parallel --out <json>

Pencere varsayilani egitim: WARMUP_ROWS .. 2018-12-31 (M2 §9.1 ile ayni bolme; 2021+ okunmaz).
Tohum bit-esligi icin OMP_NUM_THREADS/MKL_NUM_THREADS=1 torch import'undan once (test_marl_m3 [3]).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
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


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', choices=('mappo', 'ippo'), default='mappo')
    ap.add_argument('--frames', type=int, default=30_000)
    ap.add_argument('--frames-per-batch', type=int, default=6000)
    ap.add_argument('--minibatch-iters', type=int, default=45)
    ap.add_argument('--n-envs', type=int, default=10)
    ap.add_argument('--parallel', action='store_true')
    ap.add_argument('--episode-len', type=int, default=252)
    ap.add_argument('--alpha', type=float, default=0.25)
    ap.add_argument('--use-state', action='store_true', help='elestirmen girdisi ortam state() (varsayilan: tum gozlemler)')
    ap.add_argument('--pred-path', default=None)
    ap.add_argument('--train-device', default='cpu')
    ap.add_argument('--sampling-device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1, help='OMP/MKL/torch is parcacigi (bit-eslik icin 1)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', str(a.threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(a.threads))

    import numpy as np
    import torch
    from benchmarl.algorithms import IppoConfig, MappoConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    from marl import obs_spec as OS
    from marl import panel as PN
    from marl.benchmarl_task import MarlTask, make_env

    torch.set_num_threads(a.threads)
    arrays = PN.load_npz(PANEL)
    dates = arrays['dates']
    end_row = int(np.flatnonzero(dates <= TRAIN_END)[-1])
    tcfg = dict(panel_path=PANEL, start_row=OS.WARMUP_ROWS, end_row=end_row, mode='train', alpha=a.alpha,
                cost_bps_roundtrip=20.0, reward_term='excess', reward_scale=100.0, episode_len=a.episode_len,
                use_state=a.use_state, pred_path=a.pred_path)
    meta_env = make_env({k: v for k, v in tcfg.items() if k != 'use_state'})
    task = MarlTask.PORTFOLIO.update_config(tcfg)

    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_frames = a.frames
    cfg.on_policy_collected_frames_per_batch = a.frames_per_batch
    cfg.on_policy_n_envs_per_worker = a.n_envs
    cfg.on_policy_minibatch_size = min(cfg.on_policy_minibatch_size, a.frames_per_batch)
    cfg.on_policy_n_minibatch_iters = a.minibatch_iters
    cfg.parallel_collection = a.parallel
    cfg.sampling_device = a.sampling_device
    cfg.train_device = cfg.buffer_device = a.train_device
    cfg.share_policy_params = True   # K24
    cfg.evaluation = False
    cfg.render = False
    cfg.loggers = []
    cfg.create_json = False
    cfg.checkpoint_at_end = False
    cfg.save_folder = tempfile.mkdtemp(prefix='marl_m3_')
    algo = (MappoConfig if a.algo == 'mappo' else IppoConfig).get_from_yaml()
    exp = Experiment(task=task, algorithm_config=algo, model_config=MlpConfig.get_from_yaml(),
                     critic_model_config=MlpConfig.get_from_yaml(), seed=a.seed, config=cfg)
    t0 = time.perf_counter()
    exp.run()
    wall = time.perf_counter() - t0
    rec = {
        'note': 'yalniz sure ve tekrar uretim; performans sayisi YOK (M3 on kaydindan once)',
        **provenance(), 'args': vars(a), 'panel_sha256': arrays['sha256'],
        'train_window': [str(dates[OS.WARMUP_ROWS]), str(dates[end_row])],
        'wall_seconds': round(wall, 1), 'frames': exp.total_frames,
        'seconds_per_200k': round(wall / max(exp.total_frames, 1) * 200_000, 1),
        'policy_params': sum(p.numel() for p in exp.policy.parameters()), 'policy_sha256': policy_digest(exp),
        'model_meta': {k: v for k, v in meta_env.model_meta().items() if k != 'norm_stats'},
        'env': {'python': platform.python_version(), 'os': platform.platform(), 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'threads': torch.get_num_threads(), 'omp': os.environ.get('OMP_NUM_THREADS')},
    }
    print(json.dumps({k: rec[k] for k in ('wall_seconds', 'frames', 'seconds_per_200k', 'policy_params',
                                          'policy_sha256')}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
