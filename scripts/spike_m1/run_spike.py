"""Faz 10 M1 spike'i: BenchMARL MAPPO / IPPO, 30 ajan, `none` | `attn` (venv-marl'da kosar).

Olcutler (Faz 10 plani §3.2.6): 1 ozel iletisim modulu (AttnComm), 2 ayni tohumla iki CPU kosusu esit mi
(--repro), 3 Windows, 4 RTX 4060 suresi (--train-device cuda), 5 ajan ekseni tensor boyutu (AttnComm girdisi
[..., N, F]), 6 paralel ortam (--n-envs, --parallel), 7 torch CUDA cakismasi (venv-marl torch == ana venv).

Spike'ta uretilen hicbir performans sayisi rapora girmez (§4 S1); bu betik yalniz sure, parametre sayisi ve
politika agirlik ozeti yazar. Panel: prepare_panel.py (2013-06..2020-12; 2021'e dokunmaz).

  venv-marl\\Scripts\\python scripts/spike_m1/run_spike.py --panel <npz> --comm attn --frames 2000 --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from benchmarl.algorithms import IppoConfig, MappoConfig  # noqa: E402
from benchmarl.experiment import Experiment, ExperimentConfig  # noqa: E402
from benchmarl.models.mlp import MlpConfig  # noqa: E402

from benchmarl_glue import AttnCommConfig, SpikeTask, n_params  # noqa: E402


def policy_digest(exp) -> str:
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def one_run(a, seed: int) -> dict:
    task = SpikeTask.PORTFOLIO.update_config(dict(panel_path=a.panel, episode_len=a.episode_len, alpha=a.alpha,
                                                  cost=a.cost, reward_scale=100.0, n_agents=a.n_agents))
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_frames = a.frames
    cfg.on_policy_collected_frames_per_batch = a.frames_per_batch
    cfg.on_policy_n_envs_per_worker = a.n_envs
    cfg.on_policy_minibatch_size = min(cfg.on_policy_minibatch_size, a.frames_per_batch)
    cfg.on_policy_n_minibatch_iters = a.minibatch_iters
    cfg.parallel_collection = a.parallel
    cfg.sampling_device = a.sampling_device
    cfg.train_device = a.train_device
    cfg.buffer_device = a.train_device
    cfg.share_policy_params = True  # K24
    cfg.evaluation = False
    cfg.render = False
    cfg.loggers = []
    cfg.create_json = False
    cfg.checkpoint_at_end = False
    cfg.save_folder = tempfile.mkdtemp(prefix='spike_m1_')
    algo = (MappoConfig if a.algo == 'mappo' else IppoConfig).get_from_yaml()
    critic = MlpConfig.get_from_yaml()
    if a.comm == 'attn':
        model = AttnCommConfig(hidden=a.hidden, heads=4, rounds=1)
    else:
        model = MlpConfig.get_from_yaml()
    exp = Experiment(task=task, algorithm_config=algo, model_config=model, critic_model_config=critic,
                     seed=seed, config=cfg)
    t0 = time.perf_counter()
    exp.run()
    wall = time.perf_counter() - t0
    return {'seed': seed, 'wall_seconds': round(wall, 1), 'frames': exp.total_frames,
            'seconds_per_200k': round(wall / max(exp.total_frames, 1) * 200_000, 1),
            'policy_params': n_params(exp.policy), 'policy_sha256': policy_digest(exp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--comm', choices=('none', 'attn'), default='attn')
    ap.add_argument('--algo', choices=('mappo', 'ippo'), default='mappo')
    ap.add_argument('--frames', type=int, default=200_000)
    ap.add_argument('--frames-per-batch', type=int, default=6000)
    ap.add_argument('--minibatch-iters', type=int, default=45)
    ap.add_argument('--n-envs', type=int, default=10)
    ap.add_argument('--parallel', action='store_true', help='ParallelEnv ile toplama (olcut 6)')
    ap.add_argument('--episode-len', type=int, default=252)
    ap.add_argument('--n-agents', type=int, default=30)
    ap.add_argument('--alpha', type=float, default=0.25)
    ap.add_argument('--cost', type=float, default=0.001)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--sampling-device', default='cpu')
    ap.add_argument('--train-device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repro', action='store_true', help='ayni tohumla iki kosu; politika ozetleri esit mi (olcut 2)')
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    runs = [one_run(a, a.seed)]
    if a.repro:
        runs.append(one_run(a, a.seed))
    rec = {'spike': True, 'note': 'performans sayisi YOK (Faz 10 §4 S1); yalniz sure ve tekrar uretim',
           'args': vars(a), 'runs': runs,
           'repro_equal': (runs[0]['policy_sha256'] == runs[1]['policy_sha256']) if a.repro else None,
           'env': {'python': platform.python_version(), 'os': platform.platform(), 'torch': torch.__version__,
                   'cuda': torch.cuda.is_available(),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'threads': torch.get_num_threads()}}
    print(json.dumps(rec, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1)


if __name__ == '__main__':
    main()
