"""Faz 10 §3.2.8 adim 1: ortam hizi profili (venv-marl). Uzun kosulardan (M8) once sure nereye gidiyor? Basarim OKUNMAZ.

Olcumler (egitim penceresi, tahminsiz, gat icin graf; rastgele eylem ya da kisa egitim):
  A  ham ortam   : tek surecte MarlPortfolioEnv, rastgele eylemle --steps adim; adim basi ms + cProfile (kumulatif pay)
  B  sarmalayici : ayni adimlar TorchRL PettingZooWrapper icinden (rand_step); adim basi ms
  C  egitim      : train_m3.py --frames 30000 gat; (i) paralel, 10 epoch  (ii) paralel, 1 epoch  (iii) seri, 10 epoch.
                   (i) - (ii) ~ 9 epoch guncelleme; (iii) - (i) ~ paralel toplamanin kazanci; (ii) ~ toplama + 1 epoch.
Kayit: measurements/phase10-m8/env_profile.json (git izi, GPU, is parcacigi).

  venv-marl\\Scripts\\python scripts/marl/profile_env_m8.py --out docs/development/measurements/phase10-m8/env_profile.json
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
TRAIN_END = '2018-12-31'


def _git(*a) -> str:
    return subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()


def env_cfg() -> dict:
    from marl import obs_spec as OS
    from marl import panel as PN
    arrays = PN.load_npz(PANEL)
    end_row = int(np.flatnonzero(arrays['dates'] <= TRAIN_END)[-1])
    return dict(panel_path=PANEL, start_row=OS.WARMUP_ROWS, end_row=end_row, mode='train', alpha=0.25,
                cost_bps_roundtrip=20.0, reward_term='excess', reward_scale=100.0, episode_len=252)


def raw_env(steps: int, seed: int = 0) -> dict:
    from marl.benchmarl_task import make_env
    env = make_env(env_cfg())
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    agents = list(env.agents)
    shape = env.action_space(agents[0]).shape

    def loop():
        for _ in range(steps):
            acts = {a: rng.uniform(-1, 1, shape).astype(np.float32) for a in agents}
            _, _, term, trunc, _ = env.step(acts)
            if any(term.values()) or any(trunc.values()):
                env.reset(seed=seed)

    t0 = time.perf_counter()
    loop()
    wall = time.perf_counter() - t0
    env.reset(seed=seed)
    pr = cProfile.Profile()
    pr.enable()
    loop()
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, _) in st.stats.items():
        rows.append({'func': f"{os.path.relpath(fn, ROOT).replace(os.sep, '/') if fn.startswith(ROOT) else os.path.basename(fn)}:{name}",
                     'cum_share': ct / total, 'self_share': tt / total, 'calls_per_step': nc / steps})
    top_cum = sorted([r for r in rows if r['func'].startswith('marl/')], key=lambda r: -r['cum_share'])[:12]
    top_self = sorted(rows, key=lambda r: -r['self_share'])[:10]
    return {'steps': steps, 'ms_per_step': 1000 * wall / steps, 'profile_total_s': total,
            'top_marl_cumulative': top_cum, 'top_self': top_self}


def wrapped_env(steps: int, seed: int = 0) -> dict:
    from torchrl.envs import PettingZooWrapper
    from torchrl.envs.utils import step_mdp
    from marl.benchmarl_task import make_env
    env = PettingZooWrapper(env=make_env(env_cfg()), seed=seed, categorical_actions=True, return_state=False,
                            use_mask=False)
    td = env.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        td = env.rand_step(td)
        done = bool(td['next', 'done'].any())
        td = env.reset() if done else step_mdp(td)
    wall = time.perf_counter() - t0
    return {'steps': steps, 'ms_per_step': 1000 * wall / steps}


def train_runs(frames: int, device: str) -> dict:
    py = sys.executable
    out = {}
    tmp = tempfile.mkdtemp(prefix='prof_m8_')
    for name, extra in (('parallel_ep10', ['--parallel', '--epochs', '10']), ('parallel_ep1', ['--parallel', '--epochs', '1']),
                        ('serial_ep10', ['--epochs', '10'])):
        js = os.path.join(tmp, f'{name}.json')
        cmd = [py, 'scripts/marl/train_m3.py', '--frames', str(frames), '--seed', '0', '--cost-bps', '20', '--comm', 'gat',
               '--lr', '5e-4', '--train-device', device, '--out', js, *extra]
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        proc_wall = time.perf_counter() - t0
        r = json.load(open(js, encoding='utf-8'))
        out[name] = {'train_wall_s': r['wall_seconds'], 'process_wall_s': round(proc_wall, 1), 'frames': r['frames'],
                     'ms_per_frame': 1000 * r['wall_seconds'] / r['frames']}
    p10, p1, s10 = (out[k]['train_wall_s'] for k in ('parallel_ep10', 'parallel_ep1', 'serial_ep10'))
    out['decomposition'] = {'update_9_epochs_s': p10 - p1, 'update_share_parallel_ep10': (p10 - p1) / p10 * 10 / 9,
                            'collection_plus_1ep_s': p1, 'parallel_speedup_vs_serial': s10 / p10,
                            'note': 'guncelleme payi ~ (ep10 - ep1) * 10/9 / ep10 (epoch basi esit varsayimi); kaba'}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--frames', type=int, default=30000)
    ap.add_argument('--train-device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    rec = {'note': 'Faz 10 §3.2.8 adim 1: ortam hizi profili; basarim okunmaz', 'commit': _git('rev-parse', 'HEAD'),
           'git_status_porcelain': _git('status', '--porcelain'),
           'script_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           'env': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'omp': os.environ.get('OMP_NUM_THREADS'), 'cpu_count': os.cpu_count()}}
    rec['A_raw_env'] = raw_env(a.steps)
    print('A ham ortam ms/adim', round(rec['A_raw_env']['ms_per_step'], 3), flush=True)
    rec['B_wrapped_env'] = wrapped_env(a.steps)
    print('B sarmalayici ms/adim', round(rec['B_wrapped_env']['ms_per_step'], 3), flush=True)
    rec['C_training'] = train_runs(a.frames, a.train_device)
    print('C', json.dumps(rec['C_training']['decomposition'], default=float), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
