"""Faz 11 S2: politika aktarimi parite REFERANSI (venv-marl). Dondurulmus arsivdeki her politika BenchMARL ile
yuklenir (`m3_runner.load_policy`, egitimle ayni yapi) ve iki girdi kumesinde deterministik cikti kaydedilir:

  (a) 2026 test penceresi: `eval_fold_m8` ile ayni ortam (mode='eval', egitim norm_stats'i, fold_eval.json satirlari);
      her adimda gozlem, loc ve eylem; gunluk R de kaydedilir (M8 test kaydiyla esitlik `s2_parity.py`'de).
  (b) Rastgele gozlem: tohumlu N(0, 3^2) [K, N, d] (kirpma sinirlari +-5 disina da tasar; tanh doygunlugu yoklanir).

Cikti `results/marl_s2/ref/<kol>/<tohum>.npz` (git disi). Test penceresi M8'de zaten okundu; burada yeniden
OKUNMAZ, yalniz ayni seri yeniden uretilir (karar/secim yok).

  venv-marl\\Scripts\\python scripts/marl/s2_parity_ref.py [--arm M-none --seed s201] [--k 256]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ARCHIVE = os.path.join('data', 'marl', 'frozen', 'm8-2026')
OUT = os.path.join('results', 'marl_s2', 'ref')
RAND_SEED = 20261122


def rollout_capture(policy, env_cfg: dict) -> dict:
    """m3_runner.deterministic_rollout ile ayni dongu; ek olarak gozlem / loc / eylem kaydi."""
    import numpy as np
    import torch
    from torchrl.envs import PettingZooWrapper
    from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

    from marl.benchmarl_task import make_env

    base = make_env(env_cfg)
    env = PettingZooWrapper(env=base, categorical_actions=True, return_state=False, use_mask=False)
    obs, loc, act, R = [], [], [], []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        for _ in range(base.end_row - base.start_row):
            obs.append(td['agent', 'observation'].clone())
            td = policy(td)
            loc.append(td['agent', 'loc'].clone())
            act.append(td['agent', 'action'].clone())
            td = env.step(td)
            R.append(base.last_info['R'])
            if bool(td['next', 'done'].all()):
                break
            td = step_mdp(td)
    cat = lambda xs: torch.stack(xs).numpy()   # noqa: E731
    return {'obs': cat(obs), 'loc': cat(loc), 'action': cat(act), 'R': np.asarray(R)}


def random_capture(policy, shape, k: int) -> dict:
    import torch
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    g = torch.Generator().manual_seed(RAND_SEED)
    x = 3.0 * torch.randn(k, *shape, generator=g)
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = policy(TensorDict({('agent', 'observation'): x}, batch_size=[k]))
    return {'obs': x.numpy(), 'loc': td['agent', 'loc'].numpy(), 'action': td['agent', 'action'].numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default=None)
    ap.add_argument('--seed', default=None)
    ap.add_argument('--k', type=int, default=256)
    a = ap.parse_args()
    with open(os.path.join(ARCHIVE, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    if a.arm is None:
        # Kol basina AYRI surec: BenchMARL gorev yapilandirmasi sinif duzeyinde tutulur (MarlTask.PORTFOLIO.update_config
        # birlestirir); SARL kolunun single_agent anahtari ayni surecte sonraki MARL koluna tasar. eval_fold_m8 de her
        # politikayi ayri surecte yukler.
        import subprocess
        for arm in sorted({p['arm'] for p in man['policies']}):
            cmd = [sys.executable, os.path.abspath(__file__), '--arm', arm, '--k', str(a.k)]
            subprocess.run(cmd + (['--seed', a.seed] if a.seed else []), check=True)
        return
    import numpy as np

    from marl.m3_runner import load_policy

    todo = [p for p in man['policies'] if (a.arm in (None, p['arm'])) and (a.seed in (None, p['seed']))]
    for p in todo:
        t0 = time.time()
        d = os.path.join(ARCHIVE, p['arm'], p['seed'])
        exp, meta = load_policy(d, weights=p['ckpt'])
        with open(os.path.join(d, 'fold_eval.json'), encoding='utf-8') as fh:
            fe = json.load(fh)
        ts, te = fe['rows']['test']
        cfg = {k: v for k, v in meta['task_cfg'].items() if k != 'use_state'}
        cfg.update(mode='eval', norm_stats_path=os.path.join(d, 'norm_stats.json'), start_row=ts, end_row=te)
        ro = rollout_capture(exp.policy, cfg)
        rd = random_capture(exp.policy, ro['obs'].shape[1:], a.k)
        out = os.path.join(OUT, p['arm'])
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f"{p['seed']}.npz"), ckpt=p['ckpt'], params_sha=p['params_sha'],
                            rows=np.asarray([ts, te]), **{f'win_{k}': v for k, v in ro.items()},
                            **{f'rand_{k}': v for k, v in rd.items()})
        print(f"{p['arm']}/{p['seed']}: {ro['obs'].shape[0]} adim, gozlem {tuple(ro['obs'].shape[1:])}, "
              f"{time.time() - t0:.1f} sn", flush=True)


if __name__ == '__main__':
    main()
