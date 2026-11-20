"""
Faz 10 §3.2.8 adim 2: toplu yerel ortam (`marl/batched_env.py`). YALNIZ venv-marl. Sentetik panel, CPU.

Kontroller:
  1. B = 1 toplu ortam, ayni tohum ve ayni eylemlerle TorchRL `PettingZooWrapper` ile bit-es (gozlem, odul, bitis;
     bolum sonlari ve sifirlama dahil).
  2. B = 3: alt ortam i, tohum seed + i ile kurulan bagimsiz tek ortamla bit-es (cekirdek ayni, parite degismez).
  3. Gorev (MarlTask) toplu ortamda BenchMARL'a parti boyutsuz spec verir; gozlem/eylem spec'leri sarmalayiciyla ayni.
  4. Kisa MAPPO `batched=True` ile tamamlanir; ayni tohum -> ayni politika ozeti; `batched` anahtari yokken gorev
     konfigurasyonu eskisiyle ayni (varsayilan kapali).

    venv-marl\\Scripts\\python tests/test_marl_batched_env.py
"""
import hashlib
import os
import sys
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchrl.envs import PettingZooWrapper  # noqa: E402
from torchrl.envs.utils import step_mdp  # noqa: E402

from _marl_fixtures import synth_arrays  # noqa: E402
from marl import obs_spec as OS  # noqa: E402
from marl import panel as PN  # noqa: E402
from marl import m3_runner as R  # noqa: E402
from marl.batched_env import BatchedMarlEnv  # noqa: E402
from marl.benchmarl_task import MarlTask, make_env  # noqa: E402

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


tmp = tempfile.mkdtemp(prefix='marl_batched_')
N, T = 4, 260
arr = synth_arrays(T=T, N=N, seed=0, with_rf=True)
panel_path = os.path.join(tmp, 'panel.npz')
PN.save_npz(arr, panel_path)
CFG = dict(panel_path=panel_path, start_row=OS.WARMUP_ROWS, end_row=180, mode='train', alpha=0.25,
           cost_bps_roundtrip=20.0, reward_term='excess', reward_scale=100.0, episode_len=20, limit_down_gap=0.095)
STEPS = 70          # 3+ bolum (episode_len 20)


def wrap(seed):
    return PettingZooWrapper(env=make_env(CFG), seed=seed, categorical_actions=True, return_state=False, use_mask=False)


def roll(env, acts):
    td = env.reset()
    obs, rew, dones = [td['agent', 'observation'].clone()], [], []
    for a in acts:
        td.set(('agent', 'action'), a)
        td = env.step(td)
        rew.append(td['next', 'agent', 'reward'].clone())
        dones.append(td['next', 'done'].clone())
        nxt = step_mdp(td)
        if bool(td['next', 'done'].any()):
            nxt.set('_reset', td['next', 'done'].clone())
            nxt = env.reset(nxt)
        td = nxt
        obs.append(td['agent', 'observation'].clone())
    return torch.stack(obs), torch.stack(rew), torch.stack(dones)


g = torch.Generator().manual_seed(1)
print("\n[1] B = 1 <-> PettingZooWrapper")
acts1 = [torch.rand(N, 1, generator=g) * 2 - 1 for _ in range(STEPS)]
ow, rw, dw = roll(wrap(0), acts1)
ob, rb, db = roll(BatchedMarlEnv([lambda: make_env(CFG)], seed=0), [a[None] for a in acts1])
check("gozlem, odul ve bitis bit-es; en az 3 bolum sonu kapsandi",
      torch.equal(ow, ob[:, 0]) and torch.equal(rw, rb[:, 0]) and torch.equal(dw, db[:, 0]) and int(dw.sum()) >= 3,
      f"bolum sonu {int(dw.sum())}")

print("\n[2] B = 3 <-> bagimsiz tek ortamlar (tohum s + i)")
acts3 = [torch.rand(3, N, 1, generator=g) * 2 - 1 for _ in range(STEPS)]
ob3, rb3, _ = roll(BatchedMarlEnv([lambda: make_env(CFG)] * 3, seed=10), acts3)
ok = []
for i in range(3):
    o, r, _ = roll(wrap(10 + i), [a[i] for a in acts3])
    ok.append(torch.equal(o, ob3[:, i]) and torch.equal(r, rb3[:, i]))
check("her alt ortam bagimsiz tek ortamla bit-es; alt ortamlar birbirinden farkli",
      all(ok) and not torch.equal(ob3[:, 0], ob3[:, 1]), f"{ok}")

print("\n[3] Gorev spec'leri")
task = MarlTask.PORTFOLIO.update_config({**CFG, 'batched': True})
benv = task.get_env_fun(num_envs=5, continuous_actions=True, seed=0, device='cpu')()
wenv = wrap(0)
so, sa = task.observation_spec(benv), task.action_spec(benv)
check("toplu ortam [5]; gorev parti boyutsuz gozlem/eylem spec'i verir, sarmalayiciyla ayni sekil",
      benv.batch_size == torch.Size([5]) and so['agent', 'observation'].shape == wenv.observation_spec['agent', 'observation'].shape
      and sa['agent', 'action'].shape == wenv.full_action_spec['agent', 'action'].shape and task.group_map(benv) == wenv.group_map)

print("\n[4] Kisa MAPPO, tohum, varsayilan kapali")
BUILD = dict(algo='mappo', frames=400, frames_per_batch=200, epochs=2, n_envs=4, parallel=False, init_std=0.12)


def digest(exp):
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


e1 = R.build_experiment({**CFG, 'batched': True}, seed=3, **BUILD)
e1.run()
e2 = R.build_experiment({**CFG, 'batched': True}, seed=3, **BUILD)
e2.run()
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
import argparse  # noqa: E402
import train_m3 as TM  # noqa: E402
ns = argparse.Namespace(alpha=0.25, cost_bps=20.0, episode_len=252, use_state=False, pred_path=None, extras_path=None,
                        batched_env=False)
default_cfg = TM.train_task_cfg(ns, 100, OS.WARMUP_ROWS)
on_cfg = TM.train_task_cfg(argparse.Namespace(**{**vars(ns), 'batched_env': True}), 100, OS.WARMUP_ROWS)
check("toplu ortamla kosu tamamlandi, ayni tohum -> ayni politika ozeti; bayrak kapaliyken gorevde 'batched' anahtari yok",
      e1.total_frames >= 400 and digest(e1) == digest(e2) and 'batched' not in default_cfg and on_cfg.get('batched') is True)

print("\n" + "=" * 60)
print(f"Sonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
