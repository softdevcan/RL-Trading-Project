r"""Faz 10 M8 tasarimi A2 (plan K15): SARL kontrol kollari, `marl/sarl.py`. YALNIZ venv-marl.

[1] Tek-ajan gorunumu: ayni eylem dizisiyle SARL ve MARL ortaminin muhasebesi bit-es (R, X, devir, nakit, deger, satis
    engeli); SARL gozlemi MARL gozlemlerinin yan yana dizilisi; SARL odulu MARL odullerinin ortalamasi (alpha = 0'da
    her ajanin odulune esit); bolum sonu ve yeniden baslangic ayni.
[2] Toplu yol: BatchedMarlEnv(SARL) B = 1, PettingZooWrapper(SARL) ile bit-es; grup 'agent', tek uye.
[3] Esmimari: RowSharedPolicy parametre sekilleri MARL paylasimli ajan MLP'siyle ayni; MARL agirliklari kopyalaninca
    loc/scale MARL'in ajan basina ciktisina esit (eylem dagilimi ayni).
[4] Kisa MAPPO: iki SARL kolu tamamlanir, ayni tohum -> ayni politika; toplu yol egitimde calisir; kaydet/yukle +
    deterministik degerlendirme tam pencere.

    venv-marl\Scripts\python tests/test_marl_sarl.py
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
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp  # noqa: E402

from _marl_fixtures import synth_arrays  # noqa: E402
from marl import m3_runner as R  # noqa: E402
from marl import obs_spec as OS  # noqa: E402
from marl import panel as PN  # noqa: E402
from marl.batched_env import BatchedMarlEnv  # noqa: E402
from marl.benchmarl_task import make_env  # noqa: E402

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def digest(exp) -> str:
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


tmp = tempfile.mkdtemp(prefix='marl_sarl_')
N, T = 4, 260
PN.save_npz(synth_arrays(T=T, N=N, seed=0, with_rf=True), os.path.join(tmp, 'panel.npz'))
CFG = dict(panel_path=os.path.join(tmp, 'panel.npz'), start_row=OS.WARMUP_ROWS, end_row=180, mode='train', alpha=0.25,
           cost_bps_roundtrip=20.0, reward_term='excess', reward_scale=100.0, episode_len=20, limit_down_gap=0.095)
SCFG = {**CFG, 'single_agent': True}
ESM = {'kind': 'rowshared', 'n_rows': N, 'num_cells': [256, 256]}
KEYS = ('R', 'X', 'turnover', 'cash_weight', 'value', 'n_sell_blocked')

# ---------------------------------------------------------------------------------------------
print("\n[1] Tek-ajan gorunumu <-> MARL ortami (ayni eylemler)")


def parity(alpha: float, steps: int = 45):
    em, es = make_env({**CFG, 'alpha': alpha}), make_env({**SCFG, 'alpha': alpha})
    om, _ = em.reset(seed=7)
    os_, _ = es.reset(seed=7)
    cat = lambda o: np.concatenate([o[a] for a in em.possible_agents]).astype(np.float32)   # noqa: E731
    ok, ends, rng = np.array_equal(os_['agent_0'], cat(om)), 0, np.random.default_rng(3)
    eq_each = True
    for _ in range(steps):
        z = rng.uniform(-em.bound, em.bound, N).astype(np.float32)
        om, rm, tm, um, _ = em.step({a: z[i:i + 1] for i, a in enumerate(em.possible_agents)})
        os_, rs, ts, us, _ = es.step({'agent_0': z})
        ok &= all(em.last_info[k] == es.last_info[k] for k in KEYS)
        ok &= np.array_equal(os_['agent_0'], cat(om))
        ok &= rs['agent_0'] == float(np.mean([rm[a] for a in em.possible_agents]))
        eq_each &= all(rs['agent_0'] == rm[a] for a in em.possible_agents)
        ok &= ts['agent_0'] == tm['agent_0'] and us['agent_0'] == um['agent_0']
        if not em.agents:
            ok &= es.agents == []
            ends += 1
            om, _ = em.reset()
            os_, _ = es.reset()
            ok &= np.array_equal(os_['agent_0'], cat(om))
    return bool(ok), ends, bool(eq_each)


ok25, ends25, each25 = parity(0.25)
ok0, _, each0 = parity(0.0)
check("muhasebe bit-es (R, X, devir, nakit, deger, satis engeli), gozlem yan yana dizilis, odul = MARL ortalamasi, "
      "bitis ve yeniden baslangic ayni (2+ bolum); alpha = 0'da odul her ajaninkine esit, 0,25'te degil",
      ok25 and ok0 and ends25 >= 2 and each0 and not each25, f"{ok25} {ok0} {ends25} {each0} {each25}")

# ---------------------------------------------------------------------------------------------
print("\n[2] Toplu yol: BatchedMarlEnv(SARL) B = 1 <-> PettingZooWrapper(SARL)")


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
acts = [torch.rand(1, N, generator=g) * 2 - 1 for _ in range(50)]
wenv = PettingZooWrapper(env=make_env(SCFG), seed=0, categorical_actions=True, return_state=False, use_mask=False)
benv = BatchedMarlEnv([lambda: make_env(SCFG)], seed=0)
ow, rw, dw = roll(wenv, acts)
ob, rb, db = roll(benv, [a[None] for a in acts])
check("gozlem, odul, bitis bit-es (2+ bolum sonu); grup 'agent' = ['agent_0'], gozlem N*d, eylem N",
      torch.equal(ow, ob[:, 0]) and torch.equal(rw, rb[:, 0]) and torch.equal(dw, db[:, 0]) and int(dw.sum()) >= 2
      and wenv.group_map == {'agent': ['agent_0']} == benv.group_map
      and ow.shape[-1] == N * make_env(CFG).observation_space('agent_0').shape[0]
      and wenv.full_action_spec['agent', 'action'].shape[-1] == N, f"bitis {int(dw.sum())}")

# ---------------------------------------------------------------------------------------------
print("\n[3] Esmimari: RowSharedPolicy <-> MARL paylasimli ajan MLP'si")
kw = dict(algo='mappo', seed=0, frames=200, frames_per_batch=200, epochs=1, n_envs=1)
em_x, es_x = R.build_experiment(CFG, **kw), R.build_experiment(SCFG, **kw, model=ESM)
pm, ps = list(em_x.policy.parameters()), list(es_x.policy.parameters())
same_shape = [tuple(p.shape) for p in pm] == [tuple(p.shape) for p in ps] or \
    [tuple(p.squeeze(0).shape) for p in pm] == [tuple(p.shape) for p in ps]
with torch.no_grad():
    for a, b in zip(pm, ps):
        b.copy_(a.reshape(b.shape))
tdm = PettingZooWrapper(env=make_env(CFG), seed=5, categorical_actions=True, return_state=False, use_mask=False).reset()
tds = PettingZooWrapper(env=make_env(SCFG), seed=5, categorical_actions=True, return_state=False, use_mask=False).reset()
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    outm, outs = em_x.policy(tdm.clone()), es_x.policy(tds.clone())
loc_eq = torch.allclose(outm['agent', 'loc'].reshape(-1), outs['agent', 'loc'].reshape(-1), atol=1e-6)
sc_eq = torch.allclose(outm['agent', 'scale'].reshape(-1), outs['agent', 'scale'].reshape(-1), atol=1e-6)
check("parametre sekilleri ayni (d -> 256 -> 256 -> 2); ayni agirlikla loc ve scale MARL ajan basina ciktiya esit",
      same_shape and loc_eq and sc_eq and torch.equal(tds['agent', 'observation'].reshape(-1),
                                                     tdm['agent', 'observation'].reshape(-1)),
      f"{[tuple(p.shape) for p in pm]} {[tuple(p.shape) for p in ps]} {loc_eq} {sc_eq}")

# ---------------------------------------------------------------------------------------------
print("\n[4] Kisa MAPPO: SARL-agirlik ve SARL-esmimari")
build = dict(algo='mappo', seed=4, frames=400, frames_per_batch=200, epochs=2, n_envs=2, parallel=False, init_std=0.12)
ok4, info4 = True, []
for name, model in (('agirlik', None), ('esmimari', ESM)):
    e1 = R.build_experiment(SCFG, **build, model=model)   # kur -> kos -> kur -> kos: Experiment kurulumda kuresel
    e1.run()                                              # RNG'yi tohumlar (test_marl_m3 [3] ile ayni sira)
    e2 = R.build_experiment(SCFG, **build, model=model)
    e2.run()
    ok4 &= e1.total_frames >= 400 and digest(e1) == digest(e2)
    info4.append((name, e1.total_frames, sum(p.numel() for p in e1.policy.parameters())))
eb = R.build_experiment({**SCFG, 'batched': True}, **build, model=ESM)
eb.run()
rd = os.path.join(tmp, 'run_esm')
R.save_policy(e1, SCFG, {'build': {**build, 'model': ESM}, 'policy_sha256': digest(e1)}, rd)
el, meta = R.load_policy(rd)
ser = R.deterministic_rollout(el.policy, {**SCFG, 'mode': 'eval', 'start_row': 180, 'end_row': 259,
                                          'norm_stats_path': os.path.join(rd, 'norm_stats.json')})
check("iki kol tamamlandi, ayni tohum -> ayni politika; toplu yol egitimde calisir; kaydet/yukle ayni; deterministik "
      "degerlendirme tam pencere (79 adim), X sonlu; meta 'sarl'",
      ok4 and eb.total_frames >= 400 and digest(el) == digest(e1) and len(ser['X']) == 79
      and bool(np.all(np.isfinite(ser['X']))) and meta['model_meta']['kind'] == 'sarl', f"{info4}")

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
