"""
Faz 10 M3: BenchMARL gorevi (`marl/benchmarl_task.py`). YALNIZ venv-marl'da kosar. Sentetik panel, ag yok, CPU.

Kontroller:
  1. Gorev: tek grup 'agent' (N ajan), gozlem [N, local_dim], eylem [N, 1]; bilinmeyen anahtar -> hata;
     tahmin blogu panel ile hizali degilse -> hata.
  2. Bolum sonu (M2 §9 #10, hakem-rl R6): bolum sonunda grup duzeyinde terminated=False, truncated=True, done=True;
     TorchRL GAE terminated=False iken V(s_T)'yi bootstrap eder (terminated=True'da etmez) — iki durumun farki
     gamma * V(s_T); BenchMARL MAPPO kaybi grup `terminated` anahtarina bagli.
  3. Kisa MAPPO kosusu (use_state False / True) tamamlanir; ayni tohumla iki kosu ayni politika ozetini verir.
  4. M3 kosucusu: politika kaydet/yukle -> ayni agirlik; deterministik degerlendirme iki kez bit-es, tam pencere,
     egitim norm_stats'i ile; baslangic std'si (R7) olcek eslemesine gecer; baseline'lar ayni muhasebeyle
     (nakit X = 0, al-tut ilk gunden sonra devir 0, 1/N sonlu ve tam yatirimli).

    venv-marl\\Scripts\\python tests/test_marl_m3.py
"""
import hashlib
import os
import sys
import tempfile

# Bit-eslik icin MKL/OpenMP tek is parcacigi, torch/numpy import'undan ONCE (torch.set_num_threads surec
# ortasinda yetmiyor: 2026-11-16 olcumu, 2 guncellemede 1,9e-9 agirlik farki)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _marl_fixtures import synth_arrays  # noqa: E402
from marl import obs_spec as OS  # noqa: E402
from marl import panel as PN  # noqa: E402
from marl.benchmarl_task import MarlTask, load_pred, make_env  # noqa: E402

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


tmp = tempfile.mkdtemp(prefix='marl_m3_')
arr = synth_arrays(T=260, N=4, seed=0, with_rf=True)
panel_path = os.path.join(tmp, 'panel.npz')
PN.save_npz(arr, panel_path)
EP = 20
CFG = dict(panel_path=panel_path, start_row=OS.WARMUP_ROWS, end_row=180, mode='train', alpha=0.25,
           cost_bps_roundtrip=20.0, reward_term='excess', reward_scale=100.0, episode_len=EP,
           limit_down_gap=0.095)   # sentetik tarihler v0.7 takvimi disinda

# ---------------------------------------------------------------------------------------------
print("\n[1] Gorev")
task = MarlTask.PORTFOLIO.update_config(dict(CFG))
env = task.get_env_fun(1, True, 0, 'cpu')()
check("tek grup 'agent', 4 ajan", list(task.group_map(env)) == ['agent'] and len(env.group_map['agent']) == 4)
obs_shape = tuple(task.observation_spec(env)['agent', 'observation'].shape)
check("gozlem [N, local_dim], eylem [N, 1]",
      obs_shape == (4, make_env(CFG).spec.local_dim) and tuple(task.action_spec(env)['agent', 'action'].shape) == (4, 1),
      str(obs_shape))
check("max_steps = episode_len (egitim)", task.max_steps(env) == EP)
check("bilinmeyen gorev anahtari -> ValueError", raises(lambda: make_env({**CFG, 'alfa': 0.3})))
bad = os.path.join(tmp, 'pred_bad.npz')
np.savez(bad, dates=arr['dates'][:-1], symbols=arr['symbols'], p_up_cal=np.zeros((259, 4)),
         pred_logret=np.zeros((259, 4)), model_disagreement=np.zeros((259, 4)))
check("tahmin blogu panel ile hizali degil -> ValueError", raises(lambda: load_pred(bad, PN.load_npz(panel_path))))
ex = np.random.default_rng(7).normal(size=(260, 2))
ex_path = os.path.join(tmp, 'sreg.npz')
np.savez(ex_path, dates=arr['dates'], names=np.array(['sma_ratio', 'vol']), extras=ex)
ex_bad = os.path.join(tmp, 'sreg_bad.npz')
np.savez(ex_bad, dates=arr['dates'][:-1], names=np.array(['a', 'b']), extras=ex[:-1])
e0, e1 = make_env(CFG), make_env({**CFG, 'extras_path': ex_path})
o0, o1 = e0.reset(seed=0)[0], e1.reset(seed=0)[0]
ag = e1.possible_agents
same_head = all(np.array_equal(o0[a], o1[a][:-2]) for a in ag)
tail_ok = all(np.allclose(o1[a][-2:], ex[e1.t].astype(np.float32)) for a in ag)
check("M6b yayin blogu (extras_path): yerel gozlem +2 boyut, bas kismi bloksuzla bit-es, son 2 sutun her ajanda o gunun "
      "S-reg degeri; hizasiz blok -> ValueError",
      e1.spec.local_dim == e0.spec.local_dim + 2 and same_head and tail_ok
      and raises(lambda: make_env({**CFG, 'extras_path': ex_bad})), f"{e0.spec.local_dim} {e1.spec.local_dim}")

# ---------------------------------------------------------------------------------------------
print("\n[2] Bolum sonu ve bootstrap")
td = env.rollout(EP + 5, break_when_any_done=False)
term = td['next', 'agent', 'terminated'].squeeze(-1)
trunc = td['next', 'agent', 'truncated'].squeeze(-1)
done = td['next', 'agent', 'done'].squeeze(-1)
check("bolum sonu (adim EP): truncated=True, done=True, terminated=False (tum ajanlar)",
      bool(trunc[EP - 1].all() and done[EP - 1].all() and not term[EP - 1].any()), f"{trunc[EP - 1]} {term[EP - 1]}")
check("bolum icinde done=False; hicbir adimda terminated=True yok",
      bool(not done[:EP - 1].any() and not term.any()))

from torchrl.objectives.value.functional import generalized_advantage_estimate as gae  # noqa: E402
T1 = 3
r = torch.zeros(T1, 1)
v = torch.ones(T1, 1)
nv = torch.ones(T1, 1) * 2.0
d = torch.tensor([[False], [False], [True]])
adv_trunc, tgt_trunc = gae(0.9, 0.95, v, nv, r, d, torch.zeros_like(d))      # truncation: terminated=False
adv_term, tgt_term = gae(0.9, 0.95, v, nv, r, d, d.clone())                  # gercek son durum
check("GAE: son adimda terminated=False -> hedef = r + gamma V(s_T) (bootstrap); True -> hedef = r",
      abs(float(tgt_trunc[-1]) - 0.9 * 2.0) < 1e-6 and abs(float(tgt_term[-1])) < 1e-6,
      f"{tgt_trunc[-1]} {tgt_term[-1]}")

from benchmarl.algorithms import MappoConfig  # noqa: E402
from benchmarl.experiment import Experiment, ExperimentConfig  # noqa: E402
from benchmarl.models.mlp import MlpConfig  # noqa: E402


def experiment(seed, use_state=False, frames=400):
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_frames = frames
    cfg.on_policy_collected_frames_per_batch = 200
    cfg.on_policy_n_envs_per_worker = 2
    cfg.on_policy_minibatch_size = 100
    cfg.on_policy_n_minibatch_iters = 2
    cfg.parallel_collection = False
    cfg.sampling_device = cfg.train_device = cfg.buffer_device = 'cpu'
    cfg.share_policy_params = True
    cfg.evaluation = False
    cfg.render = False
    cfg.loggers = []
    cfg.create_json = False
    cfg.checkpoint_at_end = False
    cfg.save_folder = tempfile.mkdtemp(prefix='marl_m3_exp_')
    t = MarlTask.PORTFOLIO.update_config({**CFG, 'use_state': use_state})
    return Experiment(task=t, algorithm_config=MappoConfig.get_from_yaml(), model_config=MlpConfig.get_from_yaml(),
                      critic_model_config=MlpConfig.get_from_yaml(), seed=seed, config=cfg)


def digest(exp):
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


e0 = experiment(0)
keys = e0.losses['agent'].tensor_keys
check("MAPPO kaybi grup terminated/done anahtarlarina bagli (bootstrap terminated'a gore)",
      keys.terminated == ('agent', 'terminated') and keys.done == ('agent', 'done'), f"{keys}")

# ---------------------------------------------------------------------------------------------
print("\n[3] Kisa MAPPO kosusu ve tohum")
e0.run()
check("MAPPO (elestirmen = tum ajan gozlemleri) tamamlandi", e0.total_frames >= 400, str(e0.total_frames))
es = experiment(0, use_state=True)
es.run()
check("MAPPO (elestirmen = ortam state()) tamamlandi", es.total_frames >= 400)
e1 = experiment(0)
e1.run()
check("ayni tohum, iki CPU kosusu -> ayni politika ozeti", digest(e0) == digest(e1))

# ---------------------------------------------------------------------------------------------
print("\n[4] M3 kosucusu: kaydet/yukle, deterministik degerlendirme, baseline")
from marl import m3_runner as R  # noqa: E402
build = dict(algo='mappo', seed=3, frames=400, frames_per_batch=200, epochs=2, n_envs=2, parallel=False,
             init_std=0.12)
er = R.build_experiment(CFG, **build)
check("baslangic std'si olcek eslemesine gecer (biased_softplus_0.12)",
      er.algorithm_config.scale_mapping == 'biased_softplus_0.12', str(er.algorithm_config.scale_mapping))
from torchrl.envs.utils import ExplorationType, set_exploration_type  # noqa: E402


def sampled_logit_std(init_std):
    ex = R.build_experiment(CFG, **{**build, 'init_std': init_std})
    td0 = ex.task.get_env_fun(1, True, 0, 'cpu')().reset()
    with set_exploration_type(ExplorationType.RANDOM):
        acts = torch.stack([ex.policy(td0.clone())['agent', 'action'] for _ in range(200)])
    return float(acts.std(0).mean())


_s1, _s012 = sampled_logit_std(1.0), sampled_logit_std(0.12)
check("R7: baslangic gurultusu varsayilan std 1,0'da > 2,5 logit (~tam devir), 0,12'de < 1,5 logit",
      _s1 > 2.5 and _s012 < 1.5, f"{_s1:.2f} {_s012:.2f}")
er.run()
rd = os.path.join(tmp, 'run')
R.save_policy(er, CFG, {'build': build, 'policy_sha256': digest(er)}, rd)
el, meta = R.load_policy(rd)
check("kaydet/yukle: ayni politika ozeti", digest(el) == digest(er))
ecfg = {**CFG, 'mode': 'eval', 'start_row': 180, 'end_row': 259,
        'norm_stats_path': os.path.join(rd, 'norm_stats.json')}
s1 = R.deterministic_rollout(el.policy, ecfg)
s2 = R.deterministic_rollout(el.policy, ecfg)
check("deterministik degerlendirme: iki kosu bit-es, tam pencere (79 adim), X sonlu",
      np.array_equal(s1['X'], s2['X']) and len(s1['X']) == 79 and np.all(np.isfinite(s1['X'])), str(len(s1['X'])))
check("degerlendirme ortami 'fit' kabul etmez (held-out olcek egitimden)",
      raises(lambda: R.deterministic_rollout(el.policy, {**ecfg, 'norm_stats_path': None})))
_pa = PN.load_npz(panel_path)
bl = {k: R.rule_series(_pa, 120, 200, k, 20.0, limit_down_gap=0.095) for k in ('ew', 'bh', 'cash')}
check("baseline: nakit X = 0, al-tut ilk gunden sonra devir 0, 1/N X sonlu ve tam yatirimli",
      np.abs(bl['cash']['X']).max() < 1e-15 and np.abs(bl['bh']['turnover'][1:]).max() < 1e-12
      and np.all(np.isfinite(bl['ew']['X'])) and bl['ew']['cash_weight'].max() < 1e-12,
      f"{np.abs(bl['cash']['X']).max()} {np.abs(bl['bh']['turnover'][1:]).max()} {bl['ew']['cash_weight'].max()}")

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
