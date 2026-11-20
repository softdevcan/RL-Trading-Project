"""
Faz 10 M5 S-msg: guvenle agirlikli dikkat (`marl/comm_models.py` conf_mode). YALNIZ venv-marl. Sentetik panel, CPU.

Kontroller:
  1. Geriye uyum: conf_mode='none' ciktisi M4a'yi koşan commit'teki CommPolicy (git show 62b4023) ile bit-es;
     beta = 0 sabit (prob) de ayni ciktiyi verir (guvensiz dikkat = attn/gat).
  2. Guven terimi: prob = log clip(0,5 + |p - 0,5|), zlogodds = |logit p| / z_scale; tahminsiz gonderici en dusuk
     degeri alir; gat maskesi guvenle de korunur (komsu olmayan etkisiz).
  3. Mekanizma: beta > 0 iken gondericinin guveni artinca ona verilen dikkat agirligi artar; karistirilmis guven
     sabit bir derangement ile b_j <- b_pi(j); gecit baslangicta ~0,99, a > 0 iken dusuk oz-guven daha buyuk g.
  4. H3 tasarim bulgusu: zlogodds, z_scale ayni veriden olculunce kuresel sicaklik bozulmasina DEGISMEZ
     (b ayni) -> sicaklikla bozulmus "kalibresiz" kol bu modda tanim geregi ayni girdiyi alir; prob modu degisir;
     sembol basina farkli sicaklik zlogodds'ta da gondericilerin sirasini degistirir.
  5. Parametre sayisi (+1 ogrenilen beta, +2 gecit); kisa MAPPO (zlogodds + ogrenilen beta + gecit, tahmin
     bloklu ortam) tamamlanir, ayni tohum -> ayni politika ozeti.
  6. GLM gonderici sicakligi (M5 tasarimi §4; yalniz guven terimi): T_j log-tekduze [0,25; 4] tohumlu ve
     deterministik; b_j = T_j |logit p_j| / z_scale; beta = 0 iken cikti sicakliksiz modelle ayni (icerik
     degismez); egitici (`train_m3.conf_fields`) guven sutunlarini gozlem duzeninden, z_scale'i egitim
     satirlarindan (GLM'de bozulmus terimden) kurar; conf_mode olmadan guven bayragi reddedilir.

    venv-marl\\Scripts\\python tests/test_marl_m5_msg.py
"""
import hashlib
import importlib.util
import os
import subprocess
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
from tensordict import TensorDict  # noqa: E402

from _marl_fixtures import synth_arrays  # noqa: E402
from marl import obs_spec as OS  # noqa: E402
from marl import panel as PN  # noqa: E402
from marl import m3_runner as R  # noqa: E402
from marl.comm_models import (GLM_TEMP_RANGE, CommPolicy, comm_param_count, conf_scale, confidence_term,  # noqa: E402
                               derangement, sender_temperatures)

M4A_COMMIT = '62b4023'
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


tmp = tempfile.mkdtemp(prefix='marl_m5_')
N, T = 4, 260
arr = synth_arrays(T=T, N=N, seed=0, with_rf=True)
panel_path = os.path.join(tmp, 'panel.npz')
PN.save_npz(arr, panel_path)
rng = np.random.default_rng(1)
pred_path = os.path.join(tmp, 'pred.npz')
np.savez(pred_path, dates=arr['dates'], symbols=arr['symbols'], p_up_cal=rng.uniform(0.3, 0.7, (T, N)),
         pred_logret=rng.normal(0, 0.01, (T, N)), model_disagreement=rng.uniform(0, 0.01, (T, N)))
CFG = dict(panel_path=panel_path, start_row=OS.WARMUP_ROWS, end_row=180, mode='train', alpha=0.25,
           cost_bps_roundtrip=0.0, reward_term='excess', reward_scale=100.0, episode_len=20, limit_down_gap=0.095,
           pred_path=pred_path)
EDGES = [[0, 1], [1, 2]]
BUILD = dict(algo='mappo', frames=400, frames_per_batch=200, epochs=2, n_envs=2, parallel=False, init_std=0.12)
F_IN = len(OS.FEATURES_V1) + 3 + N + 4          # z, feat_ok, w_i, w_nakit, kimlik, S-obs blogu
CI, OKI = F_IN - 4, F_IN - 1                     # p_up_cal - 0,5 ve pred_ok sutunlari


def spec(kind='attn', **kw):
    return dict(kind=kind, hidden=16, heads=2, rounds=1, edges=(EDGES if kind == 'gat' else None), **kw)


def conf(**kw):
    return dict(conf_index=CI, conf_ok_index=OKI, **kw)


def module(exp):
    return next(m for m in exp.policy.modules() if isinstance(m, CommPolicy))


def out(mod, x):
    td = TensorDict({mod.in_keys[0]: x.clone()}, batch_size=[x.shape[0]])
    with torch.no_grad():
        return mod(td).get(mod.out_key)


def digest(exp):
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


torch.manual_seed(0)
x = torch.randn(3, N, F_IN)
x[..., CI] = torch.tensor([[0.2, -0.1, 0.05, 0.0]]).expand(3, N)
x[..., OKI] = 1.0

# ---------------------------------------------------------------------------------------------
print("\n[1] Geriye uyum")
src = subprocess.run(['git', 'show', f'{M4A_COMMIT}:marl/comm_models.py'], capture_output=True, text=True,
                     encoding='utf-8').stdout
old_path = os.path.join(tmp, 'comm_models_m4a.py')
with open(old_path, 'w', encoding='utf-8') as fh:
    fh.write(src)
spec_old = importlib.util.spec_from_file_location('comm_models_m4a', old_path)
old = importlib.util.module_from_spec(spec_old)
sys.modules['comm_models_m4a'] = old
spec_old.loader.exec_module(old)
for kind in ('attn', 'gat'):
    e_new = R.build_experiment(CFG, seed=0, model=spec(kind), **BUILD)
    m_new = module(e_new)
    m_old = old.CommPolicy(kind=kind, hidden=16, heads=2, rounds=1, edges=(EDGES if kind == 'gat' else None),
                           input_spec=m_new.input_spec, output_spec=m_new.output_spec, agent_group=m_new.agent_group,
                           input_has_agent_dim=True, n_agents=N, centralised=False, share_params=True,
                           device=m_new.device, action_spec=m_new.action_spec, model_index=0, is_critic=False)
    m_old.load_state_dict(m_new.state_dict())
    check(f"{kind}: conf_mode='none' ciktisi {M4A_COMMIT} CommPolicy ile bit-es", torch.equal(out(m_new, x), out(m_old, x)))
    e_b0 = R.build_experiment(CFG, seed=0, model=spec(kind, **conf(conf_mode='prob')), **BUILD)
    m_b0 = module(e_b0)
    m_b0.load_state_dict(m_new.state_dict(), strict=False)
    check(f"{kind}: beta = 0 sabit (prob) = guvensiz dikkat", torch.allclose(out(m_b0, x), out(m_new, x), atol=1e-6))

# ---------------------------------------------------------------------------------------------
print("\n[2] Guven terimi")
pm = torch.tensor([0.2, -0.1, 0.05, 0.0, 0.499])
ok = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])
bp = confidence_term(pm, ok, 'prob', 0.5, 0.9, 1.0)
check("prob: log clip(0,5 + |p - 0,5|), ust kirpma, tahminsiz -> log c_min",
      torch.allclose(bp, torch.log(torch.tensor([0.7, 0.6, 0.55, 0.5, 0.9]))))
bz = confidence_term(pm, ok, 'zlogodds', 0.5, 0.9, 2.0)
lz = torch.log((0.5 + pm) / (0.5 - pm)).abs() / 2.0
check("zlogodds: |logit p| / z_scale, tahminsiz -> 0", torch.allclose(bz[[0, 1, 2, 4]], lz[[0, 1, 2, 4]]) and bz[3] == 0)
eg = R.build_experiment(CFG, seed=0, model=spec('gat', **conf(conf_mode='prob', beta_init=2.0)), **BUILD)
mg = module(eg)
x_far = x.clone()
x_far[:, 3] += 3.0
x_far[:, 3, CI] = 0.45                                # uzak ajanin guveni de degisiyor
check("gat + guven: komsu olmayanin gozlemi/guveni ajan 0 ciktisini degistirmez",
      torch.equal(out(mg, x)[:, 0], out(mg, x_far)[:, 0]))

# ---------------------------------------------------------------------------------------------
print("\n[3] Mekanizma")
ea = R.build_experiment(CFG, seed=0, model=spec('attn', **conf(conf_mode='prob', beta_init=3.0)), **BUILD)
ma = module(ea)


def weight_to(mod, xx, j):
    xf = xx.reshape(-1, N, F_IN)
    h = mod.enc(xf)
    mask = mod._score_mask(mod.conf_terms(xf))
    with torch.no_grad():
        _, w = mod.attn[0](h, h, h, attn_mask=mask, need_weights=True)
    return w[:, :, j].mean().item()


x_hi = x.clone()
x_hi[..., 2, CI] = 0.45
check("beta > 0: gondericinin guveni artinca ona verilen dikkat artar", weight_to(ma, x_hi, 2) > weight_to(ma, x, 2))
es = R.build_experiment(CFG, seed=0, model=spec('attn', **conf(conf_mode='prob', beta_init=1.0, shuffle_conf=True,
                                                                shuffle_seed=7)), **BUILD)
ms = module(es)
perm = derangement(N, 7)
plain = confidence_term(x[..., CI], x[..., OKI], 'prob', 0.5, 0.99, 1.0)
check("karistirilmis guven: b_j <- b_pi(j), pi derangement",
      torch.equal(ms.conf_terms(x), plain[..., perm]) and not bool((perm == torch.arange(N)).any()))
egt = R.build_experiment(CFG, seed=0, model=spec('attn', **conf(conf_mode='prob', beta_mode='learned', gate=True)),
                         **BUILD)
mgt = module(egt)
g0 = mgt._gate(x)
check("gecit baslangicta ~0,99 (a = 0, b = 4,6)", bool(((g0 > 0.98) & (g0 < 1.0)).all()))
with torch.no_grad():
    mgt.gate_a.fill_(5.0)
    mgt.gate_b.fill_(0.0)
g = mgt._gate(x)[0, :, 0]
check("a > 0: dusuk oz-guven (ajan 3, p = 0,5) en buyuk g, yuksek (ajan 0) en kucuk",
      int(g.argmax()) == 3 and int(g.argmin()) == 0)

# ---------------------------------------------------------------------------------------------
print("\n[4] H3 tasarim bulgusu: standartlastirma sicakligi siler")
logit = torch.tensor(np.random.default_rng(3).normal(0, 0.3, (500, N)))
p_cal = torch.sigmoid(logit)
p_hot = torch.sigmoid(logit * 2.5)                      # kuresel sicaklik bozulmasi (asiri guven)
z_cal, z_hot = logit.abs().std().item(), (2.5 * logit).abs().std().item()
ones = torch.ones_like(logit)
b_cal = confidence_term(p_cal - 0.5, ones, 'zlogodds', 0.5, 0.99, z_cal)
b_hot = confidence_term(p_hot - 0.5, ones, 'zlogodds', 0.5, 0.99, z_hot)
check("zlogodds: kuresel sicaklik bozulmasi b'yi DEGISTIRMEZ (z_scale ayni veriden)",
      torch.allclose(b_cal, b_hot, atol=1e-6))
pb_cal = confidence_term(p_cal - 0.5, ones, 'prob', 0.5, 0.99, 1.0)
pb_hot = confidence_term(p_hot - 0.5, ones, 'prob', 0.5, 0.99, 1.0)
check("prob: ayni bozulma b'yi degistirir", not torch.allclose(pb_cal, pb_hot, atol=1e-3))
temps = torch.tensor([0.5, 1.0, 2.0, 4.0])              # sembol basina farkli sicaklik
p_het = torch.sigmoid(logit * temps)
z_het = (logit * temps).abs().std().item()
b_het = confidence_term(p_het - 0.5, ones, 'zlogodds', 0.5, 0.99, z_het)
rank_cal, rank_het = b_cal.argsort(dim=1), b_het.argsort(dim=1)
check("zlogodds: sembol basina farkli sicaklik gondericilerin sirasini degistirir",
      float((rank_cal != rank_het).any(dim=1).float().mean()) > 0.5)

# ---------------------------------------------------------------------------------------------
print("\n[5] Parametre ve kisa MAPPO")
n_in = module(egt).enc[0].in_features
pg = sum(p.numel() for p in mgt.parameters())
check("ogrenilen beta + gecit: +3 parametre",
      pg == comm_param_count(n_in, 2, 16, beta_learned=True, gate=True) == comm_param_count(n_in, 2, 16) + 3)
mspec = spec('gat', **conf(conf_mode='zlogodds', z_scale=0.5, beta_mode='learned', gate=True))
e1 = R.build_experiment(CFG, seed=5, model=mspec, **BUILD)
e1.run()
e2 = R.build_experiment(CFG, seed=5, model=mspec, **BUILD)
e2.run()
check("gat + zlogodds + ogrenilen beta + gecit: kosu tamamlandi, ayni tohum -> ayni politika ozeti",
      e1.total_frames >= 400 and digest(e1) == digest(e2))
beta = module(e1).beta.detach().item()
check("ogrenilen beta egitimde degisti (gradyan akiyor)", beta != 0.0, f"beta = {beta}")

# ---------------------------------------------------------------------------------------------
print("\n[6] GLM gonderici sicakligi + egitici guven alanlari")
T = sender_temperatures(N, 6)
check("sicaklik: log-tekduze GLM_TEMP_RANGE [0,25; 4], N uzunlukta, ayni tohum -> ayni vektor, baska tohum -> farkli",
      len(T) == N and GLM_TEMP_RANGE == (0.25, 4.0) and all(0.25 <= t <= 4.0 for t in T) and T == sender_temperatures(N, 6) != sender_temperatures(N, 7))
zkw = dict(conf_mode='zlogodds', z_scale=0.7, beta_mode='learned', beta_init=1.0)
mt = module(R.build_experiment(CFG, seed=0, model=spec('gat', **conf(conf_temp=T, **zkw)), **BUILD))
mp = module(R.build_experiment(CFG, seed=0, model=spec('gat', **conf(**zkw)), **BUILD))
mp.load_state_dict(mt.state_dict())
pq = 0.5 + x[..., CI]
want = torch.log(pq / (1 - pq)).abs() * torch.tensor(T, dtype=x.dtype) / 0.7
changed = not torch.allclose(out(mt, x), out(mp, x))
with torch.no_grad():
    mt.beta.fill_(0.0)
    mp.beta.fill_(0.0)
check("GLM: b_j = T_j |logit p_j| / z (yalniz guven terimi); beta > 0 ciktiyi degistirir, beta = 0 iken "
      "cikti sicakliksiz modelle ayni (icerik degismez)",
      torch.allclose(mt.conf_terms(x), want, atol=1e-5) and changed and torch.equal(out(mt, x), out(mp, x)))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
import argparse  # noqa: E402
import train_m3 as TM  # noqa: E402

parr = PN.load_npz(panel_path)
pr = np.load(pred_path)
okp = np.isfinite(pr['p_up_cal'])
ns = dict(comm='gat', pred_path=pred_path, conf_mode='zlogodds', beta_mode='learned', beta_init=0.0, gate=False,
          shuffle_conf=False, shuffle_seed=None, conf_temp_seed=None, seed=3)
cf = TM.conf_fields(argparse.Namespace(**ns), parr, 180)
cm = TM.conf_fields(argparse.Namespace(**{**ns, 'conf_temp_seed': 6}), parr, 180)
rows = slice(OS.WARMUP_ROWS, 181)
try:
    TM.conf_fields(argparse.Namespace(**{**ns, 'conf_mode': 'none', 'gate': True}), parr, 180)
    rejected = False
except SystemExit:
    rejected = True
check("egitici: conf_index/ok = gozlem sutunlari, z_scale egitim satirlarindan, GLM bozulmus terimden; "
      "conf_mode'suz bayrak reddedilir",
      (cf['conf_index'], cf['conf_ok_index']) == (CI, OKI) and cf['shuffle_seed'] == 3 and cf['conf_temp'] is None
      and abs(cf['z_scale'] - conf_scale(pr['p_up_cal'][rows], okp[rows])) < 1e-12
      and abs(cm['z_scale'] - conf_scale(pr['p_up_cal'][rows], okp[rows], T)) < 1e-12
      and cm['conf_temp'] == T and cm['z_scale'] != cf['z_scale'] and rejected)

print("\n" + "=" * 60)
print(f"Sonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
