"""
Faz 10 M4: iletisim kollu politika (`marl/comm_models.py`) + M3 kosucusuna baglanti. YALNIZ venv-marl. Sentetik panel, CPU.

Kontroller:
  1. Graf maskesi: simetrik, kendine dikkat acik, gecersiz kenar -> hata; attn kenar almaz, gat kenarsiz -> hata.
  2. Esit butce (plan §3.2.5): kurulan politikanin parametre sayisi = formul; attn = gat; M4 boyutlarinda
     (gozlem 48, hidden 110, heads 5) none MLP'ye gore |fark| <= %10.
  3. Mekanizma: gat'te komsu OLMAYAN ajanin gozlemi degisince ajan i'nin ciktisi bit-es kalir; attn'de degisir;
     komsunun gozlemi degisince gat ciktisi degisir.
  4. Kisa MAPPO kosusu (attn, gat) tamamlanir; ayni tohumla iki CPU kosusu ayni politika ozeti; kaydet/yukle ayni
     agirlik ve model alani kayitta.

    venv-marl\\Scripts\\python tests/test_marl_m4.py
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

import torch  # noqa: E402

from _marl_fixtures import synth_arrays  # noqa: E402
from marl import obs_spec as OS  # noqa: E402
from marl import panel as PN  # noqa: E402
from marl import m3_runner as R  # noqa: E402
from marl.comm_models import CommPolicy, comm_param_count, graph_mask, mlp_param_count  # noqa: E402

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


tmp = tempfile.mkdtemp(prefix='marl_m4_')
N = 4
arr = synth_arrays(T=260, N=N, seed=0, with_rf=True)
panel_path = os.path.join(tmp, 'panel.npz')
PN.save_npz(arr, panel_path)
CFG = dict(panel_path=panel_path, start_row=OS.WARMUP_ROWS, end_row=180, mode='train', alpha=0.25,
           cost_bps_roundtrip=0.0, reward_term='excess', reward_scale=100.0, episode_len=20, limit_down_gap=0.095)
EDGES = [[0, 1], [1, 2]]          # 3 ajan 0'in komsusu degil; 1 komsu
BUILD = dict(algo='mappo', frames=400, frames_per_batch=200, epochs=2, n_envs=2, parallel=False, init_std=0.12)


def model_spec(kind, hidden=16, heads=2):
    return dict(kind=kind, hidden=hidden, heads=heads, rounds=1, edges=(EDGES if kind == 'gat' else None))


def comm_module(exp):
    return next(m for m in exp.policy.modules() if isinstance(m, CommPolicy))


def digest(exp):
    h = hashlib.sha256()
    for name, p in sorted(exp.policy.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
print("\n[1] Graf maskesi ve arguman denetimi")
m = graph_mask(N, EDGES)
check("maske simetrik, kendine dikkat acik", bool((m == m.T).all()) and not bool(m.diagonal().any()))
check("kenar acik, kenar disi engelli", (not m[0, 1]) and (not m[2, 1]) and bool(m[0, 3]) and bool(m[0, 2]))
check("gecersiz kenar (oz-dongu, aralik disi) -> hata",
      raises(lambda: graph_mask(N, [[1, 1]])) and raises(lambda: graph_mask(N, [[0, N]])))
check("attn'e kenar -> hata; gat kenarsiz -> hata",
      raises(lambda: R.build_experiment(CFG, seed=0, model=dict(model_spec('attn'), edges=EDGES), **BUILD))
      and raises(lambda: R.build_experiment(CFG, seed=0, model=dict(model_spec('gat'), edges=None), **BUILD)))

# ---------------------------------------------------------------------------------------------
print("\n[2] Esit butce")
ea = R.build_experiment(CFG, seed=0, model=model_spec('attn'), **BUILD)
eg = R.build_experiment(CFG, seed=0, model=model_spec('gat'), **BUILD)
ca, cg = comm_module(ea), comm_module(eg)
n_in = ca.enc[0].in_features
pa = sum(p.numel() for p in ca.parameters())
pg = sum(p.numel() for p in cg.parameters())
check("parametre sayisi = formul", pa == comm_param_count(n_in, 2, 16), f"{pa} vs {comm_param_count(n_in, 2, 16)}")
check("attn = gat parametre sayisi", pa == pg, f"{pa} vs {pg}")
none_m4, comm_m4 = mlp_param_count(48, 2), comm_param_count(48, 2, 110)
check("M4 boyutlari: |comm/none - 1| <= %10", abs(comm_m4 / none_m4 - 1) <= 0.10, f"{comm_m4} / {none_m4}")

# ---------------------------------------------------------------------------------------------
print("\n[3] Mekanizma: graf maskesi bilgi akisini sinirlar")
eg.policy.load_state_dict(ea.policy.state_dict(), strict=False)   # ayni agirlik; yalniz maske farkli
torch.manual_seed(0)
obs = torch.randn(1, N, n_in)


def out(mod, x):
    from tensordict import TensorDict
    td = TensorDict({mod.in_keys[0]: x.clone()}, batch_size=[1])
    with torch.no_grad():
        return mod(td).get(mod.out_key)[0]


x_far = obs.clone()
x_far[0, 3] += 3.0     # 3, 0'in komsusu degil
x_nb = obs.clone()
x_nb[0, 1] += 3.0      # 1, 0'in komsusu
check("gat: komsu olmayanin gozlemi degisince ajan 0 ciktisi bit-es", torch.equal(out(cg, obs)[0], out(cg, x_far)[0]))
check("gat: komsunun gozlemi degisince ajan 0 ciktisi degisir", not torch.equal(out(cg, obs)[0], out(cg, x_nb)[0]))
check("attn: herhangi bir ajanin gozlemi ajan 0'i etkiler", not torch.equal(out(ca, obs)[0], out(ca, x_far)[0]))

# ---------------------------------------------------------------------------------------------
print("\n[4] Kisa MAPPO, tohum, kaydet/yukle")
for kind in ('attn', 'gat'):
    e1 = R.build_experiment(CFG, seed=5, model=model_spec(kind), **BUILD)
    e1.run()
    e2 = R.build_experiment(CFG, seed=5, model=model_spec(kind), **BUILD)
    e2.run()
    check(f"{kind}: kosu tamamlandi, ayni tohum -> ayni politika ozeti",
          e1.total_frames >= 400 and digest(e1) == digest(e2))
    d = os.path.join(tmp, f'run_{kind}')
    build = dict(BUILD, seed=5, model=model_spec(kind), lr=None)
    meta = R.save_policy(e1, CFG, {'build': build}, d)
    el, ml = R.load_policy(d)
    check(f"{kind}: kaydet/yukle ayni agirlik, model alani kayitta",
          digest(el) == digest(e1) and ml['build']['model']['kind'] == kind and meta['build']['model'] == build['model'])

print("\n" + "=" * 60)
print(f"Sonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
