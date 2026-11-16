"""
Faz 10 M2 ortami (`marl/env.py`, PettingZoo Parallel). YALNIZ venv-marl'da kosar:
    venv-marl\\Scripts\\python tests/test_marl_env.py

Kontroller:
  1. PettingZoo `parallel_api_test`; uzay boyutlari spec ile ayni; logit siniri 5 + ln N.
  2. Acik hata: rf yok, norm_stats yok, held-out ortamda 'fit', alpha = 1, gecersiz pencere.
  3. Tohum: ayni tohum + ayni eylem dizisi -> ayni yorunge (odul, deger) bit-es.
  4. K2 paritesi: ortam gozlemi = `obs_spec.build_obs_at` (panelin t'ye kadar kesilmis hali, ortamin w'si).
  5. Muhasebe: deger = prod(1 + R); alpha = 0'da tum ajanlarin odulu = scale * X;
     eksik hucre gununde hisse agirligi islem gormeden tasinir.
  6. Held-out ortam egitim istatistigini aynen kullanir (norm_stats kimligi; B9/Y13 dersi).
  7. Tam sayi modu: nakit >= 0, adet >= 0, deger = nakit + adet x kapanis; buyuk sermayede
     getiri kesirli moda yakin (yuvarlama etkisi kucuk).
  8. 'differential' odul terimi sonlu; TorchRL PettingZooWrapper sarmalar ve rollout alir.

Ag yok. ~10 sn.
"""

import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings('ignore')

try:
    import pettingzoo  # noqa: F401
except ImportError:
    print("pettingzoo yok: bu test venv-marl ile kosar (requirements-marl.txt)")
    sys.exit(2)

from marl import obs_spec as OS  # noqa: E402
from marl.env import MarlPortfolioEnv  # noqa: E402
from marl.weights import Constraints  # noqa: E402
from tests._marl_fixtures import synth_arrays  # noqa: E402

passed = 0
failed = 0


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
        return False
    except exc:
        return True


arr = synth_arrays(T=260, N=4, seed=0, with_rf=True)
N = 4
TR = (OS.WARMUP_ROWS, 180)
EV = (180, 259)


def make(mode='train', stats='fit', rows=TR, **kw):
    return MarlPortfolioEnv(arr, start_row=rows[0], end_row=rows[1], norm_stats=stats, mode=mode, **kw)


def run(env, seed, actions_seed=7, steps=None):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(actions_seed)
    traj = []
    while env.agents:
        act = {a: rng.uniform(-env.bound, env.bound, 1).astype(np.float32) for a in env.agents}
        obs, rew, term, trunc, info = env.step(act)
        traj.append((env.last_info['value'], tuple(rew.values()), dict(env.last_info)))
        if steps and len(traj) >= steps:
            break
    return traj


# ---------------------------------------------------------------------------------------------
print("\n[1] API ve uzaylar")
from pettingzoo.test import parallel_api_test  # noqa: E402
env = make(episode_len=40)
try:
    parallel_api_test(env, num_cycles=50)
    check("PettingZoo parallel_api_test", True)
except Exception as e:  # noqa: BLE001
    check("PettingZoo parallel_api_test", False, repr(e)[:300])
env = make(episode_len=40)
obs, _ = env.reset(seed=0)
check("gozlem boyutu = spec.local_dim", all(o.shape == (env.spec.local_dim,) for o in obs.values()))
check("durum boyutu = spec.state_dim", env.state().shape == (env.spec.state_dim,))
check("eylem siniri 5 + ln N", abs(env.action_space('agent_0').high[0] - (5 + np.log(N))) < 1e-5)

# ---------------------------------------------------------------------------------------------
print("\n[2] Acik hatalar")
no_rf = {k: v for k, v in arr.items() if k != 'rf'}
check("rf yoksa hata (sabite dusulmez)",
      raises(lambda: MarlPortfolioEnv(no_rf, TR[0], TR[1], norm_stats='fit')))
check("norm_stats=None hata", raises(lambda: make(stats=None)))
check("held-out ortamda 'fit' hata", raises(lambda: make(mode='eval', stats='fit', rows=EV)))
check("alpha = 1 hata", raises(lambda: make(alpha=1.0)))
check("isinma baglami yoksa hata", raises(lambda: make(rows=(10, 100))))

# ---------------------------------------------------------------------------------------------
print("\n[3] Tohum")
t1 = run(make(episode_len=60), seed=11)
t2 = run(make(episode_len=60), seed=11)
t3 = run(make(episode_len=60), seed=12)
check("ayni tohum -> ayni yorunge (bit-es)", [x[:2] for x in t1] == [x[:2] for x in t2])
check("farkli tohum -> farkli bolum baslangici", t1[0][2]['date'] != t3[0][2]['date'] or t1[0][:2] != t3[0][:2])

# ---------------------------------------------------------------------------------------------
print("\n[4] K2 paritesi (ortam <-> canli gozlem)")
env = make(mode='train', episode_len=1000)
obs, _ = env.reset(seed=0)
rng = np.random.default_rng(0)
par = True
for step in range(60):
    t = env.t
    live = OS.build_obs_at(env.spec, {k: arr[k][:t + 1] for k in OS.REQUIRED_ARRAYS}, env.norm_stats, env.w)
    par &= np.array_equal(np.stack([obs[a] for a in env.possible_agents]), live)
    act = {a: rng.uniform(-env.bound, env.bound, 1).astype(np.float32) for a in env.possible_agents}
    obs, *_ = env.step(act)
check("60 adim: ortam gozlemi = build_obs_at (bit-es)", par)

# ---------------------------------------------------------------------------------------------
print("\n[5] Muhasebe")
env = make(mode='eval', stats=make().norm_stats, rows=(120, 200), alpha=0.0)
env.reset()
rng = np.random.default_rng(1)
vals, Rs, same_rew, w2_hist = [], [], True, []
while env.agents:
    act = {a: rng.uniform(-env.bound, env.bound, 1).astype(np.float32) for a in env.agents}
    t_before = env.t
    w_before = env.w.copy()
    _, rew, *_ = env.step(act)
    li = env.last_info
    vals.append(li['value'])
    Rs.append(li['R'])
    r = np.array(list(rew.values()))
    same_rew &= np.allclose(r, env.scale * li['X'], rtol=0, atol=1e-12)
    if t_before + 1 in (150, 151, 152):
        w2_hist.append((w_before[2], env.w[2], li['n_frozen']))
check("deger = prod(1 + R)", abs(vals[-1] - np.prod(1 + np.array(Rs))) < 1e-12)
check("alpha = 0: tum ajanlar ayni odul = scale * X", same_rew)
check("eksik hucre gunu: hisse 1 dondurulur, agirligi yalniz yeniden normalize olur",
      len(w2_hist) == 3 and all(n >= 1 for *_, n in w2_hist), str(w2_hist))
check("eval bolumu tum pencere: adim sayisi = end - start", len(vals) == 200 - 120)

# Listeleme kilidi: sembol 2 satir 90'da listelenir; 90-109 alinamaz, 110 acilisinda alinabilir
for _int in (False, True):
    el = make('eval', make().norm_stats, (85, 125), integer_shares=_int, initial_capital=1e9)
    el.reset(seed=0)
    w2 = {}
    while el.agents:
        act = {a: np.array([el.bound if a == 'agent_2' else -el.bound], np.float32) for a in el.agents}
        el.step(act)
        w2[el.t] = el.w[3]   # w[0] nakit; sembol 2 -> indeks 3
    check(f"kilit ({'tam sayi' if _int else 'kesirli'}): sembol 2 kilitte agirlik 0, kilit bitince alinir",
          all(w2[t] == 0 for t in range(86, 110)) and w2[110] > 0.5, f"w2[109]={w2[109]:.3f} w2[110]={w2[110]:.3f}")

# Satis engeli (§9 #9): sembol 0 satir 115 acilisinda -%12 bosluk; o gun satilamaz, 116'da satilir
_arr_g = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in arr.items()}
_arr_g['open'][115, 0] = arr['close'][114, 0] * 0.88
_L = 115
for _gap in (None, 0.0):
    for _int in (False, True):
        kw = {} if _gap is None else {'limit_down_gap': _gap}
        eg = MarlPortfolioEnv(_arr_g, 100, 125, norm_stats=make().norm_stats, mode='eval',
                              integer_shares=_int, initial_capital=1e9, **kw)
        eg.reset(seed=0)
        hist = {}
        while eg.agents:
            sell = eg.t + 1 >= _L
            act = {a: np.array([(-eg.bound if sell else eg.bound) if a == 'agent_0' else -eg.bound], np.float32)
                   for a in eg.agents}
            h0 = eg.holdings[0] if _int else None
            eg.step(act)
            hist[eg.t] = (eg.last_info['n_sell_blocked'], eg.w[1], h0, eg.holdings[0] if _int else None)
        nb, w1, hb, ha = hist[_L]
        lbl = 'tam sayi' if _int else 'kesirli'
        if _gap is None:
            kept = (hb == ha) if _int else w1 > 0.5
            check(f"satis engeli ({lbl}): taban acilista satilmaz, ertesi gun satilir",
                  nb == 1 and kept and hist[_L + 1][1] < 0.01, f"{hist[_L]} {hist[_L + 1]}")
        else:
            check(f"satis engeli kapali ({lbl}; kontrol kolu): ayni gun satilir",
                  nb == 0 and w1 < 0.01, f"{hist[_L]}")
check("model_meta satis engeli esigini tasir", make().model_meta()['limit_down_no_sell_gap'] == 0.095)

# ---------------------------------------------------------------------------------------------
print("\n[6] Held-out olcek")
tr_env = make()
ev_env = make(mode='eval', stats=tr_env.norm_stats, rows=EV)
check("held-out ortam egitim norm_stats'ini aynen kullanir", ev_env.norm_stats is tr_env.norm_stats
      and tr_env.norm_stats['fit_rows'] == list(TR))
meta = tr_env.model_meta()
check("model meta K1 alanlari", {'kind', 'action_type', 'symbols', 'obs_spec', 'norm_stats',
                                 'risk_projection', 'logit_bound', 'train_range'} <= set(meta)
      and meta['kind'] == 'marl' and meta['symbols'] == list(arr['symbols']))

# ---------------------------------------------------------------------------------------------
print("\n[7] Tam sayi modu")
stats = tr_env.norm_stats
def int_vs_frac(bps):
    ei = make(mode='eval', stats=stats, rows=EV, integer_shares=True, initial_capital=1e9, cost_bps_roundtrip=bps)
    ef = make(mode='eval', stats=stats, rows=EV, cost_bps_roundtrip=bps)
    ei.reset(); ef.reset()
    rng = np.random.default_rng(5)
    ok_inv, cons, diffs = True, True, []
    while ei.agents:
        act = {a: rng.uniform(-ei.bound, ei.bound, 1).astype(np.float32) for a in ei.agents}
        ei.step(act); ef.step(act)
        cf = ei.px['close'][ei.t]
        v_chk = ei.cash + float(np.dot(ei.holdings, np.where(np.isfinite(cf), cf, 0)))
        ok_inv &= ei.cash >= -1e-6 and bool(np.all(ei.holdings >= 0))
        cons &= abs(v_chk - ei.value) < 1e-9 * ei.value
        diffs.append(abs(ei.last_info['R'] - ef.last_info['R']))
    return ok_inv, cons, max(diffs)


ok0, cons0, d0 = int_vs_frac(0.0)
ok20, cons20, d20 = int_vs_frac(20.0)
check("tam sayi: nakit >= 0, adet >= 0", ok0 and ok20)
check("tam sayi: deger = nakit + adet x kapanis", cons0 and cons20)
check("maliyetsiz, 1e9 sermaye: tam sayi = kesirli (yalniz yuvarlama, < 1e-6)", d0 < 1e-6, f"max {d0:.2e}")
check("20 bps: fark maliyet yaklasikligi mertebesinde (< c = 1e-3; olculen tasarim §2.4'te)",
      d20 < 1e-3, f"max {d20:.2e}")
print(f"    olcum: max |R_tam - R_kesirli| = {d0:.2e} (0 bps), {d20:.2e} (20 bps)")
ec = make(mode='eval', stats=stats, rows=EV, constraints=Constraints(max_position=0.3))
ec.reset()
while ec.agents:
    ec.step({a: np.array([ec.bound], np.float32) for a in ec.agents})
check("pozisyon siniri 0,3 (tum logitler +B): kapanista max hisse <= 0,3 x (1 + seans ici kayma)",
      ec.w[1:].max() <= 0.3 * 1.1, str(ec.w))

# ---------------------------------------------------------------------------------------------
print("\n[8] Odul terimi ve TorchRL")
check("'differential' alpha != 0 -> ValueError (birincil odul 'excess'; kullanici karari 2026-11-16)",
      raises(lambda: make(reward_term='differential', episode_len=100, alpha=0.25, dsr_eta=1.0 / 20)))
check("'differential': egitim bolumu (100) <= isinma (252) -> ValueError (G sessizce hep 0 olmasin)",
      raises(lambda: make(reward_term='differential', episode_len=100, alpha=0.0)))
check("'differential' isinma bekcisi eval'de tetiklenmez (kisa pencere, egitim istatistigi)",
      not raises(lambda: make('eval', make().norm_stats, EV, reward_term='differential', alpha=0.0)))
ed = make(reward_term='differential', episode_len=100, alpha=0.0, dsr_eta=1.0 / 20)
td = run(ed, seed=3)
check("'differential' odul sonlu", all(np.isfinite(r).all() for _, r, _ in td))
nz = np.mean([inf['dsr_raw'] != 0.0 for _, _, inf in td])
check("'differential' G bolumun isinma sonrasinda sifir degil (pay >= 0,7)", nz >= 0.7, f"{nz:.2f}")
try:
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper
    wrapped = PettingZooWrapper(env=make(episode_len=20), use_mask=False,
                                categorical_actions=False, seed=0)
    ro = wrapped.rollout(5)
    grp = list(wrapped.group_map)
    check("TorchRL PettingZooWrapper: tek grup, ajan ekseni N, rollout 5 adim",
          grp == ['agent'] and ro.batch_size[0] == 5
          and ro['agent', 'observation'].shape[-2:] == (N, make().spec.local_dim), f"{grp} {ro}")
except Exception as e:  # noqa: BLE001
    check("TorchRL PettingZooWrapper: tek grup, ajan ekseni N, rollout 5 adim", False, repr(e)[:400])

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
