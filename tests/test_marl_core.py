"""
Faz 10 M2 saf cekirdek (`marl/weights.py`, `marl/accounting.py`, `marl/orders.py`, `marl/obs_spec.py`,
`marl/panel.py`). Tasarim: docs/development/phase-10-m2-design.md. ANA venv'de kosar (pettingzoo yok).

Kontroller:
  1. Logit siniri B = 5 + ln N: tum ajanlar -B'de nakit ~ %99,33, N'den bagimsiz (M1: N=30'da %83 tavan).
  2. Projeksiyon: kisitsiz = u; dondurma, bant, pozisyon ve grup siniri; satis engeli (taban acilis,
     §9 #9); 500 rastgele durumda uzun-yalniz, toplam 1, sinirlar islem goren hissede, engelli hisse
     w~'nin altina inmez.
  3. Muhasebe: tasinan fiyat (eksik gun getiri 0, donuste aradaki hareket), elle hesap (gece + maliyet +
     seans ici + nakit faizi), tam nakit X = 0, maliyet bir kez, katki toplami = X (birinci derece).
  4. Emir (K5): sema interpret_actions_with_risk'in ust kumesi; nakit >= 0; satis <= adet;
     `portfolio.apply_decisions` ayni nakit ve adedi uretir; islem goremeyen hisseye dokunulmaz.
  5. Gozlem (K2 + B0): t sonrasi satirlar degisince t ozellikleri bit-es; canli `build_obs_at`
     ortamin onceden hesapladigi gozlemle bit-es; normalizasyon yalniz egitim satirlarindan.
  6. Panel: legacy ADX (hep 0) reddedilir; sembol sirasi korunur (sorted DEGIL).

Ag yok. ~5 sn.
    python tests/test_marl_core.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import accounting as A  # noqa: E402
from marl import obs_spec as OS  # noqa: E402
from marl import orders as O  # noqa: E402
from marl import weights as W  # noqa: E402
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
    except exc:
        return True
    return False


# ---------------------------------------------------------------------------------------------
print("\n[1] Logit siniri")
caps = []
for n in (1, 5, 30, 100):
    b = W.logit_bound(n)
    caps.append(W.target_weights(np.full(n, -b), b)[0])
check("B = 5 + ln N", abs(W.logit_bound(30) - (5 + np.log(30))) < 1e-12)
check("tum ajanlar -B: nakit = 1/(1+e^-5) ~ %99,33, N'den bagimsiz",
      all(abs(c - 1 / (1 + np.exp(-5))) < 1e-12 for c in caps), str(caps))
check("eski sabit B=5, N=30 tavani %83 (M1 olcumu yeniden uretildi)",
      abs(W.target_weights(np.full(30, -5.0), 5.0)[0] - 1 / (1 + 30 * np.exp(-5))) < 1e-12
      and 0.82 < 1 / (1 + 30 * np.exp(-5)) < 0.84)
u = W.target_weights([100.0, -100.0], 3.0)
check("logit kirpilir ve toplam 1", abs(u.sum() - 1) < 1e-12 and abs(np.log(u[1] / u[0]) - 3.0) < 1e-12)

# ---------------------------------------------------------------------------------------------
print("\n[2] Risk projeksiyonu")
n = 5
u = W.target_weights([0.3, -0.2, 1.0, 0.0, 0.5], W.logit_bound(n))
wd = np.array([0.2, 0.1, 0.2, 0.1, 0.3, 0.1])
all_tr = np.ones(n, dtype=bool)
w, rep = W.project(u, wd, all_tr)
check("kisitsiz + hepsi islem goruyor: w* = u", np.allclose(w, u, atol=1e-15))
tr = all_tr.copy()
tr[2] = False
w, rep = W.project(u, wd, tr)
check("dondurulan hisse kaymis agirligini korur", w[3] == wd[3] and rep['n_frozen'] == 1)
check("dondurmada toplam 1, kalan kutle u oraninda",
      abs(w.sum() - 1) < 1e-12 and abs(w[1] / w[0] - u[1] / u[0]) < 1e-12)
w, rep = W.project(u, wd, all_tr, W.Constraints(band=0.5))
check("genis bant: hic islem yok (w* = w~)", np.allclose(w, wd) and rep['n_banded'] == n)
u2 = np.array([0.0, 0.01, 0.01, 0.49, 0.49, 0.0])
wd2 = np.array([0.0, 0.3, 0.3, 0.2, 0.2, 0.0])
w, rep = W.project(u2, wd2, all_tr, W.Constraints(band=0.3))
check("bant toplam > 1 yaratinca islem goren kucultulur, nakit >= 0",
      w.min() >= -1e-15 and abs(w.sum() - 1) < 1e-12, str(w))
w, _ = W.project(W.target_weights([8, 0, 0, 0, 0], W.logit_bound(5)), wd, all_tr,
                 W.Constraints(max_position=0.2))
check("pozisyon siniri: max <= 0,2 ve fazla nakde", w[1:].max() <= 0.2 + 1e-15 and w[0] > 0.5, str(w))
groups = np.array([0, 0, 1, 1, 1])
w, _ = W.project(W.target_weights([0, 0, 3, 3, 3], W.logit_bound(5)), wd, all_tr,
                 W.Constraints(max_group=0.4), groups)
check("grup siniri: grup 1 toplami <= 0,4", w[3:].sum() <= 0.4 + 1e-12, str(w))
# Satis engeli (tasarim §9 #9 on kaydi): hisse 0 w~ 0,1'den 0'a inmek istiyor, engelli -> 0,1'de kalir
_us = W.target_weights([-8.0, 2.0, 0, 0, 0], W.logit_bound(5))
_ns = np.array([True, False, False, False, False])
w_ns, r_ns = W.project(_us, wd, all_tr, no_sell=_ns)
w_free, _ = W.project(_us, wd, all_tr)
check("satis engeli: hedef w~'nin altindaysa hisse dondurulur; serbestte satilir",
      w_ns[1] == wd[1] and r_ns['n_sell_blocked'] == 1 and w_free[1] < 0.01 and abs(w_ns.sum() - 1) < 1e-12)
_ub = W.target_weights([4.0, 0, 0, 0, 0], W.logit_bound(5))
w_b1, r_b1 = W.project(_ub, wd, all_tr, no_sell=_ns)
w_b0, _ = W.project(_ub, wd, all_tr)
check("satis engeli alimi kisitlamaz; engel yokken None ile bit-es",
      np.array_equal(w_b1, w_b0) and r_b1['n_sell_blocked'] == 0
      and np.array_equal(W.project(u, wd, all_tr, no_sell=np.zeros(n, bool))[0], W.project(u, wd, all_tr)[0]))
rng = np.random.default_rng(1)
ok_all = True
for _ in range(500):
    n = int(rng.integers(1, 12))
    b = W.logit_bound(n)
    u = W.target_weights(rng.uniform(-b, b, n), b)
    wd = rng.dirichlet(np.ones(n + 1))
    tr = rng.random(n) > 0.2
    g = rng.integers(0, 3, n)
    c = W.Constraints(band=float(rng.choice([0, 0.01, 0.05])), max_position=float(rng.choice([1, 0.3, 0.15])),
                      max_group=float(rng.choice([1, 0.5])))
    ns = rng.random(n) > 0.7
    w, rep = W.project(u, wd, tr, c, g, ns)
    s = w[1:]
    ok = (w.min() >= -1e-12 and abs(w.sum() - 1) < 1e-9 and np.all(s[~tr] == wd[1:][~tr])
          and np.all(s[tr & ns] >= wd[1:][tr & ns] - 1e-12))
    ok &= np.all(s[tr & ~ns] <= c.max_position + 1e-12)
    ok &= np.all((s[tr & ns] <= c.max_position + 1e-12) | (s[tr & ns] == wd[1:][tr & ns]))
    stuck = ~tr | (tr & ns & (s == wd[1:]))   # satilamayan kutle: dondurulmus + engelli (sinir asilabilir)
    for gid in np.unique(g):
        m = g == gid
        if s[m & stuck].sum() <= c.max_group:
            ok &= s[m].sum() <= c.max_group + 1e-9
    ok_all &= bool(ok)
check("500 rastgele durum: uzun-yalniz, toplam 1, sinirlar, dondurma, satis engeli", ok_all)

# ---------------------------------------------------------------------------------------------
print("\n[3] Muhasebe")
o = np.array([[10.0], [np.nan], [np.nan], [12.0]])
c = np.array([[10.0], [np.nan], [np.nan], [12.5]])
px = A.fill_prices(o, c)
g_on1, g_day1, tr1 = A.step_growth(px, 0)
g_on3, g_day3, tr3 = A.step_growth(px, 2)
check("eksik gun: g = 1 ve islem goremez", g_on1[0] == 1 and g_day1[0] == 1 and not tr1[0])
check("donus gunu: g_on = O / son gecerli kapanis", abs(g_on3[0] - 1.2) < 1e-15 and tr3[0])
po = A.fill_prices(np.array([[np.nan], [5.0]]), np.array([[np.nan], [5.5]]))
g1_, gd_, tr_ = A.step_growth(po, 0)
check("listelenmeden once g = 1, kilit 0 iken ilk gun islem gorebilir", g1_[0] == 1 and tr_[0])
# Listeleme kilidi (kullanici karari 2026-11-16, hakem-finans M2 #1): panelde sonradan listelenen sembol
_la = synth_arrays()   # sembol 2 ilk 90 satir NaN, sembol 1'de 150-152 eksik
_lk = A.fill_prices(_la['open'], _la['close'], A.LISTING_LOCKOUT)
_l0 = A.fill_prices(_la['open'], _la['close'], 0)
_first2 = int(np.flatnonzero(_lk['valid'][:, 2])[0])
check("kilit: sonradan listelenen sembol ilk gecerli gunden itibaren 20 seans islem goremez, 21.'de gorur",
      _first2 == 90 and not _lk['tradable'][90:110, 2].any() and _lk['tradable'][110, 2]
      and _lk['valid'][90:110, 2].all())
check("kilit: panel basinda listeli semboller etkilenmez; kilit 0 -> tradable = valid",
      np.array_equal(_lk['tradable'][:, [0, 1, 3]], _lk['valid'][:, [0, 1, 3]])
      and np.array_equal(_l0['tradable'], _l0['valid']))
_cut = 100
_lc = A.fill_prices(_la['open'][:_cut + 1], _la['close'][:_cut + 1], A.LISTING_LOCKOUT)
check("kilit nedensel (B0): panel t=100'de kesilince satir <= t maskesi ayni",
      np.array_equal(_lc['tradable'], _lk['tradable'][:_cut + 1]))
try:
    A.fill_prices(o, c, -1)
    _neg = False
except ValueError:
    _neg = True
check("kilit negatif -> ValueError", _neg)
# Satis engeli maskesi (§9 #9): acilis boslugu O_t / Cf_{t-1} <= 0,905; t acilisinda bilinir
_go = _la['open'].copy()
_go[120, 0] = _la['close'][119, 0] * 0.90
_go[121, 0] = _la['close'][120, 0] * 0.906
_gp = A.fill_prices(_go, _la['close'], A.LISTING_LOCKOUT, A.LIMIT_DOWN_GAP_FIXED)
_gp0 = A.fill_prices(_go, _la['close'], A.LISTING_LOCKOUT)
_gc = A.fill_prices(_go[:121], _la['close'][:121], A.LISTING_LOCKOUT, A.LIMIT_DOWN_GAP_FIXED)
check("satis engeli: -%10 bosluk engelli, -%9,4 degil; esik 0 -> hic; nedensel (B0)",
      _gp['no_sell'][120, 0] and not _gp['no_sell'][121, 0] and not _gp0['no_sell'].any()
      and np.array_equal(_gc['no_sell'], _gp['no_sell'][:121]) and not _gp['no_sell'][~_gp['valid']].any())
# v0.7 donem bagimli marj (hakem-2 H1): 2015-11-30 oncesi ve +-%20 doneminde esik 0,195, 2020-03-13'ten 0,095
# 2021+ (tasarim §2.5, 2026-11-20): yeni giris yok, %10 surer; dogrulanmis kapsam 2026-11-09'da biter
_ld = A.limit_down_gaps(['2014-01-02', '2015-11-27', '2015-11-30', '2020-03-12', '2020-03-13', '2020-12-31',
                         '2023-02-15', '2026-11-09'])
check("takvim: 0,195 x4 (BISTECH oncesi, +-%20) / 0,095 x4 (2020-03-13'ten, 2021+ dahil)",
      np.allclose(_ld, [0.195, 0.195, 0.195, 0.195, 0.095, 0.095, 0.095, 0.095]))
check("takvim: dogrulanmis kapsam disi (2026-11-10) ve tarih olmayan girdi -> ValueError",
      raises(lambda: A.limit_down_gaps(['2026-11-09', '2026-11-10'])) and raises(lambda: A.limit_down_gaps(['d0001'])))
_go2 = _la['open'].copy()
_go2[120, 0] = _la['close'][119, 0] * 0.85     # -%15: sabit kuralda engel, +-%20 doneminde engel yok
_go2[121, 0] = _la['close'][120, 0] * 0.80     # -%20: her iki kuralda engel
_row = np.full(_go2.shape[0], 0.195)
_row[121:] = 0.095
_gs = A.fill_prices(_go2, _la['close'], 0, _row)
check("satir basina esik: -%15 bosluk +-%20 doneminde engelsiz, -%20 engelli",
      not _gs['no_sell'][120, 0] and _gs['no_sell'][121, 0])
# hakem-rl M2 #8: kilit gozlemde ayri bayrak degil; kilitli hucrede feat_ok = 0 oldugu icin ajan gorur.
# Degismez: kilit penceresi feat_ok = 0 penceresinin icinde (kilit uzar ya da ozellik isinmasi kisalirsa kirilir)
from marl import obs_spec as _OS  # noqa: E402
_f, _ok = _OS.compute_raw_features(_la)
check("kilit penceresi feat_ok = 0 icinde (ajan kilidi feat_ok ile gorur; LISTING_LOCKOUT <= ozellik isinmasi)",
      _lk['lockout'].any() and not _ok[_lk['lockout']].any())

w = np.array([0.5, 0.5])
g_on, g_day, rf, cost = np.array([1.02]), np.array([0.99]), 0.0004, 0.001
u = np.array([0.2, 0.8])
res = A.step_fractional(w, u, g_on, g_day, np.array([True]), rf, cost)
g1 = 0.5 * (1 + rf) + 0.5 * 1.02
wt = np.array([0.5 * (1 + rf), 0.5 * 1.02]) / g1
tau = abs(0.8 - wt[1])
R_hand = g1 * (1 - cost * tau) * (0.2 + 0.8 * 0.99) - 1
check("elle hesap: R = g1 (1 - c tau) g2 - 1", abs(res.R - R_hand) < 1e-15, f"{res.R} vs {R_hand}")
check("X = R - rf", abs(res.X - (R_hand - rf)) < 1e-15)
check("kapanis agirligi toplam 1", abs(res.w_next.sum() - 1) < 1e-15)
res0 = A.step_fractional(np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), np.array([1.05, 0.9]),
                         np.array([1.01, 1.0]), np.array([True, True]), rf, cost)
check("tam nakit: R = rf, X = 0, devir 0, k = 0",
      abs(res0.R - rf) < 1e-15 and abs(res0.X) < 1e-15 and res0.turnover == 0 and np.all(res0.k == 0))
wa = np.array([0.4, 0.3, 0.3])
r1_ = A.step_fractional(wa, np.array([0.1, 0.45, 0.45]), np.ones(2), np.ones(2), np.ones(2, bool), 0.0, cost)
r2_ = A.step_fractional(r1_.w_next, np.array([0.1, 0.45, 0.45]), np.ones(2), np.ones(2), np.ones(2, bool),
                        0.0, cost)
check("maliyet bir kez: ilk adim c*tau, ayni hedefte ikinci adim devir 0",
      abs(r1_.R - (-cost * 0.3)) < 1e-15 and r2_.turnover < 1e-15 and abs(r2_.R) < 1e-15,
      f"{r1_.R} {r2_.turnover}")
r_ex = A.step_fractional(np.array([0.3, 0.4, 0.3]), np.array([0.3, 0.4, 0.3]), np.ones(2),
                         np.array([1.03, 0.97]), np.ones(2, bool), 0.0, cost)
check("yalniz seans ici hareket, maliyet 0: sum k = X tam", abs(r_ex.contrib_residual) < 1e-15)
rng = np.random.default_rng(2)
res_max = 0.0
for _ in range(300):
    n = 30
    w = rng.dirichlet(np.ones(n + 1))
    u = W.target_weights(rng.normal(0, 2, n), W.logit_bound(n))
    r_ = A.step_fractional(w, u, np.exp(rng.normal(0, 0.01, n)), np.exp(rng.normal(0, 0.02, n)),
                           rng.random(n) > 0.05, 0.0005, 0.001)
    res_max = max(res_max, abs(r_.contrib_residual))
check("katki ayrisimi birinci derece: |X - sum k| < 1e-3 (gunluk olcek, 300 durum)", res_max < 1e-3,
      f"max {res_max:.2e}")
w = np.array([0.2, 0.5, 0.3])
r_fz = A.step_fractional(w, np.array([1.0, 0, 0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]),
                         np.array([True, False]), 0.0, 0.0)
check("dondurulan hisse satilmaz (tam nakit hedefinde bile)", abs(r_fz.w_next[2] - 0.3) < 1e-15)
try:
    A.agent_rewards(np.zeros(2), 0.0, 1.0)
    check("alpha=1 reddedilir", False)
except ValueError:
    check("alpha=1 reddedilir", True)
check("tek yon maliyet: 20 bps gidis-donus -> 0,001", A.one_way_cost(20) == 0.001)
ds = A.DifferentialSharpe(eta=0.1)
xs = rng.normal(0.001, 0.01, 40)
outs = [ds.update(x) for x in xs]
a = b = 0.0
ok = True
for i, x in enumerate(xs, 1):
    da, db = x - a, x * x - b
    var = b - a * a
    d = 0.0 if i <= 10 else (b * da - 0.5 * a * db) / max(var, 1e-9) ** 1.5
    a, b = a + 0.1 * da, b + 0.1 * db
    ok &= abs(outs[i - 1][1] - d) < 1e-9 and abs(outs[i - 1][0] - np.tanh(d / 2)) < 1e-12
check("DifferentialSharpe = Moody-Saffell D_t (isinma 1/eta, tanh(D/2))", ok)

# ---------------------------------------------------------------------------------------------
print("\n[4] Emir uretimi (K5)")
syms = ['A', 'B', 'C']
prices = np.array([10.0, 25.0, 7.3])
hold = np.array([100, 0, 50])
cash = 5_000.0
target = np.array([0.1, 0.2, 0.5, 0.2])
dec, summ = O.rebalance_orders(target, prices, hold, cash, 0.001, syms)
legacy_keys = {'symbol', 'action', 'raw_signal', 'shares', 'price', 'cost', 'revenue', 'commission',
               'reason', 'executed'}
check("sema interpret_actions_with_risk'in ust kumesi", all(legacy_keys <= set(d) for d in dec))
check("nakit >= 0 ve nakit = nakit + gelir - maliyet",
      summ['cash_after'] >= 0 and abs(summ['cash_after'] - (cash + sum(d['revenue'] for d in dec)
                                                             - sum(d['cost'] for d in dec))) < 1e-9)
check("BUY maliyeti komisyon dahil, SELL geliri komisyon dusulmus",
      all(abs(d['cost'] - d['shares'] * d['price'] * 1.001) < 1e-9 for d in dec if d['action'] == 'BUY')
      and all(abs(d['revenue'] - d['shares'] * d['price'] * 0.999) < 1e-9 for d in dec if d['action'] == 'SELL'))
check("satis <= mevcut adet", all(d['shares'] <= hold[i] for i, d in enumerate(dec) if d['action'] == 'SELL'))
from app.services import portfolio as P  # noqa: E402
pf = P.default_portfolio(cash)
pf['positions'] = {s: {'shares': int(h), 'avg_cost': float(p)} for s, h, p in zip(syms, hold, prices) if h}
pf2, ap = P.apply_decisions(pf, dec, '2026-01-02')
h_app = np.array([pf2['positions'].get(s, {}).get('shares', 0) for s in syms])
check("portfolio.apply_decisions ayni nakit ve adedi uretir (degismeden)",
      abs(pf2['cash'] - summ['cash_after']) < 1e-9 and np.array_equal(h_app, summ['holdings_after'])
      and not ap.get('skipped'), f"{pf2['cash']} vs {summ['cash_after']}; {ap}")
h_e, c_e = O.apply_to_holdings(dec, hold, cash)
check("apply_to_holdings = apply_decisions", np.array_equal(h_e, h_app) and abs(c_e - pf2['cash']) < 1e-9)
dec2, s2 = O.rebalance_orders(np.array([1.0, 0, 0, 0]), prices, hold, cash, 0.001, syms,
                              tradable=np.array([True, True, False]))
check("islem goremeyen hisseye dokunulmaz", dec2[2]['action'] == 'HOLD' and s2['holdings_after'][2] == 50
      and dec2[0]['action'] == 'SELL')
dec3, s3 = O.rebalance_orders(np.array([1.0, 0, 0, 0]), prices, hold, cash, 0.001, syms,
                              no_sell=np.array([True, False, False]))
check("satis engeli (K5 bekcisi): engelli hisse satilmaz, digeri satilir",
      dec3[0]['action'] == 'HOLD' and 'satis engeli' in dec3[0]['reason'] and s3['holdings_after'][0] == 100
      and dec3[2]['action'] == 'SELL')
rng = np.random.default_rng(3)
ok = True
gaps = []
for _ in range(200):
    n = 30
    p_ = rng.uniform(5, 300, n)
    h_ = rng.integers(0, 500, n)
    c_ = float(rng.uniform(0, 50_000))
    t_ = W.target_weights(rng.normal(0, 2, n), W.logit_bound(n))
    d_, s_ = O.rebalance_orders(t_, p_, h_, c_, 0.001, [str(i) for i in range(n)])
    ok &= s_['cash_after'] >= -1e-9 and np.all(s_['holdings_after'] >= 0)
    big_t = W.target_weights(rng.normal(0, 2, n), W.logit_bound(n))
    _, sb = O.rebalance_orders(big_t, p_, np.zeros(n, int), 1e8, 0.001, [str(i) for i in range(n)])
    gaps.append(sb['weight_gap'])
check("200 rastgele durum: nakit >= 0, adet >= 0", bool(ok))
check("buyuk sermayede tam sayi yuvarlama farki kucuk (sum|w - w*| < %1)", max(gaps) < 0.01,
      f"max {max(gaps):.4f}")

# ---------------------------------------------------------------------------------------------
print("\n[5] Gozlem (K2 + B0)")
arr = synth_arrays()
T, N = arr['close'].shape
feat, ok = OS.compute_raw_features(arr)
check("boyut [T, N, F] ve F = 11", feat.shape == (T, N, len(OS.FEATURES_V1)) == (T, N, 11))
check("isinma satirlari gecersiz (vlog_z 60 gun)", not ok[:OS.VLOG_WIN - 1].any())
check("gec listelenen ve eksik hucre gecersiz", not ok[:90, 2].any() and not ok[150:153, 1].any())
t_cut = 170
mut = {k: v.copy() for k, v in arr.items()}
for k in mut:
    mut[k][t_cut + 1:] *= 3.7
feat_m, ok_m = OS.compute_raw_features(mut)
check("B0: t sonrasi satirlar degisince <= t ozellikleri bit-es",
      np.array_equal(np.nan_to_num(feat[:t_cut + 1], nan=-999), np.nan_to_num(feat_m[:t_cut + 1], nan=-999))
      and np.array_equal(ok[:t_cut + 1], ok_m[:t_cut + 1]))
_orig_lag = OS._lag_log


def _leaky_lag(lc, k):  # mutasyon: r_k yerine bir sonraki gunun getirisi (ileriye bakis)
    x = _orig_lag(lc, k)
    x[:-1] = lc[1:] - lc[:-1]
    return x


OS._lag_log = _leaky_lag
fl1, _ = OS.compute_raw_features(arr)
fl2, _ = OS.compute_raw_features(mut)
OS._lag_log = _orig_lag
check("mutasyon: ileriye bakan ozellik B0 kontrolunu KIRAR",
      not np.array_equal(np.nan_to_num(fl1[:t_cut + 1], nan=-999), np.nan_to_num(fl2[:t_cut + 1], nan=-999)))
train = np.zeros(T, bool)
train[60:160] = True
stats = OS.fit_norm_stats(feat, ok, train)
mut2 = {k: v.copy() for k, v in arr.items()}
for k in mut2:
    mut2[k][161:] *= 2.0
f2, ok2 = OS.compute_raw_features(mut2)
stats2 = OS.fit_norm_stats(f2, ok2, train)
check("normalizasyon yalniz egitim satirlarindan (pencere disi degisim stats'i degistirmez)",
      stats['features'] == stats2['features'])
rng = np.random.default_rng(4)
pred = {'p_up_cal': rng.uniform(0.4, 0.6, (T, N)), 'pred_logret': rng.normal(0, 0.01, (T, N)),
        'model_disagreement': np.abs(rng.normal(0, 0.005, (T, N)))}
pred['p_up_cal'][100:105, 0] = np.nan
stats_p = OS.fit_norm_stats(feat, ok, train, pred)
spec = OS.ObsSpec(n_agents=N, use_pred=True, n_groups=2, n_extras=1)
z = OS.normalize(feat, ok, stats_p)
blk, okp = OS.pred_block(pred, stats_p)
g1h = np.eye(2)[np.array([0, 1, 1, 0])]
w = np.array([0.4, 0.1, 0.2, 0.0, 0.3])
parity = True
for t in (OS.WARMUP_ROWS, 100, 102, 151, 170, T - 1):
    env_o = OS.local_obs(spec, z[t], ok[t], w, g1h, blk[t], okp[t], np.array([0.5]))
    live_o = OS.build_obs_at(spec, {k: v[:t + 1] for k, v in arr.items()}, stats_p, w, g1h,
                             {k: v[:t + 1] for k, v in pred.items()}, np.array([0.5]))
    parity &= np.array_equal(env_o, live_o)
check("K2: canli build_obs_at = ortam gozlemi (bit-es, 6 tarih, tahmin + grup + ek)", parity)
check("yerel gozlem boyutu = spec.local_dim", env_o.shape == (N, spec.local_dim) and env_o.dtype == np.float32)
check("eksik tahmin maskelenir (0,5'e dusulmez): pred_ok = 0 ve blok 0",
      not okp[102, 0] and np.all(blk[102, 0] == 0) and okp[102, 1])
check("kuresel durum boyutu = spec.state_dim", OS.global_state(spec, z[100], ok[100], w, np.array([0.5])).shape
      == (spec.state_dim,))
rep = OS.clip_report(feat, ok, stats, train)
check("gozlem sagligi raporu her ozellik icin", set(rep) == set(OS.FEATURES_V1)
      and all(v['n'] > 0 for v in rep.values()))

# ---------------------------------------------------------------------------------------------
print("\n[6] Panel")
import pandas as pd  # noqa: E402
from marl import panel as PN  # noqa: E402
dates = pd.date_range('2020-01-01', periods=5, freq='B')
order = ['ZZZ.IS', 'AAA.IS', 'MMM.IS']
rows = []
for d in dates:
    for s in order:
        rows.append({'date': d, 'symbol': s, 'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 10,
                     'rsi': 50, 'cci': 0, 'adx': 0.0, 'macd': 0})
df = pd.DataFrame(rows).set_index(['date', 'symbol'])
try:
    PN.build_arrays(df)
    check("legacy ADX (hep 0) reddedilir", False)
except ValueError:
    check("legacy ADX (hep 0) reddedilir", True)
df['adx'] = 20.0
df = df.drop(index=(dates[2], 'AAA.IS'))
out = PN.build_arrays(df)
check("sembol sirasi panelin gorunme sirasi (sorted degil)", list(out['symbols']) == order)
check("eksik hucre NaN kalir (atilmaz), takvim tam", out['close'].shape == (5, 3)
      and np.isnan(out['close'][2, 1]) and len(out['dates']) == 5)

# ---------------------------------------------------------------------------------------------
print("\n[7] M8 paneli veri duzeltmesi m8-v1 (tasarim §6.1)")
from marl import panel_fix as PF  # noqa: E402
_d = pd.bdate_range('2023-01-02', '2024-07-31')
_ds = np.asarray(_d.strftime('%Y-%m-%d'))
_rng = np.random.default_rng(7)
_truth = {s: 10.0 * np.exp(np.cumsum(_rng.normal(0, 0.01, len(_d)))) for s in ('AAA.IS', 'SKBNK.IS')}
_rows, _isy = [], {}
for _s, _tr in _truth.items():
    _c = _tr.copy()
    if _s == 'SKBNK.IS':                       # yfinance: iki sinirdan oncesi duzeltilmemis (gercek veriyle ayni)
        for _fx in PF.RIGHTS_FIXES:
            _c[_ds < _fx['yf_boundary']] /= PF.rights_factor(_fx)
    else:                                      # kesik gun cubugu: kapanis %5 yuksek
        _c[list(_ds).index('2023-01-05')] *= 1.05
    _rows += [{'symbol': _s, 'date': d, 'open': _c[i], 'high': _c[i] * 1.01, 'low': _c[i] * 0.99,
               'close': _c[i], 'volume': 100.0} for i, d in enumerate(_d)]
    _k = 2.0 if _s == 'AAA.IS' else 0.5        # iki kaynagin sabit olcek farki
    _isy[_s] = pd.DataFrame({'HGDG_KAPANIS': _tr * _k, 'HG_KAPANIS': _tr * _k, 'HG_MIN': _tr * _k * 0.98,
                             'HG_MAX': _tr * _k * 1.02}, index=_ds)
_raw = pd.DataFrame(_rows).set_index(['symbol', 'date'])
_fixd, _rep = PF.apply_m8_v1(_raw, _isy)
_keep = _ds != '2023-02-08'
_sk = _fixd.xs('SKBNK.IS', level='symbol')['close'].to_numpy()
_aaa, _raa = _fixd.xs('AAA.IS', level='symbol'), _raw.xs('AAA.IS', level='symbol')[_keep]
_i5 = list(_aaa.index.strftime('%Y-%m-%d')).index('2023-01-05')
_t5 = _truth['AAA.IS'][list(_ds).index('2023-01-05')]
check("m8-v1: iptal gunu atilir; SKBNK sinir oncesi TERP carpaniyla duzelir; kesik kapanis IsY x yerel olcege "
      "yamanir, dusuk IsY araligina genisler; kalan hucreler bit-es",
      _rep['drop_sessions']['2023-02-08']['rows'] == 2 and len(_aaa) == _keep.sum()
      and np.allclose(_sk, _truth['SKBNK.IS'][_keep], rtol=1e-12, atol=0)
      and _rep['close_patch']['n'] == 1 and np.isclose(_aaa['close'].iloc[_i5], _t5, rtol=1e-12, atol=0)
      and np.isclose(_aaa['low'].iloc[_i5], 0.98 * _t5) and _aaa['high'].iloc[_i5] >= _aaa['close'].iloc[_i5]
      and _aaa.drop(_aaa.index[_i5]).equals(_raa.drop(_raa.index[_i5])),
      f"patch {_rep['close_patch']['n']}, rights {[r['rows_scaled'] for r in _rep['rights_fixes']]}")

# ---------------------------------------------------------------------------------------------
print("\n[8] M8 kontrol noktasi secimi: marl.selection == evaluation.rl_folds (M8 tasarimi A1)")
from marl import selection as SEL  # noqa: E402
from evaluation import rl_folds as RF  # noqa: E402
_rng8, _ok8 = np.random.default_rng(11), True
for _ in range(30):
    _n = int(_rng8.integers(2, 300))
    _R, _rf = _rng8.normal(0.0005, 0.01, _n), np.abs(_rng8.normal(0.0004, 0.0001, _n))
    _dt = pd.bdate_range('2020-01-01', periods=_n)
    _a, _b = SEL.excess_sharpe(_R, _rf), RF.excess_sharpe(_R, _dt, pd.Series(_rf, index=_dt))
    _ok8 &= (_a is None and _b is None) or (_a is not None and _b is not None and np.isclose(_a, _b, rtol=1e-12))
_ok8 &= SEL.excess_sharpe(np.zeros(10), np.zeros(10)) is None and RF.excess_sharpe(np.zeros(10)) is None   # hic islem yok
for _ in range(40):                                   # esitlik (0,1'e yuvarlanmis metrik), None ve hepsi-None
    _recs = [{'step': s * 100, 'params_sha': str(_rng8.integers(0, 4)),
              'metric': (None if _rng8.random() < 0.25 else float(np.round(_rng8.normal(), 1)))} for s in range(1, 11)]
    if _rng8.random() < 0.1:
        _recs = [{**r, 'metric': None} for r in _recs]
    _x, _y = SEL.select_checkpoint(_recs), RF.select_checkpoint(_recs)
    _ok8 &= _x['step'] == _y['step'] and bool(_x['degenerate']) == bool(_y.get('degenerate'))
check("excess_sharpe ve select_checkpoint iki uygulamada ayni (rastgele seri, sabit brut, esitlik, hepsi None); "
      "N_CHECKPOINTS ayni", bool(_ok8) and SEL.N_CHECKPOINTS == RF.N_CHECKPOINTS)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
