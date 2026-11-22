"""
Faz 11 S3: sanal para defteri `marl/sim_ledger.py` (plan phase-11-integration-plan.md §3 S3, §3.4). ANA venv, torch.

Kontroller (dondurulmus arsiv ve panel_m8 yoksa ATLANIR; tam 72 politika paritesi scripts/marl/s3_parity.py):
  1. Seri paritesi: M-none s201 (mlp) ve S-agr s201 (sarl_agr) 2026 test penceresinde M8 test kaydiyla R/X/devir bit-es;
     minvar ve bh `rule_series` ile bit-es.
  2. Kaldigi yerden devam: pencere ortadan bolunur, durum JSON'a yazilip okunur, ikinci yari devam eder -> tam kosuyla
     bit-es (politika + minvar: ay basi dengeleme baslangic satirina bagli). Sermaye R'yi degistirmez, deger = sermaye x
     prod(1 + R).
  3. Mutasyon: maliyet 10 bps, durum agirliginda 1e-12 kayma ve yanlis baslangic satiri (minvar) seriyi degistirir.
  5. Tam sayi mod (S5 urun portfoyu, orders.step_integer): JSON devam bit-es, 1e9 sermayede kesirliye yakinsar.
  4. Ileri panel (varsa): fiyat marji dogrulanmis kapsami disindaki gun Market kurulumunda hata verir (S4 on kosulu;
     sessiz gecmez).

~15 sn.
    python tests/test_marl_sim_ledger.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import panel as PN  # noqa: E402
from marl import rule_baselines as RB  # noqa: E402
from marl import sim_ledger as SL  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


ARCH = os.path.join('data', 'marl', 'frozen', 'm8-2026')
PANEL = os.path.join('data', 'marl', 'panel_m8.npz')
M8 = os.path.join('results', 'marl_m8', '2026')
if not (os.path.exists(os.path.join(ARCH, 'manifest.json')) and os.path.exists(PANEL)):
    print("[ATLANDI] dondurulmus arsiv ya da panel_m8 yok")
    sys.exit(0)

man = {(p['arm'], p['seed']): p for p in json.load(open(os.path.join(ARCH, 'manifest.json'), encoding='utf-8'))['policies']}
with open(os.path.join(ARCH, 'M-none', 's201', 'fold_eval.json'), encoding='utf-8') as fh:
    TS, TE = json.load(fh)['rows']['test']
mk = SL.Market(PN.load_npz(PANEL))


def pol(arm, seed):
    return SL.PolicyStrategy(os.path.join(ARCH, arm, seed), man[(arm, seed)]['ckpt'], sid=f'policy/{arm}/{seed}')


def full(strategy, capital=SL.INITIAL_CAPITAL, start=TS, cost_mk=None):
    return SL.series(SL.run(cost_mk or mk, strategy, SL.LedgerState.start(start, mk.N, capital), TE)[1])


def split(strategy, cut):
    st = SL.LedgerState.start(TS, mk.N)
    _, a = SL.run(mk, strategy, st, cut)
    st2 = SL.LedgerState.from_json(st.to_json())
    _, b = SL.run(mk, strategy, st2, TE)
    return SL.series(a + b), st2


# 1 ---------------------------------------------------------------------------------------------------------------
ser = {}
ok1 = True
for arm in ('M-none', 'S-agr'):
    ser[arm] = full(pol(arm, 's201'))
    path = os.path.join(M8, arm, 's201', 'eval', 'test_argmax.npz')
    if os.path.exists(path):
        z = np.load(path)
        ok1 &= all(np.array_equal(ser[arm][k], z[f'policy_{k}']) for k in ('R', 'X', 'turnover', 'cash_weight'))
    else:
        print(f"  (M8 test kaydi yok: {arm} yalniz kural paritesi)")
for kind in ('minvar', 'bh'):
    ser[kind] = full(SL.RuleStrategy(kind))
    ref = RB.rule_series(mk.arrays, TS, TE, kind, SL.COST_BPS_ROUNDTRIP)
    ok1 &= all(np.array_equal(ser[kind][k], ref[k]) for k in ('R', 'X', 'turnover', 'cash_weight'))
check(f"1. seri paritesi: M-none + S-agr M8 test kaydiyla, minvar + bh rule_series ile bit-es ({TE - TS} gun)", ok1)

# 2 ---------------------------------------------------------------------------------------------------------------
cut = TS + 61            # ay ortasi degil: herhangi bir satir; minvar'da yeniden dengeleme takvimi baslangica bagli
ok2 = True
for name, strat in (('M-none', pol('M-none', 's201')), ('minvar', SL.RuleStrategy('minvar'))):
    s2, st2 = split(strat, cut)
    ok2 &= all(np.array_equal(s2[k], ser[name][k]) for k in ('R', 'X', 'turnover', 'cash_weight', 'value'))
    ok2 &= st2.t == TE and st2.n_steps == TE - TS
big = full(pol('M-none', 's201'), capital=1e9)
ok2 &= np.array_equal(big['R'], ser['M-none']['R'])
ok2 &= abs(ser['M-none']['value'][-1] / SL.INITIAL_CAPITAL - np.prod(1 + ser['M-none']['R'])) < 1e-12
check("2. kaldigi yerden devam (JSON durum) tam kosuyla bit-es (politika + minvar); sermaye R'yi degistirmez", ok2)

# 3 ---------------------------------------------------------------------------------------------------------------
m10 = SL.Market(mk.arrays, cost_bps_roundtrip=10.0)
ok3 = not np.array_equal(full(pol('M-none', 's201'), cost_mk=m10)['R'], ser['M-none']['R'])
st = SL.LedgerState.start(TS, mk.N)
SL.run(mk, pol('M-none', 's201'), st, cut)
st_b = SL.LedgerState.from_json(st.to_json())
st_b.w = st_b.w.copy()
st_b.w[1] += 1e-12
st_b.w[0] -= 1e-12
_, tail = SL.run(mk, pol('M-none', 's201'), st_b, TE)
ok3 &= not np.array_equal(SL.series(tail)['R'], ser['M-none']['R'][cut - TS:])
s_w = SL.series(SL.run(mk, SL.RuleStrategy('minvar'), SL.LedgerState.start(TS + 3, mk.N), TE)[1])   # takvim kayar
ok3 &= str(mk.dates[TS + 3])[:7] == str(mk.dates[TS + 2])[:7]       # ay basi degil: fark baslangic kuralindan
ok3 &= not np.array_equal(s_w['R'], ser['minvar']['R'][3:])
check("3. mutasyon: maliyet 10 bps, durum agirliginda 1e-12 kayma, kaydirilmis baslangic seriyi degistirir", ok3)

# 5 (S5 urun portfoyu: tam sayi mod) -------------------------------------------------------------------------------
pi = pol('M-none', 's201')
sti = SL.LedgerState.start(TS, mk.N, 100_000.0, integer=True)
_, ia = SL.run(mk, pi, sti, cut)
sti2 = SL.LedgerState.from_json(sti.to_json())
_, ib = SL.run(mk, pi, sti2, TE)
_, ifull = SL.run(mk, pi, SL.LedgerState.start(TS, mk.N, 100_000.0, integer=True), TE)
ok5 = [r['R'] for r in ia + ib] == [r['R'] for r in ifull] and sti2.integer and sti2.holdings.dtype.kind == 'i'
big = SL.series(SL.run(mk, SL.RuleStrategy('ew'), SL.LedgerState.start(TS, mk.N, 1e9, integer=True), TE)[1])
frac = full(SL.RuleStrategy('ew'))
ok5 &= abs(big['value'][-1] / 1e9 / (frac['value'][-1] / SL.INITIAL_CAPITAL) - 1) < 1e-4
ok5 &= not np.array_equal(SL.series(ifull)['R'], ser['M-none']['R'])          # 100k'da yuvarlama farki var
check("5. tam sayi mod (urun portfoyu): JSON durumla devam bit-es, adet tam sayi; 1e9 sermayede kesirliye yakinsar "
      "(< 1e-4), 100k'da fark var", ok5)

# 4 ---------------------------------------------------------------------------------------------------------------
fwd = os.path.join('data', 'marl', 'forward', 'm8', 'panel_m8_fwd.npz')
if os.path.exists(fwd):
    try:
        SL.Market(PN.load_npz(fwd))
        ok4 = False
    except ValueError as e:
        ok4 = 'fiyat marji' in str(e)
    check("4. ileri panel: marj dogrulama kapsami disindaki gun Market kurulumunda hata (S4 on kosulu)", ok4)
else:
    print("[ATLANDI] 4. ileri panel yok (scripts/marl/forward_panel_m8.py --update)")

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
