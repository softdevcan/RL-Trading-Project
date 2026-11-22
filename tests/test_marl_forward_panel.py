"""
Faz 11 S1: M8 ileri paneli `marl/forward_panel.py` (plan phase-11-integration-plan.md §3 S1, §3.3). ANA venv, ag yok.

Kontroller:
  1. Dondurulmus kaynaktan sifir ileri satirla kurulan panel panel_m8.npz ile bit-es (rf dahil; kaynak varsa).
  2. Zincir kurali (sentetik): temettu gununde zincirli getiri = toplam getiri; kesik cubuk (%4) Is Yatirim'la yamanir,
     yuksek/dusuk kapsar; %0,5 fark yamanmaz; yfinance satiri yoksa hucre eksik.
  3. Gun kapisi: tatil atlanir; Is Yatirim eksik / rf bayat / marj bandi disi / kesim saati oncesi -> yazilmaz, sebep;
     cok seanslik boslukta marj uygulanmaz; temiz gun kabul.
  4. Gercek kaynak + 3 sentetik ileri gun: onceki satirlar bit-es, 3 yeni satir; 2 gunluk panel 3 gunlugun basi (B0:
     sonraki gun eklemek oncekileri degistirmez); mutasyon: ileri satir dondurulmus gune duserse ve onceki panelin bir
     degeri degisirse hata. Fiyat marji dogrulanmis kapsam disindaki ileri gun hata verir (sessiz gecmez).

~30 sn.
    python tests/test_marl_forward_panel.py
"""

import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import accounting as A  # noqa: E402
from marl import forward_panel as FP  # noqa: E402
from marl import panel as PN  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


have_src = os.path.exists(os.path.join(FP.SOURCE, 'manifest.json')) and os.path.exists(FP.PANEL_M8)
REF = PN.load_npz(FP.PANEL_M8) if have_src else None
SRC = FP.load_source()[0] if have_src else None

# 1 ---------------------------------------------------------------------------------------------------------------
if have_src:
    a = FP.build_arrays(SRC, None, rf_csv=os.path.join('data', 'macro', 'risk_free.csv'))
    r = FP.check_prefix(a, REF)
    check(f"1. sifir ileri satir: panel_m8 ile bit-es ({r['prefix_rows']} satir, {len(r['keys'])} dizi, rf dahil)",
          r['new_rows'] == 0 and 'rf' in r['keys'])
else:
    print("[ATLANDI] 1. dondurulmus kaynak yok (scripts/marl/forward_panel_m8.py --freeze-source)")


# 2 ---------------------------------------------------------------------------------------------------------------
def yfdf(rows):
    return pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume']).set_index('date')


def isydf(rows):
    return pd.DataFrame(rows, columns=['date', 'HGDG_KAPANIS', 'HG_KAPANIS', 'HG_MIN', 'HG_MAX']).set_index('date')


A0, D = '2030-01-02', '2030-01-03'
anchor = {'X': (A0, 100.0), 'Y': (A0, 50.0), 'Z': (A0, 20.0), 'W': (A0, 10.0)}
yf = {  # X: %2 temettu (taze cekimde capa 100 -> 98 duzeltildi), ham d kapanisi 99
    'X': yfdf([(A0, 98, 99, 97, 98, 1000), (D, 98.5, 99.5, 98.0, 99.0, 1200)]),
    'Y': yfdf([(A0, 50, 50, 50, 50, 10), (D, 50, 50.2, 47.5, 48.0, 10)]),          # kesik cubuk: gercek 50
    'Z': yfdf([(A0, 20, 20, 20, 20, 10), (D, 20, 20.3, 19.9, 20.1, 10)]),          # IsY 20,0 -> %0,5 fark
}
isy = {'X': isydf([(A0, 98, 100, 97, 100), (D, 99, 99, 98, 99.5)]),
       'Y': isydf([(A0, 50, 50, 49, 50), (D, 50, 50, 49.0, 50.6)]),
       'Z': isydf([(A0, 20, 20, 20, 20), (D, 20, 20, 19.9, 20.1)]),
       'W': isydf([(A0, 10, 10, 10, 10), (D, 10, 10, 10, 10)])}
rows, rep = FP.chain_day(D, anchor, yf, isy)
R = {r['symbol']: r for r in rows}
ok2 = abs(R['X']['close'] - 100 * 99 / 98) < 1e-12 and not R['X']['patched']             # toplam getiri +%1,02
ok2 &= abs(R['X']['volume'] - 1200 / (100 / 98)) < 1e-9
ok2 &= R['Y']['patched'] and abs(R['Y']['close'] - 50.0) < 1e-12 and R['Y']['high'] >= 50.6 - 1e-12
ok2 &= R['Y']['low'] <= R['Y']['close'] <= R['Y']['high'] and R['Y']['low'] <= 47.5
ok2 &= not R['Z']['patched'] and abs(R['Z']['close'] - 20.1) < 1e-12
ok2 &= 'W' not in R and rep['W']['status'] == 'yf_yok'
check("2. zincir: temettu gunu toplam getiri, kesik cubuk Is Yatirim'la yamanir (yuksek/dusuk kapsar), %0,5 yamanmaz, "
      "yfinance yoksa hucre eksik", ok2)

# 3 ---------------------------------------------------------------------------------------------------------------
rate = pd.Series([40.0, 40.0], index=pd.to_datetime(['2029-12-30', A0]))
late = datetime(2030, 1, 3, 20, 0)
g_ok = FP.day_gate(D, A0, rows, rep, 4, rate, late)
g_skip = FP.day_gate(D, A0, [], {s: {'status': 'yf_yok'} for s in 'XYZW'}, 4, rate, late)
rep_isy = {**rep, 'Z': {'status': 'isy_yok'}}
g_isy = FP.day_gate(D, A0, rows, rep_isy, 4, rate, late)
g_rf = FP.day_gate(D, A0, rows, rep, 4, pd.Series([40.0], index=pd.to_datetime(['2029-12-20'])), late)
g_cut = FP.day_gate(D, A0, rows, rep, 4, rate, datetime(2030, 1, 3, 17, 0))
jump = [dict(r) for r in rows]
jump[0]['close'] = jump[0]['close_anchor'] * 1.12
g_band = FP.day_gate(D, A0, jump, rep, 4, rate, late)
gap = [dict(r, anchor_date='2029-12-27') for r in jump]                     # cok seanslik bosluk
g_gap = FP.day_gate(D, A0, gap, rep, 4, rate, late)
ok3 = g_ok['accept'] and g_ok['n_patched'] == 1
ok3 &= g_skip['skip'] and g_skip['reason'] == 'seans_degil'
ok3 &= (g_isy['reason'], g_rf['reason'], g_cut['reason'], g_band['reason']) == (
    'isy_bekleniyor', 'rf_bayat', 'kesim_saati_oncesi', 'marj_bandi_disi')
ok3 &= not any(g['accept'] for g in (g_isy, g_rf, g_cut, g_band)) and g_gap['accept']
check("3. gun kapisi: tatil atlanir; Is Yatirim eksik / rf bayat / marj bandi / kesim saati yazilmaz; cok seanslik "
      "boslukta marj uygulanmaz; temiz gun kabul", ok3)

# 4 ---------------------------------------------------------------------------------------------------------------
if have_src:
    anc = FP.anchors(REF)
    days = ['2026-11-10', '2026-11-11', '2026-11-12']
    rng = np.random.default_rng(7)
    fwd = []
    for d in days:
        for s, (_, c) in anc.items():
            r = float(rng.normal(0, 0.01))
            o = c * (1 + float(rng.normal(0, 0.003)))
            cl = c * (1 + r)
            fwd.append({'date': d, 'symbol': s, 'open': o, 'high': max(o, cl) * 1.004, 'low': min(o, cl) * 0.996,
                        'close': cl, 'volume': 1e6 * (1 + rng.random())})
            anc[s] = (d, cl)
    fwd = pd.DataFrame(fwd)
    with tempfile.TemporaryDirectory() as tmp:
        rfc = os.path.join(tmp, 'rf.csv')
        base = pd.read_csv(os.path.join('data', 'macro', 'risk_free.csv'), index_col=0)
        ext = pd.DataFrame({c: [base[c].iloc[-1]] * 3 for c in base.columns}, index=['2026-11-09', '2026-11-10',
                                                                                      '2026-11-11'])
        pd.concat([base, ext]).to_csv(rfc)
        a3 = FP.build_arrays(SRC, fwd, rf_csv=rfc)
        a2 = FP.build_arrays(SRC, fwd[fwd['date'] < days[-1]], rf_csv=rfc)
    p3 = FP.check_prefix(a3, REF)
    p23 = FP.check_prefix(a3, a2)
    last = fwd[fwd['date'] == days[-1]].set_index('symbol')['close'].reindex(a3['symbols']).to_numpy()
    ok4 = p3['new_rows'] == 3 and p23['new_rows'] == 1 and np.array_equal(a3['close'][-1], last)
    ok4 &= np.isfinite(a3['rsi'][-3:]).all() and np.isfinite(a3['adx'][-3:]).all() and np.isfinite(a3['rf'][-3:]).all()
    bad = fwd.copy()
    bad.loc[0, 'date'] = '2026-11-09'
    ok4 &= raises(lambda: FP.build_arrays(SRC, bad))
    mut = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REF.items()}
    mut['close'][-5, 3] *= 1.0000001
    ok4 &= raises(lambda: FP.check_prefix(a3, mut))
    ok4 &= raises(lambda: A.limit_down_gaps(list(a3['dates'])))
    check("4. gercek kaynak + 3 ileri gun: onceki 3364 satir bit-es, 2 gunluk panel 3 gunlugun basi (B0); ileri satir "
          "dondurulmus gune duserse ve onceki deger degisirse hata; marj kapsami disindaki gun hata", ok4)
else:
    print("[ATLANDI] 4. dondurulmus kaynak yok")

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
