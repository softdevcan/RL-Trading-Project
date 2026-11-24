"""
Faz 11 S4: ileri veri gunluk isi `marl/forward_job.py` + marj kapsami `marl/margin_coverage.py` (plan
phase-11-integration-plan.md §3 S4). ANA venv, torch, ag yok. Dondurulmus kaynak/arsiv yoksa 2-4 ATLANIR.

Kontroller:
  1. Marj kapsami: kapsam = min(panel son gunu, son kaynak denetimi + 31 gun); takvim degisikligi bildiren denetim ve
     kaynaksiz kayit hata; accounting.limit_down_gaps varsayilani degismedi (verified_until=None ile sabit ayni),
     kapsam geriye gidemez, kapsam disi gun hata.
  2. Gercek kaynak + 6 sentetik ileri gun, 2 politika + 2 kural: ilk ileri gun panelde yokken depo KURULMAZ; her gun
     calisma = 3 gun bosluk sonrasi doldurma = tek seferde (satirlar + durum bit-es) = sim_ledger dogrudan kosusu.
  3. Idempotent: ayni gun ikinci calisma dosyalara dokunmaz (SHA ayni).
  4. Kurtarma ve bekciler: rows.csv'ye fazladan satir (cokme) atilir ve sonuc ayni; strateji kumesi farki, panelin
     geri gitmesi ve baslangic satirinin degismesi hata.
  5. Tek yazar (plan §3.12): depoda WRITER varsa farkli kimlik 'yazar_degil' doner, depoya dokunmaz.

~30 sn.
    python tests/test_marl_forward_job.py
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from marl import accounting as A  # noqa: E402
from marl import forward_job as FJ  # noqa: E402
from marl import forward_panel as FP  # noqa: E402
from marl import margin_coverage as MC  # noqa: E402
from marl import panel as PN  # noqa: E402
from marl import sim_ledger as SL  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# 1 ---------------------------------------------------------------------------------------------------------------
chk = [{'checked': '2026-11-20', 'schedule_unchanged': True, 'sources': ['x']}]
c1 = MC.coverage_until('2026-12-01', chk)
c2 = MC.coverage_until('2027-01-15', chk)
ok1 = c1['until'] == '2026-12-01' and c1['limited_by'] == 'panel'
ok1 &= c2['until'] == '2026-12-21' and c2['limited_by'] == 'check'
ok1 &= MC.coverage_until('2026-11-05', chk)['until'] == A.PRICE_MARGIN_VERIFIED_UNTIL
ok1 &= raises(lambda: MC.coverage_until('2026-12-01', chk + [{'checked': '2026-11-25', 'schedule_unchanged': False,
                                                                'sources': ['y']}]))
real = MC.load_checks()
ok1 &= len(real) >= 1 and all(c['sources'] for c in real)
d_hist = ['2026-11-05', '2026-11-06', '2026-11-09']
ok1 &= np.array_equal(A.limit_down_gaps(d_hist), A.limit_down_gaps(d_hist, A.PRICE_MARGIN_VERIFIED_UNTIL))
ok1 &= raises(lambda: A.limit_down_gaps(d_hist, '2026-11-01'))
ok1 &= raises(lambda: A.limit_down_gaps(d_hist + ['2026-11-10']))
ok1 &= np.array_equal(A.limit_down_gaps(d_hist + ['2026-11-10'], '2026-11-30')[:3], A.limit_down_gaps(d_hist))
check("1. marj kapsami: min(panel, denetim + 31 gun); takvim degisikligi hata; limit_down_gaps varsayilani ayni, "
      "geriye gitmez, kapsam disi hata", ok1)

# 2-4 -------------------------------------------------------------------------------------------------------------
ARCH = FJ.ARCHIVE
have = os.path.exists(os.path.join(FP.SOURCE, 'manifest.json')) and os.path.exists(FP.PANEL_M8) and \
    os.path.exists(os.path.join(ARCH, 'manifest.json'))
if not have:
    print("[ATLANDI] 2-4: dondurulmus kaynak/arsiv yok")
    print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
    sys.exit(1 if failed else 0)

REF = PN.load_npz(FP.PANEL_M8)
SRC = FP.load_source()[0]
anc = FP.anchors(REF)
DAYS = ['2026-11-10', '2026-11-11', '2026-11-12', '2026-11-13', '2026-11-16', '2026-11-17']
rng = np.random.default_rng(11)
fwd = []
for d in DAYS:
    for s, (_, c) in anc.items():
        r = float(rng.normal(0, 0.012))
        o = c * (1 + float(rng.normal(0, 0.004)))
        cl = c * (1 + r)
        fwd.append({'date': d, 'symbol': s, 'open': o, 'high': max(o, cl) * 1.004, 'low': min(o, cl) * 0.996,
                    'close': cl, 'volume': 1e6 * (1 + rng.random())})
        anc[s] = (d, cl)
tmp = tempfile.mkdtemp()
rfc = os.path.join(tmp, 'rf.csv')
base = pd.read_csv(os.path.join('data', 'macro', 'risk_free.csv'), index_col=0)
ext = pd.DataFrame({c: [base[c].iloc[-1]] * 6 for c in base.columns},
                   index=['2026-11-09', '2026-11-10', '2026-11-11', '2026-11-12', '2026-11-13', '2026-11-16'])
pd.concat([base, ext]).to_csv(rfc)
FULL = FP.build_arrays(SRC, pd.DataFrame(fwd), rf_csv=rfc)
START = '2026-11-11'                     # ilk karar satiri 2026-11-10
VU = '2026-11-30'
SIDS = ['policy/M-none/s201', 'policy/S-agr/s201', 'rule/minvar', 'rule/bh']


def run_to(store, day):
    return FJ.advance(FJ.truncate(FULL, day), store, SIDS, start_date=START, verified_until=VU, now='T')


def snap(store):
    meta, st, rows = FJ.load_store(store)
    return {s: (v.start_row, v.t, v.w.tolist(), v.value, v.n_steps) for s, v in st.items()}, \
        rows.drop(columns=['written_at']).sort_values(['sid', 'row']).reset_index(drop=True)


def same(a, b):
    return a[0] == b[0] and a[1].equals(b[1])


def sha_dir(store):
    out = {}
    for f in sorted(os.listdir(store)):
        with open(os.path.join(store, f), 'rb') as fh:
            out[f] = hashlib.sha256(fh.read()).hexdigest()
    return out


S_daily, S_gap, S_once = (os.path.join(tmp, x) for x in ('daily', 'gap', 'once'))
pre = run_to(S_daily, '2026-11-10')
ok2 = pre['status'] == 'baslamadi' and not os.path.exists(S_daily)
for d in DAYS[1:]:
    run_to(S_daily, d)
run_to(S_gap, DAYS[1])
gap = run_to(S_gap, DAYS[-1])                              # 4 gun bosluk
once = run_to(S_once, DAYS[-1])
A_, B_, C_ = snap(S_daily), snap(S_gap), snap(S_once)
ok2 &= same(A_, B_) and same(A_, C_) and gap['new_dates'] == DAYS[2:] and once['n_sessions'] == 5
mk = SL.Market(FULL, verified_until=VU)
r0 = FJ.start_row(FULL['dates'], START)
ok2 &= r0 == int(np.flatnonzero(FULL['dates'] == '2026-11-10')[0])
direct = SL.series(SL.run(mk, FJ.make_strategy('policy/M-none/s201'), SL.LedgerState.start(r0, mk.N), len(FULL['dates']) - 1)[1])
mine = A_[1][A_[1]['sid'] == 'policy/M-none/s201']
ok2 &= np.array_equal(mine['R'].to_numpy(), direct['R']) and list(mine['date']) == DAYS[1:]
ok2 &= np.array_equal(mine['value'].to_numpy(), direct['value'])
check("2. ilk ileri gun yokken depo kurulmaz; her gun = 4 gun bosluk sonrasi doldurma = tek seferde = dogrudan defter "
      "(bit-es, 5 seans, 2 politika + 2 kural)", ok2)

# 3 ---------------------------------------------------------------------------------------------------------------
h0 = sha_dir(S_daily)
again = run_to(S_daily, DAYS[-1])
check("3. idempotent: ayni gun ikinci calisma 'degisiklik_yok', dosya SHA'lari ayni",
      again['status'] == 'degisiklik_yok' and sha_dir(S_daily) == h0)

# 4 ---------------------------------------------------------------------------------------------------------------
S_crash = os.path.join(tmp, 'crash')
shutil.copytree(S_daily, S_crash)
rows = pd.read_csv(os.path.join(S_crash, FJ.ROWS), dtype={'sid': str, 'date': str}, float_precision='round_trip')
pd.concat([rows, rows.tail(3)], ignore_index=True).to_csv(os.path.join(S_crash, FJ.ROWS), index=False, float_format='%.17g')
ok4 = same(snap(S_crash), A_)
ok4 &= raises(lambda: FJ.advance(FJ.truncate(FULL, DAYS[-1]), S_daily, SIDS[:3], start_date=START, verified_until=VU))
ok4 &= raises(lambda: run_to(S_daily, DAYS[3]))                                     # panel geri gitti
ok4 &= raises(lambda: FJ.advance(FULL, S_daily, SIDS, start_date='2026-11-12', verified_until=VU))
short = os.path.join(tmp, 'short')
shutil.copytree(S_daily, short)
rows.head(len(rows) - 2).to_csv(os.path.join(short, FJ.ROWS), index=False, float_format='%.17g')
ok4 &= raises(lambda: FJ.load_store(short))
check("4. cokme kurtarma (fazla satir atilir, sonuc ayni); strateji kumesi farki, panel geri gitmesi, baslangic "
      "degisimi, eksik satir hata", ok4)

# 5 ---------------------------------------------------------------------------------------------------------------
# Tek yazar (plan §3.12): depoda WRITER varsa yalniz ayni MARL_FORWARD_WRITER_ID ile yazilir; digeri hicbir dosyaya
# (gunluk ve kilit dahil) dokunmadan 'yazar_degil' doner.
from marl import forward_panel as FP5  # noqa: E402

wdir = tempfile.mkdtemp()
old_root, old_env = FP5.FWD_ROOT, os.environ.pop(FJ.WRITER_ENV, None)
try:
    FP5.FWD_ROOT = wdir
    ok5 = FJ.writer_check()['ok']                                  # WRITER yok: serbest
    with open(os.path.join(wdir, FJ.WRITER), 'w', encoding='utf-8') as fh:
        fh.write('vds\n')
    r5 = FJ.run_daily(update_panel=False)
    ok5 &= r5['status'] == 'yazar_degil' and sorted(os.listdir(wdir)) == [FJ.WRITER]
    os.environ[FJ.WRITER_ENV] = 'yerel'
    ok5 &= not FJ.writer_check()['ok']
    os.environ[FJ.WRITER_ENV] = 'vds'
    ok5 &= FJ.writer_check()['ok']
finally:
    FP5.FWD_ROOT = old_root
    os.environ.pop(FJ.WRITER_ENV, None)
    if old_env is not None:
        os.environ[FJ.WRITER_ENV] = old_env
    shutil.rmtree(wdir, ignore_errors=True)
check("5. tek yazar: WRITER yoksa serbest; varsa farkli/bos kimlik 'yazar_degil' ve depoya (gunluk, kilit) dokunmaz; "
      "ayni kimlik yazar", ok5)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
