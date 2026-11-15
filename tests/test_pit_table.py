"""
Faz 9 / P (P1-P2, K19, Y26) — nokta-zamanli orneklem disi tahmin tablosu (`prediction/pit_table.py`).

Kontroller:
  1. B0 (gunluk): t sonrasi fiyat/hacim degistirilince t'ye kadarki TUM satirlar bit-es; t sonrasi
     farkli (bozma etkili). Haftalik ufukta (h=5) da.
  2. Mutasyon (haftalik, t = capa + 1): arindirmasiz taban egitimi ve arindirmasiz meta B0'i KIRAR.
  3. K19: her satirda train_end < label_end < capa <= tarih; meta_label_end < capa (meta girdisi
     orneklem disi ve etiketi capadan once gerceklesmis).
  4. Havuzlu sicaklik: ayni capada tum semboller ayni T; p_up_cal = sigmoid(logit / T),
     p_up_raw = sigmoid(logit); pred_logret = taban ortalamasi; anlasmazlik = std.
  5. Maske: gec baslayan sembol (ODAS) no_data / no_features / insufficient_history; maskeli
     satirda p NaN (sessiz 0,5 yok), ok satirda dolu.
  6. P2 kapsam raporu ve parquet + yan JSON + manifest yazimi (gecici dizin), geri okuma bit-es.
  7. block_bootstrap_pvalue: bilgili tahminci p < 0,01, gurultu p > 0,05; bilgi kapisi + BH.

Gercek panel gerekir (`data/bist/stock_data_with_indicators.csv`, ag yok). ~2 dk.
    python tests/test_pit_table.py
"""

import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings('ignore')
import logging  # noqa: E402
logging.disable(logging.WARNING)

from prediction import pit_table as P  # noqa: E402
from evaluation import calibration as cal  # noqa: E402

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


if not os.path.exists(os.path.join('data', 'bist', 'stock_data_with_indicators.csv')):
    print("[ATLANDI] panel yok")
    sys.exit(0)

SMALL = {'xgboost': {'n_estimators': 40}, 'lightgbm': {'n_estimators': 40},
         'catboost': {'iterations': 40}}
SYMS = ['THYAO.IS', 'GARAN.IS']
panel, prep = P.load_panel(end='2014-06-30')


def build(pnl, horizon='daily', syms=SYMS):
    return P.build_pit_table(pnl, syms, horizon=horizon, params_override=SMALL, ci_boot=100)


def perturb_after(pnl, t):
    """t'den SONRAKI satirlarda fiyatlari yariya indir, hacmi 3 kat (etiketler de doner)."""
    q = pnl.copy()
    d = pd.to_datetime(q.index.get_level_values('date'))
    m = d > t
    for c in ('open', 'high', 'low', 'close'):
        q.loc[m, c] = q.loc[m, c] * 0.5
    q.loc[m, 'volume'] = q.loc[m, 'volume'] * 3
    return q


def same_upto(a, b, t):
    x = a[a['date'] <= t].reset_index(drop=True)
    y = b[b['date'] <= t].reset_index(drop=True)
    if len(x) != len(y) or list(x.columns) != list(y.columns):
        return False, 'boyut'
    for c in x.columns:
        u, v = x[c], y[c]
        if u.dtype.kind in 'fc':
            if not np.array_equal(u.to_numpy(), v.to_numpy(), equal_nan=True):
                return False, c
        elif not ((u.isna() & v.isna()) | (u == v)).all():
            return False, c
    return True, ''


def differs_after(a, b, t):
    x = a[a['date'] > t]['p_up_cal'].to_numpy()
    y = b[b['date'] > t]['p_up_cal'].to_numpy()
    return not np.array_equal(x, y, equal_nan=True)


print("\n[1] B0 kesme degismezligi")
full, rep = build(panel)
anchors = pd.DatetimeIndex(rep['anchors'])
ok_anchors = sorted({pd.Timestamp(v.split(':')[1]) for v in full.loc[full.meta_ok, 'model_version']})
cal_dates = pd.DatetimeIndex(sorted(full['date'].unique()))
a_k = ok_anchors[1]
t = cal_dates[cal_dates.get_loc(a_k) + 20]
pert, _ = build(perturb_after(panel, t))
ok, col = same_upto(full, pert, t)
check(f"gunluk: t={t.date()} (capa {a_k.date()} + 20) kadar tum sutunlar bit-es", ok, col)
check("gunluk: t sonrasi p_up_cal farkli (bozma etkili)", differs_after(full, pert, t))
t2 = cal_dates[cal_dates.get_loc(ok_anchors[-1]) - 1]
pert2, _ = build(perturb_after(panel, t2))
ok, col = same_upto(full, pert2, t2)
check(f"gunluk: t = capa - 1 ({t2.date()}): sonraki capanin egitimi t sonrasini gormuyor", ok, col)

wk, wrep = build(panel, 'weekly')
wk_ok = sorted({pd.Timestamp(v.split(':')[1]) for v in wk.loc[wk.meta_ok, 'model_version']})
tw = cal_dates[cal_dates.get_loc(wk_ok[1]) + 1]
wpert, _ = build(perturb_after(panel, tw), 'weekly')
ok, col = same_upto(wk, wpert, tw)
check(f"haftalik (h=5): t = capa + 1 ({tw.date()}) kadar bit-es", ok, col)

print("\n[2] Mutasyonlar B0'i kirmali (haftalik, t = capa + 1)")
for knob in ('PURGE_BASE', 'PURGE_META'):
    setattr(P, knob, False)
    try:
        a, _ = build(panel, 'weekly')
        b, _ = build(perturb_after(panel, tw), 'weekly')
    finally:
        setattr(P, knob, True)
    ok, col = same_upto(a, b, tw)
    check(f"{knob}=False (etiketi capadan sonra gerceklesen satirlar) -> B0 kirilir", not ok, col)

print("\n[3] K19: orneklem disi girdiler ve son egitim tarihi")
f = full[full.base_ok].copy()
f['anchor'] = pd.to_datetime(f['model_version'].str.split(':').str[1])
check("base_ok satirlarinda train_end < label_end < capa <= tarih",
      bool((f.train_end < f.label_end).all() and (f.label_end < f.anchor).all()
           and (f.anchor <= f.date).all()))
mo = f[f.meta_ok]
check("meta_ok satirlarinda meta_label_end < capa (meta + sicaklik girdisi capadan once)",
      len(mo) > 0 and bool((mo.meta_label_end < mo.anchor).all()))

print("\n[4] Havuzlu sicaklik ve sutun tanimlari")
check("ayni capada tum semboller ayni T (havuzlu)",
      bool((mo.groupby('anchor')['temperature'].nunique() == 1).all())
      and mo.groupby('anchor')['symbol'].nunique().max() == len(SYMS))
check("p_up_cal = sigmoid(logit / T), p_up_raw = sigmoid(logit)",
      np.allclose(mo.p_up_cal, cal.sigmoid(mo.logit / mo.temperature), atol=1e-12)
      and np.allclose(mo.p_up_raw, cal.sigmoid(mo.logit), atol=1e-12))
rc = [f'ret_{m}' for m in P.V1_MODELS]
check("pred_logret = taban ortalamasi, model_disagreement = std (ddof=0)",
      np.allclose(f.pred_logret, f[rc].mean(axis=1)) and np.allclose(f.model_disagreement, f[rc].std(axis=1, ddof=0)))
T_log = {e['anchor']: e.get('T_pooled') for e in rep['meta_log']}
check("rapordaki T_pooled tablodaki T ile ayni",
      all(abs(T_log[str(a.date())] - g.iloc[0]) < 1e-15 for a, g in mo.groupby('anchor')['temperature']))

print("\n[5] Maske (gec baslayan sembol)")
od, _ = build(panel, syms=['THYAO.IS', 'ODAS.IS'])
o = od[od.symbol == 'ODAS.IS']
reasons = set(o.mask_reason)
check("ODAS: no_data (listelenmeden once) + insufficient_history; ok yok (2014-06'ya kadar)",
      {'no_data', 'insufficient_history'} <= reasons and 'ok' not in reasons, str(reasons))
check("maskeli satirlarda p_up_cal / pred_logret NaN (sessiz 0,5 yok)",
      bool(od.loc[~od.meta_ok, 'p_up_cal'].isna().all() and od.loc[~od.base_ok, 'pred_logret'].isna().all()))
check("ok satirlarinda p_up_cal ve pred_logret dolu",
      bool(od.loc[od.meta_ok, ['p_up_cal', 'pred_logret']].notna().all().all()))
check("mask_reason yalniz tanimli degerler", set(od.mask_reason) <= set(P.MASK_REASONS))

print("\n[6] P2 kapsam ve yazim")
cov = P.coverage_report(od)
po = cov['per_symbol']
check("kapsam: ODAS meta_ok 0, THYAO meta_ok 1,0 (RL penceresi 2013-06..)",
      po['ODAS.IS']['meta_ok'] == 0 and po['THYAO.IS']['meta_ok'] == 1.0, str(po))
check("kapsam: maske sayilari toplam = sembol x gun",
      sum(cov['mask_reasons'].values()) == cov['n_calendar_days'] * 2)
tmp = tempfile.mkdtemp(prefix='pit_')
try:
    import prediction.manifest as M
    old = M.runs_dir
    M.runs_dir = lambda: tmp
    try:
        rep['coverage'] = P.coverage_report(full)
        paths = P.write_table(full, rep, run_id='t0', directory=tmp)
    finally:
        M.runs_dir = old
    back = pd.read_parquet(paths['table'])
    ok, col = same_upto(full, back, full['date'].max())
    check("parquet geri okuma bit-es; yan JSON + manifest yazildi",
          ok and os.path.exists(paths['sidecar']) and os.path.exists(paths['manifest']), col)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n[7] p-degeri ve bilgi kapisi")
rng = np.random.default_rng(3)
y = (rng.random(600) < 0.5).astype(float)
good = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.1, 600), 0.01, 0.99)
noise = rng.random(600)
pg = cal.block_bootstrap_pvalue(cal.auc, good, y, n_boot=300)
pn = cal.block_bootstrap_pvalue(cal.auc, noise, y, n_boot=300)
check("bilgili tahminci p < 0,01; gurultu p > 0,05", pg < 0.01 and pn > 0.05, f"{pg}, {pn}")
labels = P.realized_labels(panel)
g = P.evaluate_gate(full, labels, window=('2013-06-01', '2014-07-01'), n_boot=100)
check("evaluate_gate: BH ailesi = degerlendirilen sembol sayisi; beklenen yanlis gecis yazili",
      g['bh'] is not None and g['bh']['family_size'] == len(SYMS)
      and g['expected_false_pass_one_sided_2_5pct'] == round(0.025 * len(SYMS), 2), str(g.get('bh')))

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
