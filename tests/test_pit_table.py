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

check("evaluate_gate v2: sembol basina MDE (aile > tek > 0,5), iki tarafli p, havuzlu tarih blogu GA",
      all(v['mde_auc_family'] > v['mde_auc_single'] > 0.5 and v['p_two_sided'] is not None
          for v in g['per_symbol'].values() if 'auc' in v)
      and g['pooled']['auc']['ci'][0] < g['pooled']['auc']['value'] < g['pooled']['auc']['ci'][1])
# hakem-finans I3 #2: bloklar arasi p seviyesi farki havuzlu AUC'u sisirir, blok ici AUC'u sisirmez
pp = np.r_[np.full(200, 0.4), np.full(200, 0.6)]
yy = np.r_[(rng.random(200) < 0.3), (rng.random(200) < 0.7)].astype(float)
gg = np.r_[np.zeros(200), np.ones(200)]
check("seviye farki: havuzlu AUC > 0,6, blok ici AUC = 0,5 (sabit p blok icinde siralama tasimaz)",
      cal.auc(pp, yy) > 0.6 and abs(P._within_block_auc(pp, yy, gg) - 0.5) < 1e-12,
      f"{cal.auc(pp, yy)}, {P._within_block_auc(pp, yy, gg)}")

print("\n[P4] P tablosu -> TradingEnv tahmin arayuzu (Milestone 0' DTF kolu)")
from env.trading_env import TradingEnv  # noqa: E402
env_panel = panel[panel.index.get_level_values('symbol').isin(SYMS)]
env_panel = env_panel[pd.to_datetime(env_panel.index.get_level_values('date')) >= pd.Timestamp('2013-06-03')]
env_dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(env_panel.index.get_level_values('date')))))
pf, prep4 = P.env_prediction_features(full, SYMS, env_dates)
check("adaptor: her (gun, sembol) hucresi dolu; sayim toplami = hucre sayisi",
      len(pf) == len(env_dates) and all(set(v) == set(SYMS) for v in pf.values())
      and sum(prep4['counts'].values()) == prep4['n_cells'], str(prep4['counts']))
okrow = full[full.meta_ok & full['date'].isin(env_dates)].iloc[0]
cell = pf[str(okrow['date'])[:10]][okrow['symbol']]
d_up = float(okrow['p_up_cal'] >= 0.5)
exp_ag = np.mean([(okrow[c] > 0) == bool(d_up) for c in ('ret_xgboost', 'ret_lightgbm', 'ret_catboost')])
check("adaptor: meta_ok hucresinde esleme elle (getiri, yon, max(p,1-p), GBM isaret uyumu)",
      cell == {'predicted_return': float(okrow['pred_logret']), 'predicted_direction': d_up,
               'confidence': float(max(okrow['p_up_cal'], 1 - okrow['p_up_cal'])),
               'ensemble_agreement': float(exp_ag)}, f"{cell}")
masked = full.copy()
mi = masked.index[masked.meta_ok & masked['date'].isin(env_dates)][5]
masked.loc[mi, ['meta_ok', 'mask_reason']] = [False, 'meta_insufficient']
pf_m, rep_m = P.env_prediction_features(masked, SYMS, env_dates)
mr = masked.loc[mi]
check("adaptor: meta_ok olmayan hucre notr ve nedeni sayiliyor (yon 0,5 -> env'de 0, DOWN degil)",
      pf_m[str(mr['date'])[:10]][mr['symbol']] == P.ENV_PRED_NEUTRAL
      and rep_m['counts'].get('meta_insufficient') == 1, str(rep_m['counts']))
pert_pf, _ = P.env_prediction_features(pert, SYMS, env_dates)
check(f"adaptor B0: t={t.date()} ve oncesi gunlerde sozluk bit-es (satir t yalniz P satiri t'den)",
      all(pf[k] == pert_pf[k] for k in pf if pd.Timestamp(k) <= t)
      and any(pf[k] != pert_pf[k] for k in pf if pd.Timestamp(k) > t))


def env_pred_obs(feats, date_key):
    e = TradingEnv(env_panel, prediction_features=feats)
    e.reset(seed=0)
    i = list(e.dates.strftime('%Y-%m-%d')).index(date_key)
    e.current_step = i
    return e._get_observation()[-len(SYMS) * 4:], str(e._get_current_date())[:10], e.symbols


k_ok = str(okrow['date'])[:10]
obs, seen, env_syms = env_pred_obs(pf, k_ok)
j = env_syms.index(okrow['symbol'])   # env sembolleri kendi sirasinda (df'ten)
expv = [np.tanh(cell['predicted_return'] * 10), cell['predicted_direction'] * 2 - 1,
        cell['confidence'] * 2 - 1, cell['ensemble_agreement'] * 2 - 1]
check("env uctan uca: gozlemin tahmin dilimi adaptor degerinin env donusumu (ayni gun)",
      seen == k_ok and np.allclose(obs[4 * j:4 * j + 4], expv, atol=1e-6), f"{seen} {obs[4 * j:4 * j + 4]} {expv}")
# mutasyon: adaptorun doldurdugu bir hucreyi silince env kendi varsayilanina duser -> yon -1 (DOWN)
holed = {k: {s: dict(v) for s, v in day.items()} for k, day in pf.items()}
del holed[k_ok][okrow['symbol']]
obs_h, _, _ = env_pred_obs(holed, k_ok)
obs_n, _, _ = env_pred_obs({k: {s: dict(P.ENV_PRED_NEUTRAL) for s in SYMS} for k in pf}, k_ok)
check("mutasyon: eksik hucrede env varsayilani yon = -1 (DOWN); adaptorun notru 4 alanda 0",
      obs_h[4 * j + 1] == -1.0 and np.allclose(obs_n, 0.0), f"{obs_h[4 * j:4 * j + 4]} {obs_n[:4]}")

print("\n[PC] Milestone 0' DTF pozitif kontrol tablosu (bilerek gelecekten)")
pcs = {}
for target in (0.55, 0.60):
    pc, pinfo = P.positive_control_table(full, labels, target, seed=7)
    pcs[target] = (pc, pinfo)
lab = full.merge(labels[['date', 'symbol', 'y_up']], on=['date', 'symbol'], how='left')
m_pc = full['meta_ok'].to_numpy() & lab['y_up'].notna().to_numpy()
auc_base = cal.auc(full.loc[m_pc, 'p_up_cal'].to_numpy(), lab.loc[m_pc, 'y_up'].to_numpy())
a55, a60 = pcs[0.55][1]['achieved_auc_meta_ok'], pcs[0.60][1]['achieved_auc_meta_ok']
check("pozitif kontrol: olculen AUC hedefe yakin (+-0,03) ve siralama 0,55 < 0,60; maske ve satirlar ayni",
      abs(a55 - 0.55) < 0.03 and abs(a60 - 0.60) < 0.03 and a55 < a60
      and pcs[0.55][0]['meta_ok'].equals(full['meta_ok']) and len(pcs[0.55][0]) == len(full),
      f"taban {auc_base:.3f}, 0,55 -> {a55:.3f}, 0,60 -> {a60:.3f}")
pf_pc, _ = P.env_prediction_features(pcs[0.60][0], SYMS, env_dates)
okc = pcs[0.60][0][pcs[0.60][0].meta_ok & pcs[0.60][0]['date'].isin(env_dates)].merge(
    labels[['date', 'symbol', 'y_up']], on=['date', 'symbol']).dropna(subset=['y_up'])
dir_acc = np.mean([pf_pc[str(r.date)[:10]][r.symbol]['predicted_direction'] == r.y_up for r in okc.itertuples()])
check("P4 adaptorunden gecince yon isabeti > 0,55 (AUC 0,60 kontrolu env'e bilgili ulasir)", dir_acc > 0.55,
      f"yon isabeti {dir_acc:.3f}")

# hakem-finans I3 ikinci okuma #1: capa seviyesi uzun blok boyunca sabitken 20'lik tarih blogu GA'yi daraltir;
# capa kumesi bootstrap'i bagimsiz birimi (capa) dogru sayar -> daha genis GA
n_c, L = 12, 60
lv = np.repeat(rng.normal(0, 1, n_c), L)
yc = (rng.random(n_c * L) < 1 / (1 + np.exp(-lv))).astype(float)
pc = 1 / (1 + np.exp(-lv))
dc = np.arange(n_c * L)
cc = np.repeat(np.arange(n_c), L)
f = lambda i: cal.auc(pc[i], yc[i])
w20 = np.ptp(np.quantile(P._date_block_bootstrap(dc, f, n_boot=300, block=20), [0.025, 0.975]))
wcl = np.ptp(np.quantile(P._cluster_bootstrap(cc, f, n_boot=300), [0.025, 0.975]))
check("capa kumesi bootstrap GA'si 20'lik tarih blogundan genis (seviye capa boyunca sabit)",
      wcl > 1.2 * w20, f"kume {wcl:.4f}, blok20 {w20:.4f}")
gs = P.evaluate_gate(full, labels, window=('2013-06-01', '2014-07-01'), n_boot=50, pooled_boot=50,
                     sensitivity=True)
se_ = gs['sensitivity']
check("duyarlilik: ana alanlar degismez; blok63/kume/tie disi/Bonferroni/MDE dagilimi var",
      gs['pooled']['auc']['value'] == g['pooled']['auc']['value'] and g.get('sensitivity') is None
      and {'pooled_block63', 'pooled_anchor_cluster', 'ties_excluded', 'p_two_sided_bonferroni',
           'mde_family_block63', 'auc_within_block_per_symbol'} <= set(se_)
      and se_['ties_excluded']['n_rows'] <= gs['n_rows'], str(list(se_)))

print("\n[P5a] pit-v2: hedef K12 yurutme ufku (O_{t+1} -> O_{t+2}; Faz 10 §3.1.3)")
OO = 'open_to_open'
sym0 = SYMS[0]
d0 = P._symbol_ohlcv(panel, sym0)
sess0 = d0.index
# (a) etiket tanimi: ham acilislardan bagimsiz hesap
lo = P.realized_labels(panel, target=OO)
l0 = lo[lo.symbol == sym0].set_index('date')['y_up']
o_ = d0['open']
exp_oo = (o_.shift(-2) > o_.shift(-1)).astype(float).where(o_.shift(-2).notna())
fr0, _, _ = P.symbol_frame(d0, sym0, 'daily', OO)
k_lab = fr0['label_date'].dropna()
check("etiket: y_up = 1[O_{t+2} > O_{t+1}] (ham acilistan), label_date = sembolun t+2. seansi",
      bool(((l0 == exp_oo) | (l0.isna() & exp_oo.isna())).all())
      and bool((k_lab.index.map(lambda x: sess0[sess0.get_loc(x) + 2]) == pd.DatetimeIndex(k_lab)).all())
      and np.allclose(fr0['y_ret'].dropna(), np.log(o_.shift(-2) / o_.shift(-1)).reindex(fr0['y_ret'].dropna().index)))
lc = P.realized_labels(panel)
c_ = d0['close']
exp_cc = (c_.shift(-1) > c_).astype(float).where(c_.shift(-1).notna())
check("close_to_close (pit-v1) etiketi degismedi: 1[C_{t+1} > C_t]",
      bool(((lc[lc.symbol == sym0].set_index('date')['y_up'] == exp_cc)
            | exp_cc.isna()).all()))
try:
    P.label_steps('daily', 'open_to_close')
    bad_target = False
except ValueError:
    bad_target = True
check("taninmayan hedef -> ValueError (sessiz varsayilan yok)", bad_target)

# (b) varsayilan yol main'deki koda gore bit-es (pit-v1 dondurulmus; M0' ona bagli)
import subprocess  # noqa: E402
import importlib.util  # noqa: E402
ref = next((r for r in ('origin/main', 'main') if subprocess.run(
    ['git', 'rev-parse', '--verify', '-q', r], capture_output=True).returncode == 0), None)
src = subprocess.run(['git', 'show', f'{ref}:prediction/pit_table.py'], capture_output=True, text=True,
                     encoding='utf-8').stdout if ref else ''
if 'open_to_open' in src or not src:
    check("varsayilan yol main ile bit-es (main zaten P5a'li ya da main yok; yerine pit-v1 surum etiketi)",
          rep['table_version'] == 'pit-v1' and full['model_version'].dropna().str.startswith('pit-v1:').all())
else:
    tmpd = tempfile.mkdtemp(prefix='pit_main_')
    try:
        mp = os.path.join(tmpd, 'pit_table_main.py')
        with open(mp, 'w', encoding='utf-8') as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location('pit_table_main', mp)
        PM = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PM)
        fm, _ = PM.build_pit_table(panel, SYMS, params_override=SMALL, ci_boot=100)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    ok, col = same_upto(full, fm, full['date'].max())
    check(f"varsayilan yol (close_to_close) {ref} ile bit-es (tum sutunlar)", ok, col)
    lm = PM.realized_labels(panel)
    check(f"realized_labels varsayilan {ref} ile bit-es",
          lm.equals(lc), '')

# (c) pit-v2 tablosu, B0
oo, orep = P.build_pit_table(panel, SYMS, params_override=SMALL, ci_boot=100, target=OO)
check("pit-v2: rapor table_version/target, model_version oneki, purge h=2",
      orep['table_version'] == 'pit-v2' and orep['target'] == OO and orep['design']['purge'].endswith('(h=2)')
      and oo['model_version'].dropna().str.startswith('pit-v2:').all() and list(oo.columns) == list(full.columns))
oo_ok = sorted({pd.Timestamp(v.split(':')[1]) for v in oo.loc[oo.meta_ok, 'model_version']})
to = cal_dates[cal_dates.get_loc(oo_ok[1]) + 20]
oop, _ = P.build_pit_table(perturb_after(panel, to), SYMS, params_override=SMALL, ci_boot=100, target=OO)
ok, col = same_upto(oo, oop, to)
check(f"pit-v2 B0: t={to.date()} (capa + 20) kadar bit-es; sonrasi farkli", ok and differs_after(oo, oop, to), col)
to2 = cal_dates[cal_dates.get_loc(oo_ok[-1]) - 1]
oop2, _ = P.build_pit_table(perturb_after(panel, to2), SYMS, params_override=SMALL, ci_boot=100, target=OO)
ok, col = same_upto(oo, oop2, to2)
check(f"pit-v2 B0: t = capa - 1 ({to2.date()})", ok, col)


# (d) etiket siniri: egitimin son satirinin GERCEK etiketi (t+2 acilisi) capadan once; bagimsiz seans sayimi
def label_boundary(tab):
    viol, n = [], 0
    for s, g in tab[tab.base_ok].groupby('symbol'):
        ss = P._symbol_ohlcv(panel, s).index
        for mv, te, le in g[['model_version', 'train_end', 'label_end']].drop_duplicates().itertuples(index=False):
            a = pd.Timestamp(mv.split(':')[1])
            real = ss[ss.get_loc(pd.Timestamp(te)) + 2]
            n += 1
            if not (real < a and pd.Timestamp(le) == real):
                viol.append((s, str(a.date()), str(pd.Timestamp(te).date()), str(real.date())))
    return n, viol


nb, vb = label_boundary(oo)
check(f"etiket siniri: {nb} (sembol, capa) blogunda label_end = train_end + 2 seans < capa", nb > 0 and not vb,
      str(vb[:3]))
P.PURGE_STEPS_OVERRIDE = 1
try:
    om, _ = P.build_pit_table(panel, SYMS, params_override=SMALL, ci_boot=100, target=OO)
finally:
    P.PURGE_STEPS_OVERRIDE = None
nm, vm = label_boundary(om)
check(f"mutasyon purge 1 seans -> etiket siniri testi KIRILIR ({len(vm)}/{nm} blok ihlal)", len(vm) > 0)

# (e) bant disi acilis etiket maskesine girer (M2 open_gap_audit kurali), sessizce kullanilmaz
dd = sess0[300]
pb = panel.copy()
mk = (pb.index.get_level_values('symbol') == sym0) & (pd.to_datetime(pb.index.get_level_values('date')) == dd)
pb.loc[mk, 'open'] = pb.loc[mk, 'open'] * 1.5          # BISTECH oncesi %20 marjin disi
lb = P.realized_labels(pb, target=OO)
lb0 = lb[lb.symbol == sym0].set_index('date')['y_up']
i_d = sess0.get_loc(dd)
hit = [sess0[i_d - 1], sess0[i_d - 2]]                 # giris = dd (t = dd-1), cikis = dd (t = dd-2)
others = lb0.drop(hit)
check("bant disi acilis: giris/cikis o gun olan iki etiket NaN, digerleri degismez",
      bool(lb0.loc[hit].isna().all() and l0.loc[hit].notna().all()
           and ((others == l0.drop(hit)) | (others.isna() & l0.drop(hit).isna())).all()))
frb, _, _ = P.symbol_frame(P._symbol_ohlcv(pb, sym0), sym0, 'daily', OO)
_, rb = P.build_pit_table(pb, [sym0], params_override=SMALL, ci_boot=100, target=OO)
check("bant disi: egitim cercevesinde maskeli + y_ret NaN; raporda sayili",
      bool(frb.loc[hit, 'band_masked'].all() and frb.loc[hit, 'y_ret'].isna().all())
      and str(hit[0].date()) in rb['open_band_mask']['per_symbol'][sym0]['masked_dates']
      and rb['open_band_mask']['n_label_masked'] >= 2, str(rb['open_band_mask']['per_symbol'][sym0]))

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
