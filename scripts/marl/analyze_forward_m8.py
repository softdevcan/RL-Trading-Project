"""Faz 11 S6: M8 ileri veri okumasi (ANA venv). Kural `docs/development/phase-11-m8-forward-annex.md` (ek; koda ustun).

Bir kez okunur: defterde en az READ_SESSIONS (126) seans birikince YALNIZ ilk 126 seans; cikti dosyasi varsa DURUR.

  - Hucre: SR = marl.selection.excess_sharpe(R, rf = R - X) (tanimsiz -> 0); d = [1 x 12], eslesme ayni tohum
    (NK: M-none s201 <-> null s213, ...).
  - Birincil (§6.7): m8_stats.contrast(d) -> Δ̄, GA (bootstrap B 10.000 tohum 0), p_isaret (R 10.000 tohum 1).
    H1/H2: 'destekler' (ayni isaret + GA 6 kat Δ̄'sini kapsar) / 'celisir' (GA 0'i ters isaretle disliyor) / 'belirsiz';
    H3/NK betimsel.
  - Ek §6.1 yazim: ga0 (tohum GA 0'i kapsiyor mu), gun bloklu GA ayni karari veriyor mu (yol_ayni / yol_farkli),
    genislik / |6 kat Δ̄|, NK yan yana.
  - §6.2 tek katli tohum kumelemesi: tam isaret-cevirme 2^12, t-GA (sd n-1), MDE80.
  - §6.3 gun bloklu dairesel bootstrap: gunler tum politikalara ortak (R, rf, X birlikte), blok 10 (+ 5, 20
    duyarlilik), B 10.000, tohum 0, SR yeniden, tanimsiz -> 0.
  - §7 denetimler (biri bozuksa 'UYARILI'); §6.4 betimsel.

    python scripts/marl/analyze_forward_m8.py --self-test
    python scripts/marl/analyze_forward_m8.py --out docs/development/measurements/phase11-forward/forward_reading.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import m8_stats as MS  # noqa: E402
from marl.selection import excess_sharpe  # noqa: E402

READ_SESSIONS = 126
FORWARD_START = '2026-11-23'
CONTRASTS = {'H1': ('M-none', 'S-esm'), 'H2': ('M-none', 'S-agr'), 'H3': ('M-gat', 'M-self'), 'NK': ('M-none', 'null')}
CONFIRMATORY = ('H1', 'H2')
SEEDS = [f's{i}' for i in range(201, 213)]
NULL_SEED = {f's{i}': f's{i + 12}' for i in range(201, 213)}       # 201 <-> 213, ..., 212 <-> 224
BLOCK, BLOCKS_SENS, B_PATH, SEED_PATH = 10, (5, 20), 10_000, 0
UNDEF_MIN, MISSING_MIN = 2, 2          # ek §7: %10 kuralinin 12 hucredeki karsiligi
COST_GRID = (0, 10, 25, 50)
COST_BPS = 20.0
T975 = {11: 2.200985}
T80 = {11: 0.875539}
ANNEX = 'docs/development/phase-11-m8-forward-annex.md'
ANNEX_TAG = 'm8-forward-annex-v1'


# --- metrik ---------------------------------------------------------------------------------------------------------
def sr_cell(R, X) -> tuple[float, bool]:
    s = excess_sharpe(np.asarray(R, float), rf=np.asarray(R, float) - np.asarray(X, float))
    return (0.0, True) if s is None else (s, False)


def sr_matrix(R: np.ndarray, X: np.ndarray, periods: int = 252) -> np.ndarray:
    """Vektor excess_sharpe(R, rf = R - X) (ayni islem sirasi; tanimsiz -> 0). R, X [..., n]."""
    ex = R - (R - X)
    sd_r = R.std(axis=-1, ddof=1)
    sd = ex.std(axis=-1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = ex.mean(axis=-1) / sd * np.sqrt(periods)
    return np.where((sd_r == 0) | (sd == 0) | ~np.isfinite(s), 0.0, s)


def pair_sid(c: str, arm: str, seed: str) -> str:
    return f"policy/{arm}/{NULL_SEED[seed] if (c == 'NK' and arm == 'null') else seed}"


def verdict(delta: float, ci, delta6: float) -> str:
    lo, hi = ci
    if np.sign(delta) == np.sign(delta6) and delta != 0 and lo <= delta6 <= hi:
        return 'destekler'
    if (delta6 > 0 and hi < 0) or (delta6 < 0 and lo > 0):
        return 'celisir'
    return 'belirsiz'


def seed_cluster_1fold(d: np.ndarray) -> dict:
    """Ek §6.2: tek katli kumeleme (F tanimsiz, raporlanmaz)."""
    sm = d[np.isfinite(d)]
    n = sm.size
    obs = abs(sm.mean())
    flips = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    p = int((np.abs((flips * sm).mean(axis=1)) >= obs - 1e-12).sum()) / flips.shape[0]
    se = sm.std(ddof=1) / np.sqrt(n)
    if n - 1 in T975:
        t975, t80 = T975[n - 1], T80[n - 1]
    else:
        from scipy.stats import t
        t975, t80 = float(t.ppf(0.975, n - 1)), float(t.ppf(0.80, n - 1))
    return {'n': int(n), 'p_exact': p, 't_ci95': [float(sm.mean() - t975 * se), float(sm.mean() + t975 * se)],
            'mde80': float((t975 + t80) * se)}


def circular_block_indices(n: int, block: int, rng) -> np.ndarray:
    from evaluation.regime_contrast import circular_block_indices as cbi
    return cbi(n, block, rng)


def path_bootstrap(R: dict, X: dict, pairs: dict, block: int, B: int = B_PATH, seed: int = SEED_PATH) -> dict:
    """Gun bloklu GA: her tekrarda ayni gun indisleri tum stratejilere (R, rf, X birlikte); kontrast basina Δ̄*."""
    sids = sorted({s for pr in pairs.values() for ab in pr for s in ab})
    k = {s: i for i, s in enumerate(sids)}
    Rm = np.stack([R[s] for s in sids])
    Xm = np.stack([X[s] for s in sids])
    n = Rm.shape[1]
    rng = np.random.default_rng(seed)
    star = {c: np.empty(B) for c in pairs}
    for b in range(B):
        idx = circular_block_indices(n, block, rng)
        sr = sr_matrix(Rm[:, idx], Xm[:, idx])
        for c, pr in pairs.items():
            star[c][b] = float(np.mean([sr[k[a]] - sr[k[bb]] for a, bb in pr]))
    return {c: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for c, v in star.items()}


# --- okuma ----------------------------------------------------------------------------------------------------------
def window(rows: pd.DataFrame, n: int = READ_SESSIONS) -> tuple[list, dict]:
    """Ilk n seans (tum stratejilerde ayni tarihler). Donus: tarihler, {sid: alt-cerceve}."""
    per = {sid: g.sort_values('row') for sid, g in rows.groupby('sid')}
    dates = None
    for sid, g in per.items():
        d = list(g['date'])[:n]
        if len(d) < n:
            raise SystemExit(f"{sid}: {len(d)} seans < {n}; okuma yapilmaz")
        if dates is None:
            dates = d
        elif d != dates:
            raise SystemExit(f"{sid}: tarihler diger stratejilerden farkli")
    return dates, {sid: g.iloc[:n] for sid, g in per.items()}


def analyze(rows: pd.DataFrame, delta6: dict, labels6: dict, n_sessions: int = READ_SESSIONS) -> dict:
    dates, per = window(rows, n_sessions)
    R = {s: g['R'].to_numpy(float) for s, g in per.items()}
    X = {s: g['X'].to_numpy(float) for s, g in per.items()}
    cell = {s: sr_cell(R[s], X[s]) for s in per}
    out = {'window': {'first': dates[0], 'last': dates[-1], 'n_sessions': len(dates)}, 'contrasts': {}}
    pairs, undef = {}, {}
    for c, (a, b) in CONTRASTS.items():
        pr, d, und, miss = [], [], {a: 0, b: 0}, {a: 0, b: 0}
        for s in SEEDS:
            sa, sb = pair_sid(c, a, s), pair_sid(c, b, s)
            ok = True
            for arm, sid in ((a, sa), (b, sb)):
                if sid not in cell:
                    miss[arm] += 1
                    ok = False
                elif cell[sid][1]:
                    und[arm] += 1
            if ok:
                pr.append((sa, sb))
                d.append(cell[sa][0] - cell[sb][0])
            else:
                d.append(np.nan)
        pairs[c] = pr
        undef[c] = und
        dm = np.asarray(d, float)[None, :]
        res = MS.contrast(dm, a, b)
        r = {'arms': [a, b], 'delta_bar': res['delta_bar'], 'ci95': res['ci95'], 'p_signflip': res['p_signflip'],
             'p_bootstrap': res['p_bootstrap'], 'n_pairs': res['n_pairs'], 'cells': dm[0].tolist(),
             'undefined': und, 'missing': miss, 'cluster': seed_cluster_1fold(dm[0])}
        r['ga0'] = 'ga0_kapsar' if res['ci95'][0] <= 0 <= res['ci95'][1] else 'ga0_dislar'
        r['label6'] = labels6.get(c)
        r['delta6'] = delta6.get(c)
        r['width_ratio'] = (res['ci95'][1] - res['ci95'][0]) / abs(delta6[c]) if delta6.get(c) else None
        flags = []
        if max(und.values()) >= UNDEF_MIN or abs(und[a] - und[b]) >= UNDEF_MIN:
            flags.append('tanimsiz_uyarisi')
            keep = [i for i, s in enumerate(SEEDS) if np.isfinite(dm[0, i]) and not cell[pair_sid(c, a, s)][1]
                    and not cell[pair_sid(c, b, s)][1]]
            dd = dm[:, keep]
            r['undefined_dropped'] = {k: v for k, v in MS.contrast(dd, a, b).items() if k != 'label'} if keep else None
        if max(miss.values()) >= MISSING_MIN:
            flags.append('EKSIK')
        r['flags'] = flags
        out['contrasts'][c] = r
    for blk in (BLOCK,) + BLOCKS_SENS:
        ci = path_bootstrap(R, X, pairs, blk)
        for c, v in ci.items():
            out['contrasts'][c]['path_ci95' if blk == BLOCK else f'path_ci95_block{blk}'] = v
    for c, r in out['contrasts'].items():
        if c in CONFIRMATORY and 'EKSIK' not in r['flags']:
            r['verdict'] = verdict(r['delta_bar'], r['ci95'], r['delta6'])
            r['path_verdict'] = verdict(r['delta_bar'], r['path_ci95'], r['delta6'])
            r['path'] = 'yol_ayni' if r['verdict'] == r['path_verdict'] else 'yol_farkli'
            r['wording'] = ('cumle_kurulmaz' if r['path'] == 'yol_farkli' else
                            'celismedi_isaret_ayni' if (r['verdict'] == 'destekler' and r['ga0'] == 'ga0_kapsar')
                            else r['verdict'])
        else:
            r['verdict'] = 'EKSIK' if 'EKSIK' in r['flags'] else 'betimsel'
        r['nk_side_by_side'] = {'delta_bar': out['contrasts']['NK']['delta_bar'], 'ci95': out['contrasts']['NK']['ci95']} \
            if c in CONFIRMATORY else None
    out['descriptive'] = descriptive(per, cell)
    return out


def descriptive(per: dict, cell: dict) -> dict:
    arms = {}
    for sid, (sr, und) in cell.items():
        kind, *rest = sid.split('/')
        key = rest[0] if kind == 'policy' else sid
        arms.setdefault(key, []).append(sr)
    out = {'arms': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'n': len(v)}
                    for k, v in sorted(arms.items())}}
    cost = {}
    for sid, g in per.items():
        R, X, tau = (g[k].to_numpy(float) for k in ('R', 'X', 'turnover'))
        c0 = COST_BPS / 2 / 1e4
        row = {}
        for bps in COST_GRID:
            Rc = (1 + R) * (1 - bps / 2 / 1e4 * tau) / (1 - c0 * tau) - 1
            row[bps] = sr_cell(Rc, Rc - (R - X))[0]
        cost[sid] = row
    out['cost_grid_sr'] = cost
    out['turnover_mean'] = {s: float(g['turnover'].mean()) for s, g in per.items()}
    out['cash_mean'] = {s: float(g['cash_weight'].mean()) for s, g in per.items()}
    out['sell_blocked'] = {s: int(g['n_sell_blocked'].sum()) for s, g in per.items()}
    out['undefined_cells'] = sorted(s for s, (_, u) in cell.items() if u)
    return out


# --- denetimler (ek §7) ---------------------------------------------------------------------------------------------
def git(*a) -> str:
    return subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()


def checks(rows: pd.DataFrame, meta: dict, dates: list) -> dict:
    from marl import forward_job as FJ
    from marl import forward_panel as FP
    from marl import frozen as FZ
    from marl import panel as PN
    c = {}
    with open(os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'm8_reading.json'), encoding='utf-8') as fh:
        fwd = json.load(fh)['forward_policies_2026']
    try:
        v = FZ.verify_archive(FJ.ARCHIVE, forward=fwd)
        c['archive'] = {'ok': True, 'n': v.get('n_policies')}
    except Exception as e:   # noqa: BLE001
        c['archive'] = {'ok': False, 'error': str(e)}
    try:
        FP.verify_source()
        a = PN.load_npz(os.path.join(FP.FWD_ROOT, FP.PANEL_FWD))
        p = FP.check_prefix(a, PN.load_npz(FP.PANEL_M8))
        c['panel_prefix'] = {'ok': True, 'rows': p['prefix_rows']}
    except Exception as e:   # noqa: BLE001
        c['panel_prefix'] = {'ok': False, 'error': str(e)}
    r = subprocess.run([sys.executable, os.path.join('scripts', 'marl', 's3_parity.py')], capture_output=True, text=True)
    c['s3_parity'] = {'ok': r.returncode == 0, 'tail': r.stdout.strip().splitlines()[-1:] if r.stdout else []}
    sids = set(rows['sid'])
    c['ledger'] = {'ok': len(sids) == 76 and dates[0] == FORWARD_START, 'n_strategies': len(sids), 'first': dates[0]}
    tag_time = git('log', '-1', '--format=%cI', ANNEX_TAG)
    created = meta.get('created_at', '')
    c['store_after_freeze'] = {'ok': bool(tag_time) and created[:19] > tag_time[:19], 'annex_tag_time': tag_time,
                               'store_created': created}
    stops, last_n = [], None
    logp = os.path.join(FP.FWD_ROOT, FP.LOG)
    if os.path.exists(logp):
        with open(logp, encoding='utf-8') as fh:
            for line in fh:
                e = json.loads(line)
                if e.get('kind') == 'ledger' and (e.get('ledger') or {}).get('n_sessions') is not None:
                    last_n = e['ledger']['n_sessions']
                for d in e.get('days', []) if e.get('kind') is None else []:
                    if not d.get('accept') and not d.get('skip') and d.get('date', '') <= dates[-1] \
                            and d.get('date') not in set(rows['date']):
                        stops.append(d)
    c['log'] = {'ok': last_n is not None and last_n >= READ_SESSIONS, 'last_logged_sessions': last_n,
                'n_unresolved_stops_in_window': len({d['date'] for d in stops})}
    c['log']['ok'] &= c['log']['n_unresolved_stops_in_window'] == 0
    c['all_ok'] = all(v['ok'] for v in c.values() if isinstance(v, dict) and 'ok' in v)
    return c


def input_support(dates: list) -> dict:
    """Ek §6.4: ozellik basina, politikanin egitim norm_stats'iyla |z| > 5 (kirpilan) ve |z| > 3 orani; ileri pencere
    (karar satirlari = yurutme gunlerinden onceki satirlar) ve egitim penceresi (fit_rows). 72 politikanin ortalamasi."""
    from marl import forward_job as FJ
    from marl import forward_panel as FP
    from marl import obs_spec as OS
    from marl import panel as PN
    a = PN.load_npz(os.path.join(FP.FWD_ROOT, FP.PANEL_FWD))
    feat, ok = OS.compute_raw_features(a)
    d = np.asarray(a['dates'])
    exec_rows = np.flatnonzero(np.isin(d, dates))
    dec = exec_rows - 1
    acc = {}
    for p in FJ.manifest()['policies']:
        with open(os.path.join(FJ.ARCHIVE, p['arm'], p['seed'], 'norm_stats.json'), encoding='utf-8') as fh:
            ns = json.load(fh)
        z = OS.normalize(feat, ok, ns, clip=np.inf)
        f0, f1 = ns['fit_rows']
        for j, name in enumerate(OS.FEATURES_V1):
            zf, of = np.abs(z[dec, :, j]), ok[dec]
            zt, ot = np.abs(z[f0:f1 + 1, :, j]), ok[f0:f1 + 1]
            r = acc.setdefault(name, {'fwd_gt5': [], 'fwd_gt3': [], 'train_gt3': []})
            r['fwd_gt5'].append(float((zf[of] > 5).mean()))
            r['fwd_gt3'].append(float((zf[of] > 3).mean()))
            r['train_gt3'].append(float((zt[ot] > 3).mean()))
    return {k: {q: float(np.mean(v)) for q, v in r.items()} for k, r in acc.items()}


def provenance() -> dict:
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(ANNEX, 'rb') as fh:
        annex_sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')), 'script_sha256': sha,
            'annex_sha256': annex_sha, 'annex_tag': git('rev-parse', '--verify', '--quiet', ANNEX_TAG) or None}


def reading_6fold() -> tuple[dict, dict]:
    with open(os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'm8_reading.json'), encoding='utf-8') as fh:
        con = json.load(fh)['contrasts']
    return {c: con[c]['delta_bar'] for c in CONTRASTS}, {c: con[c]['label'] for c in CONTRASTS}


# --- oz-test --------------------------------------------------------------------------------------------------------
def _synth(n: int, eff: dict, seed: int = 3, undef_sids=(), drop_sids=(), n_extra: int = 5) -> pd.DataFrame:
    """Sentetik defter: kol etkisi eff[arm] (gunluk fazla getiri kaymasi), ortak piyasa yolu + tohum gurultusu."""
    rng = np.random.default_rng(seed)
    T = n + n_extra
    dates = pd.bdate_range(FORWARD_START, periods=T).strftime('%Y-%m-%d').tolist()
    mkt = rng.normal(0.0005, 0.012, T)
    rf = np.full(T, 0.0015)
    recs = []
    sids = [f'policy/{a}/{s}' for a in ('M-none', 'S-esm', 'S-agr', 'M-gat', 'M-self') for s in SEEDS] + \
        [f'policy/null/{NULL_SEED[s]}' for s in SEEDS] + [f'rule/{k}' for k in ('ew', 'bh', 'minvar', 'hrp')]
    for sid in sids:
        if sid in drop_sids:
            continue
        arm = sid.split('/')[1]
        if sid in undef_sids:
            R = rf.copy()
        else:
            R = rf + mkt + eff.get(arm, 0.0) + rng.normal(0, 0.004, T)
        X = R - rf
        for i in range(T):
            recs.append({'sid': sid, 'date': dates[i], 'row': 4000 + i, 'R': R[i], 'X': X[i], 'rf': rf[i],
                         'turnover': 0.1, 'cost_frac': 1e-4, 'cash_weight': 0.05, 'value': 1.0, 'n_sell_blocked': 0})
    return pd.DataFrame(recs)


def self_test() -> int:
    ok_all = []

    def chk(name, ok):
        ok_all.append(bool(ok))
        print(f"[{'OK' if ok else 'HATA'}] {name}")

    d6 = {'H1': 0.465, 'H2': 0.281, 'H3': -0.01, 'NK': 0.073}
    l6 = {'H1': 'FARK_VAR_M-none', 'H2': 'FARK_VAR_M-none', 'H3': 'FARK_GOZLENMEDI', 'NK': 'FARK_GOZLENMEDI'}
    # 1 vektor SR = excess_sharpe
    rng = np.random.default_rng(0)
    R = rng.normal(0.001, 0.01, (5, 126))
    X = R - 0.0015
    v = sr_matrix(R, X)
    chk("1. vektor SR = excess_sharpe(R, rf = R - X) (5 seri) ve tanimsiz -> 0",
        all(abs(v[i] - excess_sharpe(R[i], rf=R[i] - X[i])) < 1e-12 for i in range(5))
        and sr_matrix(np.full((1, 10), 0.001), np.zeros((1, 10)))[0] == 0.0)
    # 2 pencere: ilk 126, fazlasi okunmaz; eksikse durur
    rows = _synth(READ_SESSIONS, {'M-none': 0.0006})
    dates, per = window(rows)
    stop = False
    try:
        window(rows[rows['row'] < 4000 + 100])
    except SystemExit:
        stop = True
    chk("2. pencere tam ilk 126 seans (fazla 5 seans okunmaz); 100 seansta okuma durur",
        len(dates) == 126 and dates[0] == FORWARD_START and all(len(g) == 126 for g in per.values()) and stop)
    # 3 karar kurallari
    chk("3. karar: destekler / celisir / belirsiz (+ ters isaretli 6 kat)",
        verdict(0.4, [0.1, 0.7], 0.465) == 'destekler' and verdict(0.4, [-0.1, 0.9], 0.465) == 'destekler'
        and verdict(-0.3, [-0.5, -0.1], 0.465) == 'celisir' and verdict(0.1, [-0.2, 0.3], 0.465) == 'belirsiz'
        and verdict(0.9, [0.6, 1.2], 0.465) == 'belirsiz' and verdict(0.3, [0.1, 0.5], -0.2) == 'celisir')
    # 4 buyuk etki -> H1/H2 destekler/belirsiz, sema tam; NK etkisiz
    out = analyze(rows, d6, l6)
    need = {'delta_bar', 'ci95', 'p_signflip', 'p_bootstrap', 'verdict', 'ga0', 'path_ci95', 'path_verdict', 'path',
            'width_ratio', 'cluster', 'undefined', 'missing', 'path_ci95_block5', 'path_ci95_block20', 'wording'}
    h1 = out['contrasts']['H1']
    chk("4. sema (ek §8) tam; H1 etkisi pozitif, kume p tam sayim 4.096, NK betimsel ve H1 satirinda yan yana",
        need <= set(h1) and h1['delta_bar'] > 0 and h1['cluster']['n'] == 12
        and out['contrasts']['NK']['verdict'] == 'betimsel' and h1['nk_side_by_side'] is not None
        and h1['path'] in ('yol_ayni', 'yol_farkli'))
    # 5 tohum GA'si yalniz tohum gurultusunu gorur: ortak piyasa yolu tohum GA'sini daraltir, gun bloklu GA genis
    chk("5. gun bloklu GA (ortak piyasa yolu) tohum GA'sindan genis (§6.1 gerekcesi)",
        (h1['path_ci95'][1] - h1['path_ci95'][0]) > (h1['ci95'][1] - h1['ci95'][0]))
    # 6 tanimsiz ve eksik kurallari
    und = [f'policy/S-esm/{s}' for s in SEEDS[:3]]
    o2 = analyze(_synth(READ_SESSIONS, {'M-none': 0.0006}, undef_sids=und), d6, l6)
    o3 = analyze(_synth(READ_SESSIONS, {}, drop_sids=[f'policy/S-agr/{s}' for s in SEEDS[:2]]), d6, l6)
    chk("6. tanimsiz 3 hucre -> tanimsiz_uyarisi + dusurulmus duyarlilik (9 es); 2 eksik hucre -> EKSIK",
        'tanimsiz_uyarisi' in o2['contrasts']['H1']['flags'] and o2['contrasts']['H1']['undefined']['S-esm'] == 3
        and o2['contrasts']['H1']['undefined_dropped']['n_pairs'] == 9
        and o3['contrasts']['H2']['verdict'] == 'EKSIK' and 'EKSIK' not in o3['contrasts']['H1']['flags'])
    # 7 gun bloklu bootstrap ortak indis: ayni seri iki stratejide -> fark her tekrarda 0
    rr = {'a': R[0], 'b': R[0].copy()}
    xx = {'a': X[0], 'b': X[0].copy()}
    ci = path_bootstrap(rr, xx, {'T': [('a', 'b')]}, 10, B=200)
    chk("7. gun bloklu bootstrap gunleri stratejilere ORTAK cizer (ayni seri -> GA [0, 0])", ci['T'] == [0.0, 0.0])
    # 8 cikarim yolu (ek §8): M8 test kaydindaki bir politikayla bir kez es (varsa)
    npz = os.path.join('results', 'marl_m8', '2026', 'M-none', 's201', 'eval', 'test_argmax.npz')
    if os.path.exists(npz) and os.path.exists(os.path.join('data', 'marl', 'panel_m8.npz')):
        from marl import forward_job as FJ
        from marl import panel as PN
        from marl import sim_ledger as SL
        with open(os.path.join(FJ.ARCHIVE, 'M-none', 's201', 'fold_eval.json'), encoding='utf-8') as fh:
            ts, te = json.load(fh)['rows']['test']
        mk = SL.Market(PN.load_npz(os.path.join('data', 'marl', 'panel_m8.npz')))
        st = FJ.make_strategy('policy/M-none/s201')
        ser = SL.series(SL.run(mk, st, SL.LedgerState.start(ts, mk.N), te)[1])
        z = np.load(npz)
        from marl import obs_spec as OS
        chk("8. cikarim yolu: M-none s201 2026 defteri M8 test kaydiyla bit-es; kirpma = obs_spec.CLIP 5",
            np.array_equal(ser['R'], z['policy_R']) and OS.CLIP == 5.0 and st.spec.clip == 5.0)
    else:
        print("[ATLANDI] 8. M8 test kaydi / panel yok")
    # 9 cikti dosyasi varsa durur (tek okuma)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'r.json')
        open(p, 'w').write('{}')
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--out', p], capture_output=True, text=True)
        chk("9. cikti dosyasi varsa okuma durur (ikinci okuma yok)", r.returncode != 0 and 'zaten' in (r.stdout + r.stderr))
    n_ok = sum(ok_all)
    print(f"\n{'=' * 60}\nSonuc: {n_ok} gecti, {len(ok_all) - n_ok} kaldi")
    return 0 if all(ok_all) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not a.out:
        raise SystemExit('--out gerekli')
    if os.path.exists(a.out):
        raise SystemExit(f"okuma zaten yapildi: {a.out} (ikinci okuma yok)")
    from marl import forward_job as FJ
    meta, states, rows = FJ.load_store(FJ.LEDGER_DIR)
    if rows.empty:
        raise SystemExit("defter bos")
    d6, l6 = reading_6fold()
    out = analyze(rows, d6, l6)
    dates, _ = window(rows)
    out['descriptive']['input_support'] = input_support(dates)
    out['checks'] = checks(rows, meta, dates)
    out['status'] = 'TAMAM' if out['checks']['all_ok'] else 'UYARILI'
    out['provenance'] = provenance()
    out['annex'] = ANNEX
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    for c, r in out['contrasts'].items():
        print(f"{c}: Δ̄ {r['delta_bar']:+.3f} GA [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] yol GA "
              f"[{r['path_ci95'][0]:+.3f}, {r['path_ci95'][1]:+.3f}] -> {r['verdict']} {r.get('path', '')} {r['flags']}")
    print('durum:', out['status'])


if __name__ == '__main__':
    main()
