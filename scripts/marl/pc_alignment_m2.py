"""Faz 10 M2 tasarim §9.1 ON KAYDI (commit 0232f11, sonuctan once): PC060 yeni ortamda ufuk hizasi.

PC060 (y_up = 1[C_{t+1} > C_t], sentetik AUC 0,60) RL'siz kural portfoyunde iki yurutmeyle olculur:
  E0: t kapanisinda yurutme (eski ortam; Faz 9 §9.38 referansi), getiri Cf_{t+1}/Cf_t.
  E1: K12, t+1 acilisinda yurutme; ORTAMIN KENDI muhasebesi (`accounting.step_fractional`, `fill_prices`
      kilit 20 + taban satis engeli, `project`, A8 rf). Kopya mantik yok.
Birincil: E1 net (20 bps) dSR(S_yon - 1/N), 2013-08-27..2020-12-31, esik 0,17. < 0,17 ise on kayit geregi
PC060-K12 (ayni uretec, etiket 1[Of_{t+2} > Of_{t+1}]) ayni sekilde olculur. 2021 ve sonrasi OKUNMAZ.
Sinyal BILEREK gelecekten: ust sinir olcumu, strateji onerisi degil.

Hakem-2 M2 kapanisi sonrasi (SONUCTAN SONRA, betimsel; birincil sayilar degismez):
  D_on_next: t+1 -> t+2 gecesi (K12 ufkunun gece parcasi; #4). --extra-seeds: uretec tohum yayilimi (#3, #5);
  PC060 yeniden uretimi kapanis->kapanis etiketini paneldan kurar (orijinal etiket dosyasi degil; beyan).
  JSON'a git durumu ve betik SHA-256'si yazilir (#8).

    python scripts/marl/pc_alignment_m2.py --out docs/development/measurements/phase10-m2/pc_alignment.json
    python scripts/marl/pc_alignment_m2.py --out .../pc_alignment.json --extra-seeds 1-8 --seeds-out .../pc_alignment_seeds.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
PANEL_SHA = 'a5dcf9a607e0447df490f0a47c5187555f428faddc62b47d5eef81819e5e147b'
PROTOCOL = os.path.join('docs', 'development', 'milestone0', 'protocol-v2.json')
COST = 0.001
THRESHOLD = 0.17
ANN = np.sqrt(252.0)
WINDOWS = {'train': ('2013-08-27', '2018-12-31'), 'val': ('2019-01-02', '2020-12-31'),
           'pooled': ('2013-08-27', '2020-12-31')}
PRIMARY = 'pooled'


def signal_weights(p_up, ok, valid_t):
    """S_yon: p_up_cal >= 0,5 ve meta_ok ve t'de gecerli -> esit agirlik; hic yoksa 1/N (gecerli)."""
    sel = ok & valid_t & np.isfinite(p_up) & (p_up >= 0.5)
    base = sel if sel.any() else valid_t
    return base.astype(float) / max(float(base.sum()), 1.0)


def equal_weights(valid_t):
    return valid_t.astype(float) / max(float(valid_t.sum()), 1.0)


def run(arrays, px, P, OK, rows, A, W):
    """rows: karar satirlari t (getiri t+1'e yazilir). Donus: gunluk seriler ve ayristirma."""
    rf = arrays['rf']
    N = P.shape[1]
    out = {k: [] for k in ('E0_S', 'E0_EW', 'E0_S_gross', 'E0_EW_gross',
                           'E1_S', 'E1_EW', 'E1_S_gross', 'E1_EW_gross', 'D_on', 'D_day', 'D_on_next', 'rf',
                           'tau_E1_S', 'tau_E1_EW', 'sell_blocked_E1_S')}
    w1 = {k: np.r_[1.0, np.zeros(N)] for k in ('S', 'EW')}              # E1 kapanis agirligi (nakitten)
    w1g = {k: np.r_[1.0, np.zeros(N)] for k in ('S', 'EW')}             # E1 brut
    w0 = {k: np.zeros(N) for k in ('S', 'EW', 'Sg', 'EWg')}             # E0 kaymis onceki agirlik
    cf = px['close']
    for t in rows:
        v = px['valid'][t]
        ws = signal_weights(P[t], OK[t], v)
        ew = equal_weights(v)
        r_f = float(rf[t + 1])
        out['rf'].append(r_f)
        # E0: t kapanisinda yurutme (Faz 9 §9.38 run_strategy ile ayni maliyet tanimi)
        with np.errstate(invalid='ignore', divide='ignore'):
            g = cf[t + 1] / cf[t]
        g = np.where(np.isfinite(g), g, 1.0)
        for key, w, gross in (('E0_S', ws, False), ('E0_EW', ew, False), ('E0_S_gross', ws, True),
                              ('E0_EW_gross', ew, True)):
            slot = {'E0_S': 'S', 'E0_EW': 'EW', 'E0_S_gross': 'Sg', 'E0_EW_gross': 'EWg'}[key]
            c = 0.0 if gross else COST * float(np.abs(w - w0[slot]).sum())
            growth = float((w * g).sum())
            out[key].append(growth * (1.0 - c) - 1.0 - r_f)
            w0[slot] = w * g / growth
        # E1: ortamin muhasebesi
        g_on, g_day, tradable = A.step_growth(px, t)
        ns = px['no_sell'][t + 1]
        for k, w in (('S', ws), ('EW', ew)):
            u = np.r_[0.0, w]
            res = A.step_fractional(w1[k], u, g_on, g_day, tradable, r_f, COST, W.Constraints(), None, ns)
            out[f'E1_{k}'].append(res.X)
            out[f'tau_E1_{k}'].append(res.turnover)
            if k == 'S':
                out['sell_blocked_E1_S'].append(res.projection['n_sell_blocked'])
            w1[k] = res.w_next
            rg = A.step_fractional(w1g[k], u, g_on, g_day, tradable, r_f, 0.0, W.Constraints(), None, ns)
            out[f'E1_{k}_gross'].append(rg.X)
            w1g[k] = rg.w_next
        # betimsel ayristirma (brut): sinyal - esit agirlik farkinin gece ve seans ici katkisi
        d = ws - ew
        out['D_on'].append(float(np.dot(d, g_on - 1.0)))
        out['D_day'].append(float(np.dot(d, g_day - 1.0)))
        g_next = A.step_growth(px, t + 1)[0] if t + 2 < len(arrays['dates']) else np.ones(N)
        out['D_on_next'].append(float(np.dot(d, g_next - 1.0)))   # K12 ufkunun gecesi (t+1 -> t+2)
    dates = pd.Index(arrays['dates'][np.asarray(rows) + 1])
    return pd.DataFrame(out, index=dates)


def summarize(df):
    from evaluation.sharpe_tests import sharpe_diff_hac

    def sr(x):
        return float(x.mean() / x.std(ddof=1) * ANN)

    def diff(a, b):
        r = sharpe_diff_hac(df[a].to_numpy(), df[b].to_numpy())
        return {'delta_sharpe_annual': r['delta'] * ANN, 'se_annual': r['se'] * ANN, 'p_value': r['p_value']}

    tot = df['D_on'].mean() + df['D_day'].mean()
    tot_k12 = df['D_day'].mean() + df['D_on_next'].mean()
    return {
        'n_days': int(len(df)), 'first': str(df.index[0]), 'last': str(df.index[-1]),
        'sharpe_annual': {c: sr(df[c]) for c in df.columns if c.startswith('E')},
        'tests': {'E1 net S_yon - 1/N': diff('E1_S', 'E1_EW'), 'E1 brut S_yon - 1/N': diff('E1_S_gross', 'E1_EW_gross'),
                  'E0 net S_yon - 1/N': diff('E0_S', 'E0_EW'), 'E0 brut S_yon - 1/N': diff('E0_S_gross', 'E0_EW_gross')},
        'decomposition_gross': {'D_on_mean_daily': float(df['D_on'].mean()), 'D_day_mean_daily': float(df['D_day'].mean()),
                                'intraday_share': float(df['D_day'].mean() / tot) if tot != 0 else None,
                                'D_on_next_mean_daily': float(df['D_on_next'].mean()),
                                'intraday_share_k12_horizon': float(df['D_day'].mean() / tot_k12) if tot_k12 != 0 else None,
                                'note': 'intraday_share = D_day/(D_on+D_day): PC060 (kapanis->kapanis) ufku icin; '
                                        'K12 ufku (acilis->acilis) icin intraday_share_k12_horizon = D_day/(D_day+D_on_next)'},
        'turnover_mean': {'E1_S': float(df['tau_E1_S'].mean()), 'E1_EW': float(df['tau_E1_EW'].mean())},
        'sell_blocked_steps_E1_S': int(df['sell_blocked_E1_S'].sum()),
    }


def pivots(table, dates, symbols):
    t = table.copy()
    t['date'] = pd.to_datetime(t['date']).dt.tz_localize(None).dt.normalize().dt.strftime('%Y-%m-%d')
    p = t.pivot_table(index='date', columns='symbol', values='p_up_cal', aggfunc='first')
    o = t.pivot_table(index='date', columns='symbol', values='meta_ok', aggfunc='first')
    P = p.reindex(index=list(dates), columns=list(symbols)).to_numpy(dtype=np.float64)
    OK = o.reindex(index=list(dates), columns=list(symbols)).fillna(False).to_numpy(dtype=bool)
    return P, OK


def close_labels(arrays, px):
    """PC060 etiketinin panelden kurulumu: y(t) = 1[C_{t+1} > C_t], t ve t+1 gecerliyse (yalniz tohum yayilimi)."""
    c, v = arrays['close'], px['valid']
    rec = []
    for t in range(c.shape[0] - 1):
        for i in np.flatnonzero(v[t] & v[t + 1]):
            rec.append((arrays['dates'][t], str(arrays['symbols'][i]), float(c[t + 1, i] > c[t, i])))
    lab = pd.DataFrame(rec, columns=['date', 'symbol', 'y_up'])
    lab['date'] = pd.to_datetime(lab['date'])
    return lab


def _seeds(spec):
    a, b = (spec.split('-') + [spec])[:2]
    return list(range(int(a), int(b) + 1))


def _provenance():
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def evaluate(name, table, arrays, px, A, W, quiet=False):
    dates = arrays['dates']
    P, OK = pivots(table, dates, arrays['symbols'])
    lo, hi = WINDOWS[PRIMARY]
    rows = [t for t in range(len(dates) - 1) if lo <= dates[t + 1] <= hi]
    df = run(arrays, px, P, OK, rows, A, W)
    res = {w: summarize(df.loc[(df.index >= a) & (df.index <= b)]) for w, (a, b) in WINDOWS.items()}
    gate = res[PRIMARY]['tests']['E0 brut S_yon - 1/N']
    res['alignment_gate'] = {'rule': 'E0 brut S_yon 1/N\'in acikca ustunde (dSR > 0, p < 0,01); degilse olcum durur',
                             'passed': bool(gate['delta_sharpe_annual'] > 0 and gate['p_value'] < 0.01), **gate}
    res['primary_delta'] = res[PRIMARY]['tests']['E1 net S_yon - 1/N']['delta_sharpe_annual']
    res['primary_exceeds_threshold'] = bool(res['primary_delta'] >= THRESHOLD)
    if not quiet:
        print(f"[{name}] hizalama kapisi {res['alignment_gate']['passed']} (E0 brut dSR "
              f"{gate['delta_sharpe_annual']:+.2f}); E1 net dSR {res['primary_delta']:+.3f} (esik {THRESHOLD})")
    return res


def k12_labels(arrays, px):
    """PC060-K12 etiketi: y(t) = 1[Of_{t+2} > Of_{t+1}], t+1 ve t+2 gecerliyse; aksi halde etiket yok."""
    o, v = arrays['open'], px['valid']
    T, N = o.shape
    rec = []
    for t in range(T - 2):
        m = v[t + 1] & v[t + 2]
        for i in np.flatnonzero(m):
            rec.append((arrays['dates'][t], str(arrays['symbols'][i]), float(o[t + 2, i] > o[t + 1, i])))
    lab = pd.DataFrame(rec, columns=['date', 'symbol', 'y_up'])
    lab['date'] = pd.to_datetime(lab['date'])
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra-seeds', default=None, help='or. 1-8: betimsel tohum yayilimi (birincil karar degismez)')
    ap.add_argument('--seeds-out', default=None)
    a = ap.parse_args()
    from marl import accounting as A
    from marl import panel as PN
    from marl import weights as W
    from prediction.pit_table import load_table, positive_control_table

    arrays = PN.load_npz(PANEL)
    if arrays['sha256'] != PANEL_SHA:
        raise SystemExit(f"panel SHA {arrays['sha256']} != on kayit {PANEL_SHA}")
    if max(arrays['dates']) > '2020-12-31':
        raise SystemExit('panel 2021+ iceriyor; M2 olcumu 2021 okumaz')
    px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, A.LIMIT_DOWN_GAP)
    proto = json.load(open(PROTOCOL, encoding='utf-8'))['prediction_levels']['PC060']
    from app.auth import workspace as ws
    table, tinfo = load_table(proto['table'])
    if tinfo['sha256'] != proto['table_sha256']:
        raise SystemExit(f"PC060 tablo SHA {tinfo['sha256']} != protokol {proto['table_sha256']}")

    res = {'PC060': evaluate('PC060', table, arrays, px, A, W)}
    if not res['PC060']['alignment_gate']['passed']:
        raise SystemExit('hizalama kapisi gecmedi: olcum durur (on kayit §9.1)')
    if res['PC060']['primary_exceeds_threshold']:
        reading = 'PC060 E1 net dSR >= 0,17: PC060 yeni ortamda pozitif kontrol olarak kalir (M3 S-obs kolu).'
    else:
        side = json.load(open(os.path.join(ws.results_dir(), 'predictions_pit', f"{proto['table']}.json"),
                              encoding='utf-8'))
        base, binfo = load_table(side['base_table']['run_id'])
        if binfo['sha256'] != side['base_table']['sha256']:
            raise SystemExit('PC060 taban tablo SHA uyusmuyor')
        base = base.copy()
        base['date'] = pd.to_datetime(base['date']).dt.tz_localize(None).dt.normalize()
        k12, kinfo = positive_control_table(base, k12_labels(arrays, px), side['target_auc'], seed=side['seed'])
        res['PC060_K12'] = evaluate('PC060-K12', k12, arrays, px, A, W)
        res['PC060_K12']['generator'] = {k: kinfo[k] for k in ('target_auc', 'seed', 'achieved_auc_meta_ok',
                                                               'n_labeled')}
        res['PC060_K12']['label'] = '1[Of_{t+2} > Of_{t+1}], t+1 ve t+2 gecerli; yalniz 2013-06..2020-12 (panel)'
        if res['PC060_K12']['primary_exceeds_threshold']:
            reading = ('PC060 E1 net dSR < 0,17 -> yeni ortamda gecerli pozitif kontrol DEGIL. PC060-K12 >= 0,17 -> '
                       'M3 pozitif kontrolu PC060-K12.')
        else:
            reading = ('PC060 ve PC060-K12 E1 net dSR < 0,17: AUC 0,60 bu yurutme ve maliyetle tespit edilebilir deger '
                       'uretmiyor; kontrol gucu (AUC) kullanici kararina birakilir.')
    prov = _provenance()
    rec = {'preregistration': 'phase-10-m2-design.md §9.1, commit 0232f11 (sonuctan once)', **prov,
           'panel': PANEL.replace(os.sep, '/'), 'panel_sha256': arrays['sha256'], 'table': tinfo['run_id'],
           'table_sha256': tinfo['sha256'], 'cost_one_way': COST, 'threshold': THRESHOLD, 'windows': WINDOWS,
           'primary_window': PRIMARY, 'results': res, 'reading': reading,
           'warning': 'sinyal bilerek gelecekten; ust sinir, strateji onerisi degil; tez sonucu degil'}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False, default=str)
    print(reading)
    if a.extra_seeds:
        side = json.load(open(os.path.join(ws.results_dir(), 'predictions_pit', f"{proto['table']}.json"),
                              encoding='utf-8'))
        base, _ = load_table(side['base_table']['run_id'])
        base = base.copy()
        base['date'] = pd.to_datetime(base['date']).dt.tz_localize(None).dt.normalize()
        labs = {'PC060_panel_labels': close_labels(arrays, px), 'PC060_K12': k12_labels(arrays, px)}
        rows = []
        for sd in [side['seed']] + _seeds(a.extra_seeds):
            row = {'seed': sd}
            for nm, lab in labs.items():
                tab, inf = positive_control_table(base, lab, side['target_auc'], seed=sd)
                r = evaluate(nm, tab, arrays, px, A, W, quiet=True)
                row[nm] = {'auc': inf['achieved_auc_meta_ok'], 'E1_net_pooled': r['primary_delta'],
                           'E1_net_train': r['train']['tests']['E1 net S_yon - 1/N']['delta_sharpe_annual'],
                           'E1_net_val': r['val']['tests']['E1 net S_yon - 1/N']['delta_sharpe_annual'],
                           'E1_net_se_pooled': r[PRIMARY]['tests']['E1 net S_yon - 1/N']['se_annual']}
            rows.append(row)
            print(f"  tohum {sd}: " + "  ".join(f"{k} {v['E1_net_pooled']:+.3f}" for k, v in row.items() if k != 'seed'))
        summ = {}
        for nm in labs:
            x = np.array([r[nm]['E1_net_pooled'] for r in rows])
            summ[nm] = {'min': float(x.min()), 'mean': float(x.mean()), 'max': float(x.max()),
                        'n_ge_threshold': int((x >= THRESHOLD).sum()), 'n': int(len(x))}
        srec = {'note': ('BETIMSEL, hakem-2 M2 kapanisi #3/#5 sonrasi (sonuctan sonra); birincil karar (§9.1) '
                         'degismez. PC060_panel_labels etiketi panelden kurulur (orijinal pc_auc060 etiket dosyasi '
                         'degil); tohum ' + str(side['seed']) + ' satiri bu yuzden pc_alignment.json PC060 ile ayni olmayabilir.'),
                **prov, 'panel_sha256': arrays['sha256'], 'base_table': side['base_table'],
                'target_auc': side['target_auc'], 'threshold': THRESHOLD, 'rows': rows, 'summary': summ}
        with open(a.seeds_out, 'w', encoding='utf-8') as fh:
            json.dump(srec, fh, indent=1, ensure_ascii=False, default=str)
        print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
