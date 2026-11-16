"""Faz 10 M2: veri kapisindan gecen paneli M2 ortaminin .npz dosyasina yazar (ANA venv'de kosar).

Tasarim §6. Hat: stock_data_with_indicators.csv -> with_adx('wilder') -> prepare_rl_panel
(drop_indicator_warmup=True; rota ile ayni kapi) -> marl.panel.build_arrays -> A8 rf -> .npz (SHA-256).
Eksik gunler ATILMAZ (M1 spike'i 69 gunu atiyordu); hucre maskesi ortamda.

    python scripts/marl/prepare_panel_m2.py --end 2020-12-31 --out <dizin>/panel_m2.npz
    python scripts/marl/prepare_panel_m2.py --end 2020-12-31 --health-only --train-end 2018-12-31

--health-only: rf olmadan da kosar; .npz YAZMAZ, gozlem sagligini (ozellik basina kirpma orani,
sabitlik; K18) egitim satirlarinda olcer. rf yoksa .npz yazimi acik hatayla durur (sabite dusulmez).
Pencere 2021'e dokunmaz (m0p-v2 dogrulayici penceresi; --end varsayilani 2020-12-31).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=None, help='varsayilan RL_DATA_START')
    ap.add_argument('--end', default='2020-12-31')
    ap.add_argument('--out', default=None)
    ap.add_argument('--health-only', action='store_true')
    ap.add_argument('--train-end', default='2018-12-31', help='--health-only: olcum satirlarinin sonu')
    ap.add_argument('--json', default=None, help='olcum JSON cikisi')
    a = ap.parse_args()

    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel
    from data.technical_indicators import with_adx
    from marl import obs_spec as OS
    from marl import panel as PN

    src = os.path.join(DataFetcher().data_dir, 'stock_data_with_indicators.csv')
    df = with_adx(DataFetcher().load_data('stock_data_with_indicators.csv'), 'wilder')
    panel, gate = prepare_rl_panel(df, start=a.start, end=a.end, drop_indicator_warmup=True)
    arrays = PN.build_arrays(panel)
    dates = arrays['dates']
    valid = np.isfinite(arrays['open']) & np.isfinite(arrays['close'])
    incomplete = int((~valid.all(axis=1)).sum())
    feat, ok = OS.compute_raw_features(arrays)
    rows = (dates <= a.train_end) & (np.arange(len(dates)) >= OS.WARMUP_ROWS)
    stats = OS.fit_norm_stats(feat, ok, rows)
    health = OS.clip_report(feat, ok, stats, rows)
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    # Hakem-finans M2 #1, #4, Q3 (2026-11-16): kilit, fiyat limiti, acilis gecerli / kapanis eksik
    from marl.accounting import LISTING_LOCKOUT, fill_prices
    px = fill_prices(arrays['open'], arrays['close'], LISTING_LOCKOUT)
    with np.errstate(invalid='ignore', divide='ignore'):
        g_on = px['open'][1:] / px['close'][:-1]
    v1 = px['valid'][1:] & px['listed'][:-1]
    limits = {}
    for thr in (0.095, 0.195):   # BIST fiyat marji donemine gore +-%10 / +-%20; ikisi de raporlanir
        up = v1 & (g_on >= 1 + thr)
        dn = v1 & (g_on <= 1 - thr)
        limits[f'{thr:.3f}'] = {'open_gap_up_cells': int(up.sum()), 'open_gap_down_cells': int(dn.sum()),
                                'frac_of_valid': float((up | dn).sum() / max(v1.sum(), 1)),
                                'up_in_health_rows': int(up[rows[1:]].sum()),
                                'down_in_health_rows': int(dn[rows[1:]].sum())}
    o_ok = np.isfinite(arrays['open']) & (arrays['open'] > 0)
    c_ok = np.isfinite(arrays['close']) & (arrays['close'] > 0)
    market = {
        'listing_lockout_sessions': LISTING_LOCKOUT,
        'listing_lockout_cells': int(px['lockout'].sum()),
        'listing_lockout_symbols': [str(arrays['symbols'][i]) for i in np.flatnonzero(px['lockout'].any(0))],
        'open_gap_limit_cells': limits,
        'open_valid_close_missing_cells': int((o_ok & ~c_ok).sum()),
        'price_basis': 'yfinance history() auto_adjust varsayilani (bolunme + temettu duzeltilmis)',
    }
    rep = {
        'source_csv': src, 'source_sha256': _sha(src), 'commit': commit, 'market': market,
        'window': [str(dates[0]), str(dates[-1])], 'T': int(len(dates)), 'N': int(len(arrays['symbols'])),
        'symbols': [str(s) for s in arrays['symbols']],
        'incomplete_days': incomplete, 'missing_cell_frac': float(1 - valid.mean()),
        'obs_spec': OS.OBS_SPEC_VERSION, 'health_rows': [str(dates[rows][0]), str(dates[rows][-1])],
        'health': health, 'norm_stats': stats,
        'gate': {k: gate[k] for k in list(gate)[:12]},
    }
    if a.health_only:
        out = json.dumps(rep, indent=1, default=str, ensure_ascii=False)
        if a.json:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            open(a.json, 'w', encoding='utf-8').write(out)
        print(json.dumps({k: rep[k] for k in ('window', 'T', 'N', 'incomplete_days', 'missing_cell_frac',
                                              'health_rows', 'source_sha256', 'market')}, indent=1, default=str))
        for n, h in health.items():
            print(f"  {n:10s} kirpma {h['clip_frac']:.4%}  sabit {h['constant']}  n {h['n']}")
        return
    if not a.out:
        ap.error('--out gerekli (ya da --health-only)')
    arrays['rf'] = PN.risk_free_for(dates)   # RiskFreeUnavailable: sabite dusulmez
    info = PN.save_npz(arrays, a.out, meta={k: rep[k] for k in rep if k not in ('health', 'norm_stats')})
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
