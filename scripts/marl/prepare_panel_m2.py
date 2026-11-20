"""Faz 10 M2: veri kapisindan gecen paneli M2 ortaminin .npz dosyasina yazar (ANA venv'de kosar).

Tasarim §6. Hat: stock_data_with_indicators.csv -> with_adx('wilder') -> prepare_rl_panel
(drop_indicator_warmup=True; rota ile ayni kapi) -> marl.panel.build_arrays -> A8 rf -> .npz (SHA-256).
Eksik gunler ATILMAZ (M1 spike'i 69 gunu atiyordu); hucre maskesi ortamda.

    python scripts/marl/prepare_panel_m2.py --end 2020-12-31 --out <dizin>/panel_m2.npz
    python scripts/marl/prepare_panel_m2.py --end 2020-12-31 --health-only --train-end 2018-12-31

--health-only: rf olmadan da kosar; .npz YAZMAZ, gozlem sagligini (ozellik basina kirpma orani,
sabitlik; K18) egitim satirlarinda ve egitim istatistigiyle dogrulama satirlarinda (train_end, end] olcer. rf yoksa .npz yazimi acik hatayla durur (sabite dusulmez).
Pencere 2021'e dokunmaz (m0p-v2 dogrulayici penceresi; --end varsayilani 2020-12-31).

M8 paneli (tasarim §6.1, kullanici karari 2026-11-20): `--data-fix m8-v1` veri duzeltmesini (`marl/panel_fix.py`)
uygular ve MACD/RSI/CCI'yi duzeltilmis fiyattan yeniden hesaplar; once yeniden hesabin CSV'yi urettigi sinanir.
Varsayilan `none`: hat M2 paneliyle ayni (bit-es).

    python scripts/marl/prepare_panel_m2.py --data-fix m8-v1 --end 2026-11-09 --out data/marl/panel_m8.npz \
        --json docs/development/measurements/phase10-m8/panel_m8.json
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
    ap.add_argument('--data-fix', choices=('none', 'm8-v1'), default='none',
                    help="M8 paneli veri duzeltmesi (tasarim §6.1); 'none' -> M2 paneliyle ayni hat")
    a = ap.parse_args()

    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel
    from data.technical_indicators import with_adx
    from marl import obs_spec as OS
    from marl import panel as PN

    src = os.path.join(DataFetcher().data_dir, 'stock_data_with_indicators.csv')
    df = DataFetcher().load_data('stock_data_with_indicators.csv')
    fix = None
    if a.data_fix == 'm8-v1':   # yalniz M8 paneli (tasarim §6.1); paylasilan kapi ve M2 paneli degismez
        import pandas as pd
        from marl import panel_fix as PF
        cols = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'cci']
        same = float(np.nanmax(np.abs(PF.recompute_indicators(df)[cols].to_numpy() - df[cols].to_numpy())))
        if not same < 1e-6:
            raise RuntimeError(f"gosterge yeniden hesabi CSV'yi uretmiyor (en buyuk fark {same}); m8-v1 durdu")
        syms = list(pd.unique(df.index.get_level_values('symbol')))
        df, fix = PF.apply_m8_v1(df, {s: PF.load_isy(s) for s in syms})
        df = PF.recompute_indicators(df)
        fix['indicator_recompute_check_max_abs_diff'] = same
        fix['isy_sha256'] = PF.isy_sha256(syms)
    df = with_adx(df, 'wilder')
    panel, gate = prepare_rl_panel(df, start=a.start, end=a.end, drop_indicator_warmup=True)
    arrays = PN.build_arrays(panel)
    dates = arrays['dates']
    valid = np.isfinite(arrays['open']) & np.isfinite(arrays['close'])
    incomplete = int((~valid.all(axis=1)).sum())
    feat, ok = OS.compute_raw_features(arrays)
    rows = (dates <= a.train_end) & (np.arange(len(dates)) >= OS.WARMUP_ROWS)
    stats = OS.fit_norm_stats(feat, ok, rows)
    health = OS.clip_report(feat, ok, stats, rows)
    val_rows = (dates > a.train_end) & (np.arange(len(dates)) >= OS.WARMUP_ROWS)
    health_val = OS.clip_report(feat, ok, stats, val_rows) if val_rows.any() else {}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    # Hakem-finans M2 #1, #4, Q3 (2026-11-16): kilit, fiyat limiti, acilis gecerli / kapanis eksik
    from marl.accounting import LISTING_LOCKOUT, fill_prices, limit_down_gaps, margin_schedule_meta
    px = fill_prices(arrays['open'], arrays['close'], LISTING_LOCKOUT, limit_down_gaps(dates))   # v0.7
    with np.errstate(invalid='ignore', divide='ignore'):
        g_on = px['open'][1:] / px['close'][:-1]
    v1 = px['valid'][1:] & px['listed'][:-1]
    limits = {}
    for thr in (0.095, 0.195):   # BIST fiyat marji donemine gore +-%10 / +-%20; ikisi de raporlanir
        up = v1 & (g_on >= 1 + thr)
        dn = v1 & (g_on <= 1 - thr)
        limits[f'{thr:.3f}'] = {'open_gap_up_cells': int(up.sum()), 'open_gap_down_cells': int(dn.sum()),
                                'frac_of_valid': float((up | dn).sum() / max(v1.sum(), 1)),
                                'frac_up': float(up.sum() / max(v1.sum(), 1)),
                                'frac_down': float(dn.sum() / max(v1.sum(), 1)),
                                'up_in_health_rows': int(up[rows[1:]].sum()),
                                'down_in_health_rows': int(dn[rows[1:]].sum())}
    # hakem-finans M2 ikinci okuma #9: kilit sonrasi ilk 10 seansta tavan/taban acilis boslugu
    post = {}
    for i in np.flatnonzero(px['lockout'].any(0)):
        end = int(np.flatnonzero(px['lockout'][:, i])[-1]) + 1          # kilitten sonraki ilk satir
        seg = g_on[end - 1:end - 1 + 10, i]                               # g_on[t] = t -> t+1
        post[str(arrays['symbols'][i])] = {'first_tradable': str(dates[end]),
                                           'gap_up_0.095': int(np.sum(seg >= 1.095)),
                                           'gap_down_0.095': int(np.sum(seg <= 0.905))}
    o_ok = np.isfinite(arrays['open']) & (arrays['open'] > 0)
    c_ok = np.isfinite(arrays['close']) & (arrays['close'] > 0)
    market = {
        'listing_lockout_sessions': LISTING_LOCKOUT,
        'listing_lockout_cells': int(px['lockout'].sum()),
        'listing_lockout_symbols': [str(arrays['symbols'][i]) for i in np.flatnonzero(px['lockout'].any(0))],
        'open_gap_limit_cells': limits,
        'post_lockout_first10_gaps': post,
        'open_valid_close_missing_cells': int((o_ok & ~c_ok).sum()),
        # §9 #9 kurali uygulandiktan sonra ortamin gordugu maske (tradable & no_sell; ortam penceresi >= WARMUP_ROWS)
        'limit_down_no_sell': margin_schedule_meta(),
        'no_sell_cells': int((px['no_sell'] & px['tradable'])[OS.WARMUP_ROWS:].sum()),
        'no_sell_cells_health_rows': int((px['no_sell'] & px['tradable'])[rows].sum()),
        'no_sell_cells_val_rows': int((px['no_sell'] & px['tradable'])[val_rows].sum()),
        'price_basis': 'yfinance history() auto_adjust varsayilani (bolunme + temettu duzeltilmis)',
    }
    rep = {
        'source_csv': os.path.relpath(src, ROOT).replace(os.sep, '/'),   # yerel yol JSON'a girmez
        'source_sha256': _sha(src), 'commit': commit, 'market': market,
        'window': [str(dates[0]), str(dates[-1])], 'T': int(len(dates)), 'N': int(len(arrays['symbols'])),
        'symbols': [str(s) for s in arrays['symbols']],
        'incomplete_days': incomplete, 'missing_cell_frac': float(1 - valid.mean()),
        'obs_spec': OS.OBS_SPEC_VERSION, 'health_rows': [str(dates[rows][0]), str(dates[rows][-1])],
        'health': health, 'val_rows': [str(dates[val_rows][0]), str(dates[val_rows][-1])] if val_rows.any() else None,
        'health_val': health_val, 'norm_stats': stats,
        'gate': {k: gate[k] for k in list(gate)[:12]},
    }
    if fix is not None:
        rep['data_fix'] = fix
        rep['data_fix_summary'] = {**{k: fix[k] for k in ('version', 'close_tol', 'median_window', 'drop_sessions',
                                                           'rights_fixes', 'indicator_recompute_check_max_abs_diff',
                                                           'isy_sha256')},
                                   'close_patch_n': fix['close_patch']['n'],
                                   'residual_max_abs_dev_after': fix['verify']['residual_max_abs_dev_after']}
    def _write_json():
        if a.json:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            open(a.json, 'w', encoding='utf-8').write(json.dumps(rep, indent=1, default=str, ensure_ascii=False))

    if a.health_only:
        _write_json()
        print(json.dumps({k: rep[k] for k in ('window', 'T', 'N', 'incomplete_days', 'missing_cell_frac',
                                              'health_rows', 'source_sha256', 'market')}, indent=1, default=str))
        for n, h in health.items():
            hv = health_val.get(n, {})
            print(f"  {n:10s} egitim kirpma {h['clip_frac']:.4%} sabit {h['constant']} n {h['n']}"
                  f" | dogrulama kirpma {hv.get('clip_frac', float('nan')):.4%} sabit {hv.get('constant')} n {hv.get('n')}")
        return
    if not a.out:
        ap.error('--out gerekli (ya da --health-only)')
    arrays['rf'] = PN.risk_free_for(dates)   # RiskFreeUnavailable: sabite dusulmez
    rf = arrays['rf']
    rep['risk_free'] = {'finite_rows': int(np.isfinite(rf).sum()), 'rows': int(len(rf)),
                        'first_nan_rows': int(np.argmax(np.isfinite(rf))) if np.isfinite(rf).any() else None,
                        'per_period_mean': float(np.nanmean(rf)), 'per_period_max': float(np.nanmax(rf))}
    info = PN.save_npz(arrays, a.out, meta={k: rep[k] for k in rep
                                            if k not in ('health', 'health_val', 'norm_stats', 'data_fix')})
    info['path'] = os.path.relpath(os.path.abspath(a.out), ROOT).replace(os.sep, '/')   # yerel yol JSON'a girmez
    rep['npz'] = info
    _write_json()   # .npz SHA + rf kapsami olcum JSON'una (health-only ile ayni alanlar + npz, risk_free)
    print(json.dumps({**info, 'risk_free': rep['risk_free']}, indent=1))

if __name__ == '__main__':
    main()
