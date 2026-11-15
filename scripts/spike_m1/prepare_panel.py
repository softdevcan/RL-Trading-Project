"""Faz 10 M1 spike'i: veri kapisindan gecen paneli numpy dizilerine yazar (ANA venv'de kosar).

Spike'in ortami `venv-marl/` icinde kosar ve uretim koduna (pandas, data/, app/) bagli olmamalidir (K9).
Bu betik ana venv'de paneli `data/integrity.py::prepare_rl_panel` ile hazirlar ve sabit bir .npz yazar;
spike ortami yalniz numpy ile okur.

Pencere 2013-06-01..2020-12-31: 2021 kati Milestone 0' m0p-v2'nin dogrulayici penceresi, spike ona dokunmaz.
Spike'ta uretilen hicbir performans sayisi rapora girmez (Faz 10 plani §4 S1). Ozellikler spike'a ozgu ve
olceksizdir; asil gozlem tanimi M2 `obs_spec`'inde yapilir (K26).

    python scripts/spike_m1/prepare_panel.py --data-root <ana depo> --out <dizin>/panel_spike.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

START, END = '2013-06-01', '2020-12-31'
FEATURES = ('r1', 'r5', 'r20', 'range', 'gap', 'rsi', 'cci', 'adx', 'macd_rel', 'vol20')


def build_features(o, h, l, c, rsi, cci, adx, macd):
    """[T, N] dizilerinden [T, N, F] nedensel ozellik (yalniz <= t). Ilk 20 satir isinma, NaN kalir."""
    lc = np.log(c)
    r1 = np.full_like(c, np.nan)
    r1[1:] = lc[1:] - lc[:-1]

    def lag_ret(k):
        x = np.full_like(c, np.nan)
        x[k:] = lc[k:] - lc[:-k]
        return x

    gap = np.full_like(c, np.nan)
    gap[1:] = np.log(o[1:] / c[:-1])
    vol = np.full_like(c, np.nan)
    for t in range(20, len(c)):
        vol[t] = np.std(r1[t - 19:t + 1], axis=0)
    f = np.stack([
        np.tanh(10 * r1), np.tanh(5 * lag_ret(5)), np.tanh(2.5 * lag_ret(20)),
        np.tanh(10 * (h - l) / c), np.tanh(10 * gap),
        rsi / 100.0 - 0.5, np.tanh(cci / 200.0), adx / 100.0,
        np.tanh(100 * macd / c),  # MACD fiyat cinsinden (TL); fiyata bolunerek olceksiz
        np.tanh(30 * vol),
    ], axis=-1).astype(np.float32)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='data/bist iceren depo koku (salt okunur)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.chdir(a.data_root)
    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel

    df = DataFetcher().load_data('stock_data_with_indicators.csv')
    panel, report = prepare_rl_panel(df, start=START, end=END)
    cols = ('open', 'high', 'low', 'close', 'rsi', 'cci', 'adx', 'macd')
    wide = {k: panel[k].unstack('symbol').sort_index() for k in cols}
    symbols = list(wide['close'].columns)
    ok = np.ones(len(wide['close']), dtype=bool)
    for k in cols:
        ok &= wide[k].notna().all(axis=1).to_numpy()
    dates = wide['close'].index[ok]
    arr = {k: wide[k].loc[dates].to_numpy(dtype=np.float64) for k in cols}
    feats = build_features(arr['open'], arr['high'], arr['low'], arr['close'], arr['rsi'], arr['cci'],
                           arr['adx'], arr['macd'])
    warm = 20
    out = dict(features=feats[warm:], open=arr['open'][warm:].astype(np.float32),
               close=arr['close'][warm:].astype(np.float32),
               dates=np.array([str(d)[:10] for d in dates[warm:]]), symbols=np.array(symbols),
               feature_names=np.array(FEATURES))
    assert np.isfinite(out['features']).all(), 'ozelliklerde NaN/inf'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    sha = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
    print(json.dumps({'out': a.out, 'sha256': sha, 'T': int(out['features'].shape[0]), 'N': len(symbols),
                      'F': len(FEATURES), 'first': out['dates'][0], 'last': out['dates'][-1],
                      'dropped_incomplete_days': int((~ok).sum()), 'panel_report_keys': sorted(report)[:8]},
                     indent=1, default=str))


if __name__ == '__main__':
    main()
