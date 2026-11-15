"""
Faz 9 F (F1-F3) olcumu: gercek panelde eski guvenin kalibrasyonu, Y4 catisma orani ve
kalibre yolun (lojistik + sicaklik) test ECE/Brier'i.

Her sembol icin tam ensemble (5 model, TATS kapali, makro/temel/capraz yok — yalniz fiyat
ozellikleri) 'calibrated' modda egitilir; kalibrasyon blogu iki yolu da tasir. Modeller
gecici bir dizine yazilir (calisma alanina dokunulmaz).

    python scripts/analysis/measure_calibration.py --symbols THYAO.IS,GARAN.IS,ASELS.IS \
        --out docs/development/measurements/phase9-f/calibration.json
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='THYAO.IS,GARAN.IS,ASELS.IS')
    ap.add_argument('--horizon', default='daily')
    ap.add_argument('--start', default='2013-06-01')
    ap.add_argument('--out', default='docs/development/measurements/phase9-f/calibration.json')
    a = ap.parse_args()

    from app.auth import workspace as ws
    import prediction.models.ensemble as E

    panel_path = os.path.join('data', 'bist', 'stock_data_with_indicators.csv')
    with open(panel_path, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    panel = pd.read_csv(panel_path, parse_dates=['date'])
    tmp = tempfile.mkdtemp(prefix='meas_f_')
    ws.prediction_models_dir = lambda *x, **k: tmp
    ws.read_dirs = lambda kind, *x, **k: [tmp]
    out = {'panel': panel_path, 'panel_sha256': sha, 'horizon': a.horizon, 'start': a.start,
           'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
           'config': 'tam ensemble (5 model), use_tats=False, yalniz fiyat ozellikleri, confidence_mode=calibrated',
           'symbols': {}}
    try:
        for sym in a.symbols.split(','):
            d = panel[(panel.symbol == sym) & (panel.date >= a.start)].set_index('date').sort_index()
            d = d[['open', 'high', 'low', 'close', 'volume']].dropna()
            t0 = time.time()
            ens = E.StackingEnsemble(horizon=a.horizon, use_tats=False, confidence_mode='calibrated')
            tm = ens.train(d, sym)['ensemble_test_metrics']
            c = tm.get('calibration', {})
            leg, cc = c.get('legacy', {}), c.get('calibrated') or {}
            row = {'n_rows': int(len(d)), 'seconds': round(time.time() - t0, 1),
                   'features': c.get('features'), 'n_fit': c.get('n_fit'), 'n_calib': c.get('n_calib'),
                   'legacy': {k: leg.get(k) for k in ('n', 'ece', 'brier', 'accuracy', 'base_rate',
                                                       'head_vs_price_conflict_rate', 'has_direction_heads')},
                   'calibrated': None if not cc else {
                       'temperature': cc['temperature'],
                       'temperature_at_bound': cc.get('temperature_at_bound'),
                       'raw': {k: cc['raw'][k] for k in ('n', 'ece', 'brier', 'accuracy')},
                       'temperature_scaled': {k: cc['temperature_scaled'][k] for k in ('n', 'ece', 'brier', 'accuracy')},
                       'isotonic_compare': {k: cc['isotonic_compare'][k] for k in ('n', 'ece', 'brier', 'accuracy')},
                       'price_vs_p_conflict_rate': cc['price_vs_p_conflict_rate']},
                   'skipped': c.get('skipped'), 'error': c.get('error')}
            out['symbols'][sym] = row
            print(sym, json.dumps(row, default=float)[:400], flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('->', a.out)


if __name__ == '__main__':
    main()
