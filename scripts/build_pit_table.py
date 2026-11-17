"""
Faz 9 / P: 30 sembol PIT tahmin tablosu (K11 v1: 3 GBM + lojistik meta) + P2 kapsam + bilgi kapisi.

Tablo calisma alanina (`results_dir()/predictions_pit/<run_id>.parquet`, git disi), olcum ozeti
`docs/development/measurements/phase9-p/` altina yazilir.

    python scripts/build_pit_table.py                       # tum panel, varsayilan tohum
    python scripts/build_pit_table.py --seed 7 --out docs/development/measurements/phase9-p/pit_v1_seed7.json
    python scripts/build_pit_table.py --user-id <id>        # kullanici calisma alanina yaz
    python scripts/build_pit_table.py --target open_to_open --out docs/development/measurements/phase10-p5/pit_v2.json
                                                            # P5a: pit-v2, K12 yurutme ufku O_{t+1} -> O_{t+2}
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default=None, help='virgulle; bos = paneldeki tum semboller')
    ap.add_argument('--horizon', default='daily')
    ap.add_argument('--target', default='close_to_close', choices=['close_to_close', 'open_to_open'],
                    help='close_to_close = pit-v1 (dondurulmus); open_to_open = pit-v2 (P5a)')
    ap.add_argument('--seed', type=int, default=None, help='bos = model varsayilani (GLOBAL_SEED)')
    ap.add_argument('--user-id', default=None)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--out', default=None,
                    help='bos = phase9-p/pit_v1.json (close_to_close) ya da phase10-p5/pit_v2.json')
    a = ap.parse_args()
    if a.out is None:
        a.out = ('docs/development/measurements/phase9-p/pit_v1.json' if a.target == 'close_to_close'
                 else 'docs/development/measurements/phase10-p5/pit_v2.json')

    import logging
    logging.disable(logging.WARNING)
    from contextlib import nullcontext
    from app.auth import workspace as ws
    from prediction import pit_table as P
    from prediction.seeding import GLOBAL_SEED
    from evaluation.reproducibility import frame_sha256

    panel_path = os.path.join('data', 'bist', 'stock_data_with_indicators.csv')
    with open(panel_path, 'rb') as fh:
        panel_sha = hashlib.sha256(fh.read()).hexdigest()
    ctx = ws.use_workspace(a.user_id) if a.user_id else nullcontext()
    with ctx:
        t0 = time.time()
        panel, prep = P.load_panel()
        syms = a.symbols.split(',') if a.symbols else None
        table, rep = P.build_pit_table(panel, syms, horizon=a.horizon, seed=a.seed, progress=True,
                                       target=a.target)
        rep['coverage'] = P.coverage_report(table)
        rep['integrity'] = {k: prep.get(k) for k in ('window_start', 'window_end', 'rows_in_window', 'rows_after', 'n_artifacts', 'n_ghost_days', 'n_stale_cells_dropped', 'missing_cell_frac',
                                                     'artifacts_in_window') if k in prep}
        snap = {'name': 'pit_input_panel', 'sha256': frame_sha256(panel), 'file': panel_path,
                'file_sha256': panel_sha}
        paths = P.write_table(table, rep, run_id=a.run_id, snapshot=snap)
        labels = P.realized_labels(panel, a.horizon, a.target)
        gate = P.evaluate_gate(table, labels)
        with open(paths['table'], 'rb') as fh:
            tsha = hashlib.sha256(fh.read()).hexdigest()
    T = [e['T_pooled'] for e in rep['meta_log'] if e.get('T_pooled') is not None]
    out = {
        'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
        'panel': panel_path, 'panel_sha256': panel_sha, 'panel_frame_sha256': snap['sha256'],
        'table_sha256': tsha, 'table_rows': int(len(table)),
        'seed': a.seed if a.seed is not None else GLOBAL_SEED, 'horizon': a.horizon,
        'target': a.target, 'table_version': rep['table_version'],
        **({'open_band_mask': rep['open_band_mask']} if 'open_band_mask' in rep else {}),
        'design': rep['design'], 'models': rep['models'], 'n_anchors': len(rep['anchors']),
        'anchors_first_last': [rep['anchors'][0], rep['anchors'][-1]],
        'integrity': rep['integrity'],
        'coverage_rl_window': rep['coverage'],
        'temperature_pooled': {'per_anchor': {e['anchor']: e.get('T_pooled') for e in rep['meta_log']},
                               'n_at_bound': int(sum(bool(e.get('T_at_bound')) for e in rep['meta_log'])),
                               'min': min(T) if T else None, 'max': max(T) if T else None},
        'n_informative_asof_per_anchor': {e['anchor']: e.get('n_informative_asof') for e in rep['meta_log']},
        'gate_dev_window': gate,
        'seconds_per_symbol': rep['seconds_per_symbol'],
        'wall_seconds': round(time.time() - t0, 1),
        'note': ('K11 v1 (3 GBM + lojistik meta), yalniz fiyat/hacim ozellikleri; T capa basina '
                 'havuzlu; bilgi kapisi DEV_WINDOW (wf-v1 ilk dogrulama yili 2020 oncesi). '
                 'Tablo calisma alaninda (git disi): ' + os.path.relpath(paths['table'], ROOT).replace(os.sep, '/')),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('->', a.out, '| tablo', paths['table'], '| manifest', paths['manifest'])
    print('kapi:', gate['n_informative'], 'bilgili /', len(gate['per_symbol']), '| BH red', gate['n_bh_reject'],
          '| havuzlu AUC', gate['pooled']['auc'])


if __name__ == '__main__':
    main()
