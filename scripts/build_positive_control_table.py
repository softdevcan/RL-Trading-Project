"""
Milestone 0' DTF pozitif kontrol tablolari (BILEREK gelecekten; yalniz kontrol kolu).

    python scripts/build_positive_control_table.py --auc 0.55 0.60

Girdi: results_dir()/predictions_pit/pit_v1_seed42.parquet (maske ve olcek kaynagi).
Cikti: results_dir()/predictions_pit/pc_auc<AUC>.parquet + yan JSON (positive_control=True, SHA).
"""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auc', type=float, nargs='+', default=[0.55, 0.60])
    ap.add_argument('--base', default='pit_v1_seed42')
    ap.add_argument('--seed', type=int, default=20261115)
    a = ap.parse_args()
    import logging
    logging.disable(logging.WARNING)
    from app.auth import workspace as ws
    from prediction import pit_table as P
    base, binfo = P.load_table(a.base)
    panel, _ = P.load_panel()
    labels = P.realized_labels(panel)
    d = os.path.join(ws.results_dir(), 'predictions_pit')
    for auc in a.auc:
        run_id = f'pc_auc{int(round(auc * 100)):03d}'
        tab, info = P.positive_control_table(base, labels, auc, seed=a.seed)
        path = os.path.join(d, f'{run_id}.parquet')
        tab.to_parquet(path, index=False)
        with open(path, 'rb') as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        side = {**info, 'run_id': run_id, 'table_sha256': sha, 'base_table': binfo,
                'table_version': f'{P.TABLE_VERSION}+positive-control'}
        with open(os.path.join(d, f'{run_id}.json'), 'w', encoding='utf-8', newline='\n') as fh:
            json.dump(side, fh, indent=1, default=str)
        print(run_id, 'hedef', auc, 'olculen', round(info['achieved_auc_meta_ok'], 4), 'sha', sha[:12])


if __name__ == '__main__':
    main()
