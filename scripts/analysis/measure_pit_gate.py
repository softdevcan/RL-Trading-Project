"""
Faz 9 / P, İ3 hakem-finans #1-#2: bilgi kapisi v2 — sembol basina MDE, iki tarafli p, blok ici AUC, esit
kapanis orani; havuzlu AUC icin tarih blogu bootstrap GA/p. Var olan PIT tablosundan (yeniden egitim yok).

    python scripts/analysis/measure_pit_gate.py
"""

import hashlib
import json
import os
import subprocess
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings('ignore')

TABLE = os.path.join('results', 'predictions_pit', 'pit_v1_seed42.parquet')
OUT = os.path.join('docs', 'development', 'measurements', 'phase9-p', 'gate_v2.json')


def main():
    import logging
    logging.disable(logging.WARNING)
    import pandas as pd
    from prediction import pit_table as P

    table = pd.read_parquet(TABLE)
    with open(TABLE, 'rb') as fh:
        tsha = hashlib.sha256(fh.read()).hexdigest()
    panel, _ = P.load_panel()
    labels = P.realized_labels(panel)
    g = P.evaluate_gate(table, labels)
    out = {'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
           'table': TABLE, 'table_sha256': tsha, 'gate_dev_window': g}
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    po = g['pooled']
    print('havuzlu AUC', po['auc'], '\nblok ici', po['auc_within_block'], '\ntie', po['tie_rate'])
    print('MDE medyan tek/aile', g['mde_auc_single_median'], g['mde_auc_family_median'])
    o = g['per_symbol']['ODAS.IS']
    print('ODAS', {k: o[k] for k in ('n', 'auc', 'auc_within_block', 'p_two_sided', 'tie_rate', 'base_rate')})
    print('->', OUT)


if __name__ == '__main__':
    main()
