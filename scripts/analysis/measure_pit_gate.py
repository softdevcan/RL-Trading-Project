"""
Faz 9 / P, İ3 hakem-finans #1-#2: bilgi kapisi v2 — sembol basina MDE, iki tarafli p, blok ici AUC, esit
kapanis orani; havuzlu AUC icin tarih blogu bootstrap GA/p. Var olan PIT tablosundan (yeniden egitim yok).

    python scripts/analysis/measure_pit_gate.py                 # gate_v2.json (arsiv)
    python scripts/analysis/measure_pit_gate.py --sensitivity   # gate_v2_sensitivity.json (I3 ikinci okuma)
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
SENS = '--sensitivity' in sys.argv
OUT = os.path.join('docs', 'development', 'measurements', 'phase9-p',
                   'gate_v2_sensitivity.json' if SENS else 'gate_v2.json')


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
    g = P.evaluate_gate(table, labels, sensitivity=SENS)
    out = {'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
           'table': TABLE, 'table_sha256': tsha, 'gate_dev_window': g}
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=float)
    po = g['pooled']
    print('havuzlu AUC', po['auc'], '\nblok ici', po['auc_within_block'], '\ntie', po['tie_rate'])
    print('MDE medyan tek/aile', g['mde_auc_single_median'], g['mde_auc_family_median'])
    o = g['per_symbol']['ODAS.IS']
    print('ODAS', {k: o[k] for k in ('n', 'auc', 'auc_within_block', 'p_two_sided', 'tie_rate', 'base_rate')})
    if SENS:
        se = g['sensitivity']
        for k in ('pooled_block63', 'pooled_anchor_cluster'):
            print(k, {m: (v['value'], v['ci'], v['p_gt_half']) for m, v in se[k].items() if isinstance(v, dict)})
        print('MDE aile b20/b63', se['mde_family_block20'], se['mde_family_block63'])
        print('blok ici sembol dagilimi', se['auc_within_block_per_symbol'])
        te = se['ties_excluded']
        print('tie disi havuzlu', te['pooled_auc']['value'], 'blok ici', te['pooled_auc_within_block']['value'],
              'ODAS', te['per_symbol']['ODAS.IS'])
        print('ODAS Bonferroni', se['p_two_sided_bonferroni']['ODAS.IS'], 'p<0,05:', se['n_two_sided_p_lt_005'])
    print('->', OUT)


if __name__ == '__main__':
    main()
