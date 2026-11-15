"""
Faz 9 P4: P tablosu -> eski TradingEnv tahmin arayuzu; gercek tabloda (30 sembol, RL penceresi 2013-06..)
hucre doluluk/notr nedenleri ve env gozlemindeki dinamik aralik. Yeniden egitim yok.

    python scripts/analysis/measure_p4_env_features.py
"""

import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TABLE = os.path.join('results', 'predictions_pit', 'pit_v1_seed42.parquet')
OUT = os.path.join('docs', 'development', 'measurements', 'phase9-p', 'p4_env_features.json')
RL_START = '2013-06-01'   # K17


def main():
    import logging
    logging.disable(logging.WARNING)
    import pandas as pd
    from prediction import pit_table as P

    table = pd.read_parquet(TABLE)
    with open(TABLE, 'rb') as fh:
        tsha = hashlib.sha256(fh.read()).hexdigest()
    panel, _ = P.load_panel()
    syms = sorted(set(panel.index.get_level_values('symbol')))
    d = pd.DatetimeIndex(sorted(set(pd.to_datetime(panel.index.get_level_values('date')))))
    d = d[d >= pd.Timestamp(RL_START)]
    _, rep = P.env_prediction_features(table, syms, d)
    out = {'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
           'table': TABLE, 'table_sha256': tsha, 'window': [str(d[0].date()), str(d[-1].date())],
           'n_symbols': len(syms), 'report': rep}
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
