"""
Faz 9 / P, İ3 hakem-rl #3: meta girdisinin dagilim kaymasi.

Meta lojistigi bir capada GECMIS 504 OOF satirinin taban tahminleriyle uyduruluyor (eski blok modelleri);
yeni blokta daha cok veriyle egitilmis taban modelin tahminine uygulaniyor. Sembol x capa basina:
  oran = std(blok icindeki taban tahmini) / std(meta penceresindeki taban tahmini), model basina;
  kayma = (blok ortalamasi - pencere ortalamasi) / pencere std'si.
Tablodan (parquet) hesaplanir; model yeniden egitilmez.

    python scripts/analysis/measure_pit_shift.py --table results/predictions_pit/pit_v1_seed42.parquet
"""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=os.path.join('results', 'predictions_pit', 'pit_v1_seed42.parquet'))
    ap.add_argument('--out', default=os.path.join('docs', 'development', 'measurements', 'phase9-p',
                                                  'meta_input_shift.json'))
    a = ap.parse_args()
    from prediction.pit_table import META_WINDOW, V1_MODELS
    t = pd.read_parquet(a.table)
    rc = [f'ret_{m}' for m in V1_MODELS]
    t['anchor'] = pd.to_datetime(t['model_version'].str.split(':').str[1])
    ratios = {m: [] for m in rc}
    shifts = {m: [] for m in rc}
    for s, g in t[t.base_ok].groupby('symbol'):
        g = g.sort_values('date')
        for a_k, blk in g[g.meta_ok].groupby('anchor'):
            # meta penceresi: etiketi capadan once gerceklesmis son 504 OOF satiri (yaklasik: label_date
            # tabloda yok; tarih < capa - 1 seans, h=1 icin ayni kume)
            hist = g[g['date'] < a_k].iloc[:-1].iloc[-META_WINDOW:]
            if len(hist) < 200:
                continue
            for m in rc:
                sd_h = hist[m].std()
                if sd_h > 0:
                    ratios[m].append(blk[m].std() / sd_h)
                    shifts[m].append((blk[m].mean() - hist[m].mean()) / sd_h)

    def q(v):
        v = np.asarray(v, dtype=float)
        return {'n': int(len(v)), 'median': float(np.median(v)),
                'p10': float(np.quantile(v, 0.1)), 'p90': float(np.quantile(v, 0.9)),
                'p01': float(np.quantile(v, 0.01)), 'p99': float(np.quantile(v, 0.99))}

    with open(a.table, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    out = {'commit': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
           'table': a.table, 'table_sha256': sha,
           'note': ('sembol x capa basina: std orani = std(blok) / std(meta penceresi); kayma = (ort. blok - '
                    'ort. pencere) / std pencere. Pencere tarih < capa ile yaklasik (h=1, son satir atilir).'),
           'std_ratio': {m: q(v) for m, v in ratios.items()},
           'mean_shift_in_sd': {m: q(v) for m, v in shifts.items()}}
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    for m in rc:
        print(m, 'std orani', {k: round(v, 3) for k, v in out['std_ratio'][m].items() if k != 'n'},
              '| kayma', {k: round(v, 3) for k, v in out['mean_shift_in_sd'][m].items() if k != 'n'})
    print('->', a.out)


if __name__ == '__main__':
    main()
