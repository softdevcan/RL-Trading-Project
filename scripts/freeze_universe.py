"""Faz 9 E0 (Y11, K7) — RL panelinin bugunku sembol evrenini dondur.

    python scripts/freeze_universe.py            # deneme: yazmadan ozet
    python scripts/freeze_universe.py --write    # data/universe/bist30_panel.json

Evren = `data/bist/stock_data_with_indicators.csv` icindeki semboller, panelde GORUNME SIRASIYLA
(env durum vektoru bu sirayi kullanir; sorted() UYGULANMAZ). Bu GERCEK (tarihsel) BIST-30 DEGILDIR:
2026-11'de panelde bulunan 30 hisse. Tarihsel uyelik E1'de (`bist30_membership.csv`).
Dosya varsa ve sembol kumesi/sirasi degisiyorsa yazmaz (evren surumu elle artirilir).
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, 'data', 'bist', 'stock_data_with_indicators.csv')
RAW = os.path.join(ROOT, 'data', 'bist', 'raw_stock_data.csv')
OUT = os.path.join(ROOT, 'data', 'universe', 'bist30_panel.json')


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    d = pd.read_csv(PANEL, usecols=['symbol', 'date'])
    symbols = list(dict.fromkeys(d['symbol']))
    doc = {
        'name': 'bist30_panel_v1',
        'definition': ('RL egitim panelinde 2026-11-14 itibariyla bulunan hisseler, panelde gorunme '
                       'sirasiyla. Tarihsel BIST-30 uyeligi DEGIL (hayatta kalma yanliligi; E1).'),
        'decision': 'K7 (Faz 10 plani §6; kullanici onayi 2026-11-14)',
        'frozen_at': str(date.today()),
        'n_symbols': len(symbols),
        'symbols': symbols,
        'panel': {
            'file': 'data/bist/stock_data_with_indicators.csv', 'sha256': _sha(PANEL),
            'raw_file': 'data/bist/raw_stock_data.csv', 'raw_sha256': _sha(RAW) if os.path.exists(RAW) else None,
            'first_date': str(d['date'].min()), 'last_date': str(d['date'].max()),
        },
    }
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            old = json.load(f)
        if old['symbols'] != symbols:
            sys.exit(f"Evren degisti ({old['name']}): {sorted(set(old['symbols']) ^ set(symbols))} — "
                     f"yeni surum adi ver, eskisinin uzerine yazma")
    print(json.dumps({k: doc[k] for k in ('name', 'n_symbols', 'frozen_at')}, ensure_ascii=False),
          doc['panel']['last_date'])
    if a.write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print('yazildi:', os.path.relpath(OUT, ROOT))


if __name__ == '__main__':
    main()
