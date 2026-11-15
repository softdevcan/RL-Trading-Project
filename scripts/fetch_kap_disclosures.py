"""
Faz 9 E1: KAP VYK API'den bir sirketin (varsayilan 49 = BORSA ISTANBUL A.S.) bildirim listesini
sayfalayarak ceker (sayfa basina <= 50, dakikada 6 cagri). Ham yanitlar data/universe/sources/kap_api/
(gitignore); ozet: data/universe/sources/kap_api/list_<company>.json.

    python scripts/fetch_kap_disclosures.py [--company 49] [--start 1] [--max-pages 200]

Test gecidinde (apigwdev) veri 2023-01-02'den basliyor (ilk Borsa bildirimi #1091709; 2026-11-15 olculdu).
"""

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import kap_api as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--company', default='49')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--max-pages', type=int, default=200)
    a = ap.parse_args()
    rows, idx = [], a.start
    for page in range(a.max_pages):
        try:
            r = K.disclosures(idx, company_ids=[a.company])
        except RuntimeError as e:
            if 'ER005' in str(e):   # bu indeksten sonra bildirim yok
                break
            raise
        if not r:
            break
        rows.extend(r)
        nxt = max(int(x['disclosureIndex']) for x in r) + 1
        print(f'sayfa {page}: {len(r)} bildirim, son #{nxt - 1}', flush=True)
        if nxt <= idx:
            break
        idx = nxt
    seen = {x['disclosureIndex']: x for x in rows}
    rows = sorted(seen.values(), key=lambda x: int(x['disclosureIndex']))
    out = os.path.join(K.RAW_DIR, f'list_{a.company}.json')
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'company': a.company, 'base': K.base_url(), 'n': len(rows), 'rows': rows}, fh,
                  ensure_ascii=False, indent=1)
    c = collections.Counter(s for x in rows for s in (x.get('subReportIds') or ['<yok>']))
    print(len(rows), 'bildirim ->', out)
    for k, v in c.most_common(40):
        print(f'{v:6d}  {k}')


if __name__ == '__main__':
    main()
