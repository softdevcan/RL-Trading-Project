"""Faz 9 E2 (Y5) — evrenin 30 hissesi icin KAP sektor siniflamasi ve ajan/maruziyet gruplari.

    python scripts/build_sectors.py --fetch            # KAP'tan indir, yerel kopyayi sakla, CSV yaz
    python scripts/build_sectors.py --html <dosya>     # saklanan kopyadan yeniden uret (ag yok)

Kaynak: KAP "Sektorler" sayfasi (https://kap.org.tr/tr/Sektorler); sayfanin gomulu Next.js verisindeki
{sectorName, sectorNo, stockCode, title} kayitlari. Ham sayfa ucuncu taraf icerigi oldugu icin depoya
GIRMEZ: `data/universe/sources/` (gitignore) altina saklanir; CSV her satirda URL, erisim tarihi ve ham
sayfanin SHA-256'sini tasir (ayni kopyadan yeniden uretim dogrulanabilir).

Grup kurali (mekanik, yargi yok; K6=A altinda yalniz `gat` kolu ve sektor maruziyet siniri kullanir):
  grup = KAP alt sektoru; evrende 2'den az uyesi olan alt sektorlerin hisseleri 'DIGER' grubunda.
Kabul (plan §4 E2): her sembol tam bir grupta, grup basina >= 2 sembol, ~6-8 grup.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
URL = 'https://kap.org.tr/tr/Sektorler'
OUT = os.path.join(ROOT, 'data', 'universe', 'sectors.csv')
SRC_DIR = os.path.join(ROOT, 'data', 'universe', 'sources')
MIN_GROUP = 2
REC = re.compile(r'\{"sectorName":"([^"]*)","sectorOid":"[^"]*","sectorNo":"([^"]*)",'
                 r'"mkkMemberOid":"[^"]*","stockCode":"([^"]*)","title":"([^"]*)"')


def parse(html: str) -> dict:
    """stockCode -> [(sector_name, sector_no, title)] (bir kayit birden cok kod tasiyabilir)."""
    t = html.replace('\\"', '"')
    by = {}
    for name, no, codes, title in REC.findall(t):
        for c in (x.strip() for x in codes.split(',')):
            if c:
                by.setdefault(c, []).append((name, no, title))
    return by


def build(html: bytes, retrieved_at: str, symbols: list) -> list:
    sha = hashlib.sha256(html).hexdigest()
    by = parse(html.decode('utf-8'))
    rows, missing, multi = [], [], []
    for s in symbols:
        hits = {(n, no, t) for n, no, t in by.get(s[:-3], [])}
        if not hits:
            missing.append(s)
            continue
        if len({n for n, _, _ in hits}) > 1:
            multi.append((s, sorted(hits)))
            continue
        name, no, title = sorted(hits)[0]
        rows.append({'symbol': s, 'kap_stock_code': s[:-3], 'kap_title': title, 'kap_sector': name,
                     'kap_sector_no': no, 'kap_parent_no': no.split('.')[0],
                     'source_url': URL, 'retrieved_at': retrieved_at, 'source_sha256': sha})
    if missing or multi:
        sys.exit(f"KAP'ta bulunamayan: {missing}; birden cok sektorlu: {multi}")
    size = Counter(r['kap_sector'] for r in rows)
    for r in rows:
        r['group'] = r['kap_sector'] if size[r['kap_sector']] >= MIN_GROUP else 'DIGER'
    return rows


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--fetch', action='store_true')
    g.add_argument('--html')
    a = ap.parse_args()
    from data.bist30_symbols import BIST30_SYMBOLS
    if a.fetch:
        req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0 (research client)'})
        html = urllib.request.urlopen(req, timeout=60).read()
        retrieved = str(date.today())
        os.makedirs(SRC_DIR, exist_ok=True)
        with gzip.open(os.path.join(SRC_DIR, f'kap_sektorler_{retrieved}.html.gz'), 'wb') as f:
            f.write(html)
    else:
        opener = gzip.open if a.html.endswith('.gz') else open
        with opener(a.html, 'rb') as f:
            html = f.read()
        m = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(a.html))
        retrieved = m.group(1) if m else 'unknown'
    rows = build(html, retrieved, BIST30_SYMBOLS)
    cols = ['symbol', 'group', 'kap_sector', 'kap_sector_no', 'kap_parent_no', 'kap_stock_code',
            'kap_title', 'source_url', 'retrieved_at', 'source_sha256']
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    groups = Counter(r['group'] for r in rows)
    print(json.dumps({'n': len(rows), 'n_groups': len(groups), 'groups': dict(groups)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
