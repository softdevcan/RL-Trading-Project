"""
Faz 9 E1: KAP 'Endeks Sirketlerinde Degisiklik' (oda-11700) ve 'BIST Pay Endeksleri' (oda-13500) bildirimlerinden
BIST 30 (XU030) donem ici olaylari. Girdi: `scripts/fetch_kap_disclosures.py` ciktisi
(data/universe/sources/kap_api/list_49.json). Ayrinti cagrisi basina 10,5 sn (dakikada 6).

    python scripts/build_kap_index_events.py [--no-fetch]

Cikti: data/universe/bist30_kap_events.csv — satir basina bir XU030 olayi:
  effective_date, action (in|out), name (bildirimdeki pay adi), related_codes (bildirimin relatedStocks
  kodlari), kap_index, kap_time, kap_link, subject, html_sha256.
Ayrica tum taranan bildirimlerin XU030 satiri tasiyip tasimadigi: data/universe/bist30_kap_scan.json.
Kapsam: yalniz test gecidinin verisi (2023-01-02 .. son bildirim); 2017-2018 bu kaynakta YOK.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
from html.parser import HTMLParser
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import kap_api as K  # noqa: E402

TYPES = ('oda-11700', 'oda-13500')
OUT_CSV = os.path.join(ROOT, 'data', 'universe', 'bist30_kap_events.csv')
OUT_SCAN = os.path.join(ROOT, 'data', 'universe', 'bist30_kap_scan.json')
INDEX_CODE = 'XU030'
CHANGES_CSV = os.path.join(ROOT, 'data', 'universe', 'bist30_changes.csv')


class _Rows(HTMLParser):
    """Tablo satirlari -> hucre metinleri listesi (ic ice tablo satirlari ayri satir)."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._row = []
        elif tag in ('td', 'th') and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._row is not None and self._cell is not None:
            self._row.append(' '.join(''.join(self._cell).split()))
            self._cell = None
        elif tag == 'tr' and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def decode_html(detail: dict) -> str:
    hm = detail.get('htmlMessages') or []
    hm = hm if isinstance(hm, list) else [hm]
    parts = []
    for m in hm:
        raw = base64.b64decode(m.get('tr') or '')
        try:
            parts.append(raw.decode('utf-8'))
        except UnicodeDecodeError:
            parts.append(raw.decode('cp1254'))
    return '\n'.join(parts)


def xu030_rows(html_text: str):
    """Olay tablosu satirlari: [pay adi, dahil endeks, cikarildigi endeks, gecerlilik (gg/aa/yyyy)] + ayni
    dortlunun EN kopyasi + simge hucresi (9 hucre; 2026-11-15 #1092672'de goruldu). Ilk 4 hucre alinir,
    tarih bicimi denetlenir; (ad, yon, tarih) tekillestirilir."""
    import re
    p = _Rows()
    p.feed(html_text)
    seen, out = set(), []
    for r in p.rows:
        if len(r) < 4 or not re.fullmatch(r'\d{2}/\d{2}/\d{4}', r[3]):
            continue
        name, inc, exc, date = r[:4]
        for action, cell in (('in', inc), ('out', exc)):
            if INDEX_CODE in cell.replace(' ', '').split(','):
                key = (name, action, date)
                if key not in seen:
                    seen.add(key)
                    out.append({'name': name, 'action': action, 'effective_date': date})
    return out


def quarterly_crosscheck(detail: dict, html_text: str, changes_csv: str) -> Optional[dict]:
    """Pozitif kontrol: 'BIST Pay Endeksleri' ceyreklik duyurusunun TR PDF ekinde BIST 30 blogu, ayni ceyregin
    bist30_changes.csv (Borsa Istanbul arsivinden) kodlarini iceriyor mu. Blok = 'BIST 30 ENDEKSI' ile
    sonraki 'BIST ' basligi arasi (alinacak/cikarilacak/yedek birlikte; yon ayrica denetlenmez)."""
    import re
    import fitz
    if 'BIST 30' not in html_text:
        return None
    atts = [a for a in (detail.get('attachmentUrls') or []) if 'Donemsel' in a.get('fileName', '')]
    if not atts:
        return {'status': 'ek yok'}
    a = atts[0]
    p = K.download_attachment(a['url'].rsplit('/', 1)[1], a['fileName'])
    t = re.sub(r'\s+', ' ', ' '.join(pg.get_text() for pg in fitz.open(p)))
    m = re.search(r'BIST 30 ENDEKS\S* (.*?)(?= BIST (?!30)\S)', t)
    block = m.group(1) if m else ''
    per = re.search(r'(\d{2})[./](\d{2})[./](\d{4})-\d{2}[./]\d{2}[./]\d{4} DÖNEMİ', t)
    quarter = None if not per else f'{per.group(3)}Q{(int(per.group(2)) - 1) // 3 + 1}'
    with open(changes_csv, encoding='utf-8') as fh:
        allq = [r for r in csv.DictReader(fh) if r['quarter'] == quarter]
    want = sorted({r['code'] for r in allq if r['action'] in ('add', 'remove')})
    # degisiklik yok: alinacak/cikarilacak sutunlari bos ('-- ----- -----' iki kez), ardindan yedek listesi
    no_change_pdf = bool(re.match(r'ALINACAK PAYLAR ÇIKARILACAK PAYLAR YEDEK PAYLAR -- -+ -+ -- -+ -+ 1 ', block))
    found = {c: bool(re.search(rf'\b{c}\b', block)) for c in want}
    if want:
        agree = all(found.values()) and not no_change_pdf
    else:
        agree = no_change_pdf and any(r['action'] == 'none' for r in allq)
    with open(p, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'status': 'ok' if block and quarter and allq else 'eslesme yok', 'quarter': quarter,
            'expected_codes': want, 'found': found, 'pdf_no_change': no_change_pdf,
            'agrees_with_archive': bool(block and quarter and allq and agree),
            'pdf': a['fileName'], 'pdf_sha256': sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-fetch', action='store_true', help='yalniz onbellek (ag cagrisi yok)')
    a = ap.parse_args()
    lst = json.load(open(os.path.join(K.RAW_DIR, 'list_49.json'), encoding='utf-8'))['rows']
    cand = [x for x in lst if any(s.startswith(TYPES) for s in (x.get('subReportIds') or []))]
    events, scan = [], []
    for i, x in enumerate(cand):
        idx = int(x['disclosureIndex'])
        if a.no_fetch and not os.path.exists(os.path.join(K.RAW_DIR, K._raw_name(f'/disclosureDetail/{idx}',
                                                                                {'fileType': 'html'}))):
            scan.append({'kap_index': idx, 'status': 'not_fetched'})
            continue
        d = K.detail(idx, 'html')
        d = d[0] if isinstance(d, list) else d
        h = decode_html(d)
        rows = xu030_rows(h)
        sha = hashlib.sha256(h.encode('utf-8')).hexdigest()
        codes = ' '.join(s.get('code', '') for s in (d.get('relatedStocks') or []) if isinstance(s, dict))
        subj = (d.get('subject') or {}).get('tr', '')
        rec = {'kap_index': idx, 'time': d.get('time'), 'subject': subj, 'n_xu030': len(rows),
               'mentions_bist30': ('XU030' in h) or ('BIST 30' in h), 'html_sha256': sha, 'html_len': len(h)}
        qc = quarterly_crosscheck(d, h, CHANGES_CSV) if subj.startswith('BIST Pay') else None
        if qc:
            rec['quarterly_crosscheck'] = qc
        scan.append(rec)
        for r in rows:
            events.append({**r, 'related_codes': codes, 'kap_index': idx, 'kap_time': d.get('time'),
                           'kap_link': d.get('link'), 'subject': subj, 'html_sha256': sha})
        print(f'{i + 1}/{len(cand)} #{idx} {d.get("time")} XU030 satiri: {len(rows)}', flush=True)
    cols = ['effective_date', 'action', 'name', 'related_codes', 'kap_index', 'kap_time', 'kap_link',
            'subject', 'html_sha256']
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for e in events:
            w.writerow({c: e.get(c) for c in cols})
    with open(OUT_SCAN, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'source': K.base_url(), 'company': '49', 'types': list(TYPES), 'n_candidates': len(cand),
                   'n_events': len(events), 'coverage_note': 'test gecidi verisi 2023-01-02 .. ; 2017-2018 yok',
                   'scan': scan}, fh, ensure_ascii=False, indent=1)
    qcs = [x['quarterly_crosscheck'] for x in scan if x.get('quarterly_crosscheck')]
    print(len(events), 'XU030 olayi ->', OUT_CSV)
    print('bildirimde XU030/BIST 30 geciyor:', sum(bool(x.get('mentions_bist30')) for x in scan),
          '| ceyreklik capraz denetim:', [(q.get('quarter'), q.get('agrees_with_archive')) for q in qcs])


if __name__ == '__main__':
    main()
