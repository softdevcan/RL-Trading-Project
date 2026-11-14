"""Faz 9 E1 (K3) — tarihsel BIST 30 uyeligi: Borsa Istanbul endeks duyurulari.

    python scripts/build_membership.py index      # duyuru dizinini tara (taxonomy/term/16, sayfa sayfa)
    python scripts/build_membership.py fetch      # BIST pay endeksi duyurularinin ham sayfalarini indir
    python scripts/build_membership.py parse      # BIST 30 alinan/cikarilan tablolarini ayristir

Ham sayfalar ucuncu taraf icerigi -> `data/universe/sources/bist/` (gitignore); her sayfanin SHA-256'si
ve erisim tarihi dizinde (`index.json`) tutulur. Istekler arasi bekleme var (sunucuya yuk bindirme).
Ayrintili tasarim: plan §9.21 (E1).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SRC =os.path.join(ROOT, 'data', 'universe', 'sources', 'bist')
INDEX = os.path.join(SRC, 'index.json')
BASE = 'https://borsaistanbul.com'
UA = {'User-Agent': 'Mozilla/5.0 (research client)'}
DELAY_S = 1.0
STOP_BELOW_ID = 12100          # 2013 Q3 duyurusu #12189; K17 RL penceresi 2013-06
LINK = re.compile(r'href="(?:https?://(?:www\.)?borsaistanbul\.com)?/duyuru/(\d+)/([^"#?]+)"')
# BIST pay endeksi uyeligini etkileyen duyurular: ceyreklik donemsel + donem ici (pazar degisikligi,
# islem sirasi kapatma, kotasyondan cikma vb. "BIST pay endeksleri kapsaminda yapilacak degisiklikler")
RELEVANT = re.compile(r'bist-pay-endeksleri-(kapsaminda-yer-alacak|donemsel)|'
                      r'pay-endeksleri-kapsaminda-yapilacak-degisiklik|'
                      r'bist-pay-endeksleri-kapsamindan-cikarilmasi|endeks-degisiklik')
EXCLUDE = re.compile(r'katilim|kyd|eurobond|osba|temettu|geri-alim|surdurulebilirlik|kural-set|temel-kural')


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read()


def _load_index() -> dict:
    if os.path.exists(INDEX):
        with open(INDEX, encoding='utf-8') as f:
            return json.load(f)
    return {'links': {}, 'pages': {}, 'fetched': {}}


def _save_index(ix: dict) -> None:
    os.makedirs(SRC, exist_ok=True)
    with open(INDEX, 'w', encoding='utf-8') as f:
        json.dump(ix, f, ensure_ascii=False, indent=1)


QUARTER_SLUG = re.compile(r'^(\d{4})-yili-(birinci|ikinci|ucuncu|dorduncu)-uc-aylik-donemi-icin-'
                          r'bist-pay-endeksleri-kapsaminda-yer-alacak-paylar')


def cmd_index(max_pages: int = 0) -> None:
    """TR liste sayfasini indir (statik 334 baglanti; taxonomy/term/16 tam arsiv DEGIL) ve
    ceyreklik BIST pay endeksi duyurularinin kimligini cikar. Eksik ceyrekler: `probe`."""
    ix = _load_index()
    html = _get(f'{BASE}/bist-endeksleri/endeks-duyurulari')
    os.makedirs(SRC, exist_ok=True)
    with open(os.path.join(SRC, 'tr_list.html'), 'wb') as f:
        f.write(html)
    ix['list'] = {'url': f'{BASE}/bist-endeksleri/endeks-duyurulari',
                  'sha256': hashlib.sha256(html).hexdigest(), 'at': str(date.today())}
    inv = {v: k for k, v in QWORD.items()}
    quarters = ix.setdefault('quarters', {})
    for did, slug in LINK.findall(html.decode('utf-8', 'replace')):
        ix['links'][did] = slug
        m = QUARTER_SLUG.match(slug)
        if m:
            quarters[f'{m.group(1)}Q{inv[m.group(2)]}'] = {'id': int(did), 'slug': slug, 'via': 'tr_list'}
    _save_index(ix)
    print(f'{len(quarters)} ceyrek duyurusu listede', flush=True)


def relevant(slug: str) -> bool:
    return bool(RELEVANT.search(slug)) and not EXCLUDE.search(slug)


def cmd_fetch() -> None:
    ix = _load_index()
    todo = sorted((int(d), s) for d, s in ix['links'].items()
                  if relevant(s) and d not in ix['fetched'] and int(d) >= STOP_BELOW_ID)
    print(f'{len(todo)} duyuru indirilecek', flush=True)
    for did, slug in todo:
        html = _get(f'{BASE}/duyuru/{did}/{slug}')
        path = os.path.join(SRC, f'duyuru_{did}.html')
        with open(path, 'wb') as f:
            f.write(html)
        ix['fetched'][str(did)] = {'slug': slug, 'sha256': hashlib.sha256(html).hexdigest(),
                                   'at': str(date.today())}
        _save_index(ix)
        time.sleep(DELAY_S)
    print('bitti', flush=True)


# Ceyrek -> duyuru kimligi. Kaynak: TR liste sayfasi (/bist-endeksleri/endeks-duyurulari, statik 334
# baglanti; 2013 Q3-2025 Q3'un 45'i) + EN liste (/en/indices/index-announcements). Eksik ceyrekler
# `probe` ile bulunur: slug zorunlu (yanlis slug 404), kimlikler tarihle artar; pencere liste
# sayfalarindaki komsu duyuru tarihlerinden (2026-11-14).
SLUG_OLD = '{y}-yili-{q}-uc-aylik-donemi-icin-bist-pay-endeksleri-kapsaminda-yer-alacak-paylar-belirlenmistir'
SLUG_NEW = 'bist-pay-endeksleri-donemsel-degisiklikleri'
QWORD = {1: 'birinci', 2: 'ikinci', 3: 'ucuncu', 4: 'dorduncu'}
PROBE = [  # (yil, ceyrek, slug, id_bas, id_son)
    (2024, 3, SLUG_OLD, 14668, 14684), (2024, 4, SLUG_OLD, 14696, 14709),
    (2025, 1, SLUG_OLD, 14732, 14795), (2025, 2, SLUG_OLD, 14732, 14795),
    (2025, 4, SLUG_NEW, 15216, 15235), (2026, 1, SLUG_NEW, 15293, 15293),
    (2026, 2, SLUG_NEW, 15347, 15393), (2026, 3, SLUG_NEW, 15483, 15483),
    (2026, 4, SLUG_NEW, 15598, 15598),
]


def cmd_probe() -> None:
    ix = _load_index()
    ix.setdefault('probe', {})
    for y, q, tmpl, a, b in PROBE:
        key = f'{y}Q{q}'
        if key in ix['probe']:
            continue
        slug = tmpl.format(y=y, q=QWORD[q])
        hits = []
        for did in range(a, b + 1):
            req = urllib.request.Request(f'{BASE}/duyuru/{did}/{slug}', headers=UA)
            try:
                html = urllib.request.urlopen(req, timeout=60).read()
                title = re.search(rb'<title>([^<]*)', html)
                hits.append({'id': did, 'slug': slug,
                             'title': title.group(1).decode('utf-8', 'replace') if title else ''})
            except urllib.error.HTTPError:
                pass
            time.sleep(DELAY_S)
        ix['probe'][key] = hits
        _save_index(ix)
        print(key, [(h['id'], h['title'][:60]) for h in hits], flush=True)


COLS = ('ALINACAK', 'ÇIKARILACAK', 'YEDEK')
CODE = re.compile(r'^[A-Z0-9]{4,6}$')


def _cell_items(cell_html: str) -> list:
    """Bir tablo hucresindeki ogeler: <br>/<p>/<div> ile ayrilmis metin parcalari; '--' bos."""
    import html as H
    parts = re.split(r'<br\s*/?>|</p>|</div>|</li>', cell_html, flags=re.I)
    out = []
    for p in parts:
        t = ' '.join(H.unescape(re.sub(r'<[^>]+>', ' ', p)).split())
        if t and not re.fullmatch(r'-+', t):
            out.append(t)
    return out


def parse_tables(html: str) -> dict:
    """Endeks basligi (ör. 'BIST 30') -> {'ALINACAK': [...], 'ÇIKARILACAK': [...], 'YEDEK': [...]}.

    Yeni bicim: her kolon uclu (sira, KOD, unvan) -> KOD alinir ('kind': 'code').
    Eski bicim: kolon basina tek hucre, adlar <br> ile ayrik -> ad alinir ('kind': 'name').
    """
    import html as H
    out = {}
    for m in re.finditer(r'<table.*?</table>', html, flags=re.S | re.I):
        pre = H.unescape(re.sub(r'<[^>]+>', ' ', html[max(0, m.start() - 1500):m.start()]))
        heads = re.findall(r'(BIST [A-ZÇĞİÖŞÜ0-9 .\-]+?) ENDEKS', ' '.join(pre.split()))
        if not heads:
            continue
        name = heads[-1].strip()
        rows = re.findall(r'<tr.*?</tr>', m.group(0), flags=re.S | re.I)
        grid = [re.findall(r'<t[dh]([^>]*)>(.*?)</t[dh]>', r, flags=re.S | re.I) for r in rows]
        if not grid:
            continue
        col_of = []   # baslik hucrelerini colspan kadar ac
        for attrs, c in grid[0]:
            span = re.search(r'colspan="?(\d+)', attrs)
            h = ' '.join(H.unescape(re.sub(r'<[^>]+>', ' ', c)).split()).upper()
            col_of += [next((k for k in COLS if h.startswith(k)), None)] * (int(span.group(1)) if span else 1)
        # Yeni bicim: kolon basina 3 alt sutun (sira, KOD, unvan); eski bicim: kolon basina 1 hucre
        width = max((len(r) for r in grid[1:]), default=0)
        kind = 'code' if width > len(set(col_of) - {None}) else 'name'
        res = {k: [] for k in COLS}
        for r in grid[1:]:
            for j, (_a, c) in enumerate(r):
                k = col_of[j] if j < len(col_of) else None
                if k is None:
                    continue
                for it in _cell_items(c):
                    if kind == 'name' or CODE.match(it):
                        res[k].append(it)
        res['kind'] = kind
        out.setdefault(name, res)
    return out


KAP_INDEX_URL = 'https://kap.org.tr/tr/Endeksler?indice=QklTVCAzMA'
ANCHOR = os.path.join(ROOT, 'data', 'universe', 'bist30_anchor.json')


def cmd_anchor() -> None:
    """Zincirin capasi: BUGUNKU BIST 30 uyeleri (KAP 'Endeksler' gomulu verisi, kod XU030)."""
    html = _get(KAP_INDEX_URL)
    today = str(date.today())
    os.makedirs(SRC, exist_ok=True)
    import gzip
    with gzip.open(os.path.join(SRC, f'kap_endeksler_{today}.html.gz'), 'wb') as f:
        f.write(html)
    t = html.decode('utf-8').replace('\\"', '"')
    i = t.find('"code":"XU030"')
    j = t.find('"code":"', i + 10)
    codes = re.findall(r'"stockCode":"([^"]+)"', t[i:j])
    if i < 0 or len(codes) != 30 or len(set(codes)) != 30:
        sys.exit(f'XU030 bulunamadi ya da 30 degil: {len(codes)}')
    from data.bist30_symbols import BIST30_SYMBOLS
    panel = {s[:-3] for s in BIST30_SYMBOLS}
    doc = {'index': 'BIST 30 (XU030)', 'as_of': today, 'source_url': KAP_INDEX_URL,
           'source_sha256': hashlib.sha256(html).hexdigest(), 'codes': codes,
           'panel_not_in_index': sorted(panel - set(codes)), 'index_not_in_panel': sorted(set(codes) - panel)}
    with open(ANCHOR, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: doc[k] for k in ('as_of', 'panel_not_in_index', 'index_not_in_panel')}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['index', 'probe', 'anchor', 'fetch', 'parse'])
    ap.add_argument('--max-pages', type=int, default=200)
    a = ap.parse_args()
    if a.cmd == 'index':
        cmd_index(a.max_pages)
    elif a.cmd == 'anchor':
        cmd_anchor()
    elif a.cmd == 'probe':
        cmd_probe()
    elif a.cmd == 'fetch':
        cmd_fetch()
    else:
        sys.exit('parse: ham sayfalar incelendikten sonra yazilacak')


if __name__ == '__main__':
    main()
