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
from html.parser import HTMLParser
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
                      r'pay-endeksleri-kapsaminda-yapilacak-degisiklik|bist-pazar-endeksleri-kapsaminda|'
                      r'endeks(ler)?-kapsamindan-cikar|endeks-degisiklik|bist-30|endekslerinde-yapilacak')
EXCLUDE = re.compile(r'katilim|kyd|eurobond|osba|temettu|geri-alim|surdurulebilirlik|kural-set|temel-kural|'
                     r'metodoloji|viop|vadeli|spot-|hesaplanmaya-basla|borclanma|tahvil|bono|kira-sertifika')


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read()


def _load_index() -> dict:
    ix = {'links': {}, 'pages': {}, 'fetched': {}}
    if os.path.exists(INDEX):
        with open(INDEX, encoding='utf-8') as f:
            ix = json.load(f)
    import glob
    for shard in sorted(glob.glob(os.path.join(SRC, 'nodes_*.json'))):   # paralel `nodes` parcalari
        with open(shard, encoding='utf-8') as f:
            ix.setdefault('nodes', {}).update(json.load(f))
    return ix


def _save_index(ix: dict) -> None:
    os.makedirs(SRC, exist_ok=True)
    with open(INDEX, 'w', encoding='utf-8', newline='\n') as f:
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
    """Indirme adayi: slug'da 'endeks' gecen, endeks AILESI disi olmayan her duyuru. BIST 30'u
    etkileyip etkilemedigine slug degil GOVDE karar verir (`mentions_bist30`)."""
    return ('endeks' in slug or bool(RELEVANT.search(slug))) and not EXCLUDE.search(slug)


FETCHED = os.path.join(SRC, 'fetched.json')   # indirme manifesti (index.json'dan ayri: tarama ile cakismaz)


def _load_fetched() -> dict:
    if os.path.exists(FETCHED):
        with open(FETCHED, encoding='utf-8') as f:
            return json.load(f)
    return {}


def candidates(ix: dict) -> dict:
    """Indirilecek duyurular: kimlik -> slug. Ceyreklik (liste + probe + nodes) ve donem ici
    (`relevant` kalibina uyan, nodes taramasindan)."""
    out = {str(v['id']): v['slug'] for v in ix.get('quarters', {}).values()}
    for hits in ix.get('probe', {}).values():
        out.update({str(h['id']): h['slug'] for h in hits})
    for did, slug in ix.get('nodes', {}).items():
        if not slug.startswith('?') and (QUARTER_SLUG.match(slug) or slug.startswith(SLUG_NEW) or relevant(slug)):
            out[did] = slug
    return out


def cmd_fetch() -> None:
    ix = _load_index()
    done = _load_fetched()
    todo = sorted((int(d), s) for d, s in candidates(ix).items() if d not in done)
    print(f'{len(todo)} duyuru indirilecek', flush=True)
    for did, slug in todo:
        html = _get(f'{BASE}/duyuru/{did}/{slug}')
        with open(os.path.join(SRC, f'duyuru_{did}.html'), 'wb') as f:
            f.write(html)
        done[str(did)] = {'slug': slug, 'url': f'{BASE}/duyuru/{did}/{slug}',
                          'sha256': hashlib.sha256(html).hexdigest(), 'at': str(date.today())}
        with open(FETCHED, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(done, f, ensure_ascii=False, indent=1)
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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def cmd_nodes(a: int, b: int, delay: float = 0.5) -> None:
    """Kimlik -> kanonik TR slug: `/node/<id>` 301 ile `/duyuru/<id>/<slug>`'a yonlenir (2026-11-14
    gozlemi). Slug tahmini gerekmez; dizin dahil tum duyurular (donem ici degisiklikler de) gorunur.
    Govde indirilmez. Sonuc `nodes_<a>_<b>.json` parcasi (kimlik -> slug ya da '?HTTP kodu'); kaldigi
    yerden surer; `_load_index` parcalari birlestirir."""
    ix = _load_index()
    seen = ix.get('nodes', {})
    shard = os.path.join(SRC, f'nodes_{a}_{b}.json')     # paralel kosular cakismasin diye parca dosya
    nodes = {}
    if os.path.exists(shard):
        with open(shard, encoding='utf-8') as f:
            nodes = json.load(f)
    op = urllib.request.build_opener(_NoRedirect)

    def save():
        with open(shard, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(nodes, f, ensure_ascii=False, indent=0)
    n_new = 0
    for did in range(a, b + 1):
        if str(did) in nodes or str(did) in seen:
            continue
        try:
            op.open(urllib.request.Request(f'{BASE}/node/{did}', headers=UA, method='HEAD'), timeout=60)
            nodes[str(did)] = '?200'
        except urllib.error.HTTPError as e:
            loc = e.headers.get('Location', '') if e.code in (301, 302) else ''
            m = re.search(r'/duyuru/(\d+)/([^"#?\s]+)', loc)
            nodes[str(did)] = m.group(2) if m and m.group(1) == str(did) else f'?{e.code}:{loc[-80:]}'
        except Exception as e:  # ag hatasi: kaydetme, sonraki kosuda tekrar denenir
            print(did, 'hata', e, flush=True)
            continue
        n_new += 1
        if n_new % 50 == 0:
            save()
            print(did, flush=True)
        time.sleep(delay)
    save()
    print(f'{n_new} yeni kimlik', flush=True)


COLS = ('ALINACAK', 'ÇIKARILACAK', 'YEDEK')
CODE = re.compile(r'^[A-Z0-9]{4,6}$')


NUM = re.compile(r'^\d+\.?$')
HEAD = re.compile(r'(BIST [0-9A-ZÇĞİÖŞÜ .\-]+?) ENDEKS')


class _Tables(HTMLParser):
    """Duyuru govdesi -> sirali olaylar: ('text', metin) ve ('table', [[hucre, ...], ...]).
    Hucre = {'items': [...], 'colspan': n, 'rowspan': n}; ogeler <br>/<p>/<li>/<div> ile ayrilir.
    Ic ice tablolarda (Word ciktisi, ör. 2022 Q1) yalniz EN ICTEKI tablolar tablo olarak verilir;
    kapsayici tablonun hucre metni ('BIST 30 ENDEKSI' basligi) metin olayi olarak akar."""
    SPLIT = {'br', 'p', 'li', 'div'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.events, self.stack, self.buf = [], [], []

    def _flush(self):
        t = ' '.join(''.join(self.buf).split())
        self.buf = []
        if not t:
            return
        top = self.stack[-1] if self.stack else None
        if top is not None and top['cell'] is not None and not top['nested']:
            top['cell']['items'].append(t)
        else:
            self.events.append(('text', t))

    def handle_starttag(self, tag, attrs):
        if tag in self.SPLIT or tag in ('table', 'td', 'th', 'tr'):
            self._flush()
        top = self.stack[-1] if self.stack else None
        if tag == 'table':
            if top is not None:          # kapsayici: hucre ogeleri baslik metnidir
                top['nested'] = True
                for t in self.stack:
                    for r in t['rows']:
                        for c in r:
                            self.events.extend(('text', i) for i in c['items'])
                            c['items'] = []
            self.stack.append({'rows': [], 'cell': None, 'nested': False})
        elif top is None:
            return
        elif tag == 'tr':
            top['rows'].append([])
        elif tag in ('td', 'th'):
            d = dict(attrs)
            num = lambda k: int(m.group(0)) if (m := re.match(r'\d+', d.get(k) or '')) else 1
            top['cell'] = {'items': [], 'colspan': num('colspan'), 'rowspan': num('rowspan')}
            if not top['rows']:
                top['rows'].append([])
            top['rows'][-1].append(top['cell'])

    def handle_endtag(self, tag):
        if tag in self.SPLIT or tag in ('td', 'th', 'table'):
            self._flush()
        if not self.stack:
            return
        top = self.stack[-1]
        if tag in ('td', 'th'):
            top['cell'] = None
        elif tag == 'table':
            self.stack.pop()
            if not top['nested']:
                self.events.append(('table', top['rows']))

    def handle_data(self, data):
        self.buf.append(data)


def _grid(rows: list) -> list:
    """HTML tablo algoritmasi: rowspan/colspan acilir. Sonuc satir -> [(kok_mu, hucre) | None ...]."""
    grid, carry = [], {}   # carry: sutun -> (kalan satir, hucre)
    for r in rows:
        line, col, cells = [], 0, list(r)
        while cells or any(c >= col for c in carry):
            if col in carry:
                n, cell = carry.pop(col)
                line.append((False, cell))
                if n > 1:
                    carry[col] = (n - 1, cell)
                col += 1
                continue
            if not cells:
                line.append(None)
                col += 1
                continue
            cell = cells.pop(0)
            for k in range(cell['colspan']):
                line.append((k == 0, cell))
                if cell['rowspan'] > 1:
                    carry[col + k] = (cell['rowspan'] - 1, cell)
            col += cell['colspan']
        grid.append(line)
    return grid


def _clean(item: str) -> str:
    return re.sub(r'^\d+\.\s*', '', item).strip()


def parse_tables(html: str) -> dict:
    """Endeks basligi (ör. 'BIST 30') -> {'ALINACAK': [...], 'ÇIKARILACAK': [...], 'YEDEK': [...],
    'kind': 'code'|'name', 'names': {kod: unvan}}.

    Izgara rowspan/colspan acilarak kurulur; kolon (alinacak/cikarilacak/yedek) baslik satirindaki
    konumdan. Endeks adi tablodan onceki son 'BIST .. ENDEKSI' metninden ya da tablo ici baslik
    satirindan (2016 Q1'de tum endeksler tek tabloda). Yeni bicim (sira hucresi 'N.' var): sira
    hucresinden sonraki ilk hucre KOD, sonrakiler unvan. Eski bicim: hucredeki her oge bir ad ('1. ' atilir).
    Yalniz o satirda baslayan hucreler okunur (rowspan devami onceki satirin unvanidir).
    """
    a = html.find('id="printableArea"')
    p = _Tables()
    p.feed(html[a:] if a >= 0 else html)
    out, current = {}, None
    for kind_ev, ev in p.events:
        if kind_ev == 'text':
            h = HEAD.findall(ev.upper())
            if h:
                current = h[-1].strip()
            continue
        grid = _grid(ev)
        flat = [c for line in grid for x in line if x and x[0] for c in [x[1]]]
        is_code = any(len(c['items']) == 1 and NUM.match(c['items'][0]) for c in flat)
        colmap = None
        for line in grid:
            roots = [(j, x[1]) for j, x in enumerate(line) if x and x[0]]
            text = ' '.join(' '.join(c['items']) for _, c in roots).upper()
            h = HEAD.findall(text)
            if h and not any(k in text for k in COLS):
                current = h[-1].strip()
                continue
            if any(k in text for k in COLS) and not any(NUM.match(i) for _, c in roots for i in c['items']):
                colmap = {}
                for j, c in roots:
                    lab = next((k for k in COLS if ' '.join(c['items']).upper().startswith(k)), None)
                    for t in range(c['colspan']):
                        colmap[j + t] = lab
                continue
            if colmap is None or current is None:
                continue
            res = out.setdefault(current, {**{k: [] for k in COLS}, 'kind': 'code' if is_code else 'name',
                                           'names': {}, 'marked': []})
            for k in COLS:
                cells = [c for j, c in roots if colmap.get(j) == k]
                if is_code:
                    vals = [i for c in cells for i in c['items'] if not NUM.match(i) and not re.fullmatch(r'-+', i)]
                    if vals:
                        res[k].append(vals[0])
                        if len(vals) > 1:
                            res['names'][vals[0]] = ' '.join(vals[1:])
                        mk = MARK.findall(' '.join(vals))
                        if mk:
                            res['marked'].append((k, vals[0], mk[0]))
                else:
                    for i in (_clean(i) for c in cells for i in c['items'] if not re.fullmatch(r'-+', i)):
                        if i:
                            res[k].append(i)
                            if MARK.search(i):
                                res['marked'].append((k, i, MARK.findall(i)[0]))
    return out


MARK = re.compile(r'\(\*+\)')


def footnotes(html: str) -> list:
    """Duyuru govdesindeki dipnot TANIMLARI: tablo disi metinde '(*)' ile baslayan parcalar
    (ör. #11503: '(*) 21/06/2017 tarihinden itibaren endekslerden cikarilacaktir'). Isaretli ogeler
    `parse_tables(...)['marked']`'ta; dipnot donem ici bir olay olabilir -> `bist30_footnotes.csv` (aday)."""
    a = html.find('id="printableArea"')
    p = _Tables()
    p.feed(html[a:] if a >= 0 else html)
    return [t for kind, t in p.events if kind == 'text' and MARK.match(t)]


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
    with open(ANCHOR, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: doc[k] for k in ('as_of', 'panel_not_in_index', 'index_not_in_panel')}), flush=True)


UNI = os.path.join(ROOT, 'data', 'universe')
CHANGES = os.path.join(UNI, 'bist30_changes.csv')
NAME_MAP = os.path.join(UNI, 'bist_name_map.csv')
FOOTNOTES = os.path.join(UNI, 'bist30_footnotes.csv')   # isaretli BIST 30 ogeleri + dipnot metni (olay ADAYI)
EVENTS = os.path.join(UNI, 'bist30_events.csv')          # donem ici + kod degisikligi (elle, alintili)
MEMBERSHIP = os.path.join(UNI, 'bist30_membership.csv')
INDEX_NAME = 'BIST 30'


def _norm_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize('NFKD', s.upper().replace('İ', 'I'))
    s = ''.join(c for c in s if not unicodedata.combining(c)).replace('.', '')
    return ' '.join(re.sub(r'[^A-Z0-9 ]', ' ', s).split())


def _body_text(html: str) -> str:
    import html as H
    a = html.find('id="printableArea"')
    body = html[a:].split('<footer')[0] if a >= 0 else ''
    return ' '.join(H.unescape(re.sub(r'<[^>]+>', ' ', body)).split())


def _ann_date(html: str) -> str:
    m = re.search(r'<p class="date[^"]*">\s*(\d{2})\.(\d{2})\.(\d{4})\s*</p>', html)
    return f'{m[3]}-{m[2]}-{m[1]}' if m else ''


def _period(text: str):
    """Govdedeki ILK 'gg/aa/yyyy-gg/aa/yyyy' araligi (donem basi, sonu)."""
    m = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})', text)
    return (f'{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}', f'{m[6]}-{int(m[5]):02d}-{int(m[4]):02d}') if m else None


def _read(did: str) -> str:
    with open(os.path.join(SRC, f'duyuru_{did}.html'), encoding='utf-8', errors='replace') as f:
        return f.read()


def quarter_announcements(fetched: dict) -> dict:
    """Ceyrek ('2014Q1') -> [(kimlik, tablo, donem, ilan)]. Ceyrek eski slug'dan ya da (yeni slug)
    govdedeki donem basindan. Yalniz BIST 30 tablosu olan sayfalar (bos govdeli ikizler elenir)."""
    inv = {v: k for k, v in QWORD.items()}
    out = {}
    for did, v in fetched.items():
        m = QUARTER_SLUG.match(v['slug'])
        if not m and not v['slug'].startswith(SLUG_NEW):
            continue
        html = _read(did)
        t = parse_tables(html).get(INDEX_NAME)
        per = _period(_body_text(html))
        if t is None or per is None:
            continue
        key = (f'{m.group(1)}Q{inv[m.group(2)]}' if m
               else f'{per[0][:4]}Q{(int(per[0][5:7]) - 1) // 3 + 1}')
        out.setdefault(key, []).append((did, t, per, _ann_date(html)))
    return out


def build_name_map(fetched: dict) -> dict:
    """Eski bicim ad -> kod, MEKANIK: ayni kaynagin yeni bicim tablolarindaki (tum endeksler)
    kod-unvan ciftlerinde normalize unvan BIREBIR esit olmali ve tek koda gitmeli (onek eslesmesi YOK;
    ör. 'VESTEL' -> VESTL, VESBE degil). Isaret ('(*)') normalizasyonda duser ama `marked`'ta kalir.
    Donus: norm(unvan) -> {kod: [(kanit kimligi, unvan, ilan tarihi), ...]} (TUM kanitlar)."""
    pairs = {}
    for did in sorted(fetched, key=int):
        html = _read(did)
        ann = _ann_date(html)
        for r in parse_tables(html).values():
            if r['kind'] == 'code':
                for c, u in r['names'].items():
                    pairs.setdefault(_norm_name(u), {}).setdefault(c, []).append((did, u, ann))
    return pairs


def _row(q, per, ann, action, raw, code, kind, did, fetched, twins, marked=''):
    return {'quarter': q, 'period_start': per[0], 'period_end': per[1], 'announced': ann,
            'action': action, 'raw': raw, 'code': code, 'format': kind, 'marked': marked, 'source_id': did,
            'source_url': fetched[did]['url'], 'source_sha256': fetched[did]['sha256'],
            'accessed': fetched[did]['at'], 'twins': twins}


def cmd_parse() -> None:
    """Ceyreklik BIST 30 degisiklikleri -> `bist30_changes.csv`; eski bicim adlari -> `bist_name_map.csv`;
    isaretli ('(*)') BIST 30 ogeleri ve dipnot metinleri -> `bist30_footnotes.csv` (donem ici olay ADAYI;
    hakem-2 E #2: #11503 DOHOL '21/06/2017 tarihinden itibaren endekslerden cikarilacaktir').
    Degisikliksiz ceyrek de bir satir ('none') olarak yazilir: ceyregin OKUNDUGUNUN kaydi."""
    import csv
    fetched = _load_fetched()
    qa = quarter_announcements(fetched)
    pairs = build_name_map(fetched)
    rows, names, notes, problems = [], {}, [], []
    for q in sorted(qa):
        anns = sorted(qa[q], key=lambda x: int(x[0]))
        sig = {(tuple(t['ALINACAK']), tuple(t['ÇIKARILACAK'])) for _, t, _, _ in anns}
        if len(sig) > 1:
            problems.append(f'{q}: ikiz duyurular farkli ({[a[0] for a in anns]})')
        did, t, per, ann = anns[0]
        twins = ' '.join(a[0] for a in anns[1:])
        marked = {(k, item): m for k, item, m in t['marked']}
        fn = footnotes(_read(did))
        for (k, item), m in sorted(marked.items()):
            notes.append({'quarter': q, 'column': k, 'item': item, 'marker': m,
                          'footnote': ' | '.join(x for x in fn if x.startswith(m)),
                          'source_id': did, 'source_url': fetched[did]['url'],
                          'source_sha256': fetched[did]['sha256']})
        if marked and not fn:
            problems.append(f'{q}: isaretli oge var ama dipnot metni bulunamadi {sorted(marked)}')
        for action, col in (('add', 'ALINACAK'), ('remove', 'ÇIKARILACAK')):
            for raw in t[col]:
                code = raw
                if t['kind'] != 'code':
                    hit = pairs.get(_norm_name(raw), {})
                    if len(hit) != 1:
                        problems.append(f'{q}: ad eslenemedi {raw!r} -> {sorted(hit)}')
                        continue
                    code, ev = next(iter(hit.items()))
                    names.setdefault(raw, (code, ev, []))[2].append((q, ann))
                rows.append(_row(q, per, ann, action, raw, code, t['kind'], did, fetched, twins,
                                 marked.get((col, raw), '')))
        if not t['ALINACAK'] and not t['ÇIKARILACAK']:
            rows.append(_row(q, per, ann, 'none', '', '', t['kind'], did, fetched, twins))
    if problems:
        sys.exit('parse durdu:\n  ' + '\n  '.join(problems))
    with open(CHANGES, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    with open(NAME_MAP, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['name', 'code', 'name_quarters', 'nearest_evidence_id', 'nearest_evidence_date',
                    'year_gap', 'evidence_unvan', 'n_evidence', 'evidence_ids', 'evidence_url', 'rule'])
        for n in sorted(names):
            c, ev, used = names[n]
            first_use = min(d for _q, d in used)
            near = min(ev, key=lambda x: abs(int(x[2][:4]) - int(first_use[:4])) * 1000 + int(x[0]) % 1000)
            w.writerow([n, c, ' '.join(q for q, _d in used), near[0], near[2],
                        abs(int(near[2][:4]) - int(first_use[:4])), near[1], len(ev),
                        ' '.join(sorted({x[0] for x in ev}, key=int)), fetched[near[0]]['url'],
                        'normalize(unvan) == normalize(ad), tek kod'])
    with open(FOOTNOTES, 'w', encoding='utf-8', newline='') as f:
        cols = ['quarter', 'column', 'item', 'marker', 'footnote', 'source_id', 'source_url', 'source_sha256']
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(notes)
    print(f'{len(qa)} ceyrek, {sum(r["action"] != "none" for r in rows)} degisiklik, '
          f'{len(names)} ad eslemesi, {len(notes)} isaretli BIST 30 ogesi (dipnot adayi)', flush=True)
    print('eksik ceyrek:', sorted({f'{y}Q{q}' for y in range(2013, 2027) for q in range(1, 5)
                                  if (2013, 3) <= (y, q) <= (2026, 4)} - set(qa)), flush=True)


def load_events() -> list:
    """Elle derlenen olaylar (her biri ham duyurudan alintili): date (yururluk ilk gunu),
    action (add|remove|rename), code, new_code (rename), announced, source_url, source_sha256, quote."""
    import csv
    if not os.path.exists(EVENTS):
        return []
    with open(EVENTS, encoding='utf-8', newline='') as f:
        return [r for r in csv.DictReader(f) if r['date'] and not r['date'].startswith('#')]


def chain(anchor: dict, changes: list, events: list):
    """Capadan GERIYE zincir. Olay tarihi = yururluk ilk gunu. Geriye gidiste her olay icin:
    alinan sonraki kumede uye olmali, cikan olmamali; rename'de yeni kod uye, eski kod degil.
    Donus: (segmentler [(bas, son_haric|None, kume, olaylar)], ihlal metinleri, yapili olay ihlalleri)."""
    ev = [{'date': r['period_start'], 'action': r['action'], 'code': r['code'], 'new_code': '',
           'src': r['source_url'], 'sha': r['source_sha256'], 'kind': 'quarterly'}
          for r in changes if r['action'] in ('add', 'remove')]
    ev += [{'date': r['date'], 'action': r['action'], 'code': r['code'], 'new_code': r.get('new_code', ''),
            'src': r['source_url'], 'sha': r['source_sha256'], 'kind': 'event'} for r in events]
    by_date = {}
    for e in ev:
        by_date.setdefault(e['date'], []).append(e)
    first = min(r['period_start'] for r in changes)
    by_date.setdefault(first, [])
    cur, end, segs, bad, viol = set(anchor['codes']), None, [], [], []
    for d in sorted(by_date, reverse=True):
        if d > anchor['as_of'] or d < first:
            continue
        segs.append((d, end, frozenset(cur), by_date[d]))
        if len(cur) != 30:
            bad.append(f'{d}..{end or anchor["as_of"]}: {len(cur)} uye')
        for e in by_date[d]:
            c = e['code']
            if e['action'] == 'add':
                if c not in cur:
                    bad.append(f'{d} alinan {c} sonraki kumede uye degil')
                    viol.append({'date': d, 'code': c, 'msg': bad[-1]})
                cur.discard(c)
            elif e['action'] == 'remove':
                if c in cur:
                    bad.append(f'{d} cikan {c} sonraki kumede hala uye')
                    viol.append({'date': d, 'code': c, 'msg': bad[-1]})
                cur.add(c)
            elif e['action'] == 'rename':
                if e['new_code'] not in cur or c in cur:
                    bad.append(f'{d} kod degisikligi {c}->{e["new_code"]} tutarsiz')
                    viol.append({'date': d, 'code': e['new_code'], 'msg': bad[-1]})
                cur.discard(e['new_code'])
                cur.add(c)
        end = d
    return segs, bad, viol


MEMBERSHIP_META = os.path.join(UNI, 'bist30_membership_meta.json')


def unknown_intervals(changes: list, events: list, viol: list, segs: list, as_of: str = None) -> list:
    """hakem-2 E #1. Bir olay ihlali (d, c) yalniz sunu soyler: c'nin uyeligi d ile c'nin BIR SONRAKI
    kayitli olayi (yoksa capa) arasinda kayitsiz degisti — aralik birden cok segmente yayilabilir
    (DOHOL: 2017-01-02 alinir, 21/06/2017 cikar, 2017-07-03 yeniden alinir). Bilinmeyen aralik
    [d, c'nin sonraki kaydi) -> {'code', 'start', 'end', 'reason'}. Sonraki kayit yoksa end = capa
    gunu (`as_of`, HARIC): capa gunu KAP'tan kesin bilinir (hakem-2 E ikinci okuma #1); as_of yoksa None.
    Uye sayisi 30 olmayan segment de butunuyle bilinmeyen ('*')."""
    rec = {}
    for r in changes:
        if r['action'] in ('add', 'remove'):
            rec.setdefault(r['code'], set()).add(r['period_start'])
    for r in events:
        for c in (r['code'], r.get('new_code', '')):
            if c:
                rec.setdefault(c, set()).add(r['date'])
    out = []
    for v in viol:
        later = sorted(x for x in rec.get(v['code'], ()) if x > v['date'])
        out.append({'code': v['code'], 'start': v['date'], 'end': later[0] if later else as_of,
                    'reason': v['msg']})
    for s, e, k, _ in segs:
        if len(k) != 30:
            out.append({'code': '*', 'start': s, 'end': e, 'reason': f'{len(k)} uye'})
    return sorted(out, key=lambda x: (x['start'], x['code']))


def _overlaps(a, b, s, e) -> bool:
    """[a, b) ile [s, e) kesisiyor mu (None = sonsuz)."""
    return (e is None or a < e) and (b is None or s < b)


def count_checked_from(viol: list, segs: list):
    """SEZGISEL kesim (eski 'consistent_from'; hakem-2 E #1: ONCULU YANLIS): en gec olay ihlalinden
    sonraki ilk segment basi ve sonrasinda uye sayisi 30. Yalniz 'cayreklik-yalniz, sayi denetiminden
    gecen' anlaminda; dogruluk ya da tutarlilik iddiasi tasimaz."""
    if not viol and all(len(k) == 30 for _s, _e, k, _ in segs):
        return min(x[0] for x in segs)
    last = max([v['date'] for v in viol] + [s for s, _e, k, _ in segs if len(k) != 30])
    later = sorted(x[0] for x in segs if x[0] > last)
    return later[0] if later else None


def cmd_chain(provisional: bool = False) -> None:
    """Uyelik: ihlal yoksa KESIN (status final). Ihlal varsa varsayilan olarak yazmaz; `--provisional`
    ile GECICI yazar. Satir basina iki bayrak:
      - `consistent` (KATI): segment 30 uye ve hicbir bilinmeyen araliga (`unknown_intervals`) degmiyor.
      - `count_checked` (SEZGISEL): `count_checked_from` ve sonrasi; yalniz sayi denetiminden gecen.
    Bilinmeyen araliklar meta'da (`bist30_membership_meta.json`); `universe_mask` bunlari GUN duzeyinde uygular."""
    import csv
    with open(ANCHOR, encoding='utf-8') as f:
        anchor = json.load(f)
    with open(CHANGES, encoding='utf-8', newline='') as f:
        changes = list(csv.DictReader(f))
    events = load_events()
    segs, bad, viol = chain(anchor, changes, events)
    if bad and not provisional:
        sys.exit('zincir tutarsiz (membership YAZILMADI; gecici icin --provisional):\n  ' + '\n  '.join(bad))
    unk = unknown_intervals(changes, events, viol, segs, anchor['as_of'])
    cc = count_checked_from(viol, segs)
    with open(MEMBERSHIP, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['start', 'end', 'symbol', 'n_members', 'consistent', 'count_checked', 'start_event',
                    'source_url', 'source_sha256'])
        for s, e, k, evs in sorted(segs, key=lambda x: x[0]):
            strict = int(len(k) == 30 and not any(_overlaps(u['start'], u['end'], s, e) for u in unk))
            heur = int(cc is not None and s >= cc and len(k) == 30)
            for c in sorted(k):
                w.writerow([s, e or '', c, len(k), strict, heur, '+'.join(sorted({x['kind'] for x in evs})),
                            ' '.join(sorted({x['src'] for x in evs})), ' '.join(sorted({x['sha'] for x in evs}))])
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    strict_starts = [s for s, e, k, _ in segs
                     if len(k) == 30 and not any(_overlaps(u['start'], u['end'], s, e) for u in unk)]
    meta = {'status': 'provisional' if bad else 'final', 'anchor_as_of': anchor['as_of'],
            'first_start': min(x[0] for x in segs), 'n_segments': len(segs),
            'n_strict_segments': len(strict_starts), 'count_checked_from': cc,
            'n_quarterly_changes': sum(r['action'] in ('add', 'remove') for r in changes),
            'n_events': len(events), 'violations': bad, 'unknown_intervals': unk,
            'inputs_sha256': {os.path.basename(p): sha(p) for p in (ANCHOR, CHANGES, EVENTS) if os.path.exists(p)},
            'note': ('GECICI: donem ici olaylar (bist30_events.csv) eklenmedi. KATI modda bilinmeyen '
                     'araliklar None doner; count_checked SEZGISELDIR (yalniz sayi denetimi), dogruluk '
                     'kaniti degildir.') if bad else ''}
    with open(MEMBERSHIP_META, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{meta['status']}: {len(segs)} segment, kati {len(strict_starts)} segment, sezgisel "
          f"{cc}..{anchor['as_of']}, {len(bad)} ihlal, {len(unk)} bilinmeyen aralik", flush=True)


EXPOSURE = os.path.join(ROOT, 'docs', 'development', 'measurements', 'phase9-e1', 'exposure.json')


def cmd_exposure() -> None:
    """Hayatta kalma maruziyeti: panelin 30 hissesinin seans basina BIST 30'da olma orani.
    Seanslar RL panelinin kendi islem gunleri (yoksa durur).

    - KATI kapsam (`universe_mask(mode='strict')`): bilinmeyen araliklar disindaki seans sayisi.
    - SEZGISEL oran (`mode='count_checked'`): yalniz sayi denetiminden gecen segmentler; GECICI zincirde
      tek olculebilen oran budur ve dogruluk iddiasi tasimaz (hakem-2 E #1, #3).
    - Duyarlilik: bilinmeyen araliga dusen PANEL hisselerinin (ör. SKBNK) o seanslarda uye oldugu
      varsayimiyla ust sinir. Kayitsiz takaslarin (tespit edilmemis) etkisi sinirlanamaz.
    - Mekanik denetim: panel hissesi, paneldeki ilk islem gununden once uye gorunuyor mu (ham zincir).
    """
    import pandas as pd
    from data.bist30_symbols import BIST30_SYMBOLS, UNIVERSE, load_membership, panel_membership
    panel = os.path.join(ROOT, UNIVERSE['panel']['file'])
    if not os.path.exists(panel):
        sys.exit(f'panel yok: {panel}')
    px = pd.read_csv(panel, usecols=['symbol', 'date'])
    px['date'] = pd.to_datetime(px['date'])
    days = pd.DatetimeIndex(sorted(px['date'].unique()))
    meta = json.load(open(MEMBERSHIP_META, encoding='utf-8'))
    strict = panel_membership(days, mode='strict').dropna(how='all')
    m = panel_membership(days, mode='count_checked')
    covered = m.dropna(how='all').astype(bool)
    per_day = covered.sum(axis=1)
    n = len(covered)
    # duyarlilik: bilinmeyen araliktaki panel hisseleri
    unk_panel = {}
    for u in meta['unknown_intervals']:
        sym = f"{u['code']}.IS"
        if sym in BIST30_SYMBOLS:
            sel = covered.index[(covered.index >= u['start'])
                                & ((covered.index < u['end']) if u['end'] else True)]
            unk_panel[sym] = unk_panel.get(sym, 0) + int((~covered.loc[sel, sym]).sum())
    upper = (float(per_day.sum()) + sum(unk_panel.values())) / (30 * n) if n else None
    # ilk islem gunu denetimi (ham zincir, yalniz panel hisseleri)
    first = px.groupby('symbol')['date'].min()
    raw = panel_membership(days, mode='raw').dropna(how='all').astype(bool)
    listing = {s: str(raw.index[raw[s] & (raw.index < first[s])][0].date())
               for s in BIST30_SYMBOLS if (raw[s] & (raw.index < first[s])).any()}
    segs = load_membership()
    panel_codes = {x.split('.')[0] for x in BIST30_SYMBOLS}
    seg_rows = [{'start': s, 'end': e, 'consistent': ok, 'count_checked': cc, 'n_members': len(k),
                 'n_panel_in_index': len(panel_codes & k), 'panel_not_in_index': sorted(panel_codes - k),
                 'index_not_in_panel': sorted(k - panel_codes)} for s, e, k, ok, cc in segs]
    by_year = {str(y): round(float(g.mean() / 30), 4) for y, g in per_day.groupby(per_day.index.year)}
    doc = {
        'what': 'Panelin 30 hissesinin seans basina BIST 30 uyeligi (hayatta kalma maruziyeti), plan §9.21 E1',
        'membership_file': os.path.relpath(MEMBERSHIP, ROOT).replace('\\', '/'),
        'membership_sha256': hashlib.sha256(open(MEMBERSHIP, 'rb').read()).hexdigest(),
        'membership_status': meta['status'],
        'panel_file': UNIVERSE['panel']['file'], 'panel_sha256': UNIVERSE['panel']['sha256'],
        'sessions_total': int(len(days)),
        'strict_sessions_covered': int(len(strict)),
        'measure': 'count_checked (SEZGISEL: yalniz ceyreklik duyurular, sayi denetiminden gecen segmentler)',
        'note': ('GECICI. Oranlar count_checked modunda; donem ici olaylar (hizli giris, cikarma) dahil '
                 'degil, kayitsiz takaslar olculunce degisebilir. Hisse basina nitel iddia icin kullanilmaz.'),
        'count_checked_from': meta['count_checked_from'],
        'sessions_covered': int(n),
        'covered_range': [str(covered.index[0].date()), str(covered.index[-1].date())] if n else None,
        'mean_panel_in_index': round(float(per_day.mean()), 3),
        'share_panel_in_index': round(float(per_day.mean() / 30), 4),
        'share_upper_if_unknown_panel_members': round(upper, 4) if upper is not None else None,
        'unknown_panel_sessions': unk_panel,
        'min_panel_in_index': {'n': int(per_day.min()), 'first_date': str(per_day.idxmin().date())},
        'max_panel_in_index': {'n': int(per_day.max()), 'first_date': str(per_day.idxmax().date())},
        'share_by_year': by_year,
        'symbol_share_in_index': {s: round(float(covered[s].mean()), 4) for s in BIST30_SYMBOLS},
        'panel_member_before_first_trade': listing,
        'segments': seg_rows,
        'measured_at': str(date.today()),
    }
    os.makedirs(os.path.dirname(EXPOSURE), exist_ok=True)
    with open(EXPOSURE, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: doc[k] for k in ('strict_sessions_covered', 'covered_range', 'mean_panel_in_index',
                                          'share_panel_in_index', 'share_upper_if_unknown_panel_members',
                                          'unknown_panel_sessions', 'min_panel_in_index',
                                          'panel_member_before_first_trade')}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['index', 'probe', 'nodes', 'anchor', 'fetch', 'parse', 'chain', 'exposure'])
    ap.add_argument('--max-pages', type=int, default=200)
    ap.add_argument('--range', type=int, nargs=2, default=(11000, 15700))
    ap.add_argument('--provisional', action='store_true', help='chain: ihlal varsa gecici uyelik yaz')
    a = ap.parse_args()
    if a.cmd == 'index':
        cmd_index(a.max_pages)
    elif a.cmd == 'anchor':
        cmd_anchor()
    elif a.cmd == 'nodes':
        cmd_nodes(*a.range)
    elif a.cmd == 'probe':
        cmd_probe()
    elif a.cmd == 'fetch':
        cmd_fetch()
    elif a.cmd == 'parse':
        cmd_parse()
    elif a.cmd == 'chain':
        cmd_chain(a.provisional)
    else:
        cmd_exposure()


if __name__ == '__main__':
    main()
