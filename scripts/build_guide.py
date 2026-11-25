"""K28 R1: sistem sozlugu ureticisi. Tek kaynak `dashboard/content/glossary.yaml` -> `docs/sozluk.md` (URETILIR).

Plan: docs/development/system-guide-plan.md §2-§3, §6. Gelistirme makinesinde kosulur (imajda `scripts/` yok).
Dogrulama (`validate`, `check_guide`) testle ortak: tests/test_guide.py.

  python scripts/build_guide.py            # dogrula + docs/sozluk.md ve dashboard/content/snapshot.json yaz
  python scripts/build_guide.py --check    # dogrula + uretilmis dosyalar bayat mi (bayatsa cikis 1)

R3: snapshot.json = rehberin deney kartlari (etiket + en fazla 2 sayi, her biri kaynak JSON + alan + SHA + uretim commit'i)
ve asama cizelgesi (system-map.md durum tablosu). Kaynak JSON ya da harita degisince --check bayatligi yakalar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOSSARY = os.path.join(ROOT, 'dashboard', 'content', 'glossary.yaml')
OUT_MD = os.path.join(ROOT, 'docs', 'sozluk.md')
LIT_TABLE = os.path.join(ROOT, 'docs', 'thesis', 'literature-table.md')

CATEGORIES = {   # sira = belgedeki sira
    'sistem': 'Sistem ve aşamalar',
    'deney': 'Deney kodları ve etiketler',
    'degerlendirme': 'Değerlendirme ve istatistik',
    'marl': 'Çok ajanlı öğrenme',
    'rl': 'Tek ajan ve genel pekiştirmeli öğrenme',
    'tahmin': 'Tahmin hattı',
    'veri': 'Veri',
    'yazilim': 'Yazılım ve ürün',
}
REQUIRED = ('id', 'terim', 'kategori', 'sade', 'teknik')
OPTIONAL = ('kisaltma', 'kodda', 'belgede', 'ilgili', 'kaynak', 'benzetme', 'uyari', 'en')
LIST_FIELDS = ('kodda', 'belgede', 'ilgili', 'kaynak')
ID_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
SENTENCE_END = re.compile(r'[.!?](\s|$)')
MAX_SADE_SENTENCES = 2
# Tez kurallari (docs/thesis/CLAUDE.md, kok CLAUDE.md): bu ifadeler sozlukte gecmez. `uyari` alani taranmaz, cunku
# yasakli ifadeyi "denmez" diye anmak icin vardir.
FORBIDDEN = (r'sızıntı\s+yok', r'sizinti\s+yok', r'diferansiyel\s+sharpe', r'literatürde\s+(yok|ilk)',
             r'\bilk\s+(kez|defa|çalışma)', r'anlamlı\s+olarak\s+daha\s+iyi')
SCANNED = ('terim', 'sade', 'teknik', 'benzetme')


def load(path: str = GLOSSARY) -> list:
    with open(path, encoding='utf-8') as fh:
        terms = yaml.safe_load(fh)
    if not isinstance(terms, list):
        raise ValueError(f'{path}: kok bir liste olmali')
    return terms


def _token(abbr: str) -> re.Pattern:
    return re.compile(r'(?<![\w-])' + re.escape(abbr) + r'(?![\w])')


def check_schema(terms: list) -> list:
    """[1] zorunlu alanlar, tipler, id bicimi ve tekilligi, kategori, sade en fazla 2 cumle."""
    err, seen = [], set()
    for i, t in enumerate(terms):
        tid = t.get('id') if isinstance(t, dict) else None
        where = f'#{i} ({tid})'
        if not isinstance(t, dict):
            err.append(f'{where}: terim bir sozluk olmali')
            continue
        for k in REQUIRED:
            if not isinstance(t.get(k), str) or not t[k].strip():
                err.append(f'{where}: zorunlu alan bos ya da metin degil: {k}')
        for k in t:
            if k not in REQUIRED + OPTIONAL:
                err.append(f'{where}: bilinmeyen alan: {k}')
        for k in LIST_FIELDS:
            if k in t and not (isinstance(t[k], list) and all(isinstance(x, str) and x for x in t[k])):
                err.append(f'{where}: {k} bos olmayan metinlerden bir liste olmali')
        if isinstance(tid, str):
            if not ID_RE.match(tid):
                err.append(f'{where}: id bicimi a-z0-9 ve tire olmali')
            if tid in seen:
                err.append(f'{where}: id tekil degil')
            seen.add(tid)
        if t.get('kategori') not in CATEGORIES:
            err.append(f"{where}: kategori {sorted(CATEGORIES)} disinda: {t.get('kategori')}")
        if isinstance(t.get('sade'), str) and len(SENTENCE_END.findall(t['sade'].strip())) > MAX_SADE_SENTENCES:
            err.append(f'{where}: sade tanim en fazla {MAX_SADE_SENTENCES} cumle')
    return err


def check_links(terms: list) -> list:
    """[2] ilgili baglantilari var olan (baska) bir id'ye gidiyor; sade tanimda gecen her sozluk kisaltmasi ilgili'de."""
    err, ids = [], {t['id'] for t in terms}
    abbr = [(t['id'], t['kisaltma']) for t in terms if t.get('kisaltma')]
    for t in terms:
        rel = set(t.get('ilgili', []))
        for r in rel - ids:
            err.append(f"{t['id']}: ilgili '{r}' sozlukte yok")
        if t['id'] in rel:
            err.append(f"{t['id']}: ilgili kendisini gosteriyor")
        for oid, a in abbr:
            if oid != t['id'] and _token(a).search(t['sade']) and oid not in rel:
                err.append(f"{t['id']}: sade tanimda '{a}' geciyor ama '{oid}' ilgili'de degil")
    return err


def check_paths(terms: list, root: str = ROOT) -> list:
    """[3] kodda ve belgede yollari depoda var; `::ad` verilmisse ad dosyada tanimli bir ad olarak geciyor."""
    err = []
    for t in terms:
        for ref in t.get('kodda', []) + t.get('belgede', []):
            path, _, name = ref.partition('::')
            full = os.path.join(root, *path.split('/'))
            if not os.path.isfile(full):
                err.append(f"{t['id']}: yol yok: {path}")
                continue
            if name:
                with open(full, encoding='utf-8') as fh:
                    src = fh.read()
                if not re.search(r'^\s*(def|class)\s+' + re.escape(name) + r'\b|^\s*' + re.escape(name) + r'\s*[:=]',
                                 src, re.M):
                    err.append(f"{t['id']}: {path} icinde '{name}' tanimi yok")
    return err


def check_sources(terms: list, lit_path: str = LIT_TABLE) -> list:
    """[4] kaynak dizeleri literature-table.md'de birebir geciyor (bellekten kaynak yazilmaz)."""
    with open(lit_path, encoding='utf-8') as fh:
        lit = fh.read()
    return [f"{t['id']}: kaynak literature-table.md'de yok: {s}" for t in terms for s in t.get('kaynak', [])
            if s not in lit]


def check_forbidden(terms: list) -> list:
    """[5] yasakli ifadeler (tez kurallari); `uyari` alani haric."""
    err = []
    for t in terms:
        for k in SCANNED:
            for pat in FORBIDDEN:
                if isinstance(t.get(k), str) and re.search(pat, t[k], re.I):
                    err.append(f"{t['id']}.{k}: yasakli ifade /{pat}/")
    return err


def validate(terms: list, root: str = ROOT) -> list:
    err = check_schema(terms)
    if err:
        return err
    return check_links(terms) + check_paths(terms, root) + check_sources(terms) + check_forbidden(terms)


def _md_path(path: str) -> str:
    """Depo kokune gore yol -> docs/sozluk.md'den goreli baglanti."""
    return path[len('docs/'):] if path.startswith('docs/') else '../' + path


def render(terms: list) -> str:
    """Deterministik Markdown (ayni girdi -> ayni bayt). Elle duzenlenmez."""
    names = {t['id']: t['terim'] for t in terms}
    by_cat = {c: [t for t in terms if t['kategori'] == c] for c in CATEGORIES}
    out = ['<!-- URETILDI: scripts/build_guide.py <- dashboard/content/glossary.yaml. ELLE DUZENLEME: YAML\'i '
           'degistirip `python scripts/build_guide.py` kos. -->', '',
           '# Sistem Sözlüğü', '',
           'Projede geçen aşama adları, deney kodları, etiketler ve yöntem terimleri. Her terimde önce **sade** tanım '
           '(konuyu bilmeyen biri için), sonra **teknik** tanım (koddaki karşılığıyla) gelir. Sözlük deney sonucu '
           'içermez; sonuçlar ilgili tasarım belgelerindedir.', '',
           f'**Terim sayısı:** {len(terms)} · **Tek kaynak:** [`dashboard/content/glossary.yaml`]'
           '(../dashboard/content/glossary.yaml) · **Plan:** [system-guide-plan.md](development/system-guide-plan.md)', '',
           '## İçindekiler', '']
    for c, title in CATEGORIES.items():
        if by_cat[c]:
            links = ', '.join(f"[{t['terim']}](#{t['id']})" for t in by_cat[c])
            out.append(f'- **[{title}](#kat-{c})** — {links}')
    for c, title in CATEGORIES.items():
        if not by_cat[c]:
            continue
        out += ['', f'<a id="kat-{c}"></a>', '', f'## {title}']
        for t in by_cat[c]:
            head = t['terim'] + (f" · `{t['kisaltma']}`" if t.get('kisaltma') and t['kisaltma'] not in t['terim'] else '')
            out += ['', f'<a id="{t["id"]}"></a>', '', f'### {head}', '', f"**Sade:** {t['sade'].strip()}"]
            if t.get('benzetme'):
                out += ['', f"> *Benzetme:* {t['benzetme'].strip()}"]
            out += ['', f"**Teknik:** {t['teknik'].strip()}"]
            meta = []
            if t.get('kodda'):
                meta.append('Kodda: ' + ', '.join(f"[`{r}`]({_md_path(r.partition('::')[0])})" for r in t['kodda']))
            if t.get('belgede'):
                meta.append('Belgede: ' + ', '.join(f'[{os.path.basename(r)}]({_md_path(r)})' for r in t['belgede']))
            if t.get('ilgili'):
                meta.append('İlgili: ' + ', '.join(f'[{names[r]}](#{r})' for r in t['ilgili']))
            if t.get('kaynak'):
                meta.append('Kaynak: ' + '; '.join(t['kaynak']))
            if meta:
                out += [''] + [f'- {m}' for m in meta]
            if t.get('uyari'):
                out += ['', f"**Uyarı:** {t['uyari'].strip()}"]
    return '\n'.join(out) + '\n'


# ------------------------------------------------------------------ R3: rehber anlik goruntusu (snapshot.json)
GUIDE = os.path.join(ROOT, 'dashboard', 'content', 'guide.yaml')
SNAPSHOT = os.path.join(ROOT, 'dashboard', 'content', 'snapshot.json')
GUIDE_SECTIONS = ('ne_yapar', 'akis', 'neredeyiz', 'deneyler', 'ne_degildir', 'sss')
LABELS = {'GECTI': 'GEÇTİ', 'KISMI': 'KISMİ', 'GOZLENMEDI': 'GÖZLENMEDİ', 'PILOT_ISARETI': 'PİLOT_İŞARETİ',
          # M8 on kayitli etiketleri (phase-10-m8-design.md §6.5); etiket adinda kolun adi gecer
          'FARK_VAR_MARL': 'FARK VAR: MARL', 'FARK_VAR_M-none': 'FARK VAR: M-none', 'FARK_GOZLENMEDI': 'FARK GÖZLENMEDİ'}
MAX_CARD_NUMBERS = 2          # R-K4
BADGE = 'Tez bulgusu değil'   # R-K4: gelistirme kapisi kartlarinda zorunlu
# Faz 11 §3.14: on kayitli nihai deney (M8) kartlari "tez bulgusu degil" tasiyamaz; rozetleri ayri ve yine zorunlu.
BADGES = {'gelistirme': BADGE, 'nihai': 'Ön kayıtlı nihai deney'}
# Ana sayfanin "one cikan model" ozeti (Faz 11 §3.14): pano olcum dosyasi acmaz, sayilar buradan kaynakli gelir.
M8_READING = 'docs/development/measurements/phase10-m8/m8_reading.json'
M8_ARMS = ('M-none', 'null', 'M-self', 'M-gat', 'S-agr', 'S-esm')
M8_RULES = ('minvar', 'hrp', 'bh', 'ew')
M8_COST_BPS = 20              # on kayitli maliyet (gidis-donus), M8 §6
DONE_WORDS = ('bitti', 'donduruldu', 'kapandı', 'kapandi')


def load_guide(path: str = GUIDE) -> dict:
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def field(obj, dotted: str):
    """'a.b.c' alan yolu; yoksa KeyError."""
    for part in dotted.split('.'):
        if not isinstance(obj, dict) or part not in obj:
            raise KeyError(dotted)
        obj = obj[part]
    return obj


def fmt_number(v: float, ondalik: int, isaret: bool = False) -> str:
    """Turkce ondalik virgul, gercek eksi isareti; isaret=True ise + da yazilir."""
    s = f'{v:+.{ondalik}f}' if isaret else f'{v:.{ondalik}f}'
    return s.replace('-', '−').replace('.', ',')


def _plain(md: str) -> str:
    md = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', md)
    return re.sub(r'\s+', ' ', md.replace('**', '').replace('`', '')).strip()


def status_table_text(md: str) -> str:
    """'## Durum tablosu' bolumunun metni (bir sonraki '## ' basligina kadar)."""
    i = md.find('## Durum tablosu')
    if i < 0:
        raise ValueError("system-map.md: '## Durum tablosu' yok")
    j = md.find('\n## ', i + 1)
    return md[i:] if j < 0 else md[i:j]


def parse_status_table(md: str) -> list:
    """system-map.md '## Durum tablosu' -> [{ad, durum, sinif, siradaki}]; durum = kalin bas ya da ilk ':' / ';' / '('
    oncesi. sinif: 'suruyor' | 'bitti' | 'planli' (anahtar sozcuk). Tablo yoksa ValueError."""
    i = md.find('## Durum tablosu')
    if i < 0:
        raise ValueError("system-map.md: '## Durum tablosu' yok")
    rows = []
    for line in md[i:].splitlines()[1:]:
        if line.startswith('## '):
            break
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 4 or set(cells[0]) <= set('-: ') or cells[0].startswith('Aşama'):
            continue
        name, status, nxt = cells[0], cells[1], cells[3]
        m = re.match(r'\*\*(.+?)\*\*', status)
        head = _plain(m.group(1) if m else re.split(r'[:;(]', status, maxsplit=1)[0]).strip(' ,')
        low = head.lower()
        cls = 'suruyor' if 'sürüyor' in low else ('bitti' if any(w in low for w in DONE_WORDS) else 'planli')
        rows.append({'ad': _plain(name), 'durum': head, 'sinif': cls,
                     'siradaki': None if _plain(nxt) in ('', '—', '-') else _plain(nxt)})
    return rows


def _resolve(spec: dict, root: str, cache: dict):
    rel = spec['kaynak']
    if rel not in cache:
        with open(os.path.join(root, *rel.split('/')), encoding='utf-8') as fh:
            cache[rel] = json.load(fh)
    return field(cache[rel], spec['alan']), cache[rel]


def check_guide(guide: dict, terms: list, root: str = ROOT) -> list:
    """Rehber metni: bolumler var, terim baglantilari sozlukte, kartlar (en fazla 2 sayi, her sayi ve etiket kaynakta
    cozuluyor), belge yollari var, yasakli ifade yok."""
    err, ids, cache = [], {t['id'] for t in terms}, {}
    for s in GUIDE_SECTIONS:
        if not isinstance(guide.get(s), dict):
            err.append(f'guide.yaml: bolum yok: {s}')
    if err:
        return err
    refs = list(guide['ne_yapar'].get('terimler', [])) + [a.get('terim') for a in guide['akis']['adimlar']]
    refs += [m.get('terim') for m in guide['ne_degildir']['maddeler']]
    refs += [t for q in guide['sss']['sorular'] for t in q.get('terimler', [])]
    texts = [guide['ne_yapar']['baslik'], *guide['ne_yapar']['paragraflar'], guide['akis']['aciklama'],
             guide['ne_degildir']['aciklama'], guide['deneyler']['aciklama'], guide['neredeyiz']['aciklama']]
    texts += [x for a in guide['akis']['adimlar'] for x in (a['ad'], a['metin'])]
    texts += [x for m in guide['ne_degildir']['maddeler'] for x in (m['baslik'], m['metin'])]
    texts += [x for q in guide['sss']['sorular'] for x in (q['soru'], q['cevap'])]
    if not os.path.isfile(os.path.join(root, *guide['neredeyiz']['kaynak'].split('/'))):
        err.append(f"neredeyiz: kaynak yok: {guide['neredeyiz']['kaynak']}")
    seen = set()
    for c in guide['deneyler']['kartlar']:
        cid = c.get('id')
        if cid in seen:
            err.append(f'kart id tekil degil: {cid}')
        seen.add(cid)
        for k in ('baslik', 'soru', 'okuma', 'belge', 'etiket', 'sayilar'):
            if not c.get(k):
                err.append(f'kart {cid}: alan bos: {k}')
        if err:
            continue
        refs += c.get('terimler', [])
        texts += [c['baslik'], c['soru'], c['okuma']] + [n['ad'] for n in c['sayilar']]
        if not os.path.isfile(os.path.join(root, *c['belge'].split('/'))):
            err.append(f"kart {cid}: belge yok: {c['belge']}")
        if c.get('kademe', 'gelistirme') not in BADGES:
            err.append(f"kart {cid}: kademe {c.get('kademe')!r} bilinmiyor {sorted(BADGES)}")
        if not 1 <= len(c['sayilar']) <= MAX_CARD_NUMBERS:
            err.append(f'kart {cid}: 1-{MAX_CARD_NUMBERS} sayi olmali')
        for spec in [c['etiket'], *c['sayilar']]:
            try:
                v, _ = _resolve(spec, root, cache)
            except (KeyError, OSError, ValueError) as e:
                err.append(f"kart {cid}: kaynak/alan cozulmedi: {spec.get('kaynak')} :: {spec.get('alan')} ({e})")
                continue
            if spec is c['etiket']:
                if 'esit' not in spec and v not in LABELS:
                    err.append(f'kart {cid}: etiket {v!r} bilinen etiketlerden degil {sorted(LABELS)}')
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                err.append(f"kart {cid}: sayi degil: {spec['alan']} = {v!r}")
    err += [f'guide.yaml: terim sozlukte yok: {r}' for r in refs if r and r not in ids]
    err += [f'guide.yaml: yasakli ifade /{p}/' for x in texts for p in FORBIDDEN if re.search(p, str(x), re.I)]
    return err


def _source_rec(rel: str, js, root: str) -> dict:
    with open(os.path.join(root, *rel.split('/')), 'rb') as fh:
        # satir sonu normallesir: git autocrlf Windows'ta CRLF, Linux'ta LF verir; ozet platformdan bagimsiz kalmali
        sha = hashlib.sha256(fh.read().replace(b'\r\n', b'\n')).hexdigest()
    commit = None
    if isinstance(js, dict):
        commit = (js.get('provenance') or {}).get('commit') if isinstance(js.get('provenance'), dict) else None
        commit = commit or (js.get('commit') if isinstance(js.get('commit'), str) else None)
    return {'sha256': sha, 'uretim_commit': commit}


def build_snapshot(guide: dict, root: str = ROOT) -> dict:
    """Deterministik (zaman damgasi yok): ayni kaynaklar -> ayni bayt. Commit = olcumu ureten kodun JSON'a yazdigi commit
    (provenance); dosyanin son git commit'i KULLANILMAZ, cunku kaynak ve snapshot ayni commit'e girince kendini bayatlatir."""
    cache, sources, cards = {}, {}, []
    for c in guide['deneyler']['kartlar']:
        ev, js = _resolve(c['etiket'], root, cache)
        e = c['etiket']
        label = (e['evet'] if ev == e['esit'] else e['hayir']) if 'esit' in e else LABELS[ev]
        sources.setdefault(e['kaynak'], _source_rec(e['kaynak'], js, root))
        nums = []
        for n in c['sayilar']:
            v, js = _resolve(n, root, cache)
            sources.setdefault(n['kaynak'], _source_rec(n['kaynak'], js, root))
            nums.append({'ad': n['ad'], 'deger': fmt_number(float(v), int(n.get('ondalik', 2)), bool(n.get('isaret'))),
                         'ham': float(v), 'kaynak': n['kaynak'], 'alan': n['alan']})
        cards.append({'id': c['id'], 'baslik': c['baslik'], 'soru': c['soru'].strip(), 'okuma': c['okuma'].strip(),
                      'etiket': label, 'etiket_kaynak': {'kaynak': e['kaynak'], 'alan': e['alan'], 'ham': ev},
                      'sayilar': nums, 'belge': c['belge'], 'terimler': c.get('terimler', []),
                      'kademe': c.get('kademe', 'gelistirme'), 'rozet': BADGES[c.get('kademe', 'gelistirme')]})
    rel = guide['neredeyiz']['kaynak']
    with open(os.path.join(root, *rel.split('/')), encoding='utf-8') as fh:
        table = status_table_text(fh.read())   # metin kipi CRLF'yi zaten '\n' yapar
    stages = parse_status_table(table)
    # Yalniz durum tablosunun ozeti: haritanin baska satiri (or. 'Son guncelleme') rehberi bayatlatmaz.
    sources[rel] = {'tablo_sha256': hashlib.sha256(table.encode('utf-8')).hexdigest(), 'uretim_commit': None}
    m8, js = build_m8_summary(root, cache)
    sources.setdefault(M8_READING, _source_rec(M8_READING, js, root))
    return {'not': 'URETILDI: scripts/build_guide.py <- dashboard/content/guide.yaml + kaynaklar. ELLE DUZENLEME.',
            'kartlar': cards, 'asamalar': stages, 'm8_ozet': m8, 'kaynaklar': dict(sorted(sources.items()))}


def build_m8_summary(root: str = ROOT, cache: dict | None = None) -> tuple[dict, dict]:
    """Ana sayfa ozeti: M8 okumasindan ogrenen kollarin 72 hucre ortalama Sharpe'i (+- std) ve kural portfoylerinin 6 kat
    ortalamasi (ayni muhasebe, 20 bps; M8 §7.3(b) ile ayni tanim). Siralama YOK: M8 en iyiyi secmez (§6.5); sayfa
    ogrenen kollarin en yuksegini kurallarin yaninda gosterir. Her alan kaynak JSON'daki yoluyla birlikte yazilir."""
    cache = {} if cache is None else cache
    arms_raw, js = _resolve({'kaynak': M8_READING, 'alan': 'descriptive.arms'}, root, cache)
    base = field(js, 'baselines')
    folds = sorted(base)
    arms = [{'kol': a, 'ort': float(arms_raw[a]['mean_sr']), 'std': float(arms_raw[a]['std_sr']),
             'alan': f'descriptive.arms.{a}.mean_sr'} for a in M8_ARMS]
    rules = [{'kural': r, 'ort': float(sum(base[f][r]['sr'] for f in folds) / len(folds)),
              'alan': f'baselines.<kat>.{r}.sr (6 kat ortalamasi)'} for r in M8_RULES]
    labels = {h: field(js, f'contrasts.{h}.label_full') for h in ('H1', 'H2', 'H3', 'NK')}
    return {'kaynak': M8_READING, 'katlar': folds, 'hucre_kol_basi': int(field(js, 'contrasts.H1.n_pairs')),
            'maliyet_bps': M8_COST_BPS, 'kollar': arms, 'kurallar': rules, 'etiketler': labels,
            'on_kayit': field(js, 'prereg')}, js


def snapshot_text(snap: dict) -> str:
    return json.dumps(snap, indent=1, ensure_ascii=False) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='yazmadan denetle: dogrulama + uretilmis dosyalar bayat mi')
    a = ap.parse_args()
    terms, guide = load(), load_guide()
    err = validate(terms) or check_guide(guide, terms)
    if err:
        print('\n'.join(f'HATA: {e}' for e in err))
        sys.exit(1)
    outputs = {OUT_MD: render(terms), SNAPSHOT: snapshot_text(build_snapshot(guide))}
    if a.check:
        stale = []
        for path, text in outputs.items():
            cur = open(path, encoding='utf-8').read() if os.path.exists(path) else None
            if cur != text:
                stale.append(os.path.relpath(path, ROOT))
        if stale:
            print(f'BAYAT: {", ".join(stale)} uretilmis hal ile ayni degil; `python scripts/build_guide.py` kos')
            sys.exit(1)
        print(f'tamam: {len(terms)} terim, {len(guide["deneyler"]["kartlar"])} kart; uretilmis dosyalar guncel')
        return
    for path, text in outputs.items():
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(text)
        print(f'yazildi: {os.path.relpath(path, ROOT)}')


if __name__ == '__main__':
    main()
