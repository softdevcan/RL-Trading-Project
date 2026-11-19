"""K28 R1: sistem sozlugu ureticisi. Tek kaynak `dashboard/content/glossary.yaml` -> `docs/sozluk.md` (URETILIR).

Plan: docs/development/system-guide-plan.md §2-§3, §6. Gelistirme makinesinde kosulur (imajda `scripts/` yok).
Dogrulama (`validate`) testle ortak: tests/test_guide.py. R3'te snapshot.json (deney kartlari) buraya eklenecek.

  python scripts/build_guide.py            # dogrula + docs/sozluk.md yaz
  python scripts/build_guide.py --check    # dogrula + uretilmis dosya bayat mi (bayatsa cikis 1)
"""
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='yazmadan denetle: dogrulama + uretilmis dosya bayat mi')
    a = ap.parse_args()
    terms = load()
    err = validate(terms)
    if err:
        print('\n'.join(f'HATA: {e}' for e in err))
        sys.exit(1)
    md = render(terms)
    if a.check:
        cur = open(OUT_MD, encoding='utf-8').read() if os.path.exists(OUT_MD) else None
        if cur != md:
            print(f'BAYAT: {os.path.relpath(OUT_MD, ROOT)} uretilmis hal ile ayni degil; `python scripts/build_guide.py` kos')
            sys.exit(1)
        print(f'tamam: {len(terms)} terim, {os.path.relpath(OUT_MD, ROOT)} guncel')
        return
    with open(OUT_MD, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(md)
    print(f'yazildi: {os.path.relpath(OUT_MD, ROOT)} ({len(terms)} terim)')


if __name__ == '__main__':
    main()
