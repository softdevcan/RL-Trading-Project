"""
K28 R1: sistem sozlugu (`dashboard/content/glossary.yaml` -> `docs/sozluk.md`, `scripts/build_guide.py`). Ana venv, ag yok.
Plan: docs/development/system-guide-plan.md §6. Her kontrol gercek sozlugun gectigini VE bozulmus bir kopyanin
yakalandigini (mutasyon) birlikte sinar.

Kontroller:
  1. Sema: zorunlu alanlar, id bicimi + tekil, kategori listede, sade en fazla 2 cumle (mutasyon: cift id, bos sade).
  2. Baglantilar: ilgili var olan id'ye gidiyor; sade tanimda gecen sozluk kisaltmasi ilgili'de (mutasyon: baglantisiz
     kisaltma, olmayan id).
  3. Yollar: kodda/belgede dosyalari var; `::ad` dosyada tanimli (mutasyon: olmayan ad, olmayan dosya).
  4. Kaynaklar: her kaynak dizesi literature-table.md'de birebir (mutasyon: uydurma kaynak).
  5. Yasakli ifadeler (tez kurallari): "sizinti yok", "diferansiyel Sharpe", ozgunluk iddialari; `uyari` alani taranmaz
     (mutasyon: sade'ye yasakli ifade eklenince yakalanir).
  6. Tazelik: docs/sozluk.md ve dashboard/content/snapshot.json ureticinin ciktisiyla bit-es ve `build_guide.py --check`
     0 doner (mutasyon: degisen terim baska bir cikti uretir).
  7. Pano sayfasi (R2): `guide.layout()` hatasiz kurulur, uc bolum + sozluk; arama 'sharpe' -> Sharpe karti, 'zzz' ->
     bos durum, Turkce harfe duyarsiz ('ogrenme' ~ 'öğrenme'), kategori suzgeci; rehber metnindeki terim baglantilari
     sozlukte var ve yasakli ifade yok (mutasyon); menu 'Analiz' grubunda ve ROUTE_INDEX'te; kategori listesi uretici ile ayni.
  8. Deney kartlari ve asama cizelgesi (R3): her kartta 1-2 sayi, her sayi kaynak JSON'daki alanin degeriyle ayni, etiket
     kaynaktan, 'Tez bulgusu degil' rozeti; check_guide gercek rehberde temiz (mutasyon: cozulmeyen alan, 3 sayili kart);
     kaynak JSON degisince ya da system-map satiri degisince snapshot degisir (bayatlik yakalanir).

    python tests/test_guide.py
"""
import copy
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import build_guide as BG  # noqa: E402

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


terms = BG.load()


def mutated(fn):
    t = copy.deepcopy(terms)
    fn(t)
    return t


print(f"\nSozluk: {len(terms)} terim, kategoriler {sorted({t['kategori'] for t in terms})}")

# ---------------------------------------------------------------------------------------------
print("\n[1] Sema")
e0 = BG.check_schema(terms)
e_dup = BG.check_schema(mutated(lambda t: t.append(dict(t[0]))))
e_empty = BG.check_schema(mutated(lambda t: t[1].update(sade='  ')))
e_long = BG.check_schema(mutated(lambda t: t[2].update(sade='Bir. Iki. Uc.')))
check("sema gecerli; cift id, bos sade ve 3 cumlelik sade yakalanir",
      e0 == [] and any('tekil' in x for x in e_dup) and any('sade' in x for x in e_empty)
      and any('cumle' in x for x in e_long), f"{e0[:3]}")

# ---------------------------------------------------------------------------------------------
print("\n[2] Baglantilar")
e0 = BG.check_links(terms)
e_abbr = BG.check_links(mutated(lambda t: next(x for x in t if x['id'] == 'sharpe').update(
    sade='Belirsizligi HAC ile olculen bir oran.')))
e_dead = BG.check_links(mutated(lambda t: t[0].setdefault('ilgili', []).append('yok-boyle-terim')))
check("ilgili baglantilari gecerli; baglantisiz kisaltma (HAC) ve olmayan id yakalanir",
      e0 == [] and any("'HAC'" in x for x in e_abbr) and any('yok-boyle-terim' in x for x in e_dead), f"{e0[:3]}")

# ---------------------------------------------------------------------------------------------
print("\n[3] Yollar")
e0 = BG.check_paths(terms)
e_sym = BG.check_paths(mutated(lambda t: next(x for x in t if x['id'] == 'sharpe').update(
    kodda=['scripts/marl/analyze_m3.py::olmayan_fonksiyon'])))
e_file = BG.check_paths(mutated(lambda t: t[0].update(belgede=['docs/olmayan.md'])))
n_refs = sum(len(t.get('kodda', [])) + len(t.get('belgede', [])) for t in terms)
check(f"{n_refs} kod/belge yolu var ve adlar tanimli; olmayan ad ve dosya yakalanir",
      e0 == [] and any('olmayan_fonksiyon' in x for x in e_sym) and any('olmayan.md' in x for x in e_file), f"{e0[:3]}")

# ---------------------------------------------------------------------------------------------
print("\n[4] Kaynaklar")
e0 = BG.check_sources(terms)
e_fake = BG.check_sources(mutated(lambda t: t[0].update(kaynak=['Kimse vd. (2099)'])))
n_src = sum(len(t.get('kaynak', [])) for t in terms)
check(f"{n_src} kaynak dizesi literature-table.md'de; uydurma kaynak yakalanir",
      e0 == [] and n_src > 0 and any('2099' in x for x in e_fake), f"{e0[:3]}")

# ---------------------------------------------------------------------------------------------
print("\n[5] Yasakli ifadeler")
e0 = BG.check_forbidden(terms)
uyari_has_forbidden = any('sızıntı yok' in (t.get('uyari') or '').lower() for t in terms)
e_inj = [BG.check_forbidden(mutated(lambda t, p=p: t[3].update(sade=t[3]['sade'] + ' ' + p)))
         for p in ('Bu hatta sızıntı yok.', 'Odul diferansiyel Sharpe kullanir.', 'Literatürde ilk kez yapildi.')]
check("yasakli ifade yok (uyari haric, orada 'denmez' diye anilabiliyor); eklenen uc ifade yakalanir",
      e0 == [] and uyari_has_forbidden and all(e_inj), f"{e0[:3]}")

# ---------------------------------------------------------------------------------------------
print("\n[6] Tazelik")
with open(BG.OUT_MD, encoding='utf-8') as fh:
    on_disk = fh.read()
r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'build_guide.py'), '--check'], capture_output=True,
                   text=True, encoding='utf-8')
changed = BG.render(mutated(lambda t: t[0].update(teknik=t[0]['teknik'] + ' (degisti)')))
with open(BG.SNAPSHOT, encoding='utf-8') as fh:
    snap_disk = fh.read()
check("docs/sozluk.md ve snapshot.json uretilmis hal ile bit-es, --check 0 doner; degisen terim farkli cikti uretir",
      BG.render(terms) == on_disk and BG.snapshot_text(BG.build_snapshot(BG.load_guide())) == snap_disk
      and r.returncode == 0 and changed != on_disk, r.stdout + r.stderr)

# ---------------------------------------------------------------------------------------------
print("\n[7] Pano sayfasi (R2)")
from dashboard.components.sidebar import NAV_GROUPS  # noqa: E402
from dashboard.components.topbar import ROUTE_INDEX  # noqa: E402
from dashboard.pages import guide as GP  # noqa: E402


def ids(node, acc=None):
    acc = [] if acc is None else acc
    if isinstance(getattr(node, 'id', None), str):
        acc.append(node.id)
    kids = getattr(node, 'children', None)
    for k in (kids if isinstance(kids, (list, tuple)) else [kids] if kids is not None else []):
        if hasattr(k, 'children') or hasattr(k, 'id'):
            ids(k, acc)
    return acc


lay_ids = ids(GP.layout())
gtext, gterms = GP.load_content()
names = {t['id']: t['terim'] for t in gterms}
hit = [t['id'] for t in GP.filter_terms(gterms, 'sharpe', GP.ALL)]
empty = GP.term_list(GP.filter_terms(gterms, 'zzz', GP.ALL), names)
folded = [t['id'] for t in GP.filter_terms(gterms, 'ogrenme', GP.ALL)]
marl_only = GP.filter_terms(gterms, '', 'marl')
check("layout kurulur: 6 bolum + her deney ve terim karti; arama sharpe -> Sharpe, zzz -> bos durum, Turkce harfe duyarsiz, "
      "kategori suzgeci",
      all(f'rehber-{a}' in lay_ids for a in ('ne-yapar', 'akis', 'neredeyiz', 'deneyler', 'ne-degildir', 'sozluk'))
      and all(f"deney-{c['id']}" in lay_ids for c in gtext['deneyler']['kartlar'])
      and all(f"terim-{t['id']}" in lay_ids for t in gterms) and 'sharpe' in hit
      and 'state-block' in (getattr(empty, 'className', '') or '') and 'marl' in folded
      and bool(marl_only) and all(t['kategori'] == 'marl' for t in marl_only), f"{hit[:3]} {folded[:3]}")


def guide_refs(g):
    refs = list(g['ne_yapar'].get('terimler', []))
    refs += [a['terim'] for a in g['akis']['adimlar'] if a.get('terim')]
    refs += [m['terim'] for m in g['ne_degildir']['maddeler'] if m.get('terim')]
    return refs


def guide_texts(g):
    out = [g['ne_yapar']['baslik'], *g['ne_yapar']['paragraflar'], g['akis']['baslik'], g['akis']['aciklama'],
           g['ne_degildir']['baslik'], g['ne_degildir']['aciklama']]
    out += [x for a in g['akis']['adimlar'] for x in (a['ad'], a['metin'])]
    out += [x for m in g['ne_degildir']['maddeler'] for x in (m['baslik'], m['metin'])]
    return out


def forbidden(texts):
    import re
    return [p for x in texts for p in BG.FORBIDDEN if re.search(p, x, re.I)]


bad_refs = [r for r in guide_refs(gtext) if r not in names]
bad_txt = forbidden(guide_texts(gtext))
mut = copy.deepcopy(gtext)
mut['ne_degildir']['maddeler'][0]['metin'] += ' Bu testte sızıntı yok.'
analiz = dict(NAV_GROUPS).get('Analiz', [])
check("rehber metni: terim baglantilari sozlukte, yasakli ifade yok (eklenince yakalanir); menu Analiz'de, "
      "ROUTE_INDEX'te; kategori listesi uretici ile ayni",
      not bad_refs and not bad_txt and bool(forbidden(guide_texts(mut)))
      and any(i['href'] == '/dash/guide' for i in analiz) and ROUTE_INDEX.get('/dash/guide') == ('Analiz', 'Rehber')
      and GP.CATEGORIES == BG.CATEGORIES, f"{bad_refs} {bad_txt}")

# ---------------------------------------------------------------------------------------------
print("\n[8] Deney kartlari ve asama cizelgesi (R3)")
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

guide_y = BG.load_guide()
snap = BG.build_snapshot(guide_y)
ok_cards = True
for c in snap['kartlar']:
    ok_cards &= 1 <= len(c['sayilar']) <= BG.MAX_CARD_NUMBERS and c['rozet'] == BG.BADGE
    for n in c['sayilar']:
        src = json.load(open(os.path.join(ROOT, *n['kaynak'].split('/')), encoding='utf-8'))
        ok_cards &= bool(n['kaynak'] and n['alan'] and n['deger']) and float(BG.field(src, n['alan'])) == n['ham']
    lab_src = json.load(open(os.path.join(ROOT, *c['etiket_kaynak']['kaynak'].split('/')), encoding='utf-8'))
    ok_cards &= BG.field(lab_src, c['etiket_kaynak']['alan']) == c['etiket_kaynak']['ham']
e_clean = BG.check_guide(guide_y, terms)
g_bad = copy.deepcopy(guide_y)
g_bad['deneyler']['kartlar'][1]['sayilar'][0]['alan'] = 'train.olmayan_alan'
g_three = copy.deepcopy(guide_y)
g_three['deneyler']['kartlar'][1]['sayilar'] *= 2
e_bad, e_three = BG.check_guide(g_bad, terms), BG.check_guide(g_three, terms)

tmp = tempfile.mkdtemp(prefix='guide_r3_')
for rel in list(snap['kaynaklar']):
    dst = os.path.join(tmp, *rel.split('/'))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, *rel.split('/')), dst)
same_copy = BG.build_snapshot(guide_y, tmp) == snap
c0 = guide_y['deneyler']['kartlar'][1]['sayilar'][0]
jp = os.path.join(tmp, *c0['kaynak'].split('/'))
js = json.load(open(jp, encoding='utf-8'))
node = js
for part in c0['alan'].split('.')[:-1]:
    node = node[part]
node[c0['alan'].split('.')[-1]] = 9.99
json.dump(js, open(jp, 'w', encoding='utf-8'))
json_changes = BG.build_snapshot(guide_y, tmp) != snap
mp = os.path.join(tmp, *guide_y['neredeyiz']['kaynak'].split('/'))
md = open(mp, encoding='utf-8').read()
open(mp, 'w', encoding='utf-8').write(md.replace('| Faz 1–3, 6–8 | bitti |', '| Faz 1–3, 6–8 | sürüyor |', 1))
stages_change = [a['sinif'] for a in BG.parse_status_table(open(mp, encoding='utf-8').read())][0] == 'suruyor'
check(f"{len(snap['kartlar'])} kart: 1-2 sayi, her sayi kaynak alanina esit, etiket kaynaktan, rozet var; check_guide temiz; "
      "cozulmeyen alan ve 3 sayili kart yakalanir; kaynak JSON ya da harita degisince snapshot degisir",
      ok_cards and e_clean == [] and any('olmayan_alan' in x for x in e_bad) and any('1-2 sayi' in x for x in e_three)
      and same_copy and json_changes and stages_change and len(snap['asamalar']) > 0, f"{e_clean[:3]}")

print("\n" + "=" * 60)
print(f"Sonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
