"""Faz 9 E — sembol evreni (E0, Y11/K7) ve sektor siniflamasi (E2), tarihsel uyelik (E1).

    python tests/test_universe.py

E0: evrenin tek kaynagi `data/universe/bist30_panel.json`; kod ve pano listeleri ondan okur ve RL
panelinin sembolleriyle (gorunme SIRASIYLA) ayni. Ag yok.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

FAILED: list[str] = []
PASSED: list[str] = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}]   {name} {'' if ok else detail}")
    (PASSED if ok else FAILED).append(name)


def test_e0():
    print("\nE0 — evren tek kaynak (Y11, K7)")
    with open(os.path.join(ROOT, 'data', 'universe', 'bist30_panel.json'), encoding='utf-8') as f:
        uni = json.load(f)
    syms = uni['symbols']
    check("evren dosyasi: 30 benzersiz .IS sembol, panel sha256 ve tarih kayitli",
          len(syms) == 30 == uni['n_symbols'] and len(set(syms)) == 30
          and all(s.endswith('.IS') for s in syms) and len(uni['panel']['sha256']) == 64
          and uni['frozen_at'] and uni['panel']['last_date'], str(uni.get('panel')))

    panel = os.path.join(ROOT, uni['panel']['file'])
    if os.path.exists(panel):
        order = list(dict.fromkeys(pd.read_csv(panel, usecols=['symbol'])['symbol']))
        check("RL paneli ile ayni semboller, ayni SIRA (env durum vektoru sirasi)", order == syms,
              f"(fark {sorted(set(order) ^ set(syms))})")
    else:
        print("  [ATLA]  panel CSV yok (temiz klon); sira kontrolu atlandi")

    from data.bist30_symbols import BIST30_SYMBOLS, STOCK_INFO, PHASE1_SYMBOLS, get_symbols
    check("BIST30_SYMBOLS = evren dosyasi; get_symbols(phase=2) ayni (egitim rotasi varsayilani)",
          BIST30_SYMBOLS == syms and get_symbols(phase=2) == syms)
    check("STOCK_INFO her sembolun adini tasiyor, fazlasi yok; PHASE1 evrenin alt kumesi",
          set(STOCK_INFO) == set(syms) and all(v.get('name') for v in STOCK_INFO.values())
          and set(PHASE1_SYMBOLS) <= set(syms))
    import dashboard.pages.data as pdata
    import dashboard.pages.daily_trading as pdaily
    check("pano listeleri (veri, gunluk karar) evren dosyasindan, ayni sira",
          [s for s, _ in pdata.BIST30_SYMBOLS] == syms and list(pdaily.BIST30_SYMBOLS) == syms
          and pdata.ALL_BIST_VALUES == syms)
    old = {'ALARK.IS', 'HEKTS.IS', 'KOZA.IS', 'KOZAA.IS', 'KOZAL.IS', 'TTKOM.IS'}
    check("eski listelerin panelde olmayan sembolleri (Y11) kalmadi", not (old & set(syms)))


def test_e2():
    print("\nE2 — KAP sektor siniflamasi ve gruplar (Y5)")
    import csv
    import glob
    from collections import Counter
    from data.bist30_symbols import BIST30_SYMBOLS, SECTORS_FILE, load_sector_groups
    with open(SECTORS_FILE, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = load_sector_groups()
    size = Counter(groups.values())
    check("her evren sembolu tam bir satirda/grupta, fazlasi yok",
          [r['symbol'] for r in rows] == BIST30_SYMBOLS and set(groups) == set(BIST30_SYMBOLS))
    check(f"grup basina >= 2 sembol, 6-8 grup (olculen {len(size)} grup, en kucuk {min(size.values())})",
          min(size.values()) >= 2 and 6 <= len(size) <= 8, str(dict(size)))
    check("her satirda kaynak: KAP URL, erisim tarihi, tek ham sayfa SHA-256",
          all(r['source_url'] == 'https://kap.org.tr/tr/Sektorler' and r['retrieved_at']
              and len(r['source_sha256']) == 64 for r in rows)
          and len({r['source_sha256'] for r in rows}) == 1)
    sub = Counter(r['kap_sector'] for r in rows)
    check("grup kurali: grup = KAP alt sektoru; <2 uyeli alt sektor -> DIGER (mekanik)",
          all(r['group'] == (r['kap_sector'] if sub[r['kap_sector']] >= 2 else 'DIGER') for r in rows))
    src = sorted(glob.glob(os.path.join(ROOT, 'data', 'universe', 'sources', 'kap_sektorler_*.html.gz')))
    if src:
        import gzip
        import hashlib
        import subprocess
        raw = gzip.open(src[-1], 'rb').read()
        same_sha = hashlib.sha256(raw).hexdigest() == rows[0]['source_sha256']
        before = open(SECTORS_FILE, 'rb').read()
        subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'build_sectors.py'), '--html', src[-1]],
                       check=True, capture_output=True)
        check("saklanan ham sayfadan yeniden uretim bayt bayt ayni CSV (SHA-256 eslesir)",
              same_sha and open(SECTORS_FILE, 'rb').read() == before)
    else:
        print("  [ATLA]  yerel KAP kopyasi yok (data/universe/sources gitignore); yeniden uretim atlandi")


def test_e1_anchor():
    print("\nE1 — zincir capasi: bugunku BIST 30 (KAP XU030)")
    from data.bist30_symbols import BIST30_SYMBOLS
    with open(os.path.join(ROOT, 'data', 'universe', 'bist30_anchor.json'), encoding='utf-8') as f:
        a = json.load(f)
    codes, panel = set(a['codes']), {s[:-3] for s in BIST30_SYMBOLS}
    check("capa: 30 benzersiz kod, KAP URL, tarih, ham sayfa SHA-256",
          len(a['codes']) == 30 == len(codes) and a['source_url'].startswith('https://kap.org.tr/')
          and a['as_of'] and len(a['source_sha256']) == 64)
    check(f"panel/endeks farki dosyadaki kayitla tutarli (panelde olup endekste olmayan "
          f"{len(a['panel_not_in_index'])})",
          sorted(panel - codes) == a['panel_not_in_index'] and sorted(codes - panel) == a['index_not_in_panel'])


def _bm():
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import build_membership
    return build_membership


def test_e1_parser():
    print("\nE1 — duyuru tablosu ayristirici (sentetik HTML; uc bicim)")
    B = _bm()
    # Yeni bicim: tek tabloda iki endeks, satir ici baslik, rowspan'li unvan (2016 Q1 duzeni);
    # unvan 'VESTEL' kod kalibina uyar ama kod sutunu degildir.
    new = ('<div id="printableArea"><table><tr><td colspan="9">BIST 100 ENDEKSİ</td></tr>'
           '<tr><td colspan="3">ALINACAK PAYLAR</td><td colspan="3">ÇIKARILACAK PAYLAR</td>'
           '<td colspan="3">YEDEK PAYLAR</td></tr>'
           '<tr><td>1.</td><td>LOGO</td><td>LOGO YAZILIM</td><td>1.</td><td>NTHOL</td><td>NET</td>'
           '<td>1.</td><td>ASLAN</td><td>ASLAN</td></tr>'
           '<tr><td colspan="9"><p>BIST 30 ENDEKSİ</p></td></tr>'
           '<tr><td colspan="3">ALINACAK PAYLAR</td><td colspan="3">ÇIKARILACAK PAYLAR</td>'
           '<td colspan="3">YEDEK PAYLAR</td></tr>'
           '<tr><td>1.</td><td>VESTL</td><td rowspan="2">VESTEL</td><td>1.</td><td>TSKB</td>'
           '<td rowspan="2">T.S.K.B.</td><td>1.</td><td>OTKAR</td><td>OTOKAR</td></tr>'
           '<tr><td> </td><td> </td><td> </td><td> </td><td>2.</td><td>HEKTS</td><td>HEKTAS</td></tr>'
           '</table></div>')
    t = B.parse_tables(new)
    check("yeni bicim: satir ici endeks basligi, rowspan devami unvan sanilmaz, unvan kod sanilmaz",
          t['BIST 30']['ALINACAK'] == ['VESTL'] and t['BIST 30']['ÇIKARILACAK'] == ['TSKB']
          and t['BIST 30']['YEDEK'] == ['OTKAR', 'HEKTS'] and t['BIST 100']['ALINACAK'] == ['LOGO']
          and t['BIST 30']['names']['VESTL'] == 'VESTEL', str(t.get('BIST 30')))
    # Word ciktisi: kapsayici tablo yalniz basligi tasir, degisiklik ic tabloda (2022 Q1)
    word = ('<div id="printableArea"><table><tr><td><p>BIST 30 ENDEKSİ</p><table>'
            '<tr><td colspan="3">ALINACAK PAYLAR</td><td colspan="3">ÇIKARILACAK PAYLAR</td></tr>'
            '<tr><td>1</td><td>TOASO</td><td>TOFAS</td><td>1</td><td>VAKBN</td><td>VAKIFBANK</td></tr>'
            '</table></td></tr></table></div>')
    w = B.parse_tables(word).get('BIST 30', {})
    check("ic ice tablo: en icteki tablo okunur, kapsayici hucre basligi endeks adi olur",
          w.get('ALINACAK') == ['TOASO'] and w.get('ÇIKARILACAK') == ['VAKBN'], str(w))
    # Eski bicim: tablodan once baslik, hucrede <li>/<br> ile adlar, '1. ' numarasi, '---' bos
    old = ('<div id="printableArea"><p><strong>BIST 30 ENDEKSİ</strong></p><table>'
           '<tr><td>ALINACAK PAYLAR</td><td>ÇIKARILACAK PAYLAR</td><td>YEDEK PAYLAR</td></tr>'
           '<tr><td>1. ÜLKER BİSKÜVİ<br>2. TEKFEN HOLDİNG</td><td><p>---</p></td>'
           '<td><ol><li>TRAKYA CAM</li></ol></td></tr></table></div>')
    o = B.parse_tables(old).get('BIST 30', {})
    check("eski bicim: adlar ayri oge, numara ve '---' atilir",
          o.get('kind') == 'name' and o.get('ALINACAK') == ['ÜLKER BİSKÜVİ', 'TEKFEN HOLDİNG']
          and o.get('ÇIKARILACAK') == [] and o.get('YEDEK') == ['TRAKYA CAM'], str(o))
    check("ad normalizasyonu: Turkce harf/nokta/'(*)' esit, onek esit DEGIL",
          B._norm_name('İHLAS HOLDİNG') == B._norm_name('IHLAS HOLDING')
          and B._norm_name('DOĞAN HOLDİNG (*)') == B._norm_name('DOGAN HOLDING')
          and B._norm_name('VESTEL') != B._norm_name('VESTEL BEYAZ EŞYA'))


def test_e1_chain_logic():
    print("\nE1 — geriye zincir mantigi (sentetik)")
    B = _bm()
    base = [f'S{i:02d}' for i in range(30)]
    anchor = {'as_of': '2020-12-31', 'codes': base}
    ch = [{'period_start': '2020-10-01', 'action': 'add', 'code': 'S00', 'source_url': 'u', 'source_sha256': 'h'},
          {'period_start': '2020-10-01', 'action': 'remove', 'code': 'X', 'source_url': 'u', 'source_sha256': 'h'},
          {'period_start': '2020-07-01', 'action': 'none', 'code': '', 'source_url': 'u', 'source_sha256': 'h'}]
    segs, bad = B.chain(anchor, ch, [])
    k = {s: set(m) for s, _e, m, _ in segs}
    check("tutarli zincir: ihlal yok, eski donemde S00 yerine X, her segment 30",
          not bad and 'X' in k['2020-07-01'] and 'S00' not in k['2020-07-01'] and k['2020-10-01'] == set(base),
          str(bad))
    mut = [dict(ch[0], code='S99')] + ch[1:]   # alinan pay sonraki kumede yok -> ihlal + 31 uye
    _s, bad2 = B.chain(anchor, mut, [])
    check("mutasyon: sonraki kumede olmayan 'alinan' ve 30 disi uye sayisi yakalanir",
          any('alinan S99' in b for b in bad2) and any('31 uye' in b for b in bad2), str(bad2))
    ev = [{'date': '2020-08-15', 'action': 'rename', 'code': 'OLD1', 'new_code': 'S01', 'source_url': 'v',
           'source_sha256': 'g'}]
    segs3, bad3 = B.chain(anchor, ch, ev)
    k3 = {s: set(m) for s, _e, m, _ in segs3}
    check("kod degisikligi: olay tarihinden once eski kod, sonra yeni kod",
          not bad3 and 'OLD1' in k3['2020-07-01'] and 'S01' in k3['2020-08-15'], str(bad3))


def test_e1_membership():
    print("\nE1 — uyelik dosyalari, maske, maruziyet (depodaki ciktilar)")
    import csv
    import hashlib
    import subprocess
    B = _bm()
    from data import bist30_symbols as S
    uni = os.path.join(ROOT, 'data', 'universe')
    with open(os.path.join(uni, 'bist30_anchor.json'), encoding='utf-8') as f:
        anchor = json.load(f)
    with open(os.path.join(uni, 'bist30_membership_meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(uni, 'bist30_changes.csv'), encoding='utf-8', newline='') as f:
        changes = list(csv.DictReader(f))
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    check("meta: girdilerin SHA-256'si depodaki capa/degisiklik dosyalariyla ayni; status final|provisional",
          meta['inputs_sha256'].get('bist30_anchor.json') == sha(os.path.join(uni, 'bist30_anchor.json'))
          and meta['inputs_sha256'].get('bist30_changes.csv') == sha(os.path.join(uni, 'bist30_changes.csv'))
          and meta['status'] in ('final', 'provisional') and (meta['status'] == 'final') == (not meta['violations']))
    check("54 ceyrek okunmus (2013 Q3-2026 Q4), her satirda kaynak URL + SHA-256",
          len({r['quarter'] for r in changes}) == 54
          and all(r['source_url'].startswith('https://borsaistanbul.com/duyuru/') and len(r['source_sha256']) == 64
                  for r in changes))
    segs = S.load_membership()
    check("segmentler kesintisiz (son = sonrakinin basi), son segment = capa, tutarli segment 30 uye",
          all(a[1] == b[0] for a, b in zip(segs, segs[1:])) and segs[-1][1] is None
          and segs[-1][2] == frozenset(anchor['codes']) and all(len(k) == 30 for _s, _e, k, ok in segs if ok))
    cut = meta['consistent_from']
    check(f"maske: kapsam disi ve tutarsiz segment None; tutarli kisim {cut}..; bugun = capa",
          S.universe_mask('2013-06-28') is None and S.universe_mask(anchor['as_of']) == frozenset(anchor['codes'])
          and all((S.universe_mask(s) is not None) == ok for s, _e, _k, ok in segs)
          and S.universe_mask(segs[0][0], include_inconsistent=True) == segs[0][2])
    import pandas as pd
    viol = []
    for r in changes:
        if r['action'] in ('add', 'remove') and r['period_start'] >= cut:
            on = S.universe_mask(r['period_start'])
            before = S.universe_mask(pd.Timestamp(r['period_start']) - pd.Timedelta(days=1))
            want = r['action'] == 'add'
            if (r['code'] in on) != want or (before is not None and (r['code'] in before) == want):
                viol.append((r['quarter'], r['action'], r['code']))
    check("tutarli kisimdaki her ceyreklik degisiklik maskede yururluk gununde gorunur", not viol, str(viol))
    exp = os.path.join(ROOT, 'docs', 'development', 'measurements', 'phase9-e1', 'exposure.json')
    with open(exp, encoding='utf-8') as f:
        ex = json.load(f)
    check("maruziyet JSON'u depodaki uyelik dosyasina bagli (SHA-256) ve kapsam = tutarli kisim",
          ex['membership_sha256'] == sha(S.MEMBERSHIP_FILE) and ex['covered_from'] == cut
          and ex['membership_status'] == meta['status'])
    before = {p: open(p, 'rb').read() for p in (S.MEMBERSHIP_FILE, os.path.join(uni, 'bist30_membership_meta.json'))}
    args = ['chain'] + (['--provisional'] if meta['status'] == 'provisional' else [])
    subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'build_membership.py')] + args,
                   check=True, capture_output=True)
    check("zincir depodaki girdilerden bayt bayt ayni uyelik + meta uretir (ag yok)",
          all(open(p, 'rb').read() == b for p, b in before.items()))
    if os.path.exists(os.path.join(B.SRC, 'fetched.json')):
        outs = (os.path.join(uni, 'bist30_changes.csv'), os.path.join(uni, 'bist_name_map.csv'))
        prev = {p: open(p, 'rb').read() for p in outs}
        subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'build_membership.py'), 'parse'],
                       check=True, capture_output=True)
        check("saklanan ham duyurulardan bayt bayt ayni degisiklik + ad tablosu",
              all(open(p, 'rb').read() == b for p, b in prev.items()))
    else:
        print("  [ATLA]  yerel ham duyurular yok (sources gitignore); parse yeniden uretimi atlandi")


def main() -> int:
    test_e0()
    test_e2()
    test_e1_anchor()
    test_e1_parser()
    test_e1_chain_logic()
    test_e1_membership()
    print("\n" + "=" * 60)
    print(f"Sonuc: {len(PASSED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
