"""Makro veri kalite bayragi kaliciligi — Faz 6 (G.2/R3).

Kapsanan bosluk: kalite bayraklari yalnizca CANLI cekimde `df.attrs`'e
iliştiriliyordu. Egitim yolu makroyu neredeyse her zaman CSV cache'inden
(`load_data`) okudugu icin bayrak kayboluyor, manifest fallback'li veriyi
'temiz' gosteriyordu. Bayraklar artik CSV'nin yanindaki `*_quality.json`
dosyasinda tasiniyor.

Calistir: python tests/test_macro_quality_flag.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath('.'))

import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import pandas as pd

from data.macro_fetcher import MacroDataFetcher
from prediction.manifest import TrainingManifest

FAILS = []


def check(name, cond, detail=''):
    print(('  [OK]   ' if cond else '  [FAIL] ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def _fetcher(tmp_dir, strict=False):
    """__init__'i atlayarak fetcher kur.

    Gercek __init__ EVDS_API_KEY ister ve evdsAPI istemcisi olusturur (ag).
    Bu test yalnizca disk kalicilik yolunu olctugu icin sadece gereken
    alanlar doldurulur — test hermetik kalir.
    """
    f = object.__new__(MacroDataFetcher)
    f.data_dir = tmp_dir
    f.strict_data = strict
    f.data_quality = {}
    return f


def _frame(quality=None):
    df = pd.DataFrame(
        {'policy_rate': [50.0, 50.0], 'usd_try': [34.1, 34.2]},
        index=pd.date_range('2026-01-01', periods=2),
    )
    if quality is not None:
        df.attrs['data_quality'] = quality
    return df


def main():
    tmp = tempfile.mkdtemp(prefix='macro_quality_')
    print(f"Gecici makro dizini: {tmp}")

    # --- 1. Fallback bayragi kaydedilir ve cache'ten geri okunur ---
    print("\n1) Fallback bayragi save -> load turunu sag cikiyor")
    f = _fetcher(tmp)
    flagged = {'policy_rate': 'fallback_constant_50.0'}
    f.save_data(_frame(flagged), 'macro_data.csv')

    sidecar = os.path.join(tmp, 'macro_data_quality.json')
    check('yan dosya yazildi', os.path.exists(sidecar))
    with open(sidecar, encoding='utf-8') as fh:
        payload = json.load(fh)
    check('yan dosyada bayrak var', payload.get('data_quality') == flagged,
          str(payload.get('data_quality')))
    check('cekim zamani kaydedildi', bool(payload.get('fetched_at')))

    loaded = f.load_data('macro_data.csv')
    check('load_data bayragi attrs e geri koydu',
          loaded.attrs.get('data_quality') == flagged,
          str(loaded.attrs.get('data_quality')))
    check('veri bozulmadi', len(loaded) == 2 and 'policy_rate' in loaded.columns)

    # --- 2. Manifest bayragi gercekten kaydediyor (G.2 kabul kriteri) ---
    print("\n2) Manifest fallback'i gosteriyor")
    m = TrainingManifest(run_kind='test', symbols=['AAA'])
    m.set_data_quality(loaded.attrs.get('data_quality'))
    d = m.to_dict()
    check("manifest data_quality.policy_rate = fallback",
          d['data_quality'].get('policy_rate') == 'fallback_constant_50.0',
          str(d['data_quality']))

    # --- 3. Temiz cekim eski bayragi temizler ---
    print("\n3) Temiz cekim bayragi siler (bayat uyari kalmaz)")
    f.save_data(_frame({}), 'macro_data.csv')
    clean = f.load_data('macro_data.csv')
    check('temiz cekimden sonra bayrak yok', clean.attrs.get('data_quality') == {},
          str(clean.attrs.get('data_quality')))

    # --- 4. Yan dosyasi olmayan eski CSV 'temiz' sayilmaz ---
    print("\n4) Eski (yan dosyasiz) CSV bilinmiyor olarak isaretlenir")
    os.remove(sidecar)
    legacy = f.load_data('macro_data.csv')
    check('bilinmiyor bayragi kondu',
          legacy.attrs.get('data_quality') == {'macro_csv': 'quality_unknown_legacy_file'},
          str(legacy.attrs.get('data_quality')))

    # --- 5. Bozuk yan dosya sessizce 'temiz' donmez ---
    print("\n5) Bozuk yan dosya")
    with open(sidecar, 'w', encoding='utf-8') as fh:
        fh.write('{bozuk json')
    broken = f.load_data('macro_data.csv')
    check('okunamayan yan dosya isaretlendi',
          broken.attrs.get('data_quality') == {'macro_csv': 'quality_unreadable'},
          str(broken.attrs.get('data_quality')))

    # --- 6. strict_data cache yolunda da patlar (R3) ---
    print("\n6) strict_data cache'lenmis fallback'i reddeder")
    f.save_data(_frame(flagged), 'macro_data.csv')
    strict = _fetcher(tmp, strict=True)
    try:
        strict.load_data('macro_data.csv')
        check('strict mod fallback cache te patliyor', False, 'hata firlatilmadi')
    except ValueError as e:
        check('strict mod fallback cache te patliyor', 'strict_data' in str(e))

    lenient = _fetcher(tmp, strict=False)
    got = lenient.load_data('macro_data.csv')
    check('lenient mod (varsayilan) patlamaz, sadece isaretler',
          got.attrs.get('data_quality') == flagged)

    strict_clean = _fetcher(tmp, strict=True)
    f.save_data(_frame({}), 'macro_data.csv')
    try:
        strict_clean.load_data('macro_data.csv')
        check('strict mod temiz veriyi kabul eder', True)
    except ValueError as e:
        check('strict mod temiz veriyi kabul eder', False, str(e))

    shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + '=' * 60)
    if FAILS:
        print(f"BASARISIZ: {len(FAILS)} kontrol -> {FAILS}")
        return 1
    print("TUM KONTROLLER GECTI — kalite bayragi cache turunu sag cikiyor")
    return 0


if __name__ == '__main__':
    sys.exit(main())
