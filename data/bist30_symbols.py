"""
BIST-30 Hisse Senedi Sembolleri

Faz 9 E0 (Y11, K7): sembol evreninin TEK kaynagi `data/universe/bist30_panel.json` — RL egitim
panelinde 2026-11-14 itibariyla bulunan 30 hisse, panelde gorunme sirasiyla
(`scripts/freeze_universe.py`). Bu GERCEK/tarihsel BIST-30 uyeligi DEGILDIR (hayatta kalma
yanliligi; tarihsel uyelik E1). Pano listeleri ve egitim rotasinin varsayilan veri cekimi buradan okur.
"""

import json as _json
import os as _os

_UNIVERSE_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'universe', 'bist30_panel.json')


def _load_universe():
    with open(_UNIVERSE_FILE, encoding='utf-8') as f:
        return _json.load(f)


UNIVERSE = _load_universe()

# BIST-30 sembolleri (Yahoo Finance formatı: SEMBOL.IS) — evren dosyasindaki sirayla
BIST30_SYMBOLS = list(UNIVERSE['symbols'])

# Faz 1 için başlangıç seti (5 hisse ile test)
PHASE1_SYMBOLS = [
    'AKBNK.IS',   # Akbank - Banking
    'THYAO.IS',   # THY - Aviation
    'TUPRS.IS',   # Tüpraş - Energy
    'BIMAS.IS',   # BIM - Retail
    'ASELS.IS',   # Aselsan - Defense
]

# Hisse bilgileri (ad). Sektor E2'de kaynakli dosyadan gelir (`data/universe/sectors.csv`);
# eski 7 sembollu kaynaksiz 'sector' alani kaldirildi (Y5).
STOCK_INFO = {
    'AKBNK.IS': {'name': 'Akbank'},
    'ARCLK.IS': {'name': 'Arçelik'},
    'ASELS.IS': {'name': 'Aselsan'},
    'BIMAS.IS': {'name': 'BİM'},
    'EKGYO.IS': {'name': 'Emlak Konut GYO'},
    'ENKAI.IS': {'name': 'Enka İnşaat'},
    'EREGL.IS': {'name': 'Ereğli Demir Çelik'},
    'FROTO.IS': {'name': 'Ford Otosan'},
    'GARAN.IS': {'name': 'Garanti BBVA'},
    'HALKB.IS': {'name': 'Halkbank'},
    'ISCTR.IS': {'name': 'İş Bankası (C)'},
    'KCHOL.IS': {'name': 'Koç Holding'},
    'KRDMD.IS': {'name': 'Kardemir (D)'},
    'MGROS.IS': {'name': 'Migros'},
    'ODAS.IS': {'name': 'Odaş Elektrik'},
    'PETKM.IS': {'name': 'Petkim'},
    'PGSUS.IS': {'name': 'Pegasus'},
    'SAHOL.IS': {'name': 'Sabancı Holding'},
    'SASA.IS': {'name': 'Sasa Polyester'},
    'SISE.IS': {'name': 'Şişecam'},
    'SKBNK.IS': {'name': 'Şekerbank'},
    'TAVHL.IS': {'name': 'TAV Havalimanları'},
    'TCELL.IS': {'name': 'Turkcell'},
    'THYAO.IS': {'name': 'Türk Hava Yolları'},
    'TKFEN.IS': {'name': 'Tekfen Holding'},
    'TOASO.IS': {'name': 'Tofaş'},
    'TSKB.IS': {'name': 'TSKB'},
    'TUPRS.IS': {'name': 'Tüpraş'},
    'VAKBN.IS': {'name': 'Vakıfbank'},
    'YKBNK.IS': {'name': 'Yapı Kredi'},
}

# Altın ve döviz sembolleri
GOLD_SYMBOLS = ['GC=F', 'GOLD_GRAM_USD', 'GOLD_ONS_TRY']
FX_SYMBOLS   = ['USDTRY=X', 'EURTRY=X']
SYNTHETIC_SYMBOLS = ['GOLD_GRAM_TRY', 'GOLD_ONS_TRY', 'GOLD_GRAM_USD']

# Faz 3: BIST-30 + gram altın
PHASE3_SYMBOLS = PHASE1_SYMBOLS + ['GOLD_GRAM_TRY']

# Varlık bilgileri (altın ve döviz dahil)
ASSET_INFO = {
    'GC=F':          {'name': 'Altin Ons (USD)',   'type': 'commodity', 'currency': 'USD'},
    'USDTRY=X':      {'name': 'USD/TRY',           'type': 'fx',        'currency': 'TRY'},
    'EURTRY=X':      {'name': 'EUR/TRY',           'type': 'fx',        'currency': 'TRY'},
    'GOLD_GRAM_TRY': {'name': 'Gram Altin (TRY)',  'type': 'commodity', 'currency': 'TRY'},
    'GOLD_ONS_TRY':  {'name': 'Ons Altin (TRY)',   'type': 'commodity', 'currency': 'TRY'},
    'GOLD_GRAM_USD': {'name': 'Gram Altin (USD)',   'type': 'commodity', 'currency': 'USD'},
}


def get_symbols(phase=1, include_gold=False):
    """
    Faz numarasına göre hisse listesi döndür

    Args:
        phase (int): 1 = 5 hisse, 2 = tüm BIST-30, 3 = BIST-30 + altın
        include_gold (bool): Altın sembollerini de ekle (phase bağımsız)

    Returns:
        list: Hisse/varlık sembolleri
    """
    if phase == 1:
        symbols = PHASE1_SYMBOLS
    elif phase == 3:
        symbols = PHASE3_SYMBOLS
    else:
        symbols = BIST30_SYMBOLS

    if include_gold and 'GOLD_GRAM_TRY' not in symbols:
        symbols = symbols + ['GOLD_GRAM_TRY']

    return symbols


def get_all_tradeable_symbols():
    """Tüm işlem yapılabilir sembolleri döndür (hisse + altın + döviz)."""
    seen = set()
    result = []
    for s in BIST30_SYMBOLS + GOLD_SYMBOLS + FX_SYMBOLS + SYNTHETIC_SYMBOLS:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


if __name__ == '__main__':
    print(f"BIST-30 toplam hisse sayısı: {len(BIST30_SYMBOLS)}")
    print(f"Faz 1 için seçilen hisseler: {len(PHASE1_SYMBOLS)}")
    print("\nFaz 1 Hisseler:")
    for symbol in PHASE1_SYMBOLS:
        info = STOCK_INFO.get(symbol, {})
        print(f"  - {symbol}: {info.get('name', 'N/A')}")
