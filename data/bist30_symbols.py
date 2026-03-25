"""
BIST-30 Hisse Senedi Sembolleri
Türkiye'nin en büyük 30 şirketi
"""

# BIST-30 sembolleri (Yahoo Finance formatı: SEMBOL.IS)
BIST30_SYMBOLS = [
    'AKBNK.IS',   # Akbank
    'ALARK.IS',   # Alarko Holding
    'ARCLK.IS',   # Arçelik
    'ASELS.IS',   # Aselsan
    'BIMAS.IS',   # BIM
    'EKGYO.IS',   # Emlak Konut GYO
    'ENKAI.IS',   # Enka İnşaat
    'EREGL.IS',   # Ereğli Demir Çelik
    'FROTO.IS',   # Ford Otosan
    'GARAN.IS',   # Garanti Bankası
    'HEKTS.IS',   # Hektaş
    'ISCTR.IS',   # İş Bankası (C)
    'KCHOL.IS',   # Koç Holding
    'KRDMD.IS',   # Kardemir (D)
    'KOZAA.IS',   # Koza Altın
    'KOZAL.IS',   # Koza Anadolu Metal
    'KOZA.IS',    # Koza Madencilik (Not in BIST30 - placeholder)
    'PETKM.IS',   # Petkim
    'PGSUS.IS',   # Pegasus
    'SAHOL.IS',   # Sabancı Holding
    'SISE.IS',    # Şişe Cam
    'TAVHL.IS',   # TAV Havalimanları
    'TCELL.IS',   # Turkcell
    'THYAO.IS',   # Türk Hava Yolları
    'TKFEN.IS',   # Tekfen Holding
    'TOASO.IS',   # Tofaş
    'TTKOM.IS',   # Türk Telekom
    'TUPRS.IS',   # Tüpraş
    'VAKBN.IS',   # Vakıfbank
    'YKBNK.IS',   # Yapı Kredi Bankası
]

# Faz 1 için başlangıç seti (5 hisse ile test)
PHASE1_SYMBOLS = [
    'AKBNK.IS',   # Akbank - Banking
    'THYAO.IS',   # THY - Aviation
    'TUPRS.IS',   # Tüpraş - Energy
    'BIMAS.IS',   # BIM - Retail
    'ASELS.IS',   # Aselsan - Defense
]

# Hisse bilgileri
STOCK_INFO = {
    'AKBNK.IS': {'name': 'Akbank', 'sector': 'Banking'},
    'ALARK.IS': {'name': 'Alarko Holding', 'sector': 'Construction'},
    'ARCLK.IS': {'name': 'Arçelik', 'sector': 'Consumer Durables'},
    'ASELS.IS': {'name': 'Aselsan', 'sector': 'Defense'},
    'BIMAS.IS': {'name': 'BIM', 'sector': 'Retail'},
    'THYAO.IS': {'name': 'Turkish Airlines', 'sector': 'Aviation'},
    'TUPRS.IS': {'name': 'Tüpraş', 'sector': 'Energy'},
}

# Altın ve döviz sembolleri
GOLD_SYMBOLS = ['GC=F']
FX_SYMBOLS = ['USDTRY=X']
SYNTHETIC_SYMBOLS = ['GOLD_GRAM_TRY']

# Faz 3: BIST-30 + gram altın
PHASE3_SYMBOLS = PHASE1_SYMBOLS + ['GOLD_GRAM_TRY']

# Varlık bilgileri (altın ve döviz dahil)
ASSET_INFO = {
    'GC=F': {'name': 'Altin Ons (USD)', 'type': 'commodity', 'currency': 'USD'},
    'USDTRY=X': {'name': 'USD/TRY', 'type': 'fx', 'currency': 'TRY'},
    'GOLD_GRAM_TRY': {'name': 'Gram Altin (TRY)', 'type': 'commodity', 'currency': 'TRY'},
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
    """Tüm işlem yapılabilir sembolleri döndür (hisse + altın)."""
    return BIST30_SYMBOLS + SYNTHETIC_SYMBOLS


if __name__ == '__main__':
    print(f"BIST-30 toplam hisse sayısı: {len(BIST30_SYMBOLS)}")
    print(f"Faz 1 için seçilen hisseler: {len(PHASE1_SYMBOLS)}")
    print("\nFaz 1 Hisseler:")
    for symbol in PHASE1_SYMBOLS:
        info = STOCK_INFO.get(symbol, {})
        print(f"  - {symbol}: {info.get('name', 'N/A')} ({info.get('sector', 'N/A')})")
