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

def get_symbols(phase=1):
    """
    Faz numarasına göre hisse listesi döndür

    Args:
        phase (int): 1 = 5 hisse, 2-3 = tüm BIST-30

    Returns:
        list: Hisse sembolleri
    """
    if phase == 1:
        return PHASE1_SYMBOLS
    else:
        return BIST30_SYMBOLS

if __name__ == '__main__':
    print(f"BIST-30 toplam hisse sayısı: {len(BIST30_SYMBOLS)}")
    print(f"Faz 1 için seçilen hisseler: {len(PHASE1_SYMBOLS)}")
    print("\nFaz 1 Hisseler:")
    for symbol in PHASE1_SYMBOLS:
        info = STOCK_INFO.get(symbol, {})
        print(f"  - {symbol}: {info.get('name', 'N/A')} ({info.get('sector', 'N/A')})")
