"""
Test: Gold Fetcher
Altın verisi çekme ve gram altın hesaplaması testleri

Kullanım:
    python tests/test_gold_fetcher.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def test_gold_oz_fetch():
    """Ons altın verisi çekme testi"""
    print("\n[TEST 1] Ons altın (USD) verisi...")
    from data.gold_fetcher import GoldFetcher

    fetcher = GoldFetcher(start_date="2023-01-01")
    df = fetcher.fetch_gold_oz()

    assert df is not None, "Ons altın verisi çekilemedi!"
    assert not df.empty, "Ons altın DataFrame boş!"
    assert set(['open', 'high', 'low', 'close', 'volume']).issubset(df.columns), \
        f"Eksik kolonlar: {df.columns.tolist()}"
    assert (df['close'] > 0).all(), "Negatif/sıfır kapanış fiyatı var!"

    print(f"  OK: {len(df)} satır, son fiyat: {df['close'].iloc[-1]:.2f} USD/oz")
    print(f"  Tarih aralığı: {df.index[0].date()} - {df.index[-1].date()}")
    return df


def test_usdtry_fetch():
    """USD/TRY verisi çekme testi"""
    print("\n[TEST 2] USD/TRY verisi...")
    from data.gold_fetcher import GoldFetcher

    fetcher = GoldFetcher(start_date="2023-01-01")
    df = fetcher.fetch_usdtry()

    assert df is not None, "USD/TRY verisi çekilemedi!"
    assert not df.empty, "USD/TRY DataFrame boş!"
    assert (df['close'] > 0).all(), "Negatif/sıfır kur var!"

    print(f"  OK: {len(df)} satır, son kur: {df['close'].iloc[-1]:.4f} TRY/USD")
    return df


def test_gram_gold_computation(gold_oz_df, usdtry_df):
    """Gram altın hesaplama testi"""
    print("\n[TEST 3] Gram altın (TRY) hesaplama...")
    from data.gold_fetcher import GoldFetcher, TROY_OZ_TO_GRAM

    fetcher = GoldFetcher(start_date="2023-01-01")
    gram_df = fetcher.compute_gold_gram_try(gold_oz_df, usdtry_df)

    assert not gram_df.empty, "Gram altın DataFrame boş!"
    assert (gram_df['close'] > 0).all(), "Negatif gram altın fiyatı!"

    # Yaklaşık doğruluk kontrolü
    sample = gram_df['close'].iloc[-1]
    print(f"  OK: Son gram altın: {sample:.2f} TRY/g")
    print(f"  Formül kontrolü: (oz_usd * usdtry) / {TROY_OZ_TO_GRAM}")
    return gram_df


def test_fetch_all():
    """Tüm altın verilerini çekme testi"""
    print("\n[TEST 4] fetch_all_gold_data() testi...")
    from data.gold_fetcher import GoldFetcher, GOLD_OZ_SYMBOL, USDTRY_SYMBOL, GOLD_GRAM_SYMBOL

    fetcher = GoldFetcher(start_date="2023-01-01")
    df = fetcher.fetch_all_gold_data()

    assert not df.empty, "Birleşik altın DataFrame boş!"

    symbols = df.index.get_level_values('symbol').unique().tolist()
    print(f"  Semboller: {symbols}")

    expected = {GOLD_OZ_SYMBOL, USDTRY_SYMBOL, GOLD_GRAM_SYMBOL}
    for sym in expected:
        assert sym in symbols, f"Eksik sembol: {sym}"

    assert df.index.names == ['symbol', 'date'], "Multi-index hatalı!"

    print(f"  OK: {len(df)} toplam satır, {len(symbols)} sembol")
    return df


def test_latest_prices():
    """Güncel fiyat testi"""
    print("\n[TEST 5] get_latest_prices() testi...")
    from data.gold_fetcher import GoldFetcher

    fetcher = GoldFetcher(start_date="2024-01-01")
    prices = fetcher.get_latest_prices()

    assert len(prices) > 0, "Hiç güncel fiyat alınamadı!"
    for sym, info in prices.items():
        assert 'close' in info, f"{sym}: 'close' alanı yok"
        assert info['close'] > 0, f"{sym}: geçersiz fiyat {info['close']}"
        print(f"  {sym}: {info['close']:.4f} (değişim: {info['change_pct']:+.2f}%)")

    print("  OK!")


def test_bist30_symbols_update():
    """bist30_symbols.py güncellemesi testi"""
    print("\n[TEST 6] bist30_symbols.py güncelleme testi...")
    from data.bist30_symbols import (
        GOLD_SYMBOLS, FX_SYMBOLS, SYNTHETIC_SYMBOLS, PHASE3_SYMBOLS,
        get_symbols, get_all_tradeable_symbols, ASSET_INFO
    )

    assert 'GC=F' in GOLD_SYMBOLS, "GOLD_SYMBOLS hatalı"
    assert 'USDTRY=X' in FX_SYMBOLS, "FX_SYMBOLS hatalı"
    assert 'GOLD_GRAM_TRY' in SYNTHETIC_SYMBOLS, "SYNTHETIC_SYMBOLS hatalı"
    assert 'GOLD_GRAM_TRY' in PHASE3_SYMBOLS, "PHASE3_SYMBOLS hatalı"

    p3 = get_symbols(phase=3)
    assert 'GOLD_GRAM_TRY' in p3, "Phase 3 sembollerinde altın yok"

    all_sym = get_all_tradeable_symbols()
    assert 'GOLD_GRAM_TRY' in all_sym, "get_all_tradeable_symbols hatalı"

    for sym in ('GC=F', 'USDTRY=X', 'GOLD_GRAM_TRY'):
        assert sym in ASSET_INFO, f"ASSET_INFO'da {sym} yok"

    print(f"  Phase 3 sembolleri: {p3}")
    print("  OK!")


if __name__ == '__main__':
    print("=" * 60)
    print("GOLD FETCHER TESTLERİ")
    print("=" * 60)

    try:
        gold_oz = test_gold_oz_fetch()
        usdtry = test_usdtry_fetch()
        test_gram_gold_computation(gold_oz, usdtry)
        test_fetch_all()
        test_latest_prices()
        test_bist30_symbols_update()

        print("\n" + "=" * 60)
        print("TÜM TESTLER BAŞARILI!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\nTEST BAŞARISIZ: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nBEKLENMEDİK HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
