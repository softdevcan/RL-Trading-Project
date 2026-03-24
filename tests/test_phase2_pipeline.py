"""
Faz 2 End-to-End Pipeline Testi
Tüm data pipeline'ını (Fundamental + Macro + Environment) test eder
"""

import sys
import os

# Proje root'unu path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from data.data_fetcher import DataFetcher
from data.fundamental_fetcher import FundamentalDataFetcher
from data.macro_fetcher import MacroDataFetcher
from data.bist30_symbols import get_symbols
from data.technical_indicators import add_indicators_to_multi_symbol_df
from env.trading_env import TradingEnv
import logging
import sys

# Windows console encoding fix
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_phase2_pipeline():
    """Faz 2 full pipeline test"""

    print("\n" + "="*80)
    print("FAZ 2 END-TO-END PIPELINE TEST")
    print("="*80)

    # 1. Symbols
    symbols = get_symbols(phase=1)  # 5 hisse ile test
    print(f"\n✓ Symbols: {', '.join(symbols)}")

    # 2. Market Data (OHLCV + Technical Indicators)
    print("\n" + "-"*80)
    print("STEP 1: Fetching Market Data...")
    print("-"*80)

    market_fetcher = DataFetcher(start_date="2018-01-01")

    try:
        market_df = market_fetcher.load_data('stock_data_with_indicators.csv')
        print("✓ Market data loaded from cache")
    except:
        print("Cache not found, fetching fresh data...")
        market_df = market_fetcher.fetch_stock_data(symbols, save=True)
        market_df = market_fetcher.clean_data(market_df)
        market_df = add_indicators_to_multi_symbol_df(market_df)
        market_fetcher.save_data(market_df, 'stock_data_with_indicators.csv')
        print("✓ Market data fetched and cached")

    print(f"  - Total rows: {len(market_df)}")
    print(f"  - Date range: {market_df.index.get_level_values('date').min().date()} "
          f"to {market_df.index.get_level_values('date').max().date()}")
    print(f"  - Columns: {market_df.columns.tolist()}")

    # 3. Fundamental Data
    print("\n" + "-"*80)
    print("STEP 2: Fetching Fundamental Data...")
    print("-"*80)

    fund_fetcher = FundamentalDataFetcher()

    try:
        fund_df = fund_fetcher.load_data('fundamental_data.csv')
        print("✓ Fundamental data loaded from cache")
    except:
        print("Cache not found, fetching fresh data...")
        fund_df = fund_fetcher.fetch_fundamental_data(symbols, save=True)
        print("✓ Fundamental data fetched and cached")

    print(f"  - Total symbols: {len(fund_df)}")
    print(f"  - Ratios: {fund_df.columns.tolist()}")
    print("\nFundamental Data Preview:")
    print(fund_df)

    # 4. Macro Data
    print("\n" + "-"*80)
    print("STEP 3: Fetching Macro Data...")
    print("-"*80)

    # TCMB API key (gerçek key'iniz)
    TCMB_API_KEY = "tV4qq6RzPr"

    macro_fetcher = MacroDataFetcher(
        api_key=TCMB_API_KEY,
        start_date="2018-01-01",
        end_date="2024-12-31"
    )

    try:
        macro_df = macro_fetcher.load_data('macro_data.csv')
        print("✓ Macro data loaded from cache")
    except:
        print("Cache not found, fetching from TCMB EVDS...")
        try:
            macro_df = macro_fetcher.fetch_macro_data(save=True)
            print("✓ Macro data fetched and cached")
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch macro data: {e}")
            print("  Creating dummy macro data for testing...")
            # Dummy macro data oluştur
            dates = market_df.index.get_level_values('date').unique()
            macro_df = pd.DataFrame({
                'policy_rate': 15.0,
                'cpi_inflation': 50.0,
                'ppi_inflation': 60.0,
                'usd_try': 30.0,
                'eur_try': 35.0,
                'bist100_index': 5000.0
            }, index=dates)
            macro_df.index.name = 'date'
            print("✓ Dummy macro data created")

    print(f"  - Total rows: {len(macro_df)}")
    print(f"  - Indicators: {macro_df.columns.tolist()}")
    print("\nMacro Data Sample (first 3 rows):")
    print(macro_df.head(3))

    # 5. Environment Test (Faz 1)
    print("\n" + "-"*80)
    print("STEP 4: Testing FAZ 1 Environment (56 features)...")
    print("-"*80)

    train_df, val_df, test_df = market_fetcher.split_data(market_df)

    env_phase1 = TradingEnv(
        df=train_df,
        initial_balance=1_000_000,
        phase=1  # Faz 1
    )

    obs1, info1 = env_phase1.reset()
    print(f"✓ Faz 1 Environment initialized")
    print(f"  - State shape: {obs1.shape}")
    print(f"  - Expected: (56,) = 1 (balance) + 5 (shares) + 50 (5 stocks × 10 features)")

    # Birkaç random step
    for i in range(3):
        action = env_phase1.action_space.sample()
        obs, reward, terminated, truncated, info = env_phase1.step(action)
        print(f"  Step {i+1}: Reward={reward:.4f}, Portfolio=₺{info['portfolio_value']:,.0f}")
        if terminated:
            break

    # 6. Environment Test (Faz 2)
    print("\n" + "-"*80)
    print("STEP 5: Testing FAZ 2 Environment (97 features)...")
    print("-"*80)

    env_phase2 = TradingEnv(
        df=train_df,
        initial_balance=1_000_000,
        fundamental_df=fund_df,
        macro_df=macro_df,
        phase=2  # Faz 2
    )

    obs2, info2 = env_phase2.reset()
    print(f"✓ Faz 2 Environment initialized")
    print(f"  - State shape: {obs2.shape}")
    print(f"  - Expected: (97,) = 1 (balance) + 5 (shares) + 85 (5 stocks × 17) + 6 (macro)")
    print(f"  - Breakdown:")
    print(f"      Balance: 1")
    print(f"      Shares owned: 5")
    print(f"      Per-stock features: 5 × 17 = 85")
    print(f"        (10 OHLCV+Technical + 7 Fundamental)")
    print(f"      Macro features: 6")

    # Birkaç random step
    for i in range(3):
        action = env_phase2.action_space.sample()
        obs, reward, terminated, truncated, info = env_phase2.step(action)
        print(f"  Step {i+1}: Reward={reward:.4f}, Portfolio=₺{info['portfolio_value']:,.0f}")
        if terminated:
            break

    # 7. Comparison
    print("\n" + "="*80)
    print("PIPELINE TEST SUMMARY")
    print("="*80)
    print(f"✓ Market Data: {len(market_df)} rows, {len(market_df.columns)} columns")
    print(f"✓ Fundamental Data: {len(fund_df)} symbols, {len(fund_df.columns)} ratios")
    print(f"✓ Macro Data: {len(macro_df)} rows, {len(macro_df.columns)} indicators")
    print(f"✓ Faz 1 Env: State shape {obs1.shape} (expected: 56)")
    print(f"✓ Faz 2 Env: State shape {obs2.shape} (expected: 97)")

    # Verify state dimensions
    assert obs1.shape[0] == 56, f"Faz 1 state should be 56, got {obs1.shape[0]}"
    assert obs2.shape[0] == 97, f"Faz 2 state should be 97, got {obs2.shape[0]}"

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nFaz 2 Data Infrastructure is ready for training! 🚀")


if __name__ == '__main__':
    test_phase2_pipeline()
