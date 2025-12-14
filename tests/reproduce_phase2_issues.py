
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.trading_env import make_env
from data.fundamental_fetcher import FundamentalDataFetcher
from data.macro_fetcher import MacroDataFetcher
from data.bist30_symbols import get_symbols

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phase2_integration():
    print("\n" + "="*60)
    print("PHASE 2 INTEGRATION TEST")
    print("="*60)

    # 1. Mock Data Generation (to avoid API calls during reproduction if possible, or use fetchers if needed)
    # Let's try to use the actual fetchers first to see if they work, but fallback to mock if API keys fail
    
    symbols = get_symbols(phase=1)[:3] # Test with 3 symbols
    print(f"Testing with symbols: {symbols}")

    # Create dummy stock data
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    multi_index = pd.MultiIndex.from_product([symbols, dates], names=['symbol', 'date'])
    df = pd.DataFrame(index=multi_index)
    df['open'] = 10.0
    df['high'] = 11.0
    df['low'] = 9.0
    df['close'] = 10.5
    df['volume'] = 1000000
    # Add dummy indicators
    df['macd'] = 0.1
    df['rsi'] = 55.0
    df['cci'] = 10.0
    df['adx'] = 25.0
    df['turbulence'] = 0.0
    
    print("Stock data created.")

    # 2. Test Fundamental Fetcher
    print("\nTesting FundamentalDataFetcher (Real)...")
    fund_fetcher = FundamentalDataFetcher()
    # Use real fetcher
    try:
        fund_df = fund_fetcher.fetch_fundamental_data(symbols, save=False)
        print("Fundamental data fetched successfully.")
        print(fund_df.head())
    except Exception as e:
        print(f"Fundamental fetch failed: {e}")
        # Fallback to mock if fails (shouldn't happen if fetcher is robust)
        fund_data = {
            'roe': 15.0, 'roa': 5.0, 'debt_to_equity': 1.0, 
            'current_ratio': 1.5, 'pe_ratio': 10.0, 'pb_ratio': 1.5, 'profit_margin': 10.0
        }
        fund_df = pd.DataFrame([fund_data] * len(symbols), index=symbols)

    # 3. Test Macro Fetcher
    print("\nTesting MacroDataFetcher (Real)...")
    # Use real fetcher
    try:
        # Use a valid API key (or the one in the file)
        API_KEY = "tV4qq6RzPr"
        macro_fetcher = MacroDataFetcher(api_key=API_KEY, start_date="2023-01-01", end_date="2023-01-10")
        macro_df = macro_fetcher.fetch_macro_data(save=False)
        print("Macro data fetched successfully.")
        print(macro_df.head())
    except Exception as e:
        print(f"Macro fetch failed: {e}")
        # Fallback to mock
        macro_data = {
            'policy_rate': 45.0, 'cpi_inflation': 60.0, 'ppi_inflation': 40.0,
            'usd_try': 30.0, 'eur_try': 32.0, 'bist100_index': 8000.0
        }
        macro_df = pd.DataFrame([macro_data] * len(dates), index=dates)

    # 4. Create Environment with Phase 2 settings
    print("\nCreating TradingEnv (Phase 2)...")
    try:
        env = make_env(
            df,
            phase=2,
            reward_type='psr',
            fundamental_df=fund_df,
            macro_df=macro_df
        )
        print("Environment created successfully.")
        
        # Check observation space
        print(f"Observation space shape: {env.observation_space.shape}")
        expected_dim = 1 + len(symbols) + (len(symbols) * 17) + 6
        print(f"Expected dimension: {expected_dim}")
        
        if env.observation_space.shape[0] != expected_dim:
            print(f"ERROR: Dimension mismatch! Got {env.observation_space.shape[0]}, expected {expected_dim}")
        
        # Reset
        obs, info = env.reset()
        print(f"Reset successful. Obs shape: {obs.shape}")
        
        # Step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful. Reward: {reward}")
        print(f"Info keys: {info.keys()}")
        
        if 'reward_components' in info:
            print(f"Reward components: {info['reward_components']}")
            
    except Exception as e:
        print(f"ERROR during environment execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phase2_integration()
