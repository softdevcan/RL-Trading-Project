"""
Test Benchmark Strategies with existing data
"""

import pandas as pd
from benchmark_strategies import evaluate_benchmarks

# Load existing data
print("Loading data...")
df = pd.read_csv('data/stock_data_with_indicators.csv')

# Convert to multi-index format
if 'date' in df.columns and 'symbol' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'symbol'])
    print(f"Data loaded: {len(df)} rows")
    print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    print(f"Symbols: {df.index.get_level_values('symbol').unique().tolist()}")

    # Use last 30% as test data (similar to typical train/val/test split)
    dates = df.index.get_level_values('date').unique().sort_values()
    test_start_idx = int(len(dates) * 0.7)
    test_dates = dates[test_start_idx:]
    test_df = df.loc[df.index.get_level_values('date').isin(test_dates)]

    print(f"\nTest data: {len(test_df)} rows, {len(test_dates)} days")
    print("=" * 80)

    # Evaluate benchmarks
    try:
        benchmark_results = evaluate_benchmarks(test_df)

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        for name, (portfolio, returns, trades, metrics) in benchmark_results.items():
            print(f"\n{name}:")
            print(f"  Total Return: {metrics['total_return']*100:.2f}%")
            print(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"  Volatility: {metrics['volatility']*100:.2f}%")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Final Value: ${metrics['final_portfolio_value']:,.0f}")
            print(f"  Days: {metrics['n_days']}")

        print("\n" + "=" * 80)
        print("[SUCCESS] Benchmark strategies tested successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Benchmark evaluation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("[ERROR] Data format not recognized. Expected 'date' and 'symbol' columns.")
