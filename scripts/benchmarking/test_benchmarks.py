import pandas as pd
import json
import argparse
import sys
import numpy as np
from benchmark_strategies import evaluate_benchmarks

def parse_args():
    parser = argparse.ArgumentParser(description='Test Benchmark Strategies')
    parser.add_argument('--results_file', type=str, help='Path to RL training results JSON file for comparison')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load existing data
    print("Loading data...")
    # Try different potential data paths
    data_paths = [
        'data/stock_data_with_indicators.csv',
        '../data/stock_data_with_indicators.csv'
    ]
    
    df = None
    for path in data_paths:
        try:
            df = pd.read_csv(path)
            print(f"Loaded data from {path}")
            break
        except FileNotFoundError:
            continue
            
    if df is None:
        print("[ERROR] Could not load data file.")
        return

    # Convert to multi-index format
    if 'date' in df.columns and 'symbol' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Ensure UTC timezone if needed, or just normalize
        if df['date'].dt.tz is None:
             df['date'] = df['date'].dt.tz_localize('UTC') # Assume UTC if naive
        else:
             df['date'] = df['date'].dt.tz_convert('UTC')

        df = df.set_index(['date', 'symbol'])
        print(f"Data loaded: {len(df)} rows")
        
        # Determine test period
        test_df = None
        rl_metrics = None
        
        if args.results_file:
            try:
                with open(args.results_file, 'r') as f:
                    data = json.load(f)
                    
                print(f"Loaded RL results from {args.results_file}")
                
                # Extract RL metrics
                if 'metrics' in data:
                    rl_metrics = data['metrics']
                    # Flatten if nested
                    if 'final_portfolio_value' not in rl_metrics and 'final_portfolio_value' in data:
                        # Maybe metrics are top level?
                         pass
                else:
                    # Assume top level keys are metrics
                    rl_metrics = data
                
                # Determine date range from trades
                start_date = None
                end_date = None
                
                if 'trades' in data and len(data['trades']) > 0:
                    trades = data['trades']
                    # Parse dates from trades
                    trade_dates = [pd.to_datetime(t['date']) for t in trades]
                    start_date = min(trade_dates)
                    end_date = max(trade_dates)
                    print(f"Detected RL Trade Interval: {start_date.date()} to {end_date.date()}")
                
                if start_date and end_date:
                    # Filter df to this range
                    # We need to handle timezone matching. Assuming df is UTC.
                    # Trade dates usually strings, pd.to_datetime should handle.
                     if start_date.tzinfo is None:
                         start_date = start_date.tz_localize('UTC')
                     if end_date.tzinfo is None:
                         end_date = end_date.tz_localize('UTC')
                         
                     # Ensure df index dates are compatible
                     dates = df.index.get_level_values('date')
                     
                     # Add a small buffer to start/end to ensure we catch the range
                     mask = (dates >= start_date) & (dates <= end_date)
                     test_df = df.loc[mask]
                     
                     if len(test_df) == 0:
                         print("[WARNING] No data found for the exact date range. Using broad intersection.")
                         # fallback
                         pass
                     else:
                         print(f"Aligned test data to RL results: {len(test_df)} rows")

            except Exception as e:
                print(f"[ERROR] Failed to process results file: {e}")
                import traceback
                traceback.print_exc()

        if test_df is None or len(test_df) == 0:
            print("Using default 'last 30%' split for test data...")
            dates = df.index.get_level_values('date').unique().sort_values()
            test_start_idx = int(len(dates) * 0.7)
            test_dates = dates[test_start_idx:]
            test_df = df.loc[df.index.get_level_values('date').isin(test_dates)]

        print(f"\nTest data: {len(test_df)} rows, {test_df.index.get_level_values('date').nunique()} days")
        print("=" * 80)

        # Evaluate benchmarks
        try:
            benchmark_results = evaluate_benchmarks(test_df)

            print("\n" + "=" * 80)
            print("BENCHMARK COMPARISON RESULTS")
            print("=" * 80)
            
            # Prepare comparison table
            # Provide RL metrics if available
            if rl_metrics:
                print(f"{'Metric':<25} {'RL Agent':<15} {'Buy-and-Hold':<15} {'BIST-30':<15}")
                print("-" * 70)
                
                # Helper to safely get metric
                def get_m(m_dict, key, factor=1.0, fmt="{:.2f}"):
                    val = m_dict.get(key, 0)
                    if val is None: val = 0
                    return fmt.format(val * factor)

                metrics_map = {
                    'Total Return': ('total_return', 100, "{:.2f}%"),
                    'Annualized Return': ('annualized_return', 100, "{:.2f}%"),
                    'Sharpe Ratio': ('sharpe_ratio', 1, "{:.4f}"),
                    'Sortino Ratio': ('sortino_ratio', 1, "{:.4f}"),
                    'Max Drawdown': ('max_drawdown', 100, "{:.2f}%"),
                    'Volatility': ('volatility', 100, "{:.2f}%"),
                    'Final Value': ('final_portfolio_value', 1, "{:,.0f}"),
                    'Trades': ('total_trades', 1, "{}")
                }
                
                # Check mapping for RL agent (keys might differ)
                # JSON keys saw: 'cumulative_return', 'sharpe_ratio', 'max_drawdown', 'final_portfolio_value'
                # Inspect temp showed keys.
                rl_map = {
                    'total_return': 'cumulative_return',
                    'annualized_return': 'annualized_return', # might be missing
                    'sharpe_ratio': 'sharpe_ratio',
                    'sortino_ratio': 'sortino_ratio', # might be missing
                    'max_drawdown': 'max_drawdown',
                    'volatility': 'volatility', # might be missing
                    'final_portfolio_value': 'final_portfolio_value',
                    'total_trades': 'trades' # In JSON it is 'trades' list, but maybe 'total_trades' key exists?
                                            # Inspect showed 'trades' as key for LIST of trades.
                                            # We can calc len if it's a list.
                }

                for label, (key, factor, fmt) in metrics_map.items():
                    # RL Value
                    rl_val_raw = rl_metrics.get(rl_map.get(key, key))
                    # If rl_val_raw is a list (like trades), take len
                    if key == 'total_trades' and isinstance(rl_val_raw, list):
                        rl_val = len(rl_val_raw)
                    elif key == 'total_trades' and isinstance(rl_metrics.get('trades'), int):
                         rl_val = rl_metrics.get('trades')
                    else:
                        rl_val = rl_val_raw
                    
                    if rl_val is None:
                        rl_str = "N/A"
                    else:
                        try:
                            rl_str = fmt.format(float(rl_val) * factor)
                        except:
                            rl_str = str(rl_val)

                    # Benchmark Values
                    bah_metrics = benchmark_results['Buy-and-Hold'][3]
                    bist_metrics = benchmark_results['BIST-30'][3]
                    
                    bah_str = get_m(bah_metrics, key, factor, fmt)
                    bist_str = get_m(bist_metrics, key, factor, fmt)
                    
                    print(f"{label:<25} {rl_str:<15} {bah_str:<15} {bist_str:<15}")

            else:
                # Standard print if no RL metrics
                for name, (portfolio, returns, trades, metrics) in benchmark_results.items():
                    print(f"\n{name}:")
                    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
                    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
                    print(f"  Final Value: ${metrics['final_portfolio_value']:,.0f}")

            print("\n" + "=" * 80)
            print("[SUCCESS] Benchmark strategies tested successfully!")
            print("=" * 80)

        except Exception as e:
            print(f"\n[ERROR] Benchmark evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[ERROR] Data format not recognized. Expected 'date' and 'symbol' columns.")

if __name__ == "__main__":
    main()
