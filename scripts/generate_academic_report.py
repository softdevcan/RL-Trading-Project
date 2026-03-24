"""
Generate Comprehensive Academic Analysis Report
Compares all trained models and generates publication-ready visualizations
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from app.services.model_analysis import ModelAnalyzer
from stable_baselines3 import PPO, A2C, TD3, SAC
from app.environments.trading_env import MultiStockTradingEnv

def load_data_for_testing():
    """Load preprocessed data for testing"""
    data_file = 'data/preprocessed/train_val_test_data.pkl'

    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run data preprocessing first!")
        return None

    import pickle
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    return data['test']

def evaluate_model_on_test_set(model_path: str, test_df: pd.DataFrame,
                                initial_balance: float = 1_000_000,
                                commission_rate: float = 0.001,
                                max_shares_per_trade: int = 100):
    """
    Evaluate a trained model on test set and collect detailed metrics

    Returns:
        Tuple of (portfolio_values, returns, trades, metrics)
    """
    print(f"📊 Evaluating model: {model_path}")

    # Determine model type from name
    model_name_lower = model_path.lower()
    if "ppo" in model_name_lower:
        model = PPO.load(model_path)
    elif "a2c" in model_name_lower:
        model = A2C.load(model_path)
    elif "td3" in model_name_lower:
        model = TD3.load(model_path)
    elif "sac" in model_name_lower:
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown model type in: {model_path}")

    # Create test environment
    env = MultiStockTradingEnv(
        df=test_df,
        initial_balance=initial_balance,
        commission_rate=commission_rate,
        max_shares_per_trade=max_shares_per_trade
    )

    # Run evaluation
    obs, _ = env.reset()
    done = False
    truncated = False

    portfolio_values = [initial_balance]
    returns = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        portfolio_values.append(env.portfolio_value)
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            returns.append(daily_return)

    # Get final metrics and trades
    metrics = env.get_metrics()
    trades = env.trade_history

    # Add PnL to trades
    for trade in trades:
        if trade['action'] == 'BUY':
            trade['pnl'] = 0  # Can't calculate PnL for buy without corresponding sell
        elif trade['action'] == 'SELL':
            # Simple approximation
            trade['pnl'] = trade['total'] - trade['commission']

    return (
        np.array(portfolio_values),
        np.array(returns),
        trades,
        metrics
    )

def main():
    """Main analysis workflow"""
    print("=" * 80)
    print("🎓 ACADEMIC RESEARCH ANALYSIS - RL TRADING MODELS")
    print("=" * 80)

    # Initialize analyzer
    analyzer = ModelAnalyzer(results_dir="results")

    # Load test data
    print("\n📂 Loading test data...")
    test_df = load_data_for_testing()
    if test_df is None:
        return

    print(f"✅ Test data loaded: {len(test_df)} samples")

    # Find all trained models
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ No models directory found!")
        return

    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        print("❌ No trained models found!")
        return

    print(f"\n🤖 Found {len(model_files)} trained models")

    # Evaluate all models
    model_results = {}
    model_portfolios = {}
    model_returns = {}

    for model_file in model_files:
        model_name = model_file.stem
        model_path = str(model_file.with_suffix(''))

        try:
            # Evaluate model
            portfolio_values, returns, trades, base_metrics = evaluate_model_on_test_set(
                model_path, test_df
            )

            # Calculate advanced metrics
            advanced_metrics = analyzer.calculate_advanced_metrics(
                portfolio_values=portfolio_values,
                returns=returns,
                trades=trades,
                risk_free_rate=0.02
            )

            # Store results
            model_results[model_name] = advanced_metrics
            model_portfolios[model_name] = portfolio_values
            model_returns[model_name] = returns

            print(f"✅ {model_name}: Sharpe={advanced_metrics['sharpe_ratio']:.4f}, "
                  f"Return={advanced_metrics['total_return']*100:.2f}%")

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue

    if not model_results:
        print("\n❌ No models successfully evaluated!")
        return

    # Generate comprehensive report
    print("\n" + "=" * 80)
    analyzer.generate_comprehensive_report(
        model_results=model_results,
        model_portfolios=model_portfolios,
        model_returns=model_returns
    )

    # Save detailed results as JSON
    results_json = {}
    for model_name, metrics in model_results.items():
        results_json[model_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }

    json_path = analyzer.results_dir / "data" / "detailed_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n💾 Detailed results saved to: {json_path}")

    # Print best model summary
    print("\n" + "=" * 80)
    print("🏆 BEST MODELS BY METRIC")
    print("=" * 80)

    best_sharpe = max(model_results.items(), key=lambda x: x[1]['sharpe_ratio'])
    print(f"📈 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.4f})")

    best_return = max(model_results.items(), key=lambda x: x[1]['total_return'])
    print(f"💰 Best Total Return: {best_return[0]} ({best_return[1]['total_return']*100:.2f}%)")

    best_drawdown = max(model_results.items(), key=lambda x: x[1]['max_drawdown'])
    print(f"🛡️  Lowest Drawdown: {best_drawdown[0]} ({best_drawdown[1]['max_drawdown']*100:.2f}%)")

    best_winrate = max(model_results.items(), key=lambda x: x[1]['win_rate'])
    print(f"🎯 Best Win Rate: {best_winrate[0]} ({best_winrate[1]['win_rate']*100:.2f}%)")

    print("\n✅ Analysis complete! Check the 'results' directory for all outputs.")
    print("\n📚 For your thesis/paper:")
    print("   - Use figures from 'results/figures/' (publication quality, 300 DPI)")
    print("   - Use LaTeX tables from 'results/latex/'")
    print("   - Cite metrics from 'results/data/model_comparison.csv'")
    print("   - Read full report in 'results/ANALYSIS_REPORT.txt'")

if __name__ == "__main__":
    main()
