"""
Extract comparison table data from metrics JSON files
Outputs clean data ready for Google Docs/Sheets
"""

import json
import numpy as np
from pathlib import Path

def calculate_sortino_from_portfolio(portfolio_history, risk_free_rate=0.02):
    """Calculate Sortino ratio from portfolio history"""
    portfolio_values = np.array(portfolio_history)

    if len(portfolio_values) < 2:
        return 0.0

    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Annualization factor
    annualization_factor = np.sqrt(252)

    # Excess returns
    excess_returns = returns - (risk_free_rate / 252)

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10

    # Sortino ratio
    sortino_ratio = (np.mean(excess_returns) / (downside_deviation + 1e-10)) * annualization_factor

    return float(sortino_ratio)

def calculate_annualized_return(cumulative_return, n_days):
    """Calculate annualized return"""
    if n_days == 0:
        return 0.0
    years = n_days / 252
    if years == 0:
        return 0.0
    return ((1 + cumulative_return) ** (1 / years)) - 1

# Load all metrics files
results_dir = Path("results")
metrics_files = sorted(results_dir.glob("*_metrics.json"))

print("=" * 100)
print("MODEL COMPARISON TABLE - Ready for Google Docs/Sheets")
print("=" * 100)
print()

# Collect data
models_data = []

for metrics_file in metrics_files:
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    algorithm = metrics.get("algorithm", "Unknown").upper()

    # Calculate Sortino if not present
    sortino = metrics.get("sortino_ratio")
    if sortino is None and "portfolio_history" in metrics:
        sortino = calculate_sortino_from_portfolio(metrics["portfolio_history"])

    # Calculate annualized return
    cumulative_return = metrics.get("cumulative_return", 0)
    n_days = len(metrics.get("portfolio_history", [])) - 1
    annualized_return = calculate_annualized_return(cumulative_return, n_days) if n_days > 0 else 0

    models_data.append({
        "Algorithm": algorithm,
        "Cumulative_Return_Pct": cumulative_return * 100,
        "Annualized_Return_Pct": annualized_return * 100,
        "Sharpe_Ratio": metrics.get("sharpe_ratio", 0),
        "Sortino_Ratio": sortino if sortino else 0,
        "Max_Drawdown_Pct": abs(metrics.get("max_drawdown", 0)) * 100,
        "Total_Trades": metrics.get("total_trades", 0),
        "Learning_Rate": metrics.get("learning_rate", 0)
    })

# Add Buy-and-Hold benchmark (from test_benchmarks.py results)
models_data.append({
    "Algorithm": "BUY-AND-HOLD",
    "Cumulative_Return_Pct": 146.44,
    "Annualized_Return_Pct": 41.47,
    "Sharpe_Ratio": 1.5183,
    "Sortino_Ratio": 2.5150,
    "Max_Drawdown_Pct": 20.13,
    "Total_Trades": 4,
    "Learning_Rate": "-"
})

# Add BIST-30 (equal-weighted synthetic index from 5 stocks)
# This would be ~0 because synthetic index wasn't working properly
# You may want to recalculate this with actual data
models_data.append({
    "Algorithm": "BIST-30",
    "Cumulative_Return_Pct": 0.00,  # Placeholder - needs recalculation
    "Annualized_Return_Pct": 0.00,
    "Sharpe_Ratio": 0.00,
    "Sortino_Ratio": 0.00,
    "Max_Drawdown_Pct": 0.00,
    "Total_Trades": 1,
    "Learning_Rate": "-"
})

# Sort by algorithm name
models_data.sort(key=lambda x: (x["Algorithm"] not in ["BUY-AND-HOLD", "BIST-30"], x["Algorithm"]))

# Print in tab-separated format (easy to paste into Google Sheets)
print("TAB-SEPARATED FORMAT (Copy to Google Sheets):")
print("-" * 100)
print("\t".join([
    "Model",
    "Kümülatif Getiri (%)",
    "Yıllık Getiri (%)",
    "Sharpe Oranı",
    "Sortino Oranı",
    "Maks. Düşüş (%)",
    "İşlem Sayısı",
    "Learning Rate"
]))

for model in models_data:
    print("\t".join([
        model["Algorithm"],
        f"{model['Cumulative_Return_Pct']:.2f}",
        f"{model['Annualized_Return_Pct']:.2f}",
        f"{model['Sharpe_Ratio']:.4f}",
        f"{model['Sortino_Ratio']:.4f}",
        f"{model['Max_Drawdown_Pct']:.2f}",
        str(model["Total_Trades"]),
        str(model["Learning_Rate"]) if isinstance(model["Learning_Rate"], str) else f"{model['Learning_Rate']:.6f}"
    ]))

print()
print("=" * 100)
print()

# Print in Markdown format (for documentation)
print("MARKDOWN TABLE FORMAT:")
print("-" * 100)
print("| Model | Kümülatif Getiri (%) | Yıllık Getiri (%) | Sharpe Oranı | Sortino Oranı | Maks. Düşüş (%) | İşlem Sayısı | Learning Rate |")
print("|-------|---------------------|-------------------|--------------|---------------|-----------------|--------------|---------------|")

for model in models_data:
    lr_str = str(model["Learning_Rate"]) if isinstance(model["Learning_Rate"], str) else f"{model['Learning_Rate']:.6f}"
    print(f"| {model['Algorithm']} | {model['Cumulative_Return_Pct']:.2f} | {model['Annualized_Return_Pct']:.2f} | {model['Sharpe_Ratio']:.4f} | {model['Sortino_Ratio']:.4f} | {model['Max_Drawdown_Pct']:.2f} | {model['Total_Trades']} | {lr_str} |")

print()
print("=" * 100)
print()

# Print summary statistics
print("SUMMARY STATISTICS:")
print("-" * 100)

rl_models = [m for m in models_data if m["Algorithm"] not in ["BUY-AND-HOLD", "BIST-30"]]

if rl_models:
    best_return = max(rl_models, key=lambda x: x["Cumulative_Return_Pct"])
    best_sharpe = max(rl_models, key=lambda x: x["Sharpe_Ratio"])
    best_sortino = max(rl_models, key=lambda x: x["Sortino_Ratio"])
    lowest_drawdown = min(rl_models, key=lambda x: x["Max_Drawdown_Pct"])

    print(f"Best Cumulative Return: {best_return['Algorithm']} ({best_return['Cumulative_Return_Pct']:.2f}%)")
    print(f"Best Sharpe Ratio: {best_sharpe['Algorithm']} ({best_sharpe['Sharpe_Ratio']:.4f})")
    print(f"Best Sortino Ratio: {best_sortino['Algorithm']} ({best_sortino['Sortino_Ratio']:.4f})")
    print(f"Lowest Max Drawdown: {lowest_drawdown['Algorithm']} ({lowest_drawdown['Max_Drawdown_Pct']:.2f}%)")

print()
print("=" * 100)
print()

# Print CSV format
print("CSV FORMAT (Save as .csv file):")
print("-" * 100)
print(",".join([
    "Model",
    "Kumulative_Return_Pct",
    "Annualized_Return_Pct",
    "Sharpe_Ratio",
    "Sortino_Ratio",
    "Max_Drawdown_Pct",
    "Total_Trades",
    "Learning_Rate"
]))

for model in models_data:
    lr_str = str(model["Learning_Rate"]) if isinstance(model["Learning_Rate"], str) else f"{model['Learning_Rate']}"
    print(",".join([
        model["Algorithm"],
        f"{model['Cumulative_Return_Pct']:.2f}",
        f"{model['Annualized_Return_Pct']:.2f}",
        f"{model['Sharpe_Ratio']:.4f}",
        f"{model['Sortino_Ratio']:.4f}",
        f"{model['Max_Drawdown_Pct']:.2f}",
        str(model["Total_Trades"]),
        lr_str
    ]))

print()
print("=" * 100)
print("\nNOTE: BIST-30 values are placeholders and need recalculation with proper data.")
print("Buy-and-Hold values are from test_benchmarks.py run on test data (598 days).")
print("=" * 100)
