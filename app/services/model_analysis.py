"""
Model Analysis and Visualization for Academic Research
Generates comprehensive performance metrics, comparisons, and publication-ready visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class ModelAnalyzer:
    """
    Comprehensive model analysis for academic publications
    Generates metrics, comparisons, and publication-ready visualizations
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Academic publication directories
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "tables").mkdir(exist_ok=True)
        (self.results_dir / "latex").mkdir(exist_ok=True)
        (self.results_dir / "data").mkdir(exist_ok=True)

    def calculate_advanced_metrics(self, portfolio_values: np.ndarray,
                                   returns: np.ndarray,
                                   trades: List[Dict],
                                   risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate comprehensive performance metrics for academic analysis

        Args:
            portfolio_values: Array of portfolio values over time
            returns: Array of daily returns
            trades: List of trade dictionaries
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Dictionary with comprehensive metrics
        """

        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Annualized metrics (assuming 252 trading days)
        n_days = len(returns)
        annualization_factor = np.sqrt(252)

        # Returns analysis
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        annualized_return = mean_return * 252
        annualized_volatility = std_return * annualization_factor

        # Sharpe Ratio
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * annualization_factor

        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / (downside_deviation + 1e-10) * annualization_factor

        # Maximum Drawdown
        cumulative = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative) / cumulative
        max_drawdown = np.min(drawdown)

        # Calmar Ratio (return / max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win Rate and Profit Factor
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        # Average trade metrics
        avg_profit = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0

        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5)

        # Conditional Value at Risk (CVaR) - Expected Shortfall
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0

        # Information Ratio (assuming benchmark = 0)
        information_ratio = mean_return / (std_return + 1e-10) * annualization_factor

        # Recovery Factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Ulcer Index (measure of downside volatility)
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))

        return {
            # Return Metrics
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "mean_daily_return": float(mean_return),

            # Risk Metrics
            "volatility": float(std_return),
            "annualized_volatility": float(annualized_volatility),
            "max_drawdown": float(max_drawdown),
            "ulcer_index": float(ulcer_index),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),

            # Risk-Adjusted Returns
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "information_ratio": float(information_ratio),
            "recovery_factor": float(recovery_factor),

            # Trading Metrics
            "total_trades": len(trades),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_profit": float(avg_profit),
            "avg_loss": float(avg_loss),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),

            # Portfolio Metrics
            "final_portfolio_value": float(portfolio_values[-1]),
            "initial_portfolio_value": float(portfolio_values[0]),
            "n_days": n_days
        }

    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models and generate comparison table

        Args:
            model_results: Dictionary with model_name -> metrics

        Returns:
            DataFrame with model comparisons
        """
        comparison_data = []

        for model_name, metrics in model_results.items():
            comparison_data.append({
                "Model": model_name,
                "Total Return (%)": metrics["total_return"] * 100,
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Sortino Ratio": metrics["sortino_ratio"],
                "Max Drawdown (%)": metrics["max_drawdown"] * 100,
                "Win Rate (%)": metrics["win_rate"] * 100,
                "Profit Factor": metrics["profit_factor"],
                "Total Trades": metrics["total_trades"],
                "Calmar Ratio": metrics["calmar_ratio"],
                "VaR 95%": metrics["var_95"],
            })

        df = pd.DataFrame(comparison_data)
        df = df.round(4)

        return df

    def statistical_significance_test(self, returns_1: np.ndarray,
                                      returns_2: np.ndarray,
                                      test_type: str = "t-test") -> Dict:
        """
        Perform statistical significance tests between two models

        Args:
            returns_1: Returns from model 1
            returns_2: Returns from model 2
            test_type: "t-test" or "wilcoxon"

        Returns:
            Dictionary with test results
        """
        if test_type == "t-test":
            statistic, p_value = stats.ttest_ind(returns_1, returns_2)
            test_name = "Independent t-test"
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(returns_1, returns_2)
            test_name = "Wilcoxon signed-rank test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return {
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01
        }

    def generate_latex_table(self, df: pd.DataFrame,
                            caption: str,
                            label: str,
                            filename: str) -> str:
        """
        Generate LaTeX table for academic publication

        Args:
            df: DataFrame to convert
            caption: Table caption
            label: LaTeX label for referencing
            filename: Output filename

        Returns:
            LaTeX table string
        """
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{" + "l" + "c" * (len(df.columns) - 1) + "}\n"
        latex += "\\toprule\n"

        # Header
        latex += " & ".join(df.columns) + " \\\\\n"
        latex += "\\midrule\n"

        # Rows
        for _, row in df.iterrows():
            latex += " & ".join(str(val) for val in row.values) + " \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        # Save to file
        output_path = self.results_dir / "latex" / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)

        return latex

    def plot_portfolio_comparison(self, model_portfolios: Dict[str, np.ndarray],
                                  filename: str = "portfolio_comparison.pdf"):
        """
        Plot portfolio evolution comparison for multiple models

        Args:
            model_portfolios: Dictionary with model_name -> portfolio_values array
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for model_name, values in model_portfolios.items():
            normalized = (values / values[0]) * 100  # Normalize to 100
            ax.plot(normalized, label=model_name, linewidth=2)

        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Portfolio Value (Normalized to 100)', fontsize=12)
        ax.set_title('Portfolio Evolution Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / "figures" / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_drawdown_comparison(self, model_portfolios: Dict[str, np.ndarray],
                                 filename: str = "drawdown_comparison.pdf"):
        """
        Plot drawdown comparison for multiple models
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for model_name, values in model_portfolios.items():
            cumulative = np.maximum.accumulate(values)
            drawdown = (values - cumulative) / cumulative * 100
            ax.plot(drawdown, label=model_name, linewidth=2)

        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3)
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(self.results_dir / "figures" / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_returns_distribution(self, model_returns: Dict[str, np.ndarray],
                                  filename: str = "returns_distribution.pdf"):
        """
        Plot returns distribution comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        for model_name, returns in model_returns.items():
            axes[0].hist(returns * 100, bins=50, alpha=0.6, label=model_name)

        axes[0].set_xlabel('Daily Returns (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Box plot
        data_for_box = [returns * 100 for returns in model_returns.values()]
        axes[1].boxplot(data_for_box, labels=model_returns.keys())
        axes[1].set_ylabel('Daily Returns (%)', fontsize=12)
        axes[1].set_title('Returns Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.results_dir / "figures" / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_return_scatter(self, model_metrics: Dict[str, Dict],
                                filename: str = "risk_return_scatter.pdf"):
        """
        Plot risk-return scatter plot (annualized return vs volatility)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name, metrics in model_metrics.items():
            ax.scatter(
                metrics['annualized_volatility'] * 100,
                metrics['annualized_return'] * 100,
                s=200,
                label=model_name,
                alpha=0.7
            )
            # Add model name next to point
            ax.annotate(
                model_name,
                (metrics['annualized_volatility'] * 100, metrics['annualized_return'] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )

        ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(self.results_dir / "figures" / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_metrics_radar(self, model_metrics: Dict[str, Dict],
                                       filename: str = "performance_radar.pdf"):
        """
        Plot radar chart comparing normalized performance metrics
        """
        # Select key metrics for radar chart
        metrics_to_plot = [
            'sharpe_ratio', 'sortino_ratio', 'win_rate',
            'profit_factor', 'calmar_ratio'
        ]

        # Normalize metrics to 0-1 scale
        normalized_data = {}
        for metric in metrics_to_plot:
            values = [model_metrics[m][metric] for m in model_metrics.keys()]
            max_val = max(values) if max(values) > 0 else 1
            normalized_data[metric] = {
                m: model_metrics[m][metric] / max_val
                for m in model_metrics.keys()
            }

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for model_name in model_metrics.keys():
            values = [normalized_data[metric][model_name] for metric in metrics_to_plot]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax.set_ylim(0, 1)
        ax.set_title('Normalized Performance Metrics Comparison',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.results_dir / "figures" / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self, model_results: Dict[str, Dict],
                                     model_portfolios: Dict[str, np.ndarray],
                                     model_returns: Dict[str, np.ndarray]):
        """
        Generate comprehensive analysis report with all visualizations and tables

        Args:
            model_results: Dictionary with model_name -> metrics
            model_portfolios: Dictionary with model_name -> portfolio_values
            model_returns: Dictionary with model_name -> returns
        """
        print("🔬 Generating Comprehensive Academic Analysis Report...")

        # 1. Generate comparison table
        comparison_df = self.compare_models(model_results)
        print(f"\n📊 Model Comparison Table:\n{comparison_df.to_string()}")

        # Save as CSV
        comparison_df.to_csv(self.results_dir / "data" / "model_comparison.csv", index=False)

        # Generate LaTeX table
        self.generate_latex_table(
            comparison_df,
            caption="Performance Comparison of RL Trading Algorithms",
            label="tab:model_comparison",
            filename="model_comparison.tex"
        )

        # 2. Generate all plots
        print("\n📈 Generating visualizations...")
        self.plot_portfolio_comparison(model_portfolios)
        self.plot_drawdown_comparison(model_portfolios)
        self.plot_returns_distribution(model_returns)
        self.plot_risk_return_scatter(model_results)
        self.plot_performance_metrics_radar(model_results)

        # 3. Statistical significance tests
        print("\n📊 Performing statistical tests...")
        model_names = list(model_returns.keys())
        if len(model_names) >= 2:
            test_results = self.statistical_significance_test(
                model_returns[model_names[0]],
                model_returns[model_names[1]]
            )
            print(f"\n{test_results['test']} between {model_names[0]} and {model_names[1]}:")
            print(f"  p-value: {test_results['p_value']:.6f}")
            print(f"  Significant at α=0.05: {test_results['significant_at_0.05']}")

        # 4. Generate summary report
        report_path = self.results_dir / "ANALYSIS_REPORT.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RL TRADING MODEL ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("MODEL COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            f.write("BEST PERFORMERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Highest Sharpe Ratio: {comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Model']}\n")
            f.write(f"Highest Total Return: {comparison_df.loc[comparison_df['Total Return (%)'].idxmax(), 'Model']}\n")
            f.write(f"Lowest Drawdown: {comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmax(), 'Model']}\n")
            f.write(f"Highest Win Rate: {comparison_df.loc[comparison_df['Win Rate (%)'].idxmax(), 'Model']}\n")

        print(f"\n✅ Analysis complete! Results saved to: {self.results_dir}")
        print(f"   📁 Figures: {self.results_dir / 'figures'}")
        print(f"   📄 LaTeX tables: {self.results_dir / 'latex'}")
        print(f"   📊 Data: {self.results_dir / 'data'}")
        print(f"   📝 Report: {report_path}")


def load_model_results(model_name: str, results_file: str = "results/test_results.json") -> Dict:
    """
    Load model results from saved JSON file

    Args:
        model_name: Name of the model
        results_file: Path to results JSON file

    Returns:
        Dictionary with model metrics
    """
    try:
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        return all_results.get(model_name, {})
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return {}
