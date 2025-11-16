"""
Hyperparameter Optimization Results Analysis

Optuna study sonuçlarını analiz eder ve akademik seviyede görselleştirmeler oluşturur.

Kullanım Örnekleri:

    # 1. Tek bir study'yi analiz etme
    python analyze_results.py --study-name ppo_optimization_20240115_143022

    # 2. Tüm studies'i karşılaştırma
    python analyze_results.py --compare-all

    # 3. Belirli studies'i karşılaştırma
    python analyze_results.py --compare ppo_study_1 a2c_study_1 td3_study_1

    # 4. LaTeX tablo oluşturma
    python analyze_results.py --study-name sac_study --latex
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_contour,
    plot_parallel_coordinate,
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class HyperparameterAnalyzer:
    """
    Hiper parametre optimizasyon sonuçlarını analiz eden sınıf.
    """

    def __init__(
        self,
        storage: str = "sqlite:///results/hyperparameter_studies/optuna_studies.db",
        output_dir: str = "results/hyperparameter_studies"
    ):
        """
        Args:
            storage: Optuna storage path
            output_dir: Output directory for plots and reports
        """
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        self.plots_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

    def load_study(self, study_name: str) -> optuna.Study:
        """Optuna study'yi yükler."""
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage
            )
            print(f"✅ Study loaded: {study_name}")
            print(f"   Trials: {len(study.trials)}")
            print(f"   Best value: {study.best_value:.4f}")
            return study
        except Exception as e:
            print(f"❌ Failed to load study '{study_name}': {e}")
            raise

    def list_all_studies(self) -> List[str]:
        """Tüm study isimlerini listeler."""
        try:
            summaries = optuna.study.get_all_study_summaries(storage=self.storage)
            study_names = [s.study_name for s in summaries]
            print(f"\n📚 Found {len(study_names)} studies:")
            for i, name in enumerate(study_names, 1):
                print(f"   {i}. {name}")
            return study_names
        except Exception as e:
            print(f"❌ Failed to list studies: {e}")
            return []

    def get_study_statistics(self, study: optuna.Study) -> Dict[str, Any]:
        """Study istatistiklerini hesaplar."""
        trials = study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

        if not completed_trials:
            return {}

        values = [t.value for t in completed_trials]

        stats = {
            "study_name": study.study_name,
            "total_trials": len(trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "mean_value": np.mean(values),
            "median_value": np.median(values),
            "std_value": np.std(values),
            "min_value": np.min(values),
            "max_value": np.max(values),
            "best_params": study.best_params,
        }

        return stats

    def plot_optimization_history(self, study: optuna.Study, save: bool = True):
        """Optimization history plot."""
        fig = plot_optimization_history(study)
        fig.update_layout(
            title=f"Optimization History: {study.study_name}",
            xaxis_title="Trial",
            yaxis_title="Objective Value (Sharpe Ratio)",
            font=dict(size=12),
            width=1000,
            height=600,
        )

        if save:
            filepath = self.plots_dir / f"{study.study_name}_optimization_history.html"
            fig.write_html(str(filepath))
            print(f"✅ Saved: {filepath}")

        return fig

    def plot_param_importances(self, study: optuna.Study, save: bool = True):
        """Parameter importance plot."""
        try:
            fig = plot_param_importances(study)
            fig.update_layout(
                title=f"Hyperparameter Importances: {study.study_name}",
                xaxis_title="Importance",
                yaxis_title="Hyperparameter",
                font=dict(size=12),
                width=1000,
                height=600,
            )

            if save:
                filepath = self.plots_dir / f"{study.study_name}_param_importances.html"
                fig.write_html(str(filepath))
                print(f"✅ Saved: {filepath}")

            return fig
        except Exception as e:
            print(f"⚠️  Could not plot param importances: {e}")
            return None

    def plot_parallel_coordinate(self, study: optuna.Study, save: bool = True):
        """Parallel coordinate plot."""
        try:
            fig = plot_parallel_coordinate(study)
            fig.update_layout(
                title=f"Parallel Coordinate Plot: {study.study_name}",
                font=dict(size=12),
                width=1400,
                height=700,
            )

            if save:
                filepath = self.plots_dir / f"{study.study_name}_parallel_coordinate.html"
                fig.write_html(str(filepath))
                print(f"✅ Saved: {filepath}")

            return fig
        except Exception as e:
            print(f"⚠️  Could not plot parallel coordinate: {e}")
            return None

    def plot_contour(self, study: optuna.Study, save: bool = True):
        """Contour plot for parameter interactions."""
        try:
            fig = plot_contour(study)
            fig.update_layout(
                title=f"Parameter Contour Plot: {study.study_name}",
                font=dict(size=12),
                width=1200,
                height=800,
            )

            if save:
                filepath = self.plots_dir / f"{study.study_name}_contour.html"
                fig.write_html(str(filepath))
                print(f"✅ Saved: {filepath}")

            return fig
        except Exception as e:
            print(f"⚠️  Could not plot contour: {e}")
            return None

    def plot_slice(self, study: optuna.Study, save: bool = True):
        """Slice plot for individual parameter effects."""
        try:
            fig = plot_slice(study)
            fig.update_layout(
                title=f"Parameter Slice Plot: {study.study_name}",
                font=dict(size=12),
                width=1400,
                height=1000,
            )

            if save:
                filepath = self.plots_dir / f"{study.study_name}_slice.html"
                fig.write_html(str(filepath))
                print(f"✅ Saved: {filepath}")

            return fig
        except Exception as e:
            print(f"⚠️  Could not plot slice: {e}")
            return None

    def create_comparison_plots(self, studies: Dict[str, optuna.Study], save: bool = True):
        """Birden fazla study'yi karşılaştırır."""
        # Prepare data
        data = []
        for name, study in studies.items():
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    data.append({
                        "Algorithm": name.split("_")[0].upper(),
                        "Trial": trial.number,
                        "Value": trial.value,
                    })

        df = pd.DataFrame(data)

        # 1. Box plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=df, x="Algorithm", y="Value", ax=axes[0])
        axes[0].set_title("Sharpe Ratio Distribution by Algorithm")
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].grid(True, alpha=0.3)

        # 2. Violin plot
        sns.violinplot(data=df, x="Algorithm", y="Value", ax=axes[1])
        axes[1].set_title("Sharpe Ratio Distribution (Violin Plot)")
        axes[1].set_ylabel("Sharpe Ratio")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.plots_dir / "algorithm_comparison_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")

        plt.show()

        # 3. Best value comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        best_values = {name: study.best_value for name, study in studies.items()}
        algorithms = [name.split("_")[0].upper() for name in best_values.keys()]
        values = list(best_values.values())

        bars = ax.bar(algorithms, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithms)])
        ax.set_ylabel("Best Sharpe Ratio")
        ax.set_title("Best Sharpe Ratio by Algorithm")
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        if save:
            filepath = self.plots_dir / "algorithm_comparison_best_values.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filepath}")

        plt.show()

    def generate_latex_table(self, study: optuna.Study, save: bool = True) -> str:
        """LaTeX tablo oluşturur."""
        stats = self.get_study_statistics(study)

        latex = r"""
\begin{table}[htbp]
\centering
\caption{Hyperparameter Optimization Results: """ + study.study_name.replace("_", r"\_") + r"""}
\label{tab:hyperopt_results}
\begin{tabular}{ll}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""

        latex += f"Total Trials & {stats['total_trials']} \\\\\n"
        latex += f"Completed Trials & {stats['completed_trials']} \\\\\n"
        latex += f"Pruned Trials & {stats['pruned_trials']} \\\\\n"
        latex += f"Failed Trials & {stats['failed_trials']} \\\\\n"
        latex += r"\midrule" + "\n"
        latex += f"Best Sharpe Ratio & {stats['best_value']:.4f} \\\\\n"
        latex += f"Mean Sharpe Ratio & {stats['mean_value']:.4f} \\\\\n"
        latex += f"Median Sharpe Ratio & {stats['median_value']:.4f} \\\\\n"
        latex += f"Std Sharpe Ratio & {stats['std_value']:.4f} \\\\\n"
        latex += r"\midrule" + "\n"

        # Best hyperparameters
        latex += r"\multicolumn{2}{l}{\textbf{Best Hyperparameters:}} \\" + "\n"
        for param, value in stats['best_params'].items():
            param_name = param.replace("_", r"\_")
            if isinstance(value, float):
                latex += f"{param_name} & {value:.6f} \\\\\n"
            else:
                latex += f"{param_name} & {value} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

        if save:
            filepath = self.reports_dir / f"{study.study_name}_latex_table.tex"
            with open(filepath, 'w') as f:
                f.write(latex)
            print(f"✅ Saved: {filepath}")

        return latex

    def generate_full_report(self, study: optuna.Study):
        """Tam analiz raporu oluşturur."""
        print(f"\n{'='*80}")
        print(f"📊 Generating Full Analysis Report: {study.study_name}")
        print(f"{'='*80}\n")

        # Statistics
        stats = self.get_study_statistics(study)
        print("\n📈 Study Statistics:")
        for key, value in stats.items():
            if key != "best_params":
                print(f"   {key}: {value}")

        # Plots
        print("\n🎨 Generating Visualizations...")
        self.plot_optimization_history(study)
        self.plot_param_importances(study)
        self.plot_parallel_coordinate(study)
        self.plot_contour(study)
        self.plot_slice(study)

        # LaTeX table
        print("\n📝 Generating LaTeX Table...")
        self.generate_latex_table(study)

        # Save statistics as JSON
        stats_file = self.reports_dir / f"{study.study_name}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"✅ Saved: {stats_file}")

        print(f"\n{'='*80}")
        print(f"✅ Full Report Generated!")
        print(f"{'='*80}")
        print(f"Plots directory: {self.plots_dir}")
        print(f"Reports directory: {self.reports_dir}")
        print(f"{'='*80}\n")


def parse_args():
    """Command line argümanlarını parse eder."""
    parser = argparse.ArgumentParser(
        description="📊 Hyperparameter Optimization Results Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--study-name",
        type=str,
        help="Analiz edilecek study ismi"
    )

    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Karşılaştırılacak study isimleri"
    )

    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Tüm studies'i karşılaştır"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Tüm studies'i listele"
    )

    parser.add_argument(
        "--latex",
        action="store_true",
        help="LaTeX tablo oluştur"
    )

    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///results/hyperparameter_studies/optuna_studies.db",
        help="Optuna storage path"
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    analyzer = HyperparameterAnalyzer(storage=args.storage)

    # List all studies
    if args.list:
        analyzer.list_all_studies()
        return

    # Compare all studies
    if args.compare_all:
        study_names = analyzer.list_all_studies()
        if not study_names:
            print("❌ No studies found!")
            return

        print(f"\n{'='*80}")
        print(f"📊 Comparing All Studies")
        print(f"{'='*80}\n")

        studies = {name: analyzer.load_study(name) for name in study_names}
        analyzer.create_comparison_plots(studies)
        return

    # Compare specific studies
    if args.compare:
        print(f"\n{'='*80}")
        print(f"📊 Comparing Selected Studies")
        print(f"{'='*80}\n")

        studies = {}
        for name in args.compare:
            try:
                studies[name] = analyzer.load_study(name)
            except Exception as e:
                print(f"⚠️  Skipping {name}: {e}")

        if studies:
            analyzer.create_comparison_plots(studies)
        return

    # Single study analysis
    if args.study_name:
        study = analyzer.load_study(args.study_name)
        analyzer.generate_full_report(study)

        if args.latex:
            latex = analyzer.generate_latex_table(study)
            print("\n📝 LaTeX Table:")
            print(latex)
        return

    # No arguments provided
    print("❌ Please specify --study-name, --compare, --compare-all, or --list")
    print("Use --help for more information")


if __name__ == "__main__":
    main()
