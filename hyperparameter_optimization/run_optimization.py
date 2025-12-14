"""
Main Hyperparameter Optimization Runner

Tüm algoritmaları veya belirli bir algoritmayı optimize etmek için ana script.

Kullanım Örnekleri:

    # 1. Tek algoritma optimize etme
    python run_optimization.py --algorithm ppo --trials 50 --timesteps 100000

    # 2. Tüm algoritmaları optimize etme
    python run_optimization.py --algorithm all --trials 30

    # 3. Özel tarih aralığı ile
    python run_optimization.py --algorithm sac --trials 20 \
        --train-start 2018-01-01 --train-end 2022-12-31 \
        --val-start 2023-01-01 --val-end 2023-12-31

    # 4. Paralel çalıştırma (dikkat: GPU memory!)
    python run_optimization.py --algorithm ppo --trials 100 --jobs 2

    # 5. Kısa test (sadece 2 trial, 10k timesteps)
    python run_optimization.py --algorithm ppo --trials 2 --timesteps 10000
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from hyperparameter_optimization.optimizers import (
    PPOOptimizer,
    A2COptimizer,
    TD3Optimizer,
    SACOptimizer
)
from data.bist30_symbols import PHASE1_SYMBOLS


def parse_args():
    """Command line argümanlarını parse eder."""
    parser = argparse.ArgumentParser(
        description="🔬 RL Trading Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "a2c", "td3", "sac", "all"],
        default="ppo",
        help="Optimize edilecek algoritma (default: ppo)"
    )

    # Optimization parameters
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Toplam trial sayısı (default: 50)"
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Paralel job sayısı (default: 1, dikkat: GPU memory!)"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Her trial için training timesteps (default: 100000)"
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluation frequency (default: 5000)"
    )

    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Evaluation episode sayısı (default: 5)"
    )

    # Stock symbols
    parser.add_argument(
        "--stocks",
        type=str,
        nargs="+",
        default=PHASE1_SYMBOLS,  # AKBNK.IS, THYAO.IS, TUPRS.IS, BIMAS.IS, ASELS.IS
        help="Hisse senedi sembolleri (default: PHASE1_SYMBOLS - AKBNK, THYAO, TUPRS, BIMAS, ASELS)"
    )

    # Date ranges
    parser.add_argument(
        "--train-start",
        type=str,
        default="2018-01-01",
        help="Training başlangıç tarihi (default: 2018-01-01)"
    )

    parser.add_argument(
        "--train-end",
        type=str,
        default="2022-12-31",
        help="Training bitiş tarihi (default: 2022-12-31)"
    )

    parser.add_argument(
        "--val-start",
        type=str,
        default="2023-01-01",
        help="Validation başlangıç tarihi (default: 2023-01-01)"
    )

    parser.add_argument(
        "--val-end",
        type=str,
        default="2023-12-31",
        help="Validation bitiş tarihi (default: 2023-12-31)"
    )

    # Study configuration
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study ismi (None ise otomatik oluşturulur)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Progress bar gösterme"
    )

    # Phase 2 support
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=2,
        help="Trading phase (1=56 features, 2=97 features with fundamental+macro, default: 2)"
    )

    parser.add_argument(
        "--reward-type",
        type=str,
        choices=["simple", "psr"],
        default="psr",
        help="Reward function type (simple=baseline, psr=risk-aware, default: psr)"
    )

    return parser.parse_args()


def get_optimizer(algorithm: str, args):
    """
    Algoritma için optimizer döndürür.

    Args:
        algorithm: 'ppo', 'a2c', 'td3', veya 'sac'
        args: Command line arguments

    Returns:
        Optimizer instance
    """
    optimizer_map = {
        "ppo": PPOOptimizer,
        "a2c": A2COptimizer,
        "td3": TD3Optimizer,
        "sac": SACOptimizer,
    }

    optimizer_class = optimizer_map[algorithm.lower()]

    return optimizer_class(
        study_name=args.study_name,
        n_trials=args.trials,
        n_jobs=args.jobs,
        seed=args.seed,
        phase=args.phase,
        reward_type=args.reward_type
    )


def run_single_optimization(algorithm: str, args):
    """
    Tek bir algoritma için optimizasyon çalıştırır.

    Args:
        algorithm: 'ppo', 'a2c', 'td3', veya 'sac'
        args: Command line arguments
    """
    print(f"\n{'='*100}")
    print(f"🚀 Starting Hyperparameter Optimization: {algorithm.upper()}")
    print(f"{'='*100}")
    print(f"Configuration:")
    print(f"  Stocks: {args.stocks}")
    print(f"  Train Period: {args.train_start} to {args.train_end}")
    print(f"  Val Period: {args.val_start} to {args.val_end}")
    print(f"  Phase: {args.phase} ({'56 features' if args.phase == 1 else '97 features (Fundamental + Macro)'})")
    print(f"  Reward Type: {args.reward_type.upper()}")
    print(f"  Trials: {args.trials}")
    print(f"  Jobs: {args.jobs}")
    print(f"  Timesteps per trial: {args.timesteps:,}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*100}\n")

    # Create optimizer
    optimizer = get_optimizer(algorithm, args)

    # Run optimization
    study = optimizer.optimize(
        stock_symbols=args.stocks,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        show_progress_bar=not args.no_progress_bar
    )

    # Save best parameters
    optimizer.save_best_params()

    print(f"\n{'='*100}")
    print(f"✅ {algorithm.upper()} Optimization Complete!")
    print(f"{'='*100}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Sharpe Ratio: {study.best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nTotal Trials Completed: {len(study.trials)}")
    print(f"Pruned Trials: {len([t for t in study.trials if t.state.name == 'PRUNED'])}")
    print(f"Failed Trials: {len([t for t in study.trials if t.state.name == 'FAIL'])}")
    print(f"{'='*100}\n")

    return study


def run_all_optimizations(args):
    """
    Tüm algoritmaları optimize eder.

    Args:
        args: Command line arguments

    Returns:
        Dictionary of studies
    """
    algorithms = ["ppo", "a2c", "td3", "sac"]
    studies = {}

    print(f"\n{'='*100}")
    print(f"🚀 Starting Hyperparameter Optimization: ALL ALGORITHMS")
    print(f"{'='*100}")
    print(f"Algorithms to optimize: {', '.join([a.upper() for a in algorithms])}")
    print(f"This will take a long time! ☕️")
    print(f"{'='*100}\n")

    for i, algorithm in enumerate(algorithms, 1):
        print(f"\n\n{'#'*100}")
        print(f"# Algorithm {i}/{len(algorithms)}: {algorithm.upper()}")
        print(f"{'#'*100}\n")

        study = run_single_optimization(algorithm, args)
        studies[algorithm] = study

    # Summary
    print(f"\n\n{'='*100}")
    print(f"🎉 ALL OPTIMIZATIONS COMPLETE!")
    print(f"{'='*100}")
    print(f"\nSummary of Best Results:\n")

    for algorithm, study in studies.items():
        print(f"  {algorithm.upper():5s} - Best Sharpe Ratio: {study.best_value:.4f} (Trial {study.best_trial.number})")

    print(f"\n{'='*100}\n")

    return studies


def main():
    """Main execution."""
    args = parse_args()

    # Print header
    print("\n" + "="*100)
    print(" " * 30 + "🔬 RL TRADING HYPERPARAMETER OPTIMIZATION")
    print("="*100 + "\n")

    # Run optimization
    if args.algorithm == "all":
        studies = run_all_optimizations(args)
    else:
        study = run_single_optimization(args.algorithm, args)
        studies = {args.algorithm: study}

    # Final message
    print("\n" + "="*100)
    print("✅ Optimization pipeline completed successfully!")
    print("="*100)
    print("\nNext steps:")
    print("  1. Analyze results: python analyze_results.py")
    print("  2. View Optuna dashboard: optuna-dashboard sqlite:///results/hyperparameter_studies/optuna_studies.db")
    print("  3. Train final model with best params")
    print("="*100 + "\n")

    return studies


if __name__ == "__main__":
    main()
