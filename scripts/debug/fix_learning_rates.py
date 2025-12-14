"""
Fix hyperparameters in existing metrics JSON files
Loads hyperparameter study files and updates metrics with correct values
"""

import json
import os
from pathlib import Path

# Default learning rates for each algorithm (fallback if no study file)
ALGORITHM_LR = {
    "PPO": 0.0003,
    "A2C": 0.0007,
    "TD3": 0.001,
    "SAC": 0.0003
}

def load_hyperparameter_study(algorithm):
    """Load the best hyperparameter study for an algorithm"""
    studies_dir = Path("results/hyperparameter_studies")

    if not studies_dir.exists():
        return None

    # Find study files for this algorithm
    study_files = list(studies_dir.glob(f"best_params_{algorithm.lower()}_*.json"))

    if not study_files:
        return None

    # Use the most recent one
    latest_study = max(study_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest_study, 'r') as f:
            study_data = json.load(f)
        return study_data.get("best_params", {})
    except:
        return None

def fix_learning_rates():
    """Fix hyperparameters in all metrics JSON files"""
    results_dir = Path("results")

    if not results_dir.exists():
        print("[ERROR] Results directory not found!")
        return

    metrics_files = list(results_dir.glob("*_metrics.json"))

    if not metrics_files:
        print("[INFO] No metrics files found!")
        return

    print(f"Found {len(metrics_files)} metrics files")
    print("=" * 80)

    fixed_count = 0

    for metrics_file in metrics_files:
        try:
            # Read current metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            algorithm = metrics.get("algorithm", "").upper()
            current_lr = metrics.get("learning_rate")

            if algorithm in ALGORITHM_LR:
                # Try to load hyperparameter study
                hyperparams = load_hyperparameter_study(algorithm)

                if hyperparams:
                    # Use hyperparameters from study
                    correct_lr = hyperparams.get('learning_rate', ALGORITHM_LR[algorithm])

                    print(f"\nFile: {metrics_file.name}")
                    print(f"  Algorithm: {algorithm}")
                    print(f"  Current LR: {current_lr}")
                    print(f"  Study LR: {correct_lr}")

                    # Update with all hyperparameters from study
                    metrics["learning_rate"] = correct_lr
                    metrics["requested_learning_rate"] = current_lr
                    metrics["hyperparameters_used"] = hyperparams
                    metrics["hyperparameter_study"] = f"best_params_{algorithm.lower()}_*.json (auto-detected)"

                    # Save updated metrics
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)

                    print(f"  [FIXED with study hyperparameters]")
                    fixed_count += 1
                else:
                    # Use default LR
                    correct_lr = ALGORITHM_LR[algorithm]

                    if current_lr != correct_lr:
                        print(f"\nFile: {metrics_file.name}")
                        print(f"  Algorithm: {algorithm}")
                        print(f"  Current LR: {current_lr}")
                        print(f"  Default LR: {correct_lr}")

                        metrics["learning_rate"] = correct_lr
                        metrics["requested_learning_rate"] = current_lr
                        metrics["hyperparameters_used"] = {}

                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)

                        print(f"  [FIXED with default LR]")
                        fixed_count += 1
                    else:
                        print(f"[OK] {metrics_file.name}: {algorithm} LR is correct ({correct_lr})")
            else:
                print(f"[SKIP] {metrics_file.name}: Unknown algorithm '{algorithm}'")

        except Exception as e:
            print(f"[ERROR] Failed to process {metrics_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Fixed {fixed_count} out of {len(metrics_files)} files")
    print("=" * 80)

if __name__ == "__main__":
    fix_learning_rates()
