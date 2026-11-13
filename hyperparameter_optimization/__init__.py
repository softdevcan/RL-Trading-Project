"""
Hyperparameter Optimization Package

Akademik seviyede hiper parametre optimizasyonu sistemi.
Optuna tabanlı Bayesian optimization ile 4 farklı RL algoritmasını optimize eder.

Quick Start:
    >>> from hyperparameter_optimization import PPOOptimizer
    >>> optimizer = PPOOptimizer(n_trials=50)
    >>> study = optimizer.optimize(
    ...     stock_symbols=None,   # RL panelinin tum sembolleri (rotanin evreni)
    ...     fold='2021',          # evaluation/splits.py wf-v1 kati (Faz 9 R1)
    ...     total_timesteps=100_000
    ... )
    >>> optimizer.save_best_params()
"""

from .optimizers import (
    PPOOptimizer,
    A2COptimizer,
    TD3Optimizer,
    SACOptimizer,
)

from .search_spaces import (
    get_search_space,
    get_default_hyperparameters,
    print_search_space_info,
    SearchSpaceConfig,
)

__version__ = "1.0.0"

__all__ = [
    # Optimizers
    'PPOOptimizer',
    'A2COptimizer',
    'TD3Optimizer',
    'SACOptimizer',

    # Search Space Utilities
    'get_search_space',
    'get_default_hyperparameters',
    'print_search_space_info',
    'SearchSpaceConfig',
]
