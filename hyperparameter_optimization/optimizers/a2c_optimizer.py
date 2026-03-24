"""
A2C (Advantage Actor-Critic) Hyperparameter Optimizer
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import optuna
from stable_baselines3 import A2C

# Suppress GPU warning for MLP policies (we know GPU still provides speedup)
warnings.filterwarnings('ignore', category=UserWarning, message='.*GPU.*MlpPolicy.*')

sys.path.append(str(Path(__file__).parent.parent.parent))
from hyperparameter_optimization.base_optimizer import BaseHyperparameterOptimizer
from hyperparameter_optimization.search_spaces import SearchSpaceConfig


class A2COptimizer(BaseHyperparameterOptimizer):
    """
    A2C algoritması için hiper parametre optimizer.

    Örnek Kullanım:
        >>> optimizer = A2COptimizer(n_trials=50, n_jobs=1)
        >>> study = optimizer.optimize(
        ...     stock_symbols=['THYAO.IS', 'SAHOL.IS'],
        ...     train_start='2018-01-01',
        ...     train_end='2022-12-31',
        ...     val_start='2023-01-01',
        ...     val_end='2023-12-31',
        ...     total_timesteps=100_000
        ... )
        >>> optimizer.save_best_params()
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: BaseHyperparameterOptimizer argümanları
        """
        super().__init__(algorithm_name="a2c", **kwargs)

    def create_model(
        self,
        trial: optuna.Trial,
        env: gym.Env,
        hyperparams: Dict[str, Any]
    ) -> A2C:
        """
        A2C model oluşturur.

        Args:
            trial: Optuna trial
            env: Training environment
            hyperparams: Hiper parametreler

        Returns:
            A2C model instance
        """
        # Network architecture
        net_arch_size = hyperparams.pop("net_arch_size", "medium")
        net_arch = SearchSpaceConfig.get_network_arch(net_arch_size)

        # Policy kwargs
        policy_kwargs = dict(
            net_arch=net_arch
        )

        # Create A2C model with GPU support
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            gamma=hyperparams["gamma"],
            gae_lambda=hyperparams["gae_lambda"],
            ent_coef=hyperparams["ent_coef"],
            vf_coef=hyperparams["vf_coef"],
            max_grad_norm=hyperparams["max_grad_norm"],
            rms_prop_eps=hyperparams["rms_prop_eps"],
            use_rms_prop=True,  # A2C paper'daki gibi
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=self.seed,
            device="auto",  # Automatically use GPU if available
            tensorboard_log=f"logs/tensorboard/{self.algorithm_name}_trial_{trial.number}",
        )

        return model


if __name__ == "__main__":
    # Test
    print("🔬 Testing A2C Optimizer\n")

    optimizer = A2COptimizer(
        n_trials=2,
        n_jobs=1,
        seed=42
    )

    print(f"✅ A2C Optimizer created")
    print(f"   Search space: {len(optimizer.search_space)} parameters")
    print(f"   Study name: {optimizer.study_name}")
