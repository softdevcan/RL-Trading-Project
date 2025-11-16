"""
SAC (Soft Actor-Critic) Hyperparameter Optimizer
"""

import sys
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import optuna
from stable_baselines3 import SAC

sys.path.append(str(Path(__file__).parent.parent.parent))
from hyperparameter_optimization.base_optimizer import BaseHyperparameterOptimizer
from hyperparameter_optimization.search_spaces import SearchSpaceConfig


class SACOptimizer(BaseHyperparameterOptimizer):
    """
    SAC algoritması için hiper parametre optimizer.

    Örnek Kullanım:
        >>> optimizer = SACOptimizer(n_trials=50, n_jobs=1)
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
        super().__init__(algorithm_name="sac", **kwargs)

    def create_model(
        self,
        trial: optuna.Trial,
        env: gym.Env,
        hyperparams: Dict[str, Any]
    ) -> SAC:
        """
        SAC model oluşturur.

        Args:
            trial: Optuna trial
            env: Training environment
            hyperparams: Hiper parametreler

        Returns:
            SAC model instance
        """
        # Network architecture
        net_arch_size = hyperparams.pop("net_arch_size", "medium")
        net_arch_dict = SearchSpaceConfig.get_network_arch(net_arch_size)
        net_arch = net_arch_dict["pi"]  # SAC için actor network

        # Policy kwargs
        policy_kwargs = dict(
            net_arch=net_arch
        )

        # Create SAC model with GPU support
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams["learning_rate"],
            buffer_size=hyperparams["buffer_size"],
            learning_starts=hyperparams["learning_starts"],
            batch_size=hyperparams["batch_size"],
            tau=hyperparams["tau"],
            gamma=hyperparams["gamma"],
            ent_coef=hyperparams["ent_coef"],
            target_update_interval=hyperparams["target_update_interval"],
            train_freq=hyperparams["train_freq"],
            gradient_steps=hyperparams["gradient_steps"],
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=self.seed,
            device="auto",  # Automatically use GPU if available
            tensorboard_log=f"logs/tensorboard/{self.algorithm_name}_trial_{trial.number}",
        )

        return model


if __name__ == "__main__":
    # Test
    print("🔬 Testing SAC Optimizer\n")

    optimizer = SACOptimizer(
        n_trials=2,
        n_jobs=1,
        seed=42
    )

    print(f"✅ SAC Optimizer created")
    print(f"   Search space: {len(optimizer.search_space)} parameters")
    print(f"   Study name: {optimizer.study_name}")
