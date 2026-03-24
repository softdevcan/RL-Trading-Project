"""
Hyperparameter Search Spaces Configuration

Bu dosyada tüm algoritmaların hiper parametre arama uzaylarını tanımlarsınız.
Her parametreyi kolayca düzenleyebilir, ekleyebilir veya çıkarabilirsiniz.

Kullanım:
    from search_spaces import get_search_space
    space = get_search_space('ppo')  # veya 'a2c', 'td3', 'sac'
"""

from typing import Dict, Any, List, Optional
import optuna


class SearchSpaceConfig:
    """
    Hiper parametre arama uzayı konfigürasyonu.
    Her algoritma için özelleştirilebilir arama uzayları.
    """

    # ==================== PPO (Proximal Policy Optimization) ====================
    PPO_SEARCH_SPACE = {
        # Learning Rate - EN ÖNEMLİ PARAMETRE
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2,
            "default": 3e-4,
            "description": "Adam optimizer learning rate"
        },

        # Steps per update
        "n_steps": {
            "type": "categorical",
            "choices": [512, 1024, 2048, 4096],
            "default": 2048,
            "description": "Number of steps to run for each environment per update"
        },

        # Batch size
        "batch_size": {
            "type": "categorical",
            "choices": [32, 64, 128, 256],
            "default": 64,
            "description": "Minibatch size for SGD"
        },

        # Training epochs
        "n_epochs": {
            "type": "int",
            "low": 5,
            "high": 20,
            "default": 10,
            "description": "Number of epochs when optimizing the surrogate loss"
        },

        # Discount factor
        "gamma": {
            "type": "uniform",
            "low": 0.95,
            "high": 0.999,
            "default": 0.99,
            "description": "Discount factor for future rewards"
        },

        # GAE lambda
        "gae_lambda": {
            "type": "uniform",
            "low": 0.8,
            "high": 0.99,
            "default": 0.95,
            "description": "Factor for trade-off of bias vs variance for GAE"
        },

        # Clipping parameter
        "clip_range": {
            "type": "uniform",
            "low": 0.1,
            "high": 0.4,
            "default": 0.2,
            "description": "Clipping parameter for PPO"
        },

        # Entropy coefficient
        "ent_coef": {
            "type": "loguniform",
            "low": 1e-8,
            "high": 0.3,
            "default": 0.15,
            "description": "Entropy coefficient for exploration"
        },

        # Value function coefficient
        "vf_coef": {
            "type": "uniform",
            "low": 0.1,
            "high": 1.0,
            "default": 0.5,
            "description": "Value function coefficient for loss"
        },

        # Max gradient norm
        "max_grad_norm": {
            "type": "uniform",
            "low": 0.3,
            "high": 5.0,
            "default": 0.5,
            "description": "Maximum value for gradient clipping"
        },

        # Network architecture
        "net_arch_size": {
            "type": "categorical",
            "choices": ["small", "medium", "large"],
            "default": "medium",
            "description": "Neural network size (small: [64,64], medium: [256,256], large: [400,300])"
        }
    }

    # ==================== A2C (Advantage Actor Critic) ====================
    A2C_SEARCH_SPACE = {
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2,
            "default": 7e-4,
            "description": "RMSprop learning rate"
        },

        "n_steps": {
            "type": "categorical",
            "choices": [8, 16, 32, 64, 128, 256],
            "default": 256,
            "description": "Number of steps for each update"
        },

        "gamma": {
            "type": "uniform",
            "low": 0.95,
            "high": 0.999,
            "default": 0.99,
            "description": "Discount factor"
        },

        "gae_lambda": {
            "type": "uniform",
            "low": 0.8,
            "high": 0.99,
            "default": 0.95,
            "description": "GAE lambda parameter"
        },

        "ent_coef": {
            "type": "loguniform",
            "low": 1e-8,
            "high": 0.1,
            "default": 0.01,
            "description": "Entropy coefficient"
        },

        "vf_coef": {
            "type": "uniform",
            "low": 0.1,
            "high": 1.0,
            "default": 0.25,
            "description": "Value function coefficient"
        },

        "max_grad_norm": {
            "type": "uniform",
            "low": 0.3,
            "high": 5.0,
            "default": 0.5,
            "description": "Max gradient norm"
        },

        "rms_prop_eps": {
            "type": "loguniform",
            "low": 1e-8,
            "high": 1e-4,
            "default": 1e-5,
            "description": "RMSprop epsilon for numerical stability"
        },

        "net_arch_size": {
            "type": "categorical",
            "choices": ["small", "medium", "large"],
            "default": "medium",
            "description": "Neural network size"
        }
    }

    # ==================== TD3 (Twin Delayed DDPG) ====================
    TD3_SEARCH_SPACE = {
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2,
            "default": 1e-3,
            "description": "Learning rate for Adam optimizer"
        },

        "buffer_size": {
            "type": "categorical",
            "choices": [50000, 100000, 200000, 500000],
            "default": 100000,
            "description": "Replay buffer size"
        },

        "learning_starts": {
            "type": "int",
            "low": 500,  # Reduced from 1000 for faster hyperopt
            "high": 3000,  # Reduced from 10000 for faster trials
            "default": 1000,
            "description": "Steps before learning starts"
        },

        "batch_size": {
            "type": "categorical",
            "choices": [64, 128, 256, 512],
            "default": 256,
            "description": "Batch size for replay buffer sampling"
        },

        "tau": {
            "type": "uniform",
            "low": 0.001,
            "high": 0.02,
            "default": 0.005,
            "description": "Soft update coefficient for target networks"
        },

        "gamma": {
            "type": "uniform",
            "low": 0.95,
            "high": 0.999,
            "default": 0.99,
            "description": "Discount factor"
        },

        "action_noise_sigma": {
            "type": "uniform",
            "low": 0.05,
            "high": 0.5,
            "default": 0.2,
            "description": "Standard deviation of Gaussian action noise"
        },

        "target_policy_noise": {
            "type": "uniform",
            "low": 0.1,
            "high": 0.5,
            "default": 0.3,
            "description": "Target policy noise for smoothing"
        },

        "target_noise_clip": {
            "type": "uniform",
            "low": 0.3,
            "high": 1.0,
            "default": 0.5,
            "description": "Target noise clipping range"
        },

        "policy_delay": {
            "type": "int",
            "low": 1,
            "high": 4,
            "default": 2,
            "description": "Policy update frequency (delayed)"
        },

        "net_arch_size": {
            "type": "categorical",
            "choices": ["small", "medium", "large"],
            "default": "medium",
            "description": "Neural network size"
        }
    }

    # ==================== SAC (Soft Actor-Critic) ====================
    SAC_SEARCH_SPACE = {
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2,
            "default": 3e-4,
            "description": "Learning rate for all optimizers"
        },

        "buffer_size": {
            "type": "categorical",
            "choices": [50000, 100000, 200000, 500000],
            "default": 100000,
            "description": "Replay buffer size"
        },

        "learning_starts": {
            "type": "int",
            "low": 500,  # Reduced from 1000 for faster hyperopt
            "high": 3000,  # Reduced from 10000 for faster trials
            "default": 1000,
            "description": "Steps before learning starts"
        },

        "batch_size": {
            "type": "categorical",
            "choices": [64, 128, 256, 512],
            "default": 256,
            "description": "Batch size"
        },

        "tau": {
            "type": "uniform",
            "low": 0.001,
            "high": 0.02,
            "default": 0.005,
            "description": "Soft update coefficient"
        },

        "gamma": {
            "type": "uniform",
            "low": 0.95,
            "high": 0.999,
            "default": 0.99,
            "description": "Discount factor"
        },

        "ent_coef": {
            "type": "categorical",
            "choices": ["auto", "auto_0.1", "auto_0.5", "auto_1.0"],
            "default": "auto_0.5",
            "description": "Entropy coefficient (auto for automatic tuning)"
        },

        "target_update_interval": {
            "type": "int",
            "low": 1,
            "high": 10,
            "default": 1,
            "description": "Target network update frequency"
        },

        "train_freq": {
            "type": "int",
            "low": 1,
            "high": 10,
            "default": 1,
            "description": "Training frequency (steps)"
        },

        "gradient_steps": {
            "type": "int",
            "low": 1,
            "high": 10,
            "default": 1,
            "description": "Gradient steps per update"
        },

        "net_arch_size": {
            "type": "categorical",
            "choices": ["small", "medium", "large"],
            "default": "medium",
            "description": "Neural network size"
        }
    }

    # Network architecture mapping
    NETWORK_ARCHITECTURES = {
        "small": {"pi": [64, 64], "vf": [64, 64]},
        "medium": {"pi": [256, 256], "vf": [256, 256]},
        "large": {"pi": [400, 300], "vf": [400, 300]}
    }

    @classmethod
    def get_network_arch(cls, size: str) -> Dict[str, List[int]]:
        """Network architecture'ı döndürür."""
        return cls.NETWORK_ARCHITECTURES.get(size, cls.NETWORK_ARCHITECTURES["medium"])


def suggest_hyperparameter(trial: optuna.Trial, param_name: str, param_config: Dict[str, Any]) -> Any:
    """
    Optuna trial'dan hiper parametre önerir.

    Args:
        trial: Optuna trial objesi
        param_name: Parametre ismi
        param_config: Parametre konfigürasyonu

    Returns:
        Önerilen parametre değeri
    """
    param_type = param_config["type"]

    if param_type == "loguniform":
        return trial.suggest_float(
            param_name,
            param_config["low"],
            param_config["high"],
            log=True
        )

    elif param_type == "uniform":
        return trial.suggest_float(
            param_name,
            param_config["low"],
            param_config["high"]
        )

    elif param_type == "int":
        return trial.suggest_int(
            param_name,
            param_config["low"],
            param_config["high"]
        )

    elif param_type == "categorical":
        return trial.suggest_categorical(
            param_name,
            param_config["choices"]
        )

    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def get_search_space(algorithm: str) -> Dict[str, Dict[str, Any]]:
    """
    Algoritma için arama uzayını döndürür.

    Args:
        algorithm: 'ppo', 'a2c', 'td3', veya 'sac'

    Returns:
        Arama uzayı dictionary'si

    Example:
        >>> space = get_search_space('ppo')
        >>> print(space['learning_rate']['default'])
        0.0003
    """
    algorithm = algorithm.lower()

    if algorithm == "ppo":
        return SearchSpaceConfig.PPO_SEARCH_SPACE
    elif algorithm == "a2c":
        return SearchSpaceConfig.A2C_SEARCH_SPACE
    elif algorithm == "td3":
        return SearchSpaceConfig.TD3_SEARCH_SPACE
    elif algorithm == "sac":
        return SearchSpaceConfig.SAC_SEARCH_SPACE
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: ppo, a2c, td3, sac")


def get_default_hyperparameters(algorithm: str) -> Dict[str, Any]:
    """
    Algoritmanın varsayılan hiper parametrelerini döndürür.

    Args:
        algorithm: 'ppo', 'a2c', 'td3', veya 'sac'

    Returns:
        Varsayılan parametreler dictionary'si
    """
    search_space = get_search_space(algorithm)
    return {param_name: config["default"] for param_name, config in search_space.items()}


def print_search_space_info(algorithm: str):
    """
    Arama uzayı bilgilerini yazdırır.

    Args:
        algorithm: 'ppo', 'a2c', 'td3', veya 'sac'
    """
    search_space = get_search_space(algorithm)

    print(f"\n{'='*80}")
    print(f"Search Space for {algorithm.upper()}")
    print(f"{'='*80}\n")

    for param_name, config in search_space.items():
        print(f"📌 {param_name}")
        print(f"   Type: {config['type']}")
        print(f"   Default: {config['default']}")

        if config['type'] in ['loguniform', 'uniform']:
            print(f"   Range: [{config['low']}, {config['high']}]")
        elif config['type'] == 'int':
            print(f"   Range: [{config['low']}, {config['high']}]")
        elif config['type'] == 'categorical':
            print(f"   Choices: {config['choices']}")

        print(f"   Description: {config['description']}")
        print()


if __name__ == "__main__":
    # Test ve örnek kullanım
    print("🔬 Hyperparameter Search Spaces\n")

    algorithms = ["ppo", "a2c", "td3", "sac"]

    for algo in algorithms:
        print_search_space_info(algo)
        defaults = get_default_hyperparameters(algo)
        print(f"✅ Varsayılan parametreler yüklendi: {len(defaults)} parametre\n")
