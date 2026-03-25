"""Application configuration"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_TITLE: str = "RL Trading API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for Reinforcement Learning Trading System"

    # Server Settings
    HOST: str = "localhost"
    PORT: int = 8888
    DEBUG: bool = True

    # CORS Settings
    CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000",
                                "http://localhost:8888", "http://127.0.0.1:8888"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]

    # External API Keys
    EVDS_API_KEY: str = ""

    # Directory paths (#27)
    MODELS_DIR: str = "models"
    RESULTS_DIR: str = "results"
    DATA_DIR: str = "data"
    LOGS_DIR: str = "logs"
    HYPEROPT_DIR: str = "hyperparameter_optimization/results"
    PREDICTIONS_DIR: str = "data/predictions"
    PREDICTION_MODELS_DIR: str = "models/prediction"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
