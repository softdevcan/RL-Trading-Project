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
    HOST: str = "0.0.0.0"
    PORT: int = 8888
    DEBUG: bool = True

    # CORS Settings
    CORS_ORIGINS: list[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
