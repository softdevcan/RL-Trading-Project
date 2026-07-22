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

    # Data subdirectories
    BIST_DIR: str = "data/bist"
    GOLD_DIR: str = "data/gold"
    MACRO_DIR: str = "data/macro"
    FUNDAMENTAL_DIR: str = "data/fundamental"

    # --- Faz 6: Backend performans & egitim throughput ayarlari ---
    # Tumu env uzerinden override edilebilir; kod icinde hardcode edilmez.

    # Determinizm (G.5): tum tahmin modeli random_state'inin tek kaynagi.
    PREDICTION_SEED: int = 42

    # Paralel veri cekme (1.1): yfinance rate-limit'e saygili ThreadPool.
    # 1 = eski seri yol (byte-eş). Uygulama env: DATA_FETCH_WORKERS.
    DATA_FETCH_WORKERS: int = 8

    # In-process fetch cache sinir (1.2 · B8): batch egitimde sinirsiz
    # buyumeyi onler. 0 = sinirsiz (eski davranis). LRU tavani.
    DATA_CACHE_MAXSIZE: int = 64

    # Ensemble warm-start (2.1 · B2): %80 yeniden-egitim turu %60 modelinden
    # baslasin mi. Varsayilan False — mevcut modeller/golden bozulmaz, once
    # A/B olcum. Opt-in.
    ENSEMBLE_WARM_START: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
