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

    # Feature-selection disk cache (2.4 · B5): icerik-hash'li cache dizini.
    # Bos string = cache kapali (varsayilan, davranis degismez). MI/permutation
    # tekrar hesabini onler. Ornek deger: "results/feature_selection_cache".
    FEATURE_SELECTION_CACHE_DIR: str = ""

    # --- Faz 7: Kimlik dogrulama, yetkilendirme, kullanici calisma alanlari ---

    # Ana anahtar. False = eski davranis (herkese acik) — sadece yerel
    # gelistirme icin. Sunucuda ASLA False birakilmaz.
    AUTH_ENABLED: bool = True

    # Kullanici deposu. SQLite tek dosya; Docker volume'una bind edilir.
    # Postgres'e gecis: sadece bu URL degisir.
    AUTH_DB_URL: str = "sqlite:///data/auth/auth.db"

    # JWT imzalama. Bos birakilirsa DEBUG modda gecici anahtar dosyaya yazilir;
    # DEBUG=False iken bos olursa uygulama acilista hata verir (fail-fast).
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14

    # Cerezler. Dash WSGI altinda calistigi icin oturum cerez tabanli
    # (tarayici callback'leri Authorization header tasiyamaz).
    SESSION_COOKIE_NAME: str = "rlt_session"
    REFRESH_COOKIE_NAME: str = "rlt_refresh"
    CSRF_COOKIE_NAME: str = "rlt_csrf"
    COOKIE_SECURE: bool = False   # HTTPS arkasinda True yapilmali
    COOKIE_SAMESITE: str = "lax"  # Dash POST'lari icin lax yeterli, CSRF'i keser
    COOKIE_DOMAIN: str = ""

    # Ilk admin (bootstrap). Yalnizca hic kullanici yokken kullanilir;
    # kullanici olustuktan sonra bu degerler yok sayilir.
    BOOTSTRAP_ADMIN_EMAIL: str = ""
    BOOTSTRAP_ADMIN_PASSWORD: str = ""

    # Brute-force korumasi
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15
    PASSWORD_MIN_LENGTH: int = 10

    # Kullanici bazli calisma alani (hibrit izolasyon):
    # piyasa verisi (bist/macro/fundamental/gold) ORTAK kalir; model, sonuc,
    # tahmin ve gunluk karar dosyalari kullanici basina ayrisir.
    WORKSPACES_DIR: str = "workspaces"
    WORKSPACE_ISOLATION: bool = True
    # Eski (kullanici oncesi) models/ ve results/ iceriginin herkese salt-okunur
    # gosterilmesi. False = sadece kendi calisma alanini gorur.
    WORKSPACE_SHOW_LEGACY: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
