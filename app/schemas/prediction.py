"""
Prediction Schemas
Tahmin modülü için Pydantic request/response modelleri
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class FeatureGroupsConfig(BaseModel):
    """Egitim sirasinda kullanilacak feature grup konfigurasyonu."""
    enabled_groups: Optional[List[str]] = Field(
        default=None,
        description=(
            "Aktif feature grup id listesi. None = registry default gruplari. "
            "Olasi degerler: returns, volatility, momentum, volume, calendar, "
            "technicals, market_regime, cross_asset, macro_tr, macro_global, "
            "fundamental, fibonacci, donchian, rolling_zscore, obv, seasonality, iceemdan"
        ),
        examples=[["returns", "volatility", "momentum", "macro_tr", "macro_global"]],
    )
    target_type: str = Field(
        default='log_return',
        description="'log_return' (onerilen, stationarity) veya 'abs_price' (eski davranis)",
    )
    use_iceemdan: bool = Field(
        default=False,
        description="ICEEMDAN gurultu filtrelemesi uygula (egitimi yavaslatur, kaliteyi artirabilir)",
    )


class PredictionTrainRequest(BaseModel):
    symbol: str = Field(description="Tahmin yapılacak sembol (ör: AKBNK.IS, GOLD_GRAM_TRY)")
    horizon: str = Field(default='daily', description="'daily' veya 'weekly'")
    start_date: str = Field(default='2018-01-01', description="Eğitim verisi başlangıç tarihi")
    test_ratio: float = Field(default=0.2, description="Test seti oranı", ge=0.05, le=0.4)
    source: Optional[str] = Field(
        default=None,
        description="Veri kaynagi (gold/FX icin 'borsapy' veya 'yfinance'; hisse icin yoksayilir)",
    )
    feature_config: Optional[FeatureGroupsConfig] = Field(
        default=None,
        description="Feature grup konfigurasyonu (None = varsayilan gruplar + log_return hedef)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "GOLD_GRAM_TRY",
                "horizon": "daily",
                "start_date": "2018-01-01",
                "test_ratio": 0.2,
                "source": "yfinance",
                "feature_config": {
                    "enabled_groups": None,
                    "target_type": "log_return",
                    "use_iceemdan": False,
                }
            }
        }


class ModelPerformanceMetrics(BaseModel):
    mape: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    direction_accuracy: Optional[float] = None


class DataFreshnessReport(BaseModel):
    """Faz 4: Egitim/tahmin oncesi veri tazelik raporu (ensure_fresh_data ciktisi)."""
    symbol: Optional[str] = None
    source: Optional[str] = None
    asset_type: Optional[str] = None
    target_date: Optional[str] = None
    macro_target_date: Optional[str] = None
    price: Optional[Dict[str, Any]] = None
    macro: Optional[Dict[str, Any]] = None
    fundamental: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PredictionTrainResponse(BaseModel):
    symbol: str
    horizon: str
    n_train: int
    n_test: int
    n_features: int
    train_metrics: ModelPerformanceMetrics
    test_metrics: ModelPerformanceMetrics
    model_path: str
    trained_at: str
    data_freshness: Optional[Dict[str, Any]] = None


class PredictionTrainAcceptedResponse(BaseModel):
    """202 Accepted — egitim arka planda baslatildi."""
    symbol: str
    horizon: str
    source: Optional[str] = None
    state: str = 'running'
    started_at: str
    message: str = "Egitim arka planda baslatildi. Ilerleme icin /train/status, sonuc icin /models endpoint'ini kullanin."
    data_freshness: Optional[Dict[str, Any]] = None


class PredictionTrainStatusResponse(BaseModel):
    symbol: str
    horizon: str
    source: Optional[str] = None
    state: str  # idle | running | completed | error
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class PredictionTarget(BaseModel):
    """Tahmin hedefi: sembol + opsiyonel kaynak."""
    symbol: str
    source: Optional[str] = None


class PredictionRequest(BaseModel):
    symbols: Optional[List[str]] = Field(
        default=None,
        description="Tahmin yapilacak sembol listesi (basit kullanim)",
    )
    targets: Optional[List[PredictionTarget]] = Field(
        default=None,
        description="Detayli kullanim — her hedef icin sembol + kaynak. Verilirse 'symbols' yoksayilir.",
    )
    horizon: str = Field(default='daily', description="'daily' veya 'weekly'")
    source: Optional[str] = Field(
        default=None,
        description="Tum sembollerde kullanilacak kaynak (ops.); 'targets' verilirse yoksayilir.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "targets": [
                    {"symbol": "GOLD_GRAM_TRY", "source": "yfinance"},
                    {"symbol": "AKBNK.IS"}
                ],
                "horizon": "daily"
            }
        }


class SinglePrediction(BaseModel):
    symbol: str
    source: Optional[str] = None
    prediction_date: str
    horizon: str
    predicted_close: float
    predicted_direction: str
    predicted_change_pct: float
    confidence: float
    current_close: float
    made_at: str
    model_predictions: Optional[Dict[str, float]] = None
    ensemble_agreement: Optional[float] = None
    n_models: Optional[int] = None


class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    count: int
    data_freshness: Optional[Dict[str, Dict[str, Any]]] = None


class EnsembleTrainRequest(BaseModel):
    symbol: str = Field(description="Tahmin yapilacak sembol")
    horizon: str = Field(default='daily', description="'daily' veya 'weekly'")
    start_date: str = Field(default='2018-01-01')
    test_ratio: float = Field(default=0.2, ge=0.05, le=0.4)
    optimize: bool = Field(default=False, description="Oncelikle HPO calistir")
    n_hpo_trials: int = Field(default=30, ge=5, le=200)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AKBNK.IS",
                "horizon": "daily",
                "optimize": False
            }
        }


class EnsembleTrainResponse(BaseModel):
    symbol: str
    horizon: str
    n_total: int
    n_train: int
    n_test: int
    n_features: int
    n_models: int
    models_trained: List[str]
    model_results: Dict[str, Any]
    ensemble_test_metrics: Dict[str, Any]
    trained_at: str
    training_time_seconds: Optional[float] = None


class CrossValidateRequest(BaseModel):
    symbol: str
    horizon: str = 'daily'
    start_date: str = '2018-01-01'
    model_types: Optional[List[str]] = None
    n_splits: int = Field(default=5, ge=2, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AKBNK.IS",
                "horizon": "daily",
                "n_splits": 5
            }
        }


class HyperOptRequest(BaseModel):
    symbol: str
    horizon: str = 'daily'
    start_date: str = '2018-01-01'
    model_types: Optional[List[str]] = None
    n_trials: int = Field(default=50, ge=10, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AKBNK.IS",
                "horizon": "daily",
                "n_trials": 50
            }
        }


class PerformanceMetricsResponse(BaseModel):
    symbol: str
    horizon: str
    mape: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    direction_accuracy: Optional[float] = None
    n_predictions: int = 0


class PredictionHistoryItem(BaseModel):
    prediction_date: str
    predicted_close: float
    predicted_direction: str
    predicted_change_pct: float
    confidence: float
    current_close: Optional[float] = None
    made_at: str
    actual_close: Optional[float] = None
    error_pct: Optional[float] = None
    direction_correct: Optional[bool] = None


class PredictionHistoryResponse(BaseModel):
    symbol: str
    horizon: str
    history: List[Dict[str, Any]]
    total: int


class GoldPriceItem(BaseModel):
    symbol: str
    name: str
    close: float
    open: float
    high: float
    low: float
    change_pct: float
    date: str
    currency: str


class GoldPricesResponse(BaseModel):
    prices: List[GoldPriceItem]
    fetched_at: str


class GoldHistoryResponse(BaseModel):
    symbol: str
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]


class EvaluatePendingResponse(BaseModel):
    evaluated_count: int
    message: str


class SymbolEntry(BaseModel):
    """Egitim icin secilebilir sembol + desteklenen kaynaklar."""
    symbol: str
    name: str
    category: str  # 'bist30' | 'gold' | 'fx'
    sources: List[str]  # ['yfinance'] veya ['borsapy', 'yfinance']
    available_feature_groups: Optional[List[str]] = None  # Registry default gruplari


class TradableSymbolsResponse(BaseModel):
    symbols: List[SymbolEntry]


class TrainedModelEntry(BaseModel):
    """Tahmin icin kullanilabilir egitilmis model."""
    symbol: str
    source: Optional[str] = None
    horizon: str
    name: Optional[str] = None  # gosterim icin
    n_features: int = 0
    saved_at: Optional[str] = None
    test_mape: Optional[float] = None
    test_direction_accuracy: Optional[float] = None
    models_trained: List[str] = []
    feature_groups: Optional[List[str]] = None
    target_type: Optional[str] = None


class TrainedModelsResponse(BaseModel):
    models: List[TrainedModelEntry]
