"""
Prediction Schemas
Tahmin modülü için Pydantic request/response modelleri
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class PredictionTrainRequest(BaseModel):
    symbol: str = Field(description="Tahmin yapılacak sembol (ör: AKBNK.IS, GOLD_GRAM_TRY)")
    horizon: str = Field(default='daily', description="'daily' veya 'weekly'")
    start_date: str = Field(default='2018-01-01', description="Eğitim verisi başlangıç tarihi")
    test_ratio: float = Field(default=0.2, description="Test seti oranı", ge=0.05, le=0.4)

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AKBNK.IS",
                "horizon": "daily",
                "start_date": "2018-01-01",
                "test_ratio": 0.2
            }
        }


class ModelPerformanceMetrics(BaseModel):
    mape: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    direction_accuracy: Optional[float] = None


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


class PredictionRequest(BaseModel):
    symbols: List[str] = Field(description="Tahmin yapılacak sembol listesi")
    horizon: str = Field(default='daily', description="'daily' veya 'weekly'")

    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["AKBNK.IS", "THYAO.IS"],
                "horizon": "daily"
            }
        }


class SinglePrediction(BaseModel):
    symbol: str
    prediction_date: str
    horizon: str
    predicted_close: float
    predicted_direction: str
    predicted_change_pct: float
    confidence: float
    current_close: float
    made_at: str


class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    count: int


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


class TradableSymbolsResponse(BaseModel):
    bist30: List[str]
    gold: List[str]
    fx: List[str]
    synthetic: List[str]
    all: List[str]
