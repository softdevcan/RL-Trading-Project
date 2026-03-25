"""
Prediction API Routes
Fiyat tahmini ve altın verisi için FastAPI endpoint'leri
"""

import re
import os
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.schemas.prediction import (
    PredictionTrainRequest, PredictionTrainResponse,
    PredictionRequest, PredictionResponse, SinglePrediction,
    PerformanceMetricsResponse, PredictionHistoryResponse,
    GoldPricesResponse, GoldPriceItem, GoldHistoryResponse,
    EvaluatePendingResponse, TradableSymbolsResponse,
    ModelPerformanceMetrics,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/prediction", tags=["Prediction"])

_settings = get_settings()

# Güvenli sembol regex (path traversal koruması)
_SAFE_SYMBOL = re.compile(r'^[A-Za-z0-9_.=\-]+$')


def _validate_symbol(symbol: str) -> str:
    sym = os.path.basename(symbol)
    if not _SAFE_SYMBOL.match(sym):
        raise HTTPException(status_code=400, detail=f"Geçersiz sembol: {symbol}")
    return sym


def _validate_horizon(horizon: str) -> str:
    if horizon not in ('daily', 'weekly'):
        raise HTTPException(status_code=400, detail="horizon 'daily' veya 'weekly' olmalı")
    return horizon


# ------------------------------------------------------------------
# Model Eğitimi
# ------------------------------------------------------------------

@router.post("/train", response_model=PredictionTrainResponse)
async def train_model(request: PredictionTrainRequest):
    """Sembol ve horizon için XGBoost modeli eğit."""
    symbol = _validate_symbol(request.symbol)
    horizon = _validate_horizon(request.horizon)

    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        result = svc.train_model(
            symbol=symbol,
            horizon=horizon,
            start_date=request.start_date,
            test_ratio=request.test_ratio,
        )
        return PredictionTrainResponse(
            symbol=result['symbol'],
            horizon=result['horizon'],
            n_train=result['n_train'],
            n_test=result['n_test'],
            n_features=result['n_features'],
            train_metrics=ModelPerformanceMetrics(**result['train_metrics']),
            test_metrics=ModelPerformanceMetrics(**result['test_metrics']),
            model_path=result['model_path'],
            trained_at=result['trained_at'],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Model eğitimi başarısız: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/models")
async def list_models():
    """Eğitilmiş tahmin modellerini listele."""
    from app.services.prediction_service import PredictionService
    svc = PredictionService()
    try:
        return {"models": svc.list_trained_models()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Tahmin Üretme
# ------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Sembol listesi için günlük/haftalık fiyat tahmini üret."""
    symbols = [_validate_symbol(s) for s in request.symbols]
    horizon = _validate_horizon(request.horizon)

    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        preds = svc.predict(symbols, horizon, save=True)
        items = [SinglePrediction(**p) for p in preds]
        return PredictionResponse(predictions=items, count=len(items))
    except Exception as exc:
        logger.error(f"Tahmin başarısız: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Tahmin Geçmişi
# ------------------------------------------------------------------

@router.get("/predictions/{symbol}", response_model=PredictionHistoryResponse)
async def get_predictions(
    symbol: str,
    horizon: str = Query(default='daily'),
):
    """Sembol için tahmin geçmişini döndür."""
    symbol = _validate_symbol(symbol)
    horizon = _validate_horizon(horizon)

    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        result = svc.get_prediction_history(symbol, horizon)
        return PredictionHistoryResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Değerlendirme
# ------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    symbols: Optional[list] = None


@router.post("/evaluate", response_model=EvaluatePendingResponse)
async def evaluate_pending(request: EvaluateRequest = EvaluateRequest()):
    """Bekleyen tahminleri gerçek fiyatlarla değerlendir."""
    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        count = svc.evaluate_pending(request.symbols)
        return EvaluatePendingResponse(
            evaluated_count=count,
            message=f"{count} tahmin değerlendirildi"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Performans
# ------------------------------------------------------------------

@router.get("/performance/summary")
async def get_performance_summary():
    """Tüm semboller için özet performans."""
    from app.services.prediction_service import PredictionService
    svc = PredictionService()
    try:
        return {"summary": svc.get_summary()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/performance/{symbol}", response_model=PerformanceMetricsResponse)
async def get_performance(
    symbol: str,
    horizon: str = Query(default='daily'),
    window: int = Query(default=30, ge=5),
):
    """Sembol için performans metrikleri."""
    symbol = _validate_symbol(symbol)
    horizon = _validate_horizon(horizon)

    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        metrics = svc.get_performance(symbol, horizon, window)
        return PerformanceMetricsResponse(**metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chart-data/{symbol}")
async def get_chart_data(
    symbol: str,
    horizon: str = Query(default='daily'),
    days: int = Query(default=30, ge=7),
):
    """Tahmin vs gerçek ve doğruluk trendi chart verisi."""
    symbol = _validate_symbol(symbol)
    horizon = _validate_horizon(horizon)

    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        return svc.get_chart_data(symbol, horizon, days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Altın Fiyatları
# ------------------------------------------------------------------

@router.get("/gold/prices", response_model=GoldPricesResponse)
async def get_gold_prices():
    """Güncel ons altın, gram altın ve USD/TRY fiyatlarını döndür."""
    from app.services.gold_service import GoldService
    svc = GoldService()

    try:
        prices = svc.get_current_prices()
        items = [GoldPriceItem(**p) for p in prices]
        return GoldPricesResponse(
            prices=items,
            fetched_at=datetime.now().isoformat()
        )
    except Exception as exc:
        logger.error(f"Altın fiyatları çekilemedi: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/gold/history", response_model=GoldHistoryResponse)
async def get_gold_history(
    symbol: str = Query(default='GOLD_GRAM_TRY',
                        description="GC=F, USDTRY=X veya GOLD_GRAM_TRY"),
    days: int = Query(default=90, ge=7, le=1825),
):
    """Altın/döviz tarihsel fiyat verisi."""
    from app.services.gold_service import GoldService
    svc = GoldService()

    try:
        data = svc.get_history(symbol, days)
        return GoldHistoryResponse(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Sembol Listesi
# ------------------------------------------------------------------

@router.get("/symbols", response_model=TradableSymbolsResponse)
async def get_symbols():
    """Tahmin yapılabilir tüm sembolleri döndür."""
    from data.bist30_symbols import (
        BIST30_SYMBOLS, GOLD_SYMBOLS, FX_SYMBOLS, SYNTHETIC_SYMBOLS,
        get_all_tradeable_symbols
    )

    return TradableSymbolsResponse(
        bist30=BIST30_SYMBOLS,
        gold=GOLD_SYMBOLS,
        fx=FX_SYMBOLS,
        synthetic=SYNTHETIC_SYMBOLS,
        all=get_all_tradeable_symbols(),
    )
