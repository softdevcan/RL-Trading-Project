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
    EnsembleTrainRequest, EnsembleTrainResponse,
    CrossValidateRequest, HyperOptRequest,
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
    """Sembol icin ensemble modeli egit (XGBoost + LightGBM + CatBoost + BiLSTM + TFT)."""
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
        logger.error(f"Model egitimi basarisiz: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/train-ensemble", response_model=EnsembleTrainResponse)
async def train_ensemble(request: EnsembleTrainRequest):
    """Sembol icin ensemble modeli egit (detayli sonuc)."""
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
            optimize=request.optimize,
            n_hpo_trials=request.n_hpo_trials,
        )
        return EnsembleTrainResponse(
            symbol=result['symbol'],
            horizon=result['horizon'],
            n_total=result.get('n_train', 0) + result.get('n_test', 0),
            n_train=result.get('n_train', 0),
            n_test=result.get('n_test', 0),
            n_features=result.get('n_features', 0),
            n_models=result.get('n_models', 0),
            models_trained=result.get('models_trained', []),
            model_results=result.get('model_results', {}),
            ensemble_test_metrics=result.get('ensemble_test_metrics', {}),
            trained_at=result.get('trained_at', ''),
            training_time_seconds=result.get('training_time_seconds'),
        )
    except Exception as exc:
        logger.error(f"Ensemble egitimi basarisiz: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/cross-validate")
async def cross_validate(request: CrossValidateRequest):
    """Walk-forward cross-validation ile model karsilastirma."""
    symbol = _validate_symbol(request.symbol)
    horizon = _validate_horizon(request.horizon)

    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        result = svc.cross_validate(
            symbol=symbol,
            horizon=horizon,
            start_date=request.start_date,
            model_types=request.model_types,
            n_splits=request.n_splits,
        )
        return result
    except Exception as exc:
        logger.error(f"Cross-validation basarisiz: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/optimize")
async def optimize_hyperparameters(request: HyperOptRequest):
    """Tum modeller icin Optuna hiperparametre optimizasyonu."""
    symbol = _validate_symbol(request.symbol)
    horizon = _validate_horizon(request.horizon)

    from app.services.prediction_service import PredictionService
    svc = PredictionService()

    try:
        result = svc.optimize_hyperparameters(
            symbol=symbol,
            horizon=horizon,
            start_date=request.start_date,
            model_types=request.model_types,
            n_trials=request.n_trials,
        )
        return result
    except Exception as exc:
        logger.error(f"HPO basarisiz: {exc}", exc_info=True)
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

    all_syms = list(dict.fromkeys(get_all_tradeable_symbols()))  # deduplicate, preserve order
    return TradableSymbolsResponse(
        bist30=BIST30_SYMBOLS,
        gold=GOLD_SYMBOLS,
        fx=FX_SYMBOLS,
        synthetic=SYNTHETIC_SYMBOLS,
        all=all_syms,
    )


# ------------------------------------------------------------------
# SHAP Aciklanabilirlik  (Faz 3.4.1)
# ------------------------------------------------------------------

@router.get("/explain/{symbol}")
async def explain_prediction(
    symbol: str,
    horizon: str = Query(default='daily'),
    model_type: str = Query(default='xgboost', description="xgboost, lightgbm, catboost"),
    n_background: int = Query(default=100, ge=10, le=500),
):
    """Sembol tahmini icin SHAP feature importance aciklamasi uret.

    Returns:
        {
            'symbol': str,
            'model_type': str,
            'shap_available': bool,
            'single_prediction': {shap_values, base_value, top_positive, top_negative},
            'global_importance': {feature: mean_abs_shap},
        }
    """
    symbol = _validate_symbol(symbol)
    horizon = _validate_horizon(horizon)

    from prediction.explainability import ModelExplainer
    from prediction.models.ensemble import StackingEnsemble
    from app.services.prediction_service import PredictionService

    explainer = ModelExplainer(n_background=n_background)

    if not explainer.is_available():
        return {
            'symbol': symbol,
            'model_type': model_type,
            'shap_available': False,
            'message': "shap paketi kurulu degil. 'pip install shap' ile kurun.",
        }

    try:
        svc = PredictionService()
        df = svc._fetch_data(symbol)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"Veri bulunamadi: {symbol}")

        ensemble = StackingEnsemble(horizon=horizon)
        ensemble.load(symbol)

        if model_type not in ensemble.base_models:
            raise HTTPException(
                status_code=404,
                detail=f"'{model_type}' modeli '{symbol}' icin yuklenemedi. "
                       f"Mevcut modeller: {list(ensemble.base_models.keys())}"
            )

        model = ensemble.base_models[model_type]

        from prediction.feature_engineer import PredictionFeatureEngineer
        import numpy as np
        fe = PredictionFeatureEngineer(horizon)
        feat_df = fe.build_features(df, symbol)
        feature_cols = ensemble.feature_cols or fe.get_feature_columns(feat_df)

        X = feat_df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        background = X[:-1] if len(X) > 1 else X
        last_row = X[-1:]

        single = explainer.explain_prediction(model, last_row, feature_cols, background)
        global_imp = explainer.explain_global(model, background, feature_cols)

        return {
            'symbol': symbol,
            'model_type': model_type,
            'horizon': horizon,
            'shap_available': True,
            'n_features': len(feature_cols),
            'single_prediction': single,
            'global_importance': global_imp,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"SHAP aciklama hatasi [{symbol}]: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
