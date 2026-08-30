"""
Prediction API Routes
Fiyat tahmini ve altın verisi için FastAPI endpoint'leri
"""

import re
import os
import math
import logging
from datetime import datetime
from typing import Any, Optional
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status
from pydantic import BaseModel

from app.auth import workspace as ws
from app.auth.deps import RequireWriter


def _strip_nonfinite(obj: Any) -> Any:
    """JSON'a gidecek degerlerden NaN/Inf'i temizle (None'a cevir)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strip_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_nonfinite(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_nonfinite(v) for v in obj)
    return obj

from app.schemas.prediction import (
    PredictionTrainRequest, PredictionTrainResponse,
    PredictionTrainAcceptedResponse, PredictionTrainStatusResponse,
    PredictionRequest, PredictionResponse, SinglePrediction,
    PerformanceMetricsResponse, PredictionHistoryResponse,
    GoldPricesResponse, GoldPriceItem, GoldHistoryResponse,
    EvaluatePendingResponse, TradableSymbolsResponse, SymbolEntry,
    TrainedModelEntry, TrainedModelsResponse,
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


def _validate_source(symbol: str, source: Optional[str]) -> Optional[str]:
    """Kaynak gold/FX icin zorunlu degerse de None gelebilir (default'a duser)."""
    from app.services.prediction_service import _resolve_source
    try:
        return _resolve_source(symbol, source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ------------------------------------------------------------------
# Model Eğitimi
# ------------------------------------------------------------------

@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    request: PredictionTrainRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    user: RequireWriter,
    sync: bool = Query(default=False, description="true ise senkron egitim (timeout riski)"),
):
    """Ensemble modeli egit (XGBoost + LightGBM + CatBoost + BiLSTM + TFT).

    Varsayilan davranis: egitim arka planda baslatilir, 202 Accepted doner.
    Ilerleme `/train/status`, sonuclar `/models` endpoint'inden izlenir.
    sync=true verilirse eski davranis: senkron egitim, 200 + tam sonuc (uzun surebilir).
    """
    symbol = _validate_symbol(request.symbol)
    horizon = _validate_horizon(request.horizon)
    source = _validate_source(symbol, request.source)

    from app.services.prediction_service import (
        PredictionService, get_training_state, ensure_fresh_data
    )
    svc = PredictionService()

    # Faz 4: Egitim oncesi T-1 isgunu hedefli veri tazeleme
    try:
        freshness = ensure_fresh_data(symbol, source=source)
        logger.info(f"Veri tazelik raporu: {freshness}")
    except Exception as exc:
        logger.warning(f"ensure_fresh_data basarisiz, mevcut veriyle devam: {exc}")
        freshness = {'error': str(exc)}

    if sync:
        try:
            result = svc.train_model(
                symbol=symbol,
                horizon=horizon,
                start_date=request.start_date,
                test_ratio=request.test_ratio,
                source=source,
                feature_groups=(request.feature_config.enabled_groups
                                if request.feature_config else None),
                target_type=(request.feature_config.target_type
                             if request.feature_config else 'log_return'),
            )
            response.status_code = status.HTTP_200_OK
            payload = PredictionTrainResponse(
                symbol=result['symbol'],
                horizon=result['horizon'],
                n_train=result['n_train'],
                n_test=result['n_test'],
                n_features=result['n_features'],
                train_metrics=ModelPerformanceMetrics(**_strip_nonfinite(result['train_metrics'])),
                test_metrics=ModelPerformanceMetrics(**_strip_nonfinite(result['test_metrics'])),
                model_path=result['model_path'],
                trained_at=result['trained_at'],
                data_freshness=freshness,
            ).model_dump()
            return payload
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            logger.warning(f"Model egitimi basarisiz (veri yetersiz): {exc}")
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(f"Model egitimi basarisiz: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # Arka plan yolu: ayni (symbol, horizon, source) icin egitim zaten calisiyorsa yenisini baslatma
    current = get_training_state(symbol, horizon, source)
    if current and current.get('state') == 'running':
        return PredictionTrainAcceptedResponse(
            symbol=symbol,
            horizon=horizon,
            source=source,
            state='running',
            started_at=current.get('started_at', ''),
            message="Bu sembol+kaynak icin egitim zaten calisiyor.",
            data_freshness=freshness,
        )

    background_tasks.add_task(
        svc.train_model_async,
        symbol=symbol,
        horizon=horizon,
        start_date=request.start_date,
        test_ratio=request.test_ratio,
        source=source,
        feature_groups=(request.feature_config.enabled_groups
                        if request.feature_config else None),
        target_type=(request.feature_config.target_type
                     if request.feature_config else 'log_return'),
        # Arka plan gorevi istek baglamini devralmaz: model dosyalari dogru
        # kullanicinin calisma alanina yazilsin diye kimlik acikca tasinir.
        user_id=ws.current_user_id(),
    )
    return PredictionTrainAcceptedResponse(
        symbol=symbol,
        horizon=horizon,
        source=source,
        state='running',
        started_at=datetime.now().isoformat(),
        data_freshness=freshness,
    )


@router.get("/train/status", response_model=PredictionTrainStatusResponse)
async def get_train_status(
    symbol: str = Query(..., description="Sembol"),
    horizon: str = Query(default='daily'),
    source: Optional[str] = Query(default=None, description="Veri kaynagi"),
):
    """Arka plan egitimin durumunu dondur."""
    symbol = _validate_symbol(symbol)
    horizon = _validate_horizon(horizon)
    resolved_source = _validate_source(symbol, source) if source is not None else None

    from app.services.prediction_service import get_training_state
    info = get_training_state(symbol, horizon, resolved_source)
    if info is None:
        return PredictionTrainStatusResponse(
            symbol=symbol, horizon=horizon, source=resolved_source, state='idle',
        )
    return PredictionTrainStatusResponse(
        symbol=symbol,
        horizon=horizon,
        source=info.get('source', resolved_source),
        state=info['state'],
        started_at=info.get('started_at'),
        finished_at=info.get('finished_at'),
        error=info.get('error'),
    )


@router.get("/train/active")
async def list_active_trainings() -> dict:
    """Kullanicinin TUM tahmin egitimi kayitlari (sembol basina degil).

    `/train/status` sembol istiyor; pano sayfaya donuldugunde hangi sembolun
    egitildigini BILMIYOR (o bilgi yalnizca istemci tarafinda bir Store'da
    duruyordu ve gezinmede sifirlaniyordu). Bu uc listeyi backend'den verir,
    boylece sayfa acilista surmekte olan kosumlari bulup gosterebilir.

    Durumlar bellekte tutuluyor (`prediction_service._training_state`), uc
    ucuz: diske veya DB'ye gitmiyor.
    """
    from app.services.prediction_service import get_training_state

    runs = get_training_state() or []
    if not isinstance(runs, list):
        runs = []
    # `result` egitim ciktisinin tamami — listede tasimanin anlami yok.
    slim = [
        {k: v for k, v in run.items() if k != "result"}
        for run in runs
    ]
    running = [r for r in slim if r.get("state") == "running"]
    return {"runs": slim, "running": len(running), "count": len(slim)}


@router.post("/train-ensemble", response_model=EnsembleTrainResponse)
async def train_ensemble(request: EnsembleTrainRequest, user: RequireWriter):
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
async def cross_validate(request: CrossValidateRequest, user: RequireWriter):
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
async def optimize_hyperparameters(request: HyperOptRequest, user: RequireWriter):
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


@router.get("/models", response_model=TrainedModelsResponse)
async def list_models():
    """Egitilmis tahmin modellerini listele (sembol + source + horizon + metrikler)."""
    from app.services.prediction_service import PredictionService
    from data.bist30_symbols import STOCK_INFO, ASSET_INFO

    svc = PredictionService()
    info_map = {**STOCK_INFO, **ASSET_INFO}

    try:
        raw = svc.list_trained_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    models: list = []
    for m in raw:
        m = _strip_nonfinite(m)
        sym = m.get('symbol', '')
        src = m.get('source')
        test_metrics = m.get('test_metrics') or {}
        models.append(TrainedModelEntry(
            symbol=sym,
            source=src,
            horizon=m.get('horizon', 'daily'),
            name=info_map.get(sym, {}).get('name'),
            n_features=len(m.get('feature_cols', []) or []),
            saved_at=m.get('saved_at'),
            test_mape=test_metrics.get('mape'),
            test_direction_accuracy=test_metrics.get('direction_accuracy'),
            models_trained=m.get('models_trained', []) or [],
        ))
    return TrainedModelsResponse(models=models)


# ------------------------------------------------------------------
# Tahmin Üretme
# ------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, user: RequireWriter):
    """Sembol+source listesi icin gunluk/haftalik fiyat tahmini uret."""
    horizon = _validate_horizon(request.horizon)

    if not request.targets and not request.symbols:
        raise HTTPException(
            status_code=400,
            detail="'targets' veya 'symbols' alanlarindan biri gerekli."
        )

    targets_norm = None
    symbols_norm = None
    if request.targets:
        targets_norm = []
        for t in request.targets:
            sym = _validate_symbol(t.symbol)
            src = _validate_source(sym, t.source)
            targets_norm.append({'symbol': sym, 'source': src})
    else:
        symbols_norm = [_validate_symbol(s) for s in request.symbols]
        # Tek bir source butun sembollerde geciyorsa once dogrula
        if request.source is not None:
            for s in symbols_norm:
                _validate_source(s, request.source)

    from app.services.prediction_service import PredictionService, ensure_fresh_data
    svc = PredictionService()

    # Faz 4: Tahmin oncesi T-1 isgunu hedefli veri tazeleme (her sembol icin)
    freshness_per_symbol = {}
    target_list = (
        targets_norm
        if targets_norm
        else [{'symbol': s, 'source': request.source} for s in (symbols_norm or [])]
    )
    for t in target_list:
        try:
            freshness_per_symbol[t['symbol']] = ensure_fresh_data(
                t['symbol'], source=t.get('source'),
            )
        except Exception as exc:
            logger.warning(f"ensure_fresh_data {t['symbol']}: {exc}")
            freshness_per_symbol[t['symbol']] = {'error': str(exc)}

    try:
        preds = svc.predict(
            symbols=symbols_norm or [],
            horizon=horizon,
            save=True,
            source=request.source,
            targets=targets_norm,
        )
        items = [SinglePrediction(**p) for p in preds]
        return PredictionResponse(
            predictions=items, count=len(items),
            data_freshness=freshness_per_symbol,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
async def evaluate_pending(user: RequireWriter, request: EvaluateRequest = EvaluateRequest()):
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
    """Egitim icin secilebilecek sembolleri ve her sembolun desteklenen kaynaklarini dondur."""
    from data.bist30_symbols import (
        BIST30_SYMBOLS, GOLD_SYMBOLS, FX_SYMBOLS, SYNTHETIC_SYMBOLS,
        STOCK_INFO, ASSET_INFO,
    )
    from app.services.prediction_service import get_supported_sources
    from prediction.feature_groups import default_groups

    info_map = {**STOCK_INFO, **ASSET_INFO}
    default_grp = default_groups()

    def _entry(sym: str, category: str) -> SymbolEntry:
        meta = info_map.get(sym, {})
        return SymbolEntry(
            symbol=sym,
            name=meta.get('name', sym),
            category=category,
            sources=get_supported_sources(sym),
            available_feature_groups=default_grp,
        )

    entries: list = []
    seen = set()

    for s in BIST30_SYMBOLS:
        if s not in seen:
            entries.append(_entry(s, 'bist30'))
            seen.add(s)
    for s in GOLD_SYMBOLS + SYNTHETIC_SYMBOLS:
        if s not in seen:
            entries.append(_entry(s, 'gold'))
            seen.add(s)
    for s in FX_SYMBOLS:
        if s not in seen:
            entries.append(_entry(s, 'fx'))
            seen.add(s)

    return TradableSymbolsResponse(symbols=entries)


# ------------------------------------------------------------------
# Fiyat Geçmisi (interaktif grafik icin)
# ------------------------------------------------------------------

@router.get("/price-history")
async def get_price_history(
    symbol: str = Query(..., description="Sembol (ör: GC=F, GOLD_GRAM_TRY, AKBNK.IS)"),
    source: Optional[str] = Query(default=None, description="Veri kaynagi"),
    days: int = Query(default=90, ge=10, le=1825, description="Son kac gun"),
):
    """Sembol icin tarihsel fiyat verisi (OHLCV + MA20/MA50 + BB) dondur.

    Tracker'a bagimli olmayan dogrudan veri okuma endpoint'i.
    Dashboard interaktif grafigi bu endpoint'i kullanir.
    """
    from app.services.prediction_service import _fetch_symbol_data, _resolve_source
    import math
    from datetime import datetime, timedelta

    try:
        resolved = _resolve_source(symbol, source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y-%m-%d')

    try:
        df = _fetch_symbol_data(symbol, start_date=start_date, source=resolved)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Veri cekilemedi: {exc}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"{symbol} icin veri bulunamadi")

    df = df.tail(days)

    close = df['close']
    ma20 = close.rolling(20, min_periods=5).mean()
    ma50 = close.rolling(50, min_periods=10).mean()

    # Bollinger Bands (20g, 2 std)
    std20 = close.rolling(20, min_periods=5).std()
    bb_upper = (ma20 + 2 * std20)
    bb_lower = (ma20 - 2 * std20)

    def _safe(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(float(v), 4)

    dates = [str(d)[:10] for d in df.index]
    result = {
        "symbol": symbol,
        "source": resolved,
        "dates": dates,
        "open":     [_safe(v) for v in df.get('open', close).values],
        "high":     [_safe(v) for v in df.get('high', close).values],
        "low":      [_safe(v) for v in df.get('low', close).values],
        "close":    [_safe(v) for v in close.values],
        "volume":   [_safe(v) for v in df.get('volume', pd.Series(index=df.index)).values],
        "ma20":     [_safe(v) for v in ma20.values],
        "ma50":     [_safe(v) for v in ma50.values],
        "bb_upper": [_safe(v) for v in bb_upper.values],
        "bb_lower": [_safe(v) for v in bb_lower.values],
    }
    return result


# ------------------------------------------------------------------
# SHAP Aciklanabilirlik  (Faz 3.4.1)
# ------------------------------------------------------------------

@router.get("/explain/{symbol}")
async def explain_prediction(
    symbol: str,
    horizon: str = Query(default='daily'),
    model_type: str = Query(default='xgboost', description="xgboost, lightgbm, catboost"),
    n_background: int = Query(default=100, ge=10, le=500),
    source: Optional[str] = Query(default=None, description="Veri kaynagi (gold/FX icin)"),
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
    resolved_source = _validate_source(symbol, source)

    from prediction.explainability import ModelExplainer
    from prediction.models.ensemble import StackingEnsemble
    from app.services.prediction_service import _fetch_symbol_data

    explainer = ModelExplainer(n_background=n_background)

    if not explainer.is_available():
        return {
            'symbol': symbol,
            'source': resolved_source,
            'model_type': model_type,
            'shap_available': False,
            'message': "shap paketi kurulu degil. 'pip install shap' ile kurun.",
        }

    try:
        from datetime import timedelta as _td
        start = (datetime.now() - _td(days=365)).strftime('%Y-%m-%d')
        df = _fetch_symbol_data(symbol, start, source=resolved_source)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"Veri bulunamadi: {symbol}")

        ensemble = StackingEnsemble(horizon=horizon, source=resolved_source)
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
            'source': resolved_source,
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
