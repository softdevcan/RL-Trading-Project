"""
Trading API Routes
FastAPI endpoints for RL trading system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import asyncio
import glob
import os
import re
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

from app.schemas.trading import (
    TrainingRequest, TrainingStatus, TrainingEstimate, ModelMetrics,
    ModelInfo, TrainingResponse,
    DailyDecisionRequest, DailyDecisionResponse, TradeDecision,
    PortfolioSnapshot, PortfolioHistoryResponse,
    ModelComparisonResponse
)
from app.core.config import get_settings
from app.auth import workspace as ws
from app.auth.deps import RequireWriter
from app.auth.models import Role
from app.services import training_eta
from app.services import background_jobs as jobs

_settings = get_settings()

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading", tags=["Trading"])

# Training state — guarded by _training_lock (#23)
# state semantics: "idle" → never ran / cleared
#                  "running" → is_training=True
#                  "completed" → finished OK
#                  "error" → raised exception (detail in .error)
#
# Faz 7: durum artik KULLANICI BASINA tutulur. Tek global sozluk, bir
# kullanicinin egitimi sirasinda digerlerini "Training already in progress"
# ile bloklar ve ilerleme cubuklarini karistirirdi.
_training_lock = asyncio.Lock()


def _empty_training_state() -> Dict[str, Any]:
    return {
        "is_training": False,
        "state": "idle",
        "current_step": 0,
        "total_steps": 0,
        "start_time": None,
        "metrics": {},
        "error": None,
        # ETA alanlari: phase_name kaba faz ("preparing"/"training"/"evaluating"),
        # *_ts epoch damgalari, estimate egitim baslarken uretilen on tahmin.
        "phase_name": None,
        "run_start_ts": None,
        "learn_start_ts": None,
        "estimate": None,
        # Kosumun BITTIGI an (basarili ya da hatali). Bildirim zili "yakinda
        # bitti mi" sorusunu bununla cevapliyor; learn_end_ts degerlendirme
        # oncesini isaretledigi icin ayri tutuluyor.
        "finished_ts": None,
        # Kosumun gercek sembol evreni + kullaniciya gosterilecek uyarilar
        "n_symbols": None,
        "warnings": [],
    }


_training_states: Dict[str, Dict[str, Any]] = {}


def _training_key(user_id: Optional[str] = None) -> str:
    """Durum anahtari: kullanici id, auth kapaliysa "local"."""
    return user_id or ws.current_user_id() or "local"


def get_training_state(user_id: Optional[str] = None) -> Dict[str, Any]:
    return _training_states.setdefault(_training_key(user_id), _empty_training_state())

_SAFE_MODEL_NAME = re.compile(r'^[a-zA-Z0-9_.-]+$')


def sanitize_model_name(name: str) -> str:
    """Prevent path traversal: accept only safe filename characters."""
    name = os.path.basename(name)
    if not _SAFE_MODEL_NAME.match(name):
        raise HTTPException(status_code=400, detail="Invalid model name")
    return name


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/hyperparameters/{algorithm}")
async def get_hyperparameter_studies(algorithm: str):
    """
    Get available hyperparameter study results for a specific algorithm

    Args:
        algorithm: Algorithm name (ppo, a2c, sac, td3)

    Returns:
        List of hyperparameter study files with metadata
    """
    try:
        studies_dir = ws.hyperopt_dir()

        if not os.path.exists(studies_dir):
            return {"studies": []}

        studies = []
        algorithm_lower = algorithm.lower()

        # Search for JSON files matching the algorithm
        for filename in os.listdir(studies_dir):
            if filename.startswith(f"best_params_{algorithm_lower}_") and filename.endswith(".json"):
                file_path = os.path.join(studies_dir, filename)

                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Extract study info
                    study_info = {
                        "filename": filename,
                        "file_path": file_path,
                        "study_name": data.get("study_name", "Unknown"),
                        "best_value": data.get("best_value", 0.0),
                        "best_params": data.get("best_params", {}),
                        "n_trials": data.get("n_trials", 0),
                        "timestamp": data.get("timestamp", "Unknown"),
                        "algorithm": data.get("algorithm", algorithm_lower)
                    }

                    studies.append(study_info)
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")
                    continue

        # Sort by timestamp (newest first)
        studies.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "algorithm": algorithm,
            "studies": studies,
            "total": len(studies)
        }

    except Exception as e:
        logger.error(f"Error fetching hyperparameter studies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _check_training_data_freshness(phase: int, stale_threshold: int = 10) -> None:
    """
    Pre-flight check: refuse to start training when the underlying data is
    missing or badly stale. Without this guard the trainer happily produces a
    flat-equity model on empty/old data and the user only finds out an hour
    later.

    Raises HTTPException(400) when the situation looks unrecoverable.
    """
    # BIST data must exist and be reasonably fresh.
    bist_file = os.path.join(_settings.BIST_DIR, 'raw_stock_data.csv')
    if not os.path.exists(bist_file):
        raise HTTPException(
            status_code=400,
            detail="BIST hisse verisi yok. Önce 'Veri' sayfasından güncelleyin."
        )
    try:
        from data.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        df = fetcher.load_data('raw_stock_data.csv')
        dates = pd.to_datetime(df.index.get_level_values('date'))
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        last_date_str = dates.max().strftime('%Y-%m-%d')
        missing = _count_missing_trading_days(last_date_str)
        if missing > stale_threshold:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"BIST verisi bayat: son tarih {last_date_str}, "
                    f"{missing} iş günü eksik. 'Veri' sayfasından güncelleyin."
                )
            )
    except HTTPException:
        raise
    except Exception as exc:  # corrupt CSV, unreadable, etc.
        raise HTTPException(
            status_code=400,
            detail=f"BIST verisi okunamadı: {exc}"
        )

    # Phase 2 additionally needs fundamental + macro CSVs.
    if phase == 2:
        fund_file = os.path.join(_settings.FUNDAMENTAL_DIR, 'fundamental_data.csv')
        macro_file = os.path.join(_settings.MACRO_DIR, 'macro_data.csv')
        missing_sources = []
        if not os.path.exists(fund_file):
            missing_sources.append('fundamental_data.csv')
        if not os.path.exists(macro_file):
            missing_sources.append('macro_data.csv')
        if missing_sources:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Faz 2 için eksik veri: "
                    + ", ".join(missing_sources)
                    + ". 'Veri' sayfasından güncelleyin."
                )
            )


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    user: RequireWriter,
):
    """
    Start model training in background

    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks
        user: Aktif kullanici — egitim onun calisma alanina yazilir (viewer yapamaz)
    """
    # Veri tazeliği ön kontrolü — boş/bayat veriyle eğitim başlatmayı engeller.
    _check_training_data_freshness(request.phase)

    user_id = ws.current_user_id()
    key = _training_key(user_id)

    async with _training_lock:  # (#23) prevent race between concurrent requests
        state = get_training_state(user_id)
        if state["is_training"]:
            raise HTTPException(
                status_code=400,
                detail="Training already in progress"
            )

        # On tahmin: kullanici "Baslat"a bastigi anda ne kadar surecegini gorsun.
        # Tahmin uretilemezse egitim yine baslar — ETA sus kalir.
        try:
            prior = training_eta.estimate(
                algorithm=request.algorithm,
                phase=request.phase,
                total_timesteps=request.total_timesteps,
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning(f"ETA on tahmini uretilemedi ({exc}) — egitim tahminsiz baslatiliyor")
            prior = None

        # Reset training state inside lock
        _training_states[key] = {
            "is_training": True,
            "state": "running",
            "current_step": 0,
            "total_steps": request.total_timesteps,
            "start_time": datetime.now().isoformat(),
            "metrics": {},
            "error": None,
            "config": request.dict(),
            "phase_name": "preparing",
            "run_start_ts": time.time(),
            "learn_start_ts": None,
            "estimate": prior,
        }

    # Start training in background (outside lock — long-running).
    # Kullanici kimligi acikca tasinir: arka plan gorevi istek baglamini
    # (ContextVar) devralmaz, calisma alani yolu buna gore cozulur.
    #
    # `background_tasks.add_task` DEGIL: Starlette gorevi ayni ASGI cagrisinin
    # icinde bekler ve pano backend'i in-process ASGI ile cagirir, dolayisiyla
    # bu uc egitim bitene kadar donmuyordu (olculdu: 3000 timestep -> 15.0 sn).
    # Ayrica `run_training` async oldugu icin bloke eden model.learn() dogrudan
    # event loop'u kilitliyor, tum pano donuyordu.
    jobs.spawn(
        lambda: run_training(request, user_id),
        name=f"training-{key}",
    )

    return TrainingResponse(
        message="Training started successfully",
        training_id=_training_states[key]["start_time"],
        status="started"
    )


@router.get("/train/estimate", response_model=TrainingEstimate)
async def estimate_training_time(
    algorithm: str = "ppo",
    phase: int = 1,
    total_timesteps: int = 50_000,
):
    """Egitim baslatmadan once tahmini sureyi dondur.

    Pano bunu form degistikce cagirir; kullanici "Baslat"a basmadan once
    ne kadar surecegini gorur. Tahmin ayni makinedeki gecmis kosumlardan
    ogrenilir, gecmis yoksa yerlesik katsayilardan (`confidence='default'`).
    """
    if total_timesteps <= 0:
        raise HTTPException(status_code=400, detail="total_timesteps must be positive")
    try:
        result = training_eta.estimate(
            algorithm=algorithm, phase=phase, total_timesteps=total_timesteps,
        )
    except Exception as exc:
        logger.error(f"ETA tahmini basarisiz: {exc}")
        raise HTTPException(status_code=500, detail="Sure tahmini uretilemedi")
    return TrainingEstimate(**{k: v for k, v in result.items() if k != "step_cost"})


@router.get("/train/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status (aktif kullaniciya ait)"""
    training_state = get_training_state()

    # SB3 `total_timesteps`'i tam tutturmaz: ogrenme `n_steps` bloklariyla
    # ilerledigi icin son blok hedefi asabiliyor (olculdu: 1.001472). Sema
    # `progress`'i [0,1] ile sinirliyor, dolayisiyla ham oran YANIT
    # DOGRULAMASINI dusuruyordu -> /train/status 500 veriyor, pano egitim
    # boyunca ilerleme ve ETA gosteremiyordu. Oran burada kirpilir; sema
    # sozlesmesi (le=1) oldugu gibi kalir.
    progress = 0.0
    if training_state["total_steps"] > 0:
        progress = min(1.0, training_state["current_step"] / training_state["total_steps"])

    # ETA: devam eden kosumda gozlenen hizdan, isinma penceresinde on tahminden.
    # ETA bir ek ozellik; pano ilerleme takibi bu uca bagli oldugu icin buradaki
    # hicbir hata durumu dusurmemeli — hepsi yutulur, alanlar bos doner.
    try:
        live = training_eta.live_eta(training_state)
    except Exception as exc:
        logger.warning(f"Canli ETA hesaplanamadi ({exc}) — durum ETA'siz donuyor")
        live = {"eta_seconds": None, "eta_text": None, "finish_at": None,
                "steps_per_sec": None, "estimated_total_seconds": None,
                "eta_source": None}

    run_start = training_state.get("run_start_ts")
    elapsed = (time.time() - run_start) if run_start else None

    prior_model = None
    prior = training_state.get("estimate")
    if prior:
        try:
            prior_model = TrainingEstimate(
                **{k: v for k, v in prior.items() if k != "step_cost"}
            )
        except Exception as exc:
            # Eksik/eski sekilli tahmin sozlugu — yut, durumu dondurmeye devam et.
            logger.warning(f"On tahmin sozlugu cozulemedi ({exc}) — status ETA'siz donuyor")

    return TrainingStatus(
        is_training=training_state["is_training"],
        state=training_state.get("state", "idle"),
        current_step=training_state["current_step"],
        total_steps=training_state["total_steps"],
        progress=progress,
        start_time=training_state.get("start_time"),
        metrics=training_state.get("metrics", {}),
        error=training_state.get("error"),
        n_symbols=training_state.get("n_symbols"),
        warnings=training_state.get("warnings") or [],
        phase_name=training_state.get("phase_name"),
        elapsed_seconds=elapsed,
        elapsed_text=training_eta.humanize(elapsed) if elapsed is not None else None,
        eta_seconds=live["eta_seconds"],
        eta_text=live["eta_text"],
        finish_at=live["finish_at"],
        steps_per_sec=live["steps_per_sec"],
        estimated_total_seconds=live["estimated_total_seconds"],
        eta_source=live["eta_source"],
        estimate=prior_model,
    )


# /models cache — the home page polls this every 30s and it re-reads the
# whole models dir plus every metrics JSON each time. Invalidate on either
# a dir-mtime change (new/re-trained model) or a short TTL, so an in-place
# metrics rewrite that doesn't bump the dir mtime is still picked up soon.
_models_cache: Dict[str, Any] = {"key": None, "value": None, "ts": 0.0}
_MODELS_CACHE_TTL_SEC = 10


def _models_cache_key() -> tuple:
    def _mtime(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return None
    # Kullanici kimligi anahtarin parcasi: iki kullanici birbirinin model
    # listesini onbellekten gormemeli.
    dirs = ws.read_dirs("models") + ws.read_dirs("results")
    return (ws.current_user_id(), tuple(_mtime(d) for d in dirs))


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all trained models (kullanicinin calisma alani + ortak eski modeller)"""
    model_dirs = ws.read_dirs("models")

    if not model_dirs:
        return []

    cache_key = _models_cache_key()
    fresh = (time.time() - _models_cache["ts"]) < _MODELS_CACHE_TTL_SEC
    if _models_cache["key"] == cache_key and _models_cache["value"] is not None and fresh:
        return _models_cache["value"]

    models = []
    seen: set[str] = set()

    for models_dir in model_dirs:
        for filename in sorted(os.listdir(models_dir)):
            if not filename.endswith(".zip") or filename in seen:
                continue
            seen.add(filename)  # kendi alanindaki dosya, eski ortak dosyayi golgeler
            model_path = os.path.join(models_dir, filename)

            # Try to load metrics from results
            model_name = filename.replace(".zip", "")
            metrics_file = ws.find_file("results", f"{model_name}_metrics.json")

            metrics = {}
            if metrics_file:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

            # Algorithm/phase precedence: metrics.json → filename prefix
            algorithm = metrics.get("algorithm")
            phase = metrics.get("phase")
            if algorithm is None or phase is None:
                parts = model_name.split("_")
                if algorithm is None and parts:
                    first = parts[0].lower()
                    if first in {"ppo", "a2c", "td3", "sac"}:
                        algorithm = first.upper()
                if phase is None:
                    for p in parts:
                        if p.startswith("phase") and p[5:].isdigit():
                            phase = int(p[5:])
                            break

            models.append(ModelInfo(
                name=model_name,
                path=model_path,
                created_at=datetime.fromtimestamp(
                    os.path.getctime(model_path)
                ).isoformat(),
                algorithm=algorithm,
                phase=phase,
                metrics=metrics
            ))

    _models_cache["key"] = cache_key
    _models_cache["value"] = models
    _models_cache["ts"] = time.time()
    return models


@router.get("/models/{model_name}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    model_name = sanitize_model_name(model_name)
    metrics_file = ws.find_file("results", f"{model_name}_metrics.json")

    if not metrics_file:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for model: {model_name}"
        )

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return ModelMetrics(**metrics)


@router.post("/data/generate")
async def generate_data(
    user: RequireWriter,
    phase: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Generate fresh stock data with indicators

    Args:
        phase: Trading phase (1, 2, or 3)
        start_date: Start date in YYYY-MM-DD format (default: 2018-01-01)
        end_date: End date in YYYY-MM-DD format (default: today)
    """
    try:
        from data.data_fetcher import DataFetcher
        from data.bist30_symbols import get_symbols
        from data.technical_indicators import add_indicators_to_multi_symbol_df

        # Get symbols for phase
        symbols = get_symbols(phase=phase)

        # Create fetcher with date range
        fetcher = DataFetcher(
            start_date=start_date or "2018-01-01",
            end_date=end_date if end_date is not None else None
        )

        # Fetch data
        df = fetcher.fetch_stock_data(symbols, save=True)

        # Clean data
        df = fetcher.clean_data(df)

        # Add technical indicators
        df = add_indicators_to_multi_symbol_df(df)

        # Save
        fetcher.save_data(df, 'stock_data_with_indicators.csv')

        # Get train/val/test split info
        train_df, val_df, test_df = fetcher.split_data(df)

        return {
            "status": "success",
            "message": "Data generated successfully",
            "symbols": symbols,
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "columns": df.columns.tolist(),
            "date_range": {
                "start": str(df.index.get_level_values('date').min().date()),
                "end": str(df.index.get_level_values('date').max().date())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/info")
async def get_data_info():
    """Get information about current data"""
    try:
        from data.data_fetcher import DataFetcher

        fetcher = DataFetcher()

        # Try to load existing data
        if not os.path.exists(f'{_settings.DATA_DIR}/stock_data_with_indicators.csv'):
            return {
                "status": "no_data",
                "message": "No data found. Please generate data first."
            }

        df = fetcher.load_data('stock_data_with_indicators.csv')
        train_df, val_df, test_df = fetcher.split_data(df)

        return {
            "status": "exists",
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "columns": df.columns.tolist(),
            "symbols": df.index.get_level_values('symbol').unique().tolist(),
            "date_range": {
                "start": str(df.index.get_level_values('date').min().date()),
                "end": str(df.index.get_level_values('date').max().date())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/list")
async def list_all_datasets():
    """List all available datasets in data folder"""
    try:
        from data.data_fetcher import DataFetcher

        data_dir = "data"
        if not os.path.exists(data_dir):
            return {"datasets": []}

        # Scan all CSV files in subdirectories
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

        datasets = []
        for csv_path in csv_files:
            try:
                file_stat = os.stat(csv_path)
                rel_path = os.path.relpath(csv_path, data_dir)
                subdir = os.path.basename(os.path.dirname(csv_path))

                entry = {
                    "filepath": csv_path,
                    "filename": os.path.basename(csv_path),
                    "subdir": subdir,
                    "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                }

                # Try to get row/date info from BIST files
                if subdir == "bist":
                    fetcher = DataFetcher()
                    df = fetcher.load_data(os.path.basename(csv_path))
                    if df is not None and len(df) > 0:
                        entry.update({
                            "total_rows": len(df),
                            "symbols": df.index.get_level_values('symbol').unique().tolist(),
                            "date_range": {
                                "start": str(df.index.get_level_values('date').min().date()),
                                "end": str(df.index.get_level_values('date').max().date()),
                            },
                        })

                datasets.append(entry)
            except Exception as e:
                try:
                    file_stat = os.stat(csv_path)
                    datasets.append({
                        "filepath": csv_path,
                        "filename": os.path.basename(csv_path),
                        "subdir": os.path.basename(os.path.dirname(csv_path)),
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "error": str(e),
                    })
                except Exception:
                    pass

        return {
            "status": "success",
            "count": len(datasets),
            "datasets": datasets
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DataUpdateRequest(BaseModel):
    sources: List[str]
    mode: str = "incremental"
    start_date: Optional[str] = None
    symbols: Optional[List[str]] = None        # BIST hisse listesi; None → varsayılan faz sembolleri
    gold_source: str = "borsapy"               # "borsapy" veya "yfinance"
    gold_metrics: Optional[List[str]] = None   # None → GoldFetcher.DEFAULT_METRICS


def _count_missing_trading_days(last_date_str: str) -> int:
    """Verilen tarihten bugüne kadar iş günü sayısını döndür."""
    today = datetime.now().date()
    last = datetime.strptime(last_date_str, '%Y-%m-%d').date()
    return sum(
        1 for i in range(1, (today - last).days + 1)
        if (last + timedelta(days=i)).weekday() < 5
    )


@router.get("/data/status")
async def get_data_status():
    """Tüm veri kaynaklarının mevcut durumunu döndür (son tarih, eksik gün)."""
    logger.info("GET /data/status çağrıldı")
    try:
        status = {}

        # BIST hisseleri
        bist_file = os.path.join(_settings.BIST_DIR, 'raw_stock_data.csv')
        if os.path.exists(bist_file):
            try:
                from data.data_fetcher import DataFetcher
                fetcher = DataFetcher()
                df = fetcher.load_data('raw_stock_data.csv')
                dates = pd.to_datetime(df.index.get_level_values('date'))
                if dates.tz is not None:
                    dates = dates.tz_localize(None)
                last_date = dates.max().strftime('%Y-%m-%d')
                status['bist_stocks'] = {
                    'exists': True,
                    'file': bist_file,
                    'last_date': last_date,
                    'missing_days': _count_missing_trading_days(last_date),
                    'symbols': df.index.get_level_values('symbol').unique().tolist(),
                    'label': 'BIST-30 Hisseleri',
                }
            except Exception as exc:
                status['bist_stocks'] = {
                    'exists': True, 'file': bist_file,
                    'error': str(exc), 'label': 'BIST-30 Hisseleri',
                }
        else:
            status['bist_stocks'] = {
                'exists': False, 'file': bist_file,
                'last_date': None, 'missing_days': None,
                'symbols': [], 'label': 'BIST-30 Hisseleri',
            }

        # Altın & döviz — her kaynak için ayrı dosya taranır
        from data.gold_fetcher import GoldFetcher, GOLD_DIR
        gold_files = glob.glob(os.path.join(GOLD_DIR, 'gold_*.csv'))
        if gold_files:
            gold_status_list = []
            for gf in sorted(gold_files):
                try:
                    df = GoldFetcher.load_data(gf)
                    if df is not None and not df.empty:
                        dates = pd.to_datetime(df.index.get_level_values('date'))
                        last_date = dates.max().strftime('%Y-%m-%d')
                        gold_status_list.append({
                            'exists': True,
                            'file': gf,
                            'last_date': last_date,
                            'missing_days': _count_missing_trading_days(last_date),
                            'symbols': df.index.get_level_values('symbol').unique().tolist(),
                        })
                except Exception as exc:
                    gold_status_list.append({'exists': True, 'file': gf, 'error': str(exc)})
            status['gold'] = {
                'exists': True,
                'files': gold_status_list,
                'label': 'Altın & Döviz',
            }
        else:
            status['gold'] = {
                'exists': False, 'files': [],
                'last_date': None, 'missing_days': None,
                'symbols': [], 'label': 'Altın & Döviz',
            }

        # Makro veri / EVDS (Faiz, CPI, PPI) + yfinance (EUR/TRY, BIST-100)
        macro_file = os.path.join(_settings.MACRO_DIR, 'macro_data.csv')
        if os.path.exists(macro_file):
            try:
                df = pd.read_csv(macro_file)
                date_col = 'date' if 'date' in df.columns else df.columns[0]
                dates = pd.to_datetime(df[date_col])
                last_date = dates.max().strftime('%Y-%m-%d')
                status['macro'] = {
                    'exists': True,
                    'file': macro_file,
                    'last_date': last_date,
                    'missing_days': _count_missing_trading_days(last_date),
                    'symbols': [c for c in df.columns if c != date_col],
                    'label': 'EVDS + Makro (Faiz · CPI · PPI · EUR/TRY · BIST-100)',
                }
            except Exception as exc:
                status['macro'] = {
                    'exists': True, 'file': macro_file,
                    'error': str(exc), 'label': 'EVDS + Makro',
                }
        else:
            status['macro'] = {
                'exists': False, 'file': macro_file,
                'last_date': None, 'missing_days': None,
                'symbols': [], 'label': 'EVDS + Makro (Faiz · CPI · PPI · EUR/TRY · BIST-100)',
            }

        # Fundamental veri (yfinance — ROE, ROA, P/E, P/B vb.)
        fund_file = os.path.join(_settings.FUNDAMENTAL_DIR, 'fundamental_data.csv')
        if os.path.exists(fund_file):
            try:
                df = pd.read_csv(fund_file, index_col='symbol')
                status['fundamental'] = {
                    'exists': True,
                    'file': fund_file,
                    'last_date': None,   # fundamental veri tarihsiz (snapshot)
                    'missing_days': None,
                    'symbols': df.index.tolist(),
                    'label': 'Fundamental (ROE · ROA · P/E · P/B · D/E)',
                }
            except Exception as exc:
                status['fundamental'] = {
                    'exists': True, 'file': fund_file,
                    'error': str(exc), 'label': 'Fundamental',
                }
        else:
            status['fundamental'] = {
                'exists': False, 'file': fund_file,
                'last_date': None, 'missing_days': None,
                'symbols': [], 'label': 'Fundamental (ROE · ROA · P/E · P/B · D/E)',
            }

        return {'status': status, 'as_of': datetime.now().strftime('%Y-%m-%d')}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/data/update")
async def update_data(request: DataUpdateRequest, user: RequireWriter):
    """Seçili veri kaynaklarını güncelle (incremental veya tam)."""
    try:
        results = {}
        mode = request.mode
        start_date = request.start_date or '2018-01-01'

        if 'bist_stocks' in request.sources:
            from data.data_fetcher import DataFetcher
            from data.bist30_symbols import get_symbols
            from data.technical_indicators import add_indicators_to_multi_symbol_df

            symbols = request.symbols if request.symbols else get_symbols(phase=2)
            fetcher = DataFetcher(start_date=start_date)

            if mode == 'full':
                df = fetcher.fetch_stock_data(symbols, save=True)
                df = fetcher.clean_data(df)
                df = add_indicators_to_multi_symbol_df(df)
                fetcher.save_data(df, 'stock_data_with_indicators.csv')
                results['bist_stocks'] = {
                    'mode': 'full', 'new_rows': len(df), 'total_rows': len(df),
                }
            else:
                result = fetcher.fetch_incremental(symbols, 'raw_stock_data.csv')
                try:
                    df = fetcher.load_data('raw_stock_data.csv')
                    df = fetcher.clean_data(df)
                    df = add_indicators_to_multi_symbol_df(df)
                    fetcher.save_data(df, 'stock_data_with_indicators.csv')
                except Exception as exc:
                    logger.warning(f"İndikatörler yeniden hesaplanamadı: {exc}")
                results['bist_stocks'] = result

        if 'gold' in request.sources:
            from data.gold_fetcher import GoldFetcher

            fetcher = GoldFetcher(
                source=request.gold_source,
                metrics=request.gold_metrics or None,
                start_date=start_date,
            )

            if mode == 'full':
                df = fetcher.fetch_all(save=False)
                fetcher.save_data(df)  # saves to data/gold/gold_{source}.csv
                results['gold'] = {'mode': 'full', 'new_rows': len(df), 'total_rows': len(df)}
            else:
                result = fetcher.fetch_incremental()  # uses default_filepath()
                results['gold'] = result

        if 'macro' in request.sources:
            from data.macro_fetcher import MacroDataFetcher
            macro_api_key = _settings.EVDS_API_KEY or os.environ.get('EVDS_API_KEY', '')
            macro_fetcher = MacroDataFetcher(api_key=macro_api_key, start_date=start_date)
            try:
                df = macro_fetcher.fetch_macro_data(save=True)
                results['macro'] = {
                    'mode': 'full',
                    'rows': len(df) if df is not None else 0,
                }
            except Exception as exc:
                results['macro'] = {'error': str(exc)}

        if 'fundamental' in request.sources:
            from data.fundamental_fetcher import FundamentalDataFetcher
            from data.bist30_symbols import get_symbols
            symbols = request.symbols if request.symbols else get_symbols(phase=2)
            fund_fetcher = FundamentalDataFetcher()
            try:
                df = fund_fetcher.fetch_fundamental_data(symbols, save=True)
                results['fundamental'] = {
                    'mode': 'full',
                    'rows': len(df),
                    'symbols': df.index.tolist(),
                }
            except Exception as exc:
                results['fundamental'] = {'error': str(exc)}

        return {'status': 'success', 'results': results}

    except Exception as exc:
        logger.error(f"Veri güncelleme hatası: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# /data/earliest icin in-memory TTL cache (kaynak basina, 24h)
_EARLIEST_CACHE: Dict[str, Dict[str, Any]] = {}
_EARLIEST_TTL_SEC = 24 * 60 * 60


@router.get("/data/earliest")
def get_earliest_date(source: str = "borsapy"):
    """
    Belirtilen kaynak için her metrikte mevcut en eski veri tarihini döndür.

    source: "borsapy" | "yfinance"

    Yanıt:
      {
        "source": "borsapy",
        "earliest": "2010-01-04",          # tüm metrikler arasındaki en erken tarih
        "per_metric": {
          "gram_altin_try": "2010-01-04",
          "usd_try":        "2010-01-04",
          ...
        }
      }
    """
    # Cache hit: TTL icinde olan kayitlari direkt dondur
    cached = _EARLIEST_CACHE.get(source)
    if cached and (time.time() - cached["ts"]) < _EARLIEST_TTL_SEC:
        return cached["payload"]

    per_metric: Dict[str, str] = {}

    try:
        if source == "borsapy":
            from borsapy import FX
            checks = {
                "gram_altin_try": "gram-altin",
                "ons_altin_try":  "ons-altin",
                "usd_try":        "USD",
                "eur_try":        "EUR",
            }
            for metric, symbol in checks.items():
                try:
                    df = FX(symbol).history(period="max")
                    if df is not None and not df.empty:
                        idx = pd.to_datetime(df.index)
                        if idx.tz is not None:
                            idx = idx.tz_localize(None)
                        per_metric[metric] = str(idx.min().date())
                except Exception as exc:
                    per_metric[metric] = f"hata: {str(exc)[:60]}"

        elif source == "yfinance":
            import yfinance as yf
            checks = {
                "ons_altin_usd":  "GC=F",
                "gram_altin_usd": "GC=F",   # aynı kaynak, GC=F / 31.1
                "usd_try":        "TRY=X",
                "eur_try":        "EURTRY=X",
                "bist_stocks":    "THYAO.IS",
            }
            seen: Dict[str, str] = {}
            for metric, symbol in checks.items():
                if symbol in seen:
                    per_metric[metric] = seen[symbol]
                    continue
                try:
                    df = yf.Ticker(symbol).history(period="max")
                    if df is not None and not df.empty:
                        idx = pd.to_datetime(df.index)
                        if idx.tz is not None:
                            idx = idx.tz_localize(None)
                        date_str = str(idx.min().date())
                        per_metric[metric] = date_str
                        seen[symbol] = date_str
                except Exception as exc:
                    per_metric[metric] = f"hata: {str(exc)[:60]}"
        else:
            raise HTTPException(status_code=400, detail=f"Geçersiz kaynak: {source!r}")

        # Hata içermeyen tarihlerden en eskiyi bul
        valid_dates = [v for v in per_metric.values() if not v.startswith("hata")]
        earliest = min(valid_dates) if valid_dates else None

        # BIST tabani: bu tarihten eski istekler cekme aninda buraya cekilir
        # (2005 oncesi yfinance verisi TL/YTL denominasyonu yuzunden bozuk).
        # Panonun kullaniciya bildirebilmesi icin yanitta tasiniyor.
        from data.data_fetcher import BIST_MIN_START_DATE

        payload = {
            "source":     source,
            "earliest":   earliest,
            "per_metric": per_metric,
            "bist_min_start": BIST_MIN_START_DATE,
        }
        # Cache'e yaz (sadece basarili sonuclar — hata-only durumda da yazariz, TTL sonrasi yeniden dener)
        _EARLIEST_CACHE[source] = {"ts": time.time(), "payload": payload}
        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/models/{model_name}")
async def delete_model(model_name: str, user: RequireWriter):
    """Egitilmis modeli sil.

    Iki katman var:

    - **Kendi calisma alanindaki model** — sahibi siler (RequireWriter).
    - **Ortak (kullanici oncesi) model** — herkese gorunur oldugu icin bir
      kullanici digerlerinin gordugunu silememeli; YALNIZCA YONETICI siler.

    Ikinci madde Faz 7'de "kimse silemez" idi. Pratikte bu, kullanici sistemi
    oncesinde egitilmis deneme modellerini panodan temizlemenin HICBIR yolunu
    birakmiyordu — tek cikis kapsayicinin baglandigi dizine elle girmekti.
    Yonetici zaten operatordur (hesap silebilir, parola sifirlayabilir);
    ortak yapitlarin temizligi de onun isi. Silme ortak dizini etkiledigi
    icin denetim kaydina yazilir.
    """
    model_name = sanitize_model_name(model_name)
    own_path = os.path.join(ws.models_dir(), f"{model_name}.zip")

    if os.path.exists(own_path):
        model_path, is_shared = own_path, False
    else:
        found = ws.find_file("models", f"{model_name}.zip")
        if not found:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_name}"
            )
        if user.role != Role.ADMIN:
            raise HTTPException(
                status_code=403,
                detail=("Bu model ortak (kullanici oncesi) dizinde; herkese gorunur. "
                        "Yalnizca yonetici silebilir."),
            )
        model_path, is_shared = found, True

    os.remove(model_path)

    # Modelin yaninda duran tanim dosyasi (`<model>.meta.json`) da gider —
    # `.zip` silinip tanim kalirsa dizinde sahibi olmayan bir dosya birikir.
    from app.services.daily_trading import model_meta_path
    meta_file = model_meta_path(model_path)
    if os.path.exists(meta_file):
        os.remove(meta_file)

    # Metrik JSON'u modelin BULUNDUGU katmandan silinir: ortak bir modelin
    # metrigi ortak results/ altindadir, kullanicininki kendi alaninda.
    metrics_dir = ws.shared_dir("results") if is_shared else ws.results_dir()
    metrics_file = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    if is_shared:
        _audit_shared_model_delete(user, model_name)

    return {
        "message": f"Model {model_name} deleted successfully",
        "shared": is_shared,
    }


def _audit_shared_model_delete(user, model_name: str) -> None:
    """Ortak dizinden silme denetim kaydina yazilir — herkesi etkiler.

    Denetim yazimi basarisiz olursa SILME GERI ALINMAZ (dosya zaten gitti);
    hata yalnizca loglanir, kullaniciya basarisiz gibi gosterilmez.
    """
    try:
        from app.auth import service
        from app.auth.db import session_scope

        with session_scope() as db:
            service.audit(db, "model.delete_shared", user=user, target=model_name,
                          detail={"scope": "shared"})
    except Exception as exc:  # pragma: no cover - denetim kaydi kritik yol degil
        logger.warning(f"Ortak model silme denetim kaydina yazilamadi: {exc}")


async def run_training(request: TrainingRequest, user_id: Optional[str] = None):
    """
    Background task for model training

    Args:
        request: Training configuration
        user_id: Egitimi baslatan kullanici; model/metrik/log dosyalari onun
                 calisma alanina yazilir. None = auth kapali (eski davranis).
    """
    state_key = _training_key(user_id)
    training_state = _training_states.setdefault(state_key, _empty_training_state())

    # ContextVar arka plan gorevine tasinmadigi icin calisma alani baglami
    # burada acikca kurulur — asagidaki tum ws.* cagrilarini etkiler.
    with ws.use_workspace(user_id):
        await _run_training_inner(request, training_state)


async def _run_training_inner(request: TrainingRequest, training_state: Dict[str, Any]):
    import time

    # Track total training time
    training_start_time = time.time()
    # ETA fazlari: hazirlik/ogrenme/degerlendirme ayri olculur, kosum bitince
    # gecmise yazilir ve bir sonraki tahminin dayanagi olur.
    training_state.setdefault("run_start_ts", training_start_time)
    training_state["phase_name"] = "preparing"
    learn_start_time = None
    learn_end_time = None

    try:
        # Import training modules
        from data.data_fetcher import DataFetcher
        from data.bist30_symbols import get_symbols
        from data.technical_indicators import add_indicators_to_multi_symbol_df
        from env.trading_env import make_env
        from stable_baselines3 import A2C, PPO, TD3, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        # Custom callback to update training state
        class TrainingStateCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)

            def _on_step(self) -> bool:
                # Kullaniciya ait durum sozlugu closure ile tasinir.
                training_state["current_step"] = self.num_timesteps
                return True

        # Prepare data
        symbols = get_symbols(phase=request.phase)
        fetcher = DataFetcher()

        try:
            df = fetcher.load_data('stock_data_with_indicators.csv')
        except FileNotFoundError:
            df = fetcher.fetch_stock_data(symbols, save=True)
            df = fetcher.clean_data(df)
            df = add_indicators_to_multi_symbol_df(df)
            fetcher.save_data(df, 'stock_data_with_indicators.csv')

        train_df, val_df, test_df = fetcher.split_data(df)

        # Gozlem uzayi sembol sayisina bagli (1 + n + 5n + 5n). `split_data`
        # KRONOLOJIK bolduğu icin, paneldeki sembollerin gecmisleri esit
        # degilse bolumlerin sembol sayisi FARKLI cikar. Canli ornek:
        #   train 2018-01-01..2024-01-16 ->  5 sembol ->  56 ozellik
        #   test  2025-05-14..2026-08-28 -> 30 sembol -> 331 ozellik
        # Model 56 ile egitilip 331 ile degerlendiriliyordu; SB3 `predict`
        # asamasinda patliyordu ve HICBIR kosum tamamlanamiyordu.
        #
        # Modelin evreni EGITIM bolumunun sembolleridir; degerlendirme ve
        # dogrulama ona hizalanir. Sessizce yapmak yanlis olur — dusen sembol
        # varsa durum kaydina yazilir ve panoda gorunur.
        train_symbols = set(train_df.index.get_level_values('symbol').unique())
        panel_symbols = set(df.index.get_level_values('symbol').unique())
        dropped = sorted(panel_symbols - train_symbols)

        if dropped:
            def _align(frame):
                mask = frame.index.get_level_values('symbol').isin(train_symbols)
                return frame[mask]

            val_df, test_df = _align(val_df), _align(test_df)
            warning = (
                f"Panelde {len(panel_symbols)} sembol var ama egitim penceresinde "
                f"yalnizca {len(train_symbols)} tanesinin gecmisi bulunuyor; "
                f"model bu {len(train_symbols)} sembolle egitildi ve ayni "
                f"sembollerle degerlendirildi. Disarida kalan: {', '.join(dropped)}. "
                f"Tum sembolleri kapsamak icin Veri sayfasindan tam gecmisi indirin."
            )
            logger.warning(warning)
            training_state["warnings"] = [warning]
        training_state["n_symbols"] = len(train_symbols)

        # Load Phase 2 data if needed
        fundamental_df = None
        macro_df = None
        reward_type = 'simple'  # Default
        
        if request.phase == 2:
            try:
                from data.fundamental_fetcher import FundamentalDataFetcher
                from data.macro_fetcher import MacroDataFetcher
                
                # Load fundamental data
                fund_fetcher = FundamentalDataFetcher()
                try:
                    fundamental_df = fund_fetcher.load_data('fundamental_data.csv')
                    logger.info(f"Loaded fundamental data: {fundamental_df.shape}")
                except FileNotFoundError:
                    logger.warning("Fundamental data not found, fetching...")
                    fundamental_df = fund_fetcher.fetch_fundamental_data(symbols, save=True)
                
                # Load macro data  
                macro_fetcher = MacroDataFetcher(api_key=_settings.EVDS_API_KEY or None)
                try:
                    macro_df = macro_fetcher.load_data('macro_data.csv')
                    logger.info(f"Loaded macro data: {macro_df.shape}")
                except FileNotFoundError:
                    logger.warning("Macro data not found, fetching...")
                    macro_df = macro_fetcher.fetch_macro_data(save=True)
                
                # Phase 2 uses PSR reward
                reward_type = 'psr'
                logger.info("Phase 2: Using PSR reward with fundamental + macro data")
                
            except Exception as e:
                logger.error(f"Error loading Phase 2 data: {e}")
                raise

        # Create environment factory function
        def make_training_env():
            return make_env(
                df=train_df,
                initial_balance=request.initial_balance,
                commission_rate=request.commission_rate,
                max_shares_per_trade=request.max_shares_per_trade,
                phase=request.phase,
                reward_type=reward_type,
                fundamental_df=fundamental_df,
                macro_df=macro_df
            )

        # Create one instance to get dimensions
        temp_env = make_training_env()
        action_space_shape = temp_env.action_space.shape
        action_dim = action_space_shape[0] if action_space_shape is not None else 0
        n_stocks = temp_env.n_stocks
        logger.info(f"Environment dimensions: action_dim={action_dim}, n_stocks={n_stocks}")

        # Vectorize environment (create fresh instance)
        env = DummyVecEnv([make_training_env])

        # Create model based on algorithm with optimized hyperparameters
        model_class = {
            "A2C": A2C,
            "PPO": PPO,
            "TD3": TD3,
            "SAC": SAC
        }.get(request.algorithm, PPO)  # Default to PPO instead of A2C

        # Setup TensorBoard logging
        os.makedirs(ws.logs_dir(), exist_ok=True)
        model_name = f"{request.algorithm.lower()}_phase{request.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load hyperparameters if study is specified
        hyperparams = {}
        if request.hyperparameter_study:
            study_path = os.path.join(ws.hyperopt_dir(), request.hyperparameter_study)
            if os.path.exists(study_path):
                with open(study_path, 'r') as f:
                    study_data = json.load(f)
                    hyperparams = study_data.get("best_params", {})
                    logger.info(f"Loaded hyperparameters from {request.hyperparameter_study}")
                    logger.info(f"Best params: {hyperparams}")
            else:
                logger.warning(f"Hyperparameter study file not found: {study_path}")

        # Algorithm-specific hyperparameters optimized for trading
        # Each algorithm has its own optimal learning rate
        actual_learning_rate = None  # Track actual LR used

        if request.algorithm == "PPO":
            # PPO: Most stable for single-threaded training
            # Use hyperparameters from study or defaults
            actual_learning_rate = hyperparams.get('learning_rate', 0.0003)
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=actual_learning_rate,
                n_steps=hyperparams.get('n_steps', 2048),
                batch_size=hyperparams.get('batch_size', 64),
                n_epochs=hyperparams.get('n_epochs', 10),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.2),
                ent_coef=hyperparams.get('ent_coef', 0.15),
                vf_coef=hyperparams.get('vf_coef', 0.5),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                tensorboard_log=ws.logs_dir(),
                verbose=1
            )
        elif request.algorithm == "A2C":
            # A2C: Synchronous Advantage Actor-Critic
            # Best for: Fast training, stable convergence, trading environments
            # Paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)

            # Use hyperparameters from study or trading-optimized defaults
            actual_learning_rate = hyperparams.get('learning_rate', 0.0007)
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=actual_learning_rate,
                n_steps=hyperparams.get('n_steps', 256),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                ent_coef=hyperparams.get('ent_coef', 0.01),
                vf_coef=hyperparams.get('vf_coef', 0.25),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                normalize_advantage=True,
                use_rms_prop=True,
                rms_prop_eps=hyperparams.get('rms_prop_eps', 1e-5),
                tensorboard_log=ws.logs_dir(),
                verbose=1
            )
        elif request.algorithm == "TD3":
            # TD3: Twin Delayed DDPG - for continuous action spaces
            from stable_baselines3.common.noise import NormalActionNoise

            # Use hyperparameters from study or trading-optimized defaults
            actual_learning_rate = hyperparams.get('learning_rate', 0.001)

            # Action noise for exploration
            action_noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=0.2 * np.ones(action_dim)
            )

            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=actual_learning_rate,
                buffer_size=hyperparams.get('buffer_size', 100000),
                learning_starts=hyperparams.get('learning_starts', 1000),
                batch_size=hyperparams.get('batch_size', 256),
                tau=hyperparams.get('tau', 0.005),
                gamma=hyperparams.get('gamma', 0.99),
                train_freq=hyperparams.get('train_freq', 1),
                gradient_steps=hyperparams.get('gradient_steps', 1),
                action_noise=action_noise,
                policy_delay=hyperparams.get('policy_delay', 2),
                target_policy_noise=hyperparams.get('target_policy_noise', 0.3),
                target_noise_clip=hyperparams.get('target_noise_clip', 0.5),
                tensorboard_log=ws.logs_dir(),
                verbose=1
            )
        elif request.algorithm == "SAC":
            # SAC: Soft Actor-Critic - State of the art for continuous control
            # Use hyperparameters from study or trading-optimized defaults
            actual_learning_rate = hyperparams.get('learning_rate', 0.0003)

            # Handle ent_coef: can be 'auto', 'auto_X', or float
            ent_coef = hyperparams.get('ent_coef', 'auto_0.5')

            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=actual_learning_rate,
                buffer_size=hyperparams.get('buffer_size', 100000),
                learning_starts=hyperparams.get('learning_starts', 1000),
                batch_size=hyperparams.get('batch_size', 256),
                tau=hyperparams.get('tau', 0.005),
                gamma=hyperparams.get('gamma', 0.99),
                train_freq=hyperparams.get('train_freq', 1),
                gradient_steps=hyperparams.get('gradient_steps', 1),
                ent_coef=ent_coef,
                target_update_interval=hyperparams.get('target_update_interval', 1),
                use_sde=hyperparams.get('use_sde', False),
                tensorboard_log=ws.logs_dir(),
                verbose=1
            )
        else:
            # Fallback to default settings with high exploration
            actual_learning_rate = request.learning_rate
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=actual_learning_rate,
                tensorboard_log=ws.logs_dir(),
                verbose=1
            )

        # Train model with logging
        callback = TrainingStateCallback()
        learn_start_time = time.time()
        training_state["learn_start_ts"] = learn_start_time
        training_state["phase_name"] = "training"
        model.learn(
            total_timesteps=request.total_timesteps,
            callback=callback,
            tb_log_name=model_name
        )
        learn_end_time = time.time()
        training_state["learn_end_ts"] = learn_end_time
        training_state["phase_name"] = "evaluating"

        # Log training environment metrics (for debugging)
        from env.trading_env import TradingEnv
        train_env = env.envs[0]
        if isinstance(train_env, TradingEnv):
            train_metrics = train_env.get_metrics()
            logger.info(f"Training metrics: total_trades={train_metrics['total_trades']}, "
                       f"final_value=${train_metrics['final_portfolio_value']:,.0f}")

        # Save model
        models_root = ws.models_dir()
        model_path = os.path.join(models_root, model_name)
        model.save(model_path)

        # Modelin evrenini modelin yanina yaz. Model ADI evreni ANLATMIYOR:
        # yukarida gorulecegi gibi egitim `get_symbols(phase)` listesini yalnizca
        # veri cekerken kullanir, egitimi panelin tamamiyla yapar — "phase1" adli
        # bir model 30 sembolle egitilmis olabilir. Gunluk karar ucu evreni
        # adindan tahmin edince durum vektoru yanlis uzunlukta cikiyor ve SB3
        # "Unexpected observation shape" ile patliyordu. Sira onemli: durum
        # vektoru sembolleri env'deki sirayla diziyor.
        from app.services.daily_trading import write_model_meta
        write_model_meta(model_path, {
            "model_name": model_name,
            "algorithm": request.algorithm,
            "phase": request.phase,
            "symbols": list(temp_env.symbols),
            "n_symbols": int(temp_env.n_stocks),
            "obs_dim": int(temp_env.observation_space.shape[0]),
            "action_dim": int(action_dim),
            "total_timesteps": request.total_timesteps,
            "initial_balance": request.initial_balance,
            "max_shares_per_trade": request.max_shares_per_trade,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })

        # Evaluate on test set
        def make_test_env():
            return make_env(
                df=test_df,
                initial_balance=request.initial_balance,
                commission_rate=request.commission_rate,
                max_shares_per_trade=request.max_shares_per_trade,
                phase=request.phase,
                reward_type=reward_type,
                fundamental_df=fundamental_df,
                macro_df=macro_df
            )

        # Evaluate directly with raw TradingEnv — no DummyVecEnv workaround (#25)
        from env.trading_env import TradingEnv
        actual_env: TradingEnv = make_test_env()

        obs, _ = actual_env.reset()
        done = False
        test_step = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)  # (#24) reproducible eval
            obs, reward, done, truncated, _ = actual_env.step(action)
            test_step += 1
            if done or truncated:
                break

        logger.info(f"Test evaluation: {test_step} steps completed")

        metrics = actual_env.get_metrics()

        # Convert numpy types to Python types and handle NaN
        metrics = {
            k: (None if (isinstance(v, (np.floating, float)) and np.isnan(v))
                else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
            for k, v in metrics.items()
        }

        # Save trades history for visualization
        trades_data = []
        for idx, trade in enumerate(actual_env.trades_history):
            # Calculate total cost/revenue
            if trade['action'] == 'BUY':
                total_cost = trade.get('cost', trade['shares'] * trade['price'])
            else:  # SELL
                total_cost = trade.get('revenue', trade['shares'] * trade['price'])

            trades_data.append({
                "index": idx,
                "date": str(trade['date']),
                "symbol": trade['symbol'],
                "action": trade['action'],
                "shares": int(trade['shares']),
                "price": float(trade['price']),
                "total_cost": float(total_cost),
                "commission": float(trade.get('commission', 0))
            })

        # Save portfolio values history
        portfolio_history = [float(v) for v in actual_env.portfolio_values]

        # Calculate total training time
        total_training_time = time.time() - training_start_time

        # Save metrics
        metrics_file = os.path.join(ws.results_dir(), f'{model_name}_metrics.json')

        metrics_with_config = {
            **metrics,
            "algorithm": request.algorithm,
            "phase": request.phase,
            "total_timesteps": request.total_timesteps,
            "learning_rate": actual_learning_rate,  # Actual LR used
            "requested_learning_rate": request.learning_rate,  # Original request for reference
            "hyperparameter_study": request.hyperparameter_study,  # Study file used (if any)
            "hyperparameters_used": hyperparams if hyperparams else {},  # All hyperparams used
            "trained_at": datetime.now().isoformat(),
            "training_time_seconds": total_training_time,
            "training_time_minutes": total_training_time / 60,
            "training_time_hours": total_training_time / 3600,
            "trades": trades_data,
            "portfolio_history": portfolio_history
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_with_config, f, indent=2)

        logger.info(f"✅ Training completed in {total_training_time/60:.2f} minutes ({total_training_time:.1f}s)")

        # Update training state
        training_state["is_training"] = False
        training_state["state"] = "completed"
        training_state["metrics"] = metrics_with_config
        training_state["current_step"] = request.total_timesteps
        training_state["phase_name"] = "completed"
        training_state["finished_ts"] = time.time()

        # Bu kosumu ETA gecmisine yaz — bir sonraki tahminin dayanagi olur.
        # Kayit basarisiz olursa egitim yine basarili sayilir.
        if learn_start_time is not None:
            try:
                learn_seconds = (learn_end_time or time.time()) - learn_start_time
                # Sembol sayisi EGITILEN panelden alinir, get_symbols(phase)'ten
                # degil: rota o listeyi yalnizca veri cekerken kullanir, egitim
                # yuklenen CSV'nin tamamiyla yapilir.
                trained_symbols = train_df.index.get_level_values('symbol').nunique()
                training_eta.record_run(
                    algorithm=request.algorithm,
                    phase=request.phase,
                    n_symbols=int(trained_symbols),
                    total_timesteps=request.total_timesteps,
                    setup_seconds=learn_start_time - training_start_time,
                    learn_seconds=learn_seconds,
                    eval_seconds=max(total_training_time - (learn_seconds + (learn_start_time - training_start_time)), 0.0),
                    total_seconds=total_training_time,
                    device=str(getattr(getattr(model, "device", None), "type", "cpu")),
                )
            except Exception as exc:
                logger.warning(f"ETA gecmisine yazilamadi ({exc}) — egitim etkilenmedi")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        training_state["is_training"] = False
        training_state["state"] = "error"
        training_state["error"] = str(e)
        training_state["phase_name"] = "error"
        training_state["finished_ts"] = time.time()
        raise


# ==================== DAILY TRADING ENDPOINTS ====================

# SB3 model cache — .load() deserializes the whole policy net from disk
# (torch import + weight loading), so a cold call costs real time. Cache
# per model file, keyed on the .zip's mtime so a re-trained model (same
# filename, new weights) is picked up without a server restart.
_sb3_model_cache: Dict[str, Any] = {}  # model_path -> (model, mtime)


def _load_sb3_model(model_path: str, model_name_lower: str):
    """Load (or reuse a cached) SB3 model, matching algorithm by filename convention."""
    from stable_baselines3 import PPO, A2C, TD3, SAC

    mtime = os.path.getmtime(model_path + ".zip")
    cached = _sb3_model_cache.get(model_path)
    if cached is not None and cached[1] == mtime:
        return cached[0]

    if "ppo" in model_name_lower:
        model = PPO.load(model_path)
        logger.info("Loaded PPO model")
    elif "a2c" in model_name_lower:
        model = A2C.load(model_path)
        logger.info("Loaded A2C model")
    elif "td3" in model_name_lower:
        model = TD3.load(model_path)
        logger.info("Loaded TD3 model")
    elif "sac" in model_name_lower:
        model = SAC.load(model_path)
        logger.info("Loaded SAC model")
    else:
        return None

    _sb3_model_cache[model_path] = (model, mtime)
    return model


@router.post("/daily-decision", response_model=DailyDecisionResponse)
async def get_daily_decision(request: DailyDecisionRequest, user: RequireWriter):
    """
    Get daily trading decision from trained model

    This endpoint:
    1. Loads the specified trained model
    2. Fetches latest market data from yfinance
    3. Builds state vector from current portfolio + market data
    4. Runs model inference to get trading signals
    5. Interprets signals with risk filtering
    6. Returns trade recommendations with before/after portfolio

    Args:
        request: Daily decision request with model, portfolio, and risk params

    Returns:
        DailyDecisionResponse with trade decisions and portfolio snapshots
    """
    from app.services.daily_trading import (
        get_risk_parameters,
        resolve_trade_universe,
        fetch_latest_market_data,
        build_live_state,
        get_current_prices,
        interpret_actions_with_risk,
        calculate_portfolio_value,
        simulate_portfolio_after_trades,
        save_daily_decision
    )
    try:
        logger.info(f"Daily decision request for model: {request.model_name}")

        # 1. Load model
        safe_model_name = sanitize_model_name(request.model_name)
        # Once kullanicinin kendi alani, sonra ortak (eski) model dizini.
        model_zip = ws.find_file("models", f"{safe_model_name}.zip")
        if not model_zip:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model_name}"
            )
        model_path = model_zip[:-4]  # SB3 .zip uzantisini kendisi ekler

        # Determine model type from name (cached — see _load_sb3_model)
        model_name_lower = request.model_name.lower()
        model = _load_sb3_model(model_path, model_name_lower)
        if model is None:
            raise HTTPException(
                status_code=400,
                detail="Unknown model type. Model name must contain: ppo, a2c, td3, or sac"
            )

        # 2. Get risk parameters
        risk_params = get_risk_parameters(request.risk_mode)
        logger.info(f"Risk mode: {request.risk_mode} - {risk_params['description']}")

        # 3. Fetch latest market data
        target_date = request.date or datetime.now().strftime("%Y-%m-%d")

        # The trade universe MUST match what the model saw during training,
        # otherwise the state vector size mismatches the obs space. The model
        # NAME does not carry it: training uses `get_symbols(phase)` only to
        # fetch data and then trains on whatever the loaded panel contains, so
        # a "phase1" model can well have 30 symbols. Ask the model itself
        # (sidecar meta → training panel → name constants, each checked against
        # the action space). The user-supplied `shares` is treated as a sparse
        # declaration of current holdings — missing entries are filled with 0.
        try:
            trade_universe, universe_source = resolve_trade_universe(
                model, model_path, request.model_name
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        logger.info(f"Trade universe source: {universe_source}")

        # Normalize whatever shares the client sent to BIST `.IS` form.
        user_shares = {
            (k if "." in k else f"{k}.IS"): int(v)
            for k, v in (request.shares or {}).items()
        }
        # Align portfolio to the trade universe (fill missing with 0).
        normalized_shares = {sym: user_shares.get(sym, 0) for sym in trade_universe}
        symbols = list(normalized_shares.keys())
        logger.info(f"Trade universe ({len(symbols)} symbols): {symbols}")

        market_data = await fetch_latest_market_data(
            symbols=symbols,
            target_date=target_date,
            lookback_days=30
        )

        # 4. Build state
        state = build_live_state(
            balance=request.balance,
            shares_owned=normalized_shares,
            market_data=market_data,
            target_date=target_date,
            max_shares_per_trade=request.max_shares_per_trade
        )

        # 5. Model inference
        # Son kontrol: evren dogru sayida olsa da durum vektoru hala yanlis
        # uzunlukta cikabilir (ornegin faz 2 modeli — hisse basina 17 ozellik +
        # 6 makro bekler, `build_live_state` 10 ozellik uretir). SB3'un ham
        # "Unexpected observation shape" hatasi kullaniciya hicbir sey
        # anlatmiyordu; nedeni burada soyluyoruz.
        expected_obs = getattr(model.observation_space, "shape", None)
        if expected_obs is not None and state.shape != tuple(expected_obs):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Durum vektoru {state.shape[0]} uzunlugunda uretildi ama model "
                    f"{expected_obs[0]} bekliyor ({len(trade_universe)} sembol, evren "
                    f"kaynagi: {universe_source}). Model muhtemelen bu ucun "
                    f"uretemedigi ek ozelliklerle egitildi (faz 2: fundamental + makro). "
                    f"Faz 1/3 modeliyle deneyin."
                ),
            )
        logger.info("Running model inference...")
        action, _states = model.predict(state, deterministic=True)
        logger.info(f"Model output (raw action): {action}")

        # 6. Get current prices
        current_prices = get_current_prices(market_data, target_date)
        logger.info(f"Current prices: {current_prices}")

        # 7. Interpret actions with risk filtering
        decisions = interpret_actions_with_risk(
            action=action,
            symbols=symbols,
            current_prices=current_prices,
            balance=request.balance,
            shares_owned=normalized_shares,
            risk_params=risk_params,
            max_shares_per_trade=request.max_shares_per_trade
        )

        # 8. Calculate portfolio before/after
        portfolio_before = calculate_portfolio_value(
            balance=request.balance,
            shares=normalized_shares,
            prices=current_prices
        )

        portfolio_after = simulate_portfolio_after_trades(
            balance=request.balance,
            shares=normalized_shares,
            decisions=decisions
        )

        # 9. Create summary
        total_trades = len([d for d in decisions if d["executed"]])
        total_commission = sum(d.get("commission", 0) for d in decisions)

        daily_return_pct = 0.0
        if portfolio_before["portfolio_value"] > 0:
            daily_return_pct = (
                (portfolio_after["portfolio_value"] - portfolio_before["portfolio_value"])
                / portfolio_before["portfolio_value"] * 100
            )

        # Get actual date used (may differ from requested date)
        actual_date = market_data.attrs.get('actual_date', target_date)

        summary = {
            "total_trades": total_trades,
            "total_commission": round(total_commission, 2),
            "daily_return_pct": round(daily_return_pct, 2),
            "risk_mode": request.risk_mode,
            "max_shares_per_trade": request.max_shares_per_trade,
            "actual_date": actual_date,
            "requested_date": target_date
        }

        # 10. Save decision
        save_daily_decision(
            date=target_date,
            decisions=decisions,
            portfolio_before=portfolio_before,
            portfolio_after=portfolio_after,
            risk_mode=request.risk_mode,
            max_shares_per_trade=request.max_shares_per_trade
        )

        logger.info(f"Daily decision generated successfully: {total_trades} trades, {daily_return_pct:.2f}% return")

        # 11. Return response
        return DailyDecisionResponse(
            date=target_date,
            decisions=[TradeDecision(**d) for d in decisions],
            portfolio_before=PortfolioSnapshot(**portfolio_before),
            portfolio_after=PortfolioSnapshot(**portfolio_after),
            summary=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Daily decision failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-decision")
async def apply_decision(date: str, user: RequireWriter):
    """
    Apply a saved decision and update portfolio history

    Args:
        date: Date of the decision to apply (YYYY-MM-DD)

    Returns:
        Success message and updated portfolio
    """
    from app.services.daily_trading import append_to_portfolio_history

    try:
        logger.info(f"Applying decision for date: {date}")

        # Load decision from file
        decision_file = os.path.join(ws.live_trading_dir(), 'trade_decisions.json')

        if not os.path.exists(decision_file):
            raise HTTPException(
                status_code=404,
                detail="No decisions found"
            )

        with open(decision_file, 'r') as f:
            all_decisions = json.load(f)

        if date not in all_decisions:
            raise HTTPException(
                status_code=404,
                detail=f"No decision found for date: {date}"
            )

        decision_data = all_decisions[date]

        # Append to portfolio history
        append_to_portfolio_history(
            date=date,
            portfolio_after=decision_data["portfolio_after"],
            daily_return_pct=decision_data["summary"]["daily_return_pct"]
        )

        logger.info(f"Decision applied successfully for {date}")

        return {
            "message": f"Decision for {date} applied successfully",
            "portfolio": decision_data["portfolio_after"],
            "summary": decision_data["summary"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apply decision failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decisions-history")
async def get_decisions_history():
    """
    Return all saved daily decisions (trade_decisions.json) sorted by date desc.
    Used by the dashboard to browse past decisions, not just the most recent one.
    """
    decision_file = os.path.join(ws.live_trading_dir(), 'trade_decisions.json')
    if not os.path.exists(decision_file):
        return {"dates": [], "decisions": {}}

    def _sanitize(obj):
        # Older entries may contain NaN floats that the std json encoder
        # accepts on read but FastAPI's response serializer rejects. Coerce
        # them to None recursively before returning.
        if isinstance(obj, float):
            if obj != obj or obj in (float('inf'), float('-inf')):  # NaN/Inf
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    try:
        with open(decision_file, 'r') as f:
            all_decisions = json.load(f)
        all_decisions = _sanitize(all_decisions)
        # Newest first so the UI dropdown shows recent dates at the top.
        dates = sorted(all_decisions.keys(), reverse=True)
        return {"dates": dates, "decisions": all_decisions}
    except Exception as exc:
        logger.error(f"Failed to load decisions history: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/portfolio-history", response_model=PortfolioHistoryResponse)
async def get_portfolio_history(days: int = 30):
    """
    Get portfolio history for the last N days

    Args:
        days: Number of days to retrieve (default: 30)

    Returns:
        Portfolio history with dates, values, returns, and balances
    """
    from app.services.daily_trading import load_portfolio_history

    try:
        logger.info(f"Loading portfolio history for last {days} days")

        history = load_portfolio_history(days=days)

        return PortfolioHistoryResponse(
            dates=history["dates"],
            portfolio_values=history["portfolio_values"],
            daily_returns=history["daily_returns"],
            balances=history["balances"]
        )

    except Exception as e:
        logger.error(f"Failed to load portfolio history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest-portfolio")
async def get_latest_portfolio():
    """
    Get the latest portfolio state from history

    Returns:
        Latest portfolio snapshot or default initial state
    """
    try:
        history_file = os.path.join(ws.live_trading_dir(), 'portfolio_history.csv')

        if not os.path.exists(history_file):
            # Return default initial state
            return {
                "balance": 1000000,
                "shares": {
                    "ASELS.IS": 0,
                    "THYAO.IS": 0,
                    "EREGL.IS": 0,
                    "KCHOL.IS": 0,
                    "SAHOL.IS": 0
                },
                "portfolio_value": 1000000,
                "date": None
            }

        df = pd.read_csv(history_file)
        if df.empty:
            return {
                "balance": 1000000,
                "shares": {
                    "ASELS.IS": 0,
                    "THYAO.IS": 0,
                    "EREGL.IS": 0,
                    "KCHOL.IS": 0,
                    "SAHOL.IS": 0
                },
                "portfolio_value": 1000000,
                "date": None
            }

        latest = df.iloc[-1]

        # Extract shares
        shares = {}
        for col in df.columns:
            if col.endswith('_shares'):
                symbol = col.replace('_shares', '.IS')
                shares[symbol] = int(latest[col]) if not pd.isna(latest[col]) else 0

        return {
            "balance": float(latest['balance']),
            "shares": shares,
            "portfolio_value": float(latest['portfolio_value']),
            "date": latest['date']
        }

    except Exception as e:
        logger.error(f"Failed to load latest portfolio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ACADEMIC ANALYSIS ENDPOINTS ====================

@router.get("/analysis/model-comparison", response_model=ModelComparisonResponse)
async def get_model_comparison():
    """
    Get comprehensive model comparison for academic analysis

    Returns:
        Comparison table with all performance metrics
    """
    try:
        # Akademik rapor betigi (scripts/generate_academic_report.py) ortak
        # results/ dizinine yazar; once kullanici alanina bakilir.
        results_file = ws.find_file('results', os.path.join('data', 'detailed_results.json'))

        if not results_file:
            raise HTTPException(
                status_code=404,
                detail="No analysis results found. Please run scripts/generate_academic_report.py first"
            )

        with open(results_file, 'r') as f:
            results = json.load(f)

        return {
            "models": results,
            "count": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model comparison: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/best-models")
async def get_best_models():
    """
    Get best performing models by different metrics

    Returns:
        Dictionary with best models for each metric
    """
    try:
        # Akademik rapor betigi (scripts/generate_academic_report.py) ortak
        # results/ dizinine yazar; once kullanici alanina bakilir.
        results_file = ws.find_file('results', os.path.join('data', 'detailed_results.json'))

        if not results_file:
            raise HTTPException(
                status_code=404,
                detail="No analysis results found. Please run scripts/generate_academic_report.py first"
            )

        with open(results_file, 'r') as f:
            results = json.load(f)

        # Find best models
        best_models = {
            "best_sharpe": max(results.items(), key=lambda x: x[1]['sharpe_ratio']),
            "best_return": max(results.items(), key=lambda x: x[1]['total_return']),
            "lowest_drawdown": max(results.items(), key=lambda x: x[1]['max_drawdown']),
            "best_win_rate": max(results.items(), key=lambda x: x[1]['win_rate']),
            "best_sortino": max(results.items(), key=lambda x: x[1]['sortino_ratio']),
            "best_calmar": max(results.items(), key=lambda x: x[1]['calmar_ratio'])
        }

        return {
            metric: {
                "model": model_name,
                "value": model_data[metric.replace("best_", "").replace("lowest_", "")]
            }
            for metric, (model_name, model_data) in best_models.items()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get best models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/generate-report")
async def generate_analysis_report(background_tasks: BackgroundTasks):
    """
    Generate comprehensive academic analysis report

    This runs the analysis in the background and generates:
    - Model comparison tables
    - Performance visualizations
    - LaTeX tables for publications
    - Statistical significance tests

    Returns:
        Status message
    """
    import subprocess

    try:
        # Run the analysis script in background
        def run_analysis():
            subprocess.run(['python', 'scripts/generate_academic_report.py'], check=True)

        background_tasks.add_task(run_analysis)

        return {
            "message": "Analysis report generation started",
            "status": "processing",
            "note": "Check results/ directory for outputs when complete"
        }

    except Exception as e:
        logger.error(f"Failed to start analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
