"""
Trading API Routes
FastAPI endpoints for RL trading system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from typing import Dict, List, Optional
from pydantic import BaseModel
import asyncio
import glob
import os
import re
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

from app.schemas.trading import (
    TrainingRequest, TrainingStatus, ModelMetrics,
    ModelInfo, TrainingResponse,
    DailyDecisionRequest, DailyDecisionResponse, TradeDecision,
    PortfolioSnapshot, PortfolioHistoryResponse
)
from app.core.config import get_settings

_settings = get_settings()

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading", tags=["Trading"])

# Global training state — guarded by _training_lock (#23)
_training_lock = asyncio.Lock()
training_state = {
    "is_training": False,
    "current_step": 0,
    "total_steps": 0,
    "start_time": None,
    "metrics": {},
    "error": None
}

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
        studies_dir = f"{_settings.RESULTS_DIR}/hyperparameter_studies"

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


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start model training in background

    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks
    """
    global training_state

    async with _training_lock:  # (#23) prevent race between concurrent requests
        if training_state["is_training"]:
            raise HTTPException(
                status_code=400,
                detail="Training already in progress"
            )

        # Reset training state inside lock
        training_state = {
            "is_training": True,
            "current_step": 0,
            "total_steps": request.total_timesteps,
            "start_time": datetime.now().isoformat(),
            "metrics": {},
            "error": None,
            "config": request.dict()
        }

    # Start training in background (outside lock — long-running)
    background_tasks.add_task(run_training, request)

    return TrainingResponse(
        message="Training started successfully",
        training_id=training_state["start_time"],
        status="started"
    )


@router.get("/train/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    global training_state

    progress = 0.0
    if training_state["total_steps"] > 0:
        progress = training_state["current_step"] / training_state["total_steps"]

    return TrainingStatus(
        is_training=training_state["is_training"],
        current_step=training_state["current_step"],
        total_steps=training_state["total_steps"],
        progress=progress,
        start_time=training_state.get("start_time"),
        metrics=training_state.get("metrics", {}),
        error=training_state.get("error")
    )


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all trained models"""
    models_dir = _settings.MODELS_DIR

    if not os.path.exists(models_dir):
        return []

    models = []

    for filename in os.listdir(models_dir):
        if filename.endswith(".zip"):
            model_path = os.path.join(models_dir, filename)

            # Try to load metrics from results
            model_name = filename.replace(".zip", "")
            metrics_file = f"{_settings.RESULTS_DIR}/{model_name}_metrics.json"

            metrics = {}
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

            models.append(ModelInfo(
                name=model_name,
                path=model_path,
                created_at=datetime.fromtimestamp(
                    os.path.getctime(model_path)
                ).isoformat(),
                metrics=metrics
            ))

    return models


@router.get("/models/{model_name}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    model_name = sanitize_model_name(model_name)
    metrics_file = f"{_settings.RESULTS_DIR}/{model_name}_metrics.json"

    if not os.path.exists(metrics_file):
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for model: {model_name}"
        )

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return ModelMetrics(**metrics)


@router.post("/data/generate")
async def generate_data(
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
async def update_data(request: DataUpdateRequest):
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

        return {
            "source":     source,
            "earliest":   earliest,
            "per_metric": per_metric,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a trained model"""
    model_name = sanitize_model_name(model_name)
    model_path = f"{_settings.MODELS_DIR}/{model_name}.zip"

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_name}"
        )

    # Delete model file
    os.remove(model_path)

    # Delete metrics if exists
    metrics_file = f"{_settings.RESULTS_DIR}/{model_name}_metrics.json"
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    return {"message": f"Model {model_name} deleted successfully"}


async def run_training(request: TrainingRequest):
    """
    Background task for model training

    Args:
        request: Training configuration
    """
    global training_state
    import time

    # Track total training time
    training_start_time = time.time()

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
                global training_state
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
        os.makedirs(_settings.LOGS_DIR, exist_ok=True)
        model_name = f"{request.algorithm.lower()}_phase{request.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load hyperparameters if study is specified
        hyperparams = {}
        if request.hyperparameter_study:
            study_path = os.path.join(f"{_settings.RESULTS_DIR}/hyperparameter_studies", request.hyperparameter_study)
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
                tensorboard_log=_settings.LOGS_DIR,
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
                tensorboard_log=_settings.LOGS_DIR,
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
                tensorboard_log=_settings.LOGS_DIR,
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
                tensorboard_log=_settings.LOGS_DIR,
                verbose=1
            )
        else:
            # Fallback to default settings with high exploration
            actual_learning_rate = request.learning_rate
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=actual_learning_rate,
                tensorboard_log=_settings.LOGS_DIR,
                verbose=1
            )

        # Train model with logging
        callback = TrainingStateCallback()
        model.learn(
            total_timesteps=request.total_timesteps,
            callback=callback,
            tb_log_name=model_name
        )

        # Log training environment metrics (for debugging)
        from env.trading_env import TradingEnv
        train_env = env.envs[0]
        if isinstance(train_env, TradingEnv):
            train_metrics = train_env.get_metrics()
            logger.info(f"Training metrics: total_trades={train_metrics['total_trades']}, "
                       f"final_value=${train_metrics['final_portfolio_value']:,.0f}")

        # Save model
        os.makedirs(_settings.MODELS_DIR, exist_ok=True)
        model_path = f'{_settings.MODELS_DIR}/{model_name}'
        model.save(model_path)

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
        os.makedirs(_settings.RESULTS_DIR, exist_ok=True)
        metrics_file = f'{_settings.RESULTS_DIR}/{model_name}_metrics.json'

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
        training_state["metrics"] = metrics_with_config
        training_state["current_step"] = request.total_timesteps

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        training_state["is_training"] = False
        training_state["error"] = str(e)
        raise


# ==================== DAILY TRADING ENDPOINTS ====================

@router.post("/daily-decision", response_model=DailyDecisionResponse)
async def get_daily_decision(request: DailyDecisionRequest):
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
        fetch_latest_market_data,
        build_live_state,
        get_current_prices,
        interpret_actions_with_risk,
        calculate_portfolio_value,
        simulate_portfolio_after_trades,
        save_daily_decision
    )
    from stable_baselines3 import PPO, A2C, TD3, SAC

    try:
        logger.info(f"Daily decision request for model: {request.model_name}")

        # 1. Load model
        safe_model_name = sanitize_model_name(request.model_name)
        model_path = f"{_settings.MODELS_DIR}/{safe_model_name}"
        if not os.path.exists(model_path + ".zip"):
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model_name}"
            )

        # Determine model type from name
        model_name_lower = request.model_name.lower()
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
            raise HTTPException(
                status_code=400,
                detail="Unknown model type. Model name must contain: ppo, a2c, td3, or sac"
            )

        # 2. Get risk parameters
        risk_params = get_risk_parameters(request.risk_mode)
        logger.info(f"Risk mode: {request.risk_mode} - {risk_params['description']}")

        # 3. Fetch latest market data
        target_date = request.date or datetime.now().strftime("%Y-%m-%d")
        symbols = list(request.shares.keys())

        market_data = await fetch_latest_market_data(
            symbols=symbols,
            target_date=target_date,
            lookback_days=30
        )

        # 4. Build state
        state = build_live_state(
            balance=request.balance,
            shares_owned=request.shares,
            market_data=market_data,
            target_date=target_date,
            max_shares_per_trade=request.max_shares_per_trade
        )

        # 5. Model inference
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
            shares_owned=request.shares,
            risk_params=risk_params,
            max_shares_per_trade=request.max_shares_per_trade
        )

        # 8. Calculate portfolio before/after
        portfolio_before = calculate_portfolio_value(
            balance=request.balance,
            shares=request.shares,
            prices=current_prices
        )

        portfolio_after = simulate_portfolio_after_trades(
            balance=request.balance,
            shares=request.shares,
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
async def apply_decision(date: str):
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
        decision_file = f'{_settings.DATA_DIR}/live_trading/trade_decisions.json'

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
        history_file = f'{_settings.DATA_DIR}/live_trading/portfolio_history.csv'

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

@router.get("/analysis/model-comparison")
async def get_model_comparison():
    """
    Get comprehensive model comparison for academic analysis

    Returns:
        Comparison table with all performance metrics
    """
    try:
        results_file = f'{_settings.RESULTS_DIR}/data/detailed_results.json'

        if not os.path.exists(results_file):
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
        results_file = f'{_settings.RESULTS_DIR}/data/detailed_results.json'

        if not os.path.exists(results_file):
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
