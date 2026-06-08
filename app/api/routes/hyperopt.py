"""
Hyperparameter Optimization API Routes
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import optuna

from app.schemas.hyperopt import (
    OptimizationRequest,
    OptimizationStartResponse,
    OptimizationProgressResponse,
    OptimizationResultsResponse,
    StudyListResponse,
    StudyDetailResponse,
    StudyInfo,
    TrialInfo,
    StudyStatus,
    TrialState,
    AlgorithmSearchSpaceResponse,
    SearchSpaceInfo,
    SearchSpaceUpdateRequest,
    CancelOptimizationResponse,
    WSProgressUpdate,
    WSTrialComplete,
    WSOptimizationComplete,
    AlgorithmType,
)
from data.bist30_symbols import PHASE1_SYMBOLS
from hyperparameter_optimization.optimizers import (
    PPOOptimizer,
    A2COptimizer,
    TD3Optimizer,
    SACOptimizer,
)
from hyperparameter_optimization.search_spaces import (
    get_search_space,
    get_default_hyperparameters,
)

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hyperopt", tags=["Hyperparameter Optimization"])

# Global state management
active_studies: Dict[str, Dict] = {}  # study_id -> study info
websocket_connections: Dict[str, List[WebSocket]] = {}  # study_id -> websocket list

# Optuna storage — anchor to repo root (CWD-independent) and use forward
# slashes so the SQLAlchemy URI parses correctly on Windows.
_HYPEROPT_DB = (
    Path(__file__).resolve().parents[3]
    / "results" / "hyperparameter_studies" / "optuna_studies.db"
)
_HYPEROPT_DB.parent.mkdir(parents=True, exist_ok=True)
OPTUNA_STORAGE = f"sqlite:///{_HYPEROPT_DB.as_posix()}"


def get_optimizer_class(algorithm: AlgorithmType):
    """Algoritma için optimizer sınıfını döndürür"""
    optimizer_map = {
        AlgorithmType.PPO: PPOOptimizer,
        AlgorithmType.A2C: A2COptimizer,
        AlgorithmType.TD3: TD3Optimizer,
        AlgorithmType.SAC: SACOptimizer,
    }
    return optimizer_map[algorithm]


def create_study_info_from_request(
    study_id: str,
    request: OptimizationRequest
) -> StudyInfo:
    """Request'ten StudyInfo oluşturur"""
    stock_symbols = request.stock_symbols if request.stock_symbols else PHASE1_SYMBOLS

    return StudyInfo(
        study_id=study_id,
        study_name=request.study_name or f"{request.algorithm.value}_optimization_{study_id[:8]}",
        algorithm=request.algorithm,
        status=StudyStatus.PENDING,
        n_trials=request.n_trials,
        total_timesteps=request.total_timesteps,
        stock_symbols=stock_symbols,
        train_start=request.train_start,
        train_end=request.train_end,
        val_start=request.val_start,
        val_end=request.val_end,
        phase=request.phase,
        reward_type=request.reward_type,
        created_at=datetime.now(),
    )


async def send_websocket_message(study_id: str, message: dict):
    """WebSocket mesajı gönderir"""
    if study_id in websocket_connections:
        disconnected = []
        for ws in websocket_connections[study_id]:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                disconnected.append(ws)

        # Disconnected websockets'i temizle
        for ws in disconnected:
            websocket_connections[study_id].remove(ws)


async def run_optimization_background(
    study_id: str,
    request: OptimizationRequest,
    study_info: StudyInfo
):
    """Background'da optimization çalıştırır"""
    try:
        # Update status
        active_studies[study_id]["status"] = StudyStatus.RUNNING
        active_studies[study_id]["started_at"] = datetime.now()

        # Send WebSocket update
        await send_websocket_message(study_id, {
            "message_type": "status_update",
            "study_id": study_id,
            "status": "running",
            "message": "Optimization started"
        })

        # Get stock symbols
        stock_symbols = request.stock_symbols if request.stock_symbols else PHASE1_SYMBOLS

        # Create optimizer
        OptimizerClass = get_optimizer_class(request.algorithm)
        optimizer = OptimizerClass(
            study_name=study_info.study_name,
            storage=OPTUNA_STORAGE,
            n_trials=request.n_trials,
            n_jobs=1,  # Parallelization handled by PyTorch/TF, not Optuna
            seed=None,  # No seed for better exploration across different algorithms
            phase=request.phase,  # Phase 2 support
            reward_type=request.reward_type,  # PSR or simple reward
        )

        logger.info(f"Starting optimization for study {study_id} (Phase {request.phase}, Reward: {request.reward_type.upper()})")

        # Run optimization with callback for progress
        def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """Trial tamamlandığında çağrılır"""
            try:
                # Update progress
                completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

                active_studies[study_id].update({
                    "trials_completed": completed,
                    "trials_pruned": pruned,
                    "trials_failed": failed,
                    "progress_percentage": (completed / request.n_trials) * 100,
                    "best_value": study.best_value if completed > 0 else None,
                    "best_params": study.best_params if completed > 0 else None,
                    "best_trial": study.best_trial.number if completed > 0 else None,
                })

                # Send WebSocket progress
                asyncio.create_task(send_websocket_message(study_id, {
                    "message_type": "trial_complete",
                    "study_id": study_id,
                    "trial_number": trial.number,
                    "trial_value": trial.value if trial.value else None,
                    "trial_state": trial.state.name,
                    "is_best": trial.number == study.best_trial.number if completed > 0 else False,
                    "progress_percentage": (completed / request.n_trials) * 100,
                    "timestamp": datetime.now().isoformat(),
                }))

            except Exception as e:
                logger.error(f"Error in trial callback: {e}")

        # Optimize (callback sistemi şu an için basit tutuyoruz)
        # TODO: Add proper callback support to optimizer.optimize()
        study = optimizer.optimize(
            stock_symbols=stock_symbols,
            train_start=request.train_start,
            train_end=request.train_end,
            val_start=request.val_start,
            val_end=request.val_end,
            total_timesteps=request.total_timesteps,
            eval_freq=request.eval_freq,
            n_eval_episodes=request.n_eval_episodes,
            use_cached_data=request.use_cached_data,
            show_progress_bar=False,  # Disable for background
        )

        # Update final progress
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        active_studies[study_id].update({
            "trials_completed": completed,
            "progress_percentage": 100.0,
        })

        # Save best params
        optimizer.save_best_params()

        # Update status
        active_studies[study_id].update({
            "status": StudyStatus.COMPLETED,
            "completed_at": datetime.now(),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
        })

        # Send completion message
        await send_websocket_message(study_id, {
            "message_type": "optimization_complete",
            "study_id": study_id,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "total_trials": len(study.trials),
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(f"Optimization completed for study {study_id}")

    except Exception as e:
        logger.error(f"Optimization failed for study {study_id}: {e}")

        # Update status
        active_studies[study_id].update({
            "status": StudyStatus.FAILED,
            "completed_at": datetime.now(),
            "error_message": str(e),
        })

        # Send error message
        await send_websocket_message(study_id, {
            "message_type": "error",
            "study_id": study_id,
            "error_message": str(e),
            "timestamp": datetime.now().isoformat(),
        })


# ==================== Endpoints ====================

@router.get("/data-range")
async def get_cached_data_range():
    """
    Mevcut data/bist/raw_stock_data.csv'nin tarih aralığını ve sembol listesini
    döner. UI hyperopt sayfasındaki DatePicker'ın min/max sınırlarını ve
    sembol seçicisini buradan dolduruyor.
    """
    try:
        from data.data_fetcher import DataFetcher
        import pandas as pd

        fetcher = DataFetcher()
        df = fetcher.load_data('raw_stock_data.csv')
        dates = df.index.get_level_values('date')
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        symbols = sorted(df.index.get_level_values('symbol').unique().tolist())

        return {
            "available": True,
            "min_date": dates.min().strftime("%Y-%m-%d"),
            "max_date": dates.max().strftime("%Y-%m-%d"),
            "symbols": symbols,
            "total_rows": int(len(df)),
        }
    except FileNotFoundError:
        return {
            "available": False,
            "min_date": None,
            "max_date": None,
            "symbols": [],
            "total_rows": 0,
            "message": (
                "data/bist/raw_stock_data.csv bulunamadı. 'Veri' sayfasından "
                "veri çekin veya hyperopt için 'yfinance'tan tazele' seçeneğini "
                "işaretleyin."
            ),
        }


@router.post("/start", response_model=OptimizationStartResponse)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Hiper parametre optimizasyonu başlatır
    """
    try:
        # Create unique study ID
        study_id = str(uuid.uuid4())

        # Create study info
        study_info = create_study_info_from_request(study_id, request)

        # Store in active studies
        active_studies[study_id] = study_info.dict()

        # Add background task
        background_tasks.add_task(
            run_optimization_background,
            study_id,
            request,
            study_info
        )

        # Estimate duration (rough estimate: 2 min per trial)
        estimated_duration = (request.n_trials * 2.0)

        return OptimizationStartResponse(
            study_id=study_id,
            study_name=study_info.study_name,
            message=f"Optimization started for {request.algorithm.value}",
            estimated_duration_minutes=estimated_duration
        )

    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/studies", response_model=StudyListResponse)
async def list_studies(
    algorithm: Optional[AlgorithmType] = None,
    status: Optional[StudyStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Optimization studies'i listeler
    """
    try:
        # 1) In-memory studies (current process)
        in_mem = {sid: dict(s) for sid, s in active_studies.items()}

        # 2) Merge SQLite-persisted studies the process never started (e.g.
        #    after a server restart). Without this the grid silently goes
        #    empty even though all results are still on disk.
        known_names = {s.get("study_name") for s in in_mem.values()}
        try:
            for summary in optuna.get_all_study_summaries(storage=OPTUNA_STORAGE):
                if summary.study_name in known_names:
                    continue
                # Use study_name as the id surrogate so the detail endpoint
                # can find it (it accepts study_name fallback).
                algo_guess = summary.study_name.split("_")[0].lower()
                try:
                    algo_enum = AlgorithmType(algo_guess)
                except ValueError:
                    algo_enum = AlgorithmType.PPO
                best_value = summary.best_trial.value if summary.best_trial else None
                best_params = summary.best_trial.params if summary.best_trial else None
                best_trial_num = summary.best_trial.number if summary.best_trial else None
                in_mem[summary.study_name] = {
                    "study_id": summary.study_name,
                    "study_name": summary.study_name,
                    "algorithm": algo_enum.value,
                    "status": StudyStatus.COMPLETED.value,
                    "n_trials": summary.n_trials,
                    "total_timesteps": 0,
                    "stock_symbols": [],
                    "train_start": "—",
                    "train_end": "—",
                    "val_start": "—",
                    "val_end": "—",
                    "phase": 1,
                    "reward_type": "psr",
                    "trials_completed": summary.n_trials,
                    "trials_pruned": 0,
                    "trials_failed": 0,
                    "progress_percentage": 100.0,
                    "best_value": best_value,
                    "best_params": best_params,
                    "best_trial": best_trial_num,
                    "created_at": summary.datetime_start or datetime.now(),
                }
        except Exception as exc:
            logger.warning(f"Could not enumerate SQLite studies: {exc}")

        studies = list(in_mem.values())

        # Filter by algorithm
        if algorithm:
            studies = [s for s in studies if s["algorithm"] == algorithm]

        # Filter by status
        if status:
            studies = [s for s in studies if s["status"] == status]

        # Sort by created_at (newest first)
        studies = sorted(studies, key=lambda x: x["created_at"], reverse=True)

        # Pagination
        total = len(studies)
        studies = studies[offset:offset + limit]

        # Convert to StudyInfo
        study_infos = [StudyInfo(**s) for s in studies]

        return StudyListResponse(
            studies=study_infos,
            total=total
        )

    except Exception as e:
        logger.error(f"Failed to list studies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/studies/{study_id}", response_model=StudyDetailResponse)
async def get_study_detail(study_id: str):
    """
    Study detaylarını getirir.

    `active_studies` is process-memory only — after a server restart the
    in-memory record is gone but the Optuna SQLite DB still has all trials.
    Fall back to building a minimal StudyInfo straight from the DB so detail
    pages keep working across restarts.
    """
    try:
        # Check if study exists
        if study_id in active_studies:
            study_info = StudyInfo(**active_studies[study_id])
        else:
            # Cold-load from SQLite. `study_id` here is either the UUID or
            # the study_name; we try both since the grid uses whichever is
            # present.
            summaries = optuna.get_all_study_summaries(storage=OPTUNA_STORAGE)
            match = None
            for s in summaries:
                if s.study_name == study_id or s.study_name.endswith(study_id[:8]):
                    match = s
                    break
            if match is None:
                raise HTTPException(status_code=404, detail="Study not found")

            best_trial = None
            best_value = None
            best_params = None
            if match.best_trial is not None:
                best_trial = match.best_trial.number
                best_value = match.best_trial.value
                best_params = match.best_trial.params

            algo_guess = match.study_name.split("_")[0].lower()
            try:
                algo_enum = AlgorithmType(algo_guess)
            except ValueError:
                algo_enum = AlgorithmType.PPO

            n_complete = match.n_trials  # total trials in DB
            study_info = StudyInfo(
                study_id=study_id,
                study_name=match.study_name,
                algorithm=algo_enum,
                status=StudyStatus.COMPLETED,
                n_trials=n_complete,
                total_timesteps=0,
                stock_symbols=[],
                train_start="—",
                train_end="—",
                val_start="—",
                val_end="—",
                phase=1,
                reward_type="psr",
                trials_completed=n_complete,
                best_value=best_value,
                best_params=best_params,
                best_trial=best_trial,
                created_at=match.datetime_start or datetime.now(),
            )

        # Load study from Optuna
        try:
            study = optuna.load_study(
                study_name=study_info.study_name,
                storage=OPTUNA_STORAGE
            )

            # Get trials
            trials = []
            for trial in study.trials:
                trials.append(TrialInfo(
                    trial_number=trial.number,
                    state=TrialState(trial.state.name.lower()),
                    value=trial.value,
                    params=trial.params,
                    duration_seconds=trial.duration.total_seconds() if trial.duration else None,
                    user_attrs=trial.user_attrs,
                    datetime_start=trial.datetime_start,
                    datetime_complete=trial.datetime_complete,
                ))

            # Calculate statistics
            completed_values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            if completed_values:
                import numpy as np
                values_array = np.array(completed_values)
                mean_value = float(np.mean(values_array))
                median_value = float(np.median(values_array))
                std_value = float(np.std(values_array))
                min_value = float(np.min(values_array))
                max_value = float(np.max(values_array))
            else:
                mean_value = median_value = std_value = min_value = max_value = None

        except Exception:
            trials = []
            mean_value = median_value = std_value = min_value = max_value = None

        return StudyDetailResponse(
            study=study_info,
            trials=trials,
            mean_value=mean_value,
            median_value=median_value,
            std_value=std_value,
            min_value=min_value,
            max_value=max_value,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get study detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/studies/{study_id}/progress", response_model=OptimizationProgressResponse)
async def get_optimization_progress(study_id: str):
    """
    Optimization progress'i getirir
    """
    try:
        if study_id not in active_studies:
            raise HTTPException(status_code=404, detail="Study not found")

        study = active_studies[study_id]

        # Get last trial info
        last_trial = None
        if study["trials_completed"] > 0:
            try:
                optuna_study = optuna.load_study(
                    study_name=study["study_name"],
                    storage=OPTUNA_STORAGE
                )
                if optuna_study.trials:
                    trial = optuna_study.trials[-1]
                    last_trial = TrialInfo(
                        trial_number=trial.number,
                        state=TrialState(trial.state.name.lower()),
                        value=trial.value,
                        params=trial.params,
                        duration_seconds=trial.duration.total_seconds() if trial.duration else None,
                    )
            except Exception:
                pass

        # Estimate time remaining
        estimated_time_remaining = None
        if study["status"] == StudyStatus.RUNNING and study["trials_completed"] > 0:
            elapsed = (datetime.now() - study["started_at"]).total_seconds() / 60
            avg_time_per_trial = elapsed / study["trials_completed"]
            remaining_trials = study["n_trials"] - study["trials_completed"]
            estimated_time_remaining = avg_time_per_trial * remaining_trials

        return OptimizationProgressResponse(
            study_id=study_id,
            study_name=study["study_name"],
            status=StudyStatus(study["status"]),
            progress_percentage=study.get("progress_percentage", 0.0),
            trials_completed=study.get("trials_completed", 0),
            trials_total=study["n_trials"],
            current_best_value=study.get("best_value"),
            estimated_time_remaining_minutes=estimated_time_remaining,
            last_trial=last_trial,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/studies/{study_id}", response_model=CancelOptimizationResponse)
async def cancel_optimization(study_id: str):
    """
    Running optimization'ı iptal eder
    """
    try:
        if study_id not in active_studies:
            raise HTTPException(status_code=404, detail="Study not found")

        study = active_studies[study_id]

        if study["status"] != StudyStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Study is not running")

        # Update status
        active_studies[study_id]["status"] = StudyStatus.CANCELLED
        active_studies[study_id]["completed_at"] = datetime.now()

        # Send WebSocket message
        await send_websocket_message(study_id, {
            "message_type": "cancelled",
            "study_id": study_id,
            "message": "Optimization cancelled by user",
            "timestamp": datetime.now().isoformat(),
        })

        return CancelOptimizationResponse(
            study_id=study_id,
            message="Optimization cancelled",
            trials_completed=study.get("trials_completed", 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-spaces/{algorithm}", response_model=AlgorithmSearchSpaceResponse)
async def get_algorithm_search_space(algorithm: AlgorithmType):
    """
    Algoritmanın arama uzayını getirir
    """
    try:
        search_space = get_search_space(algorithm.value)

        parameters = []
        for param_name, config in search_space.items():
            param_info = SearchSpaceInfo(
                parameter_name=param_name,
                param_type=config["type"],
                default=config["default"],
                description=config["description"],
                low=config.get("low"),
                high=config.get("high"),
                choices=config.get("choices"),
            )
            parameters.append(param_info)

        return AlgorithmSearchSpaceResponse(
            algorithm=algorithm,
            parameters=parameters,
            total_parameters=len(parameters)
        )

    except Exception as e:
        logger.error(f"Failed to get search space: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{study_id}")
async def websocket_endpoint(websocket: WebSocket, study_id: str):
    """
    WebSocket endpoint for real-time updates
    """
    await websocket.accept()

    # Add to connections
    if study_id not in websocket_connections:
        websocket_connections[study_id] = []
    websocket_connections[study_id].append(websocket)

    logger.info(f"WebSocket connected for study {study_id}")

    try:
        # Send initial status
        if study_id in active_studies:
            study = active_studies[study_id]
            await websocket.send_json({
                "message_type": "status_update",
                "study_id": study_id,
                "status": study["status"],
                "progress_percentage": study.get("progress_percentage", 0.0),
                "trials_completed": study.get("trials_completed", 0),
                "timestamp": datetime.now().isoformat(),
            })

        # Keep connection alive
        while True:
            # Receive messages (for ping/pong)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"message_type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for study {study_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Remove from connections
        if study_id in websocket_connections and websocket in websocket_connections[study_id]:
            websocket_connections[study_id].remove(websocket)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "active_studies": len(active_studies),
        "websocket_connections": sum(len(conns) for conns in websocket_connections.values())
    }
