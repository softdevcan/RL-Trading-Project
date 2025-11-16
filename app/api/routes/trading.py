"""
Trading API Routes
FastAPI endpoints for RL trading system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import logging

from app.schemas.trading import (
    TrainingRequest, TrainingStatus, ModelMetrics,
    ModelInfo, TrainingResponse,
    DailyDecisionRequest, DailyDecisionResponse, TradeDecision,
    PortfolioSnapshot, PortfolioHistoryResponse
)

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading", tags=["Trading"])

# Global training state
training_state = {
    "is_training": False,
    "current_step": 0,
    "total_steps": 0,
    "start_time": None,
    "metrics": {},
    "error": None
}


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
        studies_dir = "results/hyperparameter_studies"

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

    if training_state["is_training"]:
        raise HTTPException(
            status_code=400,
            detail="Training already in progress"
        )

    # Reset training state
    training_state = {
        "is_training": True,
        "current_step": 0,
        "total_steps": request.total_timesteps,
        "start_time": datetime.now().isoformat(),
        "metrics": {},
        "error": None,
        "config": request.dict()
    }

    # Start training in background
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
    models_dir = "models"

    if not os.path.exists(models_dir):
        return []

    models = []

    for filename in os.listdir(models_dir):
        if filename.endswith(".zip"):
            model_path = os.path.join(models_dir, filename)

            # Try to load metrics from results
            model_name = filename.replace(".zip", "")
            metrics_file = f"results/{model_name}_metrics.json"

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
    metrics_file = f"results/{model_name}_metrics.json"

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
    start_date: str = None,
    end_date: str = None
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
            end_date=end_date
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
        if not os.path.exists('data/stock_data_with_indicators.csv'):
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
        import glob
        from datetime import datetime

        data_dir = "data"
        if not os.path.exists(data_dir):
            return {"datasets": []}

        # Find all CSV files in data directory
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

        datasets = []
        for csv_path in csv_files:
            try:
                filename = os.path.basename(csv_path)
                file_stat = os.stat(csv_path)

                # Try to load and get info
                fetcher = DataFetcher()
                df = fetcher.load_data(filename)

                if df is not None and len(df) > 0:
                    train_df, val_df, test_df = fetcher.split_data(df)

                    datasets.append({
                        "filename": filename,
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        "total_rows": len(df),
                        "train_rows": len(train_df),
                        "val_rows": len(val_df),
                        "test_rows": len(test_df),
                        "symbols": df.index.get_level_values('symbol').unique().tolist(),
                        "date_range": {
                            "start": str(df.index.get_level_values('date').min().date()),
                            "end": str(df.index.get_level_values('date').max().date())
                        },
                        "columns_count": len(df.columns)
                    })
            except Exception as e:
                # If file can't be loaded, just add basic info
                datasets.append({
                    "filename": os.path.basename(csv_path),
                    "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "error": str(e)
                })

        return {
            "status": "success",
            "count": len(datasets),
            "datasets": datasets
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a trained model"""
    model_path = f"models/{model_name}.zip"

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_name}"
        )

    # Delete model file
    os.remove(model_path)

    # Delete metrics if exists
    metrics_file = f"results/{model_name}_metrics.json"
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

        # Create environment factory function
        def make_training_env():
            return make_env(
                df=train_df,
                initial_balance=request.initial_balance,
                commission_rate=request.commission_rate,
                max_shares_per_trade=request.max_shares_per_trade
            )

        # Create one instance to get dimensions
        temp_env = make_training_env()
        action_dim = temp_env.action_space.shape[0]
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
        os.makedirs('logs', exist_ok=True)
        model_name = f"{request.algorithm.lower()}_phase{request.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load hyperparameters if study is specified
        hyperparams = {}
        if request.hyperparameter_study:
            study_path = os.path.join("results/hyperparameter_studies", request.hyperparameter_study)
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
        if request.algorithm == "PPO":
            # PPO: Most stable for single-threaded training
            # Use hyperparameters from study or defaults
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=hyperparams.get('learning_rate', 0.0003),
                n_steps=hyperparams.get('n_steps', 2048),
                batch_size=hyperparams.get('batch_size', 64),
                n_epochs=hyperparams.get('n_epochs', 10),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.2),
                ent_coef=hyperparams.get('ent_coef', 0.15),
                vf_coef=hyperparams.get('vf_coef', 0.5),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                tensorboard_log='logs',
                verbose=1
            )
        elif request.algorithm == "A2C":
            # A2C: Synchronous Advantage Actor-Critic
            # Best for: Fast training, stable convergence, trading environments
            # Paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)

            # Trading-optimized hyperparameters based on quant research:
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=0.0007,   # Slightly higher than PPO (on-policy needs faster updates)
                n_steps=256,            # REDUCED from 512: More frequent updates for volatile markets
                                        # Lower n_steps = faster reaction to market changes
                gamma=0.99,             # Standard discount factor for financial RL
                gae_lambda=0.95,        # Generalized Advantage Estimation (bias-variance tradeoff)
                ent_coef=0.01,          # REDUCED from 0.15: A2C needs LOWER entropy than PPO!
                                        # A2C is naturally more explorative due to on-policy nature
                                        # High entropy causes excessive random trading
                vf_coef=0.25,           # REDUCED from 0.5: Lower value loss weight
                                        # Prevents value function from dominating policy updates
                max_grad_norm=0.5,      # Gradient clipping (prevents exploding gradients)
                normalize_advantage=True,  # Critical for trading: normalizes profit/loss scales
                use_rms_prop=True,      # RMSprop optimizer (original A2C paper)
                                        # Better than Adam for on-policy algorithms
                rms_prop_eps=1e-5,      # RMSprop epsilon for numerical stability
                tensorboard_log='logs',
                verbose=1
            )
        elif request.algorithm == "TD3":
            # TD3: Twin Delayed DDPG - for continuous action spaces
            # Optimal learning rate for TD3: 0.001
            from stable_baselines3.common.noise import NormalActionNoise

            # Add action noise for exploration - INCREASED for more aggressive trading
            action_noise = NormalActionNoise(
                mean=np.zeros(action_dim),  # Use dimension from base_env
                sigma=0.2 * np.ones(action_dim)  # INCREASED from 0.1 to 0.2
            )

            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=0.001,    # TD3-specific optimal LR
                buffer_size=100000,     # Replay buffer size
                learning_starts=1000,   # Start learning after this many steps
                batch_size=256,         # Larger batch for off-policy learning
                tau=0.005,              # Soft update coefficient
                gamma=0.99,
                train_freq=1,           # Update the model every step
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,         # Delay policy updates (TD3 feature)
                target_policy_noise=0.3,  # INCREASED from 0.2 to 0.3 - Noise added to target policy
                target_noise_clip=0.5,  # Clip the noise
                tensorboard_log='logs',
                verbose=1
            )
        elif request.algorithm == "SAC":
            # SAC: Soft Actor-Critic - State of the art for continuous control
            # Generally the BEST algorithm for continuous action spaces like trading
            # Optimal learning rate for SAC: 0.0003
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=0.0003,   # SAC-specific optimal LR
                buffer_size=100000,     # Replay buffer size
                learning_starts=1000,   # Start learning after this many steps
                batch_size=256,         # Batch size for each gradient update
                tau=0.005,              # Soft update coefficient for target networks
                gamma=0.99,             # Discount factor
                train_freq=1,           # Update the model every step
                gradient_steps=1,       # Number of gradient steps per update
                ent_coef='auto_0.5',    # INCREASED initial entropy target (was 'auto')
                                        # This will start with higher exploration
                target_update_interval=1,  # Update target network every step
                use_sde=False,          # Don't use State Dependent Exploration (we have entropy)
                tensorboard_log='logs',
                verbose=1
            )
        else:
            # Fallback to default settings with high exploration
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=request.learning_rate,
                tensorboard_log='logs',
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
        train_metrics = env.envs[0].get_metrics()
        logger.info(f"Training metrics: total_trades={train_metrics['total_trades']}, "
                   f"final_value=${train_metrics['final_portfolio_value']:,.0f}")

        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_name}'
        model.save(model_path)

        # Evaluate on test set
        def make_test_env():
            return make_env(
                df=test_df,
                initial_balance=request.initial_balance,
                commission_rate=request.commission_rate,
                max_shares_per_trade=request.max_shares_per_trade
            )

        test_env = DummyVecEnv([make_test_env])

        obs = test_env.reset()
        done = np.array([False])
        test_step = 0

        # Track the actual environment instance
        actual_env = test_env.envs[0]

        while not done[0]:  # DummyVecEnv returns array
            action, _states = model.predict(obs, deterministic=False)  # Stochastic for better exploration

            # CRITICAL: Save state BEFORE step (which may autoreset)
            pre_step_trades = len(actual_env.trades_history)
            pre_step_trades_copy = actual_env.trades_history.copy()
            pre_step_portfolio_values = actual_env.portfolio_values.copy()

            obs, reward, done, info = test_env.step(action)
            test_step += 1

            # IMMEDIATELY restore if autoreset happened (trades_history cleared)
            if len(actual_env.trades_history) == 0 and pre_step_trades > 0:
                logger.info(f"Autoreset detected! Restoring {pre_step_trades} trades")
                actual_env.trades_history = pre_step_trades_copy
                actual_env.portfolio_values = pre_step_portfolio_values

            if done[0]:
                logger.info(f"Episode done at step {test_step}")
                break

        logger.info(f"Test evaluation: {test_step} steps completed")

        # Get metrics (from restored state if episode ended)
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
        os.makedirs('results', exist_ok=True)
        metrics_file = f'results/{model_name}_metrics.json'

        metrics_with_config = {
            **metrics,
            "algorithm": request.algorithm,
            "phase": request.phase,
            "total_timesteps": request.total_timesteps,
            "learning_rate": request.learning_rate,
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
        model_path = f"models/{request.model_name}"
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
        decision_file = 'data/live_trading/trade_decisions.json'

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
        history_file = 'data/live_trading/portfolio_history.csv'

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
        results_file = 'results/data/detailed_results.json'

        if not os.path.exists(results_file):
            raise HTTPException(
                status_code=404,
                detail="No analysis results found. Please run generate_academic_report.py first"
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
        results_file = 'results/data/detailed_results.json'

        if not os.path.exists(results_file):
            raise HTTPException(
                status_code=404,
                detail="No analysis results found. Please run generate_academic_report.py first"
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
            subprocess.run(['python', 'generate_academic_report.py'], check=True)

        background_tasks.add_task(run_analysis)

        return {
            "message": "Analysis report generation started",
            "status": "processing",
            "note": "Check results/ directory for outputs when complete"
        }

    except Exception as e:
        logger.error(f"Failed to start analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
