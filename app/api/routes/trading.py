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
    ModelInfo, TrainingResponse
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

        # Algorithm-specific hyperparameters optimized for trading
        # Each algorithm has its own optimal learning rate
        if request.algorithm == "PPO":
            # PPO: Most stable for single-threaded training
            # Optimal learning rate for PPO in trading: 0.0003
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=0.0003,   # PPO-specific optimal LR
                n_steps=2048,           # Number of steps to run for each environment per update
                batch_size=64,          # Minibatch size
                n_epochs=10,            # Number of epochs when optimizing surrogate loss
                gamma=0.99,             # Discount factor
                gae_lambda=0.95,        # Factor for trade-off of bias vs variance for GAE
                clip_range=0.2,         # Clipping parameter for PPO
                ent_coef=0.15,          # INCREASED from 0.05 to 0.15 for much more exploration!
                vf_coef=0.5,            # Value function coefficient
                max_grad_norm=0.5,      # Gradient clipping
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
            "trades": trades_data,
            "portfolio_history": portfolio_history
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_with_config, f, indent=2)

        # Update training state
        training_state["is_training"] = False
        training_state["metrics"] = metrics_with_config
        training_state["current_step"] = request.total_timesteps

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        training_state["is_training"] = False
        training_state["error"] = str(e)
        raise
