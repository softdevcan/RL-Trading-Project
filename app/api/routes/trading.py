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

from app.schemas.trading import (
    TrainingRequest, TrainingStatus, ModelMetrics,
    ModelInfo, TrainingResponse
)

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
        from stable_baselines3 import A2C, PPO, TD3
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

        # Create environment
        env = make_env(
            df=train_df,
            initial_balance=request.initial_balance,
            commission_rate=request.commission_rate,
            max_shares_per_trade=request.max_shares_per_trade
        )
        env = DummyVecEnv([lambda: env])

        # Create model based on algorithm with optimized hyperparameters
        model_class = {
            "A2C": A2C,
            "PPO": PPO,
            "TD3": TD3
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
                ent_coef=0.05,          # Increased for more exploration (trading needs active trading!)
                tensorboard_log='logs',
                verbose=1
            )
        elif request.algorithm == "A2C":
            # A2C: Synchronous version, needs careful tuning
            # Optimal learning rate for A2C: 0.0007
            model = model_class(
                'MlpPolicy',
                env,
                learning_rate=0.0007,   # A2C-specific optimal LR
                n_steps=512,            # Smaller steps for more frequent updates
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.05,          # Increased for more exploration
                vf_coef=0.5,            # Value function coefficient
                normalize_advantage=True,  # Normalize advantages for stability
                use_rms_prop=True,      # Use RMSprop optimizer (original A2C paper)
                tensorboard_log='logs',
                verbose=1
            )
        elif request.algorithm == "TD3":
            # TD3: Twin Delayed DDPG - for continuous action spaces
            # Optimal learning rate for TD3: 0.001
            from stable_baselines3.common.noise import NormalActionNoise

            # Add action noise for exploration
            action_noise = NormalActionNoise(
                mean=np.zeros(env.action_space.shape[0]),
                sigma=0.1 * np.ones(env.action_space.shape[0])
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
                target_policy_noise=0.2,  # Noise added to target policy
                target_noise_clip=0.5,  # Clip the noise
                tensorboard_log='logs',
                verbose=1
            )
        else:
            # Fallback to default settings
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

        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_name}'
        model.save(model_path)

        # Evaluate on test set
        test_env = make_env(
            df=test_df,
            initial_balance=request.initial_balance,
            commission_rate=request.commission_rate,
            max_shares_per_trade=request.max_shares_per_trade
        )
        test_env = DummyVecEnv([lambda: test_env])

        obs = test_env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)

        # Get metrics
        metrics = test_env.envs[0].get_metrics()

        # Convert numpy types to Python types and handle NaN
        metrics = {
            k: (None if (isinstance(v, (np.floating, float)) and np.isnan(v))
                else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
            for k, v in metrics.items()
        }

        # Save metrics
        os.makedirs('results', exist_ok=True)
        metrics_file = f'results/{model_name}_metrics.json'

        metrics_with_config = {
            **metrics,
            "algorithm": request.algorithm,
            "phase": request.phase,
            "total_timesteps": request.total_timesteps,
            "learning_rate": request.learning_rate,
            "trained_at": datetime.now().isoformat()
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_with_config, f, indent=2)

        # Update training state
        training_state["is_training"] = False
        training_state["metrics"] = metrics_with_config
        training_state["current_step"] = request.total_timesteps

    except Exception as e:
        training_state["is_training"] = False
        training_state["error"] = str(e)
        raise
