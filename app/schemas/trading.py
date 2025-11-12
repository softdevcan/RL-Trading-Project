"""
Trading Schemas
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime


class TrainingRequest(BaseModel):
    """Request model for starting training"""
    algorithm: str = Field(
        default="A2C",
        description="RL algorithm to use (A2C, PPO, TD3)"
    )
    phase: int = Field(
        default=1,
        description="Training phase (1, 2, or 3)",
        ge=1,
        le=3
    )
    total_timesteps: int = Field(
        default=50_000,
        description="Total training timesteps",
        gt=0
    )
    learning_rate: float = Field(
        default=0.0007,
        description="Learning rate",
        gt=0
    )
    initial_balance: float = Field(
        default=1_000_000,
        description="Initial portfolio balance",
        gt=0
    )
    commission_rate: float = Field(
        default=0.001,
        description="Transaction commission rate",
        ge=0,
        lt=1
    )
    max_shares_per_trade: int = Field(
        default=100,
        description="Maximum shares per trade",
        gt=0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "algorithm": "A2C",
                "phase": 1,
                "total_timesteps": 50000,
                "learning_rate": 0.0007,
                "initial_balance": 1000000,
                "commission_rate": 0.001,
                "max_shares_per_trade": 100
            }
        }


class TrainingResponse(BaseModel):
    """Response model for training start"""
    message: str
    training_id: str
    status: str


class TrainingStatus(BaseModel):
    """Training status model"""
    is_training: bool
    current_step: int
    total_steps: int
    progress: float = Field(ge=0, le=1)
    start_time: Optional[str] = None
    metrics: Dict = {}
    error: Optional[str] = None


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    cumulative_return: float = Field(description="Total return")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown: float = Field(description="Maximum drawdown")
    final_portfolio_value: float = Field(description="Final portfolio value")
    total_trades: int = Field(description="Number of trades")
    algorithm: Optional[str] = None
    phase: Optional[int] = None
    total_timesteps: Optional[int] = None
    learning_rate: Optional[float] = None
    trained_at: Optional[str] = None
    trades: Optional[List[Dict]] = Field(default=[], description="Trade history")
    portfolio_history: Optional[List[float]] = Field(default=[], description="Portfolio value over time")

    class Config:
        json_schema_extra = {
            "example": {
                "cumulative_return": 0.15,
                "sharpe_ratio": 1.25,
                "max_drawdown": -0.08,
                "final_portfolio_value": 1150000,
                "total_trades": 245,
                "algorithm": "A2C",
                "phase": 1,
                "total_timesteps": 50000,
                "learning_rate": 0.0007,
                "trained_at": "2024-01-15T10:30:00"
            }
        }


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    path: str
    created_at: str
    metrics: Dict = {}
