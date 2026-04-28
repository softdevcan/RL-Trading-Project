"""
Trading Schemas
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Literal
from datetime import datetime


class TrainingRequest(BaseModel):
    """Request model for starting training"""
    algorithm: str = Field(
        default="A2C",
        description="RL algorithm to use (A2C, PPO, TD3, SAC)"
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
    hyperparameter_study: Optional[str] = Field(
        default=None,
        description="Name of hyperparameter study file to use (e.g., 'best_params_ppo_ppo_optimization_753db76c.json')"
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
                "max_shares_per_trade": 100,
                "hyperparameter_study": None
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
    state: Literal["idle", "running", "completed", "error"] = "idle"
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
    algorithm: Optional[str] = None
    phase: Optional[int] = None
    metrics: Dict = {}


# ==================== DAILY TRADING SCHEMAS ====================

class DailyDecisionRequest(BaseModel):
    """Request model for daily trading decision"""
    model_name: str = Field(
        description="Name of the trained model to use"
    )
    balance: float = Field(
        description="Current cash balance",
        gt=0
    )
    shares: Dict[str, int] = Field(
        description="Current shares owned for each symbol"
    )
    risk_mode: Literal["conservative", "moderate", "aggressive"] = Field(
        default="moderate",
        description="Risk mode: conservative, moderate, or aggressive"
    )
    max_shares_per_trade: int = Field(
        default=100,
        description="Maximum shares per trade",
        gt=0
    )
    date: Optional[str] = Field(
        default=None,
        description="Target date (YYYY-MM-DD), default: today"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "ppo_phase1_20241113_143022",
                "balance": 1000000,
                "shares": {
                    "ASELS.IS": 100,
                    "THYAO.IS": 50,
                    "EREGL.IS": 200,
                    "KCHOL.IS": 80,
                    "SAHOL.IS": 120
                },
                "risk_mode": "moderate",
                "max_shares_per_trade": 100,
                "date": "2025-11-13"
            }
        }


class TradeDecision(BaseModel):
    """Single trade decision"""
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    raw_signal: float
    shares: int
    price: float
    cost: Optional[float] = 0.0
    revenue: Optional[float] = 0.0
    commission: float = 0.0
    reason: str
    executed: bool


class PortfolioSnapshot(BaseModel):
    """Portfolio snapshot at a point in time"""
    balance: float
    shares: Dict[str, int]
    portfolio_value: float


class DailyDecisionResponse(BaseModel):
    """Response model for daily trading decision"""
    date: str
    decisions: List[TradeDecision]
    portfolio_before: PortfolioSnapshot
    portfolio_after: PortfolioSnapshot
    summary: Dict

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-11-13",
                "decisions": [
                    {
                        "symbol": "ASELS.IS",
                        "action": "BUY",
                        "raw_signal": 0.87,
                        "shares": 45,
                        "price": 23.50,
                        "cost": 1057.50,
                        "commission": 1.06,
                        "reason": "Strong buy signal (0.87)",
                        "executed": True
                    }
                ],
                "portfolio_before": {
                    "balance": 1000000,
                    "shares": {"ASELS.IS": 100},
                    "portfolio_value": 1050000
                },
                "portfolio_after": {
                    "balance": 998942.50,
                    "shares": {"ASELS.IS": 145},
                    "portfolio_value": 1052000
                },
                "summary": {
                    "total_trades": 2,
                    "total_commission": 5.75,
                    "daily_return_pct": 1.45
                }
            }
        }


class PortfolioHistoryResponse(BaseModel):
    """Portfolio history response"""
    dates: List[str]
    portfolio_values: List[float]
    daily_returns: List[float]
    balances: List[float]
