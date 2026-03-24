"""
Hyperparameter Optimization API Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class AlgorithmType(str, Enum):
    """RL Algoritma tipleri"""
    PPO = "ppo"
    A2C = "a2c"
    TD3 = "td3"
    SAC = "sac"


class StudyStatus(str, Enum):
    """Optimization study durumları"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrialState(str, Enum):
    """Trial durumları"""
    RUNNING = "running"
    COMPLETE = "complete"
    PRUNED = "pruned"
    FAIL = "fail"


# ==================== Request Schemas ====================

class OptimizationRequest(BaseModel):
    """Hiper parametre optimizasyon isteği"""

    algorithm: AlgorithmType = Field(
        ...,
        description="Optimize edilecek algoritma"
    )

    n_trials: int = Field(
        50,
        ge=1,
        le=200,
        description="Toplam trial sayısı"
    )

    total_timesteps: int = Field(
        100_000,
        ge=10_000,
        le=1_000_000,
        description="Her trial için training timesteps"
    )

    stock_symbols: Optional[List[str]] = Field(
        None,
        description="Hisse senedi sembolleri (None ise PHASE1_SYMBOLS)"
    )

    train_start: str = Field(
        "2018-01-01",
        description="Training başlangıç tarihi (YYYY-MM-DD)"
    )

    train_end: str = Field(
        "2022-12-31",
        description="Training bitiş tarihi (YYYY-MM-DD)"
    )

    val_start: str = Field(
        "2023-01-01",
        description="Validation başlangıç tarihi (YYYY-MM-DD)"
    )

    val_end: str = Field(
        "2023-12-31",
        description="Validation bitiş tarihi (YYYY-MM-DD)"
    )

    eval_freq: int = Field(
        5_000,
        ge=1_000,
        description="Evaluation frequency"
    )

    n_eval_episodes: int = Field(
        5,
        ge=1,
        le=20,
        description="Evaluation episode sayısı"
    )

    study_name: Optional[str] = Field(
        None,
        description="Özel study ismi (None ise otomatik)"
    )

    phase: int = Field(
        2,
        ge=1,
        le=2,
        description="Trading phase (1=56 features, 2=97 features with fundamental+macro)"
    )

    reward_type: str = Field(
        "psr",
        description="Reward function type (simple or psr)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "algorithm": "ppo",
                "n_trials": 50,
                "total_timesteps": 100_000,
                "train_start": "2018-01-01",
                "train_end": "2022-12-31",
                "val_start": "2023-01-01",
                "val_end": "2023-12-31"
            }
        }


class SearchSpaceUpdateRequest(BaseModel):
    """Arama uzayı güncelleme isteği"""

    algorithm: AlgorithmType = Field(..., description="Algoritma")
    parameter_name: str = Field(..., description="Parametre ismi")

    # For loguniform/uniform/int
    low: Optional[float] = Field(None, description="Minimum değer")
    high: Optional[float] = Field(None, description="Maksimum değer")

    # For categorical
    choices: Optional[List[Any]] = Field(None, description="Seçenekler")

    class Config:
        json_schema_extra = {
            "example": {
                "algorithm": "ppo",
                "parameter_name": "learning_rate",
                "low": 1e-5,
                "high": 1e-2
            }
        }


# ==================== Response Schemas ====================

class TrialInfo(BaseModel):
    """Trial bilgileri"""

    trial_number: int
    state: TrialState
    value: Optional[float] = None
    params: Dict[str, Any]
    duration_seconds: Optional[float] = None
    user_attrs: Dict[str, Any] = {}
    datetime_start: Optional[datetime] = None
    datetime_complete: Optional[datetime] = None


class StudyInfo(BaseModel):
    """Study bilgileri"""

    study_id: str
    study_name: str
    algorithm: AlgorithmType
    status: StudyStatus

    # Configuration
    n_trials: int
    total_timesteps: int
    stock_symbols: List[str]
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    phase: int = 2
    reward_type: str = "psr"

    # Progress
    trials_completed: int = 0
    trials_pruned: int = 0
    trials_failed: int = 0
    progress_percentage: float = 0.0

    # Results
    best_value: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    best_trial: Optional[int] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Additional info
    error_message: Optional[str] = None


class StudyListResponse(BaseModel):
    """Study listesi response"""

    studies: List[StudyInfo]
    total: int


class StudyDetailResponse(BaseModel):
    """Study detay response"""

    study: StudyInfo
    trials: List[TrialInfo]

    # Statistics
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class SearchSpaceInfo(BaseModel):
    """Arama uzayı bilgisi"""

    parameter_name: str
    param_type: str  # loguniform, uniform, int, categorical
    default: Any
    description: str

    # For loguniform/uniform/int
    low: Optional[float] = None
    high: Optional[float] = None

    # For categorical
    choices: Optional[List[Any]] = None


class AlgorithmSearchSpaceResponse(BaseModel):
    """Algoritma arama uzayı response"""

    algorithm: AlgorithmType
    parameters: List[SearchSpaceInfo]
    total_parameters: int


class OptimizationStartResponse(BaseModel):
    """Optimizasyon başlatma response"""

    study_id: str
    study_name: str
    message: str
    estimated_duration_minutes: float


class OptimizationProgressResponse(BaseModel):
    """Optimizasyon progress response"""

    study_id: str
    study_name: str
    status: StudyStatus
    progress_percentage: float
    trials_completed: int
    trials_total: int
    current_best_value: Optional[float] = None
    estimated_time_remaining_minutes: Optional[float] = None
    last_trial: Optional[TrialInfo] = None


class OptimizationResultsResponse(BaseModel):
    """Optimizasyon sonuç response"""

    study_id: str
    study_name: str
    algorithm: AlgorithmType
    status: StudyStatus

    # Best results
    best_value: float
    best_params: Dict[str, Any]
    best_trial: int

    # Statistics
    total_trials: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int

    mean_value: float
    median_value: float
    std_value: float

    # Parameter importance (top 5)
    param_importances: Optional[Dict[str, float]] = None

    # Timing
    total_duration_minutes: float
    created_at: datetime
    completed_at: datetime


class CancelOptimizationResponse(BaseModel):
    """Optimizasyon iptal response"""

    study_id: str
    message: str
    trials_completed: int


# ==================== WebSocket Messages ====================

class WSProgressUpdate(BaseModel):
    """WebSocket progress güncellemesi"""

    message_type: str = "progress_update"
    study_id: str
    progress_percentage: float
    trials_completed: int
    current_best_value: Optional[float] = None
    last_trial_value: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class WSTrialComplete(BaseModel):
    """WebSocket trial tamamlanma mesajı"""

    message_type: str = "trial_complete"
    study_id: str
    trial_number: int
    trial_value: float
    trial_params: Dict[str, Any]
    is_best: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class WSOptimizationComplete(BaseModel):
    """WebSocket optimization tamamlanma mesajı"""

    message_type: str = "optimization_complete"
    study_id: str
    best_value: float
    best_params: Dict[str, Any]
    total_trials: int
    timestamp: datetime = Field(default_factory=datetime.now)


class WSError(BaseModel):
    """WebSocket hata mesajı"""

    message_type: str = "error"
    study_id: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.now)
