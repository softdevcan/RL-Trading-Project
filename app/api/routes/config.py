"""
Config API Routes

Sayfa-bazli hardcoded listeleri tek noktadan beslemek icin merkezi okuma uclari.
Tuketiciler: dashboard sayfalari (training, hyperopt, daily_trading, prediction).
"""

from typing import Any, Dict, List

from fastapi import APIRouter

from prediction.feature_groups import (
    DEFAULT_TARGET_TYPE,
    REGISTRY as FEATURE_REGISTRY,
    TARGET_TYPES,
    default_groups,
    groups_by_category,
)


router = APIRouter(prefix="/config", tags=["Config"])


# RL training icin desteklenen algoritmalar (SB3 destekli).
# Value: backend AlgorithmType enum (lowercase); Label: UI gosterimi (upper).
_ALGORITHMS: List[Dict[str, str]] = [
    {"value": "ppo", "label": "PPO"},
    {"value": "a2c", "label": "A2C"},
    {"value": "td3", "label": "TD3"},
    {"value": "sac", "label": "SAC"},
]

# Trading fazlari
_PHASES: List[Dict[str, Any]] = [
    {"value": 1, "label": "Faz 1 (56 ozellik, temel RL)"},
    {"value": 2, "label": "Faz 2 (97 ozellik, fundamental + makro)"},
]

# Hyperopt reward tipleri (env/reward_functions.py kabul ettikleri)
_REWARD_TYPES: List[Dict[str, str]] = [
    {"value": "psr", "label": "PSR (Probabilistic Sharpe Ratio)"},
    {"value": "simple", "label": "Simple (baseline)"},
]


@router.get("/algorithms")
async def get_algorithms() -> Dict[str, Any]:
    """RL training icin desteklenen algoritmalar."""
    return {"algorithms": _ALGORITHMS}


@router.get("/phases")
async def get_phases() -> Dict[str, Any]:
    """Trading fazlari (Phase 1 / Phase 2)."""
    return {"phases": _PHASES}


@router.get("/reward-types")
async def get_reward_types() -> Dict[str, Any]:
    """Hyperopt + RL training icin reward tipleri."""
    return {"reward_types": _REWARD_TYPES}


@router.get("/feature-groups")
async def get_feature_groups() -> Dict[str, Any]:
    """
    Prediction feature gruplari registry'si.

    Yanit:
      {
        "groups": { id: {label, description, category, default, requires}, ... },
        "by_category": { category: [{id, label, ...}, ...], ... },
        "defaults": [id, ...],
        "target_types": { id: {label, description}, ... },
        "default_target_type": "log_return"
      }
    """
    return {
        "groups": FEATURE_REGISTRY,
        "by_category": groups_by_category(),
        "defaults": default_groups(),
        "target_types": TARGET_TYPES,
        "default_target_type": DEFAULT_TARGET_TYPE,
    }
