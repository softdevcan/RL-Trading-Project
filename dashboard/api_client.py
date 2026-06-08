"""HTTP client wrappers for all FastAPI backend endpoints."""

import os
import requests
from typing import Any, Dict, List, Optional

# Base URL - can be overridden via environment variable
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8888/api")
TIMEOUT = 30  # seconds


import logging as _logging
_api_log = _logging.getLogger(__name__)


def _get(path: str, params: Optional[Dict] = None, timeout: int = TIMEOUT) -> Any:
    """GET request, returns parsed JSON or None on error."""
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        _api_log.warning("GET %s%s failed: %s", API_BASE, path, exc)
        return None


def _post(path: str, json: Optional[Dict] = None, params: Optional[Dict] = None) -> Any:
    """POST request, returns parsed JSON or None on error."""
    resp = None
    try:
        resp = requests.post(
            f"{API_BASE}{path}", json=json, params=params, timeout=TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        # FastAPI 422 (validation) bodies carry the actual fault — log them
        # otherwise debugging is a guessing game.
        body = None
        if resp is not None:
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:500] if resp.text else None
        _api_log.warning(
            "POST %s%s failed: %s | request=%s | response=%s",
            API_BASE, path, exc, json, body,
        )
        return None


# ── Trading ────────────────────────────────────────────────────────────────

def get_health() -> Dict:
    return _get("/trading/health") or {}


def get_models() -> List[Dict]:
    data = _get("/trading/models")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("models", [])
    return []


def get_model_metrics(model_name: str) -> Dict:
    return _get(f"/trading/models/{model_name}/metrics") or {}


def start_training(payload: Dict) -> Dict:
    return _post("/trading/train", json=payload) or {}


def get_training_status() -> Dict:
    return _get("/trading/train/status") or {}


def get_data_info() -> Dict:
    return _get("/trading/data/info") or {}


def get_data_list() -> List[Dict]:
    data = _get("/trading/data/list")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("datasets", [])
    return []


def generate_data(payload: Optional[Dict] = None) -> Dict:
    return _post("/trading/data/generate", json=payload or {}) or {}


def get_data_status() -> Dict:
    return _get("/trading/data/status") or {}


def update_data(payload: Dict) -> Dict:
    return _post("/trading/data/update", json=payload) or {}


def get_earliest_date(source: str = "borsapy") -> Dict:
    """Belirtilen kaynak için en eski mevcut veri tarihini döndür.
    Uzun sürebilir (period='max' sorgusu), timeout=90s."""
    return _get("/trading/data/earliest", params={"source": source}, timeout=90) or {}


def get_daily_decision(payload: Dict) -> Dict:
    return _post("/trading/daily-decision", json=payload) or {}


def apply_decision(date: str) -> Dict:
    """Backend bekler: POST /trading/apply-decision?date=YYYY-MM-DD (query param)."""
    return _post("/trading/apply-decision", params={"date": date}) or {}


def get_latest_portfolio() -> Dict:
    return _get("/trading/latest-portfolio") or {}


def get_portfolio_history() -> Dict:
    return _get("/trading/portfolio-history") or {}


def get_decisions_history() -> Dict:
    """All saved daily decisions, newest first."""
    return _get("/trading/decisions-history") or {"dates": [], "decisions": {}}


def generate_report() -> Dict:
    return _post("/trading/analysis/generate-report", json={}) or {}


def get_model_comparison() -> Dict:
    return _get("/trading/analysis/model-comparison") or {}


def get_best_models() -> Dict:
    return _get("/trading/analysis/best-models") or {}


# ── Config ─────────────────────────────────────────────────────────────────

def get_config_algorithms() -> List[Dict]:
    data = _get("/config/algorithms") or {}
    return data.get("algorithms", [])


def get_config_phases() -> List[Dict]:
    data = _get("/config/phases") or {}
    return data.get("phases", [])


def get_config_reward_types() -> List[Dict]:
    data = _get("/config/reward-types") or {}
    return data.get("reward_types", [])


def get_config_feature_groups() -> Dict:
    return _get("/config/feature-groups") or {}


# ── Hyperopt ───────────────────────────────────────────────────────────────

def start_hyperopt(payload: Dict) -> Dict:
    return _post("/hyperopt/start", json=payload) or {}


def get_hyperopt_studies() -> List[Dict]:
    data = _get("/hyperopt/studies")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("studies", [])
    return []


def get_hyperopt_study(study_id: str) -> Dict:
    return _get(f"/hyperopt/studies/{study_id}") or {}


def get_hyperopt_progress(study_id: str) -> Dict:
    return _get(f"/hyperopt/studies/{study_id}/progress") or {}


def get_search_spaces(algorithm: str) -> Dict:
    return _get(f"/hyperopt/search-spaces/{algorithm}") or {}


def get_hyperopt_data_range() -> Dict:
    """Returns {available, min_date, max_date, symbols, total_rows} from cached CSV."""
    return _get("/hyperopt/data-range") or {"available": False, "symbols": []}


# ── Prediction ─────────────────────────────────────────────────────────────

def get_prediction_models() -> List[Dict]:
    data = _get("/prediction/models")
    if isinstance(data, dict):
        return data.get("models", [])
    return []


def get_prediction_symbols() -> List[Dict]:
    """Egitilebilir sembolleri ve her sembolun desteklenen kaynaklarini dondur.

    Returns: List[{symbol, name, category, sources}]
    """
    data = _get("/prediction/symbols")
    if isinstance(data, dict):
        return data.get("symbols", [])
    if isinstance(data, list):
        return data
    return []


def train_prediction(payload: Dict) -> Dict:
    """Arka plan egitimi tetikle; 202 Accepted hemen doner."""
    return _post("/prediction/train", json=payload) or {}


def get_prediction_train_status(
    symbol: str, horizon: str = "daily", source: Optional[str] = None,
) -> Dict:
    params: Dict[str, Any] = {"symbol": symbol, "horizon": horizon}
    if source:
        params["source"] = source
    return _get("/prediction/train/status", params=params) or {}


def make_prediction(payload: Dict) -> Dict:
    return _post("/prediction/predict", json=payload) or {}


def evaluate_prediction(payload: Dict) -> Dict:
    return _post("/prediction/evaluate", json=payload) or {}


def get_prediction_performance(symbol: str) -> Dict:
    return _get(f"/prediction/performance/{symbol}") or {}


def get_prediction_history(symbol: str) -> Dict:
    return _get(f"/prediction/predictions/{symbol}") or {}


def get_prediction_chart_data(symbol: str) -> Dict:
    return _get(f"/prediction/chart-data/{symbol}") or {}


def get_price_history(symbol: str, source: Optional[str] = None, days: int = 90) -> Dict:
    """Sembol icin tarihsel fiyat verisi (OHLCV + MA + BB)."""
    params: Dict = {"symbol": symbol, "days": days}
    if source:
        params["source"] = source
    return _get("/prediction/price-history", params=params) or {}


def train_ensemble(payload: Dict) -> Dict:
    return _post("/prediction/train-ensemble", json=payload) or {}


def cross_validate(payload: Dict) -> Dict:
    return _post("/prediction/cross-validate", json=payload) or {}


def optimize_prediction(payload: Dict) -> Dict:
    return _post("/prediction/optimize", json=payload) or {}


def get_gold_prices() -> Dict:
    return _get("/prediction/gold/prices") or {}


def get_gold_history() -> Dict:
    return _get("/prediction/gold/history") or {}
