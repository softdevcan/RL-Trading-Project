"""In-process ASGI client wrappers for all FastAPI backend endpoints.

Dash and FastAPI share one Python process (Dash is mounted under /dash via
WSGIMiddleware), so routing calls through a real TCP/HTTP round-trip was
pure overhead. This module instead talks to the FastAPI `app` object
directly over httpx's ASGI transport: same routing/validation/exception
behavior as a real request, no socket involved.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx

# Base path for API routes (kept for compatibility with callers/logging).
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8888/api")
TIMEOUT = 30  # seconds

import logging as _logging
_api_log = _logging.getLogger(__name__)
# Every Dash callback makes an in-process ASGI call now; httpx's default
# INFO-level "HTTP Request: ..." line per call would otherwise flood logs.
_logging.getLogger("httpx").setLevel(_logging.WARNING)

def _asgi_transport() -> httpx.ASGITransport:
    """Lazily resolve the FastAPI app (avoids circular import at module load)."""
    from app.main import app as _fastapi_app
    return httpx.ASGITransport(app=_fastapi_app)


def _run(coro):
    """Run an async call from Dash's synchronous callback context.

    A fresh client/loop per call keeps this safe to invoke from any thread
    Dash schedules callbacks on, at the cost of the (cheap, in-process)
    ASGI handshake each time — no TCP socket, no connection pool to manage.
    """
    return asyncio.run(coro)


def _forward_auth() -> tuple[Dict[str, str], Dict[str, str]]:
    """Tarayicinin oturumunu ic ASGI cagrisina tasi.

    Dash callback'i Flask istek baglaminda calisir; oradaki cerezleri
    (oturum + CSRF) FastAPI'ye ilettigimizde AuthGateMiddleware istegi ayni
    kullanici adina dogrular ve calisma alani (workspace) baglamini kurar.
    Baglam yoksa (test, arka plan is parcacigi) bos doner — auth kapaliysa
    zaten gerekmez.
    """
    try:
        from flask import has_request_context, request as flask_request

        if not has_request_context():
            return {}, {}
        cookies = dict(flask_request.cookies)
        headers: Dict[str, str] = {}
        from app.core.config import get_settings

        csrf = cookies.get(get_settings().CSRF_COOKIE_NAME)
        if csrf:
            # /api/* yazma istekleri double-submit CSRF dogrulamasindan gecer.
            headers["X-CSRF-Token"] = csrf
        return headers, cookies
    except Exception:  # Flask baglami yok / auth kapali
        return {}, {}


def _client(cookies: Dict[str, str], headers: Dict[str, str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_asgi_transport(),
        base_url="http://dash-internal/api",
        cookies=cookies,
        headers=headers,
    )


def _get(path: str, params: Optional[Dict] = None, timeout: int = TIMEOUT) -> Any:
    """GET request, returns parsed JSON or None on error."""
    headers, cookies = _forward_auth()
    try:
        async def _do():
            async with _client(cookies, headers) as client:
                resp = await client.get(path, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
        return _run(_do())
    except Exception as exc:
        _api_log.warning("GET %s%s failed: %s", API_BASE, path, exc)
        return None


def _post(path: str, json: Optional[Dict] = None, params: Optional[Dict] = None) -> Any:
    """POST request, returns parsed JSON or None on error."""
    headers, cookies = _forward_auth()
    try:
        async def _do():
            async with _client(cookies, headers) as client:
                r = await client.post(path, json=json, params=params, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()
        return _run(_do())
    except Exception as exc:
        # FastAPI 422 (validation) bodies carry the actual fault — log them
        # otherwise debugging is a guessing game.
        body = None
        resp = getattr(exc, "response", None)
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


def _request(method: str, path: str, json: Optional[Dict] = None) -> Any:
    """PATCH/DELETE gibi diger metotlar icin ortak yol."""
    headers, cookies = _forward_auth()
    try:
        async def _do():
            async with _client(cookies, headers) as client:
                r = await client.request(method, path, json=json, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()
        return _run(_do())
    except Exception as exc:
        _api_log.warning("%s %s%s failed: %s", method, API_BASE, path, exc)
        return None


def _post_raw(path: str, json: Optional[Dict] = None) -> Dict:
    """POST — hata govdesini de dondurur (yonetim ekranlarinda mesaj gerekli)."""
    headers, cookies = _forward_auth()

    async def _do():
        async with _client(cookies, headers) as client:
            r = await client.post(path, json=json, timeout=TIMEOUT)
            try:
                body = r.json()
            except Exception:
                body = {"detail": r.text[:300]}
            return {"ok": r.is_success, "status": r.status_code, "body": body}

    try:
        return _run(_do())
    except Exception as exc:
        _api_log.warning("POST %s%s failed: %s", API_BASE, path, exc)
        return {"ok": False, "status": 0, "body": {"detail": str(exc)}}


def _request_raw(method: str, path: str, json: Optional[Dict] = None) -> Dict:
    """PATCH/DELETE — hata govdesini de dondurur.

    `_request` hatayi yutup None donuyor; hesap ekranlarinda kullaniciya
    "neden olmadi" demek gerektigi icin durum kodu ve govde saklanir.
    """
    headers, cookies = _forward_auth()

    async def _do():
        async with _client(cookies, headers) as client:
            r = await client.request(method, path, json=json, timeout=TIMEOUT)
            try:
                body = r.json()
            except Exception:
                body = {"detail": r.text[:300]}
            return {"ok": r.is_success, "status": r.status_code, "body": body}

    try:
        return _run(_do())
    except Exception as exc:
        _api_log.warning("%s %s%s failed: %s", method, API_BASE, path, exc)
        return {"ok": False, "status": 0, "body": {"detail": str(exc)}}


# ── Kullanici yonetimi (admin) ─────────────────────────────────────────────

def list_users() -> List[Dict]:
    data = _get("/admin/users") or {}
    return data.get("users", [])


def create_user(payload: Dict) -> Dict:
    return _post_raw("/admin/users", json=payload)


def update_user(user_id: str, payload: Dict) -> Dict:
    return _request("PATCH", f"/admin/users/{user_id}", json=payload) or {}


def reset_user_password(user_id: str, password: Optional[str] = None) -> Dict:
    return _post_raw(f"/admin/users/{user_id}/password", json={"password": password})


def revoke_user_sessions(user_id: str) -> Dict:
    return _post_raw(f"/admin/users/{user_id}/revoke-sessions", json=None)


def delete_user(user_id: str) -> Dict:
    return _request("DELETE", f"/admin/users/{user_id}") or {}


def get_audit_log(limit: int = 100) -> List[Dict]:
    data = _get("/admin/audit", params={"limit": limit}) or {}
    return data.get("entries", [])


# ── Kendi hesabi (her rol, viewer dahil) ───────────────────────────────────
# Yetki kontrolu uc noktada: hedef her zaman oturumdaki kullanici, govdeden
# kullanici kimligi gecilmez (bkz. app/api/routes/account.py).

def get_account() -> Dict:
    """Profil alanlari + calisma alani kullanimi (tek cagri)."""
    return _get("/account/me") or {}


def update_profile(full_name: str) -> Dict:
    return _request_raw("PATCH", "/account/profile", json={"full_name": full_name})


def get_own_sessions() -> Dict:
    return _get("/account/sessions") or {}


def revoke_other_sessions() -> Dict:
    return _post_raw("/account/sessions/revoke-others", json=None)


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


def model_return(metrics: Optional[Dict]) -> Optional[float]:
    """Modelin toplam getirisi — alan adi kaynaga gore degisir.

    RL ortami metrigi `cumulative_return` adiyla yazar (env/trading_env.py),
    akademik analiz raporu ise `total_return` adiyla
    (app/services/model_analysis.py). Panolar tek ad okuyunca eksik alan
    `.get(key, 0)` varsayilanina dusup sessizce %0.0 gosteriyordu.

    None doner: metrik yok ya da iki ad da bulunamadi (cagiran taraf "—"
    basabilsin diye; 0 dondurup gercek sifir getiriyle karistirmiyoruz).
    """
    if not metrics:
        return None
    for key in ("cumulative_return", "total_return"):
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def start_training(payload: Dict) -> Dict:
    return _post("/trading/train", json=payload) or {}


def get_training_status() -> Dict:
    return _get("/trading/train/status") or {}


def get_training_estimate(algorithm: str, phase: int, total_timesteps: int) -> Dict:
    """Egitim baslamadan once tahmini sure (bkz. app/services/training_eta.py)."""
    return _get("/trading/train/estimate", params={
        "algorithm": algorithm,
        "phase": phase,
        "total_timesteps": total_timesteps,
    }) or {}


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
