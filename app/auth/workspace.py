"""Kullanici bazli calisma alani cozumleyici (hibrit izolasyon).

Ilke
----
ORTAK (paylasilan, kullanici basina kopyalanmaz):
    data/bist, data/macro, data/fundamental, data/gold ve ham/islenmis
    OHLCV CSV'leri. Piyasa verisi herkes icin ayni; N kullanici icin N kez
    yfinance'e gitmek hem yavas hem rate-limit riski.

KULLANICI BAZLI (workspaces/<user_id>/ altinda):
    models/            RL modelleri (.zip)
    models/prediction/ tahmin modelleri
    results/           metrik JSON'lari, deney kayitlari
    data/predictions/  uretilen tahminler
    data/live_trading/ gunluk kararlar + portfoy gecmisi
    logs/              TensorBoard

Kullanim
--------
    from app.auth.workspace import models_dir, use_workspace

    path = models_dir()                 # aktif kullaniciya gore cozulur
    with use_workspace(user_id):        # arka plan is parcaciklari icin
        ...

Aktif kullanici bulunamazsa (AUTH_ENABLED=False, script, test) eski global
dizinler dondurulur — mevcut davranis birebir korunur.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from contextvars import ContextVar
from typing import Iterator

from app.core.config import get_settings

log = logging.getLogger(__name__)

# Aktif kullanici baglami: {"id", "email", "role"}
_current_user: ContextVar[dict | None] = ContextVar("rlt_current_user", default=None)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Calisma alani icindeki goreli yollar (tek dogruluk kaynagi).
_KINDS = {
    "models": "models",
    "prediction_models": os.path.join("models", "prediction"),
    "results": "results",
    "predictions": os.path.join("data", "predictions"),
    "live_trading": os.path.join("data", "live_trading"),
    "logs": "logs",
    "hyperopt": os.path.join("results", "hyperparameter_studies"),
}


# ── Baglam yonetimi ───────────────────────────────────────────────────────

def set_current_user(user: dict | None):
    """ContextVar token dondurur; reset icin saklanmali."""
    return _current_user.set(user)


def reset_current_user(token) -> None:
    _current_user.reset(token)


def get_current_user() -> dict | None:
    return _current_user.get()


def current_user_id() -> str | None:
    user = _current_user.get()
    return user.get("id") if user else None


@contextlib.contextmanager
def use_workspace(user_id: str | None, email: str = "", role: str = "user") -> Iterator[None]:
    """Arka plan gorevleri icin baglam. ContextVar thread/task sinirinda
    tasinmadigindan egitim gibi is parcaciklarinda acikca sarmalanmali."""
    token = _current_user.set({"id": user_id, "email": email, "role": role} if user_id else None)
    try:
        yield
    finally:
        _current_user.reset(token)


# ── Yol cozumleme ─────────────────────────────────────────────────────────

def _legacy_dir(kind: str) -> str:
    """Kullanici oncesi (global) dizinler — config.py'deki ayarlar."""
    s = get_settings()
    return {
        "models": s.MODELS_DIR,
        "prediction_models": s.PREDICTION_MODELS_DIR,
        "results": s.RESULTS_DIR,
        "predictions": s.PREDICTIONS_DIR,
        "live_trading": os.path.join(s.DATA_DIR, "live_trading"),
        "logs": s.LOGS_DIR,
        "hyperopt": s.HYPEROPT_DIR,
    }[kind]


def isolation_active() -> bool:
    s = get_settings()
    return bool(s.AUTH_ENABLED and s.WORKSPACE_ISOLATION and current_user_id())


def workspace_root(user_id: str | None = None) -> str | None:
    """workspaces/<user_id> yolunu dondurur; izolasyon kapaliysa None."""
    s = get_settings()
    uid = user_id or current_user_id()
    if not uid or not s.AUTH_ENABLED or not s.WORKSPACE_ISOLATION:
        return None
    if not _SAFE_ID.match(uid):
        # Yol gecisi (path traversal) korumasi: id her zaman uuid4 hex olmali.
        raise ValueError(f"Gecersiz calisma alani kimligi: {uid!r}")
    return os.path.join(s.WORKSPACES_DIR, uid)


def resolve(kind: str, user_id: str | None = None, *, create: bool = True) -> str:
    """Yazma hedefi olan dizini dondurur."""
    if kind not in _KINDS:
        raise KeyError(f"Bilinmeyen calisma alani turu: {kind}")
    root = workspace_root(user_id)
    path = _legacy_dir(kind) if root is None else os.path.join(root, _KINDS[kind])
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def read_dirs(kind: str, user_id: str | None = None) -> list[str]:
    """Listeleme/okuma icin dizinler: once kullanicinin kendi alani, sonra
    (WORKSPACE_SHOW_LEGACY ise) kullanici oncesi ortak dizin — eski egitilmis
    modeller salt-okunur olarak herkese gorunur kalir."""
    primary = resolve(kind, user_id, create=False)
    dirs = [primary]
    if get_settings().WORKSPACE_SHOW_LEGACY:
        legacy = _legacy_dir(kind)
        if os.path.abspath(legacy) != os.path.abspath(primary):
            dirs.append(legacy)
    return [d for d in dirs if os.path.isdir(d)]


def find_file(kind: str, filename: str, user_id: str | None = None) -> str | None:
    """Dosyayi once kullanici alaninda, sonra eski ortak dizinde ara."""
    for directory in read_dirs(kind, user_id):
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate
    return None


# Kisayollar — cagri yerlerinde okunakli kalsin diye.
def models_dir(user_id: str | None = None) -> str:
    return resolve("models", user_id)


def prediction_models_dir(user_id: str | None = None) -> str:
    return resolve("prediction_models", user_id)


def results_dir(user_id: str | None = None) -> str:
    return resolve("results", user_id)


def predictions_dir(user_id: str | None = None) -> str:
    return resolve("predictions", user_id)


def live_trading_dir(user_id: str | None = None) -> str:
    return resolve("live_trading", user_id)


def logs_dir(user_id: str | None = None) -> str:
    return resolve("logs", user_id)


def hyperopt_dir(user_id: str | None = None) -> str:
    return resolve("hyperopt", user_id)


def ensure_workspace(user_id: str) -> str | None:
    """Kullanici olusturulurken tum alt dizinleri hazirla."""
    root = workspace_root(user_id)
    if root is None:
        return None
    for kind in _KINDS:
        os.makedirs(os.path.join(root, _KINDS[kind]), exist_ok=True)
    log.info("Calisma alani hazir: %s", root)
    return root


def workspace_usage(user_id: str) -> dict:
    """Admin panelinde gostermek icin basit kullanim ozeti."""
    root = workspace_root(user_id)
    if root is None or not os.path.isdir(root):
        return {"root": root, "exists": False, "bytes": 0, "files": 0}
    total_bytes = 0
    total_files = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total_bytes += os.path.getsize(os.path.join(dirpath, name))
                total_files += 1
            except OSError:
                continue
    return {"root": root, "exists": True, "bytes": total_bytes, "files": total_files}
