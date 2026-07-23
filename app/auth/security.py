"""Sifre hash'leme, JWT uretimi/dogrulamasi ve CSRF token'lari.

Notlar
------
* bcrypt 5.x parolanin 72 bayttan uzun olmasina izin vermez; parola once
  SHA-256 + base64 ile sabit uzunluga indirilir (yaygin ve guvenli kalip).
* Access token stateless JWT (kisa omurlu). Refresh token da JWT ama 'jti'
  claim'i DB'de tutuldugu icin iptal edilebilir.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.core.config import get_settings

log = logging.getLogger(__name__)

ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"

_DEV_SECRET_FILE = os.path.join("data", "auth", ".jwt_secret")
_secret_cache: str | None = None


class TokenError(Exception):
    """Gecersiz/suresi dolmus token."""


# ── Gizli anahtar ──────────────────────────────────────────────────────────

def get_secret_key() -> str:
    """JWT imzalama anahtari.

    Uretimde (DEBUG=False) JWT_SECRET_KEY zorunludur — bos ise acilista
    RuntimeError. Gelistirmede anahtar uretilip data/auth/.jwt_secret
    dosyasina yazilir; boylece yeniden baslatmada oturumlar dusmez.
    """
    global _secret_cache
    if _secret_cache:
        return _secret_cache

    settings = get_settings()
    if settings.JWT_SECRET_KEY:
        _secret_cache = settings.JWT_SECRET_KEY
        return _secret_cache

    if not settings.DEBUG:
        raise RuntimeError(
            "JWT_SECRET_KEY tanimli degil. Uretim ortaminda zorunludur. "
            "Uret: python -c \"import secrets;print(secrets.token_urlsafe(64))\""
        )

    os.makedirs(os.path.dirname(_DEV_SECRET_FILE), exist_ok=True)
    if os.path.exists(_DEV_SECRET_FILE):
        with open(_DEV_SECRET_FILE, "r", encoding="utf-8") as fh:
            _secret_cache = fh.read().strip()
    if not _secret_cache:
        _secret_cache = secrets.token_urlsafe(64)
        with open(_DEV_SECRET_FILE, "w", encoding="utf-8") as fh:
            fh.write(_secret_cache)
        log.warning("Gelistirme JWT anahtari uretildi: %s (uretimde ENV kullan)", _DEV_SECRET_FILE)
    return _secret_cache


def reset_secret_cache() -> None:
    global _secret_cache
    _secret_cache = None


# ── Parola ────────────────────────────────────────────────────────────────

def _prepare(password: str) -> bytes:
    """72 bayt bcrypt sinirini asmadan tam parolayi hesaba kat."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def validate_password_strength(password: str) -> list[str]:
    """Politika ihlallerini dondurur (bos liste = gecerli)."""
    settings = get_settings()
    problems: list[str] = []
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        problems.append(f"En az {settings.PASSWORD_MIN_LENGTH} karakter olmali")
    if not any(c.islower() for c in password):
        problems.append("En az bir kucuk harf icermeli")
    if not any(c.isupper() for c in password):
        problems.append("En az bir buyuk harf icermeli")
    if not any(c.isdigit() for c in password):
        problems.append("En az bir rakam icermeli")
    return problems


# ── JWT ───────────────────────────────────────────────────────────────────

def _encode(payload: dict[str, Any]) -> str:
    settings = get_settings()
    return jwt.encode(payload, get_secret_key(), algorithm=settings.JWT_ALGORITHM)


def create_access_token(user_id: str, email: str, role: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    return _encode({
        "sub": user_id,
        "email": email,
        "role": role,
        "type": ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    })


def create_refresh_token(user_id: str) -> tuple[str, str, datetime]:
    """(token, jti, expires_at) dondurur; jti DB'ye yazilir."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    jti = uuid.uuid4().hex
    token = _encode({
        "sub": user_id,
        "jti": jti,
        "type": REFRESH_TOKEN_TYPE,
        "iat": now,
        "exp": expires,
    })
    return token, jti, expires


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token_expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("token_invalid") from exc
    if expected_type and payload.get("type") != expected_type:
        raise TokenError("token_wrong_type")
    return payload


# ── CSRF (double-submit cookie) ───────────────────────────────────────────

def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_matches(cookie_value: str | None, header_value: str | None) -> bool:
    if not cookie_value or not header_value:
        return False
    return hmac.compare_digest(cookie_value, header_value)
