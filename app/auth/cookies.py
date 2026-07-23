"""Oturum cerezlerinin tek noktadan yonetimi.

Dash WSGI altinda calistigi ve tarayici callback'leri Authorization header
tasiyamadigi icin oturum cerez tabanlidir:

  rlt_session : access JWT  — HttpOnly (JS erisemez), kisa omurlu
  rlt_refresh : refresh JWT — HttpOnly, DB'de jti kaydi ile iptal edilebilir
  rlt_csrf    : CSRF token  — HttpOnly DEGIL; istemci header'a kopyalar
"""

from __future__ import annotations

from fastapi import Response

from app.auth import security
from app.core.config import get_settings

# Refresh cerezi tum yollara gonderilir: sessiz yenileme /dash/* isteklerinde
# de calismak zorunda (Dash sayfa gezinmesi /auth'a ugramaz). HttpOnly +
# SameSite ile korunur.
REFRESH_COOKIE_PATH = "/"


def _base_kwargs() -> dict:
    s = get_settings()
    kwargs = {
        "httponly": True,
        "secure": s.COOKIE_SECURE,
        "samesite": s.COOKIE_SAMESITE,
    }
    if s.COOKIE_DOMAIN:
        kwargs["domain"] = s.COOKIE_DOMAIN
    return kwargs


def set_auth_cookies(response: Response, access_token: str, refresh_token: str | None = None) -> str:
    """Cerezleri yaz ve CSRF token'ini dondur."""
    s = get_settings()
    response.set_cookie(
        s.SESSION_COOKIE_NAME,
        access_token,
        max_age=s.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        **_base_kwargs(),
    )
    if refresh_token is not None:
        response.set_cookie(
            s.REFRESH_COOKIE_NAME,
            refresh_token,
            max_age=s.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
            # Dar kapsam: refresh token yalnizca /auth/refresh ve /auth/logout'a gider.
            path=REFRESH_COOKIE_PATH,
            **_base_kwargs(),
        )

    csrf = security.new_csrf_token()
    csrf_kwargs = _base_kwargs()
    csrf_kwargs["httponly"] = False  # istemci okuyup X-CSRF-Token header'ina koyar
    response.set_cookie(
        s.CSRF_COOKIE_NAME,
        csrf,
        max_age=s.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/",
        **csrf_kwargs,
    )
    return csrf


def clear_auth_cookies(response: Response) -> None:
    s = get_settings()
    domain = s.COOKIE_DOMAIN or None
    response.delete_cookie(s.SESSION_COOKIE_NAME, path="/", domain=domain)
    response.delete_cookie(s.REFRESH_COOKIE_NAME, path=REFRESH_COOKIE_PATH, domain=domain)
    response.delete_cookie(s.CSRF_COOKIE_NAME, path="/", domain=domain)
