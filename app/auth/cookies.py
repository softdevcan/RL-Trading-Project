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


def set_theme_cookies(response: Response, theme: str, resolved: str | None = None) -> None:
    """Gorunum tercihini cereze yaz (Faz 8, B.1).

    Oturum cerezlerinden farkli olarak **HttpOnly degil**: <head>'deki FOUC
    engelleyici script ve tema anahtari bunlari JS'ten okur.

    `resolved` yalnizca tercih light/dark iken kesin bilinir; "system" secildi
    ise karari tarayici verir ve cerezi kendisi tazeler — burada silinir ki
    sunucu eski/yanlis bir degeri okumasin.
    """
    s = get_settings()
    kwargs = {
        "httponly": False,
        "secure": s.COOKIE_SECURE,
        "samesite": s.COOKIE_SAMESITE,
        "max_age": 365 * 86400,
        "path": "/",
    }
    if s.COOKIE_DOMAIN:
        kwargs["domain"] = s.COOKIE_DOMAIN

    response.set_cookie(s.THEME_COOKIE_NAME, theme, **kwargs)

    if resolved is None:
        resolved = theme if theme in ("light", "dark") else None

    if resolved in ("light", "dark"):
        response.set_cookie(s.THEME_RESOLVED_COOKIE_NAME, resolved, **kwargs)
    else:
        response.delete_cookie(
            s.THEME_RESOLVED_COOKIE_NAME, path="/", domain=s.COOKIE_DOMAIN or None
        )


def clear_auth_cookies(response: Response) -> None:
    s = get_settings()
    domain = s.COOKIE_DOMAIN or None
    response.delete_cookie(s.SESSION_COOKIE_NAME, path="/", domain=domain)
    response.delete_cookie(s.REFRESH_COOKIE_NAME, path=REFRESH_COOKIE_PATH, domain=domain)
    response.delete_cookie(s.CSRF_COOKIE_NAME, path="/", domain=domain)
