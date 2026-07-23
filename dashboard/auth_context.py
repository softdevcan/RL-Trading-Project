"""Dash tarafinda aktif kullaniciyi cozme yardimcilari.

Dash callback'leri Starlette'in WSGIMiddleware'i araciligiyla bir is
parcaciginda calisir. AuthGateMiddleware istegi zaten dogrulamis ve
kullaniciyi ContextVar'a yazmistir; anyio bu baglami is parcacigina
kopyaladigi icin `workspace.get_current_user()` dogrudan calisir.

Kopyalanmadigi durumlar icin (surum farki, dogrudan Flask cagrisi) yedek
yol: Flask isteginin oturum cerezini cozmek.
"""

from __future__ import annotations

import logging

from app.auth.models import Role
from app.auth.workspace import get_current_user as _ctx_user
from app.core.config import get_settings

log = logging.getLogger(__name__)


def _from_cookie() -> dict | None:
    try:
        from flask import has_request_context, request

        if not has_request_context():
            return None
        token = request.cookies.get(get_settings().SESSION_COOKIE_NAME)
        if not token:
            return None
        from app.auth import security

        payload = security.decode_token(token, expected_type=security.ACCESS_TOKEN_TYPE)
        return {
            "id": payload.get("sub"),
            "email": payload.get("email", ""),
            "role": payload.get("role", Role.USER),
            "full_name": "",
        }
    except Exception:
        return None


def current_user() -> dict | None:
    """{"id", "email", "role", ...} veya None (auth kapali / baglam yok)."""
    return _ctx_user() or _from_cookie()


def current_role() -> str:
    user = current_user()
    if user:
        return user.get("role", Role.USER)
    # Auth kapaliyken eski davranis: her sey serbest.
    return Role.ADMIN if not get_settings().AUTH_ENABLED else Role.VIEWER


def is_admin() -> bool:
    return current_role() == Role.ADMIN


def can_write() -> bool:
    """viewer disindaki roller yazma islemi yapabilir (egitim, tahmin, karar)."""
    return current_role() in (Role.ADMIN, Role.USER)


def display_name() -> str:
    user = current_user() or {}
    return user.get("full_name") or user.get("email") or "Misafir"
