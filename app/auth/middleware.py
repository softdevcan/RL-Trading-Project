"""Oturum kapisi: /dash, /api ve /docs icin tek noktadan koruma.

Neden middleware?
-----------------
Dash WSGI olarak mount edildigi icin FastAPI bagimliliklari (Depends) Dash
isteklerine uygulanamaz. Middleware ASGI seviyesinde durdugundan hem
FastAPI uc noktalarini hem Dash sayfalarini/callback'lerini ayni kuralla
korur.

Yaptiklari
----------
1. Acik (public) yollar disinda gecerli access token arar.
2. Access token suresi dolmus ama refresh cerezi gecerliyse SESSIZ yeniler
   (kullanici formu doldururken oturumdan dusmesin).
3. Kimlik yoksa: tarayici gezinmesi -> /login'e yonlendirir; API -> 401 JSON.
4. must_change_password ise /change-password disina cikisi engeller.
5. /api/* uzerindeki yazma isteklerinde CSRF (double-submit) dogrular.
6. Kullaniciyi request.state.user'a ve calisma alani ContextVar'ina yazar.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from app.auth import security, service
from app.auth.cookies import clear_auth_cookies, set_auth_cookies
from app.auth.db import session_scope
from app.auth.models import Role
from app.auth.workspace import reset_current_user, set_current_user
from app.core.config import get_settings

log = logging.getLogger(__name__)

# Kimlik gerektirmeyen yollar
PUBLIC_EXACT = {
    "/login",
    "/logout",
    "/auth/login",
    "/auth/refresh",
    "/health",
    "/favicon.ico",
    "/.well-known/appspecific/com.chrome.devtools.json",
}
PUBLIC_PREFIXES = ("/static/",)

# Parola degistirmeye zorlanan kullanicinin erisebilecegi yollar
PASSWORD_CHANGE_ALLOWED = {"/change-password", "/auth/change-password", "/auth/me", "/auth/logout", "/logout"}

# Yalnizca admin: API semasi is mantigini ve uc noktalari ifsa eder
ADMIN_ONLY_PREFIXES = ("/docs", "/redoc", "/openapi.json")

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _is_public(path: str) -> bool:
    return path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIXES)


def _wants_html(request: Request) -> bool:
    """Tarayici gezinmesi mi, yoksa fetch/XHR mi?"""
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return False
    accept = request.headers.get("accept", "")
    return "text/html" in accept and request.method in SAFE_METHODS


def _login_redirect(request: Request) -> Response:
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"
    response = RedirectResponse(f"/login?next={quote(target)}", status_code=302)
    # Bozuk/expired cerezleri temizle ki dongu olusmasin.
    clear_auth_cookies(response)
    return response


def _deny(request: Request, message: str, status_code: int = 401) -> Response:
    if _wants_html(request):
        return _login_redirect(request)
    return JSONResponse({"detail": message}, status_code=status_code)


class AuthGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.AUTH_ENABLED:
            return await call_next(request)

        path = request.url.path
        if _is_public(path):
            return await call_next(request)

        token = request.cookies.get(settings.SESSION_COOKIE_NAME)
        header = request.headers.get("Authorization", "")
        bearer = header[7:].strip() if header.lower().startswith("bearer ") else None
        used_bearer = bool(bearer)
        token = bearer or token

        payload = None
        refreshed: tuple[str, str] | None = None

        if token:
            try:
                payload = security.decode_token(token, expected_type=security.ACCESS_TOKEN_TYPE)
            except security.TokenError as exc:
                if str(exc) != "token_expired" or used_bearer:
                    return _deny(request, "Gecersiz oturum")
                payload = None  # asagida sessiz yenileme denenir

        # ── Sessiz yenileme ────────────────────────────────────────────────
        if payload is None and not used_bearer:
            refresh_cookie = request.cookies.get(settings.REFRESH_COOKIE_NAME)
            if not refresh_cookie:
                return _deny(request, "Kimlik dogrulama gerekli")
            try:
                with session_scope() as db:
                    access, new_refresh, user = service.rotate_session(
                        db,
                        refresh_cookie,
                        ip=request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                        or (request.client.host if request.client else ""),
                        user_agent=request.headers.get("user-agent", ""),
                    )
                    payload = {"sub": user.id, "email": user.email, "role": user.role}
                    refreshed = (access, new_refresh)
            except (service.AuthError, security.TokenError):
                return _deny(request, "Oturum suresi doldu")

        if payload is None:
            return _deny(request, "Kimlik dogrulama gerekli")

        # ── Kullaniciyi yukle (rol/aktiflik DB'den dogrulanir) ─────────────
        with session_scope() as db:
            user = service.get_user(db, payload.get("sub", ""))
            if user is None or not user.is_active:
                return _deny(request, "Hesap devre disi", 403)
            user_ctx = {
                "id": user.id,
                "email": user.email,
                "role": user.role,
                "full_name": user.full_name,
                "must_change_password": user.must_change_password,
            }

        # ── Yetki ve durum kontrolleri ────────────────────────────────────
        if user_ctx["must_change_password"] and path not in PASSWORD_CHANGE_ALLOWED:
            if _wants_html(request):
                return RedirectResponse("/change-password", status_code=302)
            return JSONResponse({"detail": "Once parolanizi degistirin"}, status_code=403)

        if path.startswith(ADMIN_ONLY_PREFIXES) and user_ctx["role"] != Role.ADMIN:
            return JSONResponse({"detail": "Bu sayfa yalnizca yoneticiler icin"}, status_code=403)

        if (
            path.startswith("/api/")
            and request.method not in SAFE_METHODS
            and not used_bearer
        ):
            csrf_cookie = request.cookies.get(settings.CSRF_COOKIE_NAME)
            csrf_header = request.headers.get("X-CSRF-Token")
            if not security.csrf_matches(csrf_cookie, csrf_header):
                return JSONResponse({"detail": "CSRF dogrulamasi basarisiz"}, status_code=403)

        # ── Istegi calistir ───────────────────────────────────────────────
        request.state.user = user_ctx
        ctx_token = set_current_user(user_ctx)
        try:
            response = await call_next(request)
        finally:
            reset_current_user(ctx_token)

        if refreshed is not None:
            set_auth_cookies(response, refreshed[0], refreshed[1])
        return response
