"""Kimlik dogrulama uc noktalari ve HTML giris sayfalari.

Sayfalar (tarayici):   GET /login, GET /change-password, GET /logout
API (JSON):            POST /auth/login|logout|refresh|change-password, GET /auth/me

Login formu ayni POST /auth/login ucunu kullanir; icerik tipine gore
(form-encoded -> 303 redirect, JSON -> JSON govde) cevap verilir.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Annotated
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import security, service
from app.auth.cookies import clear_auth_cookies, set_auth_cookies, set_theme_cookies
from app.auth.db import get_db
from app.auth.models import Theme
from app.auth.deps import CurrentUser, get_optional_user
from app.auth.service import AuthError
from app.core.config import get_settings

log = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# ── IP bazli giris hiz siniri (hesap kilidine ek katman) ──────────────────
_LOGIN_WINDOW_SEC = 300
_LOGIN_MAX_PER_IP = 15
_login_hits: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    """Ters vekil arkasinda gercek IP. X-Forwarded-For yalnizca guvenilen
    proxy tarafindan set edilmeli (nginx/Caddy yapilandirmasi)."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _rate_limited(ip: str) -> bool:
    now = time.time()
    hits = _login_hits[ip]
    while hits and now - hits[0] > _LOGIN_WINDOW_SEC:
        hits.popleft()
    if len(hits) >= _LOGIN_MAX_PER_IP:
        return True
    hits.append(now)
    return False


def safe_next(target: str | None) -> str:
    """Acik yonlendirme (open redirect) korumasi: yalnizca site-ici yollar."""
    if not target:
        return "/dash/"
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc or not target.startswith("/") or target.startswith("//"):
        return "/dash/"
    return target


# ── Semalar ───────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=1)


class PreferencesRequest(BaseModel):
    """Gorunum tercihi. Uc gecerli deger disina cikilamaz."""

    theme: str = Field(pattern="^(light|dark|system)$")


# ── HTML sayfalari ────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    next: str = "/dash/",
    error: str = "",
):
    user = get_optional_user(request, db)
    if user is not None:
        dest = "/change-password" if user.must_change_password else safe_next(next)
        return RedirectResponse(dest, status_code=status.HTTP_303_SEE_OTHER)

    setup_needed = service.user_count(db) == 0
    return templates.TemplateResponse(
        request,
        "login.html",
        {"next": safe_next(next), "error": error, "setup_needed": setup_needed},
    )


@router.get("/change-password", response_class=HTMLResponse, include_in_schema=False)
async def change_password_page(request: Request, user: CurrentUser, error: str = ""):
    return templates.TemplateResponse(
        request,
        "change_password.html",
        {"error": error, "forced": user.must_change_password, "email": user.email},
    )


@router.get("/logout", include_in_schema=False)
async def logout_page(request: Request, db: Annotated[Session, Depends(get_db)]):
    """Tarayici icin GET cikis: cerezleri sil, /login'e don."""
    response = RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    token = request.cookies.get(get_settings().REFRESH_COOKIE_NAME)
    if token:
        service.revoke_session(db, token)
        db.commit()
    clear_auth_cookies(response)
    return response


# ── JSON / form uc noktalari ──────────────────────────────────────────────

@router.post("/auth/login")
async def login(request: Request, db: Annotated[Session, Depends(get_db)]):
    ip = client_ip(request)
    content_type = request.headers.get("content-type", "")
    is_form = "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type

    if is_form:
        form = await request.form()
        email = str(form.get("email", ""))
        password = str(form.get("password", ""))
        next_url = safe_next(str(form.get("next", "/dash/")))
    else:
        body = LoginRequest(**(await request.json()))
        email, password = body.email, body.password
        next_url = "/dash/"

    def fail(message: str, code: int):
        if is_form:
            from urllib.parse import quote
            return RedirectResponse(
                f"/login?next={quote(next_url)}&error={quote(message)}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return JSONResponse({"detail": message}, status_code=code)

    if _rate_limited(ip):
        service.audit(db, "login", email=email, success=False, ip=ip, detail={"reason": "rate_limited"})
        db.commit()
        return fail("Cok fazla deneme. Birkac dakika sonra tekrar deneyin.", 429)

    try:
        user = service.authenticate(db, email, password, ip=ip)
    except AuthError as exc:
        db.commit()  # audit kaydi korunsun
        return fail(exc.message, exc.status_code)

    access, refresh = service.create_session(
        db, user, ip=ip, user_agent=request.headers.get("user-agent", "")
    )
    db.commit()

    destination = "/change-password" if user.must_change_password else next_url
    if is_form:
        response: Response = RedirectResponse(destination, status_code=status.HTTP_303_SEE_OTHER)
    else:
        response = JSONResponse({
            "user": user.to_dict(),
            "access_token": access,
            "token_type": "bearer",
            "next": destination,
        })
    csrf = set_auth_cookies(response, access, refresh)
    # Tercih hesapta durur; cerez yalnizca okuma onbellegidir. Cerezi silinmis
    # (ya da baska makinedeki) tarayici temayi burada geri alir.
    set_theme_cookies(response, user.theme or Theme.DEFAULT)
    if not is_form:
        response.headers["X-CSRF-Token"] = csrf
    return response


@router.post("/auth/logout")
async def logout(request: Request, db: Annotated[Session, Depends(get_db)]):
    token = request.cookies.get(get_settings().REFRESH_COOKIE_NAME)
    if token:
        service.revoke_session(db, token)
        db.commit()
    response = JSONResponse({"status": "ok"})
    clear_auth_cookies(response)
    return response


@router.post("/auth/refresh")
async def refresh_tokens(request: Request, db: Annotated[Session, Depends(get_db)]):
    token = request.cookies.get(get_settings().REFRESH_COOKIE_NAME)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token yok")
    try:
        access, new_refresh, user = service.rotate_session(
            db, token, ip=client_ip(request), user_agent=request.headers.get("user-agent", "")
        )
    except (AuthError, security.TokenError) as exc:
        db.commit()
        message = getattr(exc, "message", "Oturum gecersiz")
        response = JSONResponse({"detail": message}, status_code=401)
        clear_auth_cookies(response)
        return response
    db.commit()
    response = JSONResponse({"user": user.to_dict(), "access_token": access, "token_type": "bearer"})
    response.headers["X-CSRF-Token"] = set_auth_cookies(response, access, new_refresh)
    return response


@router.get("/auth/me")
async def me(user: CurrentUser) -> dict:
    from app.auth.workspace import workspace_root
    data = user.to_dict()
    data["workspace"] = workspace_root(user.id)
    return data


@router.patch("/auth/preferences")
async def update_preferences(
    request: Request,
    body: PreferencesRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Kullanicinin kendi gorunum tercihini gunceller (Faz 8, B.1).

    Her rol kendi tercihini degistirebilir — viewer dahil; tema okuma
    yetkisiyle ilgisi olmayan kisisel bir ayardir. Kimse baskasinin
    tercihine yazamaz: hedef her zaman oturumdaki kullanicidir.

    `resolved` istege bagli: tercih "system" iken tarayicinin o an cozdugu
    deger. Sunucu Plotly figurunu bununla dogru palette uretir.
    """
    # AuthGateMiddleware CSRF'i yalnizca /api/* yazmalarinda dogruluyor; bu uc
    # /auth/* altinda ve durum degistiriyor, o yuzden kontrol burada yapilir.
    # (SameSite=Lax zaten capraz siteden PATCH'i keser — bu ikinci katman.)
    if not security.csrf_matches(
        request.cookies.get(get_settings().CSRF_COOKIE_NAME),
        request.headers.get("X-CSRF-Token"),
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF dogrulamasi basarisiz")

    db_user = service.get_user(db, user.id)
    if db_user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Kullanici bulunamadi")

    db_user.theme = body.theme
    db.commit()

    resolved = request.headers.get("X-Theme-Resolved", "")
    response = JSONResponse({"status": "ok", "theme": body.theme})
    set_theme_cookies(response, body.theme, resolved if resolved in ("light", "dark") else None)
    return response


@router.post("/auth/change-password")
async def change_password(request: Request, user: CurrentUser, db: Annotated[Session, Depends(get_db)]):
    content_type = request.headers.get("content-type", "")
    is_form = "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type

    if is_form:
        form = await request.form()
        current_password = str(form.get("current_password", ""))
        new_password = str(form.get("new_password", ""))
        confirm = str(form.get("confirm_password", ""))
    else:
        body = ChangePasswordRequest(**(await request.json()))
        current_password, new_password = body.current_password, body.new_password
        confirm = new_password

    def fail(message: str, code: int = 400):
        if is_form:
            from urllib.parse import quote
            return RedirectResponse(f"/change-password?error={quote(message)}",
                                    status_code=status.HTTP_303_SEE_OTHER)
        return JSONResponse({"detail": message}, status_code=code)

    if new_password != confirm:
        return fail("Yeni parolalar eslesmiyor")
    if not security.verify_password(current_password, user.password_hash):
        service.audit(db, "password.change", user=user, success=False, ip=client_ip(request))
        db.commit()
        return fail("Mevcut parola hatali", 403)
    if new_password == current_password:
        return fail("Yeni parola eskisiyle ayni olamaz")

    db_user = service.get_user(db, user.id)
    try:
        service.set_password(db, db_user, new_password)
    except AuthError as exc:
        return fail(exc.message, exc.status_code)

    # Parola degisti: diger tum oturumlar dusurulur, bu oturum yenilenir.
    service.revoke_all_sessions(db, db_user.id)
    service.audit(db, "password.change", user=db_user, success=True, ip=client_ip(request))
    access, refresh = service.create_session(db, db_user, ip=client_ip(request),
                                             user_agent=request.headers.get("user-agent", ""))
    db.commit()

    if is_form:
        response: Response = RedirectResponse("/dash/", status_code=status.HTTP_303_SEE_OTHER)
    else:
        response = JSONResponse({"status": "ok", "access_token": access})
    set_auth_cookies(response, access, refresh)
    return response
