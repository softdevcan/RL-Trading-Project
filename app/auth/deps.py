"""FastAPI bagimliliklari: aktif kullanici cozumleme ve rol kontrolu.

Yetki matrisi
-------------
viewer : yalnizca okuma (GET). Egitim/tahmin/karar/veri yazma yok.
user   : kendi calisma alaninda tam yetki (egitim, tahmin, gunluk karar).
admin  : + kullanici yonetimi, ortak veri guncelleme, audit log.

Token iki yerden okunur:
  1. Oturum cerezi (tarayici / Dash)  2. Authorization: Bearer (script, CI)
"""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth import security
from app.auth.db import get_db
from app.auth.models import Role, User
from app.auth.service import get_user
from app.core.config import get_settings


def extract_token(request: Request) -> str | None:
    """Access token: once Authorization header, sonra oturum cerezi."""
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return request.cookies.get(get_settings().SESSION_COOKIE_NAME)


def _unauthorized(detail: str = "Kimlik dogrulama gerekli") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_optional_user(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> User | None:
    """Kullaniciyi cozer ama yoksa hata vermez (acik uc noktalar icin)."""
    if not get_settings().AUTH_ENABLED:
        return None
    token = extract_token(request)
    if not token:
        return None
    try:
        payload = security.decode_token(token, expected_type=security.ACCESS_TOKEN_TYPE)
    except security.TokenError:
        return None
    user = get_user(db, payload.get("sub", ""))
    if user is None or not user.is_active:
        return None
    return user


def get_current_user(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> User:
    """Zorunlu kimlik. AUTH_ENABLED=False iken sanal admin dondurur ki
    mevcut kod/testler auth kapaliyken calismaya devam etsin."""
    settings = get_settings()
    if not settings.AUTH_ENABLED:
        return User(id="local", email="local@localhost", full_name="Local",
                    password_hash="", role=Role.ADMIN, is_active=True,
                    must_change_password=False)

    token = extract_token(request)
    if not token:
        raise _unauthorized()
    try:
        payload = security.decode_token(token, expected_type=security.ACCESS_TOKEN_TYPE)
    except security.TokenError as exc:
        raise _unauthorized("Oturum suresi doldu" if str(exc) == "token_expired"
                            else "Gecersiz oturum") from exc

    user = get_user(db, payload.get("sub", ""))
    if user is None:
        raise _unauthorized("Kullanici bulunamadi")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Hesap devre disi")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(*roles: str) -> Callable[..., User]:
    """Belirtilen rollerden birini zorunlu kilar. admin her zaman gecer."""
    allowed = set(roles) | {Role.ADMIN}

    def _dep(user: CurrentUser) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Bu islem icin yetkiniz yok (gerekli rol: {', '.join(sorted(roles))})",
            )
        return user

    return _dep


# Sik kullanilan hazir bagimliliklar
RequireAdmin = Annotated[User, Depends(require_role(Role.ADMIN))]
# viewer disinda herkes: yazma islemleri (egitim, tahmin, karar uygulama)
RequireWriter = Annotated[User, Depends(require_role(Role.USER))]


def require_password_ok(user: CurrentUser) -> User:
    """Ilk giriste parola degistirmeden veri ucu kullanilmasin."""
    if user.must_change_password:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Once parolanizi degistirin")
    return user
