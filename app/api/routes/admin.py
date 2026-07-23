"""Kullanici yonetimi uc noktalari (yalnizca admin).

Kayit akisi: self-signup YOK. Hesaplari admin acar; kullanici ilk giriste
parolasini degistirmek zorundadir.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Literal

import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth import service
from app.auth.db import get_db
from app.auth.deps import RequireAdmin
from app.auth.models import AuditLog, Role, User
from app.auth.service import AuthError
from app.auth.workspace import ensure_workspace, workspace_usage

router = APIRouter(prefix="/admin", tags=["admin"])

RoleName = Literal["admin", "user", "viewer"]

# Kurum ici alan adlari (ornegin sirket.local) da kabul edilsin diye
# EmailStr yerine sade bicim kontrolu — CLI ile ayni kural.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserCreate(BaseModel):
    email: str
    full_name: str = ""
    role: RoleName = "user"
    # Bos birakilirsa gucli bir parola uretilir ve YALNIZCA bu yanitta doner.
    password: str | None = Field(default=None, min_length=1)

    @field_validator("email")
    @classmethod
    def _check_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL_RE.match(value):
            raise ValueError("Gecerli bir e-posta adresi girin")
        return value


class UserUpdate(BaseModel):
    full_name: str | None = None
    role: RoleName | None = None
    is_active: bool | None = None


class PasswordReset(BaseModel):
    password: str | None = None


def _generate_password() -> str:
    """Politikayi (buyuk/kucuk/rakam/uzunluk) garanti eden gecici parola."""
    return f"Rlt{secrets.token_urlsafe(12)}9x"


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")


def _admin_count(db: Session) -> int:
    return len(list(db.execute(
        select(User.id).where(User.role == Role.ADMIN, User.is_active.is_(True))
    ).scalars()))


def _get_or_404(db: Session, user_id: str) -> User:
    user = service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Kullanici bulunamadi")
    return user


@router.get("/users")
async def list_users(admin: RequireAdmin, db: Annotated[Session, Depends(get_db)]) -> dict:
    users = []
    for user in service.list_users(db):
        row = user.to_dict()
        row["workspace"] = workspace_usage(user.id)
        users.append(row)
    return {"users": users, "count": len(users)}


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    admin: RequireAdmin,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    password = payload.password or _generate_password()
    generated = payload.password is None
    try:
        user = service.create_user(
            db,
            email=str(payload.email),
            password=password,
            full_name=payload.full_name,
            role=payload.role,
            created_by=admin.email,
        )
    except AuthError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc

    service.audit(db, "user.create", user=admin, target=user.email,
                  ip=_client_ip(request), detail={"role": user.role})
    db.commit()

    result = {"user": user.to_dict(), "workspace": ensure_workspace(user.id)}
    if generated:
        # Tek gosterim: parola hicbir yerde saklanmaz.
        result["temporary_password"] = password
    return result


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    payload: UserUpdate,
    admin: RequireAdmin,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    user = _get_or_404(db, user_id)

    # Son aktif admin'in yetkisi alinamaz / pasiflestirilemez — kilitlenme onlenir.
    losing_admin = (payload.role is not None and payload.role != Role.ADMIN) or payload.is_active is False
    if user.role == Role.ADMIN and losing_admin and _admin_count(db) <= 1:
        raise HTTPException(status.HTTP_409_CONFLICT, "Sistemdeki son admin devre disi birakilamaz")

    try:
        service.update_user(db, user, full_name=payload.full_name,
                            role=payload.role, is_active=payload.is_active)
    except AuthError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc

    service.audit(db, "user.update", user=admin, target=user.email, ip=_client_ip(request),
                  detail=payload.model_dump(exclude_none=True))
    db.commit()
    return {"user": user.to_dict()}


@router.post("/users/{user_id}/password")
async def reset_password(
    user_id: str,
    payload: PasswordReset,
    admin: RequireAdmin,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    user = _get_or_404(db, user_id)
    password = payload.password or _generate_password()
    try:
        # Admin sifirlamasi: kullanici ilk giriste tekrar degistirmeli.
        service.set_password(db, user, password, must_change=True)
    except AuthError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc

    service.revoke_all_sessions(db, user.id)
    service.audit(db, "user.password_reset", user=admin, target=user.email, ip=_client_ip(request))
    db.commit()

    result: dict = {"status": "ok", "user": user.to_dict()}
    if payload.password is None:
        result["temporary_password"] = password
    return result


@router.post("/users/{user_id}/revoke-sessions")
async def revoke_sessions(
    user_id: str,
    admin: RequireAdmin,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    user = _get_or_404(db, user_id)
    count = service.revoke_all_sessions(db, user.id)
    service.audit(db, "user.revoke_sessions", user=admin, target=user.email,
                  ip=_client_ip(request), detail={"revoked": count})
    db.commit()
    return {"status": "ok", "revoked": count}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin: RequireAdmin,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    user = _get_or_404(db, user_id)
    if user.id == admin.id:
        raise HTTPException(status.HTTP_409_CONFLICT, "Kendi hesabinizi silemezsiniz")
    if user.role == Role.ADMIN and _admin_count(db) <= 1:
        raise HTTPException(status.HTTP_409_CONFLICT, "Sistemdeki son admin silinemez")

    email = user.email
    workspace = workspace_usage(user.id)
    service.delete_user(db, user)
    service.audit(db, "user.delete", user=admin, target=email, ip=_client_ip(request),
                  detail={"workspace_kept": workspace.get("root")})
    db.commit()
    # Calisma alani dosyalari korunur; silme karari operatore birakilir.
    return {"status": "deleted", "email": email, "workspace_kept": workspace.get("root")}


@router.get("/audit")
async def audit_log(
    admin: RequireAdmin,
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(100, ge=1, le=1000),
    action: str | None = None,
) -> dict:
    stmt = select(AuditLog).order_by(desc(AuditLog.ts)).limit(limit)
    if action:
        stmt = select(AuditLog).where(AuditLog.action == action).order_by(desc(AuditLog.ts)).limit(limit)
    rows = list(db.execute(stmt).scalars())
    return {
        "entries": [
            {
                "ts": row.ts.isoformat() if row.ts else None,
                "email": row.email,
                "action": row.action,
                "target": row.target,
                "success": row.success,
                "ip": row.ip,
                "detail": row.detail,
            }
            for row in rows
        ],
        "count": len(rows),
    }
