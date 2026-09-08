"""Kimlik dogrulama tablolari (SQLAlchemy 2.0 tipli mapping)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """Naive UTC. SQLite DateTime sutunlari tz tasimadigi icin DB'ye giren ve
    DB'den cikan tum degerler naive-UTC olarak tutulur; boylece karsilastirma
    aware/naive karisikligina dusmez."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_utc(dt: datetime | None) -> datetime | None:
    """Naive-UTC degeri disariya (API/UI) verirken tz bilgisiyle isaretle."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class Theme:
    """Kullanicinin gorunum tercihi (Faz 8).

    SYSTEM gercek bir ucuncu durumdur: isletim sistemi temasi degistiginde
    pano da degisir. DOM'da damga birakilmaz, karari
    @media (prefers-color-scheme) verir.
    """

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"

    ALL = (LIGHT, DARK, SYSTEM)
    DEFAULT = SYSTEM


class Role:
    """Rol sabitleri. Yetki matrisi icin bkz. deps.py."""

    ADMIN = "admin"    # her sey + kullanici yonetimi
    USER = "user"      # kendi calisma alaninda egitim/tahmin/karar
    VIEWER = "viewer"  # yalnizca okuma

    ALL = (ADMIN, USER, VIEWER)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    # Normalize edilmis (lower + strip) e-posta; giris anahtari.
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default=Role.USER)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Admin'in actigi hesap ilk giriste sifre degistirmeye zorlanir.
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True)

    # Gorunum tercihi: light | dark | system. Tarayiciya degil hesaba bagli,
    # boylece baska makineden giren ayni temayi bulur (Faz 8, B.1).
    theme: Mapped[str] = mapped_column(String(8), default=Theme.DEFAULT,
                                       server_default=Theme.DEFAULT)

    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    created_by: Mapped[str] = mapped_column(String(255), default="")

    sessions: Mapped[list["SessionToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """API/dashboard'a donen guvenli gosterim — hash asla disari cikmaz."""
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "must_change_password": self.must_change_password,
            "theme": self.theme or Theme.DEFAULT,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }


class SessionToken(Base):
    """Refresh token kaydi. Access token stateless (JWT); iptal edilebilirlik
    bu tabloda tutulur — logout ve 'tum oturumlari kapat' bunu siler."""

    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # JWT 'jti' claim'i — token'in kendisi DB'de saklanmaz.
    jti: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    issued_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    ip: Mapped[str] = mapped_column(String(64), default="")
    user_agent: Mapped[str] = mapped_column(String(255), default="")

    user: Mapped[User] = relationship(back_populates="sessions")

    @property
    def is_valid(self) -> bool:
        if self.revoked_at is not None:
            return False
        return self.expires_at is not None and self.expires_at > utcnow()


class AuditLog(Base):
    """Guvenlik ve islem izi. Basarisiz girisler dahil."""

    __tablename__ = "auth_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    email: Mapped[str] = mapped_column(String(255), default="")
    action: Mapped[str] = mapped_column(String(64))     # login, logout, user.create...
    target: Mapped[str] = mapped_column(String(255), default="")
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    ip: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[str] = mapped_column(Text, default="")


Index("ix_audit_action_ts", AuditLog.action, AuditLog.ts)
