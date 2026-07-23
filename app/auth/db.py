"""Auth veritabani baglantisi.

SQLite tek dosya (varsayilan): Docker'da ./data volume'u zaten bind-mount
edildigi icin yeniden kurulumda kullanicilar kaybolmaz. Postgres'e gecis
icin sadece AUTH_DB_URL degistirilir — model/sorgu kodu ayni kalir.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.auth.models import Base

log = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_sqlite_dir(url: str) -> None:
    """SQLite dosyasinin klasoru yoksa olustur (ilk calistirmada gerekli)."""
    if not url.startswith("sqlite"):
        return
    path = url.split("///", 1)[-1]
    if path and path != ":memory:":
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().AUTH_DB_URL
        _ensure_sqlite_dir(url)
        is_sqlite = url.startswith("sqlite")
        _engine = create_engine(
            url,
            # FastAPI + Dash callback'leri farkli thread'lerde calisir.
            connect_args={"check_same_thread": False} if is_sqlite else {},
            pool_pre_ping=True,
            future=True,
        )
        if is_sqlite:
            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - baglanti hook'u
                cur = dbapi_conn.cursor()
                # WAL: okuma/yazma es zamanliligi (tek uvicorn worker + Dash thread'leri)
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA busy_timeout=5000")
                cur.close()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Hata halinde rollback, her durumda close."""
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Iterator[Session]:
    """FastAPI bagimliligi."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Tablolari olustur (idempotent). Uygulama acilisinda cagrilir."""
    Base.metadata.create_all(bind=get_engine())
    log.info("Auth DB hazir: %s", get_settings().AUTH_DB_URL)


def reset_engine() -> None:
    """Testlerde AUTH_DB_URL degistikten sonra engine'i tazelemek icin."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
