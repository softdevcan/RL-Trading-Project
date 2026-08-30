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


# Faz 8 sonrasi eklenen sutunlar. create_all() VAR OLAN tabloyu degistirmez;
# alembic de yok. Bu yuzden eksik sutunlar acilista elle eklenir.
#
# Kural: yalnizca ADDITIVE ve NULL-guvenli degisiklikler buraya girer
# (yeni sutun + DEFAULT). Sutun silme/tip degistirme gercek bir gec araci
# ister — buraya yazma.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # (tablo, sutun, ALTER govdesi)
    ("users", "theme", "ALTER TABLE users ADD COLUMN theme VARCHAR(8) NOT NULL DEFAULT 'system'"),
)


def _existing_columns(conn, table: str) -> set[str]:
    """Tablodaki sutun adlari. Tablo yoksa bos kume."""
    from sqlalchemy import inspect

    inspector = inspect(conn)
    if table not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table)}


def _apply_additive_columns() -> None:
    """Eksik sutunlari ekle. Idempotent: var olanlara dokunmaz."""
    from sqlalchemy import text

    engine = get_engine()
    with engine.begin() as conn:
        for table, column, statement in _ADDITIVE_COLUMNS:
            columns = _existing_columns(conn, table)
            if not columns or column in columns:
                continue
            conn.execute(text(statement))
            log.info("Auth DB gecisi: %s.%s eklendi", table, column)


def init_db() -> None:
    """Tablolari olustur ve eksik sutunlari tamamla (idempotent).

    Uygulama acilisinda cagrilir. Iki adim ayri: create_all yeni kurulumu,
    _apply_additive_columns mevcut kurulumu halleder.
    """
    Base.metadata.create_all(bind=get_engine())
    _apply_additive_columns()
    log.info("Auth DB hazir: %s", get_settings().AUTH_DB_URL)


def reset_engine() -> None:
    """Testlerde AUTH_DB_URL degistikten sonra engine'i tazelemek icin."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
