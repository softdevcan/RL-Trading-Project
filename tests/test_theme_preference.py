"""Tema tercihi ucdan uca testi (pytest degil, dogrudan `python` ile).

    python tests/test_theme_preference.py

Ne dogrular (Faz 8, E.3):
  1. Eski semali bir DB'ye `theme` sutunu ekleniyor mu, mevcut satirlar
     "system" aliyor mu, ikinci calistirma sorun cikariyor mu.
     (Projede alembic yok; create_all VAR OLAN tabloyu degistirmez. Bu adim
      atlanirsa calisan kurulum `no such column: users.theme` ile acilmaz.)
  2. Giriste tema cerezleri yaziliyor mu; tercih hesaptan geliyor mu.
  3. PATCH /auth/preferences uc gecerli degeri kabul, digerini ret ediyor mu.
  4. CSRF olmadan yazma reddediliyor mu.
  5. Viewer kendi tercihini degistirebiliyor mu (kisisel ayar, yetki degil).
  6. Kimse baskasinin tercihine yazamiyor (hedef hep oturumdaki kullanici).
  7. "system" seciliyken sunucu istemcinin cozdugu degeri kullaniyor mu.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_theme_test_")
_DB = os.path.join(_TMP, "auth.db")

os.environ["AUTH_DB_URL"] = f"sqlite:///{_DB.replace(os.sep, '/')}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["AUTH_ENABLED"] = "True"
os.environ["WORKSPACES_DIR"] = os.path.join(_TMP, "workspaces")
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = ""
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.auth import security  # noqa: E402
from app.auth.db import init_db, reset_engine, session_scope  # noqa: E402
from app.auth.models import Role, Theme, User  # noqa: E402
from app.auth.service import create_user  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [OK]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def _columns(path: str) -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {row[1] for row in con.execute("PRAGMA table_info(users)")}
    finally:
        con.close()


def _make_legacy_db(path: str) -> None:
    """Faz 8 oncesi sema: users tablosunda theme sutunu YOK."""
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE users (
            id VARCHAR(32) PRIMARY KEY,
            email VARCHAR(255) UNIQUE,
            full_name VARCHAR(120),
            password_hash VARCHAR(255),
            role VARCHAR(16),
            is_active BOOLEAN,
            must_change_password BOOLEAN,
            failed_attempts INTEGER,
            locked_until DATETIME,
            last_login_at DATETIME,
            created_at DATETIME,
            updated_at DATETIME,
            created_by VARCHAR(255)
        )"""
    )
    con.execute(
        "INSERT INTO users (id, email, role, is_active, must_change_password, failed_attempts)"
        " VALUES ('legacy01', 'eski@test.local', 'user', 1, 0, 0)"
    )
    con.commit()
    con.close()


def login(client: TestClient, email: str, password: str) -> bool:
    r = client.post("/auth/login", json={"email": email, "password": password})
    return r.status_code == 200


def patch_theme(client: TestClient, theme: str, resolved: str | None = None,
                with_csrf: bool = True):
    headers = {"X-Requested-With": "XMLHttpRequest"}
    if with_csrf:
        headers["X-CSRF-Token"] = client.cookies.get("rlt_csrf") or ""
    if resolved:
        headers["X-Theme-Resolved"] = resolved
    return client.patch("/auth/preferences", json={"theme": theme}, headers=headers)


def main() -> int:
    print("1) Eski semali DB'ye additive gec")
    _make_legacy_db(_DB)
    check("Gec oncesi theme sutunu yok", "theme" not in _columns(_DB))

    reset_engine()
    security.reset_secret_cache()
    init_db()
    check("Gec sonrasi theme sutunu var", "theme" in _columns(_DB))

    con = sqlite3.connect(_DB)
    legacy_theme = con.execute(
        "SELECT theme FROM users WHERE id = 'legacy01'"
    ).fetchone()[0]
    con.close()
    check("Mevcut satir varsayilani aldi", legacy_theme == Theme.SYSTEM,
          f"(got {legacy_theme!r})")

    init_db()  # idempotent olmali
    check("Ikinci init_db sorunsuz", "theme" in _columns(_DB))

    with session_scope() as db:
        user = db.query(User).filter(User.id == "legacy01").first()
        check("ORM eski satiri okuyabiliyor", user is not None and user.theme == Theme.SYSTEM)

    print("\n2) Kullanicilar")
    with session_scope() as db:
        create_user(db, email="user@test.local", password="UserPass123",
                    role=Role.USER, created_by="test", must_change_password=False)
        create_user(db, email="viewer@test.local", password="ViewerPass123",
                    role=Role.VIEWER, created_by="test", must_change_password=False)
    check("Kullanicilar olusturuldu", True)

    with session_scope() as db:
        fresh = db.query(User).filter(User.email == "user@test.local").first()
        check("Yeni kullanicinin varsayilani 'system'", fresh.theme == Theme.SYSTEM,
              f"(got {fresh.theme!r})")

    from app.main import app  # noqa: E402

    print("\n3) Giriste tema cerezleri")
    with TestClient(app) as c:
        check("Giris basarili", login(c, "user@test.local", "UserPass123"))
        check("rlt_theme cerezi yazildi", c.cookies.get("rlt_theme") == Theme.SYSTEM,
              f"(got {c.cookies.get('rlt_theme')!r})")
        # "system" iken cozumu tarayici yapar; sunucu resolved yazmaz
        check("system iken rlt_theme_r yazilmadi", not c.cookies.get("rlt_theme_r"),
              f"(got {c.cookies.get('rlt_theme_r')!r})")

        print("\n4) Tercih guncelleme")
        r = patch_theme(c, "dark")
        check("dark kabul edildi", r.status_code == 200, f"(got {r.status_code})")
        check("Cevap yeni temayi donuyor", r.json().get("theme") == "dark")
        check("rlt_theme cerezi guncellendi", c.cookies.get("rlt_theme") == "dark",
              f"(got {c.cookies.get('rlt_theme')!r})")
        check("light/dark iken resolved da yazildi", c.cookies.get("rlt_theme_r") == "dark",
              f"(got {c.cookies.get('rlt_theme_r')!r})")

        r = patch_theme(c, "light")
        check("light kabul edildi", r.status_code == 200, f"(got {r.status_code})")

        r = patch_theme(c, "system", resolved="dark")
        check("system kabul edildi", r.status_code == 200, f"(got {r.status_code})")
        check("system + istemci cozumu resolved'a yazildi",
              c.cookies.get("rlt_theme_r") == "dark",
              f"(got {c.cookies.get('rlt_theme_r')!r})")

        r = patch_theme(c, "midnight")
        check("Gecersiz deger reddedildi", r.status_code in (400, 422),
              f"(got {r.status_code})")

        with session_scope() as db:
            saved = db.query(User).filter(User.email == "user@test.local").first()
            check("Tercih DB'ye yazildi", saved.theme == "system", f"(got {saved.theme!r})")

        print("\n5) CSRF")
        r = patch_theme(c, "dark", with_csrf=False)
        check("CSRF'siz istek reddedildi", r.status_code == 403, f"(got {r.status_code})")
        with session_scope() as db:
            saved = db.query(User).filter(User.email == "user@test.local").first()
            check("Reddedilen istek DB'yi degistirmedi", saved.theme == "system",
                  f"(got {saved.theme!r})")

    print("\n6) Viewer kendi tercihini degistirebiliyor")
    with TestClient(app) as c:
        check("Viewer giris yapti", login(c, "viewer@test.local", "ViewerPass123"))
        r = patch_theme(c, "light")
        check("Viewer tercihini degistirdi", r.status_code == 200, f"(got {r.status_code})")
        with session_scope() as db:
            v = db.query(User).filter(User.email == "viewer@test.local").first()
            u = db.query(User).filter(User.email == "user@test.local").first()
            check("Viewer'in tercihi kaydedildi", v.theme == "light", f"(got {v.theme!r})")
            check("Baska kullanicinin tercihi degismedi", u.theme == "system",
                  f"(got {u.theme!r})")

    print("\n7) Oturumsuz erisim")
    with TestClient(app) as c:
        r = c.patch("/auth/preferences", json={"theme": "dark"},
                    headers={"X-Requested-With": "XMLHttpRequest"})
        check("Oturumsuz istek reddedildi", r.status_code in (401, 403),
              f"(got {r.status_code})")

    print("\n8) Sunucu tarafi tema cozumu (Plotly paleti)")
    from dashboard.theme import PLOT, current_theme, plot_palette

    import flask

    probe = flask.Flask(__name__)
    with probe.test_request_context("/", headers={"Cookie": "rlt_theme_r=dark"}):
        check("resolved=dark -> koyu palet", current_theme() == "dark",
              f"(got {current_theme()!r})")
        check("Palet koyu degerleri veriyor",
              plot_palette()["bg"] == PLOT["dark"]["bg"])
    with probe.test_request_context("/", headers={"Cookie": "rlt_theme_r=light"}):
        check("resolved=light -> aydinlik palet", current_theme() == "light")
    with probe.test_request_context("/", headers={"Cookie": "rlt_theme=dark"}):
        check("resolved yokken tercih kullaniliyor", current_theme() == "dark")
    with probe.test_request_context("/"):
        check("Cerez yokken aydinlik tabana duser", current_theme() == "light")

    print("\n" + "=" * 60)
    print(f"  Gecen: {len(PASSED)}   Kalan: {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"   - {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    raise SystemExit(code)
