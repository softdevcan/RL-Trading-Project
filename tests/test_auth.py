"""Auth katmani ucdan uca duman testi (pytest degil, dogrudan `python` ile).

    python tests/test_auth.py

Gecici bir SQLite DB kullanir; proje verisine dokunmaz.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Izole ortam: gercek auth.db ve workspaces/ dizinine dokunma ------------
_TMP = tempfile.mkdtemp(prefix="rlt_auth_test_")
os.environ["AUTH_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'auth.db').replace(os.sep, '/')}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["AUTH_ENABLED"] = "True"
os.environ["WORKSPACES_DIR"] = os.path.join(_TMP, "workspaces")
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = ""
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.auth.db import init_db, reset_engine, session_scope  # noqa: E402
from app.auth.models import Role  # noqa: E402
from app.auth.service import create_user  # noqa: E402
from app.auth import security  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [OK]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


def main() -> int:
    reset_engine()
    security.reset_secret_cache()
    init_db()

    with session_scope() as db:
        create_user(db, email="admin@test.local", password="AdminPass123",
                    role=Role.ADMIN, created_by="test", must_change_password=False)
        create_user(db, email="viewer@test.local", password="ViewerPass123",
                    role=Role.VIEWER, created_by="test", must_change_password=False)

    from app.main import app  # auth middleware yuklu FastAPI uygulamasi

    # TestClient: ASGI uygulamasini senkron olarak surer, cerezleri saklar.
    with TestClient(app, base_url="http://test", follow_redirects=False) as c:

        print("\n1) Korumasiz erisim engelleniyor mu")
        r = c.get("/api/trading/models")
        check("API kimliksiz -> 401", r.status_code == 401, f"(got {r.status_code})")

        r = c.get("/dash/", headers={"accept": "text/html"})
        check("Dash kimliksiz -> /login yonlendirme",
              r.status_code == 302 and "/login" in r.headers.get("location", ""),
              f"(got {r.status_code} {r.headers.get('location')})")

        r = c.get("/login")
        check("/login herkese acik", r.status_code == 200)

        r = c.get("/health")
        check("/health acik (healthcheck)", r.status_code == 200)

        print("\n2) Hatali giris")
        r = c.post("/auth/login", json={"email": "admin@test.local", "password": "yanlis"})
        check("Yanlis parola -> 401", r.status_code == 401, f"(got {r.status_code})")
        check("Cerez verilmedi", c.cookies.get("rlt_session") is None)

        print("\n3) Basarili giris")
        r = c.post("/auth/login", json={"email": "admin@test.local", "password": "AdminPass123"})
        check("Giris -> 200", r.status_code == 200, f"(got {r.status_code}: {r.text[:120]})")
        check("Oturum cerezi set edildi", c.cookies.get("rlt_session") is not None)
        check("CSRF cerezi set edildi", c.cookies.get("rlt_csrf") is not None)
        csrf = c.cookies.get("rlt_csrf")

        r = c.get("/auth/me")
        me = r.json() if r.status_code == 200 else {}
        check("/auth/me admin dondu", me.get("email") == "admin@test.local" and me.get("role") == "admin",
              f"(got {me})")
        check("Parola hash'i disari sizmiyor", "password_hash" not in me)

        r = c.get("/api/trading/models")
        check("Giris sonrasi API erisimi", r.status_code == 200, f"(got {r.status_code})")

        print("\n4) CSRF korumasi")
        r = c.post("/api/trading/data/generate", json={})
        check("CSRF header'siz POST -> 403", r.status_code == 403, f"(got {r.status_code})")

        print("\n5) Yetkilendirme (RBAC)")
        r = c.get("/api/admin/users")
        check("Admin kullanici listesini gorur", r.status_code == 200, f"(got {r.status_code})")

        r = c.post("/api/admin/users",
                   json={"email": "new@test.local", "role": "user"},
                   headers={"X-CSRF-Token": csrf})
        created = r.json() if r.status_code == 201 else {}
        check("Admin kullanici acabiliyor", r.status_code == 201, f"(got {r.status_code}: {r.text[:160]})")
        check("Gecici parola tek seferlik donuyor", bool(created.get("temporary_password")))
        check("Calisma alani olusturuldu",
              bool(created.get("workspace")) and os.path.isdir(created["workspace"]),
              f"(got {created.get('workspace')})")

        c.post("/auth/logout", headers={"X-CSRF-Token": csrf})
        c.cookies.clear()

        r = c.post("/auth/login", json={"email": "viewer@test.local", "password": "ViewerPass123"})
        check("Viewer giris yapabiliyor", r.status_code == 200)
        viewer_csrf = c.cookies.get("rlt_csrf")
        r = c.get("/api/admin/users")
        check("Viewer admin ucuna erisemiyor -> 403", r.status_code == 403, f"(got {r.status_code})")
        r = c.get("/openapi.json")
        check("Viewer /openapi.json goremiyor -> 403", r.status_code == 403, f"(got {r.status_code})")

        print("\n6) Ilk giriste parola degistirme zorunlulugu")
        c.cookies.clear()
        temp_password = created.get("temporary_password", "")
        r = c.post("/auth/login", json={"email": "new@test.local", "password": temp_password})
        body = r.json() if r.status_code == 200 else {}
        check("Yeni kullanici giris yapti", r.status_code == 200, f"(got {r.status_code})")
        check("change-password'e yonlendiriliyor", body.get("next") == "/change-password", f"(got {body.get('next')})")
        r = c.get("/api/trading/models")
        check("Parola degistirmeden API kapali -> 403", r.status_code == 403, f"(got {r.status_code})")

        r = c.post("/auth/change-password",
                   json={"current_password": temp_password, "new_password": "BrandNew123"},
                   headers={"X-CSRF-Token": c.cookies.get("rlt_csrf")})
        check("Parola degistirildi", r.status_code == 200, f"(got {r.status_code}: {r.text[:160]})")
        r = c.get("/api/trading/models")
        check("Degisiklik sonrasi API acildi", r.status_code == 200, f"(got {r.status_code})")

        print("\n7) Zayif parola reddi")
        c.cookies.clear()
        c.post("/auth/login", json={"email": "admin@test.local", "password": "AdminPass123"})
        r = c.post("/api/admin/users",
                   json={"email": "weak@test.local", "password": "12345"},
                   headers={"X-CSRF-Token": c.cookies.get("rlt_csrf")})
        check("Zayif parola reddedildi -> 400", r.status_code == 400, f"(got {r.status_code})")

        print("\n8) Cikis")
        r = c.post("/auth/logout", headers={"X-CSRF-Token": c.cookies.get("rlt_csrf")})
        check("Cikis 200", r.status_code == 200)
        c.cookies.clear()
        r = c.get("/api/trading/models")
        check("Cikis sonrasi API kapali", r.status_code == 401, f"(got {r.status_code})")

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
