"""Kullanici bazli calisma alani (hibrit izolasyon) testleri.

    python tests/test_workspace_isolation.py

Dogrulananlar:
  * Iki kullanicinin model/sonuc/karar dizinleri ayrisir
  * Piyasa verisi dizinleri ORTAK kalir
  * Eski (kullanici oncesi) modeller salt-okunur olarak herkese gorunur
  * Egitim durumu kullanici basina tutulur
  * viewer rolu yazma uclarina erisemez
  * Dash sayfasi oturum acmis kullanicinin kimligiyle render olur
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_ws_test_")
os.environ["AUTH_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'auth.db').replace(os.sep, '/')}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["AUTH_ENABLED"] = "True"
os.environ["WORKSPACES_DIR"] = os.path.join(_TMP, "workspaces")
os.environ["MODELS_DIR"] = os.path.join(_TMP, "legacy_models")
os.environ["RESULTS_DIR"] = os.path.join(_TMP, "legacy_results")
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = ""
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.auth import security, workspace as ws  # noqa: E402
from app.auth.db import init_db, reset_engine, session_scope  # noqa: E402
from app.auth.models import Role  # noqa: E402
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


def main() -> int:
    reset_engine()
    security.reset_secret_cache()
    init_db()

    with session_scope() as db:
        alice = create_user(db, email="alice@test.com", password="AlicePass123",
                            role=Role.USER, must_change_password=False)
        bob = create_user(db, email="bob@test.com", password="BobPass1234",
                          role=Role.USER, must_change_password=False)
        viewer = create_user(db, email="view@test.com", password="ViewPass123",
                             role=Role.VIEWER, must_change_password=False)
        alice_id, bob_id = alice.id, bob.id

    print("\n1) Dizin cozumleme kullanicilara gore ayrisiyor")
    with ws.use_workspace(alice_id):
        alice_models = ws.models_dir()
        alice_decisions = ws.live_trading_dir()
    with ws.use_workspace(bob_id):
        bob_models = ws.models_dir()
        bob_decisions = ws.live_trading_dir()

    check("Model dizinleri farkli", alice_models != bob_models, f"({alice_models})")
    check("Karar dizinleri farkli", alice_decisions != bob_decisions)
    check("Alice dizini kendi id'sini iceriyor", alice_id in alice_models)
    check("Dizinler diskte olusturuldu",
          os.path.isdir(alice_models) and os.path.isdir(bob_models))

    print("\n2) Piyasa verisi ORTAK kaliyor")
    settings = get_settings()
    shared_dirs = [settings.BIST_DIR, settings.MACRO_DIR, settings.FUNDAMENTAL_DIR, settings.GOLD_DIR]
    with ws.use_workspace(alice_id):
        leaks = [d for d in shared_dirs if alice_id in os.path.abspath(d)]
    check("Ortak veri dizinleri kullaniciya gore degismiyor", not leaks, f"({leaks})")

    print("\n3) Eski ortak modeller salt-okunur gorunuyor")
    legacy_models = os.environ["MODELS_DIR"]
    os.makedirs(legacy_models, exist_ok=True)
    with open(os.path.join(legacy_models, "ppo_legacy.zip"), "wb") as fh:
        fh.write(b"fake-model")
    with ws.use_workspace(alice_id):
        found = ws.find_file("models", "ppo_legacy.zip")
        read_dirs = ws.read_dirs("models")
    check("Eski model bulunabiliyor", found is not None, f"({found})")
    check("Okuma dizinleri: once kullanici, sonra ortak", len(read_dirs) == 2, f"({read_dirs})")
    check("Yazma hedefi kullanici alani", ws.workspace_root(alice_id) in alice_models)

    print("\n4) Yol gecisi (path traversal) korumasi")
    try:
        ws.workspace_root("../../etc")
        check("Gecersiz kullanici id reddediliyor", False, "(hata firlatilmadi)")
    except ValueError:
        check("Gecersiz kullanici id reddediliyor", True)

    print("\n5) Egitim durumu kullanici basina")
    from app.api.routes.trading import get_training_state

    with ws.use_workspace(alice_id):
        state_a = get_training_state()
        state_a["is_training"] = True
    with ws.use_workspace(bob_id):
        state_b = get_training_state()
    check("Bob, Alice'in egitimini gormuyor", state_b["is_training"] is False)

    from app.services.prediction_service import _training_state, get_training_state as pred_state

    _training_state[(alice_id, "THYAO.IS", "daily", None)] = {
        "state": "running", "started_at": "2026-01-01T00:00:00",
    }
    with ws.use_workspace(bob_id):
        bob_view = pred_state("THYAO.IS", "daily")
    with ws.use_workspace(alice_id):
        alice_view = pred_state("THYAO.IS", "daily")
    check("Tahmin egitimi durumu izole", bob_view is None and alice_view is not None,
          f"(bob={bob_view})")

    print("\n6) HTTP: rol bazli yazma yetkisi ve Dash render")
    from app.main import app

    with TestClient(app, base_url="http://test", follow_redirects=False) as c:
        c.post("/auth/login", json={"email": "view@test.com", "password": "ViewPass123"})
        csrf = c.cookies.get("rlt_csrf")
        r = c.post("/api/trading/train", json={"algorithm": "ppo", "total_timesteps": 100},
                   headers={"X-CSRF-Token": csrf})
        check("viewer egitim baslatamiyor -> 403", r.status_code == 403, f"(got {r.status_code})")

        r = c.get("/api/trading/models")
        check("viewer model listesini okuyabiliyor", r.status_code == 200, f"(got {r.status_code})")

        c.cookies.clear()
        c.post("/auth/login", json={"email": "alice@test.com", "password": "AlicePass123"})
        r = c.get("/dash/", headers={"accept": "text/html"})
        check("Dash sayfasi acildi", r.status_code == 200, f"(got {r.status_code})")

        # Layout istemci tarafinda cekilir; kullanici rozeti burada gorunur.
        layout = c.get("/dash/_dash-layout")
        # Dash JSON'da "/" karakterini / olarak kacirir — geri cevir.
        body = layout.text.replace("\\u002f", "/")
        check("Layout alindi", layout.status_code == 200, f"(got {layout.status_code})")
        check("Kenar cubugunda kullanici kimligi var", "alice@test.com" in body)
        check("Cikis baglantisi var", "/logout" in body)
        check("Kullanici yonetimi menusu user rolunde gizli", "/dash/users" not in body)

    print("\n" + "=" * 60)
    print(f"  Gecen: {len(PASSED)}   Kalan: {len(FAILED)}")
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
