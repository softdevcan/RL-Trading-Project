"""Faz 11 S4: M8 ileri veri uclari ve uygulama ici tetik (pytest degil).

    python tests/test_marl_forward_api.py

Ne dogrular (ag yok; is tetigi sahte fonksiyonla degistirilir):
  1. Ayar varsayilan KAPALI: uygulama acilisi zamanlayiciyi baslatmaz (testler ve Docker ag/defter islemi yapmaz).
  2. GET /api/marl/forward/status: oturumsuz 401; viewer 200 (ozet alanlari).
  3. POST /api/marl/forward/run: viewer 403 (RequireAdmin), admin 200 + tetik cagrildi + denetim kaydi.
  4. Zamanlayici kurali: hafta sonu / kesim saatinden once / gun bitti / deneme siniri -> calismaz; hafta ici 19+ calisir.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_marl_fwd_api_")
os.environ["AUTH_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'auth.db').replace(os.sep, '/')}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["AUTH_ENABLED"] = "True"
os.environ["WORKSPACES_DIR"] = os.path.join(_TMP, "workspaces")
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = ""
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = ""
os.environ.pop("MARL_FORWARD_AUTORUN", None)

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app.auth import security  # noqa: E402
from app.auth.db import init_db, reset_engine, session_scope  # noqa: E402
from app.auth.models import AuditLog, Role  # noqa: E402
from app.auth.service import create_user  # noqa: E402

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(f"[{'OK' if ok else 'HATA'}] {name}")
    passed, failed = passed + bool(ok), failed + (not ok)


def login(c, email, pw):
    return c.post("/auth/login", json={"email": email, "password": pw}).status_code == 200


def csrf(c):
    return {"X-CSRF-Token": c.cookies.get("rlt_csrf") or "", "X-Requested-With": "XMLHttpRequest"}


reset_engine()
security.reset_secret_cache()
init_db()
with session_scope() as db:
    create_user(db, email="admin@test.local", password="AdminPass1", full_name="A", role=Role.ADMIN,
                created_by="test", must_change_password=False)
    create_user(db, email="viewer@test.local", password="ViewerPass1", full_name="V", role=Role.VIEWER,
                created_by="test", must_change_password=False)

from app.main import app  # noqa: E402
from app.services import marl_forward as MF  # noqa: E402

calls = []
MF.run_now = lambda reason='manual': calls.append(reason) or True        # ag ve defter yok
MF.available = lambda: True

with TestClient(app) as c:
    check("1. MARL_FORWARD_AUTORUN varsayilan kapali: zamanlayici baslamadi",
          get_settings().MARL_FORWARD_AUTORUN is False and MF._state['scheduler'] is False)
    r0 = c.get("/api/marl/forward/status")
    ok2 = r0.status_code == 401
    ok2 &= login(c, "viewer@test.local", "ViewerPass1")
    r = c.get("/api/marl/forward/status")
    body = r.json() if r.status_code == 200 else {}
    ok2 &= r.status_code == 200 and {'forward_start', 'read_sessions', 'n_sessions', 'scheduler', 'running'} <= set(body)
    ok2 &= body.get('forward_start') == '2026-11-23' and body.get('read_sessions') == 126
    check(f"2. status: oturumsuz {r0.status_code}, viewer {r.status_code} + ozet alanlari", ok2)
    rv = c.post("/api/marl/forward/run", headers=csrf(c))
    c.post("/auth/logout", headers=csrf(c))
with TestClient(app) as c:
    login(c, "admin@test.local", "AdminPass1")
    ra = c.post("/api/marl/forward/run", headers=csrf(c))
    with session_scope() as db:
        n_audit = db.query(AuditLog).filter(AuditLog.action == "marl.forward_run").count()
    check(f"3. run: viewer {rv.status_code} (403 beklenir), admin {ra.status_code} + tetik + denetim kaydi",
          rv.status_code == 403 and ra.status_code == 200 and ra.json().get('started') is True and calls == ['manual']
          and n_audit == 1)

TRT = MF.TRT
MF._state['attempts'].clear()
MF._state['done'].clear()
mon_20 = datetime(2026, 11, 23, 20, 0, tzinfo=TRT)
ok4 = MF._due(mon_20) and not MF._due(datetime(2026, 11, 23, 18, 59, tzinfo=TRT))
ok4 &= not MF._due(datetime(2026, 11, 28, 20, 0, tzinfo=TRT))                 # cumartesi
MF._state['attempts']['2026-11-23'] = MF.MAX_ATTEMPTS_PER_DAY
ok4 &= not MF._due(mon_20)
MF._state['attempts'].clear()
MF._state['done'].add('2026-11-23')
ok4 &= not MF._due(mon_20) and MF._due(datetime(2026, 11, 24, 19, 30, tzinfo=TRT))
check("4. zamanlayici: hafta ici 19+ calisir; 19 oncesi, hafta sonu, deneme siniri, bitmis gun calismaz", ok4)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
