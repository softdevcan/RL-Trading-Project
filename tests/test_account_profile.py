"""Profil sayfasi ucdan uca testi (pytest degil, dogrudan `python` ile).

    python tests/test_account_profile.py

Ne dogrular:
  1. GET  /api/account/me         — kendi alanlari + calisma alani ozeti
  2. PATCH /api/account/profile   — ad soyad guncelleme, dogrulama, CSRF
  3. Yetki  — viewer kendi adini degistirebilir; kimse baskasinin hesabina
              yazamaz (govdeden kullanici kimligi ALINMIYOR)
  4. Rol yukseltme yolu kapali    — govdeye role/is_active konsa da yok sayilir
  5. GET  /api/account/sessions   — gruplama ve "bu tarayici" isareti
  6. POST /api/account/sessions/revoke-others
              — diger oturumlar kapanir, cagiran oturum ayakta kalir,
                kapatilan oturum grace penceresinden GERI DONEMEZ
  7. Denetim kaydi (account.update / account.revoke_sessions)
  8. _device_label birim kontrolleri
  9. Dash callback'leri gercek HTTP yolundan calisiyor mu — layout render
     testi callback GOVDESINDEKI hatayi yakalamaz (Faz 8 notu), o yuzden
     dordu de /dash/_dash-update-component uzerinden tetiklenir
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_account_test_")
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
from app.auth.models import AuditLog, Role, SessionToken, User  # noqa: E402
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


def login(client: TestClient, email: str, password: str) -> bool:
    r = client.post("/auth/login", json={"email": email, "password": password})
    return r.status_code == 200


def csrf_headers(client: TestClient, with_csrf: bool = True) -> dict:
    headers = {"X-Requested-With": "XMLHttpRequest"}
    if with_csrf:
        headers["X-CSRF-Token"] = client.cookies.get("rlt_csrf") or ""
    return headers


def name_of(email: str) -> str:
    with session_scope() as db:
        row = db.query(User).filter(User.email == email).first()
        return row.full_name if row else ""


def audit_actions(email: str) -> list[str]:
    with session_scope() as db:
        return [
            row.action
            for row in db.query(AuditLog).filter(AuditLog.email == email).all()
        ]


def main() -> int:
    print("1) Kurulum")
    reset_engine()
    security.reset_secret_cache()
    init_db()
    with session_scope() as db:
        create_user(db, email="alice@test.local", password="AlicePass123",
                    full_name="Alice Eski", role=Role.USER, created_by="test",
                    must_change_password=False)
        create_user(db, email="bob@test.local", password="BobPass1234",
                    full_name="Bob Eski", role=Role.USER, created_by="test",
                    must_change_password=False)
        create_user(db, email="view@test.local", password="ViewPass123",
                    full_name="Viewer Eski", role=Role.VIEWER, created_by="test",
                    must_change_password=False)
    check("Kullanicilar olusturuldu", True)

    from app.main import app  # noqa: E402

    print("\n2) GET /api/account/me")
    with TestClient(app) as c:
        check("Giris basarili", login(c, "alice@test.local", "AlicePass123"))

        r = c.get("/api/account/me")
        check("me 200 donuyor", r.status_code == 200, f"(got {r.status_code})")
        body = r.json() if r.status_code == 200 else {}
        user = body.get("user", {})
        check("Kendi e-postasi donuyor", user.get("email") == "alice@test.local",
              f"(got {user.get('email')!r})")
        check("Parola hash'i sizmiyor", "password_hash" not in user)
        check("Son giris alani var", "last_login_at" in user)
        check("Hesap kalici isaretli", body.get("persistent") is True)
        check("Calisma alani ozeti var", "bytes" in (body.get("workspace") or {}),
              f"(got {body.get('workspace')})")

        print("\n3) PATCH /api/account/profile")
        r = c.patch("/api/account/profile", json={"full_name": "Alice Yeni"},
                    headers=csrf_headers(c, with_csrf=False))
        check("CSRF'siz istek reddedildi", r.status_code == 403, f"(got {r.status_code})")
        check("Reddedilen istek DB'yi degistirmedi", name_of("alice@test.local") == "Alice Eski",
              f"(got {name_of('alice@test.local')!r})")

        r = c.patch("/api/account/profile", json={"full_name": "Alice Yeni"},
                    headers=csrf_headers(c))
        check("Gecerli guncelleme kabul edildi", r.status_code == 200,
              f"(got {r.status_code})")
        check("Yanit yeni adi donuyor",
              (r.json().get("user") or {}).get("full_name") == "Alice Yeni")
        check("Ad DB'ye yazildi", name_of("alice@test.local") == "Alice Yeni",
              f"(got {name_of('alice@test.local')!r})")
        check("Denetim kaydi yazildi", "account.update" in audit_actions("alice@test.local"),
              f"(got {audit_actions('alice@test.local')})")

        r = c.patch("/api/account/profile", json={"full_name": ""},
                    headers=csrf_headers(c))
        check("Bos ad reddedildi", r.status_code == 422, f"(got {r.status_code})")

        r = c.patch("/api/account/profile", json={"full_name": "x" * 200},
                    headers=csrf_headers(c))
        check("Cok uzun ad reddedildi", r.status_code == 422, f"(got {r.status_code})")
        check("Reddedilen uzun ad DB'yi degistirmedi",
              name_of("alice@test.local") == "Alice Yeni")

        print("\n4) Yetki siniri")
        # Govdeye baska kullanici kimligi / rol konsa da yok sayilmali:
        # hedef HER ZAMAN oturumdaki kullanici, rol alani semada yok.
        r = c.patch(
            "/api/account/profile",
            json={"full_name": "Alice Admin", "user_id": "bob", "role": "admin",
                  "email": "hacker@test.local", "is_active": False},
            headers=csrf_headers(c),
        )
        check("Fazladan alanlar istegi bozmadi", r.status_code == 200,
              f"(got {r.status_code})")
        check("Bob'un adi degismedi", name_of("bob@test.local") == "Bob Eski",
              f"(got {name_of('bob@test.local')!r})")
        with session_scope() as db:
            alice = db.query(User).filter(User.email == "alice@test.local").first()
            check("Rol yukseltilemedi", alice.role == Role.USER, f"(got {alice.role!r})")
            check("E-posta degismedi", alice.email == "alice@test.local")
            check("Hesap devre disi birakilamadi", alice.is_active is True)

        # Bolum 6 oturum SAYISINI dogruluyor; bu istemcinin oturumu acik
        # kalirsa oraya ucuncu bir satir olarak sizar.
        c.post("/auth/logout", headers=csrf_headers(c))

    print("\n5) Viewer kendi profilini yonetebiliyor")
    with TestClient(app) as c:
        check("Viewer giris yapti", login(c, "view@test.local", "ViewPass123"))
        r = c.patch("/api/account/profile", json={"full_name": "Viewer Yeni"},
                    headers=csrf_headers(c))
        check("Viewer adini degistirdi", r.status_code == 200, f"(got {r.status_code})")
        check("Viewer'in adi kaydedildi", name_of("view@test.local") == "Viewer Yeni",
              f"(got {name_of('view@test.local')!r})")

    print("\n6) GET /api/account/sessions")
    # Iki ayri tarayici: UA farkli olmali, yoksa (ip, user_agent) gruplamasi
    # ikisini tek satirda birlestirir.
    first = TestClient(app, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120"})
    second = TestClient(app, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) Firefox/121"})
    with first as c1, second as c2:
        check("Birinci tarayici giris yapti", login(c1, "alice@test.local", "AlicePass123"))
        check("Ikinci tarayici giris yapti", login(c2, "alice@test.local", "AlicePass123"))

        r = c1.get("/api/account/sessions")
        check("sessions 200 donuyor", r.status_code == 200, f"(got {r.status_code})")
        sessions = r.json().get("sessions", []) if r.status_code == 200 else []
        check("Iki oturum gorunuyor", len(sessions) == 2, f"(got {len(sessions)})")
        check("Tam olarak biri 'bu tarayici'",
              sum(1 for s in sessions if s.get("current")) == 1,
              f"(got {[s.get('current') for s in sessions]})")
        check("Mevcut oturum listenin basinda", bool(sessions and sessions[0].get("current")))
        check("Cihaz etiketi cozuldu",
              any("Chrome" in (s.get("device") or "") for s in sessions),
              f"(got {[s.get('device') for s in sessions]})")

        print("\n7) POST /api/account/sessions/revoke-others")
        r = c1.post("/api/account/sessions/revoke-others",
                    headers=csrf_headers(c1, with_csrf=False))
        check("CSRF'siz iptal reddedildi", r.status_code == 403, f"(got {r.status_code})")

        r = c1.post("/api/account/sessions/revoke-others", headers=csrf_headers(c1))
        check("Iptal kabul edildi", r.status_code == 200, f"(got {r.status_code})")
        check("Bir oturum kapatildi", r.json().get("revoked") == 1,
              f"(got {r.json()})")

        with session_scope() as db:
            alice = db.query(User).filter(User.email == "alice@test.local").first()
            rows = db.query(SessionToken).filter(SessionToken.user_id == alice.id).all()
            check("Geriye tek oturum kaydi kaldi", len(rows) == 1, f"(got {len(rows)})")

        check("Denetim kaydi yazildi",
              "account.revoke_sessions" in audit_actions("alice@test.local"))

        # Cagiran oturum ayakta kalmali
        r = c1.get("/api/account/me")
        check("Kendi oturumu ayakta", r.status_code == 200, f"(got {r.status_code})")

        r = c1.get("/api/account/sessions")
        check("Listede tek oturum kaldi", len(r.json().get("sessions", [])) == 1,
              f"(got {r.json().get('sessions')})")

        # Kapatilan oturum GRACE PENCERESINDEN GERI DONEMEMELI.
        # Kayit isaretlenip birakilsaydi rotate_session bunu "es zamanli
        # yenileme yarisi" sayip 30 sn boyunca yeni oturum verirdi; kayit
        # silindigi icin session_unknown -> 401.
        r = c2.post("/auth/refresh")
        check("Kapatilan oturum yenilenemiyor", r.status_code in (401, 403),
              f"(got {r.status_code})")
        with session_scope() as db:
            alice = db.query(User).filter(User.email == "alice@test.local").first()
            rows = db.query(SessionToken).filter(SessionToken.user_id == alice.id).all()
            check("Yenileme denemesi yeni oturum uretmedi", len(rows) == 1,
                  f"(got {len(rows)})")

    print("\n8) Oturumsuz erisim")
    with TestClient(app) as c:
        r = c.get("/api/account/me")
        check("Oturumsuz me reddedildi", r.status_code in (401, 403),
              f"(got {r.status_code})")
        r = c.patch("/api/account/profile", json={"full_name": "X"},
                    headers={"X-Requested-With": "XMLHttpRequest"})
        check("Oturumsuz yazma reddedildi", r.status_code in (401, 403),
              f"(got {r.status_code})")

    print("\n9) Cihaz etiketi")
    from app.api.routes.account import _device_label, _group_sessions

    check("Chrome/Windows cozuldu",
          _device_label("Mozilla/5.0 (Windows NT 10.0; Win64) Chrome/120 Safari/537") ==
          "Chrome · Windows")
    # Edge ve Opera UA'si "Chrome" de icerir; once onlar denenmeli
    check("Edge, Chrome'a karismiyor",
          _device_label("Mozilla/5.0 (Windows NT 10.0) Chrome/120 Safari/537 Edg/120") ==
          "Edge · Windows")
    check("Bos UA icin nötr etiket",
          _device_label("") == "Bilinmeyen istemci")
    check("Taninmayan UA kirpiliyor, uydurulmuyor",
          _device_label("x" * 80).startswith("x") and
          _device_label("x" * 80).endswith("..."))

    # Ayni (ip, ua) ciftinden birden fazla gecerli kayit tek satirda toplanir
    class _Row:
        def __init__(self, jti, ip, ua, issued, expires):
            self.jti, self.ip, self.user_agent = jti, ip, ua
            self.issued_at, self.expires_at = issued, expires

    from datetime import datetime

    rows = [
        _Row("b", "1.1.1.1", "UA-1", datetime(2026, 8, 30, 12, 0), datetime(2026, 9, 30)),
        _Row("a", "1.1.1.1", "UA-1", datetime(2026, 8, 30, 11, 0), datetime(2026, 9, 30)),
        _Row("c", "2.2.2.2", "UA-2", datetime(2026, 8, 30, 10, 0), datetime(2026, 9, 30)),
    ]
    grouped = _group_sessions(rows, "c")
    check("Ayni cihazin coklu kaydi tek satir", len(grouped) == 2, f"(got {len(grouped)})")
    check("Grup token sayisini tasiyor",
          any(g["tokens"] == 2 for g in grouped), f"(got {[g['tokens'] for g in grouped]})")
    check("Mevcut oturum basa alindi", grouped[0]["current"] is True)

    print("\n10) Dash callback'leri (gercek HTTP yolu)")
    # Sayfa layout'u sorunsuz render olsa bile callback GOVDESI patlayabilir
    # (Faz 8'de f-string icindeki budanmis bir import boyle kacmisti). Dordu de
    # tarayicinin kullandigi ucdan tetiklenir.
    from app.main import _dash_app

    def outputs_of(key: str):
        """`..a.p...b.q..` / `a.p` / `a.p@hash` -> Dash'in bekledigi output spec."""
        def one(part: str) -> dict:
            cid, _, prop = part.partition(".")
            return {"id": cid, "property": prop.split("@")[0]}
        if key.startswith("..") and key.endswith(".."):
            return [one(part) for part in key[2:-2].split("...")]
        return one(key)

    def find_key(fragment: str) -> str:
        return next(k for k in _dash_app.callback_map if fragment in k)

    def fire(client: TestClient, fragment: str, inputs: list, states: list | None = None):
        key = find_key(fragment)
        entry = _dash_app.callback_map[key]
        body = {
            "output": key,
            "outputs": outputs_of(key),
            "inputs": [dict(d, value=v) for d, v in zip(entry["inputs"], inputs)],
            "state": [dict(d, value=v)
                      for d, v in zip(entry.get("state") or [], states or [])],
            "changedPropIds": [f"{d['id']}.{d['property']}" for d in entry["inputs"]],
        }
        return client.post("/dash/_dash-update-component", json=body,
                           headers=csrf_headers(client))

    with TestClient(app) as c:
        check("Bob giris yapti", login(c, "bob@test.local", "BobPass1234"))
        r = c.get("/dash/account", headers={"accept": "text/html"})
        check("Hesabim sayfasi acildi", r.status_code == 200, f"(got {r.status_code})")

        r = fire(c, "account-avatar", [0])
        check("Profil callback'i calisti", r.status_code == 200, f"(got {r.status_code})")
        data = r.json().get("response", {}) if r.status_code == 200 else {}
        check("Avatar bas harfleri dolduruldu",
              data.get("account-avatar", {}).get("children") == "BE",
              f"(got {data.get('account-avatar')})")
        check("E-posta dolduruldu",
              data.get("account-email", {}).get("children") == "bob@test.local",
              f"(got {data.get('account-email')})")
        check("Girdi kutusu mevcut adi tasiyor",
              data.get("account-name-input", {}).get("value") == "Bob Eski",
              f"(got {data.get('account-name-input')})")

        r = fire(c, "account-sessions-body.children", [0])
        check("Oturum callback'i calisti", r.status_code == 200, f"(got {r.status_code})")
        check("Mevcut tarayici isaretlendi", "bu tarayici" in r.text)

        r = fire(c, "account-alert", [1], ["Bob Guncel", 0])
        check("Kaydet callback'i calisti", r.status_code == 200, f"(got {r.status_code})")
        check("Basari uyarisi dondu", '"color": "success"' in r.text or '"color":"success"' in r.text,
              f"(got {r.text[:200]})")
        check("Callback DB'ye yazdi", name_of("bob@test.local") == "Bob Guncel",
              f"(got {name_of('bob@test.local')!r})")

        r = fire(c, "account-alert", [1], ["   ", 0])
        check("Bos ad callback'te durduruldu",
              r.status_code == 200 and "bos olamaz" in r.text, f"(got {r.status_code})")
        check("Bos ad DB'yi degistirmedi", name_of("bob@test.local") == "Bob Guncel")

        r = fire(c, "account-session-alert", [1], [1])
        check("Oturum kapatma callback'i calisti", r.status_code == 200,
              f"(got {r.status_code})")

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
