"""Ust cubuk testi (pytest degil, dogrudan `python` ile).

    python tests/test_topbar.py

Ne dogrular (Faz 8, G):
  1. Ust cubuk DOM'da; gorunum anahtari TASINDI, cogaltilmadi (tek kopya).
  2. Kirinti (grup > sayfa) her rota icin dogru; bilinmeyen rotada bos kalir.
  3. Baglamsal eylem belgelenen akisi izliyor; akista karsiligi olmayan
     sayfada slot bos.
  4. Arama secenekleri role duyarli: admin grubu yalnizca admin'e gorunur.
  5. Aramadan secim url'i degistirip kutuyu bosaltiyor; ayni sayfa secilirse
     sayfa bosuna yeniden cizilmiyor (no_update).
  6. Yapisal bekci: kenar cubugundaki her menu maddesinin kirinti karsiligi
     var. Menuye madde eklenip buraya eklenmezse ust cubuk sessizce bosalir.
  7. Bildirimler: /api/account/notifications bellekteki calisma durumlarindan
     ne uretiyor (suren/hatali/yeni biten), "yakinda bitti" penceresi disinda
     kalan kosum zilde asili kaliyor mu, baska kullanicinin kosumu siziyor mu,
     ve zil callback'i rozeti/menuyu dogru dolduruyor mu.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_topbar_test_")

os.environ["AUTH_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'auth.db').replace(os.sep, '/')}"
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


def login(client: TestClient, email: str, password: str) -> bool:
    return client.post("/auth/login",
                       json={"email": email, "password": password}).status_code == 200


def _outputs_of(key: str):
    """`..a.p...b.q..` / `a.p` -> Dash'in bekledigi output spec."""
    def one(part: str) -> dict:
        cid, _, prop = part.partition(".")
        return {"id": cid, "property": prop.split("@")[0]}
    if key.startswith("..") and key.endswith(".."):
        return [one(part) for part in key[2:-2].split("...")]
    return one(key)


def main() -> int:
    print("1) Kurulum")
    reset_engine()
    security.reset_secret_cache()
    init_db()
    with session_scope() as db:
        create_user(db, email="user@test.local", password="UserPass1234",
                    full_name="Ali Veli", role=Role.USER, created_by="test",
                    must_change_password=False)
        create_user(db, email="root@test.local", password="RootPass1234",
                    full_name="Root Admin", role=Role.ADMIN, created_by="test",
                    must_change_password=False)
    check("Kullanicilar olusturuldu", True)

    from app.main import app, _dash_app  # noqa: E402
    from dashboard.components.sidebar import ADMIN_GROUP, NAV_GROUPS  # noqa: E402
    from dashboard.components.topbar import NEXT_STEP, ROUTE_INDEX  # noqa: E402

    def fire(client: TestClient, fragment: str, inputs: list, states: list | None = None):
        key = next(k for k in _dash_app.callback_map if fragment in k)
        entry = _dash_app.callback_map[key]
        body = {
            "output": key,
            "outputs": _outputs_of(key),
            "inputs": [dict(d, value=v) for d, v in zip(entry["inputs"], inputs)],
            "state": [dict(d, value=v)
                      for d, v in zip(entry.get("state") or [], states or [])],
            "changedPropIds": [f"{d['id']}.{d['property']}" for d in entry["inputs"]],
        }
        return client.post("/dash/_dash-update-component", json=body,
                           headers={"X-CSRF-Token": client.cookies.get("rlt_csrf") or ""})

    print("\n2) Yerlesim ve tasinan gorunum anahtari")
    with TestClient(app) as c:
        check("Giris basarili", login(c, "user@test.local", "UserPass1234"))
        layout = c.get("/dash/_dash-layout").text
        check("Ust cubuk DOM'da", '"topbar"' in layout)
        # Tasima, cogaltma degil: iki kopya kalirsa biri tiklaninca digerinin
        # etiketi guncellenmez ve tema anahtari tutarsiz gorunur.
        check("Gorunum anahtari tek kopya", layout.count('"theme-toggle"') == 1,
              f"(got {layout.count(chr(34) + 'theme-toggle' + chr(34))})")
        check("Arama kutusu DOM'da", '"topbar-search"' in layout)
        check("Kenar cubugu hesap satiri duruyor", "sidebar-account" in layout)

        print("\n3) Kirinti")
        r = fire(c, "topbar-crumb", ["/dash/training"])
        check("Kirinti callback'i calisti", r.status_code == 200, f"(got {r.status_code})")
        crumb = r.json()["response"]["topbar-crumb"]["children"]
        check("Grup dogru", crumb[0]["props"]["children"] == "Sistem",
              f"(got {crumb[0]['props']['children']!r})")
        check("Sayfa adi dogru", crumb[2]["props"]["children"] == "Egitim",
              f"(got {crumb[2]['props']['children']!r})")

        r = fire(c, "topbar-crumb", ["/dash/"])
        crumb = r.json()["response"]["topbar-crumb"]["children"]
        check("Dashboard kirintisi", crumb[2]["props"]["children"] == "Dashboard",
              f"(got {crumb[2]['props']['children']!r})")

        r = fire(c, "topbar-crumb", ["/dash/bilinmeyen"])
        check("Bilinmeyen rotada kirinti bos",
              r.json()["response"]["topbar-crumb"]["children"] == [],
              f"(got {r.json()['response']['topbar-crumb']['children']})")

        print("\n4) Baglamsal eylem")
        r = fire(c, "topbar-crumb", ["/dash/data"])
        actions = r.json()["response"]["topbar-actions"]["children"]
        check("Veri sayfasinda sonraki adim var", bool(actions))
        check("Sonraki adim egitime goturuyor",
              actions and actions[0]["props"]["href"] == "/dash/training",
              f"(got {actions[0]['props']['href'] if actions else None})")

        r = fire(c, "topbar-crumb", ["/dash/account"])
        check("Hesabim akista degil, slot bos",
              r.json()["response"]["topbar-actions"]["children"] == [])

        print("\n5) Arama")
        r = fire(c, "url.pathname", ["/dash/models"], ["/dash/"])
        check("Secim yonlendiriyor",
              r.json()["response"].get("url", {}).get("pathname") == "/dash/models",
              f"(got {r.json()['response']})")
        check("Kutu bosaltiliyor",
              r.json()["response"].get("topbar-search", {}).get("value") is None)

        # Ayni sayfa secilirse url'e ayni degeri yazmak display_page'i bosuna
        # tetiklerdi; no_update ile yanitta url hic yer almaz.
        r = fire(c, "url.pathname", ["/dash/models"], ["/dash/models"])
        check("Ayni sayfa yeniden cizilmiyor", "url" not in r.json()["response"],
              f"(got {r.json()['response']})")

    print("\n6) Arama secenekleri role duyarli")
    with TestClient(app) as c:
        login(c, "user@test.local", "UserPass1234")
        layout = c.get("/dash/_dash-layout").text
        check("user Yonetim grubunu gormuyor", "Yonetim" not in layout)
    with TestClient(app) as c:
        login(c, "root@test.local", "RootPass1234")
        layout = c.get("/dash/_dash-layout").text
        check("admin Yonetim grubunu goruyor", "Yonetim" in layout)

    print("\n7) Yapisal bekci")
    # Menuye madde eklenip ROUTE_INDEX'e eklenmezse ust cubuk o sayfada
    # sessizce bosalir — hicbir hata vermez, bu yuzden burada denetleniyor.
    menu_routes = {item["href"]
                   for _group, items in list(NAV_GROUPS) + [ADMIN_GROUP]
                   for item in items}
    missing = sorted(menu_routes - set(ROUTE_INDEX))
    check("Her menu maddesinin kirintisi var", not missing, f"eksik: {missing}")

    # Sonraki-adim haritasi var olmayan bir rotaya isaret etmemeli
    known = menu_routes | {"/dash/account"}
    bad = sorted(route for route, (_t, _i, href) in NEXT_STEP.items()
                 if route not in known or href not in known)
    check("Sonraki adim haritasi gecerli rotalara isaret ediyor", not bad,
          f"gecersiz: {bad}")

    print("\n8) Bildirimler (/api/account/notifications + zil)")
    # Bildirimler kalici bir tablodan degil, BELLEKTEKI calisma durumlarindan
    # uretiliyor. Test de o sozlukleri dogrudan kurup ucun ne cikardigina bakar.
    import time as _time

    from app.api.routes.account import NOTIFY_RECENT_SECONDS
    from app.api.routes.trading import _empty_training_state, _training_states
    from app.services.prediction_service import _training_state as pred_state
    from app.auth.models import User

    with session_scope() as db:
        uid = db.query(User).filter(User.email == "user@test.local").first().id
        other = db.query(User).filter(User.email == "root@test.local").first().id

    def notifications(client: TestClient) -> list:
        r = client.get("/api/account/notifications")
        return r.json().get("items", []) if r.status_code == 200 else [{"http": r.status_code}]

    with TestClient(app) as c:
        check("Giris (bildirim)", login(c, "user@test.local", "UserPass1234"))

        _training_states.clear()
        pred_state.clear()
        check("Hicbir sey calismiyorken zil bos", notifications(c) == [],
              f"(got {notifications(c)})")

        _training_states[uid] = {**_empty_training_state(), "is_training": True,
                                 "state": "running", "current_step": 43,
                                 "total_steps": 100}
        items = notifications(c)
        check("Suren egitim bildiriliyor",
              any(i["id"] == "rl-training" for i in items), f"(got {items})")
        check("Ilerleme yuzdesi tasiniyor",
              any("%43" in (i.get("body") or "") for i in items), f"(got {items})")

        _training_states[uid] = {**_empty_training_state(), "state": "error",
                                 "error": "CUDA out of memory"}
        items = notifications(c)
        check("Hata bildiriliyor", any(i["kind"] == "error" for i in items),
              f"(got {items})")
        check("Hata metni tasiniyor",
              any("CUDA" in (i.get("body") or "") for i in items), f"(got {items})")

        _training_states[uid] = {**_empty_training_state(), "state": "completed",
                                 "finished_ts": _time.time()}
        check("Yeni biten egitim bildiriliyor",
              any(i["id"] == "rl-done" for i in notifications(c)))

        # Pencere disinda kalan kosum zilde asili kalmamali
        _training_states[uid] = {**_empty_training_state(), "state": "completed",
                                 "finished_ts": _time.time() - NOTIFY_RECENT_SECONDS - 60}
        check("Eski kosum zilde asili kalmiyor", notifications(c) == [],
              f"(got {notifications(c)})")

        # Damgasiz "completed" gosterilmez: ne zaman bittigini bilmiyoruz
        _training_states[uid] = {**_empty_training_state(), "state": "completed"}
        check("Zaman damgasiz bitis gosterilmiyor", notifications(c) == [],
              f"(got {notifications(c)})")

        print("\n  -- tahmin egitimi --")
        _training_states.clear()
        pred_state[(uid, "AKBNK.IS", "daily", "yfinance")] = {
            "state": "running", "source": "yfinance",
            "started_at": "2026-08-30T10:00:00", "finished_at": None,
            "error": None, "result": None,
        }
        items = notifications(c)
        check("Suren tahmin egitimi bildiriliyor",
              any("AKBNK.IS" in i["title"] for i in items), f"(got {items})")

        # Baska kullanicinin kosumu sizmamali
        pred_state[(other, "THYAO.IS", "daily", "yfinance")] = {
            "state": "running", "source": "yfinance",
            "started_at": "2026-08-30T10:00:00", "finished_at": None,
            "error": None, "result": None,
        }
        _training_states[other] = {**_empty_training_state(), "is_training": True,
                                   "state": "running", "current_step": 1,
                                   "total_steps": 10}
        items = notifications(c)
        check("Baska kullanicinin tahmin egitimi sizmiyor",
              not any("THYAO.IS" in i["title"] for i in items), f"(got {items})")
        check("Baska kullanicinin RL egitimi sizmiyor",
              not any(i["id"] == "rl-training" for i in items), f"(got {items})")

        print("\n  -- zil callback'i --")
        r = fire(c, "topbar-notify.children", [1])
        check("Zil callback'i calisti", r.status_code == 200, f"(got {r.status_code})")
        resp = r.json()["response"] if r.status_code == 200 else {}
        check("Rozet sayaci dolduruldu",
              resp.get("topbar-notify-badge", {}).get("children") == "1",
              f"(got {resp.get('topbar-notify-badge')})")
        check("Rozet turu sinifa yansiyor",
              "is-info" in (resp.get("topbar-notify-badge", {}).get("className") or ""),
              f"(got {resp.get('topbar-notify-badge')})")
        check("Menu ogesi uretildi", "AKBNK.IS" in r.text)

        pred_state.clear()
        _training_states.clear()
        r = fire(c, "topbar-notify.children", [2])
        resp = r.json()["response"]
        check("Bos durumda rozet gizli",
              "is-empty" in resp["topbar-notify-badge"]["className"],
              f"(got {resp['topbar-notify-badge']})")
        check("Bos durumda aciklayici satir var", "Yeni bildirim yok" in r.text)

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
