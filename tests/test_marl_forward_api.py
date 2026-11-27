"""Faz 11 S4: M8 ileri veri uclari ve uygulama ici tetik (pytest degil).

    python tests/test_marl_forward_api.py

Ne dogrular (ag yok; is tetigi sahte fonksiyonla degistirilir):
  1. Ayar varsayilan KAPALI: uygulama acilisi zamanlayiciyi baslatmaz (testler ve Docker ag/defter islemi yapmaz).
  2. GET /api/marl/forward/status: oturumsuz 401; viewer 200 (ozet alanlari).
  3. POST /api/marl/forward/run: viewer 403 (RequireAdmin), admin 200 + tetik cagrildi + denetim kaydi.
  4. Zamanlayici kurali: hafta sonu / kesim saatinden once / gun bitti / deneme siniri -> calismaz; hafta ici 19+ calisir.
  5. S5 gorunum ve urun portfoyu uclari (401/200/422) + pano sayfasi ve callback'leri (arsiv varsa).
"""
from __future__ import annotations

import json
import re
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

# 5 S5: gorunum + urun portfoyu uclari ve pano sayfasi (dondurulmus arsiv varsa)
HAVE = os.path.exists(os.path.join('data', 'marl', 'frozen', 'm8-2026', 'manifest.json')) and     os.path.exists(os.path.join('data', 'marl', 'panel_m8.npz'))
if HAVE:
    from app.main import _dash_app

    def fire(client, fragment, inputs, states=None):
        # kimlik sinirinda eslesme: 'sim-chart.figure' ana sayfanin 'home-sim-chart.figure'ini secmesin (§3.14)
        key = next(k for k in _dash_app.callback_map if re.search(r'(^|\.)' + re.escape(fragment), k))
        entry = _dash_app.callback_map[key]

        def one(part):
            cid, _, prop = part.partition(".")
            return {"id": cid, "property": prop.split("@")[0]}
        outs = [one(x) for x in key[2:-2].split("...")] if key.startswith("..") else one(key)
        body = {"output": key, "outputs": outs,
                "inputs": [dict(d, value=v) for d, v in zip(entry["inputs"], inputs)],
                "state": [dict(d, value=v) for d, v in zip(entry.get("state") or [], states or [])],
                "changedPropIds": [f"{d['id']}.{d['property']}" for d in entry["inputs"]]}
        return client.post("/dash/_dash-update-component", json=body, headers=csrf(client))

    with TestClient(app) as c:
        u0 = c.get("/api/marl/forward/view", params={"window": "test2026"}).status_code
        login(c, "viewer@test.local", "ViewerPass1")
        rv = c.get("/api/marl/forward/view", params={"window": "test2026", "keys": "M-none,rule/ew"})
        v = rv.json() if rv.status_code == 200 else {}
        ok5 = u0 == 401 and rv.status_code == 200 and v.get('n_sessions') == 207 and v['lines']['M-none']['n'] == 12
        ok5 &= c.get("/api/marl/forward/view", params={"window": "test2026", "keys": "yok"}).status_code == 422
        ok5 &= c.get("/api/marl/forward/view", params={"window": "baska"}).status_code == 422
        rp = c.get("/api/marl/forward/product", params={"sid": "rule/ew", "capital": 100000, "window": "test2026"})
        ok5 &= rp.status_code == 200 and len(rp.json().get('dates', [])) == 207
        pj = rp.json() if rp.status_code == 200 else {}
        hold = pj.get('holdings') or []
        # Lot tablosu muhasebesi: sum(lot * kapanis) + nakit = tam sayi defterinin son degeri; adetler tam sayi, eksi yok
        ok5 &= bool(hold) and all(isinstance(h['lots'], int) and h['lots'] >= 0 for h in hold)
        ok5 &= abs(sum(h['market_value'] for h in hold) + pj['cash_integer_tl'] - pj['value_integer'][-1]) \
            <= 1e-6 * pj['value_integer'][-1]
        ok5 &= c.get("/api/marl/forward/product", params={"sid": "rule/ew", "capital": 0}).status_code == 422
        ok5 &= c.get("/api/marl/forward/product", params={"sid": "policy/x/s1", "capital": 1e5,
                                                           "window": "test2026"}).status_code == 422
        # §3.13 gunluk defter: her gun lot x degerleme + nakit = deger; kol satirinda 12 tohum
        bk = pj.get('book') or {}
        ok5 &= len(bk.get('dates', [])) == 207 and all(
            abs(sum(l * m for l, m in zip(bk['lots'][i], bk['mark'][i])) + bk['cash'][i] - bk['value'][i])
            <= 1e-6 * bk['value'][i] for i in range(len(bk['dates'])))
        ok5 &= len(next(r for r in v['table'] if r['key'] == 'M-none').get('members', [])) == 12
        rpage = fire(c, "page-content.children", ["/dash/simulation"])
        keys = ["M-none", "rule/ew", "rule/minvar"]
        rall = fire(c, "sim-view-all.data", ["test2026"])
        vall = rall.json().get("response", {}).get("sim-view-all", {}).get("data") if rall.status_code == 200 else None
        rview = fire(c, "sim-view.data", [vall, keys, "M-none"], ["test2026"])
        vresp = rview.json().get("response", {}) if rview.status_code == 200 else {}
        vdata = vresp.get("sim-view", {}).get("data")
        rbook = fire(c, "sim-book.data", ["test2026", "M-none", 201, 100000])
        bdata = rbook.json().get("response", {}).get("sim-book", {}).get("data") if rbook.status_code == 200 else None
        rrend = fire(c, "sim-chart.figure", [bdata, vdata, None, "all"], ["M-none", 201])
        fig = (rrend.json().get("response", {}).get("sim-chart", {}).get("figure", {}) if rrend.status_code == 200
               else {})
        rpan = fire(c, "sim-kpis.children", [bdata, vdata, None], ["M-none", 201])
        ok5 &= rpage.status_code == 200 and "Sanal Para" in rpage.text and "Portföyümde ne var" in json.dumps(
            rpage.json(), ensure_ascii=False)          # yanit unicode kacisli olabilir
        ok5 &= bool(vdata) and bool(bdata) and rrend.status_code == 200 and rpan.status_code == 200 \
            and "Toplam" in rpan.text
        # tablo 10 stratejinin hepsi (S-esm dahil), grafik yalniz secilen 3 (aralik yalniz grafigi yeniden cizer)
        ok5 &= len(vall.get("table", [])) == 10 and list(vdata.get("lines", {})) == keys \
            and "S-esm" in json.dumps(vresp.get("sim-table", {}), ensure_ascii=False)
        # M-none bant (2) + ort. + 2 kural + incelenen + secili gun isareti + gunluk K/Z + dusus
        ok5 &= len(fig.get("data", [])) == 9
    check(f"5. S5 + §3.13: view 401/200 (207 seans, M-none N=12 + 12 tohum satiri), gecersiz anahtar/pencere 422, urun "
          f"200 / sermaye 0 / bilinmeyen strateji 422; lot tablosu ve gunluk defter: lot x kapanis + nakit = deger; "
          f"pano sayfasi + gorunum + defter + render callback'leri 200 (grafik 9 iz)", ok5)

    # 6 §3.14 ana sayfa: sayfa + M8 ozeti (snapshot) + simulasyon kutusu + durum seridi (TRT saati)
    from dashboard.pages import home as H
    with TestClient(app) as c:
        login(c, "viewer@test.local", "ViewerPass1")
        hp = fire(c, "page-content.children", ["/dash/"])
        htxt = json.dumps(hp.json(), ensure_ascii=False) if hp.status_code == 200 else ""
        rm8 = fire(c, "home-m8-chart.figure", [0])
        m8fig = rm8.json().get("response", {}).get("home-m8-chart", {}).get("figure", {}) if rm8.status_code == 200 else {}
        rsim = fire(c, "home-sim-chart.figure", ["test2026", 0])
        sresp = rsim.json().get("response", {}) if rsim.status_code == 200 else {}
        rst = fire(c, "home-strip-wrap.children", [0])
        st = c.get("/api/marl/forward/status").json()
    m8 = H.load_m8_summary()
    f = H.featured(m8)
    bars = (m8fig.get("data") or [{}])[0]
    ok6 = hp.status_code == 200 and "Öne çıkan model" in htxt and "MARL iletişimsiz (MAPPO)" in htxt
    ok6 &= "Bu proje ne yapıyor?" in htxt and "Yapay zekâ, basit kurallardan" in htxt     # §3.18 sade ozet ustte
    # one cikan = bos kol haric en yuksek ortalama; kurallar ondan yukarida; cubuk degerleri ozetle ayni (10 cubuk)
    ok6 &= f["arm"]["kol"] == "M-none" and f["rules_above"] == 4 and len(bars.get("x", [])) == 10
    ok6 &= sorted(bars.get("x", [])) == sorted([a["ort"] for a in m8["kollar"]] + [r["ort"] for r in m8["kurallar"]])
    # 2026 test: M-none bant + 3 cizgi; tablo 3 satir; durum seridi 200
    ok6 &= len(sresp.get("home-sim-chart", {}).get("figure", {}).get("data", [])) == 4 and rst.status_code == 200
    ok6 &= "Toplam getiri" in json.dumps(sresp.get("home-sim-table", {}), ensure_ascii=False)
    lr = st.get("last_run") or {}
    ok6 &= (not lr.get("finished")) or (lr.get("finished_trt") or "").endswith("+03:00")
    check("6. §3.14 ana sayfa: sayfa 200 + one cikan M-none (bos kol haric), 4 kural ustte, M8 grafigi 10 cubuk = "
          "snapshot ozeti; 2026 test kutusu 4 iz + tablo; durum seridi 200; son calisma TRT damgali", ok6)
else:
    print("[ATLANDI] 5. dondurulmus arsiv yok")

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
