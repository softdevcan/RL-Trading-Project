"""Egitim durumu ucu regresyon testi (pytest degil, `python` ile).

    python tests/test_training_status.py

Neden var: SB3 `total_timesteps`'i tam tutturmuyor — ogrenme `n_steps`
bloklariyla ilerledigi icin son blok hedefi asabiliyor. Canlida gorulen deger
1.001472 idi. `TrainingStatus.progress` semasi [0,1] ile sinirli oldugu icin
ham oran YANIT DOGRULAMASINI dusuruyor, `/train/status` 500 veriyor ve pano
egitim boyunca ne ilerleme ne ETA gosterebiliyordu:

    GET /api/trading/train/status failed: 1 validation error for TrainingStatus
    progress: Input should be less than or equal to 1 [input_value=1.001472]

Sinsi tarafi: hata YALNIZCA egitim bitisine yakin ciktigi icin kisa duman
testlerinde gorunmuyor.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_status_test_")

os.environ["AUTH_DB_URL"] = f"sqlite:///{os.path.join(_TMP, 'auth.db').replace(os.sep, '/')}"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["AUTH_ENABLED"] = "False"
os.environ["WORKSPACES_DIR"] = os.path.join(_TMP, "workspaces")
os.environ["BOOTSTRAP_ADMIN_EMAIL"] = ""
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

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
    from app.main import app
    from app.api.routes.trading import _empty_training_state, _training_states

    def set_state(**kwargs) -> None:
        _training_states.clear()
        _training_states["local"] = {**_empty_training_state(), **kwargs}

    with TestClient(app) as c:
        print("1) Hedefi asan adim sayisi")
        # Canlida 1.001472 gorulmustu; asagidaki cift ayni sinifi temsil
        # ediyor (oran > 1), birebir ayni sayilar degil.
        set_state(is_training=True, state="running",
                  current_step=50_074, total_steps=50_000)
        r = c.get("/api/trading/train/status")
        check("Asan adimda uc 200 donuyor", r.status_code == 200,
              f"(got {r.status_code} {r.text[:160]})")
        if r.status_code == 200:
            check("progress 1.0'a kirpildi", r.json().get("progress") == 1.0,
                  f"(got {r.json().get('progress')})")

        print("\n2) Normal durumlar bozulmadi")
        set_state(is_training=True, state="running",
                  current_step=25_000, total_steps=50_000)
        r = c.get("/api/trading/train/status")
        check("Yarida progress 0.5", r.status_code == 200 and r.json()["progress"] == 0.5,
              f"(got {r.status_code} {r.text[:120]})")

        set_state(current_step=0, total_steps=0)
        r = c.get("/api/trading/train/status")
        check("Hic baslamamisken progress 0", r.status_code == 200 and r.json()["progress"] == 0.0,
              f"(got {r.status_code} {r.text[:120]})")

        set_state(is_training=False, state="completed",
                  current_step=50_000, total_steps=50_000)
        r = c.get("/api/trading/train/status")
        check("Tam bitiste progress 1.0",
              r.status_code == 200 and r.json()["progress"] == 1.0,
              f"(got {r.status_code} {r.text[:120]})")

        print("\n3) Sembol evreni ve uyarilar disari cikiyor mu")
        # `split_data` kronolojik boldugu icin, sembollerin gecmisleri esit
        # degilse bolumlerin sembol sayisi farkli cikiyor ve model bir
        # gozlem boyutunda egitilip baskasiyla degerlendiriliyordu. Rota
        # artik hizaliyor; kullanicinin BUNU GORMESI sart, yoksa 30 sembol
        # sandigi modeli 5 sembolle egitmis olur.
        set_state(is_training=False, state="completed",
                  current_step=10, total_steps=10, n_symbols=5,
                  warnings=["Panelde 30 sembol var ama egitim penceresinde 5 tanesi bulunuyor"])
        r = c.get("/api/trading/train/status")
        body = r.json() if r.status_code == 200 else {}
        check("n_symbols yanitta var", body.get("n_symbols") == 5,
              f"(got {body.get('n_symbols')})")
        check("Uyari yanitta tasiniyor",
              body.get("warnings") and "30 sembol" in body["warnings"][0],
              f"(got {body.get('warnings')})")

        set_state(is_training=True, state="running", current_step=1, total_steps=10)
        r = c.get("/api/trading/train/status")
        check("Uyari yokken alan bos liste",
              r.status_code == 200 and r.json().get("warnings") == [],
              f"(got {r.text[:120]})")

        print(chr(10) + "4) Sayfaya donuldugunde ilerleme geri geliyor mu")
        # `display_page` yalnizca page-content'i degistirir; sayfa her
        # gezinmede YENIDEN uretilir. dcc.Interval `disabled=True` dogup
        # yalnizca "Baslat" dugmesiyle aciliyordu, dolayisiyla baska bir
        # sayfaya gidip donen kullanici surmekte olan egitimi bir daha
        # goremiyordu. layout() artik acilista durumu soruyor.
        import dashboard.pages.training as training_page

        def find_by_id(node, target):
            if getattr(node, "id", None) == target:
                return node
            children = getattr(node, "children", None)
            if children is None:
                return None
            if not isinstance(children, (list, tuple)):
                children = [children]
            for child in children:
                found = find_by_id(child, target)
                if found is not None:
                    return found
            return None

        set_state(is_training=True, state="running", phase_name="training",
                  current_step=7_000, total_steps=10_000)
        tree = training_page.layout()
        poll = find_by_id(tree, "training-poll")
        content = find_by_id(tree, "training-status-content")
        check("Kosum surerken yoklama ACIK dogar",
              poll is not None and poll.disabled is False,
              f"(got {getattr(poll, 'disabled', 'yok')})")
        check("Kosum surerken ilerleme blogu basiliyor",
              content is not None and "egitiliyor" in str(content.children).lower(),
              f"(got {str(getattr(content, 'children', ''))[:120]})")

        set_state(is_training=False, state="idle", current_step=0, total_steps=0)
        tree = training_page.layout()
        poll = find_by_id(tree, "training-poll")
        check("Bosta yoklama KAPALI dogar",
              poll is not None and poll.disabled is True,
              f"(got {getattr(poll, 'disabled', 'yok')})")

        _training_states.clear()

        print(chr(10) + "5) Tahmin egitimi durumu")
        # Tahmin sayfasi yalnizca tek seferlik "arka planda baslatildi" uyarisi
        # basiyor, bir daha guncellemiyordu: kullanici bitti mi suruyor mu
        # anlamiyordu ve sayfadan cikip donunce o uyari da kayboluyordu.
        # `/train/status` sembol istedigi icin sayfa donuste NEYI soracagini
        # bilmiyordu; `/train/active` listeyi backend'den verir.
        from app.services.prediction_service import _training_state as pred_state

        pred_state.clear()
        r = c.get("/api/prediction/train/active")
        check("active ucu bos listeyle 200", r.status_code == 200 and r.json()["runs"] == [],
              f"(got {r.status_code} {r.text[:100]})")

        pred_state[("local", "AKBNK.IS", "daily", "yfinance")] = {
            "state": "running", "source": "yfinance",
            "started_at": "2026-08-30T10:00:00", "finished_at": None,
            "error": None, "result": {"buyuk": "cikti"},
        }
        pred_state[("baska-kullanici", "THYAO.IS", "daily", "yfinance")] = {
            "state": "running", "source": "yfinance",
            "started_at": "2026-08-30T10:00:00", "finished_at": None,
            "error": None, "result": None,
        }
        body = c.get("/api/prediction/train/active").json()
        check("Kendi kosumu listeleniyor",
              any(r0.get("symbol") == "AKBNK.IS" for r0 in body["runs"]),
              f"(got {body})")
        check("Baska kullanicinin kosumu sizmiyor",
              not any(r0.get("symbol") == "THYAO.IS" for r0 in body["runs"]),
              f"(got {body})")
        check("running sayaci dogru", body["running"] == 1, f"(got {body['running']})")
        check("Egitim ciktisi listede tasinmiyor",
              all("result" not in r0 for r0 in body["runs"]), f"(got {body['runs']})")

        import dashboard.pages.prediction as prediction_page

        tree = prediction_page.layout()
        poll = find_by_id(tree, "pred-train-poll")
        status_box = find_by_id(tree, "pred-train-status")
        check("Kosum surerken tahmin yoklamasi ACIK dogar",
              poll is not None and poll.disabled is False,
              f"(got {getattr(poll, 'disabled', 'yok')})")
        check("Sayfaya donuldugunde durum blogu basiliyor",
              status_box is not None and "AKBNK.IS" in str(status_box.children),
              f"(got {str(getattr(status_box, 'children', ''))[:120]})")

        pred_state.clear()
        tree = prediction_page.layout()
        poll = find_by_id(tree, "pred-train-poll")
        check("Kosum yokken tahmin yoklamasi KAPALI dogar",
              poll is not None and poll.disabled is True,
              f"(got {getattr(poll, 'disabled', 'yok')})")

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
