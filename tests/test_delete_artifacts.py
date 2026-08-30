"""Egitilmis model ve optimizasyon kaydi silme testi (pytest degil, `python` ile).

    python tests/test_delete_artifacts.py

Neden var: deneme amacli kosumlar panoyu kalici olarak kirletiyordu. Model
silme ucu vardi ama arayuzu yoktu; optimizasyon icin silme yetenegi HIC yoktu
(`DELETE /studies/{id}` aslinda IPTAL ediyordu, calismayan kayda 404 donuyordu).

Ne dogrular:
  1. Model silme — sahibi silebiliyor, dosya ve metrik JSON'u gidiyor.
  2. Viewer silemiyor (RequireWriter).
  3. Ortak (kullanici oncesi) dizindeki model salt-okunur: 403, dosya duruyor.
  4. Olmayan model 404; yol gecisi (../) reddediliyor.
  5. Optimizasyon kaydi silme — Optuna deposundan ve bellekten dusuyor.
  6. Calisan kosum 409 ile reddediliyor (once iptal edilmeli).
  7. Iptal artik POST /studies/{id}/cancel; DELETE ile ayrik.
  8. /hyperopt/start artik RequireWriter — viewer optimizasyon baslatamiyor.

Optuna deposu testte GECICI bir dosyaya yonlendirilir; depodaki gercek
`results/hyperparameter_studies/optuna_studies.db` dosyasina dokunulmaz.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="rlt_delete_test_")

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
from app.auth.models import Role, User  # noqa: E402
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


def csrf(client: TestClient) -> dict:
    return {"X-CSRF-Token": client.cookies.get("rlt_csrf") or "",
            "X-Requested-With": "XMLHttpRequest"}


def main() -> int:
    print("1) Kurulum")
    reset_engine()
    security.reset_secret_cache()
    init_db()
    with session_scope() as db:
        create_user(db, email="writer@test.local", password="WriterPass1", full_name="W",
                    role=Role.USER, created_by="test", must_change_password=False)
        create_user(db, email="viewer@test.local", password="ViewerPass1", full_name="V",
                    role=Role.VIEWER, created_by="test", must_change_password=False)
    check("Kullanicilar olusturuldu", True)

    # Optuna deposunu gecici dosyaya yonlendir — gercek depo dosyasina
    # dokunulmamali. Uc modul seviyesindeki degiskeni cagri aninda okuyor.
    import app.api.routes.hyperopt as ho

    ho.OPTUNA_STORAGE = f"sqlite:///{os.path.join(_TMP, 'optuna.db').replace(os.sep, '/')}"

    from app.main import app  # noqa: E402
    from app.auth import workspace as ws  # noqa: E402

    print("\n2) Model silme")
    with TestClient(app) as c:
        check("Writer giris yapti", login(c, "writer@test.local", "WriterPass1"))
        with session_scope() as db:
            uid = db.query(User).filter(User.email == "writer@test.local").first().id

        with ws.use_workspace(uid):
            models_dir, results_dir = ws.models_dir(), ws.results_dir()
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "test_model.zip")
        metrics_path = os.path.join(results_dir, "test_model_metrics.json")
        open(model_path, "wb").write(b"PK\x03\x04 sahte model")
        open(metrics_path, "w").write("{}")
        check("Sahte model olusturuldu", os.path.exists(model_path))

        r = c.delete("/api/trading/models/test_model", headers=csrf(c))
        check("Silme kabul edildi", r.status_code == 200, f"(got {r.status_code} {r.text[:120]})")
        check("Model dosyasi gitti", not os.path.exists(model_path))
        check("Metrik JSON'u da gitti", not os.path.exists(metrics_path))

        r = c.delete("/api/trading/models/yok_boyle_bir_model", headers=csrf(c))
        check("Olmayan model 404", r.status_code == 404, f"(got {r.status_code})")

        # sanitize_model_name yol gecisini kesmeli
        r = c.delete("/api/trading/models/..%2F..%2Fetc%2Fpasswd", headers=csrf(c))
        check("Yol gecisi reddedildi", r.status_code in (400, 404),
              f"(got {r.status_code})")

        r = c.delete("/api/trading/models/test_model")
        check("CSRF'siz silme reddedildi", r.status_code == 403, f"(got {r.status_code})")

    print("\n3) Yetki")
    with TestClient(app) as c:
        check("Viewer giris yapti", login(c, "viewer@test.local", "ViewerPass1"))
        r = c.delete("/api/trading/models/herhangi", headers=csrf(c))
        check("Viewer model silemiyor", r.status_code == 403, f"(got {r.status_code})")
        r = c.post("/api/hyperopt/start",
                   json={"algorithm": "ppo", "n_trials": 1, "total_timesteps": 100},
                   headers=csrf(c))
        check("Viewer optimizasyon baslatamiyor", r.status_code == 403,
              f"(got {r.status_code})")

    print("\n4) Ortak (kullanici oncesi) model salt-okunur")
    with TestClient(app) as c:
        login(c, "writer@test.local", "WriterPass1")
        legacy_dir = get_settings().MODELS_DIR
        os.makedirs(legacy_dir, exist_ok=True)
        legacy = os.path.join(legacy_dir, "ortak_test_model.zip")
        open(legacy, "wb").write(b"PK\x03\x04 ortak")
        try:
            r = c.delete("/api/trading/models/ortak_test_model", headers=csrf(c))
            check("Ortak model 403", r.status_code == 403, f"(got {r.status_code})")
            check("Ortak model yerinde duruyor", os.path.exists(legacy))
            check("403 mesaji sebebi soyluyor",
                  "salt-okunur" in r.text or "ortak" in r.text.lower(),
                  f"(got {r.text[:120]})")
        finally:
            if os.path.exists(legacy):
                os.remove(legacy)

    print("\n5) Optimizasyon kaydi silme")
    import optuna

    with TestClient(app) as c:
        login(c, "writer@test.local", "WriterPass1")

        study = optuna.create_study(study_name="ppo_test_calismasi",
                                    storage=ho.OPTUNA_STORAGE)
        study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
        names = {s.study_name for s in optuna.get_all_study_summaries(ho.OPTUNA_STORAGE)}
        check("Depoda calisma var", "ppo_test_calismasi" in names, f"(got {names})")

        r = c.delete("/api/hyperopt/studies/ppo_test_calismasi", headers=csrf(c))
        check("Silme kabul edildi", r.status_code == 200,
              f"(got {r.status_code} {r.text[:150]})")
        names = {s.study_name for s in optuna.get_all_study_summaries(ho.OPTUNA_STORAGE)}
        check("Depodan dustu", "ppo_test_calismasi" not in names, f"(got {names})")

        r = c.delete("/api/hyperopt/studies/hic_olmayan", headers=csrf(c))
        check("Olmayan calisma 404", r.status_code == 404, f"(got {r.status_code})")

    print("\n6) Calisan kosum once iptal edilmeli")
    with TestClient(app) as c:
        login(c, "writer@test.local", "WriterPass1")
        ho.active_studies["calisan-1"] = {
            "study_id": "calisan-1", "study_name": "calisan-1",
            "status": ho.StudyStatus.RUNNING.value, "trials_completed": 0,
        }
        r = c.delete("/api/hyperopt/studies/calisan-1", headers=csrf(c))
        check("Calisan kosum silinemiyor (409)", r.status_code == 409,
              f"(got {r.status_code})")
        check("Kayit bellekte duruyor", "calisan-1" in ho.active_studies)

        r = c.post("/api/hyperopt/studies/calisan-1/cancel", headers=csrf(c))
        check("Iptal kabul edildi", r.status_code == 200,
              f"(got {r.status_code} {r.text[:120]})")
        check("Durum iptal edildi olarak isaretlendi",
              str(ho.active_studies["calisan-1"]["status"]).lower().endswith("cancelled"),
              f"(got {ho.active_studies['calisan-1']['status']})")

        r = c.delete("/api/hyperopt/studies/calisan-1", headers=csrf(c))
        check("Iptalden sonra silinebiliyor", r.status_code == 200,
              f"(got {r.status_code} {r.text[:120]})")
        check("Bellekten dustu", "calisan-1" not in ho.active_studies)

    print("\n7) Viewer optimizasyon kaydi silemiyor")
    with TestClient(app) as c:
        login(c, "viewer@test.local", "ViewerPass1")
        r = c.delete("/api/hyperopt/studies/herhangi", headers=csrf(c))
        check("Viewer silemiyor", r.status_code == 403, f"(got {r.status_code})")

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
