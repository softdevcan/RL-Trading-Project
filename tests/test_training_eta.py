"""
Egitim suresi tahmini (ETA) testleri

Kapsam:
  - humanize bicimlendirmesi
  - gecmis kayit yokken yerlesik katsayidan tahmin (confidence='default')
  - gecmis kayit varken olculmus tahmin (confidence='measured')
  - farkli sembol sayisina dogrusal olcekleme (confidence='scaled')
  - canli ETA: isinma penceresinde on tahmin, sonra gozlenen hiz
  - bozuk/eksik gecmis dosyasinin tahmini dusurmemesi
  - kullanici calisma alani izolasyonu (Faz 7)

Calistirma:
    python tests/test_training_eta.py
"""

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# Calisma alanini gecici dizine yonlendir — gercek results/ kirletilmesin.
_TMP = tempfile.mkdtemp(prefix="eta_test_")
os.environ["WORKSPACE_ROOT"] = _TMP

from app.services import training_eta as eta  # noqa: E402

# results_dir() gercek cozumleyiciye gider; testte gecici dizine sabitle.
_orig_results_dir = eta.ws.results_dir
_current_ws = {"dir": os.path.join(_TMP, "u1")}


def _fake_results_dir(user_id=None):
    target = os.path.join(_TMP, user_id) if user_id else _current_ws["dir"]
    os.makedirs(target, exist_ok=True)
    return target


eta.ws.results_dir = _fake_results_dir


def _clear_history(user_id=None):
    path = eta.history_path(user_id)
    if os.path.exists(path):
        os.remove(path)


def test_humanize():
    print("\n[1] Sure bicimlendirme")
    check("45 sn", eta.humanize(45) == "45 sn", eta.humanize(45))
    check("tam dakika", eta.humanize(120) == "2 dk", eta.humanize(120))
    check("dakika + saniye", eta.humanize(125) == "2 dk 05 sn", eta.humanize(125))
    check("saat", eta.humanize(3725) == "1 sa 02 dk", eta.humanize(3725))
    check("None -> tire", eta.humanize(None) == "—")
    check("negatif -> tire", eta.humanize(-5) == "—")


def test_default_estimate():
    print("\n[2] Gecmis yokken yerlesik katsayi")
    _clear_history()
    ref = eta._DEFAULT_STEP_COST_SYMBOLS
    est = eta.estimate("ppo", phase=1, total_timesteps=50_000, n_symbols=ref)
    check("confidence=default", est["confidence"] == "default", est["confidence"])
    check("sample_size=0", est["sample_size"] == 0)
    expected_learn = eta._DEFAULT_STEP_COST["ppo"] * 50_000
    check("referans sembol sayisinda katsayi birebir kullaniliyor",
          abs(est["learn_seconds"] - expected_learn) < 1e-6,
          f"{est['learn_seconds']} != {expected_learn}")
    check("toplam = hazirlik + ogrenme + degerlendirme",
          abs(est["total_seconds"]
              - (est["setup_seconds"] + est["learn_seconds"] + est["eval_seconds"])) < 1e-6)
    check("metin uretildi", est["total_text"] and est["total_text"] != "—")

    # Varsayilan katsayi da sembol sayisiyla dogrusal olceklenmeli
    half = eta.estimate("ppo", phase=1, total_timesteps=50_000, n_symbols=ref // 2)
    check("yari sembol -> yari ogrenme suresi",
          abs(half["learn_seconds"] - expected_learn / 2) < 1e-6,
          f"{half['learn_seconds']} != {expected_learn / 2}")

    # Her algoritmanin kendi katsayisi var (bu konfigurasyonda degerler yakin,
    # ama tablo algoritma bazinda okunmali — karistirilmamali).
    sac = eta.estimate("sac", phase=1, total_timesteps=50_000, n_symbols=ref)
    check("SAC kendi katsayisini kullaniyor",
          abs(sac["learn_seconds"] - eta._DEFAULT_STEP_COST["sac"] * 50_000) < 1e-6)
    check("SAC ve PPO tahminleri ayni degil",
          sac["learn_seconds"] != est["learn_seconds"])


def test_measured_estimate():
    print("\n[3] Gecmis varken olculmus tahmin")
    _clear_history()
    # 3 kosum: 10k timestep, 20 sn ogrenme -> step_cost = 0.002
    for _ in range(3):
        eta.record_run(
            algorithm="ppo", phase=1, n_symbols=5, total_timesteps=10_000,
            setup_seconds=5.0, learn_seconds=20.0, eval_seconds=1.0, total_seconds=26.0,
        )
    est = eta.estimate("ppo", phase=1, total_timesteps=50_000, n_symbols=5)
    check("confidence=measured", est["confidence"] == "measured", est["confidence"])
    check("sample_size=3", est["sample_size"] == 3, str(est["sample_size"]))
    check("ogrenme = 0.002 x 50k = 100 sn",
          abs(est["learn_seconds"] - 100.0) < 1e-6, str(est["learn_seconds"]))
    check("hazirlik gecmisten alindi", abs(est["setup_seconds"] - 5.0) < 1e-6)
    check("degerlendirme gecmisten alindi", abs(est["eval_seconds"] - 1.0) < 1e-6)
    check("toplam 106 sn", abs(est["total_seconds"] - 106.0) < 1e-6, str(est["total_seconds"]))


def test_scaled_estimate():
    print("\n[4] Farkli sembol sayisina olcekleme")
    _clear_history()
    eta.record_run(
        algorithm="ppo", phase=1, n_symbols=5, total_timesteps=10_000,
        setup_seconds=5.0, learn_seconds=20.0, eval_seconds=1.0, total_seconds=26.0,
    )
    est = eta.estimate("ppo", phase=2, total_timesteps=10_000, n_symbols=30)
    check("confidence=scaled", est["confidence"] == "scaled", est["confidence"])
    # 5 sembolde 20 sn -> 30 sembolde 6x = 120 sn
    check("6x sembol -> 6x ogrenme suresi",
          abs(est["learn_seconds"] - 120.0) < 1e-6, str(est["learn_seconds"]))
    check("kaynak metni olceklemeyi belirtiyor", "olcekle" in est["source"].lower(),
          est["source"])


def test_algorithm_isolation():
    print("\n[5] Algoritmalar birbirine karismiyor")
    _clear_history()
    eta.record_run(
        algorithm="ppo", phase=1, n_symbols=5, total_timesteps=10_000,
        setup_seconds=5.0, learn_seconds=20.0, eval_seconds=1.0, total_seconds=26.0,
    )
    sac = eta.estimate("sac", phase=1, total_timesteps=10_000, n_symbols=5)
    check("PPO gecmisi SAC tahminine sizmiyor", sac["confidence"] == "default",
          sac["confidence"])


def test_live_eta():
    print("\n[6] Canli ETA")
    # Faz kirilimi sart: live_eta faz farkindadir, tek 'total' yetmez.
    prior = {"total_seconds": 600.0, "setup_seconds": 20.0,
             "learn_seconds": 570.0, "eval_seconds": 10.0}

    # Hazirlik: yalniz hazirlik payi erir, ogrenme + degerlendirme tam durur.
    state = {
        "current_step": 0, "total_steps": 100_000, "estimate": prior,
        "run_start_ts": time.time() - 5, "learn_start_ts": None,
        "phase_name": "preparing",
    }
    live = eta.live_eta(state)
    check("hazirlik fazinda kaynak=prior", live["eta_source"] == "prior", live["eta_source"])
    check("hazirlikta sadece hazirlik payi eridi",
          abs(live["eta_seconds"] - (15 + 570 + 10)) < 1.0, str(live["eta_seconds"]))

    # Hazirlik tahmininden uzun surse bile ogrenme payi silinmemeli
    state["run_start_ts"] = time.time() - 300
    check("hazirlik asilsa da ogrenme payi duruyor",
          abs(eta.live_eta(state)["eta_seconds"] - 580.0) < 1.0)

    # Isinma esiginin altinda: hala prior, ama ogrenmede gecen sure dusulur
    state.update({"current_step": 50, "learn_start_ts": time.time() - 5,
                  "phase_name": "training"})
    warm = eta.live_eta(state)
    check("isinma esigi altinda hala prior", warm["eta_source"] == "prior", warm["eta_source"])
    check("isinmada ogrenme payindan gecen sure dusuldu",
          abs(warm["eta_seconds"] - (565 + 10)) < 1.0, str(warm["eta_seconds"]))

    # Gozlem var: 10.000 adim / 10 sn = 1000 adim/sn -> kalan 90.000 adim = 90 sn
    state.update({"current_step": 10_000, "learn_start_ts": time.time() - 10.0})
    live = eta.live_eta(state)
    check("gozlem varken kaynak=measured", live["eta_source"] == "measured", live["eta_source"])
    check("hiz ~1000 adim/sn", 900 < live["steps_per_sec"] < 1100, str(live["steps_per_sec"]))
    check("ETA ~100 sn (90 ogrenme + 10 degerlendirme)",
          90 < live["eta_seconds"] < 110, str(live["eta_seconds"]))
    check("bitis saati uretildi", live["finish_at"] is not None)

    # SB3 rollout yuvarlamasi: current_step toplami asabilir (n_steps kati).
    # Eskiden bu durumda ETA 0 gorunup egitim devam ediyordu.
    over = eta.live_eta({
        "current_step": 4096, "total_steps": 4000, "estimate": prior,
        "run_start_ts": time.time() - 40, "learn_start_ts": time.time() - 30,
        "phase_name": "training",
    })
    check("adim toplami asinca ETA 0'a dusmuyor", over["eta_seconds"] >= 10.0,
          str(over["eta_seconds"]))
    check("asim durumunda kalan = degerlendirme payi",
          abs(over["eta_seconds"] - 10.0) < 0.5, str(over["eta_seconds"]))

    # Degerlendirme fazi: ogrenme bitti ama is bitmedi -> ETA 0 olmamali
    ev = eta.live_eta({
        "current_step": 4000, "total_steps": 4000, "estimate": prior,
        "run_start_ts": time.time() - 50, "learn_start_ts": time.time() - 40,
        "learn_end_ts": time.time() - 4, "phase_name": "evaluating",
    })
    check("degerlendirme fazinda ETA > 0", ev["eta_seconds"] > 0, str(ev["eta_seconds"]))
    check("degerlendirmede gecen sure dusuldu",
          abs(ev["eta_seconds"] - 6.0) < 1.0, str(ev["eta_seconds"]))

    # Tamamlandi
    check("tamamlaninca ETA=0",
          eta.live_eta({"phase_name": "completed"})["eta_seconds"] == 0.0)

    # Tahmin de gozlem de yoksa sessizce dusmeli
    empty = eta.live_eta({"current_step": 0, "total_steps": 0})
    check("veri yoksa ETA None", empty["eta_seconds"] is None)
    check("veri yoksa metin tire", empty["eta_text"] == "—")


def test_partial_estimate_dict():
    """Eksik alanli tahmin sozlugu canli ETA'yi dusurmemeli.

    Durum sozlugu surece ozgudur; eski sekilli bir 'estimate' (orn. sunucu
    yeniden baslamadan once yazilmis) ETA'yi patlatirsa pano ilerleme takibini
    komple kaybeder. Eksik alanlar 0 kabul edilir.
    """
    print("\n[7] Eksik alanli tahmin sozlugu")
    partial = {"total_seconds": 150.0}  # kirilim yok
    live = eta.live_eta({
        "current_step": 0, "total_steps": 10_000, "estimate": partial,
        "run_start_ts": time.time() - 5, "phase_name": "preparing",
    })
    check("eksik kirilimla patlamiyor", live["eta_seconds"] is not None)

    live = eta.live_eta({
        "current_step": 5_000, "total_steps": 10_000, "estimate": partial,
        "run_start_ts": time.time() - 20, "learn_start_ts": time.time() - 10,
        "phase_name": "training",
    })
    check("ogrenme fazinda eksik kirilimla calisiyor",
          live["eta_source"] == "measured" and live["eta_seconds"] > 0,
          str(live))

    check("estimate hic yoksa da patlamiyor",
          eta.live_eta({"current_step": 5, "total_steps": 10,
                        "phase_name": "training"})["eta_seconds"] is None)


def test_status_endpoint_resilience():
    """Bozuk on tahmin /train/status'u 500'e dusurmemeli."""
    print("\n[8] Durum ucu dayanikliligi")
    import asyncio
    from app.api.routes import trading as tr

    state = tr.get_training_state()
    original = dict(state)
    try:
        state.update({
            "is_training": True, "state": "running",
            "current_step": 5_000, "total_steps": 50_000,
            "phase_name": "training",
            "run_start_ts": time.time() - 20,
            "learn_start_ts": time.time() - 15,
            "estimate": {"total_seconds": 150.0},  # sema ile uyumsuz
        })
        status = asyncio.run(tr.get_training_status())
        check("bozuk tahminle status donuyor", status.is_training is True)
        check("bozuk tahmin None'a dusuruldu", status.estimate is None)
        check("ETA yine de hesaplandi", status.eta_text not in (None, "—"),
              str(status.eta_text))
        check("ilerleme dogru", abs(status.progress - 0.1) < 1e-9, str(status.progress))
    finally:
        state.clear()
        state.update(original)


def test_corrupt_history():
    print("\n[9] Bozuk gecmis dosyasi")
    _clear_history()
    path = eta.history_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{bozuk json")
    check("bozuk dosya bos gecmis gibi okunuyor", eta.load_history() == [])
    est = eta.estimate("ppo", phase=1, total_timesteps=10_000, n_symbols=5)
    check("bozuk dosyaya ragmen tahmin uretiliyor", est["confidence"] == "default")

    # JSON gecerli ama liste degil
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"beklenmeyen": "sozluk"}, fh)
    check("liste olmayan JSON bos kabul ediliyor", eta.load_history() == [])


def test_history_cap_and_invalid_records():
    print("\n[8] Kayit sinirlari")
    _clear_history()
    for i in range(eta.MAX_HISTORY + 25):
        eta.record_run(
            algorithm="ppo", phase=1, n_symbols=5, total_timesteps=1000,
            setup_seconds=1.0, learn_seconds=2.0, eval_seconds=0.5, total_seconds=3.5,
        )
    check(f"gecmis {eta.MAX_HISTORY} kayitla sinirli",
          len(eta.load_history()) == eta.MAX_HISTORY, str(len(eta.load_history())))

    _clear_history()
    eta.record_run(algorithm="ppo", phase=1, n_symbols=5, total_timesteps=0,
                   setup_seconds=1.0, learn_seconds=2.0, eval_seconds=0.5, total_seconds=3.5)
    check("timestep=0 kaydedilmiyor", eta.load_history() == [])
    eta.record_run(algorithm="ppo", phase=1, n_symbols=5, total_timesteps=1000,
                   setup_seconds=1.0, learn_seconds=0.0, eval_seconds=0.5, total_seconds=1.5)
    check("ogrenme suresi 0 kaydedilmiyor", eta.load_history() == [])


def test_workspace_isolation():
    print("\n[9] Kullanici izolasyonu (Faz 7)")
    _clear_history("alice")
    _clear_history("bob")
    eta.record_run(
        algorithm="ppo", phase=1, n_symbols=5, total_timesteps=10_000,
        setup_seconds=5.0, learn_seconds=20.0, eval_seconds=1.0, total_seconds=26.0,
        user_id="alice",
    )
    check("alice'in kaydi yazildi", len(eta.load_history("alice")) == 1)
    check("bob alice'in kaydini gormuyor", eta.load_history("bob") == [])
    a = eta.estimate("ppo", phase=1, total_timesteps=10_000, n_symbols=5, user_id="alice")
    b = eta.estimate("ppo", phase=1, total_timesteps=10_000, n_symbols=5, user_id="bob")
    check("alice olculmus tahmin aliyor", a["confidence"] == "measured")
    check("bob varsayilan tahmin aliyor", b["confidence"] == "default")


if __name__ == "__main__":
    print("=" * 70)
    print("Egitim suresi tahmini (ETA) testleri")
    print("=" * 70)
    try:
        test_humanize()
        test_default_estimate()
        test_measured_estimate()
        test_scaled_estimate()
        test_algorithm_isolation()
        test_live_eta()
        test_partial_estimate_dict()
        test_status_endpoint_resilience()
        test_corrupt_history()
        test_history_cap_and_invalid_records()
        test_workspace_isolation()
    finally:
        eta.ws.results_dir = _orig_results_dir
        shutil.rmtree(_TMP, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"SONUC: {passed} gecti, {failed} kaldi")
    print("=" * 70)
    sys.exit(1 if failed else 0)
