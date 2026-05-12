"""
Smoke test - Faz 5 UI dogrulamasi icin temel endpoint kontrolu.

Calistir: backend acikken (python run_server.py) ardindan:
    python tests/test_smoke.py

Cikti: PASS/FAIL satir satir. Hata varsa exit code 1.

Test edilen endpoint'ler:
  - /health, /api (health.py)
  - /api/trading/{models, data/status, portfolio-history, latest-portfolio,
                  analysis/best-models, analysis/generate-report (POST)}
  - /api/prediction/{symbols, models, gold/prices}
  - /api/hyperopt/{studies, search-spaces/PPO}
  - /api/config/{algorithms, phases, reward-types, feature-groups}    -- Faz B yeni
"""

import os
import sys

# Repo kokunu sys.path'e ekle (testler farkli cwd'lerden cagrilabilir)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests  # noqa: E402

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8888")
TIMEOUT = 15


def _check(method: str, path: str, expected_codes=(200,), payload=None) -> bool:
    """Tek bir endpoint cek; PASS/FAIL yazdir; bool dondur."""
    url = f"{API_BASE}{path}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, timeout=TIMEOUT)
        else:
            resp = requests.post(url, json=payload or {}, timeout=TIMEOUT)
    except Exception as exc:
        print(f"  FAIL  {method:4s} {path:55s} -> connection error: {exc}")
        return False

    if resp.status_code in expected_codes:
        print(f"  PASS  {method:4s} {path:55s} -> {resp.status_code}")
        return True
    else:
        print(
            f"  FAIL  {method:4s} {path:55s} -> {resp.status_code} "
            f"(beklenen: {expected_codes})"
        )
        return False


def main() -> int:
    print("=" * 78)
    print("FAZ 5 SMOKE TEST — Backend endpoint dogrulamasi")
    print(f"Hedef: {API_BASE}")
    print("=" * 78)

    results = []

    print("\n[1] Health & root")
    results.append(_check("GET", "/health"))
    results.append(_check("GET", "/api"))

    print("\n[2] Trading endpoint'leri")
    results.append(_check("GET", "/api/trading/models"))
    results.append(_check("GET", "/api/trading/data/status"))
    results.append(_check("GET", "/api/trading/portfolio-history"))
    results.append(_check("GET", "/api/trading/latest-portfolio"))
    # best-models: detailed_results.json yoksa 404 dondurur (akademik rapor henuz uretilmemis)
    results.append(_check("GET", "/api/trading/analysis/best-models",
                          expected_codes=(200, 404)))
    # generate-report POST — Faz A.1 sonrasi 202 Accepted donmeli
    results.append(_check("POST", "/api/trading/analysis/generate-report",
                          expected_codes=(200, 202)))

    print("\n[3] Prediction endpoint'leri")
    results.append(_check("GET", "/api/prediction/symbols"))
    results.append(_check("GET", "/api/prediction/models"))
    results.append(_check("GET", "/api/prediction/gold/prices"))

    print("\n[4] Hyperopt endpoint'leri")
    results.append(_check("GET", "/api/hyperopt/studies"))
    # AlgorithmType enum lowercase ister ("ppo", "a2c", "td3", "sac")
    results.append(_check("GET", "/api/hyperopt/search-spaces/ppo"))

    print("\n[5] Config endpoint'leri (Faz B yeni)")
    results.append(_check("GET", "/api/config/algorithms"))
    results.append(_check("GET", "/api/config/phases"))
    results.append(_check("GET", "/api/config/reward-types"))
    results.append(_check("GET", "/api/config/feature-groups"))

    print("\n" + "=" * 78)
    passed = sum(1 for r in results if r)
    total = len(results)
    if passed == total:
        print(f"SONUC: {passed}/{total} PASS — tum endpoint'ler saglikli.")
        print("=" * 78)
        return 0
    else:
        print(f"SONUC: {passed}/{total} PASS — {total - passed} endpoint hatali.")
        print("=" * 78)
        return 1


if __name__ == "__main__":
    sys.exit(main())
