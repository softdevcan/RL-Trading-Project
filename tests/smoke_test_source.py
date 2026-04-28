"""
Smoke test for source parameter integration.
Run from project root: venv/Scripts/python.exe tests/smoke_test_source.py
Or from tests/ dir:    python smoke_test_source.py
"""
import os
import sys
import traceback

# Proje koku her iki calisma dizininden de erisilebilir olsun
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

errors = []

print("=" * 60)
print("SMOKE TEST: source parameter integration")
print("=" * 60)

# --- 1. IMPORTS ---
print("\n[1] Testing imports...")
try:
    from app.schemas.prediction import (
        PredictionTrainRequest, PredictionRequest, PredictionTarget,
        TradableSymbolsResponse, SymbolEntry, TrainedModelEntry, TrainedModelsResponse
    )
    print("  OK: app.schemas.prediction")
except Exception as e:
    print(f"  FAIL: app.schemas.prediction -> {e}")
    traceback.print_exc()
    errors.append("schemas import")

try:
    from app.services.prediction_service import (
        PredictionService, get_training_state, get_supported_sources,
        is_gold_symbol, _resolve_source
    )
    print("  OK: app.services.prediction_service")
except Exception as e:
    print(f"  FAIL: app.services.prediction_service -> {e}")
    traceback.print_exc()
    errors.append("prediction_service import")

try:
    from prediction.models.ensemble import StackingEnsemble
    print("  OK: prediction.models.ensemble")
except Exception as e:
    print(f"  FAIL: prediction.models.ensemble -> {e}")
    traceback.print_exc()
    errors.append("ensemble import")

try:
    from prediction.models.base import BasePredictionModel
    print("  OK: prediction.models.base")
except Exception as e:
    print(f"  FAIL: prediction.models.base -> {e}")
    traceback.print_exc()
    errors.append("base import")

try:
    from prediction.tracker import PredictionTracker
    print("  OK: prediction.tracker")
except Exception as e:
    print(f"  FAIL: prediction.tracker -> {e}")
    traceback.print_exc()
    errors.append("tracker import")

try:
    from prediction.trainer import WalkForwardTrainer
    print("  OK: prediction.trainer")
except Exception as e:
    print(f"  FAIL: prediction.trainer -> {e}")
    traceback.print_exc()
    errors.append("trainer import")

try:
    from app.api.routes.prediction import router
    print("  OK: app.api.routes.prediction")
except Exception as e:
    print(f"  FAIL: app.api.routes.prediction -> {e}")
    traceback.print_exc()
    errors.append("routes import")

# --- 2. SOURCES ---
print("\n[2] Testing get_supported_sources()...")
try:
    from app.services.prediction_service import get_supported_sources, is_gold_symbol, _resolve_source

    cases = [
        ("GOLD_GRAM_TRY", ["borsapy", "yfinance"]),
        ("AKBNK.IS", ["yfinance"]),
        ("GC=F", ["yfinance"]),
        ("EURTRY=X", ["borsapy", "yfinance"]),  # eur_try: both sources supported
    ]
    for sym, expected in cases:
        result = get_supported_sources(sym)
        status = "OK" if set(result) == set(expected) else f"MISMATCH (got {result})"
        print(f"  {status}: get_supported_sources('{sym}') = {result}")

    # is_gold
    assert is_gold_symbol("GOLD_GRAM_TRY") == True, "GOLD_GRAM_TRY should be gold"
    assert is_gold_symbol("AKBNK.IS") == False, "AKBNK.IS should not be gold"
    print("  OK: is_gold_symbol checks")

    # _resolve_source
    assert _resolve_source("GOLD_GRAM_TRY", "borsapy") == "borsapy"
    assert _resolve_source("GOLD_GRAM_TRY", None) == "borsapy"   # default for gold
    assert _resolve_source("AKBNK.IS", None) is None    # hisse: source ayrimi yok -> None
    assert _resolve_source("AKBNK.IS", "yfinance") is None  # hisse: yfinance passed -> still None
    print("  OK: _resolve_source checks")

    # reject invalid
    try:
        _resolve_source("GOLD_GRAM_USD", "borsapy")
        print("  FAIL: should have raised ValueError for GOLD_GRAM_USD+borsapy")
        errors.append("resolve_source validation")
    except ValueError as e:
        print(f"  OK: rejected invalid combo: {str(e)[:60]}")

except Exception as e:
    print(f"  FAIL: sources test -> {e}")
    traceback.print_exc()
    errors.append("sources test")

# --- 3. SCHEMA VALIDATION ---
print("\n[3] Testing schema validation...")
try:
    from app.schemas.prediction import PredictionTrainRequest, PredictionRequest

    req1 = PredictionTrainRequest(symbol="GOLD_GRAM_TRY", source="borsapy")
    print(f"  OK: TrainRequest(GOLD_GRAM_TRY, borsapy) -> source={req1.source}")

    req2 = PredictionTrainRequest(symbol="AKBNK.IS")
    print(f"  OK: TrainRequest(AKBNK.IS, no source) -> source={req2.source}")

    req3 = PredictionRequest(symbols=["GOLD_GRAM_TRY"], source="borsapy")
    print(f"  OK: PredictionRequest(GOLD_GRAM_TRY, borsapy) -> source={req3.source}")

    req4 = PredictionRequest(targets=[{"symbol": "GOLD_GRAM_TRY", "source": "yfinance"}, {"symbol": "AKBNK.IS"}])
    print(f"  OK: PredictionRequest(targets) -> targets count={len(req4.targets)}")

except Exception as e:
    print(f"  FAIL: schema validation -> {e}")
    traceback.print_exc()
    errors.append("schema validation")

# --- 4. ENSEMBLE PATH NAMING ---
print("\n[4] Testing StackingEnsemble path naming...")
try:
    from prediction.models.ensemble import StackingEnsemble

    e1 = StackingEnsemble(horizon="daily", source="yfinance")
    path1 = e1._ensemble_meta_path("GOLD_GRAM_TRY")
    print(f"  Ensemble(source=yfinance) path: {path1}")
    assert "yfinance" in str(path1), f"Expected 'yfinance' in path: {path1}"
    print("  OK: source in path")

    e2 = StackingEnsemble(horizon="daily")
    path2 = e2._ensemble_meta_path("AKBNK.IS")
    print(f"  Ensemble(no source) path: {path2}")
    print("  OK: no-source path")

except Exception as e:
    print(f"  FAIL: ensemble path naming -> {e}")
    traceback.print_exc()
    errors.append("ensemble path naming")

# --- SUMMARY ---
print("\n" + "=" * 60)
if errors:
    print(f"FAILED ({len(errors)} errors): {errors}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
