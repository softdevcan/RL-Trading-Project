"""
Faz 6 — Tahmin Pipeline Regresyon Testi (altın-dosya / golden file)

Sprintin GÜVENLİK AĞI. Her performans optimizasyonu (warm-start, paralellik,
feature-eng refactor, AMP ...) davranışı KORUMALI. Bu test, sabit seed +
sentetik veriyle üretilen:
  1. feature matrisini      (deterministik, sıkı tolerans)
  2. ensemble tahmin/metriklerini (DL kaynaklı ufak sapmaya makul tolerans)
kaydedilmiş bir altın-dosyaya karşı doğrular.

Kullanım:
    # İlk sefer / bilinçli davranış değişikliğinden sonra altın-dosyayı üret:
    python tests/test_prediction_regression.py --update

    # Regresyon kontrolü (CI / optimizasyon sonrası):
    python tests/test_prediction_regression.py

Çıkış kodu: 0 = geçti, 1 = regresyon. Altın-dosya:
    tests/golden/prediction_regression.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden', 'prediction_regression.json')

# Toleranslar: feature-eng deterministik olmalı; ensemble DL modelleri
# (torch/cudnn) run'lar arası ufak sapma gösterebilir → gevşek eşik.
FEATURE_RTOL = 1e-6
FEATURE_ATOL = 1e-6
METRIC_RTOL = 0.05   # %5 — DL non-determinizmine karşı tampon
SEED = 20260722


def _make_df(n: int = 700) -> pd.DataFrame:
    """Deterministik sentetik OHLCV (seed sabit)."""
    rng = np.random.RandomState(SEED)
    idx = pd.bdate_range('2019-01-01', periods=n)
    close = 100 * np.cumprod(1 + rng.randn(n) * 0.01)
    return pd.DataFrame({
        'open': close * (1 + rng.randn(n) * 0.002),
        'high': close * (1 + np.abs(rng.randn(n)) * 0.005),
        'low':  close * (1 - np.abs(rng.randn(n)) * 0.005),
        'close': close,
        'volume': np.abs(rng.randn(n) * 1e6) + 1e5,
    }, index=idx)


def _seed_everything():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass


def _compute_snapshot() -> dict:
    """Feature matrisi özeti + ensemble tahmin/metriklerini üret."""
    from prediction.feature_engineer import PredictionFeatureEngineer
    from prediction.models.ensemble import StackingEnsemble

    df = _make_df()

    # --- 1. Feature matrisi imzası ---
    fe = PredictionFeatureEngineer('daily')
    feat = fe.build_features(df, 'REG')
    cols = fe.get_feature_columns(feat)
    X = feat[cols].fillna(0).values.astype(np.float64)
    feature_sig = {
        'n_rows': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'columns': cols,
        # Kolon-bazlı özet (tam matris yerine): ortalama + std imzası
        'col_mean': np.nan_to_num(X.mean(axis=0)).tolist(),
        'col_std': np.nan_to_num(X.std(axis=0)).tolist(),
    }

    # --- 2. Ensemble tahmin/metrikleri ---
    _seed_everything()
    ens = StackingEnsemble(horizon='daily', use_tats=True)
    res = ens.train(df, 'REG')
    pred = ens.predict_next(df, 'REG')

    ensemble_sig = {
        'models_trained': sorted(res.get('models_trained', [])),
        'n_models': res.get('n_models'),
        'test_mape': res.get('ensemble_test_metrics', {}).get('mape'),
        'test_dir_acc': res.get('ensemble_test_metrics', {}).get('direction_accuracy'),
        'predicted_direction': pred.get('predicted_direction'),
        'predicted_change_pct': pred.get('predicted_change_pct'),
        'n_models_predict': pred.get('n_models'),
    }

    # temizlik
    import glob
    for f in glob.glob('models/prediction/REG_*'):
        try:
            os.remove(f)
        except OSError:
            pass

    return {'feature': feature_sig, 'ensemble': ensemble_sig}


def _compare(golden: dict, current: dict) -> list:
    """Farkları döndür (boş liste = geçti)."""
    fails = []

    g_f, c_f = golden['feature'], current['feature']
    if g_f['columns'] != c_f['columns']:
        gset, cset = set(g_f['columns']), set(c_f['columns'])
        fails.append(
            f"feature kolonları değişti: eklenen={sorted(cset - gset)[:5]} "
            f"çıkan={sorted(gset - cset)[:5]}"
        )
    else:
        for key in ('col_mean', 'col_std'):
            g = np.array(g_f[key]); c = np.array(c_f[key])
            if not np.allclose(g, c, rtol=FEATURE_RTOL, atol=FEATURE_ATOL):
                idx = int(np.argmax(np.abs(g - c)))
                fails.append(
                    f"feature.{key} sapması: kolon '{c_f['columns'][idx]}' "
                    f"golden={g[idx]:.6g} current={c[idx]:.6g}"
                )

    g_e, c_e = golden['ensemble'], current['ensemble']
    if g_e['models_trained'] != c_e['models_trained']:
        fails.append(
            f"models_trained değişti: golden={g_e['models_trained']} "
            f"current={c_e['models_trained']}"
        )
    if g_e['predicted_direction'] != c_e['predicted_direction']:
        fails.append(
            f"predicted_direction değişti: golden={g_e['predicted_direction']} "
            f"current={c_e['predicted_direction']}"
        )
    for key in ('test_mape', 'test_dir_acc', 'predicted_change_pct'):
        g = g_e.get(key); c = c_e.get(key)
        if g is None or c is None:
            continue
        if abs(g) < 1e-9:
            ok = abs(c) < 1e-6
        else:
            ok = abs(g - c) / abs(g) <= METRIC_RTOL
        if not ok:
            fails.append(f"ensemble.{key} sapması: golden={g} current={c} (>{METRIC_RTOL:.0%})")

    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true', help='Altın-dosyayı yeniden üret')
    args = ap.parse_args()

    print("[REGRESYON] Snapshot üretiliyor (5-model ensemble, seed sabit)...")
    current = _compute_snapshot()
    print(f"  Modeller: {current['ensemble']['models_trained']}")
    print(f"  Feature: {current['feature']['n_rows']}x{current['feature']['n_features']}")
    print(f"  Test MAPE: {current['ensemble']['test_mape']} | "
          f"Dir: {current['ensemble']['predicted_direction']}")

    if args.update or not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=2)
        action = "güncellendi" if args.update else "ilk kez oluşturuldu"
        print(f"[REGRESYON] Altın-dosya {action}: {GOLDEN_PATH}")
        return 0

    with open(GOLDEN_PATH, 'r', encoding='utf-8') as f:
        golden = json.load(f)

    fails = _compare(golden, current)
    if fails:
        print("\n[REGRESYON] BASARISIZ (X) - davranis degisti:")
        for msg in fails:
            print(f"  - {msg}")
        print("\nBu beklenen bir değişiklikse: python tests/test_prediction_regression.py --update")
        return 1

    print("\n[REGRESYON] GECTI (OK) - davranis korunmus.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
