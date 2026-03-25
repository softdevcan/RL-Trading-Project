"""
Test: Prediction Module
XGBoost fiyat tahmin modülü testleri

Kullanım:
    python tests/test_prediction.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Test için sentetik veri üret
def _make_synthetic_df(n: int = 500) -> pd.DataFrame:
    """Basit sentetik OHLCV + indikatör DataFrame üret."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    close = 100 * (1 + np.random.randn(n) * 0.01).cumprod()

    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(n) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n)) * 0.008),
        'low': close * (1 - np.abs(np.random.randn(n)) * 0.008),
        'close': close,
        'volume': np.random.randint(100_000, 1_000_000, n).astype(float),
        'macd': np.random.randn(n) * 0.5,
        'rsi': 50 + np.random.randn(n) * 10,
        'cci': np.random.randn(n) * 50,
        'adx': 25 + np.abs(np.random.randn(n)) * 5,
        'turbulence': np.abs(np.random.randn(n)),
    }, index=dates)

    return df


def test_feature_engineer():
    """PredictionFeatureEngineer testi"""
    print("\n[TEST 1] PredictionFeatureEngineer...")
    from prediction.feature_engineer import PredictionFeatureEngineer

    df = _make_synthetic_df(300)

    for horizon in ('daily', 'weekly'):
        eng = PredictionFeatureEngineer(horizon)
        feat_df = eng.build_features(df, 'TEST')
        cols = eng.get_feature_columns(feat_df)

        assert 'target' in feat_df.columns, "target kolonu yok"
        assert len(cols) > 5, f"Az özellik: {len(cols)}"
        assert feat_df['target'].notna().all(), "target'te NaN var"

        print(f"  {horizon}: {len(feat_df)} satır, {len(cols)} özellik")

    print("  OK!")


def test_price_predictor_train():
    """PricePredictor eğitim testi"""
    print("\n[TEST 2] PricePredictor.train()...")
    try:
        import xgboost  # noqa
    except ImportError:
        print("  ATLA: xgboost kurulu değil")
        return None

    from prediction.models import PricePredictor

    df = _make_synthetic_df(400)
    predictor = PricePredictor('daily')

    result = predictor.train(df, 'SYNTHETIC_TEST', test_ratio=0.2)

    assert 'train_metrics' in result, "train_metrics yok"
    assert 'test_metrics' in result, "test_metrics yok"
    assert result['n_train'] > 0, "n_train=0"
    assert result['n_test'] > 0, "n_test=0"
    assert result['test_metrics']['mape'] >= 0, "negatif MAPE"
    assert 0 <= result['test_metrics']['direction_accuracy'] <= 100, "direction_acc aralık dışı"

    print(f"  Train MAPE: {result['train_metrics']['mape']:.2f}%")
    print(f"  Test MAPE : {result['test_metrics']['mape']:.2f}%")
    print(f"  Dir Acc   : {result['test_metrics']['direction_accuracy']:.1f}%")
    print(f"  Model path: {result['model_path']}")
    print("  OK!")
    return predictor


def test_price_predictor_predict(predictor):
    """PricePredictor tahmin testi"""
    print("\n[TEST 3] PricePredictor.predict_next()...")
    if predictor is None:
        print("  ATLA: model yok")
        return

    df = _make_synthetic_df(400)
    pred = predictor.predict_next(df, 'SYNTHETIC_TEST')

    required = ['symbol', 'prediction_date', 'horizon', 'predicted_close',
                'predicted_direction', 'predicted_change_pct', 'confidence',
                'current_close', 'made_at']
    for key in required:
        assert key in pred, f"Eksik alan: {key}"

    assert pred['predicted_close'] > 0, "negatif tahmin"
    assert pred['predicted_direction'] in ('UP', 'DOWN'), "geçersiz direction"
    assert 0 <= pred['confidence'] <= 1, "güven aralık dışı"

    print(f"  Tahmin   : {pred['predicted_close']:.4f}")
    print(f"  Yön      : {pred['predicted_direction']}")
    print(f"  Değişim  : {pred['predicted_change_pct']:+.2f}%")
    print(f"  Güven    : {pred['confidence']:.3f}")
    print("  OK!")
    return pred


def test_tracker(pred):
    """PredictionTracker testi"""
    print("\n[TEST 4] PredictionTracker...")
    import tempfile
    from prediction.tracker import PredictionTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = PredictionTracker(predictions_dir=tmpdir)

        if pred is None:
            # Sahte tahmin oluştur
            pred = {
                'symbol': 'FAKE.IS',
                'horizon': 'daily',
                'prediction_date': '2026-04-01',
                'predicted_close': 12.5,
                'predicted_direction': 'UP',
                'predicted_change_pct': 1.5,
                'confidence': 0.7,
                'current_close': 12.3,
                'made_at': '2026-03-25T18:00:00',
            }

        tracker.store_prediction(pred)

        # Geri oku
        history = tracker.get_prediction_history(pred['symbol'])
        assert len(history) > 0, "Tahmin kaydedilmedi"

        # Değerlendirme
        result = tracker.evaluate_prediction(
            pred['symbol'],
            pred['prediction_date'],
            actual_close=pred['predicted_close'] * 1.01,  # +1% actual
            horizon=pred['horizon']
        )
        assert result is not None, "Değerlendirme başarısız"
        assert result['actual_close'] is not None, "actual_close kaydedilmedi"
        assert result['error_pct'] is not None, "error_pct hesaplanmadı"

        # Performans metrikleri
        metrics = tracker.get_rolling_metrics(
            pred['symbol'], pred['horizon'], window=30
        )
        assert 'n_predictions' in metrics, "n_predictions yok"

        print(f"  Kayıt  : OK")
        print(f"  Değer. : hata={result['error_pct']:.2f}%")
        print(f"  Metrik : {metrics}")
        print("  OK!")


def test_evaluator():
    """PredictionEvaluator testi"""
    print("\n[TEST 5] PredictionEvaluator...")
    from prediction.evaluator import PredictionEvaluator

    preds = [
        {'symbol': 'A', 'predicted_close': 110, 'current_close': 100, 'horizon': 'daily',
         'predicted_direction': 'UP', 'prediction_date': '2026-01-02',
         'made_at': '2026-01-01T18:00:00', 'predicted_change_pct': 10},
        {'symbol': 'B', 'predicted_close': 95, 'current_close': 100, 'horizon': 'daily',
         'predicted_direction': 'DOWN', 'prediction_date': '2026-01-02',
         'made_at': '2026-01-01T18:00:00', 'predicted_change_pct': -5},
    ]

    actuals = {'A': 112, 'B': 93}
    results = PredictionEvaluator.evaluate_batch(preds, actuals)

    assert len(results) == 2, "Sonuç sayısı hatalı"
    for r in results:
        assert r['actual_close'] is not None, "actual_close yok"
        assert r['error_pct'] is not None, "error_pct yok"
        assert isinstance(r['direction_correct'], bool), "direction_correct hatalı"

    print(f"  A: tahmin=110, gercek=112, hata={results[0]['error_pct']:.2f}%")
    print(f"  B: tahmin=95, gercek=93, hata={results[1]['error_pct']:.2f}%")
    print("  OK!")


def test_walk_forward_cv():
    """WalkForwardTrainer CV testi"""
    print("\n[TEST 6] WalkForwardTrainer CV...")
    try:
        import xgboost  # noqa
    except ImportError:
        print("  ATLA: xgboost kurulu değil")
        return

    from prediction.trainer import WalkForwardTrainer

    df = _make_synthetic_df(500)
    trainer = WalkForwardTrainer(horizon='daily', n_splits=3)
    cv_result = trainer.cross_validate(df, 'SYNTHETIC_TEST')

    assert 'avg_mape' in cv_result, "avg_mape yok"
    assert cv_result['n_folds'] == 3, f"fold sayısı: {cv_result['n_folds']}"

    print(f"  CV MAPE: {cv_result['avg_mape']:.2f}%")
    print(f"  CV Dir : {cv_result['avg_direction_accuracy']:.1f}%")
    print("  OK!")


if __name__ == '__main__':
    print("=" * 60)
    print("PREDICTION MODULE TESTLERİ")
    print("=" * 60)

    failed = []

    tests = [
        ('FeatureEngineer', test_feature_engineer),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  BAŞARISIZ [{name}]: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    # Bağımlı testler
    try:
        predictor = test_price_predictor_train()
        pred = test_price_predictor_predict(predictor)
        test_tracker(pred)
        test_evaluator()
        test_walk_forward_cv()
    except Exception as e:
        print(f"  BAŞARISIZ: {e}")
        import traceback
        traceback.print_exc()
        failed.append('prediction_chain')

    print("\n" + "=" * 60)
    if failed:
        print(f"BAŞARISIZ TESTLER: {failed}")
        sys.exit(1)
    else:
        print("TÜM TESTLER BAŞARILI!")
    print("=" * 60)
