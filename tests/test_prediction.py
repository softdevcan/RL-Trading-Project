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

    # Faz 9 Y23: ensemble sonuc semasi (eski tek-model anahtarlari train_metrics/model_path yok)
    tm = result.get('ensemble_test_metrics', {})
    assert result['n_train'] > 0, "n_train=0"
    assert result['n_test'] > 0, "n_test=0"
    assert result.get('n_models', 0) >= 1, f"model egitilmedi: {result.get('failed_models')}"
    assert tm.get('mape', -1) >= 0, "negatif MAPE"
    assert 0 <= tm.get('direction_accuracy', -1) <= 100, "direction_acc aralık dışı"

    print(f"  Modeller  : {result.get('models_trained')} (hata: {result.get('failed_models')})")
    print(f"  Test MAPE : {tm['mape']:.2f}%")
    print(f"  Dir Acc   : {tm['direction_accuracy']:.1f}%")
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


def test_tracker_target_date(pred, df_last):
    """Faz 9 Y29: hedef tarih veriden (feature_date + h seans), duvar saatinden degil; tracker
    gerceklesen kapanisi feature_date'ten sonraki h. GERCEK seanstan alir; gecersiz kayitlar
    (Y28 oncesi / stale_features) degerlendirilmez ve metriğe girmez. Mutasyon uretim yolundan."""
    print("\n[TEST 7] Y29 tracker hedef tarihi...")
    import tempfile
    from datetime import datetime, timedelta
    import prediction.tracker as T
    from prediction.tracker import PredictionTracker
    from prediction.tracker_rules import horizon_sessions

    # (a) predict_next: veri 2021'de bitiyor, saat 2026 -> hedef veriden
    if pred is not None:
        exp = (pd.Timestamp(df_last) + pd.offsets.BDay(1)).strftime('%Y-%m-%d')
        old = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        assert pred['feature_date'] == str(df_last)[:10], pred.get('feature_date')
        assert pred['prediction_date'] == exp != old, (pred['prediction_date'], exp, old)
        print(f"  predict_next: feature_date={pred['feature_date']} -> prediction_date={exp} (eski kural {old})")
    fri = pd.Timestamp('2026-11-06')  # Cuma
    assert horizon_sessions('weekly') == 5 and (fri + pd.offsets.BDay(5)).strftime('%Y-%m-%d') == '2026-11-13'

    # (b) taslak kapanislar: 12.10 tatil (satir yok), 20.10 seans kapanmamis taslak (NaN)
    idx = pd.to_datetime(['2026-10-08', '2026-10-09', '2026-10-13', '2026-10-14', '2026-10-15',
                          '2026-10-16', '2026-10-19', '2026-10-20']).tz_localize('Europe/Istanbul')
    closes = pd.Series([99, 100, 103, 104, 105, 106, 107, np.nan], index=idx, dtype=float)

    def fake(symbol, start, end):
        d = closes.index.tz_localize(None)
        return closes[(d >= pd.Timestamp(start)) & (d < pd.Timestamp(end))]

    base = {'symbol': 'Y29.IS', 'predicted_direction': 'UP', 'predicted_change_pct': 1.0,
            'confidence': 0.6, 'current_close': 100.0, 'predicted_close': 101.0,
            'made_at': '2026-10-10T10:00:00'}
    recs = [
        dict(base, horizon='daily', prediction_date='2026-10-12', feature_date='2026-10-09', stale_features=False),
        dict(base, horizon='weekly', prediction_date='2026-10-16', feature_date='2026-10-09', stale_features=False),
        dict(base, symbol='OLD.IS', horizon='daily', prediction_date='2026-10-05'),            # Y28 oncesi
        dict(base, symbol='OL2.IS', horizon='daily', prediction_date='2026-10-07'),            # Y28 oncesi, bekleyen
        dict(base, symbol='STL.IS', horizon='daily', prediction_date='2026-10-06',
             feature_date='2026-10-02', stale_features=True),
    ]

    def run(mutate=False):
        orig_fetch, orig_rule = PredictionTracker._fetch_closes, T.actual_after_sessions
        PredictionTracker._fetch_closes = staticmethod(fake)
        if mutate:  # eski anlayis: takvimdeki tahmini tarihin (feature_date + h is gunu) kapanisi
            def calendar_rule(c, fd, h):
                d = (pd.Timestamp(fd) + pd.offsets.BDay(h)).strftime('%Y-%m-%d')
                m = c[c.index.tz_localize(None).strftime('%Y-%m-%d') == d].dropna()
                return (d, float(m.iloc[0])) if len(m) else None
            T.actual_after_sessions = calendar_rule
        try:
            tmp = tempfile.mkdtemp(prefix='rlt_y29_')
            tr = PredictionTracker(predictions_dir=tmp)
            for r in recs:
                tr.store_prediction(r)
            # Y28 oncesi, zaten degerlendirilmis eski kayit (metriğe girmemeli)
            tr.evaluate_prediction('OLD.IS', '2026-10-05', 99.0, 'daily')
            n = tr.evaluate_pending()
            h = tr.get_prediction_history('Y29.IS')
            return n, tr, h
        finally:
            PredictionTracker._fetch_closes, T.actual_after_sessions = orig_fetch, orig_rule

    n, tr, h = run()
    d, w = h['2026-10-12']['daily'], h['2026-10-16']['weekly']
    assert n == 2, n
    assert (d.get('actual_date'), d.get('actual_close')) == ('2026-10-13', 103.0), d   # tatil atlandi
    assert (w.get('actual_date'), w.get('actual_close')) == ('2026-10-19', 107.0), w   # 5. gercek seans
    assert tr.last_skipped_invalid == {'pre_y28_no_feature_date': 1, 'stale_features': 1}, tr.last_skipped_invalid
    assert tr.get_prediction_history('OL2.IS')['2026-10-07']['daily']['actual_close'] is None
    old = tr.get_prediction_history('OLD.IS')['2026-10-05']['daily']
    assert old['actual_close'] is not None                                      # eski kayit duruyor...
    assert tr.get_rolling_metrics('OLD.IS', 'daily')['n_predictions'] == 0      # ...ama metriğe girmiyor
    assert tr.get_performance_history('Y29.IS', 'daily', days=100000)[0]['actual_date'] == '2026-10-13'
    print(f"  gunluk 12.10 tatil -> 13.10 (103); haftalik 5. seans 19.10 (107); gecersiz atlanan {tr.last_skipped_invalid}")

    n_m, _, h_m = run(mutate=True)
    dm, wm = h_m['2026-10-12']['daily'], h_m['2026-10-16']['weekly']
    assert dm.get('actual_close') is None and wm.get('actual_date') == '2026-10-16', (dm, wm)
    print(f"  mutasyon (takvim tarihi): gunluk degerlendirilemedi, haftalik 16.10 = 4. seans -> kontroller kirilir")
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

    # Faz 9 Y23: guncel ozet semasi (model basina sonuc + en iyi model)
    assert cv_result['n_folds'] == 3, f"fold sayısı: {cv_result['n_folds']}"
    assert cv_result.get('model_results'), f"model sonucu yok: {cv_result.get('failed_folds')}"
    assert cv_result.get('feature_selection') == 'per_fold_train', "B8 protokol etiketi yok"
    best = cv_result['model_results'][cv_result['best_model']]

    print(f"  En iyi   : {cv_result['best_model']} ({len(cv_result['model_results'])} model, "
          f"basarisiz kat {cv_result['n_failed_folds']})")
    print(f"  CV MAPE  : {best['avg_mape']:.2f}%")
    print(f"  CV Dir   : {best['avg_direction_accuracy']:.1f}%")
    print("  OK!")


if __name__ == '__main__':
    # Faz 9 Y23: egitilen sentetik modeller gercek models/prediction dizinine yazilmasin
    import tempfile
    from app.auth import workspace as _ws
    _TMP_MODELS = tempfile.mkdtemp(prefix="rlt_pred_test_")
    _ws.prediction_models_dir = lambda *a, **k: _TMP_MODELS
    _ws.read_dirs = lambda kind, *a, **k: [_TMP_MODELS]
    _ws.results_dir = lambda *a, **k: _TMP_MODELS  # CV deney kaydi da gecici dizine

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
        test_tracker_target_date(pred, _make_synthetic_df(400).index[-1])
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
