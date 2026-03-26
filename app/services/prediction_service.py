"""
Gelismis Tahmin Servisi

Ensemble tabanli model egitim, tahmin uretme ve performans takibi.
Mevcut PredictionService API'sini korurken yeni ensemble mimarisini kullanir.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


def _fetch_symbol_data(symbol: str, start_date: str) -> pd.DataFrame:
    """Sembol icin OHLCV + teknik indikator verisi cek."""
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    from data.gold_fetcher import BORSAPY_SYMBOLS, YFINANCE_SYMBOLS, OUTPUT_SYMBOL

    _gold_output = set(OUTPUT_SYMBOL.values())
    _gold_raw = set(BORSAPY_SYMBOLS.values()) | set(YFINANCE_SYMBOLS.values())
    is_gold = symbol in _gold_output or symbol in _gold_raw

    if is_gold:
        from data.gold_fetcher import GoldFetcher, METRIC_INFO, OUTPUT_SYMBOL as _OUT

        metric = next(
            (m for m, out in _OUT.items() if out == symbol),
            None,
        )
        metrics = [metric] if metric else None

        if metric and not METRIC_INFO[metric]["borsapy"]:
            source = "yfinance"
        else:
            source = "borsapy"

        fetcher = GoldFetcher(source=source, metrics=metrics, start_date=start_date)
        df_multi = fetcher.fetch_all()
    else:
        from data.data_fetcher import DataFetcher
        fetcher = DataFetcher(start_date=start_date)
        df_multi = fetcher.fetch_stock_data([symbol], save=False)

    df_multi = add_indicators_to_multi_symbol_df(df_multi)

    available = df_multi.index.get_level_values('symbol').unique()
    if symbol in available:
        df = df_multi.xs(symbol, level='symbol').copy()
    else:
        raise ValueError(f"{symbol} verisi bulunamadi. Mevcut: {list(available)}")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = _clean_ohlcv(df)
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisi icin temel temizleme."""
    df = df[~df.index.duplicated(keep='last')].copy()
    df = df.sort_index()

    df = df[df['close'] > 0]

    flat_mask = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
    if flat_mask.any():
        logger.debug(f"  {flat_mask.sum()} duz (flat) satir bulundu, forward fill uygulaniyor")
        df.loc[flat_mask, ['open', 'high', 'low', 'close']] = None

    rolling_med = df['close'].rolling(30, min_periods=5).median()
    lower = rolling_med * 0.5
    upper = rolling_med * 1.5
    outlier_mask = (df['close'] < lower) | (df['close'] > upper)
    if outlier_mask.any():
        logger.warning(f"  {outlier_mask.sum()} aykiri close degeri kirpiliyor")
        df.loc[outlier_mask, 'close'] = None

    df = df.ffill().bfill()

    return df


def _fetch_macro_data(start_date: str = '2018-01-01') -> Optional[pd.DataFrame]:
    """Makroekonomik verileri cek veya yukle (opsiyonel)."""
    try:
        from data.macro_fetcher import MacroDataFetcher
        fetcher = MacroDataFetcher(start_date=start_date)
        try:
            return fetcher.load_data()
        except FileNotFoundError:
            return fetcher.fetch_macro_data(save=True)
    except Exception as exc:
        logger.warning(f"Makro veri yuklenemedi: {exc}")
        return None


def _fetch_fundamental_data(symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Fundamental verileri cek veya yukle (opsiyonel)."""
    try:
        from data.fundamental_fetcher import FundamentalDataFetcher
        fetcher = FundamentalDataFetcher()
        try:
            return fetcher.load_data()
        except FileNotFoundError:
            if symbols:
                return fetcher.fetch_fundamental_data(symbols, save=True)
            return None
    except Exception as exc:
        logger.warning(f"Fundamental veri yuklenemedi: {exc}")
        return None


def _fetch_cross_asset_data(start_date: str = '2018-01-01') -> Optional[pd.DataFrame]:
    """BIST-100 ve USD/TRY capraz varlik verilerini cek."""
    try:
        import yfinance as yf

        cross_data = {}
        for name, symbol in [('bist100', 'XU100.IS'), ('usd_try', 'TRY=X'), ('eur_try', 'EURTRY=X')]:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date)
            if not hist.empty:
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                cross_data[name] = hist['Close']

        if cross_data:
            return pd.DataFrame(cross_data).ffill().bfill()
        return None
    except Exception as exc:
        logger.warning(f"Capraz varlik verisi yuklenemedi: {exc}")
        return None


class PredictionService:
    """Gelismis tahmin pipeline'i servis katmani.

    Ensemble (XGBoost + LightGBM + CatBoost + BiLSTM + TFT) tabanli tahmin.
    Eski API'yi koruyarak geriye uyumlu calisir.
    """

    def train_model(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01', test_ratio: float = 0.2,
        optimize: bool = False, n_hpo_trials: int = 30,
    ) -> Dict[str, Any]:
        """Sembol icin ensemble modeli egit."""
        from prediction.trainer import WalkForwardTrainer

        logger.info(f"Ensemble egitimi basliyor: {symbol} {horizon}")

        df = _fetch_symbol_data(symbol, start_date)
        macro_df = _fetch_macro_data(start_date)
        fundamental_df = _fetch_fundamental_data([symbol])
        cross_asset_df = _fetch_cross_asset_data(start_date)

        trainer = WalkForwardTrainer(horizon=horizon)
        result = trainer.train_final_models(
            df, symbol,
            test_ratio=test_ratio,
            optimize_first=optimize,
            n_hpo_trials=n_hpo_trials,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )

        logger.info(f"  Egitim tamamlandi: {symbol} {horizon}")

        # Geriye uyumluluk: eski API formatinda sonuc dondur
        test_metrics = result.get('ensemble_test_metrics', {})
        return {
            'symbol': result.get('symbol', symbol),
            'horizon': result.get('horizon', horizon),
            'n_train': result.get('n_train', 0),
            'n_test': result.get('n_test', 0),
            'n_features': result.get('n_features', 0),
            'train_metrics': test_metrics,
            'test_metrics': test_metrics,
            'model_path': f'models/prediction/{symbol}_{horizon}_ensemble',
            'trained_at': result.get('trained_at', datetime.now().isoformat()),
            'n_models': result.get('n_models', 0),
            'models_trained': result.get('models_trained', []),
            'model_results': result.get('model_results', {}),
            'ensemble_test_metrics': test_metrics,
            'training_time_seconds': result.get('training_time_seconds'),
        }

    def predict(
        self, symbols: List[str], horizon: str = 'daily',
        save: bool = True
    ) -> List[Dict[str, Any]]:
        """Sembol listesi icin tahmin uret ve tracker'a kaydet."""
        from prediction.models import PricePredictor
        from prediction.tracker import PredictionTracker

        predictor = PricePredictor(horizon)
        tracker = PredictionTracker()

        predictions = []
        today = datetime.now()
        start = (today - timedelta(days=120)).strftime('%Y-%m-%d')

        for symbol in symbols:
            try:
                if not predictor.is_trained(symbol):
                    logger.warning(f"  {symbol} icin egitilmis model yok, atlaniyor")
                    continue

                df = _fetch_symbol_data(symbol, start)
                pred = predictor.predict_next(df, symbol)
                predictions.append(pred)

                if save:
                    tracker.store_prediction(pred)

            except Exception as exc:
                logger.error(f"  {symbol} tahmini basarisiz: {exc}")

        return predictions

    def cross_validate(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01',
        model_types: Optional[List[str]] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Walk-forward cross-validation ile model degerlendirme."""
        from prediction.trainer import WalkForwardTrainer

        df = _fetch_symbol_data(symbol, start_date)
        macro_df = _fetch_macro_data(start_date)
        fundamental_df = _fetch_fundamental_data([symbol])
        cross_asset_df = _fetch_cross_asset_data(start_date)

        trainer = WalkForwardTrainer(
            horizon=horizon,
            n_splits=n_splits,
        )
        return trainer.cross_validate(
            df, symbol,
            model_types=model_types,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )

    def optimize_hyperparameters(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01',
        model_types: Optional[List[str]] = None,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Tum modeller icin hiperparametre optimizasyonu."""
        from prediction.hyperopt import PredictionHyperOptimizer
        from prediction.feature_engineer import PredictionFeatureEngineer
        import numpy as np

        df = _fetch_symbol_data(symbol, start_date)
        macro_df = _fetch_macro_data(start_date)
        fundamental_df = _fetch_fundamental_data([symbol])
        cross_asset_df = _fetch_cross_asset_data(start_date)

        fe = PredictionFeatureEngineer(horizon)
        feat_df = fe.build_features(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )
        feature_cols = fe.get_feature_columns(feat_df)
        X = feat_df[feature_cols].values.astype(np.float32)
        y = feat_df['target_price'].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        optimizer = PredictionHyperOptimizer(n_trials=n_trials)
        return optimizer.optimize_all_models(X, y, horizon, model_types)

    def evaluate_pending(self, symbols: Optional[List[str]] = None) -> int:
        """Bekleyen tahminleri degerlendir."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.evaluate_pending(symbols)

    def get_performance(
        self, symbol: str, horizon: str = 'daily', window: int = 30
    ) -> Dict[str, Any]:
        """Sembol icin performans metrikleri."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.get_rolling_metrics(symbol, horizon, window)

    def get_summary(self) -> List[Dict[str, Any]]:
        """Tum semboller icin ozet."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.get_summary()

    def list_trained_models(self) -> List[Dict[str, Any]]:
        """Egitilmis modelleri listele."""
        from prediction.models import PricePredictor
        models = []
        for horizon in ('daily', 'weekly'):
            p = PricePredictor(horizon)
            models.extend(p.list_trained_models())
        return models

    def get_prediction_history(
        self, symbol: str, horizon: str = 'daily'
    ) -> Dict[str, Any]:
        """Sembol icin tahmin gecmisi."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        raw = tracker.get_prediction_history(symbol)

        history = []
        for pred_date, horizons in sorted(raw.items()):
            entry = horizons.get(horizon)
            if entry:
                history.append({'prediction_date': pred_date, **entry})

        return {
            'symbol': symbol,
            'horizon': horizon,
            'history': history,
            'total': len(history),
        }

    def get_chart_data(
        self, symbol: str, horizon: str = 'daily', days: int = 30
    ) -> Dict[str, Any]:
        """Tahmin vs gercek chart verisi."""
        from prediction.tracker import PredictionTracker
        from prediction.evaluator import PredictionEvaluator
        tracker = PredictionTracker()
        history = tracker.get_performance_history(symbol, horizon, days)
        chart = PredictionEvaluator.build_prediction_chart_data(history)
        trend = PredictionEvaluator.build_accuracy_trend_data(history)
        return {'chart': chart, 'trend': trend}
