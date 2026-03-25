"""
Tahmin Servisi
Model eğitim, tahmin üretme ve performans takibi için servis katmanı
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


def _fetch_symbol_data(symbol: str, start_date: str) -> pd.DataFrame:
    """Sembol için OHLCV + teknik indikatör verisi çek."""
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    from data.gold_fetcher import BORSAPY_SYMBOLS, YFINANCE_SYMBOLS, OUTPUT_SYMBOL

    # Gold sembollerini tanı: borsapy/yfinance output sembol adları + ham semboller
    _gold_output = set(OUTPUT_SYMBOL.values())  # GOLD_GRAM_TRY, GC=F, USDTRY=X, EURTRY=X vb.
    _gold_raw    = set(BORSAPY_SYMBOLS.values()) | set(YFINANCE_SYMBOLS.values())
    is_gold = symbol in _gold_output or symbol in _gold_raw

    if is_gold:
        from data.gold_fetcher import GoldFetcher, METRIC_INFO, OUTPUT_SYMBOL as _OUT

        # Hangi metrik bu output sembolüne karşılık gelir?
        metric = next(
            (m for m, out in _OUT.items() if out == symbol),
            None,
        )
        # Metrik bulunamadıysa tüm metrikleri çek ve sembol indexinden al
        metrics = [metric] if metric else None

        # Kaynak seçimi: borsapy desteklemiyorsa yfinance kullan
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

    # Tek sembol için DataFrame çıkart
    available = df_multi.index.get_level_values('symbol').unique()
    if symbol in available:
        df = df_multi.xs(symbol, level='symbol').copy()
    else:
        raise ValueError(f"{symbol} verisi bulunamadı. Mevcut: {list(available)}")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = _clean_ohlcv(df)
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisi için temel temizleme.

    - Duplicate tarihler kaldırılır (en son değer tutulur)
    - Sıfır veya negatif close kaldırılır
    - OHLC aynı olan satırlar (veri boşluğu) forward fill ile düzeltilir
    - Aykırı değerler (30 günlük median'ın %50'sinden fazla sapma) kırpılır
    - Eksik değerler forward fill → backward fill
    """
    df = df[~df.index.duplicated(keep='last')].copy()
    df = df.sort_index()

    # Sıfır / negatif fiyat kaldır
    df = df[df['close'] > 0]

    # OHLC tamamen aynı olan satırlar → muhtemel veri boşluğu, close ffill ile doldur
    flat_mask = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
    if flat_mask.any():
        logger.debug(f"  {flat_mask.sum()} düz (flat) satır bulundu, forward fill uygulanıyor")
        df.loc[flat_mask, ['open', 'high', 'low', 'close']] = None

    # Aykırı değer kırpma: rolling median ± %50
    rolling_med = df['close'].rolling(30, min_periods=5).median()
    lower = rolling_med * 0.5
    upper = rolling_med * 1.5
    outlier_mask = (df['close'] < lower) | (df['close'] > upper)
    if outlier_mask.any():
        logger.warning(f"  {outlier_mask.sum()} aykırı close değeri kırpılıyor")
        df.loc[outlier_mask, 'close'] = None

    # Forward fill → backward fill
    df = df.ffill().bfill()

    return df


class PredictionService:
    """Tahmin pipeline'ı servis katmanı."""

    def train_model(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01', test_ratio: float = 0.2
    ) -> Dict[str, Any]:
        """Sembol ve horizon için XGBoost modeli eğit."""
        from prediction.models import PricePredictor

        logger.info(f"Model eğitimi başlıyor: {symbol} {horizon}")

        df = _fetch_symbol_data(symbol, start_date)
        predictor = PricePredictor(horizon)
        result = predictor.train(df, symbol, test_ratio=test_ratio)

        logger.info(f"  Eğitim tamamlandı: {symbol} {horizon}")
        return result

    def predict(
        self, symbols: List[str], horizon: str = 'daily',
        save: bool = True
    ) -> List[Dict[str, Any]]:
        """Sembol listesi için tahmin üret ve tracker'a kaydet."""
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
                    logger.warning(f"  {symbol} için eğitilmiş model yok, atlanıyor")
                    continue

                df = _fetch_symbol_data(symbol, start)
                pred = predictor.predict_next(df, symbol)
                predictions.append(pred)

                if save:
                    tracker.store_prediction(pred)

            except Exception as exc:
                logger.error(f"  {symbol} tahmini başarısız: {exc}")

        return predictions

    def evaluate_pending(self, symbols: Optional[List[str]] = None) -> int:
        """Bekleyen tahminleri değerlendir."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.evaluate_pending(symbols)

    def get_performance(
        self, symbol: str, horizon: str = 'daily', window: int = 30
    ) -> Dict[str, Any]:
        """Sembol için performans metrikleri."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.get_rolling_metrics(symbol, horizon, window)

    def get_summary(self) -> List[Dict[str, Any]]:
        """Tüm semboller için özet."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.get_summary()

    def list_trained_models(self) -> List[Dict[str, Any]]:
        """Eğitilmiş modelleri listele."""
        from prediction.models import PricePredictor
        models = []
        for horizon in ('daily', 'weekly'):
            p = PricePredictor(horizon)
            models.extend(p.list_trained_models())
        return models

    def get_prediction_history(
        self, symbol: str, horizon: str = 'daily'
    ) -> Dict[str, Any]:
        """Sembol için tahmin geçmişi."""
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
        """Tahmin vs gerçek chart verisi."""
        from prediction.tracker import PredictionTracker
        from prediction.evaluator import PredictionEvaluator
        tracker = PredictionTracker()
        history = tracker.get_performance_history(symbol, horizon, days)
        chart = PredictionEvaluator.build_prediction_chart_data(history)
        trend = PredictionEvaluator.build_accuracy_trend_data(history)
        return {'chart': chart, 'trend': trend}
