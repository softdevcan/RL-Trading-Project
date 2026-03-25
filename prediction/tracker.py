"""
Tahmin Takip Sistemi
Tahminleri JSON olarak saklar, gerçek fiyatlarla karşılaştırır.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from filelock import FileLock

from prediction.evaluator import PredictionEvaluator

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = os.path.join('data', 'predictions')


class PredictionTracker:
    """Tahmin depolama ve performans takip sistemi.

    Her sembol için ayrı JSON dosyası tutar.
    Dosya formatı:
    {
        "2026-03-25": {
            "daily": { ...tahmin dict... },
            "weekly": { ...tahmin dict... }
        }
    }
    """

    def __init__(self, predictions_dir: str = PREDICTIONS_DIR):
        self.predictions_dir = predictions_dir
        os.makedirs(self.predictions_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dosya yolları
    # ------------------------------------------------------------------

    def _filepath(self, symbol: str) -> str:
        safe = symbol.replace('/', '_').replace('=', '_').replace('.', '_')
        return os.path.join(self.predictions_dir, f'{safe}_predictions.json')

    # ------------------------------------------------------------------
    # Okuma / Yazma
    # ------------------------------------------------------------------

    def _load(self, symbol: str) -> Dict:
        path = self._filepath(symbol)
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save(self, symbol: str, data: Dict):
        path = self._filepath(symbol)
        lock = FileLock(path + '.lock')
        with lock:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Tahmin kaydetme
    # ------------------------------------------------------------------

    def store_prediction(self, prediction: Dict[str, Any]):
        """Tahmini JSON dosyasına yaz.

        prediction dict formatı: predict_next() çıktısı
        """
        symbol = prediction['symbol']
        horizon = prediction['horizon']
        pred_date = prediction['prediction_date']

        data = self._load(symbol)

        if pred_date not in data:
            data[pred_date] = {}

        data[pred_date][horizon] = {
            'predicted_close': prediction['predicted_close'],
            'predicted_direction': prediction['predicted_direction'],
            'predicted_change_pct': prediction['predicted_change_pct'],
            'confidence': prediction['confidence'],
            'current_close': prediction['current_close'],
            'made_at': prediction['made_at'],
            'actual_close': None,
            'error_pct': None,
            'direction_correct': None,
        }

        self._save(symbol, data)
        logger.info(f"  Tahmin kaydedildi: {symbol} {horizon} {pred_date}")

    # ------------------------------------------------------------------
    # Değerlendirme
    # ------------------------------------------------------------------

    def evaluate_prediction(
        self, symbol: str, pred_date: str, actual_close: float,
        horizon: str = 'daily'
    ) -> Optional[Dict[str, Any]]:
        """Belirtilen tarih/sembol tahminine gerçek fiyatı yaz ve hata hesapla."""
        data = self._load(symbol)

        if pred_date not in data or horizon not in data[pred_date]:
            logger.warning(f"Tahmin bulunamadı: {symbol} {horizon} {pred_date}")
            return None

        entry = data[pred_date][horizon]
        predicted = entry['predicted_close']
        current = entry.get('current_close', predicted)

        error_pct = (predicted - actual_close) / actual_close * 100 if actual_close != 0 else None
        pred_dir = predicted > current
        actual_dir = actual_close > current
        direction_correct = pred_dir == actual_dir

        entry['actual_close'] = round(float(actual_close), 4)
        entry['error_pct'] = round(float(error_pct), 2) if error_pct is not None else None
        entry['direction_correct'] = direction_correct
        entry['evaluated_at'] = datetime.now().isoformat()

        data[pred_date][horizon] = entry
        self._save(symbol, data)

        logger.info(
            f"  {symbol} {horizon} {pred_date}: "
            f"tahmin={predicted:.4f}, gerçek={actual_close:.4f}, "
            f"hata=%{error_pct:.2f}, yön={'doğru' if direction_correct else 'yanlış'}"
        )

        return entry

    def evaluate_pending(self, symbols: Optional[List[str]] = None):
        """Gerçek fiyatı bilinmeyen bekleyen tahminleri değerlendir.

        Bugünden önceki tarihli tahminler için yfinance'dan fiyat çeker.
        """
        import yfinance as yf

        if symbols is None:
            symbols = self.list_symbols()

        today = datetime.now().date()
        evaluated_count = 0

        for symbol in symbols:
            data = self._load(symbol)

            for pred_date, horizons in data.items():
                try:
                    pred_dt = datetime.strptime(pred_date, '%Y-%m-%d').date()
                except ValueError:
                    continue

                if pred_dt >= today:
                    continue

                for horizon, entry in horizons.items():
                    if entry.get('actual_close') is not None:
                        continue

                    try:
                        yf_sym = symbol if symbol not in ('GOLD_GRAM_TRY',) else 'GC=F'
                        ticker = yf.Ticker(yf_sym)
                        end = (pred_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                        hist = ticker.history(start=pred_date, end=end)
                        if hist.empty:
                            continue

                        actual = float(hist['Close'].iloc[-1])
                        self.evaluate_prediction(symbol, pred_date, actual, horizon)
                        evaluated_count += 1

                    except Exception as exc:
                        logger.warning(
                            f"  {symbol} {horizon} {pred_date} değerlendirilemedi: {exc}"
                        )

        logger.info(f"Toplam {evaluated_count} tahmin değerlendirildi")
        return evaluated_count

    # ------------------------------------------------------------------
    # Performans
    # ------------------------------------------------------------------

    def get_performance_history(
        self, symbol: str, horizon: str = 'daily', days: int = 30
    ) -> List[Dict[str, Any]]:
        """Son N gün için değerlendirilmiş tahmin geçmişini döndür."""
        data = self._load(symbol)

        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        history = []

        for pred_date, horizons in sorted(data.items()):
            if pred_date < cutoff:
                continue
            entry = horizons.get(horizon)
            if entry is None:
                continue
            history.append({'prediction_date': pred_date, **entry})

        return history

    def get_rolling_metrics(
        self, symbol: str, horizon: str = 'daily', window: int = 30
    ) -> Dict[str, Any]:
        """Sembol için rolling performans metrikleri döndür."""
        history = self.get_performance_history(symbol, horizon, days=window * 2)
        metrics = PredictionEvaluator.compute_rolling_metrics(history, window=window)
        return {'symbol': symbol, 'horizon': horizon, **metrics}

    def get_summary(self) -> List[Dict[str, Any]]:
        """Tüm semboller için özet performans tablosu."""
        summary = []
        for symbol in self.list_symbols():
            for horizon in ('daily', 'weekly'):
                metrics = self.get_rolling_metrics(symbol, horizon)
                if metrics.get('n_predictions', 0) > 0:
                    summary.append(metrics)
        return summary

    def list_symbols(self) -> List[str]:
        """Kayıtlı sembol listesi."""
        symbols = []
        for fname in os.listdir(self.predictions_dir):
            if fname.endswith('_predictions.json'):
                safe = fname[:-len('_predictions.json')]
                # ters dönüşüm: _IS → .IS, GC_F → GC=F, USDTRY_X → USDTRY=X
                symbol = safe.replace('_IS', '.IS').replace('GC_F', 'GC=F').replace('USDTRY_X', 'USDTRY=X')
                symbols.append(symbol)
        return symbols

    def get_prediction_history(self, symbol: str) -> Dict[str, Any]:
        """Ham tahmin geçmişini döndür."""
        return self._load(symbol)
