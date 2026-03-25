"""
Tahmin Değerlendirici
Gerçek fiyatlarla tahminleri karşılaştırıp metrik üretir.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PredictionEvaluator:
    """Tahmin performans değerlendirici."""

    @staticmethod
    def evaluate_batch(
        predictions: List[Dict[str, Any]],
        actuals: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Tahmin listesini gerçek kapanış fiyatlarıyla değerlendir.

        Args:
            predictions: Tahmin dict listesi (predict_next çıktısı)
            actuals: {symbol: actual_close} dict'i

        Returns:
            Her tahmine hata metrikleri eklenerek döndürülür
        """
        results = []
        for pred in predictions:
            symbol = pred['symbol']
            actual = actuals.get(symbol)
            if actual is None:
                results.append({**pred, 'actual_close': None,
                                'error_pct': None, 'direction_correct': None})
                continue

            predicted = pred['predicted_close']
            error_pct = (predicted - actual) / actual * 100 if actual != 0 else None

            # Yön doğruluğu
            current = pred.get('current_close', actual)
            pred_dir = predicted > current
            actual_dir = actual > current
            direction_correct = pred_dir == actual_dir

            results.append({
                **pred,
                'actual_close': round(float(actual), 4),
                'error_pct': round(float(error_pct), 2) if error_pct is not None else None,
                'direction_correct': direction_correct,
            })

        return results

    @staticmethod
    def compute_rolling_metrics(
        history: List[Dict[str, Any]], window: int = 30
    ) -> Dict[str, float]:
        """Son `window` tahmin üzerinden rolling metrik hesapla.

        Args:
            history: Değerlendirilmiş tahmin geçmişi
            window: Rolling pencere boyutu

        Returns:
            mape, rmse, mae, direction_accuracy
        """
        evaluated = [
            h for h in history
            if h.get('actual_close') is not None and h.get('error_pct') is not None
        ]

        if not evaluated:
            return {
                'mape': None, 'rmse': None, 'mae': None,
                'direction_accuracy': None, 'n_predictions': 0
            }

        recent = evaluated[-window:]

        errors = [abs(r['error_pct']) for r in recent if r['error_pct'] is not None]
        mape = float(np.mean(errors)) if errors else None

        pred_arr = np.array([r['predicted_close'] for r in recent])
        actual_arr = np.array([r['actual_close'] for r in recent])
        rmse = float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_arr - actual_arr)))

        dir_correct = [r['direction_correct'] for r in recent
                       if r.get('direction_correct') is not None]
        dir_acc = float(np.mean(dir_correct) * 100) if dir_correct else None

        return {
            'mape': round(mape, 4) if mape is not None else None,
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'direction_accuracy': round(dir_acc, 2) if dir_acc is not None else None,
            'n_predictions': len(recent),
        }

    @staticmethod
    def build_prediction_chart_data(
        history: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Tahmin vs gerçek chart verisi oluştur.

        Returns:
            {dates, predicted, actual} listeler
        """
        evaluated = [
            h for h in history
            if h.get('actual_close') is not None
        ]
        evaluated = sorted(evaluated, key=lambda x: x.get('prediction_date', ''))

        return {
            'dates': [h['prediction_date'] for h in evaluated],
            'predicted': [h['predicted_close'] for h in evaluated],
            'actual': [h['actual_close'] for h in evaluated],
            'direction_correct': [h.get('direction_correct') for h in evaluated],
        }

    @staticmethod
    def build_accuracy_trend_data(
        history: List[Dict[str, Any]], window: int = 10
    ) -> Dict[str, List]:
        """Rolling MAPE ve yön doğruluk trendi oluştur."""
        evaluated = [
            h for h in history
            if h.get('actual_close') is not None and h.get('error_pct') is not None
        ]
        evaluated = sorted(evaluated, key=lambda x: x.get('prediction_date', ''))

        dates, mapes, dir_accs = [], [], []

        for i in range(window, len(evaluated) + 1):
            batch = evaluated[i - window:i]
            errors = [abs(r['error_pct']) for r in batch if r['error_pct'] is not None]
            mape = float(np.mean(errors)) if errors else None
            dir_vals = [r['direction_correct'] for r in batch
                        if r.get('direction_correct') is not None]
            dir_acc = float(np.mean(dir_vals) * 100) if dir_vals else None

            dates.append(evaluated[i - 1]['prediction_date'])
            mapes.append(round(mape, 2) if mape is not None else None)
            dir_accs.append(round(dir_acc, 2) if dir_acc is not None else None)

        return {'dates': dates, 'mape': mapes, 'direction_accuracy': dir_accs}
