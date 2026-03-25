"""
Walk-Forward Eğitim Modülü
Zaman serisi uyumlu cross-validation ile XGBoost modeli eğitir.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from prediction.feature_engineer import PredictionFeatureEngineer
from prediction.models import PricePredictor

logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """Zaman serisi walk-forward cross-validation."""

    def __init__(self, horizon: str = 'daily', n_splits: int = 5):
        """
        Args:
            horizon: 'daily' veya 'weekly'
            n_splits: Walk-forward fold sayısı
        """
        self.horizon = horizon
        self.n_splits = n_splits
        self.feature_engineer = PredictionFeatureEngineer(horizon)

    def cross_validate(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Walk-forward CV uygula.

        Args:
            df: Ham OHLCV + indikatör verisi
            symbol: Sembol

        Returns:
            Her fold için metrikler ve ortalama performans
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost kurulu değil.")

        feat_df = self.feature_engineer.build_features(df, symbol)
        feature_cols = self.feature_engineer.get_feature_columns(feat_df)

        X = feat_df[feature_cols].values
        y = feat_df['target'].values

        n = len(X)
        fold_size = n // (self.n_splits + 1)

        fold_results: List[Dict[str, float]] = []

        for fold in range(self.n_splits):
            train_end = fold_size * (fold + 1)
            test_end = min(train_end + fold_size, n)

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(X_test) == 0:
                continue

            params = {k: v for k, v in PricePredictor.XGB_PARAMS.items()
                      if k not in ('early_stopping_rounds', 'eval_metric')}
            model = XGBRegressor(
                **params,
                early_stopping_rounds=PricePredictor.XGB_PARAMS['early_stopping_rounds'],
                eval_metric=PricePredictor.XGB_PARAMS['eval_metric'],
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            preds = model.predict(X_test)
            metrics = PricePredictor._compute_metrics(y_test, preds)
            metrics['fold'] = fold + 1
            fold_results.append(metrics)

            logger.info(
                f"  Fold {fold + 1}/{self.n_splits}: "
                f"MAPE={metrics['mape']:.2f}% "
                f"DirAcc={metrics['direction_accuracy']:.1f}%"
            )

        if not fold_results:
            return {'error': 'Yeterli veri yok'}

        avg_mape = float(np.mean([r['mape'] for r in fold_results]))
        avg_dir = float(np.mean([r['direction_accuracy'] for r in fold_results]))
        avg_rmse = float(np.mean([r['rmse'] for r in fold_results]))

        logger.info(
            f"[{symbol}] CV Ortalama - MAPE: {avg_mape:.2f}% | "
            f"DirAcc: {avg_dir:.1f}% | RMSE: {avg_rmse:.4f}"
        )

        return {
            'symbol': symbol,
            'horizon': self.horizon,
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'avg_mape': round(avg_mape, 4),
            'avg_rmse': round(avg_rmse, 4),
            'avg_direction_accuracy': round(avg_dir, 2),
        }

    def train_final_model(
        self, df: pd.DataFrame, symbol: str, test_ratio: float = 0.2
    ) -> Dict[str, Any]:
        """Son modeli tüm veriye eğit (CV sonrasında)."""
        predictor = PricePredictor(self.horizon)
        return predictor.train(df, symbol, test_ratio=test_ratio)
