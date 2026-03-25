"""
XGBoost Fiyat Tahmin Modeli
Günlük ve haftalık kapanış fiyatı tahmini yapar.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from prediction.feature_engineer import PredictionFeatureEngineer

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join('models', 'prediction')


class PricePredictor:
    """XGBoost tabanlı fiyat tahmin modeli.

    Her sembol ve horizon için ayrı model eğitilir.
    """

    XGB_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
        'eval_metric': 'rmse',
    }

    def __init__(self, horizon: str = 'daily'):
        if horizon not in ('daily', 'weekly'):
            raise ValueError("horizon 'daily' veya 'weekly' olmalı")
        self.horizon = horizon
        self.feature_engineer = PredictionFeatureEngineer(horizon)
        self._models: Dict[str, Any] = {}  # symbol → xgb model
        self._feature_cols: Dict[str, list] = {}
        os.makedirs(MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, symbol: str,
              test_ratio: float = 0.2) -> Dict[str, Any]:
        """Modeli eğit ve test metriklerini döndür.

        Args:
            df: Ham OHLCV + teknik indikatörlü tek-sembol DataFrame
            symbol: Sembol adı
            test_ratio: Test seti oranı

        Returns:
            Eğitim ve test metrikleri
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost kurulu değil. `pip install xgboost` çalıştırın.")

        logger.info(f"[{symbol}] {self.horizon} modeli eğitiliyor...")

        feat_df = self.feature_engineer.build_features(df, symbol)
        feature_cols = self.feature_engineer.get_feature_columns(feat_df)

        X = feat_df[feature_cols].values
        y = feat_df['target'].values

        n = len(X)
        split = int(n * (1 - test_ratio))

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        params = {k: v for k, v in self.XGB_PARAMS.items()
                  if k != 'early_stopping_rounds' and k != 'eval_metric'}
        model = XGBRegressor(
            **params,
            early_stopping_rounds=self.XGB_PARAMS['early_stopping_rounds'],
            eval_metric=self.XGB_PARAMS['eval_metric'],
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = self._compute_metrics(y_train, train_pred)
        test_metrics = self._compute_metrics(y_test, test_pred)

        self._models[symbol] = model
        self._feature_cols[symbol] = feature_cols

        logger.info(
            f"  Train MAPE: {train_metrics['mape']:.2f}% | "
            f"Test MAPE: {test_metrics['mape']:.2f}% | "
            f"Test Dir Acc: {test_metrics['direction_accuracy']:.1f}%"
        )

        self.save(symbol)

        return {
            'symbol': symbol,
            'horizon': self.horizon,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'n_features': int(len(feature_cols)),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_path': self._model_path(symbol),
            'trained_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------

    def predict_next(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Sonraki periyot için fiyat tahmini yap.

        Args:
            df: Güncel OHLCV + indikatör verisi
            symbol: Sembol

        Returns:
            Tahmin dict'i (predicted_close, direction, change_pct, confidence vb.)
        """
        if symbol not in self._models:
            self.load(symbol)

        model = self._models[symbol]
        feature_cols = self._feature_cols[symbol]

        feat_df = self.feature_engineer.build_features(df, symbol)

        # Hedef değişken oluşturulamadıysa son satırı kullan
        last_row = feat_df.iloc[-1]
        X_last = last_row[feature_cols].values.reshape(1, -1)

        predicted_close = float(model.predict(X_last)[0])
        current_close = float(df['close'].iloc[-1])

        change_pct = (predicted_close - current_close) / current_close * 100
        direction = 'UP' if change_pct > 0 else 'DOWN'

        # Basit güven skoru: tahmin gücü / son vol bazlı
        try:
            booster = model.get_booster()
            scores = booster.get_score(importance_type='gain')
            top_gain = sorted(scores.values(), reverse=True)[:5]
            confidence = min(0.95, max(0.50, np.mean(top_gain) / (np.mean(top_gain) + 1)))
        except Exception:
            confidence = 0.65

        # Tahmin tarihi
        if self.horizon == 'daily':
            pred_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            pred_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

        return {
            'symbol': symbol,
            'prediction_date': pred_date,
            'horizon': self.horizon,
            'predicted_close': round(predicted_close, 4),
            'predicted_direction': direction,
            'predicted_change_pct': round(change_pct, 2),
            'confidence': round(float(confidence), 4),
            'current_close': round(current_close, 4),
            'made_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Kaydetme / Yükleme
    # ------------------------------------------------------------------

    def _model_path(self, symbol: str) -> str:
        safe = symbol.replace('/', '_').replace('=', '_').replace('.', '_')
        return os.path.join(MODELS_DIR, f'{safe}_{self.horizon}_xgb.json')

    def save(self, symbol: str):
        """Modeli JSON formatında kaydet."""
        model = self._models.get(symbol)
        if model is None:
            raise ValueError(f"{symbol} için eğitilmiş model yok")

        path = self._model_path(symbol)
        model.save_model(path)

        meta_path = path.replace('.json', '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': symbol,
                'horizon': self.horizon,
                'feature_cols': self._feature_cols.get(symbol, []),
                'saved_at': datetime.now().isoformat(),
            }, f, indent=2)

        logger.info(f"  Model kaydedildi: {path}")

    def load(self, symbol: str):
        """Kaydedilmiş modeli yükle."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost kurulu değil.")

        path = self._model_path(symbol)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model bulunamadı: {path}. Önce train() çağırın."
            )

        model = XGBRegressor()
        model.load_model(path)
        self._models[symbol] = model

        meta_path = path.replace('.json', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self._feature_cols[symbol] = meta.get('feature_cols', [])

        logger.info(f"  Model yüklendi: {path}")

    def is_trained(self, symbol: str) -> bool:
        """Sembol için kayıtlı model var mı?"""
        return os.path.exists(self._model_path(symbol))

    def list_trained_models(self) -> list:
        """Kayıtlı tüm modelleri listele."""
        if not os.path.exists(MODELS_DIR):
            return []
        models = []
        for fname in os.listdir(MODELS_DIR):
            if fname.endswith(f'_{self.horizon}_xgb.json'):
                meta_path = os.path.join(MODELS_DIR, fname.replace('.json', '_meta.json'))
                info = {'file': fname, 'horizon': self.horizon}
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        info.update(json.load(f))
                models.append(info)
        return models

    # ------------------------------------------------------------------
    # Metrikler
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """MAPE, RMSE, MAE ve yön doğruluğu hesapla."""
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # Yön doğruluğu: gerçek ve tahmin aynı yönde mi?
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        direction_acc = float(np.mean(true_dir == pred_dir) * 100) if len(true_dir) > 0 else 0.0

        return {
            'mape': round(mape, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'direction_accuracy': round(direction_acc, 2),
        }
