"""
Eski XGBoost Fiyat Tahmin Modeli (Legacy)

Bu dosya geriye uyumluluk icin saklanmaktadir.
Yeni sistem prediction.models paketini kullanir.
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


class LegacyPricePredictor:
    """Eski XGBoost tabanli fiyat tahmin modeli.

    Geriye uyumluluk icin saklanir. Yeni kod prediction.models kullanmalidir.
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
            raise ValueError("horizon 'daily' veya 'weekly' olmali")
        self.horizon = horizon
        self.feature_engineer = PredictionFeatureEngineer(horizon)
        self._models: Dict[str, Any] = {}
        self._feature_cols: Dict[str, list] = {}
        os.makedirs(MODELS_DIR, exist_ok=True)

    def _model_path(self, symbol: str) -> str:
        safe = symbol.replace('/', '_').replace('=', '_').replace('.', '_')
        return os.path.join(MODELS_DIR, f'{safe}_{self.horizon}_xgb.json')

    def is_trained(self, symbol: str) -> bool:
        return os.path.exists(self._model_path(symbol))

    def list_trained_models(self) -> list:
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

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        direction_acc = float(np.mean(true_dir == pred_dir) * 100) if len(true_dir) > 0 else 0.0
        return {
            'mape': round(mape, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'direction_accuracy': round(direction_acc, 2),
        }
