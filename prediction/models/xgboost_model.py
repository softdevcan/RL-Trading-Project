"""
XGBoost Tahmin Modeli

GPU destekli, Optuna ile optimize edilebilir XGBoost regressor.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

from prediction.models.base import BasePredictionModel
from prediction.seeding import GLOBAL_SEED

logger = logging.getLogger(__name__)


class XGBoostModel(BasePredictionModel):
    """XGBoost tabanli fiyat tahmin modeli."""

    MODEL_TYPE = 'xgboost'

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.0,
            'random_state': GLOBAL_SEED,
            'tree_method': 'hist',
            'early_stopping_rounds': 50,
        }

    def get_optuna_search_space(self, trial) -> Dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': GLOBAL_SEED,
            'tree_method': 'hist',
            'early_stopping_rounds': 50,
        }

    def _build_model(self, params: Dict[str, Any]):
        from xgboost import XGBRegressor

        fit_params = {k: v for k, v in params.items()
                      if k not in ('early_stopping_rounds',)}
        self.model = XGBRegressor(**fit_params)
        self._early_stopping = params.get('early_stopping_rounds', 50)

    def _supports_warm_start(self) -> bool:
        return True

    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        self.model.set_params(early_stopping_rounds=self._early_stopping)
        # Faz 6 (2.1): warm-start — onceki booster'dan devam et (yeni agaclar
        # ekle). Sadece ensemble warm_start=True verdiginde dolu olur; aksi
        # halde None -> sifirdan (mevcut davranis, bit-es).
        xgb_init = None
        ws = getattr(self, '_warm_start_from', None)
        if ws is not None and getattr(ws, 'model', None) is not None:
            try:
                xgb_init = ws.model.get_booster()
            except Exception:
                xgb_init = None
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            xgb_model=xgb_init,
        )
        val_pred = self.model.predict(X_val)
        return self.compute_metrics(y_val, val_pred)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _save_model(self, path: str):
        self.model.save_model(path + '.json')

    def _load_model(self, path: str):
        from xgboost import XGBRegressor
        self.model = XGBRegressor()
        self.model.load_model(path + '.json')

    def get_feature_importance(self) -> Dict[str, float]:
        """Ozellik onemleri."""
        if not self.is_fitted or not self.feature_cols:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_cols, importance))
