"""
CatBoost Tahmin Modeli

GPU destekli, kategorik ozellik islemeli CatBoost regressor.
"""

import logging
import os
from typing import Dict, Any

import numpy as np

from prediction.models.base import BasePredictionModel

logger = logging.getLogger(__name__)


def _default_task_type() -> str:
    """GPU'yu sadece env degiskeni ile secin, aksi halde CPU kullanilir.

    CatBoost'un GPU backendi sisteme yuklu CUDA driver surumune uyumsuzsa
    egitim .fit() sirasinda patliyor ve fallback tetiklenemiyor — bu yuzden
    varsayilan guvenli seçenek CPU.
    """
    return 'GPU' if os.environ.get('CATBOOST_USE_GPU', '').lower() in ('1', 'true', 'yes') else 'CPU'


class CatBoostModel(BasePredictionModel):
    """CatBoost tabanli fiyat tahmin modeli."""

    MODEL_TYPE = 'catboost'

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'border_count': 128,
            'random_seed': 42,
            'verbose': 0,
            'task_type': _default_task_type(),
            'early_stopping_rounds': 50,
        }

    def get_optuna_search_space(self, trial) -> Dict[str, Any]:
        return {
            'iterations': trial.suggest_int('iterations', 200, 2000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42,
            'verbose': 0,
            'task_type': _default_task_type(),
            'early_stopping_rounds': 50,
        }

    def _build_model(self, params: Dict[str, Any]):
        from catboost import CatBoostRegressor

        fit_params = {k: v for k, v in params.items()
                      if k not in ('early_stopping_rounds',)}

        self.model = CatBoostRegressor(**fit_params)
        self._early_stopping = params.get('early_stopping_rounds', 50)

    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        from catboost import CatBoostRegressor, Pool

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        try:
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=self._early_stopping,
                verbose=False,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if 'cuda' in msg or 'gpu' in msg:
                logger.warning("  CatBoost GPU hatasi, CPU'ya gecilip yeniden deneniyor")
                params = self.model.get_params()
                params['task_type'] = 'CPU'
                self.model = CatBoostRegressor(**params)
                self.model.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=self._early_stopping,
                    verbose=False,
                )
            else:
                raise

        val_pred = self.model.predict(X_val)
        return self.compute_metrics(y_val, val_pred)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _save_model(self, path: str):
        self.model.save_model(path + '.cbm')

    def _load_model(self, path: str):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor()
        self.model.load_model(path + '.cbm')

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted or not self.feature_cols:
            return {}
        importance = self.model.get_feature_importance()
        return dict(zip(self.feature_cols, importance))
