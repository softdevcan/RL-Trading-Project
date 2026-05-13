"""
LightGBM Tahmin Modeli

Hizli egitim, kategorik ozellik destegi ve GPU destekli LightGBM regressor.
"""

import logging
from typing import Dict, Any

import numpy as np

from prediction.models.base import BasePredictionModel

logger = logging.getLogger(__name__)


class LightGBMModel(BasePredictionModel):
    """LightGBM tabanli fiyat tahmin modeli."""

    MODEL_TYPE = 'lightgbm'

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 1000,
            'num_leaves': 63,
            'max_depth': -1,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
        }

    def get_optuna_search_space(self, trial) -> Dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
        }

    def _build_model(self, params: Dict[str, Any]):
        from lightgbm import LGBMRegressor

        fit_params = {k: v for k, v in params.items()
                      if k not in ('early_stopping_rounds',)}
        self.model = LGBMRegressor(**fit_params)
        self._early_stopping = params.get('early_stopping_rounds', 50)

    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                self._lgbm_early_stopping(),
                self._lgbm_log_eval(),
            ],
        )
        val_pred = self.model.predict(X_val)
        return self.compute_metrics(y_val, val_pred)

    def _lgbm_early_stopping(self):
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=self._early_stopping, verbose=False)

    @staticmethod
    def _lgbm_log_eval():
        from lightgbm import log_evaluation
        return log_evaluation(period=-1)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _save_model(self, path: str):
        self.model.booster_.save_model(path + '.lgbm')

    def _load_model(self, path: str):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=path + '.lgbm')
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor()
        self.model._Booster = booster
        self.model.fitted_ = True
        # sklearn wrapper'i dogru feature sayisiyla kurmak icin zorunlu;
        # eksik olunca n_features_in_=-1 ve predict sirasinda shape hatasi cikar.
        n_feat = booster.num_feature()
        self.model.n_features_in_ = n_feat
        self.model._n_features = n_feat

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted or not self.feature_cols:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_cols, importance))
