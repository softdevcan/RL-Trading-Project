"""
Tahmin Modeli Hiperparametre Optimizasyonu

Optuna ile TimeSeriesSplit cross-validation kullanarak
her model tipi icin en iyi hiperparametreleri bulur.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit

from prediction.models.base import BasePredictionModel, MODELS_DIR
from prediction.seeding import GLOBAL_SEED

logger = logging.getLogger(__name__)

HYPEROPT_DIR = os.path.join('results', 'prediction_hyperopt')


class PredictionHyperOptimizer:
    """Optuna tabanli hiperparametre optimizasyonu.

    TimeSeriesSplit ile walk-forward CV kullanir.
    TPESampler ile verimli arama, MedianPruner ile
    basarisiz denemeleri erken durdurma.
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_cv_splits: int = 3,
        timeout: Optional[int] = 3600,
    ):
        """
        Args:
            n_trials: Optuna deneme sayisi
            n_cv_splits: TimeSeriesSplit fold sayisi
            timeout: Maksimum sure (saniye), None = sinir yok
        """
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.timeout = timeout
        os.makedirs(HYPEROPT_DIR, exist_ok=True)

    def optimize(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        horizon: str = 'daily',
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Model icin en iyi hiperparametreleri bul.

        Args:
            model_type: Model tipi ('xgboost', 'lightgbm', 'catboost', 'bilstm', 'tft')
            X: Ozellik matrisi
            y: Hedef degisken
            horizon: 'daily' veya 'weekly'
            study_name: Optuna study adi (None = otomatik)

        Returns:
            En iyi parametreler ve optimizasyon sonuclari
        """
        if study_name is None:
            study_name = f'{model_type}_{horizon}_hpo'

        logger.info(f"[{model_type}] HPO basliyor: {self.n_trials} trial, {self.n_cv_splits} CV fold")

        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(seed=GLOBAL_SEED),
            pruner=MedianPruner(n_warmup_steps=5),
        )

        def objective(trial):
            return self._objective(trial, model_type, X, y, horizon)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        best = study.best_trial
        logger.info(
            f"  En iyi trial #{best.number}: "
            f"MAPE={best.value:.4f}% | Params={best.params}"
        )

        result = {
            'model_type': model_type,
            'horizon': horizon,
            'best_params': best.params,
            'best_mape': best.value,
            'n_trials': len(study.trials),
            'study_name': study_name,
        }

        self._save_results(result, study_name)

        return result

    def _objective(
        self,
        trial: optuna.Trial,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        horizon: str,
    ) -> float:
        """Optuna objective fonksiyonu."""
        from prediction.models import MODEL_REGISTRY

        model_cls = MODEL_REGISTRY.get(model_type)
        if model_cls is None:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")

        temp_model = model_cls(horizon=horizon)
        params = temp_model.get_optuna_search_space(trial)

        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        fold_mapes = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if len(X_val) < 10:
                continue

            try:
                model = model_cls(horizon=horizon, params=params)
                result = model.train(X_train, y_train, X_val, y_val)
                mape = result['val_metrics']['mape']
                fold_mapes.append(mape)

                trial.report(mape, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.debug(f"  Trial {trial.number} fold {fold_idx} hatasi: {exc}")
                fold_mapes.append(999.0)

        if not fold_mapes:
            return 999.0

        return float(np.mean(fold_mapes))

    def optimize_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon: str = 'daily',
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Tum modeller icin HPO calistir.

        Args:
            X: Ozellik matrisi
            y: Hedef degisken
            horizon: 'daily' veya 'weekly'
            model_types: Optimize edilecek model tipleri (None = hepsi)

        Returns:
            Model tipi -> optimizasyon sonucu dict'i
        """
        from prediction.models import MODEL_REGISTRY

        if model_types is None:
            model_types = list(MODEL_REGISTRY.keys())

        results = {}
        for model_type in model_types:
            try:
                result = self.optimize(model_type, X, y, horizon)
                results[model_type] = result
            except Exception as exc:
                logger.error(f"  [{model_type}] HPO hatasi: {exc}")
                results[model_type] = {'error': str(exc)}

        return results

    def _save_results(self, result: Dict[str, Any], study_name: str):
        """Sonuclari JSON dosyasina kaydet."""
        path = os.path.join(HYPEROPT_DIR, f'{study_name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"  HPO sonuclari kaydedildi: {path}")

    @staticmethod
    def load_best_params(model_type: str, horizon: str = 'daily') -> Optional[Dict[str, Any]]:
        """Kaydedilmis en iyi parametreleri yukle."""
        study_name = f'{model_type}_{horizon}_hpo'
        path = os.path.join(HYPEROPT_DIR, f'{study_name}.json')
        if not os.path.exists(path):
            return None

        with open(path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return result.get('best_params')
