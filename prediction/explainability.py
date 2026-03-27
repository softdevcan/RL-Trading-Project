"""
Model Aciklanabilirligi — SHAP Entegrasyonu

Her model tipi icin uygun SHAP explainer secilir:
- TreeExplainer  : XGBoost, LightGBM, CatBoost
- DeepExplainer  : BiLSTM, TFT  (background ornekleme gerektirir)
- LinearExplainer: Ridge meta-learner
- KernelExplainer: fallback

Bagimlilık: shap >= 0.42.0  (pip install shap)
Kurulu degilse tum metodlar None doner; cagiran kod bunu kontrol etmeli.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.debug("shap bulunamadi. SHAP aciklamasi devre disi.")


class ModelExplainer:
    """Model tipi bazli SHAP aciklayici.

    Kullanim:
        explainer = ModelExplainer()
        shap_vals = explainer.explain_prediction(model, X_last_row, feature_cols)
        global_imp = explainer.explain_global(model, X_train, feature_cols)
    """

    def __init__(self, n_background: int = 100):
        """
        Args:
            n_background: DeepExplainer/KernelExplainer icin background ornek sayisi
        """
        self.n_background = n_background
        self._available = _SHAP_AVAILABLE

    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Tek Tahmin Aciklamasi
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        model,
        X: np.ndarray,
        feature_cols: List[str],
        X_background: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """Tek bir tahmin icin SHAP degerleri uret.

        Args:
            model: Egitilmis BasePredictionModel alt sinifi
            X: Son satir ozellik vektoru (1, n_features) veya (n_features,)
            feature_cols: Ozellik isim listesi
            X_background: Background ornekler (DeepExplainer icin gerekli)

        Returns:
            {
                'shap_values': {feature: shap_value},
                'base_value': float,
                'top_positive': [{feature, value, shap}],   # tahmini artiranlar
                'top_negative': [{feature, value, shap}],   # tahmini azaltanlar
            }
            veya None (SHAP kurulu degilse)
        """
        if not self._available:
            return None

        X_2d = np.atleast_2d(X)
        model_obj = self._get_sklearn_model(model)
        if model_obj is None:
            return None

        try:
            explainer, shap_vals, base_val = self._compute_shap(
                model_obj, X_2d, X_background
            )
            if shap_vals is None:
                return None

            # Son satir shap degerleri (1-D)
            sv = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
            sv = np.asarray(sv, dtype=float)

            n = min(len(feature_cols), len(sv))
            shap_dict = {feature_cols[i]: round(float(sv[i]), 6) for i in range(n)}

            sorted_items = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
            x_vals = X_2d[0]

            top_positive = [
                {'feature': f, 'value': round(float(x_vals[feature_cols.index(f)]), 4), 'shap': v}
                for f, v in sorted_items if v > 0
            ][:10]

            top_negative = [
                {'feature': f, 'value': round(float(x_vals[feature_cols.index(f)]), 4), 'shap': v}
                for f, v in sorted_items if v < 0
            ][-10:]

            return {
                'shap_values': shap_dict,
                'base_value': round(float(base_val), 4) if base_val is not None else None,
                'top_positive': top_positive,
                'top_negative': top_negative,
            }

        except Exception as exc:
            logger.warning(f"explain_prediction hatasi: {exc}")
            return None

    # ------------------------------------------------------------------
    # Global Ozellik Onem
    # ------------------------------------------------------------------

    def explain_global(
        self,
        model,
        X_train: np.ndarray,
        feature_cols: List[str],
        max_samples: int = 500,
    ) -> Optional[Dict[str, float]]:
        """Egitim verisi uzerinde global feature importance hesapla.

        Args:
            model: Egitilmis BasePredictionModel alt sinifi
            X_train: Egitim ozellik matrisi
            feature_cols: Ozellik isim listesi
            max_samples: Kullanilacak maksimum ornek sayisi

        Returns:
            {feature: mean_abs_shap_value} sozlugu (buyukten kucuge sirali)
            veya None
        """
        if not self._available:
            return None

        model_obj = self._get_sklearn_model(model)
        if model_obj is None:
            return None

        # Ornek sayisini sinirla
        if len(X_train) > max_samples:
            idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_sample = X_train[idx]
        else:
            X_sample = X_train

        try:
            _, shap_vals, _ = self._compute_shap(model_obj, X_sample, None)
            if shap_vals is None:
                return None

            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            n = min(len(feature_cols), len(mean_abs))
            importance = {feature_cols[i]: round(float(mean_abs[i]), 6) for i in range(n)}
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        except Exception as exc:
            logger.warning(f"explain_global hatasi: {exc}")
            return None

    # ------------------------------------------------------------------
    # Yardimci metodlar
    # ------------------------------------------------------------------

    def _get_sklearn_model(self, model):
        """BasePredictionModel'den icerideki sklearn/xgb/lgbm modelini cikar."""
        # Tree modelleri (XGBoost, LightGBM, CatBoost) -> model.model
        if hasattr(model, 'model') and model.model is not None:
            return model.model
        # Dogrudan sklearn uyumlu nesne (meta-learner Ridge)
        if hasattr(model, 'predict') and hasattr(model, 'fit'):
            return model
        return None

    def _compute_shap(self, sklearn_model, X: np.ndarray, X_background: Optional[np.ndarray]):
        """Model tipine gore uygun SHAP explainer sec ve hesapla."""
        model_type = type(sklearn_model).__name__.lower()

        # Tree-based modeller
        tree_types = ('xgbregressor', 'xgbclassifier', 'lgbmregressor', 'lgbmclassifier',
                      'catboostregressor', 'catboostclassifier',
                      'randomforest', 'gradientboosting', 'extratrees', 'decisiontree')
        if any(t in model_type for t in tree_types):
            explainer = shap.TreeExplainer(sklearn_model)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[0]
            return explainer, np.asarray(sv), float(explainer.expected_value
                                                    if not isinstance(explainer.expected_value, (list, np.ndarray))
                                                    else explainer.expected_value[0])

        # Linear modeller (Ridge, Lasso, vb.)
        linear_types = ('ridge', 'lasso', 'linearregression', 'elasticnet')
        if any(t in model_type for t in linear_types):
            background = X_background if X_background is not None else X
            explainer = shap.LinearExplainer(sklearn_model, background)
            sv = explainer.shap_values(X)
            return explainer, np.asarray(sv), float(explainer.expected_value)

        # Fallback: KernelExplainer (model-agnostic, yavas)
        bg = X_background if X_background is not None else X[:min(50, len(X))]
        explainer = shap.KernelExplainer(sklearn_model.predict, bg)
        sv = explainer.shap_values(X[:min(100, len(X))])
        return explainer, np.asarray(sv), float(explainer.expected_value)
