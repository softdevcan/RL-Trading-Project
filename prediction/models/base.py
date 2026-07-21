"""
Tahmin Modeli Soyut Temel Sinifi

Tum tahmin modelleri bu ABC'yi implement eder.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join('models', 'prediction')


class BasePredictionModel(ABC):
    """Tum tahmin modelleri icin soyut temel sinif.

    Her alt sinif su islemleri implement etmelidir:
    - _build_model: Modeli olustur
    - _fit: Modeli egit
    - _predict_raw: Ham tahmin uret
    - _save_model / _load_model: Model kaydet/yukle
    - get_default_params: Varsayilan parametreleri dondur
    - get_optuna_search_space: HPO arama uzayi
    """

    MODEL_TYPE: str = 'base'

    def __init__(
        self,
        horizon: str = 'daily',
        params: Optional[Dict] = None,
        source: Optional[str] = None,
    ):
        if horizon not in ('daily', 'weekly'):
            raise ValueError("horizon 'daily' veya 'weekly' olmali")
        self.horizon = horizon
        self.source = source  # gold metrikleri icin 'borsapy' / 'yfinance'; hisse icin None
        self.params = params or self.get_default_params()
        self.model = None
        self.feature_cols: List[str] = []
        self.is_fitted = False
        os.makedirs(MODELS_DIR, exist_ok=True)

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Varsayilan model parametrelerini dondur."""

    @abstractmethod
    def get_optuna_search_space(self, trial) -> Dict[str, Any]:
        """Optuna trial icin hiperparametre arama uzayi."""

    @abstractmethod
    def _build_model(self, params: Dict[str, Any]):
        """Modeli verilen parametrelerle olustur."""

    @abstractmethod
    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Modeli egit, validation metrikleri dondur."""

    @abstractmethod
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Ham tahmin uret (fiyat)."""

    def _predict_direction_raw(self, _X: np.ndarray) -> Optional[np.ndarray]:
        """Direction head'den ham yon olasiligi uret (0-1 arasi sigmoid cikti).

        Alt siniflar bu metodu override edebilir. Varsayilan: None (desteklenmiyor).
        """
        return None

    @abstractmethod
    def _save_model(self, path: str):
        """Modeli dosyaya kaydet."""

    @abstractmethod
    def _load_model(self, path: str):
        """Modeli dosyadan yukle."""

    # ------------------------------------------------------------------
    # Ortak API
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Modeli egit ve metrikleri dondur."""
        self.feature_cols = feature_cols or []
        self._build_model(self.params)

        val_metrics = self._fit(X_train, y_train, X_val, y_val)

        train_pred = self._predict_raw(X_train)
        val_pred = self._predict_raw(X_val)

        train_metrics = self.compute_metrics(y_train, train_pred)
        full_val_metrics = self.compute_metrics(y_val, val_pred)

        self.is_fitted = True

        return {
            'model_type': self.MODEL_TYPE,
            'horizon': self.horizon,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_features': len(self.feature_cols),
            'train_metrics': train_metrics,
            'val_metrics': full_val_metrics,
            'params': self.params,
            'trained_at': datetime.now().isoformat(),
        }

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Tahmin uret: fiyat, yon ve guven skoru.

        Returns:
            {
                'price': float,
                'direction': str ('UP' / 'DOWN'),
                'confidence': float (0-1),
                'raw_predictions': ndarray,
            }
        """
        assert self.is_fitted, "Model henuz egitilmemis"
        raw = self._predict_raw(X)

        if len(raw) == 1:
            return {
                'price': float(raw[0]),
                'direction': None,
                'confidence': 0.5,
                'raw_predictions': raw,
            }

        return {
            'price': float(raw[-1]),
            'direction': None,
            'confidence': 0.5,
            'raw_predictions': raw,
        }

    # ------------------------------------------------------------------
    # Kaydetme / Yukleme
    # ------------------------------------------------------------------

    def _safe_symbol(self, symbol: str) -> str:
        return symbol.replace('/', '_').replace('=', '_').replace('.', '_')

    def _model_id(self, symbol: str) -> str:
        """Sembol + source kombinasyonu — dosya isminde kullanilir."""
        safe = self._safe_symbol(symbol)
        return f'{safe}__{self.source}' if self.source else safe

    def model_path(self, symbol: str) -> str:
        return os.path.join(MODELS_DIR, f'{self._model_id(symbol)}_{self.horizon}_{self.MODEL_TYPE}')

    def save(self, symbol: str, metrics: Optional[Dict] = None):
        """Modeli ve meta verisini kaydet."""
        path = self.model_path(symbol)
        self._save_model(path)

        meta = {
            'symbol': symbol,
            'source': self.source,
            'horizon': self.horizon,
            'model_type': self.MODEL_TYPE,
            'feature_cols': self.feature_cols,
            'params': self.params,
            'saved_at': datetime.now().isoformat(),
        }
        if metrics:
            meta['metrics'] = metrics

        meta_path = path + '_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"  [{self.MODEL_TYPE}] Model kaydedildi: {path}")

    def load(self, symbol: str):
        """Kaydedilmis modeli yukle."""
        path = self.model_path(symbol)
        self._load_model(path)

        meta_path = path + '_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.feature_cols = meta.get('feature_cols', [])
            self.params = meta.get('params', self.params)

        self.is_fitted = True
        logger.info(f"  [{self.MODEL_TYPE}] Model yuklendi: {path}")

    def is_trained_for(self, symbol: str) -> bool:
        """Sembol icin kayitli model var mi?"""
        path = self.model_path(symbol)
        meta_path = path + '_meta.json'
        return os.path.exists(meta_path)

    # ------------------------------------------------------------------
    # Metrikler
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """MAPE, RMSE, MAE ve yon dogrulugu hesapla."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_true.size == 0 or y_pred.size == 0:
            return {'mape': 0.0, 'rmse': 0.0, 'mae': 0.0, 'direction_accuracy': 0.0}

        # Sekans-tabanli modeller (BiLSTM/TFT) lookback yuzunden y_true'dan kisa
        # bir tahmin dizisi dondurur. Son-N hizalama ile uzunlugu esitle —
        # aksi halde bool maske uyumsuzlugu IndexError firlatir ve ensemble bu
        # modelleri sessizce duser (bkz. Faz 6 R1).
        if len(y_true) != len(y_pred):
            m = min(len(y_true), len(y_pred))
            y_true = y_true[-m:]
            y_pred = y_pred[-m:]

        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = 0.0

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        if len(y_true) > 1:
            true_dir = np.diff(y_true) > 0
            pred_dir = np.diff(y_pred) > 0
            direction_acc = float(np.mean(true_dir == pred_dir) * 100)
        else:
            direction_acc = 0.0

        def _finite(v: float) -> float:
            return float(v) if np.isfinite(v) else 0.0

        return {
            'mape': round(_finite(mape), 4),
            'rmse': round(_finite(rmse), 4),
            'mae': round(_finite(mae), 4),
            'direction_accuracy': round(_finite(direction_acc), 2),
        }
