"""
TATS — Trend-Adjusted Time Series

Ensemble'in fiyat tahmini, trend sınıflandırıcısı ile çelisiyorsa
tahmini trend yönüne dogru yumusatır.

Mimari:
    1. Trend sınıflandırıcı (XGBoost): UP / DOWN / FLAT
    2. Düzeltme: tahminin trende aykırı oldugu durumda
       correction_strength = trend_confidence * correction_factor ile fiyat kaydırılır

Kullanım:
    tats = TATSCorrector()
    tats.fit(X_meta, y_meta, current_prices_meta)
    corrected_price = tats.correct(predicted_price, current_price, X_last_row)
"""

import logging
from typing import Optional, Dict, Any

import numpy as np

from prediction.seeding import GLOBAL_SEED

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.debug("XGBoost bulunamadi, TATS devre disi.")


class TATSCorrector:
    """Trend-Adjusted Time Series düzeltici.

    Attributes:
        correction_factor: Maksimum düzeltme kuvveti (0-1). Optuna ile optimize edilebilir.
        flat_threshold: Bu yüzdenin altındaki getiriler FLAT sayılır (|return| < threshold).
    """

    def __init__(
        self,
        correction_factor: float = 0.5,
        flat_threshold: float = 0.005,
    ):
        self.correction_factor = correction_factor
        self.flat_threshold = flat_threshold
        self.classifier: Optional[Any] = None
        self._is_fitted = False
        # Faz 6: fit'te veride gorulen semantik siniflar (0=DOWN,1=FLAT,2=UP).
        # predict_proba kolonlarini semantik etikete geri cevirmek icin.
        self._semantic_classes = np.array([0, 1, 2])

    def fit(
        self,
        X: np.ndarray,
        y_prices: np.ndarray,
        current_prices: np.ndarray,
    ) -> Dict[str, Any]:
        """Trend sınıflandırıcısını egit.

        Args:
            X: Meta-train ozellik matrisi (base model tahminleri)
            y_prices: Gercek gelecek kapanış fiyatları
            current_prices: İlgili gunlerin mevcut kapanış fiyatları

        Returns:
            Egitim metrikleri
        """
        if not _XGBOOST_AVAILABLE:
            logger.warning("XGBoost bulunamadi, TATS classifier egitilmiyor")
            return {'trained': False}

        returns = (y_prices - current_prices) / np.where(current_prices == 0, 1, current_prices)
        labels = np.where(
            returns > self.flat_threshold, 2,       # UP
            np.where(returns < -self.flat_threshold, 0, 1)  # DOWN=0, FLAT=1
        )

        if len(np.unique(labels)) < 2:
            logger.warning("TATS: Yeterli sinif cesitliligi yok, egitim atlaniyor")
            return {'trained': False}

        # XGBoost >= 2.0 hedef siniflari ardisik [0..k-1] bekler; eger veride
        # sadece {0,2} gibi bir alt kume varsa dogrudan fit patlar
        # (ValueError: Expected [0 1], got [0 2]). Semantik etiketleri (0=DOWN,
        # 1=FLAT, 2=UP) sakla, egitimi encode edilmis uzayda yap, tahminde geri
        # cevir. (Faz 6: DL modelleri devreye girince ilk kez tetiklenen bug.)
        self._semantic_classes = np.unique(labels)  # or. [0 2]
        remap = {orig: i for i, orig in enumerate(self._semantic_classes)}
        labels_enc = np.array([remap[l] for l in labels])

        self.classifier = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=GLOBAL_SEED,
            verbosity=0,
            eval_metric='mlogloss',
        )
        self.classifier.fit(X, labels_enc)
        self._is_fitted = True

        train_acc = float(np.mean(self.classifier.predict(X) == labels_enc))
        class_dist = {int(c): int(np.sum(labels == c)) for c in np.unique(labels)}
        logger.info(f"  [TATS] Classifier egitildi. Train acc: {train_acc:.3f}, dagilim: {class_dist}")
        return {'trained': True, 'train_accuracy': train_acc, 'class_distribution': class_dist}

    def correct(
        self,
        predicted_price: float,
        current_price: float,
        X_last: np.ndarray,
    ) -> tuple:
        """Tahmini trend sınıflandırıcısı ile düzelt.

        Args:
            predicted_price: Ensemble ham fiyat tahmini
            current_price: Guncel kapanış
            X_last: Son satır ozellik vektoru (1, n_features)

        Returns:
            (corrected_price, tats_meta) — meta: classifier bilgileri
        """
        if not self._is_fitted or not _XGBOOST_AVAILABLE:
            return predicted_price, {'applied': False}

        try:
            proba = self.classifier.predict_proba(X_last.reshape(1, -1))[0]
            # predict_proba encode edilmis uzayda kolon dondurur; semantik
            # etikete (0=DOWN, 1=FLAT, 2=UP) geri cevir (fit'teki remap'in tersi).
            enc_idx = int(np.argmax(proba))
            trend_confidence = float(proba[enc_idx])
            classes = getattr(self, '_semantic_classes', np.array([0, 1, 2]))
            trend_label = int(classes[enc_idx]) if enc_idx < len(classes) else enc_idx

            predicted_return = (predicted_price - current_price) / (current_price or 1)
            trend_map = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
            trend = trend_map[trend_label]

            # Çelisim tespiti: tahmin UP ama trend DOWN, veya tam tersi
            conflict = (
                (predicted_return > self.flat_threshold and trend == 'DOWN') or
                (predicted_return < -self.flat_threshold and trend == 'UP')
            )

            corrected_price = predicted_price
            if conflict and trend != 'FLAT':
                correction_strength = trend_confidence * self.correction_factor
                # Trende dogru kaydır: trend DOWN ise tahmini asagi çek, UP ise yukari
                direction_sign = 1.0 if trend == 'UP' else -1.0
                delta = abs(predicted_price - current_price) * correction_strength * direction_sign
                corrected_price = predicted_price + delta

            return corrected_price, {
                'applied': conflict,
                'trend': trend,
                'trend_confidence': round(trend_confidence, 4),
                'correction_strength': round(trend_confidence * self.correction_factor, 4) if conflict else 0.0,
                'original_price': round(predicted_price, 4),
            }

        except Exception as exc:
            logger.warning(f"TATS düzeltme hatasi: {exc}")
            return predicted_price, {'applied': False, 'error': str(exc)}

    def is_fitted(self) -> bool:
        return self._is_fitted
