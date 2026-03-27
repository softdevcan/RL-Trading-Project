"""
ICEEMDAN Gurultu On-Isleme

Improved Complete Ensemble EMD with Adaptive Noise (ICEEMDAN) kullanarak
fiyat serisindeki gurultuyu filtreler, trend ve enerji ozelliklerini uretir.

Bagimlilık: EMD-signal >= 1.5.0 (pip install EMD-signal)
Kurulu degilse otomatik olarak devre disi kalir (use_iceemdan=False ile ayni davranis).

Referans: Torres et al. (2011), Colominas et al. (2014)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from PyEMD import CEEMDAN
    _EMD_AVAILABLE = True
except ImportError:
    _EMD_AVAILABLE = False
    logger.debug("EMD-signal bulunamadi. ICEEMDAN devre disi.")


class ICEEMDANProcessor:
    """ICEEMDAN tabanli fiyat serisi gurultu filtreleyici.

    Kullanim:
        proc = ICEEMDANProcessor()
        filtered_close = proc.filter_noise(close_series)
        imf_features = proc.extract_imf_features(close_series)
    """

    def __init__(
        self,
        noise_imfs: int = 2,
        trials: int = 100,
        epsilon: float = 0.005,
        max_imf: int = 8,
    ):
        """
        Args:
            noise_imfs: Gürültu IMF sayisi (ilk N IMF atilir)
            trials: ICEEMDAN ensemble deneme sayisi
            epsilon: Eklenen gurultu std katsayisi
            max_imf: Maksimum IMF sayisi
        """
        self.noise_imfs = noise_imfs
        self.trials = trials
        self.epsilon = epsilon
        self.max_imf = max_imf
        self._available = _EMD_AVAILABLE

    def is_available(self) -> bool:
        return self._available

    def decompose(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Sinyali IMF'lere ayir.

        Returns:
            IMF matrisi (n_imfs, n_samples) veya None (kurulu degilse)
        """
        if not self._available:
            return None
        try:
            ceemdan = CEEMDAN(
                trials=self.trials,
                epsilon=self.epsilon,
                ext_EMD=None,
            )
            ceemdan.noise_seed(42)
            imfs = ceemdan.ceemdan(signal)
            return imfs
        except Exception as exc:
            logger.warning(f"ICEEMDAN ayristirma hatasi: {exc}")
            return None

    def filter_noise(self, close: pd.Series) -> pd.Series:
        """Fiyat serisinden ilk N gurultu IMF'ini kaldir.

        Args:
            close: Ham kapanış fiyati serisi

        Returns:
            Filtrelenmis fiyat serisi (orijinalin uzunlugunda, index korunur)
        """
        if not self._available:
            return close

        arr = close.dropna().values.astype(float)
        if len(arr) < 50:
            return close

        imfs = self.decompose(arr)
        if imfs is None or len(imfs) <= self.noise_imfs:
            return close

        # Ilk noise_imfs IMF'i at, kalanları topla
        filtered = imfs[self.noise_imfs:].sum(axis=0)

        # Orijinal index ve uzunluga geri esle
        result = close.copy()
        valid_idx = close.dropna().index
        n = min(len(filtered), len(valid_idx))
        result.loc[valid_idx[:n]] = filtered[:n]

        correlation = float(np.corrcoef(arr[:n], filtered[:n])[0, 1])
        if correlation < 0.95:
            logger.warning(
                f"ICEEMDAN korelasyon dusuk ({correlation:.3f}), orijinal seri kullaniliyor"
            )
            return close

        logger.debug(f"ICEEMDAN: {len(imfs)} IMF, {self.noise_imfs} gurultu IMF atildi (r={correlation:.4f})")
        return result

    def extract_imf_features(self, close: pd.Series) -> pd.DataFrame:
        """IMF bazlı ek ozellikler uret.

        Uretilen ozellikler:
        - imf_trend_component: Trend IMF'inin kapanisa orani
        - imf_energy_ratio: Gurultu IMF'lerinin toplam enerjiye orani (gurultu seviyesi gostergesi)

        Returns:
            Ozellik DataFrame'i (close ile ayni index)
        """
        result = pd.DataFrame(index=close.index)
        result['imf_trend_component'] = 0.0
        result['imf_energy_ratio'] = 0.5

        if not self._available:
            return result

        arr = close.dropna().values.astype(float)
        if len(arr) < 50:
            return result

        imfs = self.decompose(arr)
        if imfs is None or len(imfs) <= self.noise_imfs:
            return result

        valid_idx = close.dropna().index
        n = min(imfs.shape[1], len(valid_idx))

        # Trend IMF: son IMF (en dusuk frekans)
        trend_imf = imfs[-1, :n]
        safe_close = arr[:n]
        safe_close = np.where(safe_close == 0, np.nan, safe_close)
        trend_ratio = np.where(np.isfinite(safe_close), trend_imf / safe_close, 0.0)
        result.loc[valid_idx[:n], 'imf_trend_component'] = trend_ratio

        # Enerji orani: gurultu IMF'leri / toplam
        noise_energy = float(np.sum(imfs[:self.noise_imfs, :n] ** 2))
        total_energy = float(np.sum(imfs[:, :n] ** 2))
        energy_ratio = noise_energy / total_energy if total_energy > 0 else 0.5
        result['imf_energy_ratio'] = energy_ratio

        return result
