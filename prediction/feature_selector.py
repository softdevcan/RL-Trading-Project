"""
Otomatik Ozellik Secimi

Mutual information, permutation importance ve korelasyon filtresi
kullanarak en bilgilendirici ozellikleri secer.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from prediction.seeding import GLOBAL_SEED

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Zaman serisi icin ozellik secimi pipeline'i.

    Uc asamali filtreleme:
    1. Korelasyon filtresi: yuksek korelasyonlu ciftlerden birini at
    2. Mutual information skoru: hedefle iliskisi dusuk olanlari at
    3. Permutation importance: model bazli onem degerlendirmesi
    """

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        mi_percentile: float = 10.0,
        max_features: Optional[int] = None,
    ):
        """
        Args:
            correlation_threshold: Korelasyon esik degeri (bu ustundekiler atilir)
            mi_percentile: Mutual info alt yuzdelik (bu altindakiler atilir)
            max_features: Maksimum ozellik sayisi (None = sinir yok)
        """
        self.correlation_threshold = correlation_threshold
        self.mi_percentile = mi_percentile
        self.max_features = max_features

    def fit_select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        use_permutation_importance: bool = False,
    ) -> Dict[str, Any]:
        """Ozellik secimi pipeline'ini calistir.

        Args:
            X: Ozellik DataFrame'i
            y: Hedef degisken
            feature_cols: Aday ozellik kolon isimleri

        Returns:
            Secilen ozellikler ve analiz sonuclari
        """
        valid_cols = [c for c in feature_cols if c in X.columns]
        X_feat = X[valid_cols].copy()

        X_feat = X_feat.replace([np.inf, -np.inf], np.nan)
        X_feat = X_feat.fillna(0)

        zero_var = X_feat.columns[X_feat.std() == 0].tolist()
        if zero_var:
            logger.info(f"  {len(zero_var)} sifir varyansli ozellik atildi")
            X_feat = X_feat.drop(columns=zero_var)

        logger.info(f"  Baslangic: {len(X_feat.columns)} ozellik")

        corr_survivors, corr_dropped = self._correlation_filter(X_feat)
        logger.info(
            f"  Korelasyon filtresi: {len(corr_survivors)} kaldi, {len(corr_dropped)} atildi"
        )
        X_feat = X_feat[corr_survivors]

        mi_scores = self._mutual_info_scores(X_feat, y)
        mi_threshold = np.percentile(list(mi_scores.values()), self.mi_percentile)
        mi_survivors = [c for c, s in mi_scores.items() if s >= mi_threshold]
        mi_dropped = [c for c, s in mi_scores.items() if s < mi_threshold]
        logger.info(
            f"  MI filtresi: {len(mi_survivors)} kaldi, {len(mi_dropped)} atildi"
        )
        X_feat = X_feat[mi_survivors]

        if self.max_features and len(X_feat.columns) > self.max_features:
            top_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            top_cols = [c for c, _ in top_mi[:self.max_features] if c in X_feat.columns]
            X_feat = X_feat[top_cols]
            logger.info(f"  Max features siniri: {len(X_feat.columns)} ozellik")

        pi_scores: Dict[str, float] = {}
        pi_dropped: List[str] = []

        if use_permutation_importance and len(X_feat.columns) > 1:
            surrogate = Ridge(alpha=1.0)
            surrogate.fit(X_feat.fillna(0), y.fillna(0))
            pi_scores = self.compute_permutation_importance(surrogate, X_feat.fillna(0), y.fillna(0))
            pi_threshold = 0.0  # Negatif onem skorlu ozellikleri at (performansi dusuruyorlar)
            pi_survivors = [c for c, s in pi_scores.items() if s >= pi_threshold]
            pi_dropped = [c for c, s in pi_scores.items() if s < pi_threshold]
            if pi_dropped:
                X_feat = X_feat[pi_survivors]
                logger.info(
                    f"  PI filtresi: {len(pi_survivors)} kaldi, {len(pi_dropped)} atildi"
                )

        selected = X_feat.columns.tolist()
        logger.info(f"  Sonuc: {len(selected)} ozellik secildi")

        return {
            'selected_features': selected,
            'n_original': len(valid_cols),
            'n_selected': len(selected),
            'dropped_zero_var': zero_var,
            'dropped_correlation': corr_dropped,
            'dropped_mi': mi_dropped,
            'mi_scores': mi_scores,
            'pi_scores': pi_scores,
            'dropped_pi': pi_dropped,
        }

    def _correlation_filter(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Yuksek korelasyonlu ciftlerden birini at."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.correlation_threshold].tolist()
            if high_corr:
                to_drop.add(col)

        survivors = [c for c in X.columns if c not in to_drop]
        return survivors, list(to_drop)

    @staticmethod
    def _mutual_info_scores(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Her ozellik icin mutual information skoru hesapla."""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        y_clean = y.fillna(0)

        mi = mutual_info_regression(X_clean, y_clean, random_state=GLOBAL_SEED, n_neighbors=5)
        return dict(zip(X.columns, mi))

    @staticmethod
    def compute_permutation_importance(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
    ) -> Dict[str, float]:
        """Egitilmis model ile permutation importance hesapla.

        Args:
            model: Egitilmis sklearn-uyumlu model
            X: Test ozellikleri
            y: Test hedefi
            n_repeats: Tekrar sayisi

        Returns:
            Ozellik -> onem skoru dict'i
        """
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=GLOBAL_SEED,
            scoring='neg_mean_absolute_error',
        )
        return dict(zip(X.columns, result.importances_mean))

    @staticmethod
    def get_top_features(
        importance_dict: Dict[str, float],
        top_n: int = 20,
    ) -> List[str]:
        """Onem sirasina gore en iyi N ozellik."""
        sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in sorted_feats[:top_n]]
