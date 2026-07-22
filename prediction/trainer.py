"""
Gelismis Egitim Pipeline

Walk-forward cross-validation, purge gap, embargo,
multi-model egitim ve deney takibi.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from prediction.feature_engineer import PredictionFeatureEngineer
from prediction.feature_selector import FeatureSelector
from prediction.models.base import BasePredictionModel
from prediction.seeding import GLOBAL_SEED, seed_everything

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = os.path.join('results', 'prediction_experiments')


class WalkForwardTrainer:
    """Zaman serisi walk-forward cross-validation ile egitim.

    Ozellikler:
    - Expanding window walk-forward
    - Purge gap (train/test arasi bosluk) — data leakage onleme
    - Embargo periodu (test sonrasi)
    - Multi-model paralel egitim
    - Ozellik secimi entegrasyonu
    - Deney takibi (JSON log)
    """

    def __init__(
        self,
        horizon: str = 'daily',
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 3,
        select_features: bool = True,
        max_features: Optional[int] = 80,
        source: Optional[str] = None,
        feature_groups: Optional[list] = None,
        target_type: str = 'log_return',
    ):
        """
        Args:
            horizon: 'daily' veya 'weekly'
            n_splits: Walk-forward fold sayisi
            purge_days: Train/test arasi gun boşlugu
            embargo_days: Test seti sonrasi gun boşlugu
            select_features: Otomatik ozellik secimi uygula
            max_features: Maksimum ozellik sayisi
            feature_groups: Aktif feature grup id listesi (None = registry default)
            target_type: 'log_return' (onerilen) veya 'abs_price'
        """
        self.horizon = horizon
        self.source = source
        self.feature_groups = feature_groups
        self.target_type = target_type
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.select_features = select_features
        self.max_features = max_features
        self.feature_engineer = PredictionFeatureEngineer(horizon)

        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    def cross_validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        model_types: Optional[List[str]] = None,
        macro_df: Optional[pd.DataFrame] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
        cross_asset_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Walk-forward CV ile model degerlendirme.

        Args:
            df: OHLCV + indikator verisi (tek sembol)
            symbol: Sembol adi
            model_types: Degerlendirilecek model tipleri (None = hepsi)
            macro_df: Makro veriler
            fundamental_df: Fundamental veriler
            cross_asset_df: Capraz varlik verileri

        Returns:
            Her model ve fold icin metrikler
        """
        from prediction.models import MODEL_REGISTRY

        if model_types is None:
            model_types = list(MODEL_REGISTRY.keys())

        logger.info(f"[{symbol}] Walk-forward CV basliyor ({self.n_splits} fold)...")

        feat_df = self.feature_engineer.build_features(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )
        feature_cols = self.feature_engineer.get_feature_columns(feat_df)

        X = feat_df[feature_cols].values.astype(np.float32)
        y = feat_df['target_price'].values.astype(np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)

        if self.select_features:
            selector = FeatureSelector(max_features=self.max_features)
            X_df = pd.DataFrame(X, columns=feature_cols)
            y_series = pd.Series(y)
            selection = selector.fit_select(X_df, y_series, feature_cols)
            selected_cols = selection['selected_features']
            selected_indices = [feature_cols.index(c) for c in selected_cols if c in feature_cols]
            X = X[:, selected_indices]
            feature_cols = selected_cols
            logger.info(f"  Ozellik secimi: {selection['n_original']} -> {selection['n_selected']}")

        n = len(X)
        fold_size = n // (self.n_splits + 1)

        all_results = {mt: [] for mt in model_types}

        prev_test_end = 0  # Onceki fold'un test_end'i (embargo icin)
        for fold in range(self.n_splits):
            # Expanding window: fold_size * (fold+1) noktasina kadar egit
            # Embargo: bir onceki fold'un test bolgesi bitmeden train setine ekleme
            train_end = fold_size * (fold + 1)
            if prev_test_end > 0:
                embargo_boundary = prev_test_end + self.embargo_days
                train_end = min(train_end, embargo_boundary)

            test_start = fold_size * (fold + 1) + self.purge_days
            test_end = min(test_start + fold_size, n)

            if test_start >= n or test_end - test_start < 20 or train_end < 20:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]

            val_split = int(len(X_train) * 0.85)
            X_tr, X_val = X_train[:val_split], X_train[val_split:]
            y_tr, y_val = y_train[:val_split], y_train[val_split:]

            logger.info(
                f"  Fold {fold + 1}/{self.n_splits}: "
                f"train=[0:{train_end}], purge={self.purge_days}d, "
                f"test=[{test_start}:{test_end}], embargo={self.embargo_days}d"
            )

            prev_test_end = test_end

            for model_type in model_types:
                try:
                    model_cls = MODEL_REGISTRY[model_type]
                    model = model_cls(horizon=self.horizon)
                    model.train(X_tr, y_tr, X_val, y_val, feature_cols=feature_cols)

                    test_pred = model._predict_raw(X_test)
                    if len(test_pred) != len(y_test):
                        test_pred = test_pred[-len(y_test):]

                    metrics = BasePredictionModel.compute_metrics(y_test, test_pred)
                    metrics['fold'] = fold + 1
                    all_results[model_type].append(metrics)

                    logger.info(
                        f"    [{model_type}] MAPE={metrics['mape']:.2f}% "
                        f"DirAcc={metrics['direction_accuracy']:.1f}%"
                    )
                except Exception as exc:
                    logger.error(f"    [{model_type}] Fold {fold + 1} hatasi: {exc}")

        summary = self._summarize_cv_results(all_results, symbol)

        self._save_experiment(summary, symbol)

        return summary

    def train_final_models(
        self,
        df: pd.DataFrame,
        symbol: str,
        test_ratio: float = 0.2,
        optimize_first: bool = False,
        n_hpo_trials: int = 30,
        macro_df: Optional[pd.DataFrame] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
        cross_asset_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Son modelleri egit (CV sonrasi).

        Args:
            df: OHLCV + indikator verisi
            symbol: Sembol adi
            test_ratio: Test seti orani
            optimize_first: Oncelikle HPO calistir
            n_hpo_trials: HPO deneme sayisi
            macro_df: Makro veriler
            fundamental_df: Fundamental veriler
            cross_asset_df: Capraz varlik verileri

        Returns:
            Ensemble egitim sonuclari
        """
        from prediction.models.ensemble import StackingEnsemble

        # Faz 6 (G.5): kosum basinda surec RNG'lerini tohumla — "ayni seed +
        # seri mod -> tekrar uretilebilir". Model random_state'leri zaten
        # GLOBAL_SEED'den geliyor; bu, numpy/torch/random global durumunu da
        # sabitler (DL agirlik init, feature-sel permutation vb.).
        seed_everything(GLOBAL_SEED)

        logger.info(f"[{symbol}] Final ensemble egitimi basliyor...")
        start_time = time.time()

        if optimize_first:
            self._run_hpo(df, symbol, n_hpo_trials, macro_df, fundamental_df, cross_asset_df)

        # NOTE: test_ratio parametresi StackingEnsemble'in 3-way 60/20/20 split'ine
        # gecilmesiyle devre disi kaldi; imzada tutuluyor ki upstream callers kirilmasin.
        _ = test_ratio
        ensemble = StackingEnsemble(
            horizon=self.horizon,
            source=self.source,
            feature_groups=self.feature_groups,
            target_type=self.target_type,
        )
        result = ensemble.train(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )

        elapsed = time.time() - start_time
        result['training_time_seconds'] = round(elapsed, 1)
        logger.info(f"  Egitim tamamlandi: {elapsed:.1f}s")

        return result

    def _run_hpo(
        self,
        df: pd.DataFrame,
        symbol: str,
        n_trials: int,
        macro_df=None,
        fundamental_df=None,
        cross_asset_df=None,
    ):
        """HPO calistir ve en iyi parametreleri kaydet."""
        from prediction.hyperopt import PredictionHyperOptimizer

        logger.info(f"  HPO basliyor ({n_trials} trial)...")

        feat_df = self.feature_engineer.build_features(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )
        feature_cols = self.feature_engineer.get_feature_columns(feat_df)
        X = feat_df[feature_cols].values.astype(np.float32)
        y = feat_df['target_price'].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        optimizer = PredictionHyperOptimizer(n_trials=n_trials, n_cv_splits=3)
        optimizer.optimize_all_models(X, y, self.horizon)

    def _summarize_cv_results(
        self,
        all_results: Dict[str, List[Dict]],
        symbol: str,
    ) -> Dict[str, Any]:
        """CV sonuclarini ozetle."""
        summary = {
            'symbol': symbol,
            'horizon': self.horizon,
            'n_folds': self.n_splits,
            'purge_days': self.purge_days,
            'embargo_days': self.embargo_days,
            'model_results': {},
            'best_model': None,
            'best_avg_mape': float('inf'),
        }

        for model_type, fold_results in all_results.items():
            if not fold_results:
                continue

            avg_mape = float(np.mean([r['mape'] for r in fold_results]))
            avg_rmse = float(np.mean([r['rmse'] for r in fold_results]))
            avg_dir = float(np.mean([r['direction_accuracy'] for r in fold_results]))
            std_mape = float(np.std([r['mape'] for r in fold_results]))

            model_summary = {
                'model_type': model_type,
                'n_folds_completed': len(fold_results),
                'avg_mape': round(avg_mape, 4),
                'std_mape': round(std_mape, 4),
                'avg_rmse': round(avg_rmse, 4),
                'avg_direction_accuracy': round(avg_dir, 2),
                'fold_results': fold_results,
            }
            summary['model_results'][model_type] = model_summary

            if avg_mape < summary['best_avg_mape']:
                summary['best_avg_mape'] = round(avg_mape, 4)
                summary['best_model'] = model_type

            logger.info(
                f"  [{model_type}] CV Ortalama — "
                f"MAPE: {avg_mape:.2f}% (±{std_mape:.2f}) | "
                f"DirAcc: {avg_dir:.1f}%"
            )

        if summary['best_model']:
            logger.info(f"  En iyi model: {summary['best_model']} (MAPE: {summary['best_avg_mape']:.2f}%)")

        return summary

    def _save_experiment(self, result: Dict[str, Any], symbol: str):
        """Deney sonuclarini kaydet."""
        safe = symbol.replace('/', '_').replace('=', '_').replace('.', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(EXPERIMENTS_DIR, f'{safe}_{self.horizon}_{timestamp}.json')

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"  Deney sonuclari kaydedildi: {path}")
