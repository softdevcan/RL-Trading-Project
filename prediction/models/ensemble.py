"""
Stacking Ensemble Modeli

Birden fazla base modelin out-of-fold tahminlerini birlestirerek
bir meta-learner ile nihai tahmini uretir.

Level 1: XGBoost, LightGBM, CatBoost, BiLSTM, TFT
Level 2: Ridge Regression veya XGBoost (meta-learner)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from prediction.models.base import (
    BasePredictionModel, MODELS_DIR, models_dir, model_dirs, find_model_file,
)
from prediction.feature_engineer import PredictionFeatureEngineer
from prediction.tats import TATSCorrector
from prediction.seeding import GLOBAL_SEED

logger = logging.getLogger(__name__)


def _json_safe(obj):
    """JSON dump oncesi NaN/Inf'i None'a cevir (allow_nan=False uyumu)."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


BASE_MODEL_CONFIGS = {
    'xgboost': {'enabled': True, 'weight': 1.0},
    'lightgbm': {'enabled': True, 'weight': 1.0},
    'catboost': {'enabled': True, 'weight': 1.0},
    'bilstm': {'enabled': True, 'weight': 1.0},
    'tft': {'enabled': True, 'weight': 1.0},
}


class StackingEnsemble:
    """Stacking tabanli ensemble tahmin sistemi.

    Walk-forward out-of-fold tahminlerle meta-learner egitir.
    Geriye uyumlu PricePredictor arayuzu saglar.
    """

    def __init__(
        self,
        horizon: str = 'daily',
        model_configs: Optional[Dict[str, Dict]] = None,
        n_folds: int = 3,
        meta_learner_type: str = 'xgboost',
        use_tats: bool = True,
        source: Optional[str] = None,
        feature_groups: Optional[List] = None,
        target_type: str = 'log_return',
        strict: bool = False,
        warm_start: Optional[bool] = None,
    ):
        if horizon not in ('daily', 'weekly'):
            raise ValueError("horizon 'daily' veya 'weekly' olmali")
        if meta_learner_type not in ('ridge', 'xgboost'):
            raise ValueError("meta_learner_type 'ridge' veya 'xgboost' olmali")

        self.horizon = horizon
        self.source = source  # gold icin 'borsapy' / 'yfinance'; hisse icin None
        self.feature_groups = feature_groups  # None = registry default
        self.target_type = target_type        # 'log_return' | 'abs_price'
        self.model_configs = model_configs or BASE_MODEL_CONFIGS.copy()
        self.n_folds = n_folds
        self.meta_learner_type = meta_learner_type
        self.use_tats = use_tats
        # Faz 6 (G.1): strict=True ise bir base model egitimde patlayinca fail-fast
        # (sessiz dusme yerine). Varsayilan False — geriye uyumlu; result nesnesi
        # her durumda failed_models + status tasir.
        self.strict = strict
        self.failed_models: Dict[str, str] = {}

        # Faz 6 (2.1/B2): warm_start=True ise %80 deployment turu, %60 modelinin
        # agirliklarindan devam eder (DL state_dict / agac init_model) — sifirdan
        # degil. None verilirse config ENSEMBLE_WARM_START'tan okunur (varsayilan
        # False -> mevcut davranis, bit-es). NOT: asil hiz/kalite kazanci GPU'lu
        # makinede olculur; bu iskele default kapali.
        if warm_start is None:
            try:
                from app.core.config import get_settings
                warm_start = bool(get_settings().ENSEMBLE_WARM_START)
            except Exception:
                warm_start = False
        self.warm_start = warm_start

        self.feature_engineer = PredictionFeatureEngineer(horizon)
        self.base_models: Dict[str, BasePredictionModel] = {}
        self.meta_learner = None  # Ridge veya XGBRegressor
        self.tats_corrector: Optional[TATSCorrector] = None
        self.feature_cols: List[str] = []
        self._is_trained = False

        # Model dizini yazma aninda cozulur (kullanici calisma alani).

    # ------------------------------------------------------------------
    # Meta-learner fabrikasi
    # ------------------------------------------------------------------

    def _build_meta_learner(self):
        """meta_learner_type'a gore uygun meta-learner olustur."""
        if self.meta_learner_type == 'xgboost':
            if not _XGBOOST_AVAILABLE:
                logger.warning("XGBoost bulunamadi, Ridge'e donuluyor")
                return Ridge(alpha=1.0)
            return XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=GLOBAL_SEED,
                verbosity=0,
            )
        return Ridge(alpha=1.0)

    # ------------------------------------------------------------------
    # Model yukleme
    # ------------------------------------------------------------------

    def _get_model_class(self, model_type: str):
        from prediction.models.xgboost_model import XGBoostModel
        from prediction.models.lightgbm_model import LightGBMModel
        from prediction.models.catboost_model import CatBoostModel
        from prediction.models.lstm_model import BiLSTMModel
        from prediction.models.tft_model import TFTModel

        registry = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'catboost': CatBoostModel,
            'bilstm': BiLSTMModel,
            'tft': TFTModel,
        }
        return registry.get(model_type)

    def _create_model(self, model_type: str, params: Optional[Dict] = None) -> BasePredictionModel:
        cls = self._get_model_class(model_type)
        if cls is None:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")
        return cls(horizon=self.horizon, params=params, source=self.source)

    # ------------------------------------------------------------------
    # Egitim
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        symbol: str,
        macro_df: Optional[pd.DataFrame] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
        cross_asset_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Tum base modelleri egit ve meta-learner olustur.

        Args:
            df: OHLCV + indikator verisi (tek sembol)
            symbol: Sembol adi
            test_ratio: Test seti orani
            macro_df: Makro veriler (opsiyonel)
            fundamental_df: Fundamental veriler (opsiyonel)
            cross_asset_df: Capraz varlik verileri (opsiyonel)

        Returns:
            Egitim metrikleri
        """
        logger.info(f"[{symbol}] Ensemble egitimi basliyor ({self.horizon})...")

        feat_df = self.feature_engineer.build_features(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
            feature_groups=self.feature_groups,
            target_type=self.target_type,
        )
        self.feature_cols = self.feature_engineer.get_feature_columns(feat_df)

        if len(feat_df) < 50:
            raise ValueError(
                f"[{symbol}] Egitim icin yetersiz veri: {len(feat_df)} satir. "
                f"Feature engineering sonrasi verinin cogu NaN nedeniyle düsmüs olabilir — "
                f"makro/cross-asset veri hizalamasini kontrol edin."
            )

        X = feat_df[self.feature_cols].values.astype(np.float32)
        y = feat_df['target_price'].values.astype(np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)

        n = len(X)
        # 3-lu kronolojik split: %60 base egitim / %20 meta egitim (OOF) / %20 nihai test
        base_end = int(n * 0.60)
        meta_end = int(n * 0.80)
        split = base_end  # _build_result icin geriye uyumlu

        X_base, y_base = X[:base_end], y[:base_end]
        X_meta, y_meta = X[base_end:meta_end], y[base_end:meta_end]
        X_test, y_test = X[meta_end:], y[meta_end:]

        val_split = int(len(X_base) * 0.85)
        X_tr, X_val = X_base[:val_split], X_base[val_split:]
        y_tr, y_val = y_base[:val_split], y_base[val_split:]

        model_results = {}
        meta_predictions = {}  # base modellerin X_meta uzerindeki OOF tahminleri

        enabled_models = [
            m for m, cfg in self.model_configs.items() if cfg.get('enabled', True)
        ]

        for model_type in enabled_models:
            try:
                logger.info(f"  [{model_type}] egitiliyor...")
                model = self._create_model(model_type)
                result = model.train(X_tr, y_tr, X_val, y_val, feature_cols=self.feature_cols)
                model_results[model_type] = result

                # OOF: meta-train bolumu uzerinde tahmin (base model bu veriyi hic gormedi)
                meta_pred = model._predict_raw(X_meta)
                if len(meta_pred) != len(y_meta):
                    meta_pred = meta_pred[-len(y_meta):]
                meta_predictions[model_type] = meta_pred

                model.save(symbol, metrics=result.get('val_metrics'))
                self.base_models[model_type] = model

                logger.info(
                    f"  [{model_type}] Val MAPE: {result['val_metrics']['mape']:.2f}% | "
                    f"Dir Acc: {result['val_metrics']['direction_accuracy']:.1f}%"
                )
            except Exception as exc:
                # Faz 6 (G.1): sessiz dusme yerine kaydet; strict ise fail-fast
                self.failed_models[model_type] = str(exc)
                logger.error(f"  [{model_type}] egitim hatasi: {exc}")
                if self.strict:
                    raise RuntimeError(
                        f"[{symbol}] strict mod: '{model_type}' base modeli egitilemedi: {exc}"
                    ) from exc
                continue

        if len(meta_predictions) < 2:
            logger.warning("  Yeterli model egitilmedi, meta-learner atlanıyor")
            if meta_predictions:
                first_model = list(self.base_models.keys())[0]
                # Nihai test seti uzerinde degerlendirme
                test_pred = self.base_models[first_model]._predict_raw(X_test)
                if len(test_pred) != len(y_test):
                    test_pred = test_pred[-len(y_test):]
                test_metrics = BasePredictionModel.compute_metrics(y_test, test_pred)
            else:
                test_metrics = {'mape': 999, 'rmse': 999, 'mae': 999, 'direction_accuracy': 0}

            self._is_trained = True
            self._save_ensemble_meta(symbol, model_results, test_metrics)
            return self._build_result(symbol, model_results, test_metrics, n, split)

        # Meta-learner: X_meta OOF tahminleri uzerinde egit
        # Sondan hizala: DL modelleri kisa OOF dondurur, hepsini min_len'e indir.
        meta_X_train, meta_min_len = self._stack_aligned(meta_predictions)
        meta_y_train = y_meta[-meta_min_len:]

        self.meta_learner = self._build_meta_learner()
        self.meta_learner.fit(meta_X_train, meta_y_train)

        # ------------------------------------------------------------------
        # 1) Test metrikleri: %60-trained base modeller + mevcut meta-learner
        #    ile hesapla (donusum yapisal olarak temiz).
        # ------------------------------------------------------------------
        test_preds_per_model = {}
        for model_type, model in self.base_models.items():
            try:
                tp = model._predict_raw(X_test)
                if len(tp) != len(y_test):
                    tp = tp[-len(y_test):]
                test_preds_per_model[model_type] = tp
            except Exception as exc:
                logger.warning(f"  [{model_type}] test tahmin hatasi: {exc}")

        if len(test_preds_per_model) >= 2:
            test_meta_X, _ = self._stack_aligned(test_preds_per_model)
            ensemble_pred = self.meta_learner.predict(test_meta_X)
        elif test_preds_per_model:
            only = list(test_preds_per_model.values())[0]
            ensemble_pred = only
        else:
            ensemble_pred = np.zeros_like(y_test)

        test_metrics = BasePredictionModel.compute_metrics(y_test, ensemble_pred)

        logger.info(
            f"  [ENSEMBLE] Test MAPE: {test_metrics['mape']:.2f}% | "
            f"Dir Acc: {test_metrics['direction_accuracy']:.1f}%"
        )

        model_agreement = self._compute_model_agreement(test_preds_per_model)
        test_metrics['model_agreement'] = model_agreement

        # TATS: meta-train seti uzerinde trend siniflandirici egit
        if self.use_tats and len(meta_predictions) >= 1:
            self.tats_corrector = TATSCorrector()
            meta_X_for_tats, tats_min_len = self._stack_aligned(meta_predictions)
            tats_y = y_meta[-tats_min_len:]
            current_prices_meta = y_base[-tats_min_len:]
            tats_result = self.tats_corrector.fit(meta_X_for_tats, tats_y, current_prices_meta)
            test_metrics['tats_trained'] = tats_result.get('trained', False)

        # ------------------------------------------------------------------
        # 2) Deployment: base modelleri %80 veri ile yeniden egit.
        #    Sebep: %60-trained modeller non-stationary varliklarda (altin,
        #    doviz) guncel fiyat seviyesini hic gormedigi icin predict
        #    zamaninda distribution shift yasanir. Test metrikleri yukarida
        #    temiz sekilde hesaplandi; bu adim sadece diske yazilan modeli
        #    gunceller.
        # ------------------------------------------------------------------
        X_full_train = X[:meta_end]
        y_full_train = y[:meta_end]
        full_val_split = int(len(X_full_train) * 0.85)
        X_ft, X_fv = X_full_train[:full_val_split], X_full_train[full_val_split:]
        y_ft, y_fv = y_full_train[:full_val_split], y_full_train[full_val_split:]

        new_meta_preds: Dict[str, np.ndarray] = {}
        for model_type in list(self.base_models.keys()):
            try:
                logger.info(f"  [{model_type}] tam egitim seti uzerinde yeniden egitiliyor (%80)...")
                # Faz 6 (2.1): warm_start ise %60-modelini kaynak ver — bu satirda
                # self.base_models[model_type] hala %60-trained model (asagida
                # overwrite ediliyor). Kapaliyken None -> sifirdan (bit-es).
                prev_model = self.base_models.get(model_type) if self.warm_start else None
                full_model = self._create_model(model_type)
                full_result = full_model.train(
                    X_ft, y_ft, X_fv, y_fv,
                    feature_cols=self.feature_cols,
                    warm_start_from=prev_model,
                )
                # Meta-set uzerinde yeni tahminler (meta-learner yeniden egitimi icin)
                mp = full_model._predict_raw(X_meta)
                if len(mp) != len(y_meta):
                    mp = mp[-len(y_meta):]
                new_meta_preds[model_type] = mp
                full_model.save(symbol, metrics=full_result.get('val_metrics'))
                self.base_models[model_type] = full_model
                logger.info(
                    f"  [{model_type}] Yeniden egitim Val MAPE: "
                    f"{full_result['val_metrics']['mape']:.2f}%"
                )
            except Exception as exc:
                logger.warning(f"  [{model_type}] yeniden egitim hatasi, orijinal model korunuyor: {exc}")

        # Meta-learner'i yeni base model tahminleri uzerinde yeniden fit et
        if len(new_meta_preds) >= 2:
            new_meta_X, new_min_len = self._stack_aligned(new_meta_preds)
            self.meta_learner = self._build_meta_learner()
            self.meta_learner.fit(new_meta_X, y_meta[-new_min_len:])
            logger.info("  [META] Meta-learner %80-trained modeller ile yeniden egitildi")

        self._is_trained = True
        self._save_ensemble_meta(symbol, model_results, test_metrics)

        return self._build_result(symbol, model_results, test_metrics, n, split)

    @staticmethod
    def _stack_aligned(predictions: Dict[str, np.ndarray]):
        """Model tahminlerini ortak minimum uzunluga (sondan) hizalayip stack'le.

        Sekans modelleri (BiLSTM/TFT) lookback yuzunden agac modellerinden kisa
        OOF tahmini dondurur; `column_stack` esit uzunluk ister. Hepsini sondan
        `min_len` kadar kirparak hizalar ve (stacked_X, min_len) dondururuz.
        Bu olmadan DL modelleri meta-katmanda column_stack'i patlatir ve sessizce
        duserler (bkz. Faz 6 R1).
        """
        keys = sorted(predictions.keys())
        min_len = min(len(predictions[k]) for k in keys)
        stacked = np.column_stack([predictions[k][-min_len:] for k in keys])
        return stacked, min_len

    def _compute_model_agreement(self, predictions: Dict[str, np.ndarray]) -> float:
        """Modellerin yon tahmini uzerindeki uzlasma orani."""
        if len(predictions) < 2:
            return 1.0

        arrays = list(predictions.values())
        min_len = min(len(a) for a in arrays)
        dirs = []
        for arr in arrays:
            d = np.diff(arr[-min_len:]) > 0
            dirs.append(d)
        dirs = np.array(dirs)
        agreement = np.mean(np.all(dirs == dirs[0], axis=0))
        return round(float(agreement), 4)

    def _build_result(self, symbol, model_results, test_metrics, n, split):
        # Faz 6 (G.1): eksik model varsa status='degraded' — sessiz "basarili" yok
        expected = {m for m, cfg in self.model_configs.items() if cfg.get('enabled', True)}
        trained = set(model_results.keys())
        missing = sorted(expected - trained)
        status = 'ok' if not missing else 'degraded'
        return {
            'symbol': symbol,
            'horizon': self.horizon,
            'status': status,
            'missing_models': missing,
            'failed_models': dict(self.failed_models),
            'n_total': n,
            'n_train': split,
            'n_test': n - split,
            'n_features': len(self.feature_cols),
            'n_models': len(self.base_models),
            'models_trained': list(model_results.keys()),
            'model_results': {k: v.get('val_metrics', {}) for k, v in model_results.items()},
            'ensemble_test_metrics': test_metrics,
            'warm_start': self.warm_start,   # Faz 6 (2.1): %80 turu warm-start miydi
            'trained_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------

    def predict_next(
        self,
        df: pd.DataFrame,
        symbol: str,
        macro_df: Optional[pd.DataFrame] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
        cross_asset_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Sonraki periyot icin ensemble tahmini uret.

        Geriye uyumlu: eski PricePredictor.predict_next() ile ayni cikti.
        """
        if not self._is_trained:
            self.load(symbol)

        feat_df = self.feature_engineer.build_features(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
            feature_groups=self.feature_groups,
            target_type=self.target_type,
        )

        # Egitimdeki feature listesiyle hizala: eksik sutunlari 0 ile doldur, sirayi koru.
        # (Predict zamaninda kisa pencere/eksik makro yuzunden bazi feature'lar tamamen NaN olup
        # build_features tarafindan drop edilmis olabilir.)
        missing_cols = [c for c in self.feature_cols if c not in feat_df.columns]
        if missing_cols:
            logger.warning(
                f"  [{symbol}] {len(missing_cols)} feature predict zamaninda uretilemedi, "
                f"0 ile doldurulacak: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}"
            )
            for c in missing_cols:
                feat_df[c] = 0.0

        aligned = feat_df.reindex(columns=self.feature_cols)

        last_row = aligned.iloc[-1:].values.astype(np.float32)
        last_row = np.nan_to_num(last_row, nan=0.0, posinf=0.0, neginf=0.0)

        model_predictions = {}
        direction_probs = {}
        full_X = aligned.values.astype(np.float32)
        full_X = np.nan_to_num(full_X, nan=0.0, posinf=0.0, neginf=0.0)

        for model_type, model in self.base_models.items():
            try:
                pred = model._predict_raw(full_X)
                model_predictions[model_type] = float(pred[-1])

                dir_raw = model._predict_direction_raw(full_X)
                if dir_raw is not None and len(dir_raw) > 0:
                    direction_probs[model_type] = float(dir_raw[-1])
            except Exception as exc:
                logger.warning(f"  [{model_type}] tahmin hatasi: {exc}")
                continue

        if self.meta_learner and len(model_predictions) >= 2:
            meta_input = np.array([[model_predictions[m] for m in sorted(model_predictions.keys())]])
            expected_n = self._meta_learner_n_features()
            if expected_n is not None and meta_input.shape[1] != expected_n:
                logger.warning(
                    f"  Meta-learner {expected_n} model bekliyor, "
                    f"{meta_input.shape[1]} mevcut ({list(model_predictions.keys())}). "
                    f"Basit ortalamaya donuluyor."
                )
                raw_pred = float(np.mean(list(model_predictions.values())))
            else:
                raw_pred = float(self.meta_learner.predict(meta_input)[0])
        elif model_predictions:
            raw_pred = float(np.mean(list(model_predictions.values())))
        else:
            raise RuntimeError("Hicbir model tahmin uretemedi")

        current_close = float(df['close'].iloc[-1])

        # log_return hedefi kullanildiysa gercek fiyata cevir
        if self.target_type == 'log_return':
            predicted_close = current_close * np.exp(raw_pred)
        else:
            predicted_close = raw_pred

        # TATS: trend siniflandirici ile tahmin duzelt
        tats_meta: Dict[str, Any] = {'applied': False}
        if self.use_tats and self.tats_corrector and self.tats_corrector.is_fitted():
            meta_input_for_tats = np.array(
                [model_predictions[m] for m in sorted(model_predictions.keys())]
            )
            predicted_close, tats_meta = self.tats_corrector.correct(
                predicted_close, current_close, meta_input_for_tats
            )

        change_pct = (predicted_close - current_close) / current_close * 100
        direction = 'UP' if change_pct > 0 else 'DOWN'

        # Confidence: direction head varsa ondan, yoksa fiyat oylamasiyla
        total_models = len(model_predictions)
        if direction_probs:
            mean_dir_prob = float(np.mean(list(direction_probs.values())))
            confidence = round(max(mean_dir_prob, 1 - mean_dir_prob), 4)
            agreement = mean_dir_prob if direction == 'UP' else (1 - mean_dir_prob)
        else:
            up_votes = sum(1 for p in model_predictions.values() if p > current_close)
            agreement = up_votes / total_models if total_models > 0 else 0.5
            confidence = round(max(agreement, 1 - agreement), 4)
        agreement = round(float(agreement), 4)

        if self.horizon == 'daily':
            pred_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            pred_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

        return {
            'symbol': symbol,
            'prediction_date': pred_date,
            'horizon': self.horizon,
            'predicted_close': round(predicted_close, 4),
            'predicted_direction': direction,
            'predicted_change_pct': round(change_pct, 2),
            'confidence': confidence,
            'current_close': round(current_close, 4),
            'model_predictions': {k: round(v, 4) for k, v in model_predictions.items()},
            'ensemble_agreement': round(agreement, 4),
            'n_models': total_models,
            'tats': tats_meta,
            'made_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Kaydetme / Yukleme
    # ------------------------------------------------------------------

    def _meta_learner_n_features(self) -> Optional[int]:
        """Meta-learner'in bekledighi girdi feature (model) sayisi."""
        if self.meta_learner is None:
            return None
        try:
            if hasattr(self.meta_learner, 'n_features_in_'):
                return int(self.meta_learner.n_features_in_)
            if hasattr(self.meta_learner, 'coef_'):
                return int(self.meta_learner.coef_.shape[0])
        except Exception:
            pass
        return None

    def _model_id(self, symbol: str) -> str:
        safe = symbol.replace('/', '_').replace('=', '_').replace('.', '_')
        return f'{safe}__{self.source}' if self.source else safe

    def _ensemble_meta_name(self, symbol: str) -> str:
        return f'{self._model_id(symbol)}_{self.horizon}_ensemble_meta.json'

    def _ensemble_meta_path(self, symbol: str) -> str:
        """Yazma yolu. Okuma icin _find_ensemble_meta kullanilir."""
        return os.path.join(models_dir(), self._ensemble_meta_name(symbol))

    def _find_ensemble_meta(self, symbol: str) -> Optional[str]:
        """Once kullanicinin calisma alani, sonra ortak (eski) dizin."""
        return find_model_file(self._ensemble_meta_name(symbol))

    def _meta_learner_path(self, symbol: str) -> str:
        return os.path.join(models_dir(), f'{self._model_id(symbol)}_{self.horizon}_meta_learner.json')

    def _save_ensemble_meta(self, symbol, model_results, test_metrics):
        meta = {
            'symbol': symbol,
            'source': self.source,
            'horizon': self.horizon,
            'feature_cols': self.feature_cols,
            'models_trained': list(self.base_models.keys()),
            'model_configs': self.model_configs,
            'meta_learner_type': self.meta_learner_type,
            'test_metrics': _json_safe(test_metrics),
            'saved_at': datetime.now().isoformat(),
            # Faz 4: feature konfigurasyon kaydi
            'feature_groups': self.feature_groups,
            'target_type': self.target_type,
        }
        if self.meta_learner:
            if self.meta_learner_type == 'xgboost' and _XGBOOST_AVAILABLE:
                self.meta_learner.save_model(self._meta_learner_path(symbol))
                meta['meta_learner_saved'] = True
            else:
                # Ridge: katsayilari JSON'a göm (geriye uyumluluk)
                meta['meta_learner_coefs'] = self.meta_learner.coef_.tolist()
                meta['meta_learner_intercept'] = float(self.meta_learner.intercept_)

        path = self._ensemble_meta_path(symbol)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, default=str)

    def save(self, symbol: str, **kwargs):
        """Tum modelleri kaydet."""
        for model_type, model in self.base_models.items():
            try:
                model.save(symbol)
            except Exception as exc:
                logger.warning(f"  [{model_type}] kaydetme hatasi: {exc}")

    def load(self, symbol: str):
        """Ensemble meta ve tum base modelleri yukle.

        Once kullanicinin calisma alanina, yoksa ortak (kullanici oncesi)
        model dizinine bakilir.
        """
        meta_path = self._find_ensemble_meta(symbol)
        if not meta_path:
            raise FileNotFoundError(
                f"Ensemble meta bulunamadi: {self._ensemble_meta_path(symbol)}"
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.feature_cols = meta.get('feature_cols', [])
        models_trained = meta.get('models_trained', [])
        # Faz 4: feature konfigurasyon geri yukle
        self.feature_groups = meta.get('feature_groups')
        self.target_type = meta.get('target_type', 'log_return')

        for model_type in models_trained:
            try:
                model = self._create_model(model_type)
                model.load(symbol)
                self.base_models[model_type] = model
            except Exception as exc:
                logger.warning(f"  [{model_type}] yukleme hatasi: {exc}")

        self.meta_learner_type = meta.get('meta_learner_type', 'ridge')
        # Meta-learner dosyasi meta ile ayni dizinde durur.
        ml_path = os.path.join(
            os.path.dirname(meta_path),
            f'{self._model_id(symbol)}_{self.horizon}_meta_learner.json',
        )
        if meta.get('meta_learner_saved') and _XGBOOST_AVAILABLE and os.path.exists(ml_path):
            self.meta_learner = XGBRegressor()
            self.meta_learner.load_model(ml_path)
        elif meta.get('meta_learner_coefs'):
            self.meta_learner = Ridge(alpha=1.0)
            self.meta_learner.coef_ = np.array(meta['meta_learner_coefs'])
            self.meta_learner.intercept_ = meta['meta_learner_intercept']
            self.meta_learner.n_features_in_ = len(meta['meta_learner_coefs'])

        self._is_trained = True
        logger.info(f"  Ensemble yuklendi: {len(self.base_models)} model")

    def is_trained(self, symbol: str) -> bool:
        return self._find_ensemble_meta(symbol) is not None

    def list_trained_models(self) -> list:
        models = []
        seen = set()
        # Kullanicinin kendi modelleri once; ayni isimli eski ortak model golgelenir.
        for directory in model_dirs():
            for fname in sorted(os.listdir(directory)):
                if not fname.endswith(f'_{self.horizon}_ensemble_meta.json') or fname in seen:
                    continue
                seen.add(fname)
                path = os.path.join(directory, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    models.append(info)
                except Exception:
                    models.append({'file': fname, 'horizon': self.horizon})
        return models
