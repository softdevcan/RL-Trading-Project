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

from prediction.tracker_rules import horizon_sessions
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


CONFIDENCE_MODES = ('legacy', 'calibrated')
MIN_CALIB_ROWS = 50  # F2: siniflandirici ve sicaklik yarilarinin her biri icin en az satir


class StackingEnsemble:
    """Stacking tabanli ensemble tahmin sistemi.

    Walk-forward out-of-fold tahminlerle meta-learner egitir.
    Geriye uyumlu PricePredictor arayuzu saglar.
    """

    # F2: yon siniflandiricisi/sicaklik yarilari arasinda ve meta sonunda arindirma (None = h).
    # Yalniz test mutasyonu icin degistirilir.
    CALIB_PURGE: Optional[int] = None

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
        confidence_mode: str = 'legacy',
    ):
        if horizon not in ('daily', 'weekly'):
            raise ValueError("horizon 'daily' veya 'weekly' olmali")
        if meta_learner_type not in ('ridge', 'xgboost'):
            raise ValueError("meta_learner_type 'ridge' veya 'xgboost' olmali")

        # Faz 9 F1-F4: 'legacy' (varsayilan; davranis dondurma) yon = fiyat tahmininin isareti,
        # guven = DL yon baslarinin ortalamasindan max(p, 1-p) (Y4: ikisi zit yonu gosterebilir).
        # 'calibrated': p(UP) meta duzeyde lojistik siniflandiricidan (OOF'un ilk yarisi),
        # sicaklik olcekleme ikinci yarida; yon = 1[p_cal >= 0,5], guven = max(p_cal, 1-p_cal).
        # Fiyat tahmini iki modda da AYNI. Her iki modda test kalibrasyonu olculur (F3).
        if confidence_mode not in CONFIDENCE_MODES:
            raise ValueError(f"confidence_mode {confidence_mode!r}: {CONFIDENCE_MODES}")
        self.confidence_mode = confidence_mode
        self.direction_model: Optional[Dict[str, Any]] = None
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
        model = cls(horizon=self.horizon, params=params, source=self.source)
        model.target_type = self.target_type  # B11: DL yon basi etiketi ham hedeften
        return model

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
            # Faz 9 B10 (Y20): "mevcut fiyat" meta satirlarinin KENDI kapanisi (eskiden
            # taban bolumunun son hedefleri y_base idi) ve etiket gerceklesen getiriden.
            # Kapanis HAM df'ten alinir, y'den turetilmez (hakem-2 B10 #B1): ICEEMDAN
            # kolunda feat_df['close'] filtreli seriyle uzerine yaziliyor; Makale 2'nin
            # TATS kollari (K3/K4) tam da o kollar. Hizalama OOF ile ayni (sondan k).
            close_meta, future_meta = self._tats_inputs(df, feat_df, base_end, meta_end, self.horizon)
            close_meta, future_meta = close_meta[-tats_min_len:], future_meta[-tats_min_len:]
            # K21: meta bolumunun son h satirinin etiketi test donemindeki kapanisa bakar
            # (hakem-finans B10 #3) -> TATS test metrigine girdigi icin uyumdan atilir.
            h = 1 if self.horizon == 'daily' else 5
            tats_result = self.tats_corrector.fit(
                meta_X_for_tats[:-h], future_meta[:-h], close_meta[:-h])
            test_metrics['tats_trained'] = tats_result.get('trained', False)
            # K21: test bolumunde TATS'li ve TATS'siz birlikte (ayni %60 modelleri, ayni satirlar)
            if test_metrics['tats_trained'] and len(test_preds_per_model) >= 2:
                test_metrics['tats'] = self._evaluate_tats_on_test(
                    df, feat_df, meta_end, test_meta_X, ensemble_pred, y_test)

        # Faz 9 F1-F4: test kalibrasyonu (her iki modda) + 'calibrated' modda yon modeli.
        # %60 modelleri hala self.base_models'ta (asagida %80 ile degisir).
        if len(test_preds_per_model) >= 2:
            try:
                test_metrics['calibration'] = self._direction_calibration(
                    df, feat_df, base_end, meta_end, n, meta_predictions, test_preds_per_model,
                    ensemble_pred, X_meta, X_test)
            except Exception as exc:
                logger.warning(f"  [KALIBRASYON] olculemedi: {exc}")
                test_metrics['calibration'] = {'error': str(exc)}
                if self.confidence_mode == 'calibrated' and self.strict:
                    raise

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

    def _evaluate_tats_on_test(self, df, feat_df, meta_end, test_meta_X, ensemble_pred, y_test):
        """K21: test bolumunde TATS duzeltmesini uygula; TATS'siz ile ayni satirlarda metrik.

        Duzeltme `predict_next` ile ayni yoldan: hedef fiyata cevrilir, `correct` uygulanir,
        hedef birimine geri cevrilir. Mevcut fiyat ham kapanis (B10).
        """
        k = len(test_meta_X)
        raw = df['close'].sort_index()
        cur = raw.reindex(feat_df.index[meta_end:][-k:]).to_numpy(dtype=np.float64)
        base = np.asarray(ensemble_pred, dtype=np.float64)[-k:]
        corrected = base.copy()
        n_applied = 0
        for i in range(k):
            pc = cur[i] * np.exp(base[i]) if self.target_type == 'log_return' else base[i]
            new_pc, meta = self.tats_corrector.correct(pc, cur[i], test_meta_X[i])
            if meta.get('applied'):
                n_applied += 1
                corrected[i] = np.log(new_pc / cur[i]) if self.target_type == 'log_return' else new_pc
        y_t = np.asarray(y_test, dtype=np.float64)[-k:]

        def tda(pred):
            # Makaledeki TDA (Kazemdehbashi 2026, Denk. 3.1): tahmin edilen DEGISIMIN isareti.
            # compute_metrics'in direction_accuracy'si fark tabanli (np.diff) ve TDA degil
            # (hakem-2 K21 B1). log_return: sign(pred) == sign(y); fiyat: cari kapanisa gore.
            if self.target_type == 'log_return':
                pd_, td_ = pred, y_t
            else:
                pd_, td_ = pred - cur, y_t - cur
            return float(np.mean(np.sign(pd_) == np.sign(td_)) * 100)

        m_base = BasePredictionModel.compute_metrics(y_t, base)
        m_corr = BasePredictionModel.compute_metrics(y_t, corrected)
        m_base['tda'], m_corr['tda'] = tda(base), tda(corrected)
        return {'n_rows': int(k), 'n_applied': int(n_applied),
                'without_tats': m_base, 'with_tats': m_corr}

    # ------------------------------------------------------------------
    # Faz 9 F1-F4: yon olasiligi ve kalibrasyon
    # ------------------------------------------------------------------

    def _ret_of(self, pred: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Hedef birimindeki tahmin -> log getiri (log_return hedefinde aynen)."""
        pred = np.asarray(pred, dtype=np.float64)
        if self.target_type == 'log_return':
            return pred
        return np.log(np.clip(pred, 1e-12, None) / np.clip(close, 1e-12, None))

    def _direction_heads(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        heads = {}
        for k, m in self.base_models.items():
            try:
                d = m._predict_direction_raw(X)
            except Exception:
                d = None
            if d is not None and len(d) > 0:
                heads[k] = np.asarray(d, dtype=np.float64)
        return heads

    def _direction_features(self, preds: Dict[str, np.ndarray], heads: Dict[str, np.ndarray],
                            close: np.ndarray):
        """Siniflandirici girdisi: taban getiri tahminleri + (varsa) DL yon baslari - 0,5.

        Hepsi SONDAN ortak uzunluga hizalanir (DL modelleri kisa dondurur). close da sondan.
        Returns: (anahtarlar, X (L x k), L)
        """
        cols = {f'ret_{k}': v for k, v in preds.items()}
        cols.update({f'dir_{k}': v for k, v in heads.items()})
        keys = sorted(cols)
        L = min([len(cols[k]) for k in keys] + [len(close)])
        c = np.asarray(close, dtype=np.float64)[-L:]
        X = np.column_stack([
            self._ret_of(cols[k][-L:], c) if k.startswith('ret_') else np.asarray(cols[k][-L:]) - 0.5
            for k in keys])
        return keys, X, L

    def _legacy_p_up(self, preds: Dict[str, np.ndarray], heads: Dict[str, np.ndarray],
                     ens_pred: np.ndarray, close: np.ndarray):
        """Eski yolun ima ettigi p(UP) ve yon catismasi (Y4), predict_next ile ayni kural.

        Eski cikti: yon fiyat tahmininden, guven DL yon baslarinin ortalamasindan
        max(p, 1-p) (bas yoksa fiyat oylamasi). Guven 'tahmin edilen yonun olasiligi' gibi
        sunuldugu icin ima edilen p(UP) = guven (yon UP) ya da 1 - guven (yon DOWN).
        """
        L = min([len(v) for v in list(preds.values()) + list(heads.values())]
                + [len(ens_pred), len(close)])
        c = np.asarray(close, dtype=np.float64)[-L:]
        up_price = self._ret_of(np.asarray(ens_pred)[-L:], c) > 0
        if heads:
            p_head = np.mean([v[-L:] for v in heads.values()], axis=0)
            conf = np.maximum(p_head, 1 - p_head)
            conflict = (p_head >= 0.5) != up_price
        else:
            votes = np.mean([self._ret_of(v[-L:], c) > 0 for v in preds.values()], axis=0)
            conf = np.maximum(votes, 1 - votes)
            conflict = np.zeros(L, dtype=bool)
        p_up = np.where(up_price, conf, 1 - conf)
        return p_up, conflict, up_price, L

    def _direction_calibration(self, df, feat_df, base_end, meta_end, n, meta_preds,
                               test_preds, ens_pred_test, X_meta, X_test) -> Dict[str, Any]:
        """F1-F3: test bolumunde eski ve kalibre yon olasiliginin kalibrasyonu.

        Etiket ham kapanistan: y = 1[kapanis_{t+h} > kapanis_t] (B10 ile ayni girdi).
        Siniflandirici OOF meta bolumunun ilk yarisinda, sicaklik ikinci yarisinda uydurulur;
        yarilar arasinda ve meta bolumunun sonunda h satir arindirilir (etiket bir sonraki
        bolume bakar). Test bolumunde yalniz degerlendirme (Guo vd. 2017 §4).
        """
        from sklearn.linear_model import LogisticRegression
        from evaluation import calibration as cal

        h = 1 if self.horizon == 'daily' else 5
        h = h if self.CALIB_PURGE is None else int(self.CALIB_PURGE)
        close_m, fut_m = self._tats_inputs(df, feat_df, base_end, meta_end, self.horizon)
        close_t, fut_t = self._tats_inputs(df, feat_df, meta_end, n, self.horizon)
        y_m_all = (fut_m > close_m).astype(np.float64)
        y_t_all = (fut_t > close_t).astype(np.float64)
        heads_m, heads_t = self._direction_heads(X_meta), self._direction_heads(X_test)

        # Test: eski yolun kalibrasyonu ve bas-fiyat catisma orani (her iki modda; F3, Y4)
        p_leg, conflict_leg, _, Lt = self._legacy_p_up(test_preds, heads_t, ens_pred_test, close_t)
        # BSS referansi: meta bolumunun (testten ONCE) taban orani, arindirilan son h satir haric
        base_ref = float(y_m_all[:max(len(y_m_all) - h, 1)].mean())
        out: Dict[str, Any] = {
            'n_bins': cal.N_BINS, 'horizon_purge': h, 'base_rate_ref': base_ref,
            'legacy': {**cal.summary(p_leg, y_t_all[-Lt:], base_rate=base_ref),
                       'head_vs_price_conflict_rate': float(np.mean(conflict_leg)),
                       'has_direction_heads': bool(heads_t)},
        }

        keys, Xm, Lm = self._direction_features(meta_preds, heads_m, close_m)
        ym = y_m_all[-Lm:]
        half = Lm // 2
        fit_idx = np.arange(0, max(half - h, 0))
        cal_idx = np.arange(half, max(Lm - h, half))
        out.update({'features': keys, 'n_fit': int(len(fit_idx)), 'n_calib': int(len(cal_idx))})
        if (len(fit_idx) < MIN_CALIB_ROWS or len(cal_idx) < MIN_CALIB_ROWS
                or len(np.unique(ym[fit_idx])) < 2 or len(np.unique(ym[cal_idx])) < 2):
            out['calibrated'] = None
            out['skipped'] = f'yetersiz satir/sinif (fit {len(fit_idx)}, calib {len(cal_idx)})'
            return out
        mu = Xm[fit_idx].mean(axis=0)
        sd = Xm[fit_idx].std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit((Xm[fit_idx] - mu) / sd, ym[fit_idx])
        z_cal = clf.decision_function((Xm[cal_idx] - mu) / sd)
        T = cal.fit_temperature(z_cal, ym[cal_idx])
        iso = cal.fit_isotonic(cal.sigmoid(z_cal), ym[cal_idx])

        tkeys, Xt, Lt2 = self._direction_features(test_preds, heads_t, close_t)
        if tkeys != keys:
            raise ValueError(f"test ozellikleri {tkeys} != uyum ozellikleri {keys}")
        yt = y_t_all[-Lt2:]
        z_t = clf.decision_function((Xt - mu) / sd)
        p_raw = cal.sigmoid(z_t)
        p_temp = cal.apply_temperature(z_t, T)
        p_iso = iso.predict(p_raw)
        up_price_t = self._ret_of(np.asarray(ens_pred_test)[-Lt2:], close_t[-Lt2:]) > 0
        at_bound = bool(T >= 0.99 * 20.0 or T <= 0.05 / 0.99)
        s_temp = cal.summary(p_temp, yt, base_rate=base_ref)
        info = cal.informative(s_temp, at_bound)
        out['calibrated'] = {
            'temperature': T,
            # T arama sinirinda: en iyi NLL p -> 0,5 (bilgisiz) yonunde; "kalibre" guven sabite yakin
            'temperature_at_bound': at_bound,
            # hakem-finans I2 #1: 'kalibre' sifati yalniz bu kapi gecilirse (T sinirda degil,
            # BSS > 0, AUC GA'si 0,5'i dislar); aksi halde "sicaklik olcekli, ayirt edici degil"
            'informative': info,
            'label': 'kalibre (ayirt edici)' if info else 'sicaklik olcekli, ayirt edici degil',
            'raw': cal.summary(p_raw, yt, base_rate=base_ref), 'temperature_scaled': s_temp,
            'isotonic_compare': cal.summary(p_iso, yt, with_bins=False, base_rate=base_ref,
                                            ci=False),
            # F1: fiyat tahmininin yonu ile p'nin yonu catisiyor mu (ayri bayrak)
            'price_vs_p_conflict_rate': float(np.mean((p_temp >= 0.5) != up_price_t)),
        }
        if self.confidence_mode == 'calibrated':
            self.direction_model = {
                'features': keys, 'mean': mu.tolist(), 'scale': sd.tolist(),
                'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]),
                'temperature': T, 'informative': info,
                'fit_on': 'OOF meta bolumu ilk yari (%60 modelleri)',
                'calibrated_on': 'OOF meta bolumu ikinci yari',
            }
        logger.info(f"  [KALIBRASYON] T={T:.3f} test ECE ham {out['calibrated']['raw']['ece']:.4f} -> "
                    f"sicaklik {out['calibrated']['temperature_scaled']['ece']:.4f} "
                    f"(eski yol {out['legacy']['ece']:.4f})")
        return out

    @staticmethod
    def _tats_inputs(df: pd.DataFrame, feat_df: pd.DataFrame, base_end: int, meta_end: int,
                     horizon: str):
        """Meta bolumunun (X[base_end:meta_end]) HAM kapanisi ve h sonraki ham kapanis (B10).

        feat_df['close'] ICEEMDAN acikken filtrelenmis olabilir; burada kullanilmaz.
        """
        h = 1 if horizon == 'daily' else 5
        raw = df['close'].sort_index()
        idx = feat_df.index[base_end:meta_end]
        close = raw.reindex(idx).to_numpy(dtype=np.float64)
        future = raw.shift(-h).reindex(idx).to_numpy(dtype=np.float64)
        if not (np.isfinite(close).all() and np.isfinite(future).all()):
            # Sessiz NaN etiket yerine acik hata: feat_df indeksi ham df ile eslesmiyor
            raise ValueError("TATS: meta satirlarinin ham kapanisi bulunamadi (indeks uyusmazligi)")
        return close, future

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

        # Y28: son satir = df'in son tarihi (hedefi NaN olan son h satir tutulur); current_close
        # ile ayni gun. build_features bu satirlari attigi icin tahmin bir ufuk bayattı.
        feat_df = self.feature_engineer.build_inference_features(
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
        # Y28 degismezi (hakem-2 Y28 #1): son ozellik satiri df'in son tarihi olmali. Son gunde bir
        # ozellik NaN ise _build o satiri dusurur ve bayatlik sessizce geri gelir -> gorunur yap.
        feature_date = pd.Timestamp(aligned.index[-1])
        df_last = pd.Timestamp(pd.to_datetime(df.index).max())
        stale_features = bool(feature_date != df_last)
        if stale_features:
            logger.warning(
                f"  [{symbol}] son ozellik satiri {feature_date.date()} != son veri {df_last.date()} "
                f"(son gunde NaN ozellik); tahmin bayat, stale_features=True"
            )

        # log_return hedefi kullanildiysa gercek fiyata cevir
        if self.target_type == 'log_return':
            predicted_close = current_close * np.exp(raw_pred)
        else:
            predicted_close = raw_pred

        # TATS: trend siniflandirici ile tahmin duzelt
        tats_available = bool(self.use_tats and self.tats_corrector and self.tats_corrector.is_fitted())
        tats_meta: Dict[str, Any] = {'applied': False}
        if tats_available:
            meta_input_for_tats = np.array(
                [model_predictions[m] for m in sorted(model_predictions.keys())]
            )
            predicted_close, tats_meta = self.tats_corrector.correct(
                predicted_close, current_close, meta_input_for_tats
            )
        tats_meta['available'] = tats_available  # Y25: "TATS yok" != "uygulanmadi"

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

        # Faz 9 F1/F4: kalibre yon olasiligi ('calibrated' mod). Fiyat tahmini degismez.
        calib_out: Dict[str, Any] = {'confidence_mode': self.confidence_mode}
        if self.confidence_mode == 'calibrated':
            dm = self.direction_model
            feats = {f'ret_{k}': v for k, v in model_predictions.items()}
            feats.update({f'dir_{k}': v for k, v in direction_probs.items()})
            if dm is None or sorted(feats) != list(dm['features']):
                calib_out['calibration_unavailable'] = (
                    'yon modeli yok' if dm is None else
                    f"ozellikler {sorted(feats)} != {dm['features']}")
                logger.warning(f"  [{symbol}] kalibre yon kullanilamadi: "
                               f"{calib_out['calibration_unavailable']}")
            else:
                x = np.array([
                    float(self._ret_of(np.array([feats[k]]), np.array([current_close]))[0])
                    if k.startswith('ret_') else feats[k] - 0.5
                    for k in dm['features']])
                z = float(np.dot((x - np.array(dm['mean'])) / np.array(dm['scale']), dm['coef'])
                          + dm['intercept'])
                p_raw = float(1 / (1 + np.exp(-z)))
                p_cal = float(1 / (1 + np.exp(-z / dm['temperature'])))
                direction_price = direction
                direction_p = 'UP' if p_cal >= 0.5 else 'DOWN'
                info = bool(dm.get('informative', False))
                # hakem-finans I2 #4: bilgisiz modelde (kapi gecilmedi) p ~ 0,5 ve yonu gurultuyle
                # cevirirdi -> yon fiyat isaretinde kalir; guven p'den ama 'bilgisiz' isaretli
                if info:
                    direction = direction_p
                confidence = round(max(p_cal, 1 - p_cal), 4)
                calib_out.update({'p_up_raw': round(p_raw, 6), 'p_up_cal': round(p_cal, 6),
                                  'direction_price': direction_price,
                                  'direction_from': 'p_cal' if info else 'price (bilgisiz yon modeli)',
                                  'confidence_informative': info,
                                  'direction_conflict': direction_price != direction_p,
                                  'temperature': dm['temperature']})

        # Y29: hedef tarih verinin son seansindan (feature_date) h is gunu sonra; duvar saatinden
        # DEGIL. Resmi BIST tatilleri bilinmiyor -> tahmini tarih; tracker gercek h. seansi kullanir.
        pred_date = (feature_date + pd.offsets.BDay(horizon_sessions(self.horizon))).strftime('%Y-%m-%d')

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
            'calibration': calib_out,
            # Y28 kokeni: kullanilan son ozellik satirinin tarihi (= df'in son tarihi).
            # Bu alani tasimayan tracker kayitlari Y28 oncesidir (bir ufuk bayat), gecersiz.
            'feature_date': str(feature_date)[:10],
            'stale_features': stale_features,
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

    def _tats_path(self, symbol: str) -> str:
        return os.path.join(models_dir(), f'{self._model_id(symbol)}_{self.horizon}_tats.json')

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

        # Faz 9 F1-F4: yon modeli JSON'da (lojistik katsayilar + sicaklik)
        meta['confidence_mode'] = self.confidence_mode
        if self.direction_model is not None:
            meta['direction_model'] = self.direction_model

        # K21: TATS diske yazilir (eskiden yuklenen modelde TATS hic calismiyordu)
        meta['use_tats'] = bool(self.use_tats)
        tc = self.tats_corrector
        if self.use_tats and tc is not None and tc.is_fitted():
            tc.classifier.save_model(self._tats_path(symbol))
            meta['tats_saved'] = True
            meta['tats_params'] = {
                'correction_factor': float(tc.correction_factor),
                'flat_threshold': float(tc.flat_threshold),
                'semantic_classes': [int(c) for c in tc._semantic_classes],
            }

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

        # K21: TATS geri yuklenir. Dosyasi olmayan eski modelde TATS yok (onceki davranis).
        # Y25: kurucudaki use_tats=False (ablasyon kolu) meta ile EZILMEZ; ikisi de True olmali
        self.use_tats = bool(self.use_tats) and bool(meta.get('use_tats', True))
        self.tats_corrector = None
        tats_path = os.path.join(
            os.path.dirname(meta_path), f'{self._model_id(symbol)}_{self.horizon}_tats.json')
        if meta.get('tats_saved') and _XGBOOST_AVAILABLE and os.path.exists(tats_path):
            from xgboost import XGBClassifier
            tp = meta.get('tats_params', {})
            tc = TATSCorrector(correction_factor=tp.get('correction_factor', 0.5),
                               flat_threshold=tp.get('flat_threshold', 0.005))
            tc.classifier = XGBClassifier()
            tc.classifier.load_model(tats_path)
            tc._semantic_classes = np.array(tp.get('semantic_classes', [0, 1, 2]))
            tc._is_fitted = True
            self.tats_corrector = tc

        # Faz 9 F1-F4: mod modelin kendisinden (kalibre egitilmis model kalibre tahmin uretir)
        self.confidence_mode = meta.get('confidence_mode', 'legacy')
        self.direction_model = meta.get('direction_model')

        self._is_trained = True
        logger.info(f"  Ensemble yuklendi: {len(self.base_models)} model"
                    f"{' + TATS' if self.tats_corrector else ''}")

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
