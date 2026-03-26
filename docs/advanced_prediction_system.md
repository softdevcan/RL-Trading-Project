---
name: Advanced Prediction System
overview: Mevcut XGBoost-only tahmin sistemini, multi-model ensemble mimarisine (XGBoost + LightGBM + CatBoost + BiLSTM + TFT), gelismis feature engineering, Optuna HPO ve RL entegrasyonu ile yeniden yapilandirma.
todos:
  - id: phase1-feature-engineering
    content: "Phase 1: Gelismis Feature Engineering - feature_engineer.py yeniden yazimi + feature_selector.py"
    status: completed
  - id: phase2-multi-model
    content: "Phase 2: Multi-Model Mimarisi - prediction/models/ dizini (base, xgboost, lightgbm, catboost, lstm, tft, ensemble)"
    status: completed
  - id: phase3-hyperopt
    content: "Phase 3: Hiperparametre Optimizasyonu - prediction/hyperopt.py + Optuna entegrasyonu"
    status: completed
  - id: phase4-training-pipeline
    content: "Phase 4: Egitim Pipeline - trainer.py yeniden yazimi (walk-forward, purge gap, multi-model)"
    status: completed
  - id: phase5-evaluation
    content: "Phase 5: Degerlendirme Cercevesi - evaluator.py genisletme (direction acc, profit factor, IC, backtest)"
    status: completed
  - id: phase6-rl-integration
    content: "Phase 6: RL Entegrasyonu - trading_env.py + daily_trading.py observation space genisletme"
    status: completed
  - id: phase7-api-dashboard
    content: "Phase 7: API ve Dashboard - yeni endpoint'ler + dashboard guncelleme"
    status: completed
isProject: false
---

# Gelismis Tahmin Sistemi - Uygulama Plani

## Mevcut Durum

- Tek XGBoost modeli, sabit hiperparametrelerle (`prediction/models.py`)
- Basit feature engineering: lagged close, rolling stats, volume, calendar (`prediction/feature_engineer.py`)
- Walk-forward CV mevcut ama basic (`prediction/trainer.py`)
- RL ile entegrasyon yok
- Makro/fundamental veriler tahmin pipeline'inda kullanilmiyor

## Hedef Mimari

**Data Layer** -> **Feature Engineering v2** -> **Multi-Model (5 base model)** -> **Stacking Ensemble** -> **Fiyat + Yon + Guven** -> **RL Observation Space**

## Phase 1: Gelismis Feature Engineering

- **Dosya:** `prediction/feature_engineer.py` (yeniden yazim)
- **Yeni dosya:** `prediction/feature_selector.py`
- Log returns, Parkinson/Garman-Klass volatilite, momentum divergence
- Cross-asset features (BIST-100, USD/TRY korelasyonu)
- Makro entegrasyon (EVDS: faiz, enflasyon, doviz)
- Fundamental features (P/E, P/B degisim)
- Market regime detection (volatilite + trend rejimi)
- Mutual information + permutation importance ile otomatik feature selection

## Phase 2: Multi-Model Mimarisi

- **Yeni dizin:** `prediction/models/` (mevcut `models.py` refactor)
- `base.py` - BasePredictionModel ABC
- `xgboost_model.py` - Gelistirilmis XGBoost
- `lightgbm_model.py` - LightGBM
- `catboost_model.py` - CatBoost
- `lstm_model.py` - BiLSTM (PyTorch, CUDA)
- `tft_model.py` - Temporal Fusion Transformer (PyTorch, CUDA)
- `ensemble.py` - Stacking meta-learner (Ridge + XGBoost)
- Her model: train, predict -> (price, direction, confidence), HPO, save/load

## Phase 3: Hiperparametre Optimizasyonu

- **Yeni dosya:** `prediction/hyperopt.py`
- Optuna + TimeSeriesSplit + TPESampler + MedianPruner
- Model-bazli arama uzaylari (XGB, LGBM, CatBoost, BiLSTM, TFT)
- Optuna Dashboard entegrasyonu (mevcut)

## Phase 4: Egitim Pipeline

- **Dosya:** `prediction/trainer.py` (yeniden yazim)
- Expanding window walk-forward + purge gap (5 gun) + embargo (3 gun)
- Multi-model paralel egitim
- GPU otomatik algilama (CUDA)
- Experiment tracking (JSON log)

## Phase 5: Degerlendirme Cercevesi

- **Dosya:** `prediction/evaluator.py` (genisletme)
- Direction accuracy, Profit Factor, Information Coefficient
- Calibration metrikleri, regime-based evaluation
- Diebold-Mariano testi, backtest simulasyonu

## Phase 6: RL Entegrasyonu

- **Dosyalar:** `env/trading_env.py`, `app/services/daily_trading.py`
- Observation space'e: predicted_return, predicted_direction, prediction_confidence, ensemble_agreement (4 x N sembol)

## Phase 7: API ve Dashboard

- **Dosyalar:** `app/api/routes/prediction.py`, `app/services/prediction_service.py`, `dashboard/pages/prediction.py`
- Yeni endpoint'ler: train-all, optimize, ensemble-predict, model karsilastirma
- Dashboard: performans karsilastirma grafikleri, ensemble weights, egitim ilerleme

## Yeni Bagimliliklar

- `lightgbm>=4.0.0`
- `catboost>=1.2.0`
- PyTorch ve Optuna zaten mevcut

## Donanim (RTX 4060 8GB + Ryzen 7 7800X3D)

- Gradient Boosting: CPU, cok hizli
- BiLSTM: ~200MB VRAM
- TFT: ~500MB-1GB VRAM
- Toplam egitim: ~2-4 saat (HPO dahil)

