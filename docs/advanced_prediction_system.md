# Gelişmiş Tahmin Sistemi — Mimari Referans

**Durum:** Tamamlandı (Faz 2)
**Donanım:** RTX 4060 8GB + Ryzen 7 7800X3D

---

## Mimari Özet

```
Veri Katmanı → Feature Engineering v2 → Multi-Model (5 base) → Stacking Ensemble
    → Fiyat + Yön + Güven → RL Observation Space
```

---

## Veri Katmanı

| Modül | Kaynak | Çıktı |
|---|---|---|
| `data/data_fetcher.py` | yfinance | OHLCV (multi-index: symbol × date), retry + incremental + cache |
| `data/macro_fetcher.py` | TCMB EVDS + yfinance | policy_rate, cpi_inflation, ppi_inflation, usd_try, eur_try, bist100_index |
| `data/fundamental_fetcher.py` | yfinance | ROE, ROA, debt_to_equity, current_ratio, P/E, P/B, profit_margin |
| `data/gold_fetcher.py` | borsapy / yfinance | gram_altin_try, ons_altin_usd, usd_try, eur_try |

### data_fetcher.py — Önemli Özellikler
- `fetch_incremental()`: Son tarihi okur, sadece delta'yı çeker, mevcut CSV'ye append eder
- `get_source_status()`: Eksik gün sayısı ve son tarih raporu
- `_fetch_with_retry()`: Exponential backoff (1s → 2s → 4s), max 3 deneme
- Coverage check: Beklenen trading günlerinin %80'inden azsa uyarı verir
- Class-level cache: Aynı sembol/tarih için tekrar çekim yapmaz

---

## Feature Engineering (`prediction/feature_engineer.py`)

Tüm özellikler en az 1 gün gecikmeli (shift) — data leakage önlenir.

| Grup | Özellikler |
|---|---|
| Getiri | log return, basit return, çoklu gecikme (1/5/10/20 gün) |
| Volatilite | Parkinson, Garman-Klass, realized vol (5/10/20 gün) |
| Momentum | RSI türevleri, MACD histogram gradyanı |
| Hacim | Hacim oranları, hacim-fiyat korelasyonu |
| Takvim | Gün, ay, çeyrek (cyclical encoding) |
| Teknik | Gecikmeli indikatörler + türevler |
| Cross-asset | BIST-100 ve USD/TRY korelasyonu |
| Makro | Faiz değişimi, enflasyon trendi |
| Fundamental | P/E, P/B değişim |
| Rejim | Volatilite rejimi, trend rejimi |

---

## Feature Selector (`prediction/feature_selector.py`)

- Mutual information ile ilk eleme
- Permutation importance ile final seçim
- TimeSeriesSplit ile overfitting önleme

---

## Multi-Model Mimarisi (`prediction/models/`)

| Model | Dosya | Donanım | Notlar |
|---|---|---|---|
| XGBoost | `xgboost_model.py` | CPU | Geliştirilmiş, Optuna HPO |
| LightGBM | `lightgbm_model.py` | CPU | lightgbm>=4.0.0 |
| CatBoost | `catboost_model.py` | CPU | catboost>=1.2.0 |
| BiLSTM | `lstm_model.py` | GPU ~200MB | PyTorch, CUDA otomatik algılar |
| TFT | `tft_model.py` | GPU ~500MB–1GB | Temporal Fusion Transformer |
| Stacking | `ensemble.py` | CPU | Ridge + XGBoost meta-learner |

Her model çıktısı: `(predicted_price, predicted_direction, confidence)`

`prediction/legacy_models.py` — Eski tek-XGBoost implementasyonu (referans/arşiv).

---

## Hiperparametre Optimizasyonu (`prediction/hyperopt.py`)

- Optuna TPESampler + MedianPruner
- TimeSeriesSplit ile zaman serisi güvenli CV
- Model bazlı arama uzayları (XGB / LGBM / CatBoost / BiLSTM / TFT)
- Optuna Dashboard entegrasyonu

---

## Eğitim Pipeline (`prediction/trainer.py`)

- **Expanding window walk-forward**: Her fold önceki tüm veriden öğrenir
- **Purge gap**: 5 gün (train-val arası sızıntı önleme)
- **Embargo**: 3 gün (validation sonrası bekleme)
- Multi-model paralel eğitim
- GPU otomatik algılama (CUDA)
- Experiment tracking: JSON log (`prediction/tracker.py`)

---

## Değerlendirme (`prediction/evaluator.py`)

| Metrik | Açıklama |
|---|---|
| Direction Accuracy | Yön doğruluğu (%) |
| Profit Factor | Kazanç/kayıp oranı |
| Information Coefficient (IC) | Tahmin-gerçek korelasyonu |
| Calibration | Güven skorunun güvenilirliği |
| Regime-based | Boğa/ayı/yatay piyasa ayrımı |
| Diebold-Mariano | Modeller arası istatistiksel fark testi |
| Backtest simülasyonu | Gerçekçi komisyon + slippage ile |

---

## RL Entegrasyonu (`env/trading_env.py`)

Observation space'e eklenen özellikler (N sembol başına):

| Özellik | Açıklama |
|---|---|
| `predicted_return` | Tahmin edilen günlük getiri |
| `predicted_direction` | 1=yukarı, -1=aşağı, 0=yatay |
| `prediction_confidence` | Ensemble güven skoru [0,1] |
| `ensemble_agreement` | 5 modelin uzlaşı oranı [0,1] |

**Toplam state space (Faz 2):** 56 (Faz 1) + 4×N (tahmin) features

---

## Bağımlılıklar

```
lightgbm>=4.0.0
catboost>=1.2.0
optuna
evds          # TCMB EVDS API (pip install evds)
borsapy       # Borsa verileri (opsiyonel, gold_fetcher için)
torch         # PyTorch (CUDA önerilir)
```

---

## İlgili Dokümanlar

- [docs/PHASE2_HYPEROPT_UPDATE.md](PHASE2_HYPEROPT_UPDATE.md) — HPO detayları
- [docs/SPRINT2_COMPLETE.md](SPRINT2_COMPLETE.md) — Sprint tamamlanma notu
