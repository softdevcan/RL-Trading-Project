# Faz 3: Gelişmiş Tahmin Kalitesi ve Risk Yönetimi — Plan ve Uygulama

## Bağlam

Faz 2'de tamamlanan ensemble tahmin sistemi (XGBoost + LightGBM + CatBoost + BiLSTM + TFT + Ridge meta-learner), araştırma dokümanındaki gap analizi sonucunda hem mevcut hataların düzeltilmesi hem de 2025-2026 araştırma bulgularına dayanan iyileştirmelerle güçlendirildi. Plan 4 alt fazdan oluşur; her biri bağımsız olarak test edilebilir.

**Durum:** ✅ Tamamlandı (2026-03-27)

---

## Faz 3.1: Temel Düzeltmeler (Foundation) ✅

> Sonraki tüm fazların üzerine inşa edileceği temel. Tüm maddeler birbirinden bağımsız, paralel uygulanabilir.

### 3.1.1 — PSR Trade Frequency Bug ✅
- **Dosya:** `env/reward_functions.py:119-121`
- **Sorun:** `total_trades=sum(1 for _ in range(len(portfolio_values)))` → her zaman `len(portfolio_values)` döner, gerçek işlem sayısı değil
- **Çözüm:** `total_trades=trades_executed` olarak düzeltildi (parametre zaten geçiliyordu, kullanılmıyordu)
- **Sonuç:** 10 step, 3 trade senaryosunda `trades_per_100 = 30` doğru hesaplanıyor

### 3.1.2 — Meta-Learner Data Leakage ✅
- **Dosya:** `prediction/models/ensemble.py`
- **Sorun:** Base modeller `X_test` üzerinde tahmin üretip, aynı `y_test` ile Ridge eğitiliyordu. Meta-learner için train=test.
- **Çözüm:** 3'lü kronolojik split uygulandı:
  - %60 base model eğitimi (`X_tr` / `y_tr`)
  - %20 meta-learner eğitimi (`X_meta` / `y_meta`) — base modellerin OOF tahminleri
  - %20 nihai test (`X_test` / `y_test`) — meta-learner hiç görmedi
- **Ek:** `TimeSeriesSplit` kullanımı kaldırıldı (artık gerekmiyor)

### 3.1.3 — Embargo Uygulaması ✅
- **Dosya:** `prediction/trainer.py`
- **Sorun:** `embargo_days=3` parametre olarak vardı ama split mantığında hiç uygulanmıyordu
- **Çözüm:** `prev_test_end` değişkeni ile her fold'un train seti önceki fold'un `test_end + embargo_days` sonrasını içermeyecek şekilde güncellendi. Log mesajları train/purge/test/embargo aralıklarını gösteriyor.

### 3.1.4 — TFT VSN 50-Değişken Sınırı ✅
- **Dosya:** `prediction/models/tft_model.py`
- **Sorun:** `n_vars=min(input_size, 50)` — 80 özellikten 30'u sessizce yok sayılıyordu
- **Çözüm:** `n_vars=input_size` olarak değiştirildi. Feature selector zaten 80'e indiriyor, ikinci sınıra gerek yok.

### 3.1.5 — Direction Head Kullanımı ✅
- **Dosyalar:** `prediction/models/base.py`, `prediction/models/lstm_model.py`, `prediction/models/tft_model.py`, `prediction/models/ensemble.py`
- **Sorun:** BiLSTM ve TFT direction head eğitiyordu ama inference'da `_` ile atılıyordu
- **Çözüm:**
  - `BasePredictionModel`'e `_predict_direction_raw(X) -> Optional[np.ndarray]` metodu eklendi (default `None`)
  - BiLSTM/TFT'de `_predict_direction_raw()` override edildi, sigmoid direction olasılıkları döndürülüyor
  - `ensemble.predict_next`'te `direction_probs` dict toplanıyor, confidence hesabında kullanılıyor

### 3.1.6 — Permutation Importance Entegrasyonu ✅
- **Dosya:** `prediction/feature_selector.py`
- **Sorun:** `compute_permutation_importance` standalone metoddu, `fit_select` pipeline'ına entegre değildi
- **Çözüm:** `fit_select`'e `use_permutation_importance=False` parametresi ile opsiyonel 3. aşama eklendi. Ridge surrogate model üzerinde PI hesaplanıyor, negatif skora sahip özellikler eleniyor. Return dict'e `pi_scores` ve `dropped_pi` eklendi.

---

## Faz 3.2: Tahmin Kalitesi İyileştirmeleri ✅

### 3.2.1 — Meta-Learner Yükseltmesi: Ridge → XGBoost ✅
- **Dosya:** `prediction/models/ensemble.py`
- **Çözüm:**
  - `meta_learner_type: str = 'xgboost'` parametresi eklendi (geriye uyumluluk için `'ridge'` da destekleniyor)
  - `XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8)` — sığ ve regularized
  - `_build_meta_learner()` factory metodu
  - XGBoost native `save_model`/`load_model`, `_meta_learner_path()` yardımcı metodu
  - Ridge fallback backward compat için korunuyor

### 3.2.2 — TATS (Trend-Adjusted Time Series) Sistemi ✅
- **Yeni dosya:** `prediction/tats.py`
- **Değişen:** `prediction/models/ensemble.py`
- **Mimari:**
  1. XGBoost UP/DOWN/FLAT trend sınıflandırıcı (eşik: `|return| > 0.005`)
  2. `fit(X, y_prices, current_prices)`: Classifier eğitir, metrik döner
  3. `correct(predicted_price, current_price, X_last)`: Tahmin ile trend çelişirse `correction_strength = trend_confidence * correction_factor` ile düzeltir
  4. Güvenli fallback: XGBoost yoksa veya classifier eğitilmemişse düzeltme yapmaz
- **Entegrasyon:** `ensemble.train()` içinde `use_tats=True` ile TATS fit edilir; `predict_next()` çıktısında `tats` alanı eklendi

### 3.2.3 — ICEEMDAN Gürültü Ön-İşleme ✅
- **Yeni dosya:** `prediction/iceemdan_processor.py`
- **Değişen:** `prediction/feature_engineer.py`
- **Yeni bağımlılık:** `EMD-signal==1.9.0`
- **Çözüm:**
  - `ICEEMDANProcessor` sınıfı: `decompose()`, `filter_noise(noise_imfs=2)` (r>0.95 korelasyon güvenlik kontrolü), `extract_imf_features()` → `imf_trend_component`, `imf_energy_ratio`
  - `PyEMD.CEEMDAN` kullanıyor, paket yoksa graceful fallback
  - `feature_engineer.py`'de `use_iceemdan=False` parametresi ile opsiyonel; IMF kolonları DataFrame'e ekleniyor

### 3.2.4 — Global Makro Göstergeler: VIX, US10Y, DXY ✅
- **Dosyalar:** `data/macro_fetcher.py`, `prediction/feature_engineer.py`
- **Eklenenler:** `vix` (`^VIX`), `us10y` (`^TNX`), `dxy` (`DX-Y.NYB`) — yfinance üzerinden
- **Not:** Bu göstergeler yalnızca prediction pipeline'a gider. RL state space'e eklenmedi (eğitilmiş model uyumluluğu korunuyor).

---

## Faz 3.3: Risk Yönetimi ve Pozisyon Boyutlandırma ✅

### 3.3.1 — ATR-Tabanlı Dinamik Pozisyon Boyutlandırma ✅
- **Dosya:** `env/trading_env.py`
- **Sorun:** `action * max_shares_per_trade(100)` — 5 TL ve 500 TL hisse aynı lot
- **Çözüm:**
  - Yeni parametreler: `use_atr_sizing=False`, `risk_per_trade=0.02`, `atr_multiplier=2.0`, `max_position_pct=0.20`, `atr_period=14`
  - `_get_atr(symbol)`: env'in df'inden ATR_14 hesaplar (high/low/close)
  - `_atr_position_size(symbol, action_signal)`: `position_value = portfolio * risk_per_trade * |action|`, `shares = position_value / (price * atr_stop / price)`, `max_position_pct` ile sınırlanır
- **Varsayılan:** `use_atr_sizing=False` — mevcut eğitilmiş modeller bozulmaz

### 3.3.2 — Kelly Fraksiyonel Boyutlandırma ✅
- **Dosya:** `env/trading_env.py`
- **Çözüm:**
  - Yeni parametreler: `use_kelly=False`, `kelly_fraction=0.25`
  - `_kelly_position_size(symbol, action_signal, win_prob)`: `f* = (p*b - q*a)/(a*b)`, quarter-Kelly, `max_position_pct` ile sınırlanır
  - `step()` öncelik sırası: `use_kelly > use_atr_sizing > sabit lot`
  - Kelly `prediction_features['confidence']` kullanır; yoksa `0.5 + |action| * 0.3` fallback
- **Varsayılan:** `use_kelly=False`

---

## Faz 3.4: Açıklanabilirlik ve İzleme ✅

### 3.4.1 — SHAP Entegrasyonu ✅
- **Yeni dosya:** `prediction/explainability.py`
- **Değişen:** `app/api/routes/prediction.py`
- **Yeni bağımlılık:** `shap==0.51.0`
- **Çözüm:**
  - `ModelExplainer(n_background=100)` sınıfı
  - `_compute_shap()`: TreeExplainer (XGB/LGBM/CatBoost), LinearExplainer (Ridge), KernelExplainer (fallback)
  - `explain_prediction(model, X, feature_cols, background)`: Tek tahmin SHAP değerleri + top_positive/top_negative listesi
  - `explain_global(model, X, feature_cols)`: Mean absolute SHAP (max 500 örnek)
  - `shap` paketi yoksa graceful `None` döner
- **API:** `GET /prediction/explain/{symbol}?horizon=daily&model_type=xgboost&n_background=100`
  - `shap_available: false` ile 200 yanıtı döner (paket yoksa)

### 3.4.2 — Gelişmiş Değerlendirme Metrikleri ✅
- **Dosya:** `prediction/evaluator.py`
- **Eklenenler:**
  - `compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)`
  - `compute_calmar_ratio(equity_curve, periods_per_year=252)`
  - `compute_deflated_sharpe_ratio(returns, n_trials=1, periods_per_year=252)` — Bailey & Lopez de Prado (2014), `{sr, sr_benchmark, psr, dsr}` döner
  - `compute_turnover(y_pred, current_prices, threshold=0.0)` — `{turnover_rate, n_trades, avg_trade_size}`
- **Entegrasyon:** `compute_comprehensive_metrics()` tüm 4 metriği çağırıyor, output dict'e ekliyor

---

## Bağımlılık Haritası

```
3.1.1 (PSR bug)           ─┐
3.1.2 (Meta-learner leak)  ─┤
3.1.3 (Embargo)            ─┤── Paralel (birbirinden bağımsız)
3.1.4 (TFT VSN cap)        ─┤
3.1.5 (Direction head)     ─┤
3.1.6 (Permutation imp.)   ─┘
         │
         ▼
3.2.1 (XGBoost meta)       ← 3.1.2'ye bağımlı
3.2.2 (TATS)               ← 3.1.5'e bağımlı
3.2.3 (ICEEMDAN)           ← Bağımsız
3.2.4 (VIX/Global)         ← Bağımsız
         │
         ▼
3.3.1 (ATR pozisyon)       ← 3.1.1'e bağımlı
3.3.2 (Kelly)              ← 3.3.1'e bağımlı
         │
         ▼
3.4.1 (SHAP)               ← 3.2.1'e bağımlı (final modeller)
3.4.2 (Gelişmiş metrikler) ← Bağımsız
```

---

## Yeni Bağımlılıklar

| Paket | Versiyon | Faz | Amaç |
|-------|----------|-----|-------|
| `shap` | `==0.51.0` | 3.4.1 | SHAP explainability |
| `EMD-signal` | `==1.9.0` | 3.2.3 | ICEEMDAN gürültü filtreleme |

Kaldırılan / güncellenen bağımlılıklar:

| Değişiklik | Açıklama |
|-----------|----------|
| `torch-tb-profiler` kaldırıldı | 2023'ten beri terk edilmiş; `tensorboard>=2.17.0` ile değiştirildi |
| `nvidia-ml-py3==7.352.0` → `nvidia-ml-py>=12.0.0` | Deprecated paket; aktif fork |
| `evds>=0.4.0` → `evds==0.4.0` | Sabit versiyon pinleme |
| `borsapy>=0.8.3` → `borsapy==0.8.3` | Sabit versiyon pinleme |

---

## Etkilenen Dosyalar

| Dosya | Değişiklik Türü | Faz |
|-------|----------------|-----|
| `env/reward_functions.py` | Bug fix | 3.1.1 |
| `env/trading_env.py` | Yeni özellik (ATR + Kelly) | 3.3 |
| `prediction/models/base.py` | Yeni metod | 3.1.5 |
| `prediction/models/lstm_model.py` | Direction head | 3.1.5 |
| `prediction/models/tft_model.py` | VSN fix + direction head | 3.1.4, 3.1.5 |
| `prediction/models/ensemble.py` | OOF split + XGB meta + TATS | 3.1.2, 3.2.1, 3.2.2 |
| `prediction/trainer.py` | Embargo fix | 3.1.3 |
| `prediction/feature_selector.py` | Permutation importance | 3.1.6 |
| `prediction/feature_engineer.py` | ICEEMDAN + VIX/US10Y/DXY | 3.2.3, 3.2.4 |
| `prediction/evaluator.py` | Sortino/Calmar/DSR/Turnover | 3.4.2 |
| `prediction/iceemdan_processor.py` | **Yeni dosya** | 3.2.3 |
| `prediction/tats.py` | **Yeni dosya** | 3.2.2 |
| `prediction/explainability.py` | **Yeni dosya** | 3.4.1 |
| `data/macro_fetcher.py` | VIX/US10Y/DXY eklendi | 3.2.4 |
| `app/api/routes/prediction.py` | /explain endpoint | 3.4.1 |
| `requirements.txt` | Yeni paketler + temizlik | — |
