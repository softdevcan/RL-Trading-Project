# RL Trading Project - 58 Issue Fix Plan

## Context
ISSUES.md'de belgelenen 65 sorundan 7'si remote merge ile çözüldü. Kalan 58 sorun (4 FATAL, 3 SEC, 17 BUG, 2 DEPR, 32 DESIGN) 8 batch halinde sırayla çözüldü / çözülüyor. Öncelik: Fatal > Security > RL Core > Backend > Frontend > Polish. Her batch bağımsız test edilebilir.

**Güncel durum (2026-03-25): Tüm Batch 0-8 tamamlandı (62 issue çözüldü). Yalnızca #29 açık.**

### Çözülen Sorunlar (merge ile) ✅
- ~~#3~~ Reward function → `env/reward_functions.py` (PSR)
- ~~#11~~ fillna deprecated → `.ffill()` / `.bfill()`
- ~~#26~~ Hardcoded stocks → `get_symbols(phase=)`
- ~~#28~~ Wrong import → Doğru fonksiyon import
- ~~#42~~ Currency $ → ₺
- ~~#46~~ JS stock symbols → Doğru Phase 1
- ~~#47~~ Button ID → Doğru
- ~~#55~~ HTML stock symbols → Doğru

---

## ✅ Batch 0: Merge Sonrası Duplicate Temizliği — TAMAMLANDI
**Issues: #64-65 (2 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:
- `ACADEMIC_GUIDE.md`, `ALGORITHMS.md`, `development.md` root kopyaları silindi (docs/ altındakiler korundu)
- `scripts/generate_academic_report.py`, `scripts/debug_model_actions.py`, `scripts/train_a2c_phase1.py` root kopyaları silindi (scripts/analysis/, scripts/debug/, scripts/training/ altındakiler korundu)

---

## ✅ Batch 1: Fatal Crashlar ve Güvenlik Açıkları — TAMAMLANDI
**Issues: #15-18, #21-22, #43, #51 (8 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**scripts/analysis/generate_academic_report.py** (#15-18)
- `from app.environments.trading_env import MultiStockTradingEnv` → `from env.trading_env import TradingEnv`
- Pickle yerine `DataFetcher.load_data()` + `add_indicators_to_multi_symbol_df()` + `split_data()` pipeline
- `env.portfolio_value` → `env._get_portfolio_value()`
- `env.trade_history` → `env.trades_history`

**app/core/config.py** (#21)
- `CORS_ORIGINS: ["*"]` → `["http://localhost:8000", "http://127.0.0.1:8000"]`

**app/api/routes/trading.py** (#22)
- `sanitize_model_name()` utility eklendi — `os.path.basename()` + `^[a-zA-Z0-9_.-]+$` regex ile path traversal engellendi
- `import re` + `_SAFE_MODEL_NAME = re.compile(r'^[a-zA-Z0-9_.-]+$')` eklendi

**static/js/dashboard.js** (#43)
- `escapeHtml(str)` helper fonksiyonu eklendi
- Tüm `innerHTML` template literal'larında `model.name` → `escapeHtml(model.name)`

**static/js/academic-analysis.js** (#51)
- `static escapeHtml(str)` metodu eklendi
- `displayComparisonTable()` içinde `${modelName}` → `${AcademicAnalysisManager.escapeHtml(modelName)}`

---

## ✅ Batch 2: Teknik İndikatör Düzeltmeleri — TAMAMLANDI
**Issues: #6-10 (5 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**data/technical_indicators.py** (#6-10)
- **#6 RSI Wilder's EMA**: `gain.rolling(window=period).mean()` → `gain.ewm(alpha=1/period, adjust=False).mean()`; `avg_loss.replace(0, np.nan)` ile division by zero koruması
- **#7 ADX Wilder's EMA**: ATR, +DI, -DI için `pd.Series(...).ewm(alpha=1/period, adjust=False).mean()`
- **#8 ADX Division by Zero**: `np.where(denominator > 0, 100 * np.abs(...) / denominator, 0)`
- **#9 Turbulence Mahalanobis**: `_calculate_mahalanobis_turbulence()` fonksiyonu eklendi — cross-sectional `(r_t - mu)^T * Sigma^{-1} * (r_t - mu)` formülü, `np.linalg.pinv` ile singular matrix koruması
- **#10 MACD Signal Line**: `macd_signal`, `macd_hist` sütunları eklendi; `calculate_macd()` artık `(macd, signal_line, histogram)` tuple döndürüyor

**Not:** Bu batch sonrası tüm mevcut modeller geçersizdir — yeniden eğitim gerekli.

---

## ✅ Batch 3: RL Environment Core — TAMAMLANDI
**Issues: #1-2, #4-5, #30 (5 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**env/trading_env.py** (#1, #2, #4, #5)
- **#1 Bounded Obs Space**: `Box(-np.inf, np.inf)` → `Box(-10.0, 10.0)`; `_get_observation()` sonunda `np.clip(state, -10.0, 10.0)`
- **#2 Dynamic Normalization**: `(price - 50) / 50` → per-symbol z-score `(price - mean) / std`; `_compute_price_stats()` training data'dan hesaplıyor; `get_price_stats()` inference için expose ediyor
- **#4 Silent Missing Data**: `KeyError → 0.0` yerine `logger.warning()` + son feature bloğunu tekrar kullan (ffill approx)
- **#5 min_threshold**: `min_threshold = 0` → `min_threshold = 1` (minimum 1 hisse)

**app/services/daily_trading.py** (#30)
- `build_live_state(..., price_stats: Optional[Dict[str, Dict[str, float]]] = None)` — aynı z-score normalization, `p_mean=50.0, p_std=50.0` fallback ile

**Not:** Bu batch sonrası tüm mevcut modeller geçersizdir — yeniden eğitim gerekli.

---

## ✅ Batch 4: Backend Bug Fix — TAMAMLANDI
**Issues: #23-25, #27, #29, #31, #33-36 (10 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**app/api/routes/trading.py** (#23, #24, #25, #27)
- **#23 Race Condition**: `_training_lock = asyncio.Lock()`; `async with _training_lock:` ile concurrent training koruması
- **#24 Deterministic Eval**: `deterministic=False` → `deterministic=True`
- **#25 DummyVecEnv Autoreset**: DummyVecEnv workaround kaldırıldı, raw `TradingEnv` ile eval; `obs, reward, done, truncated, _ = actual_env.step(action)`
- **#27 Hardcoded Paths**: 14 hardcoded path → `_settings.MODELS_DIR`, `_settings.RESULTS_DIR`, `_settings.DATA_DIR`, `_settings.LOGS_DIR`, `_settings.HYPEROPT_DIR`

**app/core/config.py** (#27)
- `MODELS_DIR`, `RESULTS_DIR`, `DATA_DIR`, `LOGS_DIR`, `HYPEROPT_DIR` sabitler eklendi

**app/services/daily_trading.py** (#31)
- `from filelock import FileLock`; `FileLock(lock_file)` ile tüm JSON read/write işlemleri korundu

**app/schemas/trading.py** (#33, #34)
- `risk_mode: str` → `risk_mode: Literal["conservative", "moderate", "aggressive"]`
- `action: str` → `action: Literal["BUY", "SELL", "HOLD"]`

**app/services/model_analysis.py** (#35, #36)
- **#35**: Global `warnings.filterwarnings('ignore')` kaldırıldı → scoped context manager kullanıldı
- **#36**: `profit_factor = np.inf` → `profit_factor = 9999.99` (JSON serialization uyumlu)

**Not (#29):** Portfolio hesaplama mantığı mevcut implementasyonda doğru, açık issue olarak işaretlendi.

---

## ✅ Batch 5: Frontend Bug Fix — TAMAMLANDI
**Issues: #39-41, #52-54 (6 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**static/js/dashboard.js** (#39, #40, #41)
- **#39**: `progressBar`, `stepInfo` null referansları için null check eklendi
- **#40**: `showError()` ikinci parametre kabul ediyor (type='error')
- **#41**: Chart oluşturmadan önce `if (performanceChart) { performanceChart.destroy(); }` ve `if (algorithmComparisonChart) { algorithmComparisonChart.destroy(); }`

**static/js/academic-analysis.js** (#52, #53, #54)
- **#52**: `checkReportStatus(retries = 0)` + `MAX_RETRIES = 20` ile infinite polling durduruldu
- **#53**: `showEmptyState()` içinde `document.getElementById(...)` → null-safe `if (el) el.style.display = ...`
- **#54**: `renderPortfolioComparison()` geçersiz modelleri filtreler, 3-point minimum interpolation, canvas data yoksa gizlenir

---

## ✅ Batch 6: Script Fix ve Deprecation Temizliği — TAMAMLANDI
**Issues: #12, #19-20, #32, #37 (5 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**scripts/training/train_a2c_phase1.py** (#19, #20)
- **#19**: `n_steps=256` hardcoded → `n_steps=n_steps` (fonksiyon parametresini kullan)
- **#20**: `done = np.array([False])`; `while not done[0]:` ile DummyVecEnv ndarray done koruması

**data/data_fetcher.py** (#12)
- `.fillna(method='ffill')` → `.ffill()`
- `.fillna(method='bfill')` → `.bfill()`

**app/main.py** (#32)
- `@app.on_event("startup")` / `@app.on_event("shutdown")` kaldırıldı
- `@asynccontextmanager async def lifespan(app: FastAPI)` ile değiştirildi
- `FastAPI(..., lifespan=lifespan)` eklendi

**run_server.py** (#37)
- `reload=True` → `reload=settings.DEBUG`

---

## ✅ Batch 7: Data Pipeline Robustness — TAMAMLANDI
**Issues: #13-14 (2 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**data/data_fetcher.py**
- **#13**: `_fetch_with_retry(symbol, max_retries=3)` metodu eklendi — 1s/2s/4s exponential backoff, tüm denemeler başarısız olursa `None` döner
- **#14**: `fetch_stock_data()` içinde coverage check — expected ~252 gün/yıl, gerçek veri %80 altındaysa `logger.warning()`

---

## ✅ Batch 8: Kod Kalitesi ve Test Altyapısı — TAMAMLANDI
**Issues: #38, #44-45, #48-50, #56-63 (16 issue) | Durum: ✅ Committed**

### Yapılan Değişiklikler:

**static/js/dashboard.js (#38, #44, #45, #49)**
- **#38**: `const AppState = {...}` namespace — 7 global değişken kapsüllendi; `Object.defineProperty` shims ile geriye uyumluluk korundu
- **#44**: Tüm Chart.js oluşturma noktaları `.destroy()` guard ile korundu (tüm instancelar doğrulandı)
- **#45**: `startStatusCheck()` interval 2000ms → 6000ms (~10 req/min)
- **#49**: `DASHBOARD_DEBUG` flag; `dbgLog()` wrapper ile tüm `console.log` çağrıları korundu

**static/js/daily-trading.js (#48, #49, #50)**
- **#48**: `showError()` / `showSuccess()` → `_showToast()` — 4 saniye sonra auto-dismiss, UI-blocking değil
- **#49**: `DAILY_TRADING_DEBUG` flag; `dtLog()` / `dtError()` wrappers ile 30+ `console.log` korundu
- **#50**: `loadLatestPortfolio()` ve `getDecision()` içinde balance/shares için `isFinite()` + non-negative sanitization

**static/css/styles.css (#56, #57)**
- **#56**: Duplicate `.comparison-table` bloğu kaldırıldı; academic bölüm için `#academic .comparison-table` override'ları eklendi
- **#57**: `@media (max-width: 480px)` breakpoint eklendi — navbar, metric-value, metrics-grid, modal, decision-summary

**tests/ (#58, #59, #60, #61)**
- **#58-59**: `tests/test_env.py` — obs.shape, bounds, obs_space.contains assertions eklendi
- **#58-59**: `tests/test_env_pytest.py` — tam pytest suite (6 test: obs shape, bounds, step types, random steps, balance, metrics)
- **#60**: `tests/fixtures.py` — `make_mock_ohlcv()` + `make_mock_df_with_indicators()` — ağ gerektirmeyen sentetik veri
- **#61**: `pytest>=8.0.0` requirements.txt'e eklendi

**requirements.txt (#62, #63)**
- **#62**: `black>=24.0.0`, `flake8>=7.0.0` eklendi
- **#63**: `peewee`, `frozendict`, `annotated-doc` kaldırıldı

---

## Batch Bağımlılıkları

```
✅ Batch 0 (Duplicates) → ✅ Batch 1 (Fatal+Sec) → ✅ Batch 2 (İndikatörler) → ✅ Batch 3 (RL Env) → ✅ Batch 4 (Backend)
                                                                                                              |
                           ✅ Batch 6 (Scripts) ← ✅ Batch 3                                           ✅ Batch 5 (Frontend)
                                 |                                                                             |
                          ✅ Batch 7 (Data)                                                           ✅ Batch 8 (Kalite)
```

## Toplam Özet

| Batch | Issue Sayısı | Durum | Anahtar Dosyalar |
|-------|-------------|-------|-----------------|
| 0 Duplicates | 2 | ✅ Tamamlandı | docs/*, scripts/* root kopyalar |
| 1 Fatal+Sec | 8 | ✅ Tamamlandı | generate_academic_report.py, config.py, trading.py, JS |
| 2 İndikatörler | 5 | ✅ Tamamlandı | technical_indicators.py |
| 3 RL Env | 5 | ✅ Tamamlandı | trading_env.py, daily_trading.py |
| 4 Backend | 10 | ✅ Tamamlandı | trading.py, daily_trading.py, model_analysis.py, schemas |
| 5 Frontend | 6 | ✅ Tamamlandı | dashboard.js, academic-analysis.js |
| 6 Scripts | 5 | ✅ Tamamlandı | train_a2c_phase1.py, data_fetcher.py, main.py, run_server.py |
| 7 Data | 2 | ✅ Tamamlandı | data_fetcher.py |
| 8 Kalite | 16 | ✅ Tamamlandı | tests/*, requirements.txt, styles.css, JS dosyaları |
| **Tamamlanan** | **54** (+8 merge=62 fix) | ✅ | — |
| **Açık** | **1** (#29 BUG) | ⚠️ | daily_trading.py |
| **Toplam** | **58** (+7 merge=65) | | |
