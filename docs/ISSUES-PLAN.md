# RL Trading Project - 63 Issue Fix Plan

## Context
ISSUES.md'de belgelenen 63 sorun (5 FATAL, 3 SEC, 22 BUG, 3 DEPR, 30 DESIGN) 8 batch halinde sirayla cozulecek. Oncelik: Fatal > Security > RL Core > Backend > Frontend > Polish. Her batch bagimsiz test edilebilir. Batch 2-3 sonrasi tum modeller yeniden egitilmeli.

---

## Batch 1: Fatal Crashlar ve Guvenlik Aciklari
**Kapsam: KUCUK | Issues: #15-18, #28, #21-22, #43, #51 (10 issue)**

### Dosyalar ve Degisiklikler:

**scripts/generate_academic_report.py** (#15-18)
- L13: `from app.environments.trading_env import MultiStockTradingEnv` → `from env.trading_env import TradingEnv`
- L17: Pickle yerine `DataFetcher.load_data()` + `split_data()` pipeline kullan
- L75: `env.portfolio_value` → `env._get_portfolio_value()`
- L82: `env.trade_history` → `env.trades_history`

**app/services/daily_trading.py** (#28)
- L110: `TechnicalIndicatorCalculator` → `TechnicalIndicators`

**app/core/config.py** (#21)
- L21-22: `CORS_ORIGINS: ["*"]` → `["http://localhost:8000", "http://127.0.0.1:8000"]`

**app/api/routes/trading.py** (#22)
- L323, L146: `sanitize_model_name()` utility ekle - `os.path.basename()` + `^[a-zA-Z0-9_.-]+$` regex ile path traversal engelle

**static/js/dashboard.js** (#43)
- L440: `innerHTML` → `textContent` (veya `escapeHtml()` helper)

**static/js/academic-analysis.js** (#51)
- L164: innerHTML icinde `${modelName}` sanitize et

**Dogrulama:**
- `python scripts/generate_academic_report.py` ImportError/AttributeError vermemeli
- Path traversal denemesi (`../../etc/passwd`) 400 donmeli
- CORS header'da `*` olmamali

---

## Batch 2: Teknik Indikator Duzeltmeleri
**Kapsam: ORTA | Issues: #6-11 (6 issue)**

### Dosyalar ve Degisiklikler:

**data/technical_indicators.py**

**#6 RSI - Wilder's EMA** (L58-59)
```python
# ONCEKI: gain.rolling(window=period).mean()
# SONRAKI:
avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
```
Ref: Wilder 1978 "New Concepts in Technical Trading Systems"

**#7 ADX - Wilder's EMA** (L112-114)
```python
atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
```

**#8 ADX Division by Zero** (L117)
```python
denominator = plus_di + minus_di
dx = np.where(denominator > 0, 100 * np.abs(plus_di - minus_di) / denominator, 0)
```

**#9 Turbulence - Mahalanobis Distance** (L135-143)
- `add_indicators_to_multi_symbol_df()` seviyesinde hesapla (tek hisse degil, cross-sectional)
- Formula: `turb_t = (r_t - mu)^T * Sigma^{-1} * (r_t - mu)`
- Ref: Liu et al. 2020 "FinRL", Kritzman & Li 2010

**#10 MACD Signal Line**
- Signal line (`macd.ewm(span=9).mean()`) ve histogram ekle
- State space sadece `macd` kullandiginden geriye uyumlu (additive)

**#11 Deprecated fillna** (L166-167)
- `.fillna(method='ffill')` → `.ffill()`
- `.fillna(method='bfill')` → `.bfill()`

**Dogrulama:**
- RSI ciktisini `pandas_ta.rsi()` ile karsilastir
- ADX ciktisini `pandas_ta.adx()` ile karsilastir
- Turbulence > 0 ve bilinen volatil donemlerde spike

---

## Batch 3: RL Environment Core
**Kapsam: ORTA | Issues: #1-5, #30 (6 issue)**

### Dosyalar ve Degisiklikler:

**env/trading_env.py**

**#1 Bounded Observation Space** (L71-76)
- `Box(-np.inf, np.inf)` → `Box(-10, 10)`
- `_get_observation()` sonunda `np.clip(state, -10, 10)` ekle

**#2 Dynamic Normalization** (L305-308)
- `(price - 50) / 50` → rolling z-score: `(price - rolling_mean) / (rolling_std + 1e-8)`
- Training data'dan rolling istatistikler hesapla ve kaydet
- Inference'da ayni istatistikleri yukle

**#3 PSR Reward Function** (L180-188)
- Portfolio change % yerine Differential Sharpe Ratio (Moody & Saffell 2001)
- Bu, PSR'nin step-wise yaklasimi
- Ref: Ansari et al. 2024, Bailey & Lopez de Prado 2012

**#4 Silent Missing Data** (L335-337)
- `KeyError → 0.0` yerine: son bilinen veriden ffill + `logger.warning()`

**#5 min_threshold** (L149)
- `min_threshold = 0` → `min_threshold = 1` (minimum 1 hisse)

**app/services/daily_trading.py** (#30)
- L246-256: Ayni dynamic normalization'i `build_live_state()`'e uygula

**KRITIK: Batch 2-3 sonrasi tum modeller gecersiz olur - yeniden egitim gerekli!**

**Dogrulama:**
- `env.observation_space.contains(obs)` her adimda True
- Normalize fiyatlar %99 [-3, 3] araliginda
- 1000 adim egitim - reward dagilimi degenere degil
- `daily_trading.py` ve `trading_env.py` normalization'i eslesiyor

---

## Batch 4: Backend Bug Fix
**Kapsam: ORTA | Issues: #23-27, #29, #31, #33-36 (11 issue)**

### Dosyalar ve Degisiklikler:

**app/api/routes/trading.py**
- #23 (L28-35): `training_state` icin `asyncio.Lock()` ekle
- #24 (L561): `deterministic=False` → `deterministic=True`
- #25 (L563-575): DummyVecEnv workaround kaldir, raw `TradingEnv` ile eval
- #26 (L914-937): Hardcoded semboller → `get_symbols(phase=1)` kullan
- #27: 22+ hardcoded path → `config.py`'da `MODELS_DIR`, `RESULTS_DIR`, `DATA_DIR` tanimla

**app/services/daily_trading.py**
- #29 (L529-533): `decision["price"]` → `current_prices[symbol]` ile portfolio hesapla
- #31: JSON dosya I/O'ya `filelock` (Windows uyumlu) ekle

**app/schemas/trading.py**
- #33 (L172): `action: str` → `action: Literal["BUY", "SELL", "HOLD"]`
- #34 (L136): `risk_mode: str` → `risk_mode: Literal["conservative", "moderate", "aggressive"]`

**app/services/model_analysis.py**
- #35 (L16): Global `warnings.filterwarnings('ignore')` kaldir, spesifik context manager kullan
- #36 (L93): `np.inf` → `9999.99` (JSON serialization uyumlu)

**Dogrulama:**
- 2 esz zamanli training istegi: ikincisi 400 donmeli
- Gecersiz `risk_mode` → 422 validation error
- `profit_factor` JSON'da finite sayi

---

## Batch 5: Frontend Bug Fix
**Kapsam: ORTA | Issues: #39-42, #46-47, #52-55 (10 issue)**

### Dosyalar ve Degisiklikler:

**static/js/dashboard.js**
- #39 (L197-200): `progressBar`, `stepInfo` referanslarini kaldir/null check
- #40 (L177,438): `showError()` 2 parametre kabul etsin (message, type='error')
- #41 (L99-134): Chart olusturmadan once `.destroy()` cagir
- #42 (L988): `$` → `₺`

**static/js/daily-trading.js**
- #46 (L156): `['ASELS', 'THYAO', 'EREGL', 'KCHOL', 'SAHOL']` → Phase 1 dogru sembolleri
- #47 (L242): `'apply-decision-btn'` → `'apply-decisions'`

**static/js/academic-analysis.js**
- #52 (L51): Infinite polling → `maxRetries=20` counter ekle
- #53 (L470-473): Null check ekle: `const el = document.getElementById(...); if (el) el.style.display = ...`
- #54 (L204): 2-point chart icin minimum veri kontrolu

**static/index.html**
- #55 (L479-497): EREGL/KCHOL/SAHOL → AKBNK/TUPRS/BIMAS (Phase 1 dogru semboller)

**Dogrulama:**
- Dashboard yukleme: console error yok
- Chart degistirince memory leak yok
- Daily trading dogru semboller gosteriyor
- Apply decisions butonu calisiyor
- Academic analysis polling 20 denemeden sonra duruyor

---

## Batch 6: Script Fix ve Deprecation Temizligi
**Kapsam: KUCUK | Issues: #12, #19-20, #32, #37 (5 issue)**

### Dosyalar ve Degisiklikler:

**scripts/train_a2c_phase1.py**
- #19 (L121): `n_steps=256` → `n_steps=n_steps` (parametre kullan)
- #20 (L180-182): `done = done[0] if isinstance(done, np.ndarray) else done`

**data/data_fetcher.py**
- #12 (L151,154): `.fillna(method='ffill')` → `.ffill()`

**app/main.py**
- #32 (L63,70): `@app.on_event()` → `@asynccontextmanager` lifespan

**run_server.py**
- #37: `reload=True` → `reload=settings.DEBUG`

---

## Batch 7: Data Pipeline Robustness
**Kapsam: KUCUK | Issues: #13-14 (2 issue)**

### Dosyalar ve Degisiklikler:

**data/data_fetcher.py**
- #13: Exponential backoff retry (max 3 deneme, 2^n saniye bekleme)
- #14: Minimum veri kapsami dogrulama (%80+ coverage gerekli)

---

## Batch 8: Kod Kalitesi ve Test Altyapisi
**Kapsam: ORTA | Issues: #38, #44-45, #48-50, #56-63 (16 issue)**

### Dosyalar ve Degisiklikler:

**Frontend:**
- #38: dashboard.js global degiskenler → `const AppState = {...}` namespace
- #44: Tum chart re-creation'larda `.destroy()` (Batch 5'in genislemesi)
- #45: Polling frekansini 10 req/min'e dusur
- #48: daily-trading.js `alert()` → toast notification div
- #49: console.log'lari `if (DEBUG)` icine al
- #50: Input sanitization (balance, shares)

**CSS:**
- #56: Duplicate table style'lari birlestir
- #57: Responsive breakpoint ekle (768px, 480px)

**Tests:**
- #58-59: Test script'lerine assertion ekle (`assert obs.shape == (56,)` vb.)
- #60: Mock data fixture'lari ekle (offline test icin)

**Requirements:**
- #61: `pytest` ekle
- #62: `black`, `flake8`, `filelock` ekle
- #63: Kullanilmayan dep'leri temizle (`peewee`, `annotated-doc`, `frozendict`)

---

## Batch Bagimliliklari

```
Batch 1 (Fatal+Sec) → Batch 2 (Indikatorler) → Batch 3 (RL Env) → Batch 4 (Backend)
                                                                         |
                       Batch 6 (Scripts) ← Batch 3                  Batch 5 (Frontend)
                           |                                             |
                       Batch 7 (Data)                               Batch 8 (Kalite)
```

## Toplam Ozet

| Batch | Issue Sayisi | Anahtar Dosyalar |
|-------|-------------|-----------------|
| 1 Fatal+Sec | 10 | generate_academic_report.py, daily_trading.py, config.py, trading.py, JS dosyalari |
| 2 Indikatorler | 6 | technical_indicators.py |
| 3 RL Env | 6 | trading_env.py, daily_trading.py |
| 4 Backend | 11 | trading.py, daily_trading.py, model_analysis.py, schemas, config.py |
| 5 Frontend | 10 | dashboard.js, daily-trading.js, academic-analysis.js, index.html |
| 6 Scripts | 5 | train_a2c_phase1.py, data_fetcher.py, main.py, run_server.py |
| 7 Data | 2 | data_fetcher.py |
| 8 Kalite | 16 | tests/*, requirements.txt, styles.css, JS dosyalari |
| **Toplam** | **63** | |
