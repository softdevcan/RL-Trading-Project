# DRL ile Algoritmik Ticaret - 3 Fazlı Geliştirme Planı

**Referans Makale**: Ansari et al. (2024) - "A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning"

Bu belge, BIST-30 endeksi için DRL tabanlı alım-satım sistemi geliştirme planını içerir.

---

## 🎯 Proje Vizyonu

Ansari et al. (2024) makalesindeki **multifaceted (çok yönlü) yaklaşımı** BIST-30 endeksine adapte ederek, günlük hisse verileri, teknik indikatörler ve temel analiz verilerini birleştiren bir Reinforcement Learning sistemi geliştirmek.

### Ana Özellikler (Hedef)
- ✅ **Multi-Stock Trading**: Aynı anda birden fazla hisse ile işlem
- ✅ **Multifaceted State**: Daily + Technical + Fundamental (Faz 2'de)
- ⏳ **PSR Reward Function**: Portfolio + Sharpe + Daily Returns (Faz 2'de)
- ✅ **RL Algoritmaları**: A2C, PPO, TD3
- ✅ **FastAPI Backend**: Model sunumu ve real-time tracking

---

## 📊 Ansari et al. (2024) Metodolojisi - Uygunluk Kontrolü

### ✅ FAZ 1 - Tamamlanan Özellikler

| Özellik | Ansari et al. | Bizim Uygulama | Durum | Notlar |
|---------|---------------|----------------|-------|--------|
| **Environment** | Custom Gym | Gymnasium | ✅ | Modern Gym wrapper |
| **Multi-Stock** | S&P500 (birden fazla) | BIST-30 (5 hisse) | ✅ | Proof-of-concept |
| **State - Daily** | OHLCV | OHLCV | ✅ | `open, high, low, close, volume` |
| **State - Technical** | 5 indikatör | 5 indikatör | ✅ | MACD, RSI, CCI, ADX, Turbulence |
| **Action Space** | Continuous (shares) | Continuous [-100, +100] | ✅ | Box space |
| **RL Algorithms** | A2C, PPO, TD3 | A2C, PPO, TD3 | ✅ | Stable-Baselines3 |
| **Commission** | Yes (0.1%) | Yes (0.1%) | ✅ | `commission_rate=0.001` |
| **Portfolio Tracking** | Yes | Yes | ✅ | Balance + shares_owned |
| **Metrics** | Sharpe, Return, DD | Sharpe, Return, DD | ✅ | `get_metrics()` |

### ⚠️ FAZ 1 - EKSİKLİKLER (Ansari et al'a göre)

| Özellik | Ansari et al. | Bizim Uygulama | Durum | Faz |
|---------|---------------|----------------|-------|-----|
| **State - Fundamental** | 11 ratios (bilanço, gelir, nakit akış) | ❌ YOK | 🔜 | Faz 2 |
| **Reward Function** | PSR (Portfolio + Sharpe + αReturn) | ❌ Basit (ΔPortfolio) | 🔜 | Faz 2 |
| **Alpha Parameter** | Tuned (0.9 → 0.1) | ❌ YOK | 🔜 | Faz 2 |
| **Ablation Studies** | DTF vs DT vs TF | ❌ YOK | 🔜 | Faz 2 |
| **Full Stock Coverage** | S&P500 (çok sayıda) | 5 hisse (pilot) | 🔜 | Faz 2 (30 hisse) |
| **Hyperparameter Tuning** | Grid search / Optuna | ❌ Manual | 🔜 | Faz 2-3 |
| **Risk Management** | Position sizing, stop-loss | ❌ Basit | 🔜 | Faz 3 |

---

## ⚡ FAZ 1: Proof-of-Concept (✅ TAMAMLANDI)

**Süre**: 3-4 Hafta
**Hedef**: Temel sistem çalışabilir durumda

### 1.1 Veri Altyapısı ✅

- [x] BIST-30 hisse listesi ✅ `data/bist30_symbols.py`
- [x] `yfinance` ile günlük OHLCV verisi çekme (2018-2024) ✅ `data/data_fetcher.py`
- [x] Veri temizleme ve normalizasyon ✅ `DataFetcher.clean_data()`
- [x] Train/Validation/Test split (70/15/15) ✅ `DataFetcher.split_data()`
- [x] Eksik veri işleme (forward fill, interpolation) ✅

### 1.2 Teknik İndikatörler ✅

Ansari et al.'dan: **MACD, RSI, CCI, ADX, Turbulence**

- [x] Teknik indikatör hesaplayıcı ✅ `data/technical_indicators.py`
- [x] MACD (Moving Average Convergence Divergence) ✅
- [x] RSI (Relative Strength Index) ✅
- [x] CCI (Commodity Channel Index) ✅
- [x] ADX (Average Directional Index) ✅
- [x] Turbulence (Volatilite metriği) ✅

### 1.3 Trading Environment ✅

- [x] Gymnasium environment ✅ `env/trading_env.py`
- [x] Multi-stock environment (5 hisse) ✅
- [x] State: balance + shares_owned + OHLCV + 5 technical indicators ✅
- [x] Action space: [-100, +100] shares per stock ✅
- [x] Basit reward: Portfolio value değişimi ✅
- [x] Transaction costs: %0.1 komisyon ✅
- [x] Günlük adım boyutu ✅
- [x] Portfolio tracking ve metrics ✅

**State Space (Faz 1)**:
```python
state = [
    balance,              # 1
    shares_owned[5],     # 5
    OHLCV[5],           # 25 (5 stocks × 5 features)
    Technicals[5]       # 25 (5 stocks × 5 indicators)
]
# Total: 56 features
```

### 1.4 İlk RL Modeli ✅

- [x] A2C model eğitimi ✅ `scripts/train_a2c_phase1.py`
- [x] PPO model eğitimi ✅ (En kararlı - önerilen)
- [x] TD3 model eğitimi ✅ (Experience replay ile)
- [x] Stable-Baselines3 entegrasyonu ✅
- [x] Model kaydetme (`.zip`) ✅
- [x] Metrik hesaplama (Sharpe, Return, Drawdown) ✅
- [x] Tensorboard logging ✅

**⚠️ Önemli Bulgular - Algoritma Performansları**:
- **PPO**: ✅ En kararlı ve güvenilir (39.59% return, 67 işlem, Sharpe: 1.26)
- **A2C**: ⚠️ Tek thread ile zayıf performans (makale 16 parallel thread öneriyor)
- **TD3**: ✅ Off-policy learning, experience replay ile iyi sonuçlar

**Optimized Hyperparameters**:
```python
# PPO (Önerilen)
PPO(n_steps=2048, batch_size=64, n_epochs=10, ent_coef=0.01)

# A2C (Dikkatli ayar gerekir)
A2C(n_steps=512, normalize_advantage=True, use_rms_prop=True, ent_coef=0.01)

# TD3 (Gelişmiş)
TD3(buffer_size=100000, batch_size=256, action_noise=NormalActionNoise)
```

### 1.5 Basit Backtesting ✅

- [x] Test setinde model performansı ✅ `evaluate_model()`
- [x] Metrics: Cumulative Return, Sharpe, Drawdown ✅

### 1.6 FastAPI & Web UI ✅

- [x] Trading API endpoints ✅ `app/api/routes/trading.py`
- [x] Pydantic schemas (validation) ✅ `app/schemas/trading.py`
- [x] Background training tasks ✅
- [x] Model management endpoints ✅
- [x] Single-page web UI ✅ `static/index.html`
- [x] Modüler yapı (HTML/CSS/JS ayrı) ✅
- [x] Real-time training progress ✅
- [x] Chart.js grafikler ✅

**API Endpoints**:
- `POST /api/trading/train` - Eğitim başlat
- `GET /api/trading/train/status` - Eğitim durumu
- `GET /api/trading/models` - Modelleri listele
- `GET /api/trading/models/{name}/metrics` - Model metrikleri
- `DELETE /api/trading/models/{name}` - Model sil
- `POST /api/trading/data/generate?phase=X&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD` - Veri oluştur (date range ile)
- `GET /api/trading/data/info` - Aktif veri seti durumu
- `GET /api/trading/data/list` - Tüm veri setlerini listele (NEW! ✨)

**Deliverables (1. Ara Değerlendirme)** ✅:
- ✅ Çalışan Trading Environment
- ✅ A2C/PPO/TD3 desteği
- ✅ Temel performans metrikleri
- ✅ FastAPI backend & Web UI
- ✅ Model training & management API
- ✅ GitHub repository

---

## 🚀 FAZ 2: Multifaceted Approach (📌 SONRAKİ ADIM - 2. Ara Değerlendirme)

**Süre**: 4-5 Hafta
**Hedef**: Ansari et al. metodolojisinin tam implementasyonu

### 2.1 Fundamental Data Entegrasyonu

**Kritik**: Ansari et al. makalesinin en önemli katkısı!

**Alpha Vantage API** ile çeyreklik veriler:

```python
# Fundamental Indicators (Ansari et al. Table 1)
LIQUIDITY_RATIOS = [
    'current_ratio',           # Cari Oran
    'acid_test_ratio',         # Asit-Test Oranı
    'operating_cashflow_ratio' # Faaliyet Nakit Akış Oranı
]

LEVERAGE_RATIOS = [
    'debt_ratio',              # Borç Oranı
    'debt_to_equity',          # Borç/Özkaynak
    'interest_coverage'        # Faiz Karşılama
]

EFFICIENCY_RATIOS = [
    'asset_turnover',          # Aktif Devir Hızı
    'inventory_turnover',      # Stok Devir Hızı
    'days_sales_inventory'     # Ortalama Stok Günü
]

PROFITABILITY_RATIOS = [
    'return_on_assets',        # ROA
    'return_on_equity'         # ROE
]
```

**Görevler**:
- [ ] Alpha Vantage API key alma (ücretsiz tier: 25 calls/day)
- [ ] `data/fundamental_fetcher.py` modülü oluşturma
- [ ] Çeyreklik veriyi günlük veriye dönüştürme (forward fill)
- [ ] 11 fundamental özellik ekleme
- [ ] Veri normalizasyonu (StandardScaler)

**Zorluklar**:
- BIST hisseleri için fundamental data bulma (Alpha Vantage US odaklı)
- Alternatif: Yahoo Finance, Investing.com, KAP (Kamu Aydınlatma Platformu)
- Çeyreklik → Günlük interpolasyon stratejisi

### 2.2 PSR Reward Function ⭐

**Ansari et al. Equation (1)** - En kritik katkı!

```python
def calculate_psr_reward(portfolio_change, sharpe_ratio, daily_return, alpha=0.9):
    """
    PSR = ΔPortfolio + Sharpe + α * DailyReturn

    Args:
        alpha: BIST-30 için tune edilecek (Ansari: 0.9 S&P500'de optimal)
    """
    reward = portfolio_change + sharpe_ratio + alpha * daily_return
    return reward

# Sharpe Ratio (Ansari Eq. 2)
def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Sharpe = (mean(returns) - rf) / std(returns)
    """
    excess_return = returns.mean() - risk_free_rate
    std_dev = returns.std()
    return excess_return / std_dev if std_dev > 0 else 0
```

**Görevler**:
- [ ] `trading_env.py`'de `step()` fonksiyonunu güncelleme
- [ ] Rolling Sharpe ratio hesaplama (window=30 gün)
- [ ] Alpha parameter tuning (0.1 → 1.0 grid search)
- [ ] Risk-free rate için TCMB faiz oranı kullanma

### 2.3 Enhanced State Representation

Ansari et al. **Figure 2**'ye göre:

```python
# State Vector (Faz 2)
state = [
    balance,                    # 1 feature
    shares_owned[0:N],         # N features (30 hisse)

    # Daily Data (5*N features)
    open[0:N], high[0:N], low[0:N], close[0:N], volume[0:N],

    # Technical Indicators (5*N features)
    macd[0:N], rsi[0:N], cci[0:N], adx[0:N], turbulence[0:N],

    # Fundamental Indicators (11*N features) ← YENİ!
    current_ratio[0:N], acid_test[0:N], ..., roe[0:N]
]

# Total: 1 + N + 5N + 5N + 11N = 1 + 22N features
# BIST-30 (N=30): 1 + 22*30 = 661 features
```

**Görevler**:
- [ ] `trading_env.py` güncelleme (fundamental features ekleme)
- [ ] State normalization strategy
- [ ] Feature selection (PCA / feature importance)

### 2.4 Çoklu RL Modelleri

```python
# A2C, PPO, TD3 ile karşılaştırmalı eğitim
models = {
    'A2C': A2C('MlpPolicy', env, learning_rate=0.0007, n_steps=5),
    'PPO': PPO('MlpPolicy', env, learning_rate=0.0003, n_steps=2048),
    'TD3': TD3('MlpPolicy', env, learning_rate=0.001, buffer_size=1_000_000)
}

for name, model in models.items():
    model.learn(total_timesteps=200_000)
    model.save(f'models/{name}_multifaceted')
```

**Görevler**:
- [ ] Her algoritma için hyperparameter tuning
- [ ] `train_multi_models.py` script oluşturma
- [ ] Karşılaştırmalı evaluation

### 2.5 Ablation Studies ⭐

**Ansari et al. Figure 10** - Hangi veri kaynağı en önemli?

```python
experiments = [
    'DTF',  # Daily + Technical + Fundamental (Full model)
    'DT',   # Daily + Technical (No fundamental)
    'TF',   # Technical + Fundamental (No daily price)
    'D',    # Daily only
    'T',    # Technical only
    'F'     # Fundamental only
]
```

**Deliverables (2. Ara Değerlendirme)**:
- [ ] Fundamental data entegrasyonu
- [ ] PSR reward function
- [ ] 3 RL modeli (A2C, PPO, TD3) karşılaştırması
- [ ] Kapsamlı backtesting sonuçları
- [ ] Ablation study grafikleri
- [ ] Teknik rapor (Ansari et al. formatında)

---

## 🎯 FAZ 3: Production System (Final Raporu)

**Süre**: 3-4 Hafta
**Hedef**: API ile model sunumu ve canlı tahminler

### 3.1 Real-Time Prediction API
### 3.2 WebSocket Real-Time Updates
### 3.3 Risk Management
### 3.4 Deployment (Docker, CI/CD)

**Deliverables (Final Raporu)**:
- [ ] Production-ready FastAPI backend
- [ ] Real-time trading signals
- [ ] WebSocket live updates
- [ ] Risk management system
- [ ] Docker deployment
- [ ] Final project report (IEEE format)

---

## 📋 SONRAKİ ADIMLAR (Yarınki Geliştirme)

### 🔴 Öncelik 1 - Kritik Eksiklikler

1. **PSR Reward Function** (`env/trading_env.py`)
   - Basit reward'ı PSR ile değiştir
   - Alpha parameter tuning
   - Rolling Sharpe ratio hesaplama

2. **Fundamental Data Pipeline** (`data/fundamental_fetcher.py`)
   - Alpha Vantage / KAP entegrasyonu
   - 11 fundamental ratio hesaplama
   - Çeyreklik → Günlük dönüştürme

3. **Enhanced State Space** (`env/trading_env.py`)
   - Fundamental features ekleme
   - 56 → 661 features (30 hisse)
   - Normalization strategy

### 🟡 Öncelik 2 - Performans

4. **Hyperparameter Tuning**
   - Optuna ile otomatik tuning
   - Learning rate, n_steps, batch_size

5. **PPO ve TD3 Eğitimi**
   - A2C dışında diğer algoritmaları da test et
   - Karşılaştırmalı analiz

6. **Full BIST-30 Coverage**
   - 5 hisse → 30 hisse
   - Model kapasitesini artırma

### 🟢 Öncelik 3 - Analiz

7. **Ablation Studies**
   - DTF vs DT vs TF
   - Her kombinasyon için eğitim

8. **Comprehensive Metrics**
   - Sortino, Calmar, Omega ratios
   - Ansari et al. Table 5-6 formatında

---

---

## 🔧 Son Değişiklikler (2025-11-12)

### Duplicate Symbol Column Bug Fix ✅

**Problem**: CSV dosyasında `symbol` sütunu iki kez kaydediliyordu:
- Biri multi-index'ten (`to_csv(index=True)`)
- Biri de kod içinde manuel eklenen sütundan

**Düzeltme**:
1. [data/data_fetcher.py:167](data/data_fetcher.py#L167) - `symbol_df['symbol'] = symbol` satırı kaldırıldı
2. [data/technical_indicators.py:196](data/technical_indicators.py#L196) - Aynı satır kaldırıldı
3. `pd.concat(..., keys=symbols)` zaten multi-index olarak symbol ekliyor

**Sonuç**: Artık CSV'de sadece index olarak symbol var (doğru format)

### Web Arayüzünden Veri Yönetimi ✅

**Yeni Özellikler**:
- Veri İstatistikleri sekmesine "Veri Oluştur" butonu eklendi
- "Veri Durumunu Kontrol Et" butonu ile mevcut veri bilgisi görüntüleme
- API endpoints: `POST /api/trading/data/generate`, `GET /api/trading/data/info`

**Kullanım**:
1. Web arayüzünü aç: `http://localhost:8000`
2. "📈 Veri İstatistikleri" sekmesine git
3. "🔄 Veri Oluştur" butonuna tıkla
4. Veri otomatik olarak indirilir, temizlenir ve teknik indikatörler eklenir

### Test Loop Bug Fix ✅

**Problem**: DummyVecEnv `done` array döndürüyor ama boolean gibi kullanılıyordu
```python
done = False
while not done:  # [False] array her zaman truthy!
```

**Düzeltme**: [app/api/routes/trading.py:469](app/api/routes/trading.py#L469)
```python
done = np.array([False])
while not done[0]:  # Array index kontrolü
```

### Reward Function İyileştirmesi ✅

**Problem**: Model overtrading yapıyordu (712 trade) - trade başına bonus vardı!

**Eski Reward**:
```python
reward = portfolio_change * 100
if trades_executed > 0:
    reward += 0.01 * trades_executed  # ❌ YANLIŞ - overtrading teşvik ediyor!
```

**Yeni Reward**: [env/trading_env.py:186-194](env/trading_env.py#L186-L194)
```python
portfolio_change_pct = (ΔPortfolio / prev_value) * 100
commission_penalty = (total_commission / initial_balance) * 100
reward = portfolio_change_pct - commission_penalty  # ✅ Komisyon cezası
```

**Sonuç**: Artık model gereksiz trade yapmaktan cezalandırılıyor

### DummyVecEnv Autoreset Bug Fix ✅

**Problem**: Test evaluation sırasında model trade yapıyor ama metrics=0 çıkıyor!

**Kök Sebep**: DummyVecEnv `done=True` olunca **otomatik reset** yapıyor
```python
obs, reward, done, info = test_env.step(action)  # done=True olunca
# DummyVecEnv içinde: env.reset() çağrılıyor!
# trades_history = [] oldu

metrics = test_env.envs[0].get_metrics()  # total_trades = 0 ❌
```

**Çözüm**: [app/api/routes/trading.py:481-494](app/api/routes/trading.py#L481-L494)
```python
# Her step öncesi state kaydet
pre_step_trades = len(actual_env.trades_history)
pre_step_portfolio_values = actual_env.portfolio_values.copy()

obs, reward, done, info = test_env.step(action)

if done[0]:
    # State'i geri yükle
    actual_env.trades_history = actual_env.trades_history[:pre_step_trades]
    actual_env.portfolio_values = pre_step_portfolio_values
```

**Sonuç**: Artık test metrics doğru alınıyor

---

## 🎨 UI/UX İyileştirmeleri (2025-11-13)

### Trade History Modal & Visualization ✅

**Özellikler**:
- Model detayları modal ekranı
- Gerçek trade geçmişi tablosu (tarih, sembol, işlem, miktar, fiyat, toplam)
- Portfolio değeri line chart (test set boyunca)
- Trade aktivitesi scatter plot (hisse bazında alım/satım)

**Dosyalar**:
- [static/index.html:313-358](static/index.html#L313-L358) - Modal HTML
- [static/js/dashboard.js:561-586](static/js/dashboard.js#L561-L586) - `showModelDetails()`
- [app/schemas/trading.py:94-95](app/schemas/trading.py#L94-L95) - `trades` ve `portfolio_history` fields
- [app/api/routes/trading.py:511-543](app/api/routes/trading.py#L511-L543) - Trade history kaydetme

**Chart.js Canvas Reuse Bug Fix**: `Chart.getChart()` ile mevcut chart'ı kontrol edip destroy etme

### Interactive Model Comparison ✅

**Öncesi**: Tüm modeller otomatik karşılaştırılıyordu
**Sonrası**: İstenen modeller seçilebiliyor

**Özellikler**:
- Model seçim grid'i (checkbox benzeri görsel)
- Select All / Deselect All butonları
- Dinamik karşılaştırma (sadece seçili modeller)
- Model bilgileri (algoritma, return, sharpe, trades)

**Dosyalar**:
- [static/index.html:279-324](static/index.html#L279-L324) - Model seçim UI
- [static/js/dashboard.js:771-807](static/js/dashboard.js#L771-L807) - `loadAvailableModels()`
- [static/js/dashboard.js:809-835](static/js/dashboard.js#L809-L835) - Selection yönetimi

### Navbar-Style Tabs & Header Consolidation ✅

**Değişiklik**: Büyük header kaldırıldı, title navbar'ın soluna taşındı

**Öncesi**:
```
[------------ Header ------------]
[Tab1] [Tab2] [Tab3] [Tab4]
```

**Sonrası**:
```
[🚀 Title + Subtitle | Tab1 Tab2 Tab3 Tab4 →]
```

**Faydası**: Önemli miktarda dikey alan kazanıldı

**Dosyalar**:
- [static/index.html:16-28](static/index.html#L16-L28) - Navbar yapısı
- [static/css/styles.css:27-80](static/css/styles.css#L27-L80) - Navbar stilleri

### Dynamic Data Generation with Date Range ✅

**Özellikler**:
- Başlangıç ve bitiş tarihi seçimi
- Quick range butonları (1y, 3y, 5y)
- Form validation (start < end)
- API ve UI tarafı entegrasyonu

**Dosyalar**:
- [app/api/routes/trading.py:158-187](app/api/routes/trading.py#L158-L187) - Date parametreleri
- [static/index.html:179-206](static/index.html#L179-L206) - Date picker UI
- [static/js/dashboard.js:454-506](static/js/dashboard.js#L454-L506) - `generateDataWithDates()`
- [static/js/dashboard.js:509-543](static/js/dashboard.js#L509-L543) - Quick range helpers

### Non-Scrollable Single-Page Layout ✅

**Hedef**: Tüm içerik sayfaya sığmalı, sayfa scroll olmamalı

**Değişiklikler**:
- `body`: `overflow: hidden`, `height: 100vh`
- Container: flexbox column layout
- Tab content: `flex: 1`, `overflow: hidden`
- Sadece gerekli bölümler scroll (model listesi, tablolar)
- Sticky table headers
- Tüm padding/margin azaltıldı

**Dosyalar**:
- [static/css/styles.css:9-25](static/css/styles.css#L9-L25) - Body & container
- [static/css/styles.css:87-96](static/css/styles.css#L87-L96) - Tab content flexible
- [static/css/styles.css:115-136](static/css/styles.css#L115-L136) - Cards overflow control
- [static/css/styles.css:297-310](static/css/styles.css#L297-L310) - Scrollable sections
- [static/css/styles.css:424-446](static/css/styles.css#L424-L446) - Sticky headers

### Training Progress Removed ✅

**Neden**: Eğitim senkron (blocking) çalışıyor, gerçek zamanlı ilerleme takibi yapılamıyor

**Kaldırılanlar**:
- Progress bar (dashboard)
- Training progress chart
- Status polling

**Eklenen**: Bilgilendirme mesajı (eğitim 5-30 dakika sürebilir)

**Dosyalar**:
- [static/index.html:34-40](static/index.html#L34-L40) - Progress container kaldırıldı
- [static/index.html:157-164](static/index.html#L157-L164) - Info box eklendi
- [static/js/dashboard.js:84](static/js/dashboard.js#L84) - Chart initialization kaldırıldı
- [static/js/dashboard.js:193-200](static/js/dashboard.js#L193-L200) - Training start güncellendi

### Dataset Management & Listing ✅

**Özellikler**:
- `data/` klasöründeki tüm CSV dosyalarını listele
- Her dataset için detaylı bilgi:
  - Dosya adı, boyut (MB)
  - Tarih aralığı, hisse listesi
  - Satır sayıları (total, train, val, test)
  - Oluşturulma tarihi
- Tablo formatında gösterim

**API Endpoint**: `GET /api/trading/data/list`

**Dosyalar**:
- [app/api/routes/trading.py:254-315](app/api/routes/trading.py#L254-L315) - `/data/list` endpoint
- [static/index.html:203-223](static/index.html#L203-L223) - Datasets tablo UI
- [static/js/dashboard.js:578-635](static/js/dashboard.js#L578-L635) - `loadDatasetsList()`

### Performance Chart Fixed ✅

**Problem**: Dashboard'daki "Performans Trendi" sahte veri gösteriyordu

**Öncesi**:
- ❌ Rastgele noise ile simüle edilmiş veri
- ❌ Sadece final_portfolio_value'dan lineer interpolasyon
- ❌ Gerçek performansı yansıtmıyordu

**Sonrası**:
- ✅ Test sırasında kaydedilen **gerçek portfolio_history** kullanılıyor
- ✅ Her günün portfolio değeri gösteriliyor
- ✅ Trade'lerin etkisi görünüyor
- ✅ Model adı label'da gösteriliyor

**Dosyalar**:
- [static/js/dashboard.js:267-288](static/js/dashboard.js#L267-L288) - `updatePerformanceChart()` tamamen yeniden yazıldı
- [static/js/dashboard.js:326-329](static/js/dashboard.js#L326-L329) - Otomatik model seçimi

**Veri Akışı**:
```
init() → loadModels() → selectModel(0) → updateMetrics() → updatePerformanceChart()
                                                                         ↓
                                                              Real portfolio_history
```

### Dark Mode Fixes ✅

**Problem**: "Mevcut Veri Bilgisi" kartı beyaz background + beyaz text (okunamıyor)

**Çözüm**: Dark theme renkleri
- Background: `#1e3a5f` (dark blue)
- Text: `#e2e8f0` (light gray)
- Border: `#334155`

**Dosya**: [static/index.html:204-206](static/index.html#L204-L206)

---

**Son Güncelleme**: 2025-11-13
**Durum**: Faz 1 Tamamlandı ✅ + UI/UX İyileştirmeleri ✅
**Sonraki Milestone**: Faz 2 (Full BIST-30) + Hyperparameter optimization
