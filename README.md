# 🚀 RL Trading System - BIST-30 Algoritmik Ticaret

Deep Reinforcement Learning kullanarak BIST-30 hisseleri için algoritmik ticaret sistemi. **Ansari et al. (2024) - A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning** makalesine dayanmaktadır.

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Web Dashboard](#web-dashboard)
- [API Kullanımı](#api-kullanımı)
- [Proje Yapısı](#proje-yapısı)
- [Geliştirme Planı](#geliştirme-planı)

---

## 🎯 Proje Hakkında

Bu proje, **3 fazlı** bir geliştirme süreciyle BIST-30 endeksi için DRL tabanlı trading sistemi geliştirmeyi hedefler:

- **Faz 1 (POC)**: 5 hisse ile temel sistem (✅ Tamamlandı)
- **Faz 2**: Advanced Prediction System — Ensemble tahmin + RL entegrasyonu (✅ Tamamlandı)
- **Faz 3**: Production-ready sistem

### Temel Özellikler

✅ **Multi-Stock Trading Environment** (Gymnasium)
✅ **3 RL Algoritması**: A2C, PPO, TD3 (Stable-Baselines3)
✅ **5 Teknik İndikatör**: MACD, RSI, CCI, ADX, Turbulence
✅ **FastAPI Backend**: Model eğitimi ve yönetimi
✅ **Web Dashboard**: Dinamik görselleştirme (Chart.js)
✅ **Real-time Training**: Background task ile eğitim
✅ **Ensemble Tahmin Sistemi**: XGBoost + LightGBM + CatBoost + BiLSTM + TFT
✅ **Hiperparametre Optimizasyonu**: Optuna + Walk-forward CV + purge/embargo
✅ **Çoklu Veri Kaynağı**: yfinance (OHLCV), TCMB EVDS (makro), fundamental, altın/döviz

### Teknoloji Stack

**Backend:**
- FastAPI + Uvicorn
- Stable-Baselines3 (PyTorch)
- Gymnasium (OpenAI Gym)
- yfinance (BIST verileri)

**Frontend:**
- Vanilla JavaScript (ES6+)
- Chart.js 4.4.0
- Responsive CSS Grid

**Data & ML:**
- pandas, numpy, scikit-learn
- pandas-ta (Teknik indikatörler)
- XGBoost, LightGBM, CatBoost
- Optuna (Hiperparametre optimizasyonu)
- evds (TCMB EVDS API — faiz, enflasyon)
- borsapy (Altın/döviz verisi)

---

## 📦 Kurulum

### 1. Repository'yi Klonlayın

```bash
git clone <repo-url>
cd RL-Trading-Project
```

### 2. Virtual Environment Oluşturun

```bash
python -m venv venv
```

### 3. Virtual Environment'ı Aktif Edin

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

**Ana Paketler:**
- fastapi, uvicorn, pydantic
- stable-baselines3, gymnasium, torch
- yfinance, pandas-ta, scikit-learn
- numpy, pandas, matplotlib

---

## 🚀 Hızlı Başlangıç

### Web Dashboard ile Kullanım (Önerilen)

```bash
# Sunucuyu başlat
python run_server.py
```

Tarayıcınızda açın: **http://localhost:8000/**

**Dashboard Özellikleri:**
- 📊 Model eğitimi (A2C/PPO/TD3)
- ⏱️ Gerçek zamanlı progress tracking
- 📈 Performans metrikleri (Sharpe, Return, Drawdown)
- 🤖 Model karşılaştırma
- 📉 Chart.js grafikleri

### TensorBoard ile İzleme

Eğitim sürecini gerçek zamanlı izlemek için:

```bash
# TensorBoard'u başlatın
tensorboard --logdir=logs

# Tarayıcıda açın: http://localhost:6006
```

**TensorBoard'da görüntülenecekler:**
- 📉 Loss curves (policy loss, value loss, entropy loss)
- 📊 Reward progression
- 🎯 Episode statistics
- 📈 Learning rate schedule
- 🔍 Gradient norms

### Test Scriptleri

```bash
# Environment testi
python tests/test_env.py

# PPO algoritması testi
python tests/test_ppo.py

# Tüm algoritmaları karşılaştır
python tests/test_all_algorithms.py

# Benchmark stratejileri
python scripts/benchmarking/test_benchmarks.py

# Akademik rapor oluştur
python scripts/analysis/generate_academic_report.py
```

---

## 🎨 Web Dashboard

### 4 Ana Sekme

#### 1️⃣ Dashboard
- Sistem durumu (eğitim/hazır)
- Aktif model bilgisi
- 5 metrik kartı (Return, Sharpe, Drawdown, Portfolio, Trades)
- Portfolio performans grafiği
- Model listesi (tıklanabilir)

#### 2️⃣ Model Eğitimi
- Algoritma seçimi (A2C, PPO, TD3)
- Faz seçimi (1, 2, 3)
- Hyperparameter ayarları
- Real-time eğitim progress grafiği

#### 3️⃣ Veri İstatistikleri
- BIST-30 hisse bilgileri
- Sektör dağılımı
- Teknik indikatörler özeti
- Veri periyodu istatistikleri

#### 4️⃣ Model Karşılaştırma
- Tüm modellerin karşılaştırma tablosu
- Algoritma performans grafiği (bar chart)
- Ortalama Sharpe Ratio karşılaştırması

### Frontend Yapısı

```
static/
├── index.html          # Temiz HTML yapısı
├── css/
│   └── styles.css     # Tüm CSS stilleri
└── js/
    └── dashboard.js   # Tüm JavaScript logic
```

---

## 🔌 API Kullanımı

### API Dokümantasyonu

```
http://localhost:8000/docs       # Swagger UI
http://localhost:8000/redoc      # ReDoc
```

### Eğitim Başlatma

```bash
curl -X POST "http://localhost:8000/api/trading/train" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "A2C",
    "phase": 1,
    "total_timesteps": 50000,
    "learning_rate": 0.0007,
    "initial_balance": 1000000
  }'
```

### Eğitim Durumu Sorgulama

```bash
curl "http://localhost:8000/api/trading/train/status"
```

### Modelleri Listeleme

```bash
curl "http://localhost:8000/api/trading/models"
```

### Ana Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | Web Dashboard |
| POST | `/api/trading/train` | Eğitim başlat |
| GET | `/api/trading/train/status` | Eğitim durumu |
| GET | `/api/trading/models` | Modelleri listele |
| GET | `/api/trading/models/{name}/metrics` | Model metrikleri |
| DELETE | `/api/trading/models/{name}` | Model sil |

---

## 📂 Proje Yapısı

```
RL-Trading-Project/
│
├── 📁 app/                      # FastAPI Backend
│   ├── api/routes/
│   │   └── trading.py          # Trading API endpoints
│   ├── schemas/
│   │   └── trading.py          # Pydantic models
│   ├── core/config.py          # Konfigürasyon
│   └── main.py                 # FastAPI app
│
├── 📁 static/                   # Frontend
│   ├── index.html              # Ana sayfa
│   ├── css/styles.css          # Stiller
│   └── js/dashboard.js         # JavaScript
│
├── 📁 data/                     # Veri İşleme
│   ├── bist30_symbols.py       # Hisse listesi
│   ├── data_fetcher.py         # OHLCV (yfinance, retry + incremental)
│   ├── technical_indicators.py # Teknik indikatörler
│   ├── macro_fetcher.py        # TCMB EVDS (faiz, enflasyon) + yfinance (döviz, BIST100)
│   ├── fundamental_fetcher.py  # Fundamental oranlar (ROE, ROA, P/E, P/B, ...)
│   └── gold_fetcher.py         # Altın/döviz (borsapy veya yfinance)
│
├── 📁 prediction/               # Gelişmiş Tahmin Sistemi (Faz 2)
│   ├── feature_engineer.py     # 10 özellik grubu (getiri, volatilite, makro, fundamental, rejim)
│   ├── feature_selector.py     # Otomatik feature selection (MI + permutation importance)
│   ├── models/                 # Multi-model mimarisi
│   │   ├── base.py             # BasePredictionModel ABC
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── lstm_model.py       # BiLSTM (PyTorch, CUDA)
│   │   ├── tft_model.py        # Temporal Fusion Transformer
│   │   └── ensemble.py         # Stacking meta-learner (Ridge + XGBoost)
│   ├── hyperopt.py             # Optuna HPO
│   ├── trainer.py              # Walk-forward + purge gap (5g) + embargo (3g)
│   ├── evaluator.py            # Direction acc, Profit Factor, IC, Diebold-Mariano
│   └── tracker.py              # Experiment tracking (JSON log)
│
├── 📁 env/                      # RL Environment
│   └── trading_env.py          # Gymnasium environment (Phase 2: +4×N prediction features)
│
├── 📁 scripts/                  # Utility Scripts
│   ├── training/               # Model eğitimi
│   ├── benchmarking/           # Performans testleri
│   ├── analysis/               # Analiz ve raporlama
│   └── debug/                  # Debug araçları
│
├── 📁 docs/                     # Dokümantasyon
│   ├── guides/                 # Kullanım kılavuzları
│   ├── development/            # Geliştirme notları
│   └── phase2/                 # Faz 2 dokümanları
│
├── 📁 models/                   # Eğitilmiş modeller (.zip)
├── 📁 results/                  # Metrikler ve raporlar
├── 📁 logs/                     # TensorBoard logs
├── 📁 tests/                    # Unit tests
│
├── 📄 run_server.py            # Server launcher
├── 📄 requirements.txt         # Dependencies
└── 📄 README.md                # Ana README
```

### Modül Bağımlılıkları

```
Web UI (static/)
    ↓ API calls
FastAPI Backend (app/)
    ↓ Background tasks
RL Training Pipeline
    ├─► data_fetcher.py → yfinance (OHLCV, incremental)
    ├─► technical_indicators.py → pandas-ta
    ├─► trading_env.py → Gymnasium
    └─► Stable-Baselines3 → A2C/PPO/TD3
        └─► Modeller (.zip) + Metrikler (.json)

Tahmin Pipeline (prediction/)
    ├─► data_fetcher.py      (OHLCV)
    ├─► macro_fetcher.py     (EVDS faiz/enflasyon + yfinance döviz)
    ├─► fundamental_fetcher.py (yfinance ROE/ROA/P/E/P/B)
    ├─► gold_fetcher.py      (borsapy/yfinance altın+döviz)
    ↓
    feature_engineer.py → feature_selector.py
    ↓
    trainer.py (walk-forward + purge/embargo)
    ↓
    XGBoost + LightGBM + CatBoost + BiLSTM + TFT
    ↓
    ensemble.py (stacking meta-learner)
    ↓
    trading_env.py (RL obs: +predicted_return/direction/confidence/agreement)
```

---

## 🎓 Akademik Analiz ve Raporlama

### Tez ve Makale için Kapsamlı Analiz

Projeye **akademik yayın kalitesinde** analiz ve görselleştirme sistemi eklenmiştir:

```bash
# Tüm modelleri karşılaştır ve akademik rapor oluştur
python generate_academic_report.py
```

**Oluşturulan Çıktılar:**

📁 **results/**
├── **figures/** (Publication-ready, 300 DPI)
│   ├── portfolio_comparison.pdf
│   ├── drawdown_comparison.pdf
│   ├── returns_distribution.pdf
│   ├── risk_return_scatter.pdf
│   └── performance_radar.pdf
│
├── **latex/** (Tez için hazır tablolar)
│   └── model_comparison.tex
│
├── **data/** (CSV ve JSON)
│   ├── model_comparison.csv
│   └── detailed_results.json
│
└── **ANALYSIS_REPORT.txt** (Özet rapor)

### Hesaplanan Metrikler (Akademik)

**Return Metrikleri:**
- Total Return
- Annualized Return
- Mean Daily Return

**Risk Metrikleri:**
- Volatility (std)
- Annualized Volatility
- Maximum Drawdown
- Ulcer Index
- Value at Risk (VaR 95%)
- Conditional VaR (CVaR)

**Risk-Adjusted Returns:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio
- Recovery Factor

**Trading Metrikleri:**
- Win Rate
- Profit Factor
- Average Profit/Loss
- Total Trades

### İstatistiksel Testler

- **T-test**: Modeller arası getiri karşılaştırması
- **Wilcoxon test**: Non-parametrik alternatif
- **p-value < 0.05**: İstatistiksel anlamlılık

### API Endpoints (Akademik Analiz)

```bash
# Model karşılaştırma verilerini al
GET /api/trading/analysis/model-comparison

# En iyi modelleri metrik bazında al
GET /api/trading/analysis/best-models

# Arka planda kapsamlı rapor oluştur
POST /api/trading/analysis/generate-report
```

### LaTeX Entegrasyonu

Oluşturulan `.tex` dosyalarını doğrudan tezinize ekleyebilirsiniz:

```latex
\input{results/latex/model_comparison.tex}
```

---

## 📊 Trading Environment (Ansari et al. Metodolojisi)

### State Space (Faz 1)

```python
state = [
    balance,                 # 1 feature
    shares_owned[N],        # N features (5 hisse)
    # OHLCV (5*N features)
    open[N], high[N], low[N], close[N], volume[N],
    # Technical Indicators (5*N features)
    macd[N], rsi[N], cci[N], adx[N], turbulence[N]
]
# Total: 1 + 5 + 25 + 25 = 56 features
```

### Action Space

```python
action = [-100, ..., 0, ..., +100]  # Her hisse için
# Negative: Sell
# Positive: Buy
# Zero: Hold
```

### Reward Function (Faz 1 - Basit)

```python
reward = (current_portfolio - previous_portfolio) / previous_portfolio
```

**Faz 2'de PSR Reward:**
```python
# Ansari et al. Equation (1)
reward = ΔPortfolio + Sharpe + α * DailyReturn
```

---

## 📈 Performans Metrikleri

Dashboard ve API şu metrikleri sağlar:

- **Cumulative Return**: Toplam getiri (%)
- **Sharpe Ratio**: Risk ayarlı getiri
- **Max Drawdown**: Maksimum düşüş (%)
- **Final Portfolio Value**: Son portföy değeri (₺)
- **Total Trades**: Toplam işlem sayısı

**Hesaplama:**
```python
# Sharpe Ratio (annualized)
sharpe = (mean(returns) / std(returns)) * sqrt(252)

# Max Drawdown
max_drawdown = min((portfolio - cummax(portfolio)) / cummax(portfolio))
```

---

## 🧪 Test ve Çalıştırma

### Environment Test

```bash
python -m env.trading_env
```

### Veri Testi

```bash
python -m data.data_fetcher
```

### Tensorboard

```bash
tensorboard --logdir logs/tensorboard/
```

---

## 🎓 Ansari et al. (2024) Uygulaması

### Faz 1 - Tamamlanan

| Özellik | Ansari et al. | Bizim Uygulama | Durum |
|---------|---------------|----------------|-------|
| Multi-Stock | ✅ S&P500 | ✅ BIST-30 (5 hisse) | ✅ |
| Technical Indicators | MACD, RSI, CCI, ADX, Turbulence | ✅ Aynısı | ✅ |
| RL Algorithms | A2C, PPO, TD3 | ✅ Aynısı | ✅ |
| Environment | Custom Gym | Gymnasium | ✅ |
| Reward | PSR | Basit (portfolio change) | ⚠️ Faz 2 |
| Fundamental Data | ✅ 11 ratios | ❌ | 🔜 Faz 2 |

### Faz 2 - Tamamlandı ✅

- [x] Gelişmiş Feature Engineering (10 grup: log return, volatilite, momentum, makro, fundamental, rejim)
- [x] Multi-model ensemble (XGBoost + LightGBM + CatBoost + BiLSTM + TFT + Stacking)
- [x] Optuna HPO (walk-forward CV, purge gap 5 gün, embargo 3 gün)
- [x] Çoklu veri kaynağı (OHLCV + makro EVDS + fundamental + altın/döviz)
- [x] RL entegrasyonu (observation space'e 4×N tahmin özellikleri eklendi)
- [x] API endpoints + dashboard (train-all, optimize, ensemble-predict)

### Faz 3 - Production

- [ ] Real-time trading signals
- [ ] WebSocket updates
- [ ] Docker deployment
- [ ] Model versioning

---

## 📝 Geliştirme Notları

### Veri Kaynağı

- **BIST-30 Hisseleri**: yfinance (`.IS` suffix)
- **Periyot**: 2018-2024 (6 yıl)
- **Split**: 70% train / 15% val / 15% test

### BIST-30 Hisseleri (Faz 1)

1. AKBNK.IS - Akbank (Bankacılık)
2. THYAO.IS - Türk Hava Yolları (Havacılık)
3. TUPRS.IS - Tüpraş (Enerji)
4. BIMAS.IS - BIM (Perakende)
5. ASELS.IS - Aselsan (Savunma)

### Hyperparameters (Faz 1)

Her algoritma için optimize edilmiş learning rate'ler:

```python
# PPO (Önerilen)
learning_rate = 0.0003
n_steps = 2048
total_timesteps = 50_000+

# A2C
learning_rate = 0.0007
n_steps = 512
total_timesteps = 50_000+

# TD3
learning_rate = 0.001
buffer_size = 100_000
total_timesteps = 50_000+

# Environment
initial_balance = 1_000_000 TL
commission_rate = 0.001 (0.1%)
max_shares_per_trade = 100
```

---

## 🤝 Katkı

Bu proje akademik bir çalışmadır. Katkılarınız için:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📚 Dokümantasyon

Detaylı dokümantasyon için [docs/](docs/) klasörüne bakın:

### Kullanım Kılavuzları
- [**Algoritma Karşılaştırması**](docs/guides/ALGORITHMS.md) - PPO, A2C, TD3
- [**Akademik Analiz**](docs/guides/ACADEMIC_GUIDE.md) - Raporlama ve metrikler
- [**Hyperparameter Optimization**](docs/guides/API_HYPEROPT_GUIDE.md) - Optuna entegrasyonu
- [**GPU Performans**](docs/guides/GPU_PERFORMANCE_GUIDE.md) - GPU testleri

### Geliştirme
- [**Geliştirme Planı**](docs/development/development.md) - Roadmap ve sprint planı
- [**Hyperopt İyileştirmeleri**](docs/development/HYPEROPT_IMPROVEMENTS_SUMMARY.md)

---

## 📚 Referanslar

**Ana Makale:**
- Ansari et al. (2024) - "A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning"

**Kütüphaneler:**
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Chart.js](https://www.chartjs.org/)

---

## ⚠️ Disclaimer

Bu proje **eğitim amaçlıdır**. Gerçek para ile ticaret yapmak için:
- Risk yönetimi ekleyin
- Backtesting yapın
- Profesyonel mali danışmanlık alın

**Yatırım tavsiyesi değildir.**

---

## 📄 Lisans

MIT License - Eğitim amaçlı kullanım için serbesttir.

---

## 📞 İletişim

Sorularınız için: development.md dosyasına bakın veya issue açın.

---

**Son Güncelleme:** Faz 2 Tamamlandı ✅ (2026-03-26)
