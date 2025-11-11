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
- **Faz 2**: Multifaceted approach (Fundamental data + PSR reward)
- **Faz 3**: Production-ready sistem

### Temel Özellikler

✅ **Multi-Stock Trading Environment** (Gymnasium)
✅ **3 RL Algoritması**: A2C, PPO, TD3 (Stable-Baselines3)
✅ **5 Teknik İndikatör**: MACD, RSI, CCI, ADX, Turbulence
✅ **FastAPI Backend**: Model eğitimi ve yönetimi
✅ **Web Dashboard**: Dinamik görselleştirme (Chart.js)
✅ **Real-time Training**: Background task ile eğitim

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

### Komut Satırı ile Eğitim

```bash
# Standalone A2C eğitimi (Faz 1)
python train_a2c_phase1.py
```

**Çıktılar:**
- Model: `models/a2c_bist30_phase1.zip`
- Metrikler: `results/a2c_phase1_*.txt`
- Tensorboard: `logs/tensorboard/`

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
│   ├── index.html              # Ana sayfa (292 satır)
│   ├── css/styles.css          # Stiller (392 satır)
│   └── js/dashboard.js         # JavaScript (465 satır)
│
├── 📁 data/                     # Veri işleme
│   ├── bist30_symbols.py       # Hisse listesi
│   ├── data_fetcher.py         # Veri çekme (yfinance)
│   └── technical_indicators.py # Teknik indikatörler
│
├── 📁 env/                      # RL Environment
│   └── trading_env.py          # Gymnasium environment
│
├── 📁 models/                   # Eğitilmiş modeller (.zip)
├── 📁 results/                  # Metrikler (.json)
│
├── train_a2c_phase1.py         # Standalone eğitim
├── run_server.py               # Server launcher
├── requirements.txt            # Dependencies
├── README.md                   # Bu dosya
└── development.md              # Geliştirme planı
```

### Modül Bağımlılıkları

```
Web UI (static/)
    ↓ API calls
FastAPI Backend (app/)
    ↓ Background tasks
Training Pipeline
    ├─► data_fetcher.py → yfinance
    ├─► technical_indicators.py → pandas-ta
    ├─► trading_env.py → Gymnasium
    └─► Stable-Baselines3 → A2C/PPO/TD3
        └─► Modeller (.zip) + Metrikler (.json)
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

### Faz 2 - Gelecek

- [ ] Fundamental data (Alpha Vantage API)
- [ ] PSR Reward function
- [ ] Ablation studies (DTF vs DT vs TF)
- [ ] Full BIST-30 (30 hisse)

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

```python
# A2C
learning_rate = 0.0007
n_steps = 5
total_timesteps = 50_000

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

**Son Güncelleme:** Faz 1 Tamamlandı ✅ (2024)
