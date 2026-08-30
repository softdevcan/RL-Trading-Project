# 🔬 Hiper Parametre Optimizasyonu - API & UI Kılavuzu

## 🎉 Özellikler

✅ **Web Tabanlı UI** - Modern, responsive web arayüzü
✅ **RESTful API** - FastAPI ile güçlü API
✅ **Real-time Updates** - WebSocket ile canlı progress tracking
✅ **Background Processing** - Asenkron optimizasyon
✅ **Study Management** - Tüm optimizasyonları takip edin
✅ **Interactive Dashboard** - Sonuçları görsel olarak inceleyin

---

## 🚀 Hızlı Başlangıç

### 1. Sunucuyu Başlatın

```bash
python run_server.py
```

Sunucu başladığında:
- **UI:** http://localhost:8000/static/hyperopt.html
- **API Docs:** http://localhost:8000/docs
- **Main Dashboard:** http://localhost:8000/

### 2. Web Arayüzünden Kullanım

1. http://localhost:8000/static/hyperopt.html adresine gidin
2. Formu doldurun:
   - Algoritma seçin (PPO, A2C, TD3, SAC)
   - Trial sayısını ayarlayın (1-200)
   - Timesteps belirleyin
   - Tarih aralığını seçin
3. "Optimizasyonu Başlat" butonuna tıklayın
4. Real-time progress takibi yapın
5. Sonuçları inceleyin

---

## 📡 API Endpoints

### 1. Optimizasyon Başlat

```http
POST /api/hyperopt/start
Content-Type: application/json

{
  "algorithm": "ppo",
  "n_trials": 50,
  "total_timesteps": 100000,
  "train_start": "2018-01-01",
  "train_end": "2022-12-31",
  "val_start": "2023-01-01",
  "val_end": "2023-12-31",
  "seed": 42
}
```

**Response:**
```json
{
  "study_id": "550e8400-e29b-41d4-a716-446655440000",
  "study_name": "ppo_optimization_550e8400",
  "message": "Optimization started for ppo",
  "estimated_duration_minutes": 100.0
}
```

### 2. Studies Listele

```http
GET /api/hyperopt/studies?algorithm=ppo&status=running&limit=50&offset=0
```

**Response:**
```json
{
  "studies": [
    {
      "study_id": "550e8400-e29b-41d4-a716-446655440000",
      "study_name": "ppo_optimization_550e8400",
      "algorithm": "ppo",
      "status": "running",
      "n_trials": 50,
      "trials_completed": 25,
      "progress_percentage": 50.0,
      "best_value": 1.8734,
      "best_params": {...},
      "created_at": "2025-01-14T10:00:00",
      ...
    }
  ],
  "total": 1
}
```

### 3. Study Detayı

```http
GET /api/hyperopt/studies/{study_id}
```

**Response:**
```json
{
  "study": {
    "study_id": "550e8400-e29b-41d4-a716-446655440000",
    "study_name": "ppo_optimization_550e8400",
    "status": "completed",
    "best_value": 1.8734,
    "best_params": {
      "learning_rate": 0.000342,
      "n_steps": 2048,
      "batch_size": 64,
      ...
    },
    ...
  },
  "trials": [
    {
      "trial_number": 0,
      "state": "complete",
      "value": 1.5234,
      "params": {...},
      "duration_seconds": 120.5,
      ...
    }
  ],
  "mean_value": 1.6543,
  "median_value": 1.6234,
  "std_value": 0.1234
}
```

### 4. Progress Takibi

```http
GET /api/hyperopt/studies/{study_id}/progress
```

**Response:**
```json
{
  "study_id": "550e8400-e29b-41d4-a716-446655440000",
  "study_name": "ppo_optimization_550e8400",
  "status": "running",
  "progress_percentage": 50.0,
  "trials_completed": 25,
  "trials_total": 50,
  "current_best_value": 1.8734,
  "estimated_time_remaining_minutes": 50.0,
  "last_trial": {...}
}
```

### 5. Optimizasyonu İptal Et

```http
POST /api/hyperopt/studies/{study_id}/cancel
```

> ⚠️ **Değişti.** Bu iş eskiden `DELETE /api/hyperopt/studies/{study_id}` idi.
> `DELETE` artık **kaydı kalıcı olarak siler** (aşağıda, madde 6). İkisi durum
> bakımından ayrık: çalışan bir study'ye `DELETE` atarsanız 409 alırsınız ve
> buraya yönlendirilirsiniz — veri kaybı olmaz.

### 6. Optimizasyon Kaydını Sil

```http
DELETE /api/hyperopt/studies/{study_id}
```

Kaydı Optuna deposundan **kalıcı olarak** siler (deneme geçmişi ve en iyi
parametreler dahil). Çalışan bir koşum `409` ile reddedilir — önce iptal edin.

`user` veya `admin` rolü gerekir (`viewer` silemez). Panoda: HiperParam
sayfasındaki çalışma kartında çöp kutusu düğmesi.

> **İzolasyon notu:** Optuna deposu depo köküne sabit bağlıdır
> (`results/hyperparameter_studies/optuna_studies.db`), `app/auth/workspace.py`
> ile çözülmez. Yani çalışmalar **tüm kullanıcılar arasında ortaktır**: burada
> silinen kayıt herkesten silinir. Liste de zaten herkese aynı çalışmaları
> gösteriyor.

**Response:**
```json
{
  "study_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Optimization cancelled",
  "trials_completed": 25
}
```

### 6. Arama Uzayı

```http
GET /api/hyperopt/search-spaces/{algorithm}
```

**Response:**
```json
{
  "algorithm": "ppo",
  "parameters": [
    {
      "parameter_name": "learning_rate",
      "param_type": "loguniform",
      "default": 0.0003,
      "description": "Adam optimizer learning rate",
      "low": 1e-5,
      "high": 1e-2
    },
    ...
  ],
  "total_parameters": 11
}
```

### 7. WebSocket (Real-time Updates)

```javascript
const ws = new WebSocket('ws://localhost:8000/api/hyperopt/ws/{study_id}');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.message_type === 'trial_complete') {
        console.log(`Trial ${data.trial_number} completed`);
        console.log(`Value: ${data.trial_value}`);
        console.log(`Progress: ${data.progress_percentage}%`);
    }

    if (data.message_type === 'optimization_complete') {
        console.log(`Best value: ${data.best_value}`);
        console.log(`Best params:`, data.best_params);
    }
};
```

---

## 🐍 Python Client Örneği

```python
import requests
import time

API_BASE = "http://localhost:8000/api/hyperopt"

# 1. Optimizasyon başlat
response = requests.post(f"{API_BASE}/start", json={
    "algorithm": "ppo",
    "n_trials": 50,
    "total_timesteps": 100_000,
    "train_start": "2018-01-01",
    "train_end": "2022-12-31",
    "val_start": "2023-01-01",
    "val_end": "2023-12-31",
})

result = response.json()
study_id = result["study_id"]
print(f"Study started: {study_id}")

# 2. Progress takibi
while True:
    response = requests.get(f"{API_BASE}/studies/{study_id}/progress")
    progress = response.json()

    print(f"Progress: {progress['progress_percentage']:.1f}% - "
          f"{progress['trials_completed']}/{progress['trials_total']} trials")

    if progress['status'] in ['completed', 'failed', 'cancelled']:
        break

    time.sleep(10)  # 10 saniyede bir kontrol et

# 3. Sonuçları al
response = requests.get(f"{API_BASE}/studies/{study_id}")
details = response.json()

print(f"\nOptimization completed!")
print(f"Best Sharpe Ratio: {details['study']['best_value']:.4f}")
print(f"Best parameters:")
for key, value in details['study']['best_params'].items():
    print(f"  {key}: {value}")
```

---

## 📊 Web UI Özellikleri

### Ana Sayfa (hyperopt.html)

1. **Optimizasyon Formu**
   - Algoritma seçimi
   - Trial sayısı ayarı
   - Timesteps ayarı
   - Tarih aralığı seçimi
   - Random seed ayarı

2. **Aktif Optimizasyonlar**
   - Tüm studies'lerin kartları
   - Real-time progress bar'lar
   - Durum göstergeleri (pending, running, completed, failed)
   - En iyi Sharpe ratio
   - Tamamlanan trial sayısı

3. **Study Detay Modal**
   - Study bilgileri
   - En iyi sonuç ve parametreler
   - Trial geçmişi tablosu
   - İstatistikler

4. **WebSocket Status**
   - Bağlantı durumu göstergesi
   - Real-time güncellemeler

### Dashboard Entegrasyonu

Ana dashboard'dan (http://localhost:8000/) "🔬 Hiper Parametre" butonuna tıklayarak hiper parametre sayfasına gidebilirsiniz.

---

## 🔧 Teknik Detaylar

### Dosya Yapısı

```
app/
├── api/routes/
│   └── hyperopt.py           # API endpoints
├── schemas/
│   └── hyperopt.py           # Pydantic schemas
└── main.py                   # FastAPI app (router eklendi)

static/
├── hyperopt.html             # Web UI
└── index.html                # Main dashboard (link eklendi)

hyperparameter_optimization/
├── optimizers/               # Optimizer sınıfları
├── search_spaces.py          # Parametre tanımları
└── base_optimizer.py         # Base optimizer

results/hyperparameter_studies/
├── optuna_studies.db         # Optuna SQLite database
├── plots/                    # Görselleştirmeler
└── reports/                  # Raporlar
```

### Teknoloji Stack'i

- **Backend:** FastAPI + Python
- **Database:** Optuna SQLite
- **WebSocket:** FastAPI WebSockets
- **Background Tasks:** FastAPI BackgroundTasks
- **Frontend:** Vanilla JavaScript + Modern CSS
- **Optimization:** Optuna 4.6.0
- **RL:** Stable-Baselines3

---

## 💡 Kullanım Senaryoları

### Senaryo 1: Hızlı Test

```python
# Web UI veya API ile
{
    "algorithm": "ppo",
    "n_trials": 2,
    "total_timesteps": 10_000
}
```

**Süre:** ~5 dakika

### Senaryo 2: Production Optimizasyon

```python
{
    "algorithm": "ppo",
    "n_trials": 50,
    "total_timesteps": 100_000
}
```

**Süre:** ~6-12 saat

### Senaryo 3: Tüm Algoritmaları Karşılaştırma

Her algoritma için ayrı ayrı optimize edin:
1. PPO optimize et (50 trials)
2. A2C optimize et (50 trials)
3. TD3 optimize et (50 trials)
4. SAC optimize et (50 trials)

Web UI'den hepsini görebilir ve karşılaştırabilirsiniz.

---

## 🎓 Akademik Kullanım

### Sonuçları Alma

```python
import requests
import json

# En iyi parametreleri al
response = requests.get(f"{API_BASE}/studies/{study_id}")
study = response.json()

# JSON'a kaydet
with open(f'best_params_{algorithm}.json', 'w') as f:
    json.dump(study['study']['best_params'], f, indent=2)

# LaTeX tablosu için CLI kullan
# python hyperparameter_optimization/analyze_results.py \
#     --study-name {study_name} --latex
```

### Görselleştirmeler

1. **Web UI'den:** Study kartına tıklayın
2. **Optuna Dashboard:** `optuna-dashboard sqlite:///results/hyperparameter_studies/optuna_studies.db`
3. **CLI Tool:** `python hyperparameter_optimization/analyze_results.py --study-name {study_name}`

---

## ❓ Sık Sorulan Sorular

### Q: Birden fazla optimizasyon aynı anda çalıştırabilir miyim?

**A:** Evet! Her optimizasyon background'da çalışır. Ancak GPU memory dikkat edin.

### Q: WebSocket bağlantısı koptu, nasıl tekrar bağlanırım?

**A:** Sayfa otomatik olarak 5 saniye sonra tekrar bağlanır. Manuel refresh de yapabilirsiniz.

### Q: Study'leri nasıl filtreleyebilirim?

**A:** API'de `algorithm` ve `status` parametrelerini kullanın:
```
GET /api/hyperopt/studies?algorithm=ppo&status=completed
```

### Q: Optimizasyonu nasıl durdururum?

**A:**
```http
POST /api/hyperopt/studies/{study_id}/cancel
```

### Q: Biten bir çalışmayı listeden nasıl kaldırırım?

**A:** Kalıcı olarak silin — tamamlanan çalışmalar Optuna deposunda kalır ve
liste onları her açılışta geri getirir:
```http
DELETE /api/hyperopt/studies/{study_id}
```

### Q: Best parameters'ı nasıl kullanırım?

**A:** Study detail'den alın ve normal training'de kullanın:
```python
best_params = study['study']['best_params']
model = PPO(policy="MlpPolicy", env=env, **best_params)
```

---

## 🔒 Güvenlik Notları

- API şu an authentication yok (local development için)
- Production'da API key veya OAuth ekleyin
- CORS ayarları `app/core/config.py`'de
- WebSocket'ler study_id ile izole edilmiş

---

## 📚 Daha Fazla Bilgi

- [Hyperparameter Optimization README](hyperparameter_optimization/README.md)
- [API Documentation](http://localhost:8000/docs)
- [Optuna Documentation](https://optuna.readthedocs.io/)

---

**Başarılar! Artık web arayüzünden hiper parametre optimizasyonu yapabilirsiniz!** 🚀
