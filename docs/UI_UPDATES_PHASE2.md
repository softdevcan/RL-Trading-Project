# WEB ARAYÜZÜ FAZ 2 GÜNCELLEMESİ

**Tarih:** 2025-12-14
**Proje:** RL Trading System - BIST-30
**Güncelleme:** Hiperparametre Optimizasyonu Web Arayüzü - Faz 2 Desteği

---

## ÖZET

Web arayüzündeki hiperparametre optimizasyonu sayfası, Faz 2 özelliklerini (Phase selection ve Reward Type) destekleyecek şekilde güncellendi.

---

## GÜNCELLENEN DOSYALAR

### 1. **static/hyperopt.html** ✅

#### Yeni Form Alanları:

**Phase Seçimi:**
```html
<div class="form-group">
    <label for="phase">Phase *</label>
    <select id="phase" required>
        <option value="2" selected>Phase 2 (97 features - Fundamental + Macro)</option>
        <option value="1">Phase 1 (56 features - Baseline)</option>
    </select>
</div>
```

**Reward Function Seçimi:**
```html
<div class="form-group">
    <label for="reward_type">Reward Function *</label>
    <select id="reward_type" required>
        <option value="psr" selected>PSR (Risk-Aware)</option>
        <option value="simple">Simple (Baseline)</option>
    </select>
</div>
```

#### JavaScript Request Body:
```javascript
const data = {
    algorithm: document.getElementById('algorithm').value,
    phase: parseInt(document.getElementById('phase').value),          // YENİ
    reward_type: document.getElementById('reward_type').value,        // YENİ
    n_trials: parseInt(document.getElementById('n_trials').value),
    total_timesteps: parseInt(document.getElementById('timesteps').value),
    train_start: document.getElementById('train_start').value,
    train_end: document.getElementById('train_end').value,
    val_start: document.getElementById('val_start').value,
    val_end: document.getElementById('val_end').value,
};
```

#### Study Card Display:
```javascript
<div class="study-info">
    <strong>Phase:</strong> ${study.phase || 2} |
    <strong>Reward:</strong> ${(study.reward_type || 'psr').toUpperCase()}
</div>
```

#### Study Detail Modal:
```javascript
<div class="study-info">
    <strong>Phase:</strong> ${study.phase || 2}
    (${study.phase === 1 ? '56 features' : '97 features - Fundamental + Macro'})
</div>
<div class="study-info">
    <strong>Reward Function:</strong> ${(study.reward_type || 'psr').toUpperCase()}
    (${study.reward_type === 'simple' ? 'Baseline' : 'Risk-Aware'})
</div>
```

---

### 2. **app/schemas/hyperopt.py** ✅

#### OptimizationRequest Schema:
```python
class OptimizationRequest(BaseModel):
    ...

    phase: int = Field(
        2,
        ge=1,
        le=2,
        description="Trading phase (1=56 features, 2=97 features with fundamental+macro)"
    )

    reward_type: str = Field(
        "psr",
        description="Reward function type (simple or psr)"
    )
```

#### StudyInfo Schema:
```python
class StudyInfo(BaseModel):
    ...

    # Configuration
    n_trials: int
    total_timesteps: int
    stock_symbols: List[str]
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    phase: int = 2              # YENİ
    reward_type: str = "psr"    # YENİ
```

---

### 3. **app/api/routes/hyperopt.py** ✅

#### create_study_info_from_request:
```python
return StudyInfo(
    study_id=study_id,
    study_name=request.study_name or f"{request.algorithm.value}_optimization_{study_id[:8]}",
    algorithm=request.algorithm,
    status=StudyStatus.PENDING,
    n_trials=request.n_trials,
    total_timesteps=request.total_timesteps,
    stock_symbols=stock_symbols,
    train_start=request.train_start,
    train_end=request.train_end,
    val_start=request.val_start,
    val_end=request.val_end,
    phase=request.phase,              # YENİ
    reward_type=request.reward_type,  # YENİ
    created_at=datetime.now(),
)
```

---

## KULLANIM

### Web Arayüzünden Optimizasyon Başlatma:

1. **Tarayıcıda aç:**
   ```
   http://localhost:8000/static/hyperopt.html
   ```

2. **Formu doldur:**
   - **Algoritma:** PPO (önerilen)
   - **Phase:** Phase 2 (97 features - Fundamental + Macro)
   - **Reward Function:** PSR (Risk-Aware)
   - **Trial Sayısı:** 50
   - **Timesteps:** 100,000
   - **Training Period:** 2018-01-01 to 2022-12-31
   - **Validation Period:** 2023-01-01 to 2023-12-31

3. **"Optimizasyonu Başlat" butonuna tıkla**

4. **Canlı izle:**
   - WebSocket bağlantısı otomatik kurulur
   - Progress bar real-time güncellenir
   - Her trial tamamlandığında bildirim gelir

---

## EKRAN GÖRÜNÜMLERİ

### Optimizasyon Formu:
```
┌─────────────────────────────────────────────────────────┐
│ Yeni Optimizasyon Başlat                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ [Algoritma: PPO ▼] [Phase: Phase 2 ▼] [Reward: PSR ▼] │
│                                                          │
│ [Trial: 50] [Timesteps: 100000]                        │
│                                                          │
│ [Train Start: 2018-01-01] [Train End: 2022-12-31]      │
│ [Val Start: 2023-01-01]   [Val End: 2023-12-31]        │
│                                                          │
│         [🚀 Optimizasyonu Başlat]                       │
└─────────────────────────────────────────────────────────┘
```

### Study Card:
```
┌─────────────────────────────────────────────────────────┐
│ PPO                                      [Çalışıyor] 🟢 │
├─────────────────────────────────────────────────────────┤
│ Study: ppo_optimization_a1b2c3d4                        │
│ Hisseler: AKBNK.IS, THYAO.IS, TUPRS.IS, ...           │
│ Phase: 2 | Reward: PSR                                 │
│                                                          │
│ [████████████████░░░░░░░░░░] 65%                       │
│ İlerleme: 32 / 50 trials (65%)                         │
│                                                          │
│ ┌──────────────┐ ┌──────────────┐                      │
│ │ En İyi Sharpe│ │ Tamamlanan   │                      │
│ │    2.3451    │ │      32      │                      │
│ └──────────────┘ └──────────────┘                      │
│                                                          │
│ Başlatıldı: 14.12.2024 15:30                           │
└─────────────────────────────────────────────────────────┘
```

### Study Detail Modal:
```
┌─────────────────────────────────────────────────────────┐
│ Study Detayları                                     [×] │
├─────────────────────────────────────────────────────────┤
│ Algoritma: PPO                                          │
│ Phase: 2 (97 features - Fundamental + Macro)           │
│ Reward Function: PSR (Risk-Aware)                      │
│ Durum: completed                                        │
│ Trial Sayısı: 50                                        │
│ Tamamlanan: 50                                          │
│                                                          │
│ En İyi Sonuç                                            │
│ ─────────────                                           │
│ Sharpe Ratio: 2.3451                                   │
│ Trial: 23                                               │
│                                                          │
│ En İyi Parametreler                                     │
│ ─────────────────                                       │
│ {                                                        │
│   "learning_rate": 0.0003,                             │
│   "n_steps": 2048,                                     │
│   "batch_size": 256,                                   │
│   "net_arch_size": "medium",                           │
│   ...                                                   │
│ }                                                        │
│                                                          │
│ Trial Geçmişi (Son 10)                                 │
│ ────────────────────────                                │
│ # │ Durum    │ Sharpe  │ Süre                          │
│ ───┼──────────┼─────────┼──────                         │
│ 50│ complete │ 2.1234  │ 127s                          │
│ 49│ complete │ 2.0145  │ 134s                          │
│ ...                                                     │
└─────────────────────────────────────────────────────────┘
```

---

## VARSAYILAN DEĞERLER

Arayüzde varsayılan olarak:
- **Phase:** 2 (Faz 2 - 97 features)
- **Reward Type:** PSR (Risk-aware)

Bu değerler en iyi performans için optimize edilmiştir.

---

## TEST SENARYOLARI

### Senaryo 1: Hızlı Test (Web UI)
1. Web arayüzünü aç: `http://localhost:8000/static/hyperopt.html`
2. Formu doldur:
   - Algorithm: PPO
   - Phase: 2
   - Reward: PSR
   - Trials: 2
   - Timesteps: 10,000
3. "Optimizasyonu Başlat" butonuna tıkla
4. WebSocket üzerinden canlı izle
5. Tamamlandığında sonuçları görüntüle

### Senaryo 2: A/B Testing (Faz 1 vs Faz 2)
1. **İlk Optimizasyon:**
   - Phase: 1
   - Reward: Simple
   - Trials: 30

2. **İkinci Optimizasyon:**
   - Phase: 2
   - Reward: PSR
   - Trials: 30

3. **Sonuçları Karşılaştır:**
   - Her iki study'nin best_value'larını karşılaştır
   - Trial history'leri incele

### Senaryo 3: Production Optimizasyon
1. **Form Ayarları:**
   - Algorithm: PPO
   - Phase: 2
   - Reward: PSR
   - Trials: 50
   - Timesteps: 100,000
   - Train: 2018-01-01 to 2022-12-31
   - Val: 2023-01-01 to 2023-12-31

2. **Başlat ve İzle:**
   - WebSocket canlı bağlantı
   - Her trial'ı takip et
   - 10-12 saat bekle

3. **Sonuçları Al:**
   - Best parameters'ı kaydet
   - JSON dosyasından detayları indir

---

## BACKEND ENTEGRASYONU

Web arayüzü şu backend endpoint'leri kullanır:

### POST /api/hyperopt/start
```json
{
  "algorithm": "ppo",
  "phase": 2,
  "reward_type": "psr",
  "n_trials": 50,
  "total_timesteps": 100000,
  "train_start": "2018-01-01",
  "train_end": "2022-12-31",
  "val_start": "2023-01-01",
  "val_end": "2023-12-31"
}
```

### GET /api/hyperopt/studies
Response içerir: `phase`, `reward_type`

### GET /api/hyperopt/studies/{study_id}
Response içerir: `phase`, `reward_type` detayları

### WebSocket /api/hyperopt/ws/{study_id}
Real-time progress updates

---

## NOTLAR

1. **Varsayılan Değerler:** Phase 2 ve PSR reward varsayılan olarak seçilidir
2. **Backward Compatibility:** Eski çalışmalar için phase ve reward_type default değerlerle gösterilir
3. **WebSocket:** Real-time progress tracking otomatik
4. **Responsive:** Mobil cihazlarda da çalışır

---

## SONRAKI ADIMLAR

1. ✅ Web arayüzü güncellendi
2. ✅ Backend entegrasyonu tamamlandı
3. ⏳ Kullanıcı testi yapılacak
4. ⏳ Production'da test edilecek

---

**Güncelleme Durumu:** ✅ TAMAMLANDI
**Test Durumu:** ⏳ BEKLİYOR
