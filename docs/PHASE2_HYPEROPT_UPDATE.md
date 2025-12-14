# FAZ 2 HİPERPARAMETRE OPTİMİZASYONU GÜNCELLEMESİ

**Tarih:** 2025-12-14
**Proje:** RL Trading System - BIST-30
**Güncelleme:** Faz 2 için Hiperparametre Optimizasyonu Desteği

---

## ÖZET

Hiperparametre optimizasyon sistemi, Faz 2 özelliklerini (Fundamental + Macro data + PSR reward) destekleyecek şekilde güncellendi.

---

## DEĞİŞİKLİKLER

### 1. **base_optimizer.py** ✅

#### Güncellenen Metodlar:

**`__init__` Metodu:**
```python
def __init__(
    self,
    algorithm_name: str,
    ...
    phase: int = 2,              # YENİ
    reward_type: str = 'psr'     # YENİ
):
```

**`create_env` Metodu:**
```python
def create_env(
    self,
    stock_symbols: list,
    start_date: str,
    end_date: str,
    phase: int = 2,              # YENİ
    reward_type: str = 'psr'     # YENİ
) -> gym.Env:
    # 1. Market data çek
    df = data_fetcher.fetch_stock_data(stock_symbols)

    # 2. Faz 2 için fundamental + macro data yükle
    if phase == 2:
        fundamental_df = fund_fetcher.load_data('fundamental_data.csv')
        macro_df = macro_fetcher.load_data('macro_data.csv')

    # 3. Environment oluştur (Phase 2 destekli)
    env = TradingEnv(
        df=df,
        phase=phase,
        reward_type=reward_type,
        fundamental_df=fundamental_df,
        macro_df=macro_df
    )

    return env
```

**`objective` Metodu:**
```python
# Environment oluşturulurken phase ve reward_type parametreleri eklendi
train_env = self.create_env(..., phase=self.phase, reward_type=self.reward_type)
val_env = self.create_env(..., phase=self.phase, reward_type=self.reward_type)
```

---

### 2. **run_optimization.py** ✅

#### Yeni Command Line Arguments:

```python
parser.add_argument(
    "--phase",
    type=int,
    choices=[1, 2],
    default=2,
    help="Trading phase (1=56 features, 2=97 features with fundamental+macro, default: 2)"
)

parser.add_argument(
    "--reward-type",
    type=str,
    choices=["simple", "psr"],
    default="psr",
    help="Reward function type (simple=baseline, psr=risk-aware, default: psr)"
)
```

#### Optimizer Oluşturma:

```python
return optimizer_class(
    study_name=args.study_name,
    n_trials=args.trials,
    n_jobs=args.jobs,
    seed=args.seed,
    phase=args.phase,              # YENİ
    reward_type=args.reward_type   # YENİ
)
```

---

### 3. **app/schemas/hyperopt.py** ✅

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

---

### 4. **app/api/routes/hyperopt.py** ✅

#### Background Optimization:

```python
optimizer = OptimizerClass(
    study_name=study_info.study_name,
    storage=OPTUNA_STORAGE,
    n_trials=request.n_trials,
    n_jobs=1,
    seed=None,
    phase=request.phase,              # YENİ
    reward_type=request.reward_type,  # YENİ
)

logger.info(f"Starting optimization for study {study_id} (Phase {request.phase}, Reward: {request.reward_type.upper()})")
```

---

## KULLANIM

### 1. **Command Line (CLI)**

#### Faz 2 + PSR Reward (Önerilen):
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --phase 2 \
    --reward-type psr \
    --trials 50 \
    --timesteps 100000
```

#### Faz 1 + Simple Reward (Baseline):
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --phase 1 \
    --reward-type simple \
    --trials 50 \
    --timesteps 100000
```

#### Hızlı Test (2 trial):
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --phase 2 \
    --reward-type psr \
    --trials 2 \
    --timesteps 10000
```

---

### 2. **API (Web Interface)**

#### POST /hyperopt/start

**Request Body:**
```json
{
  "algorithm": "ppo",
  "n_trials": 50,
  "total_timesteps": 100000,
  "phase": 2,
  "reward_type": "psr",
  "train_start": "2018-01-01",
  "train_end": "2022-12-31",
  "val_start": "2023-01-01",
  "val_end": "2023-12-31",
  "eval_freq": 5000,
  "n_eval_episodes": 5
}
```

**Response:**
```json
{
  "study_id": "uuid-here",
  "study_name": "ppo_optimization_20251214",
  "message": "Optimization started for ppo",
  "estimated_duration_minutes": 100.0
}
```

---

## FAZ 1 vs FAZ 2 KARŞILAŞTIRMASI

| Özellik | Faz 1 | Faz 2 |
|---------|-------|-------|
| **State Space** | 56 features | 97 features |
| **Market Data** | OHLCV + Technical (10) | OHLCV + Technical (10) |
| **Fundamental Data** | ❌ Yok | ✅ 7 ratio × 5 hisse = 35 |
| **Macro Data** | ❌ Yok | ✅ 6 indicators |
| **Reward Function** | Simple (baseline) | PSR (risk-aware) |
| **Learning Rate** | ~1e-5 | ~3e-4 (daha yüksek) |
| **Network Size** | Small | Medium/Large |
| **Batch Size** | 128 | 256-512 |
| **Entropy Coef** | ~1e-5 | ~0.01 (daha yüksek) |

---

## BEKLENEN SONUÇLAR

### Faz 2 Optimizasyonu Sonrası:

1. **Network Kapasitesi:** Medium veya Large network (97 feature için yeterli)
2. **Learning Rate:** 10-50x daha yüksek (3e-4 to 1e-3 arası)
3. **Batch Size:** 256-512 (gradient variance azaltmak için)
4. **Exploration:** 1000x daha yüksek entropy coefficient (0.001-0.01)
5. **Performance:** Daha iyi Sharpe Ratio, daha düşük MDD

---

## TEST SENARYOLARI

### Senaryo 1: Hızlı Test (5 dakika)
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --phase 2 \
    --reward-type psr \
    --trials 2 \
    --timesteps 10000 \
    --train-start 2023-01-01 \
    --train-end 2023-06-30 \
    --val-start 2023-07-01 \
    --val-end 2023-12-31
```

### Senaryo 2: Production Optimizasyonu (10-12 saat)
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --phase 2 \
    --reward-type psr \
    --trials 50 \
    --timesteps 100000 \
    --train-start 2018-01-01 \
    --train-end 2022-12-31 \
    --val-start 2023-01-01 \
    --val-end 2023-12-31
```

### Senaryo 3: A/B Testing (Faz 1 vs Faz 2)
```bash
# Faz 1 optimizasyonu
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo --phase 1 --reward-type simple --trials 30

# Faz 2 optimizasyonu
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo --phase 2 --reward-type psr --trials 30

# Sonuçları karşılaştır
python hyperparameter_optimization/analyze_results.py
```

---

## ÇIKTILAR

### Dosyalar:
- `results/hyperparameter_studies/best_params_ppo_*.json` - En iyi parametreler
- `results/hyperparameter_studies/optuna_studies.db` - SQLite database
- `logs/hyperopt_ppo_phase2/` - Training logs

### Optuna Dashboard:
```bash
optuna-dashboard sqlite:///results/hyperparameter_studies/optuna_studies.db
```

---

## NOTLAR

1. **Faz 2 Veri Gereksinimleri:**
   - `data/fundamental_data.csv` mevcut olmalı
   - `data/macro_data.csv` mevcut olmalı
   - Yoksa otomatik olarak çekilir

2. **Varsayılan Değerler:**
   - `--phase 2` (Faz 2 varsayılan)
   - `--reward-type psr` (PSR reward varsayılan)
   - İlk deneme Faz 2 ile yapılmalı

3. **GPU Kullanımı:**
   - `--jobs 1` önerili (GPU memory için)
   - Paralel çalıştırma PyTorch tarafından yapılır

4. **Optimizasyon Süresi:**
   - Trial başına ~2-5 dakika
   - 50 trial ≈ 2-4 saat (hızlı PC)
   - 50 trial ≈ 6-12 saat (normal PC)

---

## SONRAKI ADIMLAR

1. ✅ Kod güncellemeleri tamamlandı
2. ⏳ Hızlı test çalıştır (2 trial)
3. ⏳ Sonuçları doğrula
4. ⏳ Production optimizasyonu başlat (50 trial)
5. ⏳ En iyi hiperparametreleri kullanarak model train et

---

**Güncelleme Durumu:** ✅ TAMAMLANDI
**Test Durumu:** ⏳ BEKLİYOR
