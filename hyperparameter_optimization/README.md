# 🔬 Hyperparameter Optimization System

Akademik seviyede hiper parametre optimizasyonu sistemi. Optuna tabanlı Bayesian optimization ile 4 farklı RL algoritmasını (PPO, A2C, TD3, SAC) optimize eder.

## 📋 İçindekiler

- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Detaylı Kullanım](#detaylı-kullanım)
- [Parametre Uzayları](#parametre-uzayları)
- [Sonuç Analizi](#sonuç-analizi)
- [Akademik Kullanım](#akademik-kullanım)
- [Sistem Mimarisi](#sistem-mimarisi)

---

## 🚀 Kurulum

### 1. Gerekli Paketleri Yükleyin

```bash
pip install -r requirements.txt
```

Bu şunları yükler:
- `optuna==4.2.0` - Bayesian optimization framework
- `optuna-dashboard==0.19.0` - Web-based visualization dashboard
- `plotly==5.28.0` - Interactive plotting
- `kaleido==0.2.1` - Static image export

### 2. Dizin Yapısını Kontrol Edin

```
hyperparameter_optimization/
├── search_spaces.py          # Parametre arama uzayları (buradan düzenleyin!)
├── base_optimizer.py         # Base optimizer sınıfı
├── optimizers/
│   ├── ppo_optimizer.py      # PPO optimizer
│   ├── a2c_optimizer.py      # A2C optimizer
│   ├── td3_optimizer.py      # TD3 optimizer
│   └── sac_optimizer.py      # SAC optimizer
├── run_optimization.py       # ANA SCRIPT - optimizasyonu çalıştırır
├── analyze_results.py        # Sonuç analizi ve görselleştirme
└── README.md                 # Bu dosya
```

---

## ⚡ Hızlı Başlangıç

### Örnek 1: PPO'yu 50 Trial ile Optimize Etme

```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --trials 50 \
    --timesteps 100000
```

### Örnek 2: Test Çalıştırması (Hızlı)

```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --trials 2 \
    --timesteps 10000
```

### Örnek 3: Tüm Algoritmaları Optimize Etme

```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm all \
    --trials 30
```

### Örnek 4: Sonuçları Analiz Etme

```bash
# Study'leri listele
python hyperparameter_optimization/analyze_results.py --list

# Belirli bir study'yi analiz et
python hyperparameter_optimization/analyze_results.py \
    --study-name ppo_optimization_20240115_143022

# Tüm studies'i karşılaştır
python hyperparameter_optimization/analyze_results.py --compare-all
```

---

## 📚 Detaylı Kullanım

### A. Parametreleri Özelleştirme

[search_spaces.py](search_spaces.py) dosyasını açın ve parametre aralıklarını düzenleyin:

```python
# Örnek: PPO learning rate aralığını değiştirme
PPO_SEARCH_SPACE = {
    "learning_rate": {
        "type": "loguniform",
        "low": 1e-5,        # ← Burası minimum değer
        "high": 1e-2,       # ← Burası maksimum değer
        "default": 3e-4,
        "description": "Adam optimizer learning rate"
    },
    # ... diğer parametreler
}
```

**Parametre Tipleri:**

1. **`loguniform`** - Logaritmik ölçekte uniform dağılım (learning rate için ideal)
   ```python
   "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2}
   ```

2. **`uniform`** - Linear ölçekte uniform dağılım (gamma, clip_range için ideal)
   ```python
   "gamma": {"type": "uniform", "low": 0.95, "high": 0.999}
   ```

3. **`int`** - Integer değerler (n_epochs için ideal)
   ```python
   "n_epochs": {"type": "int", "low": 5, "high": 20}
   ```

4. **`categorical`** - Belirli seçenekler (batch_size için ideal)
   ```python
   "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]}
   ```

### B. Optimization Çalıştırma

#### Komut Satırı Argümanları

```bash
python hyperparameter_optimization/run_optimization.py [OPTIONS]
```

**Ana Parametreler:**

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--algorithm` | Algoritma seçimi (`ppo`, `a2c`, `td3`, `sac`, `all`) | `ppo` |
| `--trials` | Toplam trial sayısı | `50` |
| `--jobs` | Paralel çalıştırma sayısı (GPU memory dikkat!) | `1` |
| `--timesteps` | Her trial için training timesteps | `100000` |
| `--eval-freq` | Evaluation frequency (pruning için) | `5000` |
| `--n-eval-episodes` | Evaluation episode sayısı | `5` |

**Veri Parametreleri:**

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--stocks` | Hisse senedi sembolleri | `PHASE1_SYMBOLS` (AKBNK, THYAO, TUPRS, BIMAS, ASELS) |
| `--train-start` | Training başlangıç tarihi | `2018-01-01` |
| `--train-end` | Training bitiş tarihi | `2022-12-31` |
| `--val-start` | Validation başlangıç tarihi | `2023-01-01` |
| `--val-end` | Validation bitiş tarihi | `2023-12-31` |

**Diğer Parametreler:**

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--study-name` | Study ismi (None ise otomatik) | `None` |
| `--seed` | Random seed | `42` |
| `--no-progress-bar` | Progress bar gösterme | `False` |

#### Örnek Kullanımlar

**1. Kısa Test Çalıştırması (5 dakika):**
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --trials 2 \
    --timesteps 10000
# Default olarak PHASE1_SYMBOLS kullanılır (AKBNK, THYAO, TUPRS, BIMAS, ASELS)
```

**2. Orta Ölçekli Optimizasyon (2-3 saat):**
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --trials 30 \
    --timesteps 50000
```

**3. Tam Ölçekli Akademik Çalışma (6-12 saat):**
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --trials 100 \
    --timesteps 200000 \
    --eval-freq 10000
```

**4. Tüm Algoritmaları Karşılaştırma (1-2 gün):**
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm all \
    --trials 50 \
    --timesteps 100000
```

**5. Özel Tarih ve Hisse Senedi ile:**
```bash
python hyperparameter_optimization/run_optimization.py \
    --algorithm sac \
    --trials 40 \
    --stocks AAPL MSFT GOOGL TSLA \
    --train-start 2019-01-01 \
    --train-end 2021-12-31 \
    --val-start 2022-01-01 \
    --val-end 2022-12-31
```

### C. Optuna Dashboard Kullanımı

Optuna web tabanlı dashboard ile sonuçları interaktif olarak inceleyin:

```bash
# Dashboard'u başlat
optuna-dashboard sqlite:///results/hyperparameter_studies/optuna_studies.db
```

Tarayıcınızda açın: `http://localhost:8080`

**Dashboard Özellikleri:**
- 📊 Real-time optimization progress
- 📈 Interactive parameter importance plots
- 🔍 Trial history ve state tracking
- 🎯 Best trial details
- 📉 Parallel coordinate plots

---

## 🎯 Parametre Uzayları

### PPO (Proximal Policy Optimization)

```python
PPO_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-2],        # EN ÖNEMLİ
    "n_steps": [512, 1024, 2048, 4096],
    "batch_size": [32, 64, 128, 256],
    "n_epochs": [5, 20],
    "gamma": [0.95, 0.999],
    "gae_lambda": [0.8, 0.99],
    "clip_range": [0.1, 0.4],
    "ent_coef": [1e-8, 0.3],              # Exploration
    "vf_coef": [0.1, 1.0],
    "max_grad_norm": [0.3, 5.0],
    "net_arch_size": ["small", "medium", "large"]
}
```

**Network Architectures:**
- `small`: [64, 64] - Hızlı training, basit problemler
- `medium`: [256, 256] - **Önerilen** - Dengeli
- `large`: [400, 300] - Kompleks problemler, yavaş

### A2C (Advantage Actor-Critic)

```python
A2C_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-2],
    "n_steps": [8, 16, 32, 64, 128, 256],  # A2C'de küçük değerler
    "gamma": [0.95, 0.999],
    "gae_lambda": [0.8, 0.99],
    "ent_coef": [1e-8, 0.1],               # PPO'dan daha düşük
    "vf_coef": [0.1, 1.0],
    "max_grad_norm": [0.3, 5.0],
    "rms_prop_eps": [1e-8, 1e-4],          # RMSprop stabilite
    "net_arch_size": ["small", "medium", "large"]
}
```

### TD3 (Twin Delayed DDPG)

```python
TD3_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-2],
    "buffer_size": [50000, 100000, 200000, 500000],
    "learning_starts": [1000, 10000],
    "batch_size": [64, 128, 256, 512],
    "tau": [0.001, 0.02],                  # Soft update coefficient
    "gamma": [0.95, 0.999],
    "action_noise_sigma": [0.05, 0.5],     # Exploration noise
    "target_policy_noise": [0.1, 0.5],     # TD3 smoothing
    "target_noise_clip": [0.3, 1.0],
    "policy_delay": [1, 4],                # Delayed policy updates
    "net_arch_size": ["small", "medium", "large"]
}
```

### SAC (Soft Actor-Critic)

```python
SAC_SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-2],
    "buffer_size": [50000, 100000, 200000, 500000],
    "learning_starts": [1000, 10000],
    "batch_size": [64, 128, 256, 512],
    "tau": [0.001, 0.02],
    "gamma": [0.95, 0.999],
    "ent_coef": ["auto", "auto_0.1", "auto_0.5", "auto_1.0"],  # Automatic tuning
    "target_update_interval": [1, 10],
    "train_freq": [1, 10],
    "gradient_steps": [1, 10],
    "net_arch_size": ["small", "medium", "large"]
}
```

---

## 📊 Sonuç Analizi

### A. Temel Analiz

```bash
# Tüm studies'i listele
python hyperparameter_optimization/analyze_results.py --list
```

**Çıktı:**
```
📚 Found 4 studies:
   1. ppo_optimization_20240115_143022
   2. a2c_optimization_20240115_150133
   3. td3_optimization_20240115_153244
   4. sac_optimization_20240115_160355
```

### B. Tam Analiz Raporu

```bash
python hyperparameter_optimization/analyze_results.py \
    --study-name ppo_optimization_20240115_143022
```

**Oluşturulacak Dosyalar:**

```
results/hyperparameter_studies/
├── plots/
│   ├── ppo_optimization_20240115_143022_optimization_history.html
│   ├── ppo_optimization_20240115_143022_param_importances.html
│   ├── ppo_optimization_20240115_143022_parallel_coordinate.html
│   ├── ppo_optimization_20240115_143022_contour.html
│   └── ppo_optimization_20240115_143022_slice.html
└── reports/
    ├── ppo_optimization_20240115_143022_statistics.json
    └── ppo_optimization_20240115_143022_latex_table.tex
```

**Görselleştirmeler:**

1. **Optimization History** - Trial bazında objective value progression
2. **Parameter Importances** - Hangi parametrelerin en önemli olduğu (akademik için kritik!)
3. **Parallel Coordinate** - Tüm parametrelerin ve sonuçların etkileşimi
4. **Contour Plot** - 2D parameter etkileşimleri (örn: learning_rate vs gamma)
5. **Slice Plot** - Her parametrenin individual etkisi

### C. Karşılaştırma Analizi

**Tüm algoritmaları karşılaştır:**
```bash
python hyperparameter_optimization/analyze_results.py --compare-all
```

**Belirli studies'i karşılaştır:**
```bash
python hyperparameter_optimization/analyze_results.py \
    --compare \
    ppo_optimization_20240115_143022 \
    a2c_optimization_20240115_150133
```

**Oluşturulacak Görselleştirmeler:**
- Box plot: Algoritma bazında Sharpe ratio dağılımı
- Violin plot: Detaylı dağılım analizi
- Bar chart: En iyi sonuçların karşılaştırması

### D. LaTeX Tablo Oluşturma

```bash
python hyperparameter_optimization/analyze_results.py \
    --study-name ppo_optimization_20240115_143022 \
    --latex
```

**Örnek LaTeX Çıktısı:**
```latex
\begin{table}[htbp]
\centering
\caption{Hyperparameter Optimization Results: ppo\_optimization\_20240115\_143022}
\label{tab:hyperopt_results}
\begin{tabular}{ll}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Trials & 50 \\
Completed Trials & 48 \\
Pruned Trials & 2 \\
Best Sharpe Ratio & 1.8734 \\
Mean Sharpe Ratio & 1.4521 \\
\midrule
\multicolumn{2}{l}{\textbf{Best Hyperparameters:}} \\
learning\_rate & 0.000342 \\
n\_steps & 2048 \\
batch\_size & 64 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 🎓 Akademik Kullanım

### 1. Citation (Kaynak Gösterme)

Makalenizde Optuna'yı cite edin:

```bibtex
@inproceedings{akiba2019optuna,
  title={Optuna: A next-generation hyperparameter optimization framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}
}
```

### 2. Metodoloji (Methods Bölümü için)

**Örnek Metin:**

> We employed Bayesian optimization using Optuna [1] to systematically search for optimal hyperparameters across four state-of-the-art deep reinforcement learning algorithms: PPO, A2C, TD3, and SAC. The optimization objective was to maximize the Sharpe ratio on a validation set. We used Tree-structured Parzen Estimator (TPE) as the sampling algorithm and implemented early stopping via Median Pruner to efficiently allocate computational resources. Each trial was trained for 100,000 timesteps on a training set (2020-2022) and evaluated on a separate validation set (2023). We conducted 50 trials per algorithm, resulting in a total of 200 optimization runs.

### 3. Sonuçlar (Results Bölümü için)

```python
# İstatistikleri JSON'dan çıkarın
import json

with open('results/hyperparameter_studies/reports/ppo_optimization_statistics.json') as f:
    stats = json.load(f)

print(f"Best Sharpe Ratio: {stats['best_value']:.4f}")
print(f"Mean ± Std: {stats['mean_value']:.4f} ± {stats['std_value']:.4f}")
```

**Örnek Tablo (LaTeX):**

| Algorithm | Best Sharpe | Mean ± Std | Completed Trials | Pruned |
|-----------|-------------|------------|------------------|--------|
| PPO       | 1.87        | 1.45 ± 0.32| 48/50            | 2      |
| A2C       | 1.65        | 1.28 ± 0.41| 46/50            | 4      |
| TD3       | 1.92        | 1.51 ± 0.38| 47/50            | 3      |
| SAC       | **2.04**    | **1.63 ± 0.29** | **49/50**   | 1      |

### 4. Ablation Study (Parametre Hassasiyet Analizi)

Parameter importance plot'ları kullanarak ablation study yapın:

```bash
python hyperparameter_optimization/analyze_results.py \
    --study-name ppo_optimization_20240115_143022
```

`param_importances.html` dosyasını açın ve en önemli parametreleri belirleyin.

**Örnek Metin:**

> Parameter importance analysis revealed that learning rate (importance: 0.42) and entropy coefficient (importance: 0.28) were the most critical hyperparameters for PPO, while network architecture size had minimal impact (importance: 0.05). This suggests that proper exploration-exploitation balance is more crucial than model capacity for our trading environment.

### 5. Sensitivity Analysis

Contour plot ve slice plot kullanarak parameter interactions analiz edin.

### 6. Reproducibility (Tekrarlanabilirlik)

```bash
# Seed kullanarak sonuçları tekrarlanabilir yapın
python hyperparameter_optimization/run_optimization.py \
    --algorithm ppo \
    --trials 50 \
    --seed 42  # ← Sabit seed
```

---

## 🏗️ Sistem Mimarisi

### A. Genel Akış

```
┌─────────────────────────────────────────────────────────────┐
│                     Optuna Study                            │
│  (Bayesian Optimization with TPE Sampler)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Suggests hyperparameters
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Trial Execution                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Create TradingEnv (train data)                    │   │
│  │ 2. Initialize RL Model (PPO/A2C/TD3/SAC)            │   │
│  │ 3. Train with model.learn(timesteps)                │   │
│  │ 4. Evaluate on validation set                        │   │
│  │ 5. Return Sharpe Ratio                              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Reports objective value
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Optuna Pruning                             │
│  - Early stopping for poor trials                          │
│  - Median Pruner: prunes if below median performance       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ After N trials
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Best Hyperparameters Found                     │
│  - Saved to JSON                                            │
│  - Stored in SQLite database                               │
└─────────────────────────────────────────────────────────────┘
```

### B. Class Hierarchy

```
BaseHyperparameterOptimizer
├── PPOOptimizer
├── A2COptimizer
├── TD3Optimizer
└── SACOptimizer
```

### C. Key Components

1. **search_spaces.py** - Parametre tanımları ve arama uzayları
2. **base_optimizer.py** - Ortak optimization logic
3. **optimizers/** - Algoritma-specific implementation
4. **run_optimization.py** - CLI interface
5. **analyze_results.py** - Post-hoc analysis

---

## 💡 Tips & Best Practices

### 1. Training Süresi

| Configuration | Trials | Timesteps | Estimated Time |
|---------------|--------|-----------|----------------|
| Quick Test    | 2      | 10,000    | 5 min          |
| Development   | 10     | 50,000    | 1 hour         |
| Small Study   | 30     | 100,000   | 3 hours        |
| Medium Study  | 50     | 100,000   | 6 hours        |
| Large Study   | 100    | 200,000   | 24 hours       |
| Full Study    | 200    | 200,000   | 48 hours       |

**Not:** Süreler tek GPU (RTX 3080) için yaklaşık tahminlerdir.

### 2. Hyperparameter Tuning Stratejisi

**Phase 1: Broad Search (İlk 30 trial)**
- Geniş arama uzayı
- Tüm parametreleri optimize et
- Pruning agresif (Median Pruner)

**Phase 2: Focused Search (Sonraki 50 trial)**
- Phase 1'den en iyi bölgeyi belirle
- Arama uzayını daralt
- Daha az pruning

**Phase 3: Fine-tuning (Son 20 trial)**
- En iyi konfigürasyonun etrafında
- Çok dar arama uzayı
- Pruning yok

### 3. GPU Memory Yönetimi

```bash
# Paralel çalıştırma: Sadece GPU memory yeterliyse
python run_optimization.py --algorithm ppo --trials 100 --jobs 2

# GPU out-of-memory hatası alırsanız:
# 1. --jobs 1 kullanın (sequential)
# 2. --timesteps azaltın (örn: 50000)
# 3. net_arch_size "small" kullanın
```

### 4. Debug ve Monitoring

```bash
# TensorBoard ile training izleme
tensorboard --logdir logs/tensorboard

# Optuna Dashboard ile optimization izleme
optuna-dashboard sqlite:///results/hyperparameter_studies/optuna_studies.db
```

### 5. Sonuçları Kaydetme

```python
# Best parametreleri manuel olarak kullanma
import json

with open('results/hyperparameter_studies/best_params_ppo_xxx.json') as f:
    best_params = json.load(f)

# Model oluştur
model = PPO(
    policy="MlpPolicy",
    env=env,
    **best_params['best_params']
)
```

---

## ❓ Sık Sorulan Sorular (FAQ)

### Q1: Optimizasyon ne kadar sürer?

**A:**
- Quick test: 5-10 dakika
- Production study: 6-24 saat
- Full ablation study: 2-3 gün

### Q2: GPU gerekli mi?

**A:** Hayır ama şiddetle önerilir. CPU'da 10-20x daha yavaştır.

### Q3: Optuna dashboard nasıl başlatılır?

**A:**
```bash
optuna-dashboard sqlite:///results/hyperparameter_studies/optuna_studies.db
```

### Q4: Study'yi nasıl durdurur ve devam ettiririm?

**A:** `Ctrl+C` ile durdurun. Aynı `--study-name` ile tekrar çalıştırın, kaldığı yerden devam eder.

### Q5: Parametre aralıklarını nasıl değiştiririm?

**A:** [search_spaces.py](search_spaces.py) dosyasını düzenleyin.

### Q6: En iyi sonuçları nasıl bulurum?

**A:**
```bash
python analyze_results.py --list
python analyze_results.py --study-name <study_name>
```

### Q7: Parallel çalıştırma güvenli mi?

**A:** Evet ama GPU memory dikkat! `--jobs 2` genelde güvenli.

### Q8: Pruning nedir ve neden kullanılır?

**A:** Pruning, kötü performans gösteren trial'ları erken durdurarak hesaplama süresini %30-50 azaltır.

---

## 📧 Destek

Sorularınız için:
1. Bu README'yi okuyun
2. [Optuna Documentation](https://optuna.readthedocs.io/)
3. GitHub Issues

---

## 📄 Lisans

Bu proje MIT lisansı altındadır. Akademik çalışmalarda kullanırken lütfen Optuna'yı cite edin.

---

**Son Güncelleme:** 2025-01-14

**Versiyon:** 1.0.0

**Yazar:** RL Trading Project Team
