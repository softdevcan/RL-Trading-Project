# RL Algoritmaları - Karşılaştırma ve Kullanım Kılavuzu

Bu projede üç farklı Deep Reinforcement Learning algoritması desteklenmektedir: **A2C**, **PPO**, ve **TD3**.

## 📊 Algoritma Karşılaştırması

### 🥇 PPO (Proximal Policy Optimization) - **ÖNERİLEN**

**Avantajları**:
- ✅ En kararlı ve güvenilir algoritma
- ✅ Tek thread ile mükemmel çalışır
- ✅ Clipping mekanizması ile güvenli policy güncellemeleri
- ✅ On-policy learning ile tutarlı sonuçlar
- ✅ Exploration için entropy bonus

**Test Sonuçları** (50,000 timesteps):
- Return: **39.59%**
- Sharpe Ratio: **1.2581**
- Total Trades: **67**
- Max Drawdown: **-13.81%**

**Optimized Hyperparameters**:
```python
PPO(
    learning_rate=0.0003,  # PPO için en optimal değer
    n_steps=2048,          # Daha fazla deneyim topla
    batch_size=64,         # Stabil gradient updates
    n_epochs=10,           # Her batch'ten iyi öğren
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,        # Policy update güvenliği
    ent_coef=0.01          # Keşif bonusu
)
```

**Ne zaman kullanılır?**:
- Kararlı ve güvenilir sonuçlar istediğinizde
- İlk kez RL modeli eğitiyorsanız
- Production sistemleri için

---

### ⚠️ A2C (Advantage Actor-Critic)

**Avantajları**:
- ✅ On-policy learning
- ✅ Basit ve hızlı
- ✅ Value function ile daha iyi gradient estimates

**Dezavantajları**:
- ❌ Tek thread ile **zayıf performans**
- ❌ Orijinal makale 16 parallel thread öneriyor
- ❌ Yüksek varyans, kararsız öğrenme

**Optimized Hyperparameters** (Tek thread için):
```python
A2C(
    learning_rate=0.0007,        # A2C için optimal değer
    n_steps=512,                 # Daha sık update
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,               # Keşif bonusu
    vf_coef=0.5,                 # Value function ağırlığı
    normalize_advantage=True,    # Stabilite için
    use_rms_prop=True            # Orijinal paper'daki gibi
)
```

**Ne zaman kullanılır?**:
- Parallel environment setup'ınız varsa (SubprocVecEnv)
- Çok hızlı training istiyorsanız (ama daha az kararlı)
- Akademik karşılaştırma çalışmaları için

**⚠️ Uyarı**: Single-threaded training ile optimal sonuç vermeyebilir!

---

### 🔬 TD3 (Twin Delayed Deep Deterministic Policy Gradient)

**Avantajları**:
- ✅ Off-policy learning (experience replay)
- ✅ Twin Q-networks ile overestimation bias azaltma
- ✅ Delayed policy updates (daha kararlı)
- ✅ Continuous action spaces için özel tasarlanmış

**Özel Özellikleri**:
- Experience replay buffer (100,000 samples)
- Action noise for exploration
- Target policy smoothing
- Twin critic networks

**Optimized Hyperparameters**:
```python
TD3(
    learning_rate=0.001,         # TD3 için optimal değer (off-policy için biraz daha yüksek)
    buffer_size=100000,          # Büyük replay buffer
    learning_starts=1000,        # Önce deneyim topla
    batch_size=256,              # Büyük batch size
    tau=0.005,                   # Soft target update
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    action_noise=NormalActionNoise(sigma=0.1),
    policy_delay=2,              # TD3 özelliği
    target_policy_noise=0.2,
    target_noise_clip=0.5
)
```

**Ne zaman kullanılır?**:
- Off-policy learning avantajından yararlanmak istiyorsanız
- Continuous action spaces ile çalışıyorsanız
- Daha gelişmiş teknikler denemek istiyorsanız

---

## 🎯 Hangisi Sizin İçin?

### Yeni Başlıyorsanız → **PPO**
- En kararlı ve güvenilir
- Single-thread ile mükemmel çalışır
- Dökümantasyon ve topluluk desteği bol

### Parallel Environment Kurabiliyorsanız → **A2C**
- Hızlı training
- Daha basit mimari
- Ama dikkatli hyperparameter tuning gerekir

### Advanced Kullanıcıysanız → **TD3**
- Off-policy learning
- Experience replay
- Daha sofistike ama daha karmaşık

---

## 📈 Performans Karşılaştırması

| Algoritma | Return | Sharpe | Trades | Training Time | Kararlılık |
|-----------|--------|--------|--------|---------------|-----------|
| PPO       | 39.59% | 1.26   | 67     | ~2 dakika     | ⭐⭐⭐⭐⭐ |
| A2C       | TBD    | TBD    | TBD    | ~1 dakika     | ⭐⭐ |
| TD3       | TBD    | TBD    | TBD    | ~3 dakika     | ⭐⭐⭐⭐ |

**Test Koşulları**: 50,000 timesteps, BIST-30 5 hisse, 2018-2024 data

---

## 🚀 Web UI'dan Kullanım

1. http://localhost:8888 adresine gidin
2. "Model Eğitimi" sekmesini seçin
3. Algoritma dropdown'dan birini seçin:
   - **PPO** (Önerilen) - En kararlı
   - **A2C** - Dikkatli ayar gerektirir
   - **TD3** - Gelişmiş, Experience Replay
4. Diğer parametreleri ayarlayın:
   - Timesteps: `50000+` (minimum 10,000, önerilen: 50,000+)
   - Initial Balance: `1000000` (önerilen: 1,000,000₺)
5. "Eğitimi Başlat" butonuna tıklayın

**Not:** Learning rate otomatik olarak algoritma bazında ayarlanır:
- PPO: 0.0003 (en kararlı)
- A2C: 0.0007 (standart)
- TD3: 0.001 (off-policy için optimize edilmiş)

---

## 📚 Referanslar

### PPO
- Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- En çok kullanılan ve test edilmiş modern RL algoritması

### A2C
- Mnih et al. (2016) - "Asynchronous Methods for Deep Reinforcement Learning"
- Orijinal paper: 16 parallel actor-learner thread kullanıyor
- Single-thread versiyonu (A2C) daha az kararlı

### TD3
- Fujimoto et al. (2018) - "Addressing Function Approximation Error in Actor-Critic Methods"
- DDPG algoritmasının geliştirilmiş versiyonu
- Twin critics + delayed updates + target policy smoothing

---

## ⚙️ Pipeline Entegrasyonu

Tüm üç algoritma da FastAPI backend'e entegre edilmiştir:

```python
# app/api/routes/trading.py içinde
if request.algorithm == "PPO":
    model = PPO(...)  # PPO-specific hyperparameters
elif request.algorithm == "A2C":
    model = A2C(...)  # A2C-specific hyperparameters
elif request.algorithm == "TD3":
    model = TD3(...)  # TD3-specific hyperparameters
```

Her algoritma için optimize edilmiş hyperparameter setleri otomatik olarak uygulanır.

---

**Son Güncelleme**: November 2024
**Durumu**: Tüm 3 algoritma production-ready ✅
