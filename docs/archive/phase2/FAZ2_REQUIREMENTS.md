# FAZ 2 GEREKSİNİMLERİ: RİSKE DUYARLI VE YORUMLANABİLİR DRL TİCARET SİSTEMİ

**Kaynak:** Ansari et al. (2024) - A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning
**Hedef:** BIST-30 Endeksi için risk-odaklı, açıklanabilir DRL sistemi
**Tarih:** 2025-12-14

---

## 1. FAZ 2'NİN ANA HEDEFLERİ

Faz 2'nin temel amacı, DRL ajanlarının BIST-30 piyasasının yüksek gürültü, oynaklık ve durağan olmama (non-stationarity) sorunlarına karşı sağlamlığını (robustness) artırmak ve finansal kurumlar için zorunlu olan yorumlanabilirliği (Explainability - XAI) sağlamaktır.

### Temel Hedefler:

1. **Risk Odaklı Performans**
   - Sadece kümülatif getiriyi maksimize etmek yerine
   - Maksimum Düşüşü (MDD) anlamlı ölçüde azaltan
   - Sharpe/Sortino Oranlarını yükselten politikalar öğrenmek

2. **Zenginleştirilmiş Durum (State) Temsili**
   - Ajanın piyasayı, ham fiyattan veya sadece teknik göstergelerden öte
   - Yapısal risk faktörleri (fundamental) bağlamında
   - Makroekonomik rejim değişimleri perspektifinde algılaması

3. **Algoritmik Şeffaflık (XAI)**
   - Ajanın aldığı alım/satım kararlarını post-hoc şeffaf bir şekilde açıklayabilen mekanizmalar

---

## 2. DRL ALGORİTMASI VE MİMARİ SEÇİMİ

### Algoritma Seçimi

Faz 1 sonuçlarına göre en dengeli performansı (en düşük MDD) gösteren **PPO (Proximal Policy Optimization)** veya en yüksek getiriyi (Sharpe Oranı) sunan **TD3 (Twin Delayed DDPG)** algoritması temel alınacaktır.

**Neden Bu Algoritmalar?**
- BIST-30'un çoklu varlık portföy optimizasyonu için gerekli olan **sürekli eylem uzayını** desteklemektedir
- PPO: Policy stability ve sample efficiency
- TD3: Off-policy learning, twin critics (Q-value overestimation azaltma)

### Mimari Gereklilikler

#### 1. Sürekli Eylem Uzayı
- BIST-30 endeksindeki tüm varlıkların portföy ağırlıklarını temsil eden
- **[-1, +1] aralığında sürekli değerler** üretilmelidir
- -1: Maksimum short (veya satış)
- +1: Maksimum long (veya alım)

#### 2. Actor-Critic (A-C) Mimarisi
- **Politika (Actor)** ve **Değer (Critic)** ağları
- Öğrenmenin istikrarı için entegre edilmelidir
- TD3 kullanılıyorsa: **İkiz eleştirmenler (twin critics)** Q-değeri aşırı tahminini azaltmak için korunmalıdır

---

## 3. MDP TASARIMI VE FAKTÖR ENTEGRASYONU

### A. Durum Uzayının Zenginleştirilmesi (State Representation)

Durum uzayı **S<sub>t</sub>**, yalnızca fiyat ve teknik göstergelerden oluşmamalı; ajanın kurumsal sağlık ve sistematik risk maruziyetini anlamasını sağlayan yeni veri katmanları eklenmelidir.

#### 1. Sistematik Risk Faktörleri (Fundamental/BARRA Entegrasyonu)

**Girdi:**
- Şirketlerin bilanço, gelir tablosu ve nakit akışı tablolarından türetilen temel finansal oranlar
- Örnekler:
  - ROE (Return on Equity) - Özkaynak karlılığı
  - ROA (Return on Assets) - Aktif karlılığı
  - Kaldıraç Oranları (Debt/Equity)
  - Cari Oran (Current Ratio)
  - F/K Oranı (P/E Ratio)
  - PD/DD Oranı (P/B Ratio)
  - Kar Marjı (Profit Margin)

**Amaç:**
Bu faktörlerin entegrasyonu, ajana **aşağı yönlü risk kontrolü (MDD)** konusunda önemli ölçüde fayda sağlayarak:
- Sadece teknik göstergelere dayalı reaktif bir sistem olmaktan çıkarır
- Proaktif bir sisteme dönüştürür

**Literatür Desteği:**
- Fundamental faktörler MDD azalmasında %15-20 iyileşme sağlar [Ansari et al. 2024]
- BARRA ve Fama-French faktör modelleri risk kontrolünde kanıtlanmış

#### 2. Makroekonomik Göstergeler

**Girdi:**
Türkiye piyasasına özgü dışsal faktörler:
- **TCMB Politika Faizi**: Para politikası rejimi
- **TÜFE Enflasyon**: Tüketici fiyat endeksi (YoY %)
- **ÜFE Enflasyon**: Üretici fiyat endeksi (YoY %)
- **USD/TRY**: Dolar kuru hareketleri
- **EUR/TRY**: Euro kuru hareketleri
- **BIST-100 Endeksi**: Genel piyasa rejimi göstergesi

**Amaç:**
Makroekonomik göstergeler, ajanın piyasa rejimi değişimlerini yakalamasına yardımcı olarak:
- Faiz artışları dönemlerini algılar
- Yüksek volatilite dönemlerini öngörür
- Sistemin durağan olmama durumuna karşı dayanıklılığını artırır

**Örnek Senaryo:**
```
Eğer:
  TCMB politika_faizi > %45 VE
  USD/TRY artış_oranı > %10 (son 30 gün)
O zaman:
  Risk → YÜKSEK
  Ajan kararı → Pozisyonları azalt, nakit tut
```

---

### B. Riske Duyarlı Ödül Fonksiyonu Optimizasyonu (Reward Shaping)

Ödül fonksiyonu **R<sub>t</sub>**, salt getiri yerine **riske ayarlı getiri** sinyallerini içermelidir.

#### 1. Sharpe/Sortino Oranı Odaklı PSR Ödülü

**PSR (Portfolio-Sharpe-Returns) Formülü:**
```
R_t = w1 * ΔPortfolio + w2 * Sharpe_ratio + w3 * (-MDD) + w4 * (-Volatility) + w5 * (-Commission)
```

**Ağırlıklar (Önerilen):**
- w1 = 0.5: Portfolio getirisi
- w2 = 0.3: Sharpe ratio (risk-ayarlı getiri)
- w3 = -0.1: MDD cezası
- w4 = -0.05: Volatilite cezası
- w5 = -0.05: İşlem maliyeti cezası

**Neden PSR?**
- İşlem maliyeti cezaları içerir
- Sharpe/Sortino oranlarını birleştirir
- Riske ayarlı getirilerin kullanılması, model performansında kayda değer iyileşme sağlar

#### 2. Doğrudan Risk Cezaları

Öğrenme sürecini doğrudan aşağı yönlı riske karşı hassas hale getirmek için ödül fonksiyonuna eklenmelidir:

**Maksimum Düşüş (MDD) Cezası:**
```python
MDD = min((Portfolio_t - Cummax(Portfolio)) / Cummax(Portfolio))
MDD_penalty = -abs(MDD)  # Negatif değer ceza olarak
```

**Volatilite Cezası:**
```python
Volatility = std(returns_last_30_days) * sqrt(252)  # Annualized
Vol_penalty = -Volatility / target_volatility  # Normalize edilmiş
```

**Trade Frequency Bonus/Penalty:**
```python
# Aşırı muhafazakarlığı önle
if trades_per_episode < 20:
    penalty = -0.1
elif trades_per_episode > 100:
    penalty = -0.05  # Overtrading
else:
    penalty = 0  # İdeal aralık
```

---

## 4. GELİŞMİŞ ENTEGRASYON VE SAĞLAMLIK

### A. Yorumlanabilirlik (XAI) Mekanizması

DRL ajanının kararlarının denetlenebilirliğini ve güvenilirliğini artırmak için **iki katmanlı XAI entegrasyonu** zorunludur.

#### 1. İçsel Şeffaflık (Attention Layer) - OPSIYONEL

**Yaklaşım:**
- Model mimarisine (LSTM katmanları öncesine) **Dikkat Katmanı (Attention Layer)** eklenmelidir
- Bu katman, ajanın her bir ticaret kararını verirken hangi temel/teknik/makro faktörlere ne kadar ağırlık verdiğini gösteren bir **Dikkat Vektörü** sağlamalıdır

**Çıktı:**
```python
attention_weights = {
    'roe_AKBNK': 0.23,        # ROE'ye %23 dikkat
    'usd_try': 0.18,          # Kur'a %18 dikkat
    'rsi_THYAO': 0.15,        # RSI'ya %15 dikkat
    ...
}
```

**Değer:**
- Akademik yayınlar için güçlü
- Feature importance'ı öğrenme süreci boyunca takip
- Karmaşık implementasyon (Custom PPO policy gerektirir)

#### 2. Post-Hoc Açıklama - ÖNCELİKLİ

Ajanın eğitim sonrası (gerçek ticaret anında) verdiği kararların nedenlerini açıklayabilmek için modelden bağımsız (model-agnostic) analiz teknikleri entegre edilmelidir.

**SHAP (SHapley Additive exPlanations):**
- Global feature importance
- Shapley values teorisine dayalı
- Finansal kurumlar tarafından kabul edilen standart
- **Öncelik: P1 (Mutlaka yapılmalı)**

**LIME (Local Interpretable Model-agnostic Explanations):**
- Lokal (tek bir karar) açıklama
- Lineer yaklaşım
- SHAP'e tamamlayıcı
- **Öncelik: P2 (Yapılmalı)**

**Kullanım Senaryosu:**
```
User: "25 Aralık'ta neden AKBNK sattınız?"

XAI Modülü:
"Bu karar 5 faktörden etkilendi:
  1. AKBNK ROE düşük (0.08 < ortalam 0.12) → -0.15 etki
  2. USD/TRY yükseldi (%5 artış, 30 gün) → -0.12 etki
  3. AKBNK RSI aşırı alımda (78 > 70) → -0.10 etki
  4. TCMB faizi arttı (%2 artış) → -0.08 etki
  5. Piyasa volatilitesi yüksek (30% > hedef 25%) → -0.06 etki

Toplam: Negatif sinyal (-0.51) → SATIM kararı"
```

---

### B. Gürültüye Karşı Sağlamlık (Anti-Risk Robustness) - OPSIYONEL

Finansal verilerin gürültülü doğası nedeniyle hatalı kararlar alınmasını önlemek için özellik çıkarım (feature extraction) aşamasına bir **gürültüden arındırma (denoising)** modülü eklenebilir.

#### SSDAE (Stacked Sparse Denoising Autoencoder)

**Yapı:**
```
Raw Features (97-dim) → SSDAE Encoder → Clean Features (32-dim) → DRL Agent
```

**Avantajlar:**
- Geleneksel veya zenginleştirilmiş girdiler, DRL ajanına sunulmadan önce
- SSDAE gibi otomatik kodlayıcı ağlar kullanılarak gürültüye dayanıklı (anti-risk) özelliklere dönüştürülür
- Bu, modelin risk kontrol yeteneğini (MDD) önemli ölçüde artırır

**Literatür:**
- SSDAE, MDD azalmasında %10-15 iyileşme göstermiş [Ansari et al. 2024]
- Finansal time series forecasting'te yaygın kullanım

**Öncelik:**
- **P3 (Opsiyonel)** - Eğer MDD hedeflere ulaşmazsa ekle
- Sprint 5'te test edilebilir

---

## 5. KRİTİK STRATEJİK UYARILAR VE İNCE AYARLAR

### A. Stratejik Uyarı 1: Negatif Öğrenme Dışsallığı

**Tehdit:**
Simülasyonlar genellikle tek bir ajanın piyasayı etkilemediğini (partial-equilibrium) varsayar. Ancak, canlı piyasada çok sayıda yapay zeka ajanı etkileşime girdiğinde, ajanların keşif (exploration) amaçlı işlemleri fiyat sürecine gürültü enjekte ederek diğer ajanların öğrenme sinyallerini bozar. Bu "negatif öğrenme dışsallığı", algoritmik ticaret stratejilerinin karlılığını abartabilir ve performansın düşmesine neden olabilir.

**Öneri:**
- **Faz 2'de GEREKSİZ** (canlı trading yok, backtest ortamı)
- **Faz 3'te kritik** (production deployment için)
- Faz 3'te: Multi-agent simülasyon, market impact modeling

### B. Stratejik Uyarı 2: Aşırı Muhafazakarlıktan Kaçınma

**Tehdit:**
MDD'yi agresif bir şekilde düşürmeye odaklanan ödül fonksiyonları, ajanların aşırı muhafazakar hale gelmesine neden olabilir. Örneğin, bir Dueling DDQN V3 modelinin **1.2278 gibi yüksek bir Sharpe oranını sadece üç işlemle elde etmesi**, ajanın kârlı fırsatları kaçırdığını gösterir.

**Çözümler:**

1. **Trade Frequency Monitoring:**
   ```python
   assert trades_per_episode > 20, "Model too conservative!"
   ```

2. **Trade Frequency Bonus:**
   ```python
   if trades < 20:
       reward -= 0.1 * (20 - trades) / 20
   ```

3. **Entropy Bonus (Zaten PPO'da var):**
   ```python
   PPO(..., ent_coef=0.01)  # Exploration teşvik
   ```

**Öneri:**
Politika stabilizasyonu (PPO/TD3) ve risk kontrolü (MDD cezası) sürdürülürken, modelin kabul edilebilir bir işlem sıklığını (Trade Frequency) koruması sağlanmalıdır.

### C. Teknik İnce Ayarlar (Stabilizasyon)

#### 1. PPO/TD3 Ayarları

**PPO:**
- ✅ **Clipping mekanizması** aktif: `clip_range=0.2`
- ✅ Policy güncelleme güvenliği sağlanıyor

**TD3:**
- ✅ **Twin critics** aktif: Q-value overestimation engelleniyor
- ✅ **Target policy smoothing**: Noise ekleniyor
- ✅ **Delayed policy updates**: `policy_delay=2`

#### 2. Keşif ve Aşırı İşlem Kontrolü

**Önerilen Teknikler:**
- **Softmax Örneklemesi (Softmax Sampling)**:
  ```python
  # Action selection with temperature
  action_probs = softmax(Q_values / temperature)
  action = np.random.choice(actions, p=action_probs)
  ```

- **Soğuma Mantığı (Cooldown Logic)**:
  ```python
  # Bir hisseyi satıldıktan sonra N gün bekle
  if symbol in recently_sold and days_since_sale < 5:
      action[symbol] = 0  # No trade
  ```

**Değer:**
- Yüksek kaliteli, ancak daha az sıklıkta işlem yapılmasını teşvik eder
- DDQN V3'te başarıyla uygulanmış (literatür)

**Öncelik:**
- **P3 (Opsiyonel)** - Sprint 5'te eklenebilir

---

## 6. BAŞARI KRİTERLERİ

### Minimum Başarı (Acceptable)

| Metrik | Faz 1 Baseline | Faz 2 Hedef | İyileşme |
|--------|----------------|-------------|----------|
| **Max Drawdown (MDD)** | -13.81% | ≤ -12.5% | ≥10% iyileşme |
| **Sharpe Ratio** | 1.26 | ≥ 1.26 | Korunuyor |
| **Cumulative Return** | 39.59% | ≥ 35% | Kabul edilebilir |
| **Trade Frequency** | 67 | 30-100 | Kontrollü |

### Hedef Başarı (Target)

| Metrik | Faz 2 Hedef | İyileşme |
|--------|-------------|----------|
| **Max Drawdown** | ≤ -10% | **%27 iyileşme** |
| **Sharpe Ratio** | ≥ 1.5 | **%19 artış** |
| **Cumulative Return** | ≥ 45% | **%14 artış** |
| **Sortino Ratio** | ≥ 2.0 | Yeni metrik |

### Mükemmel Başarı (Excellent - Makale Kalitesi)

| Metrik | Faz 2 Hedef | İyileşme |
|--------|-------------|----------|
| **Max Drawdown** | ≤ -8% | **%42 iyileşme** |
| **Sharpe Ratio** | ≥ 2.0 | **%59 artış** |
| **Cumulative Return** | ≥ 55% | **%39 artış** |
| **Win Rate** | ≥ 60% | Yeni metrik |

---

## 7. ÖNCELİK SIRALA MASI

### P1: Kritik (Mutlaka Yapılmalı)
1. ✅ Fundamental data entegrasyonu
2. ✅ Makroekonomik göstergeler
3. ✅ PSR reward fonksiyonu
4. ✅ SHAP explainability

### P2: Önemli (Yapılmalı)
5. ✅ LIME explainability
6. ✅ Hyperparameter tuning (Optuna)
7. ✅ Ablation studies

### P3: Opsiyonel (İyileştirme)
8. 🟡 SSDAE denoising
9. 🟡 Attention layer
10. 🟡 Cooldown logic

---

## 8. BAĞIMLILIKLAR VE GEREKSİNİMLER

### Yazılım Gereksinimleri

```python
# Yeni kütüphaneler (requirements.txt'e eklenecek)
shap==0.42.0              # SHAP explainability
lime==0.2.0.1             # LIME explainability
evds==0.2.0               # TCMB EVDS API (makro data)
optuna==4.6.0             # Hyperparameter tuning (zaten var)
```

### Veri Gereksinimleri

1. **Fundamental Data:**
   - Kaynak: yfinance.info (7-8 oran)
   - Alternatif: Manuel CSV (eksikleri doldurmak için)
   - Güncelleme: Üç aylık

2. **Makroekonomik Data:**
   - Kaynak: TCMB EVDS API (resmi)
   - API Key: Gerekli (ücretsiz kayıt)
   - Güncelleme: Günlük/haftalık

3. **Teknik Data:**
   - Kaynak: yfinance (mevcut)
   - Güncelleme: Günlük

### Compute Gereksinimleri

- **Eğitim:**
  - GPU: Opsiyonel (CPU'da ~5 dakika/100K timesteps)
  - RAM: 8 GB yeterli
  - Disk: 5 GB (modeller + results)

- **XAI Analizi:**
  - SHAP: Background data için 100-500 MB RAM
  - LIME: Minimal

---

## SONUÇ

Faz 2, sistemimizi **reaktif** bir teknik analiz botundan **proaktif** bir risk-yönetimli trading sistemine dönüştürecek.

**Ana Değişiklikler:**
1. ✅ **State Space:** 56 → 97 features (+fundamental +macro)
2. ✅ **Reward Function:** Basit → PSR (risk-adjusted)
3. ✅ **Explainability:** Yok → SHAP/LIME (post-hoc)
4. 🟡 **Denoising:** Yok → SSDAE (opsiyonel)

**Beklenen Sonuç:**
- MDD: -13.81% → **≤ -10%** (%27 iyileşme)
- Sharpe: 1.26 → **≥ 1.5** (%19 artış)
- Akademik yayın kalitesinde sonuçlar

---

**Doküman Durumu:** ✅ Onaylandı
**Sonraki Adım:** [FAZ2_IMPLEMENTATION_PLAN.md](FAZ2_IMPLEMENTATION_PLAN.md) - Sprint detayları
**Referans:** [FAZ2_ANALYSIS.md](FAZ2_ANALYSIS.md) - Teknik analiz
