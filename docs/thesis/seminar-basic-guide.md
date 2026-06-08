# Danışman Görüşmesi — Basit Anlatım Rehberi

> Bu doküman seminer-overview.md'nin sade özetidir. Görüşme sırasında **sıra ile** anlatılacak şekilde düzenlendi. Her bölüm kısa, sözel anlatıma uygun.

---

## 1. Sistem Nedir? (30 saniye)

BIST-30 hisseleri için **pekiştirmeli öğrenme tabanlı al-sat sistemi** geliştirdim. Sistem üç katmandan oluşuyor:

1. **Veri katmanı:** OHLCV + makro (faiz, enflasyon, döviz, VIX, US10Y, DXY) + fundamental (P/E, ROE, ...) verilerini topluyor.
2. **Tahmin katmanı:** 5 farklı modelden oluşan ensemble, her hisse için ertesi gün getirisini ve yönünü tahmin ediyor.
3. **RL katmanı:** Bir RL ajanı, hem teknik göstergeleri hem de tahmin çıktısını gözlemleyerek **al / sat / tut** kararı veriyor.

Faz 1-2-3 tamamlandı: tek-ajanlı baseline çalışıyor, dashboard hazır. **Tez hedefi:** bunu sektör-bazlı çoklu-ajan + ajanlar arası iletişim sistemine evirmek.

---

## 2. RL Modelleri (sıra: A2C → PPO → TD3)

Üç farklı RL algoritması denedim. Hepsi Stable-Baselines3 kütüphanesinden.

### A2C — Advantage Actor-Critic
**Mantık:** İki sinir ağı paralel çalışır — *actor* (aksiyon seçer) ve *critic* (durumun ne kadar iyi olduğunu skorlar). Critic'in tahmininden actor'ün gerçek ödülünü çıkararak "avantaj" hesaplanır; actor bu avantaja göre güncellenir.
**Fark:** En basit ve en hızlı; on-policy çalışır (sadece güncel politikanın verisini kullanır). Genellikle baseline olarak kullanılır, sample efficiency düşüktür.

### PPO — Proximal Policy Optimization
**Mantık:** A2C ile aynı actor-critic mantığı, ama her güncellemede politikanın **eski versiyondan çok uzaklaşmasını engelleyen** bir kırpma (clipping) mekanizması ekler. Bu sayede eğitim kararlı, ani çöküşler yaşamıyor.
**Fark:** A2C'den daha kararlı ve daha güvenilir; finans gibi gürültülü ortamlarda **fiili standart** haline geldi. Halen on-policy, ama clipping sayesinde aynı veriyi birkaç epoch boyunca güvenle kullanabiliyor.

### TD3 — Twin Delayed DDPG
**Mantık:** İki critic ağı kullanır (twin), aksiyona gürültü ekler ve actor'ü critic'ten daha seyrek günceller (delayed). Bu üç hile, sürekli aksiyon uzayında değer tahmininin "şişirilmesini" (overestimation bias) önler.
**Fark:** Off-policy (geçmiş verileri replay buffer'dan tekrar kullanır) → sample-efficient. Sürekli aksiyon uzayında (ör. portföy ağırlığı 0.0-1.0) PPO'dan daha iyi çalışabilir; ama hiperparametre ayarı daha hassastır.

### Özet Tablo

| Model | Tip | Aksiyon | Avantaj | Dezavantaj |
|---|---|---|---|---|
| A2C | on-policy | Discrete/Continuous | Basit, hızlı | Düşük sample efficiency |
| PPO | on-policy | Discrete/Continuous | Kararlı, robust | Yine de on-policy |
| TD3 | off-policy | Continuous | Sample-efficient, doğru değer tahmini | Hiperparametre hassas |

---

## 3. Tahmin Modelleri (sıra: ağaç tabanlılar → derin öğrenme → ensemble)

5 model paralel eğitilip ensemble ile birleştirildi. Her biri farklı bir güçlü yan getiriyor.

- **XGBoost:** Gradient boosting ağaç ailesi; ardışık ağaçlar bir öncekinin hatasını tahmin ederek genel doğruluğu artırır. Tablo verisinde sektör standardı.
- **LightGBM:** XGBoost ile aynı mantık, ama leaf-wise büyüme + histogram tabanlı bölme sayesinde çok daha hızlı; büyük veri ve yüksek feature sayısında avantajlı.
- **CatBoost:** Yine gradient boosting, ama kategorik feature'ları otomatik kodlar ve symmetric tree kullanarak overfitting'e karşı dayanıklıdır.
- **BiLSTM:** İki yönlü LSTM; zaman serisini hem geçmişten geleceğe hem de geleceğden geçmişe okuyarak uzun-vadeli bağımlılıkları yakalar.
- **TFT (Temporal Fusion Transformer):** Attention tabanlı zaman serisi modeli; statik/dinamik feature'ları ayrıştırıp hangi zaman adımının önemli olduğunu kendisi öğrenir.
- **Stacking Ensemble:** Yukarıdaki 5 modelin çıktısı bir **meta-learner**'a (Ridge veya XGBoost) verilir; meta-learner hangi modele ne kadar güveneceğini öğrenir. 3-yönlü kronolojik split (60/20/20) ile data leakage engellenir.

### Ekstra modüller
- **ICEEMDAN:** Fiyat serisinden yüksek-frekans gürültüyü atan sinyal işleme tekniği — tahmin girdisini temizler.
- **TATS:** Trend-adjusted düzeltici — ensemble çıktısının trend yönünü ayrı bir XGBoost classifier ile doğrular.
- **SHAP:** Hangi feature'ın tahmini ne kadar etkilediğini açıklar; yorumlanabilirlik için.

---

## 4. Faz Faz Ne Yaptım? (kronolojik anlatım)

**Faz 1 — POC.** 5 hisse, 56-boyutlu state, A2C/PPO/TD3 baseline. Çalışan tek-ajan + dashboard.

**Faz 2 — Tahmin Sistemi.** 5-modelli ensemble + stacking + walk-forward (purge + embargo) + RL entegrasyonu (sembol başına +4 feature: tahmini getiri, yön, güven, ensemble uyumu).

**Faz 3 — Production.**
- 3.1: Bug fix'ler (reward sayım hatası, meta-learner leakage, embargo, TFT VSN, direction head, permutation importance).
- 3.2: ICEEMDAN + TATS + global makro (VIX, US10Y, DXY).
- 3.3: ATR tabanlı pozisyon boyutu + Kelly Criterion (opt-in).
- 3.4: SHAP açıklanabilirlik + Sortino/Calmar/Deflated Sharpe/Turnover metrikleri.

---

## 5. Tez Vizyonu — Bundan Sonra Ne?

**Problem:** Mevcut sistem tek-ajanlı. BIST-30'da tek politikanın 30 farklı hisseyi yönetmesi ölçeklenmez; banka hissesi ile teknoloji hissesi aynı mekanizmayı kullanmak zorunda kalır.

**Çözüm:** Her sektör için ayrı bir RL ajanı (~8 ajan: Banka, Enerji, Sanayi, Perakende, Holding, Telekom, Teknoloji, Diğer). Ajanlar **birbiriyle iletişim kurar** — özellikle ensemble tahmin güvenini birbirlerine mesaj olarak gönderir.

**Karşılaştırılacak iletişim mimarileri:**
1. No-comm (iletişimsiz baseline, IPPO)
2. Attention (TarMAC-benzeri)
3. GNN/GAT (graf tabanlı)
4. **Tahmin-destekli** (özgün katkı: tahmin güveni → mesaj ağırlığı)

**Beklenen katkı:** Literatürde MARL-finansta iletişim mimarisi karşılaştırması ve tahmin entegrasyonu boşluğu var; özellikle gelişen piyasalarda (BIST) çalışma yok denecek kadar az.

---

## 6. Takvim (Haziran 2026 → Haziran 2027)

| Dönem | İçerik |
|---|---|
| Haz–Eyl '26 | Baseline dondurma + literatür + tez önerisi onayı |
| Eyl–Ara '26 | Framework geçişi (SB3 → PettingZoo/RLlib) + no-comm baseline + Makale 2 submit |
| Ara '26–Mar '27 | İletişim mimarileri (attention + GNN) + tahmin-destekli iletişim |
| Mar–Haz '27 | Rejim analizi + tez yazımı + Makale 1 submit + savunma |

---

## 7. Danışmana Sorulacak Sorular

1. Sektör-bazlı (~8 ajan) granülarite yeterli mi, yoksa hiyerarşik mimari daha mı uygun?
2. 3 iletişim mimarisi karşılaştırması fazla mı? 2'ye indirilsin mi?
3. Makale 2'yi (ensemble + ICEEMDAN + TATS) erken submit etmek mantıklı mı?
4. Framework olarak PettingZoo+RLlib mi, yoksa SB3 üzerinde custom wrapper mı?
5. Klasik baseline'lar (buy&hold, equal-weight, Markowitz) eklensin mi?

---

## 8. Hızlı Cevap Kartı (danışman sorarsa)

- **"PSR nedir?"** → Probabilistic Sharpe Ratio; Sharpe'ın istatistiksel anlamlılığını ölçer (Ansari Eq. 1).
- **"Walk-forward nedir?"** → Zaman serisinde gelecek bilgisi sızmasın diye train/test'i kronolojik kaydırarak yapma yöntemi. Purge gap (5 gün) + embargo (3 gün) eklenmiş.
- **"Data leakage nasıl engellendi?"** → Tüm feature'lar ≥1 gün gecikmeli; meta-learner 3-yönlü kronolojik split (60/20/20) ile eğitiliyor.
- **"Neden Ansari makalesi?"** → BIST için en yakın metodoloji (multifaceted prediction + RL); replikasyon + genişletme şeklinde kuruldu.
- **"Açıklanabilirlik ne durumda?"** → SHAP entegre, `/prediction/explain/{symbol}` API endpoint'i ile her tahmin için feature katkısı görselleştirilebiliyor.
