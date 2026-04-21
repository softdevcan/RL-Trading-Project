# Tez Vizyonu ve Kapsam Belgesi

**Proje:** RL Trading — BIST-30 Çoklu-Ajan Portföy Yönetim Sistemi
**Belge Türü:** Vizyon, kapsam ve teknik fizibilite analizi
**Durum:** Taslak (2026-04-21)
**Amaç:** Tez konusunun sınırlarını, literatürdeki konumunu, teknik yapılabilirliğini ve ilerleme yol haritasını tek bir başvuru belgesinde toplamak.

---

## 0. Yönetici Özeti

Mevcut sistem; BIST-30 endeksinde derin pekiştirmeli öğrenme (DRL) tabanlı alım-satım için **tek-ajanlı** bir altyapıdır ([Ansari et al., 2024] metodolojisi BIST-30'a uyarlanmış; ensemble tahmin, ICEEMDAN gürültü filtresi, TATS trend düzeltici, ATR/Kelly pozisyon boyutlandırma ve SHAP açıklanabilirlik modülleri dahil edilmiştir). Tezin vizyonu, bu altyapının **çoklu-ajan pekiştirmeli öğrenme (MARL)** paradigmasına evrilerek her varlığın ya da sektörün bir ajan olarak modellendiği, ajanlar arasında öğrenilmiş iletişim protokollerinin bulunduğu ve tahmin sisteminin bu iletişime sinyal besleyerek koordinasyonu zenginleştirdiği **"Prediction-Augmented Communicating Agents for Portfolio Management"** çerçevesini üretmektir.

Tez "dipsiz kuyu" olmamalıdır. Bu belge:
- Her araştırma sorusunu **akademik, finansal ve teknik** açıdan değerlendirir.
- Her tasarım seçeneği için **doğru-yanlış-eksik** analizi yapar ve **yapılabilirlik (fizibilite)** değerlendirmesi sunar.
- Tezin **kesin kapsamını** (scope) ve **kapsam dışını** (out-of-scope) belirler.
- Başarı kriterlerini ve milestone haritasını tanımlar.

**Bu belge karar belgesi değildir.** Süreç boyunca güncellenecek, her milestone'da gözden geçirilecek, gerekirse gevşetilecek / sıkılaştırılacaktır.

---

## 1. Problem Tanımı ve Motivasyon

### 1.1 Portföy Yönetimi Problemi

Portföy yönetimi, belirli bir bütçe altında, çoklu riskli varlıklar arasında zaman içinde ağırlık dağılımı yapma problemidir. Klasik yaklaşımlar (Markowitz ortalama-varyans, Black-Litterman, Risk Parity) parametrik varsayımlar (getirilerin normalliği, durağan kovaryans vs.) üzerine kuruludur; gerçek piyasa dinamikleri (rejim değişimleri, fat-tailed dağılımlar, likidite şokları) bu varsayımları sık sık bozar.

Derin pekiştirmeli öğrenme (DRL) portföy yönetiminde model-serbest yaklaşım sunar: ajan, piyasa dinamiklerini önceden modellemeden, getiri-risk ödünleşimini doğrudan deneyim üzerinden öğrenir. 2023-2026 döneminde bu alan hızla olgunlaşmış; FinRL, ElegantRL gibi açık kaynak çatıları benchmark altyapısı sunmaya başlamıştır ([FinRL, 2024]).

### 1.2 Tek-Ajanlı Yaklaşımın Sınırları

Mevcut sistemimiz dahil literatürdeki çalışmaların çoğu, portföyü **tek bir ajan**ın gözünden modellemektedir: ajan bütün varlıkları tek observation vektöründe görür ve tek bir action vektörüyle alım-satım kararları verir.

Bu yaklaşımın yapısal zayıflıkları:

1. **Ölçeklenme sorunu:** Varlık sayısı (N) arttıkça state ve action uzayı doğrusal olarak büyür; politika ağı karmaşıklaşır, veri verimliliği düşer. BIST-30 için state boyutu ~800'ü aşmaktadır. S&P500 ölçeğinde pratik limitler zorlanır.
2. **Heterojen rol eksikliği:** Tek ajan, "bankacılık hissesi" ile "teknoloji hissesi" için aynı karar mekanizmasını kullanır. Sektörel dinamik farklılıkları (yüksek vs düşük beta, faiz duyarlılığı) tek politikada çözülmesi zor örüntülerdir.
3. **Yorumlanabilirlik:** Tek ajanın neden belirli bir hisseye ağırlık verdiği, SHAP-tipi feature importance analizleriyle sınırlı biçimde açıklanabilir.
4. **İletişim/koordinasyon yok:** Portföy bileşenleri arasındaki korelasyon-kovaryans yapısı sadece observation içinde "çakılı" (implicit) biçimde yer alır. Varlıklar arası haber akışının açık modellenmesi yoktur.

### 1.3 Multi-Agent Yaklaşımın Değeri

Finansal piyasalar doğası gereği **çoklu-ajan** sistemlerdir: her varlığın fiyatını, farklı bilgi setlerine sahip farklı katılımcılar biçimlendirir. Bu yüzden her varlığı veya sektörü bir "yapay ajan" olarak modellemek yapay olarak dayatılmış değil, doğal bir soyutlamadır.

Bu soyutlamanın getirdiği araştırma olanakları:

- **Ölçeklenebilirlik:** Her ajan kendi lokal observation'ı üzerinde öğrenir; sistem N'e göre daha iyi ölçeklenebilir.
- **Heterojenlik:** Ajan mimarileri sektöre / varlık türüne göre farklılaştırılabilir (parameter sharing esnekliği).
- **İletişim boyutu:** Ajanlar arası mesajlaşma — hangi bilgiyi, kiminle, ne zaman paylaşacaklarını — öğrenilebilir bir problem haline gelir. Bu, tez için özgün katkı potansiyeli en yüksek boyuttur.
- **Yorumlanabilirlik:** İletişim graph'ı / attention ağırlıkları ajanlar arası "bilgi akışını" görselleştirir ve sektörel bulaşmaların (contagion) veya yayılma etkilerinin modellenmesini mümkün kılar.

### 1.4 Tez Araştırma Sorusu

> **"BIST-30 portföy yönetiminde her varlığın/sektörün bağımsız bir ajan olarak modellendiği ve ajanlar arasında tahmin-destekli iletişimin bulunduğu çoklu-ajan pekiştirmeli öğrenme çerçevesi, tek-ajanlı alternatiflere göre (i) risk-ayarlı getiri, (ii) piyasa rejimlerine uyum ve (iii) yorumlanabilirlik açılarından ne düzeyde üstünlük sağlar?"**

Bu soru üç alt-soruya ayrılır:

- **AQ1 (Mimari):** Hangi iletişim mimarisi (no-comm / attention-based / graph-based / learned protocol) BIST-30'da en iyi performansı verir?
- **AQ2 (Tahmin sinyali):** Ensemble tahmin sisteminin (güven, yön, regime) iletişime beslenmesi koordinasyonu nasıl etkiler?
- **AQ3 (Rejim-duyarlılık):** İletişimin değeri boğa / ayı / yatay rejimlerde nasıl değişir?

---

## 2. Literatür Durumu

2024-2026 arasında yapılan taramaya göre MARL-finans çalışmalarını dört eksende gruplandırabiliriz. Her eksende literatürün doğruları, eksikleri ve **tezimizin boşluğu nereden doldurduğu** açıklanmıştır.

### 2.1 MARL Portföy Yönetimi (Doğrudan İlgili Alan)

**Güncel örnekler:**

- **Cooperative MARL for ETFs** ([Collaborative MARL Model, 2025] ACM): Her ETF bir ajan; reward ajanın kendi getirisi + diğer ajanların getirileri ile karışır. Basit ortak-ödül (shared-reward) tasarımı; iletişim yok, ajanlar birbirinden habersizdir.
- **Graph Attention Heterogeneous MARL** ([Nature Sci Reports, 2025]): Üç özelleşmiş heterojen ajan (risk değerlendirme, getiri tahmini, piyasa algısı). GAT ile varlık ilişkileri modelleniyor. **Bu çalışma bize yakın.** Ancak ajanlar varlık-bazlı değil rol-bazlı; iletişim mimarisi karşılaştırılması yapılmamış.
- **Hierarchical MARL Portfolio** ([Complex & Intelligent Systems, 2025]): Hiyerarşik DQN, her alt-ağ bir varlık alt-kümesinden sorumlu. Eylem uzayını küçültmede etkili; ancak ajanlar arası açık iletişim yok.
- **StockMARL** ([Zou & Siebers, EMSS 2025]): Çok-ajanlı hisse seçim çerçevesi; daha pazarlama amaçlı bir "adlandırma" içeriyor, iletişim tasarımı sığ.
- **HARLF** ([arXiv 2507.18560, 2025]): Hiyerarşik RL + LLM sentiment; base agents → meta-agents → super-agent. İletişim değil, hiyerarşik agregasyon.
- **RL-BHRP** ([arXiv 2508.11856, 2025]): RL ile Bayesian Hierarchical Risk Parity; sektör → varlık iki seviyeli allocation. İstatistiksel temelli, ama iletişim kavramı yok.
- **Data-Aware MARL with Adaptive Risk Control** ([Preprints.org 202511.0447]): Dinamik portfolio optimization için MARL; risk kontrolü odaklı.

**Literatürdeki ortak eksiklikler:**

- **İletişim mimarilerinin karşılaştırması yok.** Genellikle tek bir mimari seçilip baseline'a karşı test ediliyor.
- **Tahmin sistemi ile entegrasyon sığ.** Çoğu çalışmada observation'a ek feature olarak eklenir; iletişime beslenmez.
- **Gelişen piyasalar üzerinde sınırlı çalışma.** Büyük çoğunluk S&P500 / ETF üzerinde; BIST, NIFTY, Bovespa gibi EM örnekleri az.
- **Rejim-bağımlı analiz yok.** İletişimin faydasının boğa/ayı ayrımında nasıl değiştiği araştırılmamış.

**Tezin boşluğu:** Bu belgedeki AQ1, AQ2, AQ3'ün üçü birden literatürde birlikte ele alınmamış. BIST-30 zemininde tahmin sistemiyle entegre iletişim mimarisi karşılaştırması **özgün katkı** oluşturur.

### 2.2 Multi-Agent Communication Mimarileri

MARL'ta iletişim tasarımı aktif bir araştırma alanıdır. Temel yaklaşımlar:

| Mimari | Temel Fikir | Tarihsel Referans | Güçlü Yanı | Zayıf Yanı |
|---|---|---|---|---|
| **CommNet** | Her ajan diğer ajanların gizli durumlarının ortalamasını mesaj olarak kullanır | Sukhbaatar et al. 2016 | Basit, diferansiyellenebilir | Seçici değil, gürültüye açık |
| **DIAL / RIAL** | Diferansiyellenebilir öğrenilmiş discrete mesajlar | Foerster et al. 2016 | Öğrenilmiş protokol | Eğitimi zor, küçük ajan sayısı |
| **TarMAC** | Sender-receiver signature; query-key-value ile hedef seçimi | Das et al. 2019 | Seçici, attention-benzeri | Transformer kadar zengin değil |
| **MAAC / G2ANet** | Attention-based centralized critic, seçici mesaj okuma | Iqbal & Sha 2019 | İyi ölçeklenir | Centralized critic gerektirir |
| **GNN-tabanlı (GAT / GraphSAGE)** | Portföy bir graph; mesajlar düğümler arası akar | Hamilton et al. 2017; Veličković et al. 2018 | Yapısal bilgi (sektör) entegrasyonu | Graph kurulumu zor |
| **MACTAS** | Self-attention modülü, CTDE uyumlu | 2025, arXiv 2508.13661 | Modern, uyarlamalı | Yeni, az benchmark |
| **Mean-Field MARL** | Büyük N için "ortalama alan" yaklaşımı | Yang et al. 2018 | N>100 için idealdir | Bireysel etkileşim kaybolur |

**2024-2026 eğilimleri:**
- TarMAC türevleri + Transformer mimarileri yaygınlaşmıştır. CTDE paradigması standart haline gelmiştir.
- "Öğrenilen iletişim" (DIAL/RIAL tarzı) hâlâ zordur; finansta neredeyse kimse bunu denememiştir — **bu bir fırsat**.
- Scalable communication (2023 IJCAI, "Scalable Communication for MARL") — 30+ ajan için iletişim mimarisi optimizasyonu aktif alan.

**Tezimiz için imalar:**
- Üç mimari karşılaştırması realist: **no-comm baseline (IPPO), attention-based (TarMAC-benzeri), GNN-based (GAT)**. Mean-field BIST-30 için çok büyük; learned discrete protokol (DIAL) riskli / zor eğitim.
- Attention mimarilerinin yorumlanabilirliği (attention weight görselleştirme) tez için değerli bir nicel analiz dayanağı.

### 2.3 Prediction-Augmented RL (Hibrit Tahmin + RL)

**Güncel örnekler:**

- **LSTM + DQN hibrit:** LSTM zamansal bağımlılıkları yakalar, DQN action üretir ([Nature Sci Reports, 2025]).
- **Ensemble DRL Trading** ([Yang et al., 2021; güncel versiyon arXiv 2511.12120, 2025]): A2C + PPO + DDPG ensemble; her ajan farklı rejime daha iyi uyum gösterir. **Bizim ensemble'ımıza benziyor ama RL tarafında.**
- **HAELT** ([arXiv 2506.13981, 2025]): Transformer tabanlı hybrid attentive ensemble; high-frequency stock forecasting.
- **Data Augmentation via RL + Prediction:** Tahmin çıktısı eğitim verisini genişletmekte kullanılır (S&P500'de %137.94 kümülatif getiri raporlanmış).
- **HARLF** (yukarıda): LLM-sentiment hiyerarşik RL entegrasyonu.

**Literatürdeki ortak eksiklikler:**
- Tahmin çıktısı RL observation'a "ham özellik" olarak eklenir. Tahmin güveni, yönü ve rejimi iletişim kanalına beslenmiyor.
- Tahmin sistemlerinin açıklanabilirliği (SHAP) RL kararlarına nadiren bağlanıyor.

**Tezimiz için imalar:**
- Mevcut ensemble sistemimiz (5 model + stacking + ICEEMDAN + TATS + SHAP) **zaten güçlü bir ön-çalışma**. Tezin hikâyesini *"prediction-first MARL"* olarak konumlandırmak hem teknik hem de yayın stratejisi açısından avantaj sağlar.
- Tahmin güveninin iletişime beslenmesi ("güvenim düşükse diğer ajanlara daha çok kulak verirmin") **özgün bir katkı yoludur** ve literatürde doğrudan karşılığı bulunmamıştır.

### 2.4 Gelişen Piyasalar / BIST Özelinde DRL

**Güncel örnekler:**

- **BIST endeks tahmini** ([Journal of Emerging Economies and Policy, 2024]): ANN ile %58 yön doğruluğu. Klasik ML, RL değil.
- **CNN-DRL emerging market** ([Decision Science Letters, 2025]): Emerging market DRL; buy&hold'a göre %67 yön doğruluğu iyileşmesi. BIST değil genel EM.
- **NIFTY 50 DRL Framework** ([MDPI AI, 2025]): Hint piyasası için DRL; replike edilebilecek kurulum.

**Literatürdeki ortak eksiklikler:**
- BIST üzerinde **multi-agent DRL çalışması neredeyse yok**. Tez bu boşluğu dolduruyor.
- Türk piyasasının özgün dinamikleri (yüksek enflasyon, TL volatilitesi, TCMB faiz politikaları) genellikle ihmal edilir. Bizim makro veri pipeline'ımız (EVDS + VIX + US10Y + DXY) bu boşluğu kısmen dolduruyor.

**Tezimiz için imalar:**
- "EM-spesifik MARL portföy yönetimi" başlı başına bir katkı. Tezin ikinci ana unsuru: sadece mimari değil, **BIST'e özel rejim-duyarlı davranış analizi**.

---

## 3. Tasarım Seçenekleri ve Değerlendirme

Her tasarım ekseni için seçenekleri doğruluk, eksiklik, yapılabilirlik ve proje mantığıyla uyumu açısından değerlendiriyoruz. **Bu aşamada karar verilmiyor;** seçenekler donduruluyor, fizibilite çalışması sonucuna göre Milestone 2 başlangıcında tercihler netleşecek.

### 3.1 Ajan Granülaritesi: Kaç ajan, ne ajan?

#### Seçenek A — Her hisse bir ajan (N=30)

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | Literatürde yaygın; "natural abstraction" savunusu kolay yapılır |
| **Finansal mantık** | Her hisse farklı şirket, farklı fundamental; bağımsız politika mantıklı |
| **Teknik fizibilite** | ⚠️ **Orta-zor.** 30 ajan, CTDE critic ağının girdi boyutunu şişirir (joint action=30 continuous). Parameter sharing şart |
| **Eğitim maliyeti** | Yüksek. Her ajan ayrı rollout'a ihtiyaç duymaz ama joint training karmaşık |
| **Özgünlük** | Düşük — literatürde çok örnek var |
| **Risk** | Convergence instability; 30 ajanlı MARL'de non-stationarity ciddi sorun |

#### Seçenek B — Her sektör bir ajan (~8 ajan: Bank, Enerji, Sanayi, Perakende, Holding, Telekom, Teknoloji, Diğer)

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | **Özgün.** Sektör-bazlı MARL literatürde az ([MDPI Electronics 2025] bir örnek) |
| **Finansal mantık** | Çok güçlü. Sektörler aynı makro şoka benzer tepki verir; doğal gruplama |
| **Teknik fizibilite** | ✅ **En uygun.** 8 ajan makul, CTDE critic bu boyutta sağlam eğitilir |
| **Eğitim maliyeti** | Orta |
| **Özgünlük** | Yüksek — sektör iletişimi finansta anlamlı (bulaşma, rotasyon) |
| **Risk** | Her sektörün içindeki varlık dağılımı kararı alt-mekanizma gerektirir (allocator) |

#### Seçenek C — Hiyerarşik (Sektör allocator üstte + Hisse selector altta)

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | Feudal RL / hierarchical MARL; 2025'te popüler ([HARLF, RL-BHRP]) |
| **Finansal mantık** | Çok güçlü. Gerçek portföy yöneticisinin karar hiyerarşisine benzer |
| **Teknik fizibilite** | ⚠️ **Zor.** İki seviyeli hiyerarşi eğitim karmaşıklığını ciddi artırır. Birinci seviye ödülü propagate etmek zor |
| **Eğitim maliyeti** | Yüksek |
| **Özgünlük** | En yüksek ama risk de en yüksek |
| **Risk** | Tez süresinde bitirilememe riski belirgin |

**Tavsiye (seçenek donduruluyor):** **Seçenek B (sektör-bazlı) + C'nin hafif versiyonu** (sektör içi eşit ağırlık veya risk-parity). Milestone 2 fizibilite sonunda C'ye geçilip geçilmeyeceği kararlaştırılır.

---

### 3.2 İletişim Mimarisi

#### Seçenek 1 — No-Communication Baseline (IPPO / IPPO + Param Sharing)

| Açı | Değerlendirme |
|---|---|
| **Rolü** | **Zorunlu baseline.** İletişim etkisini ölçebilmek için olmalı |
| **Teknik fizibilite** | ✅ Kolay. PettingZoo + RLlib IPPO veya MAPPO (central critic, no comm) hazır |
| **Risk** | Yok |

#### Seçenek 2 — Attention-based Communication (TarMAC-benzeri)

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | Standard hale gelmiş; 2024-2025 literatürünün çoğu buna dayanır |
| **Yapılabilirlik** | ✅ Orta. PyTorch'ta multi-head attention ile yazılabilir; ancak mevcut SB3 stack'ine custom policy gerektirir |
| **Özgünlük** | Orta. Finansa uyarlama boyutuyla özgünlük üretilir |
| **Yorumlanabilirlik** | ✅ **Yüksek.** Attention ağırlıkları görselleştirilir (sektör bulaşma haritası) |

#### Seçenek 3 — Graph Neural Network (GAT / GraphSAGE)

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | 2025'te çok popüler, Nature Sci Reports'ta yayın yapıldı |
| **Finansal mantık** | Sektör graph'ı + korelasyon kenarları ile çok anlamlı |
| **Yapılabilirlik** | ⚠️ **Orta-zor.** PyTorch Geometric kurulumu + dinamik graph tasarımı |
| **Özgünlük** | Yüksek (graph'ın nasıl kurulduğuna bağlı: statik sektör vs dinamik korelasyon) |
| **Risk** | Graph'ın kötü tasarımı sonuçları bozar |

#### Seçenek 4 — Learned Discrete Protocol (DIAL / RIAL)

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | Klasik (2016); finansta uygulanmamış denecek kadar az |
| **Yapılabilirlik** | ❌ **Zor.** Gradient propagation noise channel üzerinden, convergence nadir |
| **Özgünlük** | En yüksek (kimse yapmamış) ama başarı riski yüksek |
| **Risk** | Tez süresinde başarılamama ihtimali ciddi |

**Tavsiye (seçenek donduruluyor):** **1 + 2 + 3 karşılaştırması** tez için optimal. 4'ü "future work" olarak bırakıyoruz. Milestone 2 sonunda 2 ve 3'ten hangisinin daha verimli olduğu fizibilite çıktılarına göre kesinleşir.

---

### 3.3 Ortak Amaç vs Bencil Amaç (Reward Design)

#### Seçenek α — Fully Cooperative (Shared Reward)

Her ajan portföyün toplam Sharpe'ını maksimize eder.

| Açı | Değerlendirme |
|---|---|
| **Finansal mantık** | ✅ Güçlü. Portföy birlik; toplam performans hedefimiz |
| **Credit assignment** | ❌ Zor. Hangi ajan ne kadar katkı yaptığı karışır |
| **Yapılabilirlik** | Orta. CTDE ile yönetilebilir ama sinyal seyrek |

#### Seçenek β — Fully Selfish (Individual Reward)

Her ajan kendi hissesinin getirisini maksimize eder.

| Açı | Değerlendirme |
|---|---|
| **Finansal mantık** | ⚠️ Zayıf. Portföy düzeyinde koordinasyon olmaz; tüm ajanlar pahalı hisseye koşar |
| **Yapılabilirlik** | Kolay ama anlamlı sonuç vermez |
| **Özgünlük** | Düşük |

#### Seçenek γ — Mixed (Ağırlıklı Kombinasyon)

`reward_i = α · individual_return_i + (1-α) · portfolio_sharpe`

| Açı | Değerlendirme |
|---|---|
| **Finansal mantık** | Güçlü. α ile takım-bireysel ödünleşimi ayarlanır |
| **Yapılabilirlik** | ✅ **Önerilen.** Optuna ile α tune edilir |
| **Özgünlük** | Orta; literatürde var ama sistematik ablation yapılmamış |

#### Seçenek δ — Risk-Adjusted Individual + Turnover Penalty

`reward_i = individual_sortino_i − λ · turnover_i − μ · deviation_from_target_weight`

| Açı | Değerlendirme |
|---|---|
| **Finansal mantık** | Çok güçlü; pratik portföy yöneticisinin gerçek amaç fonksiyonuna yakın |
| **Yapılabilirlik** | Orta. Multi-term ödül tune etmesi zor |
| **Özgünlük** | Yüksek (risk-parity hibridine benziyor) |

**Tavsiye:** **γ temel, δ ablation varyantı.** Milestone 3'te karar.

---

### 3.4 Tahmin Sisteminin Entegrasyonu

#### Seçenek I — Ham Feature Olarak (Mevcut yaklaşım)

Tahmin çıktısı (predicted_return, direction, confidence) observation'a ek feature.

| Açı | Değerlendirme |
|---|---|
| **Yapılabilirlik** | ✅ Zaten yapılmış |
| **Özgünlük** | Düşük |

#### Seçenek II — İletişim Kanalına Besleme (YENİ — özgün katkı)

Her ajan kendi tahmin güvenini ve yönünü diğer ajanlara mesaj olarak gönderir; düşük güvenli ajanlar diğerlerine daha çok kulak verir (trust-weighted attention).

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | **Özgün.** Literatürde doğrudan karşılığı yok |
| **Yapılabilirlik** | Orta. Attention weight'i confidence ile modüle etmek standart attention üzerine ek mekanizma |
| **Yayın potansiyeli** | Çok yüksek — tek başına makale konusu |

#### Seçenek III — Meta-Learner → Regime Sinyali

Ensemble'ın meta-learner çıktısı "market regime" indikatörü olarak tüm ajanlara broadcast edilir.

| Açı | Değerlendirme |
|---|---|
| **Finansal mantık** | Güçlü. Boğa/ayı/yatay rejim ortak bilgi |
| **Yapılabilirlik** | ✅ Kolay (mevcut stacking meta-learner yeterli) |
| **Özgünlük** | Orta-yüksek |

**Tavsiye:** **II + III birlikte.** Tezin özgün katkısının çekirdeği. Milestone 3'ün ana uğraşı.

---

### 3.5 Framework ve Altyapı Seçimi

#### Seçenek F1 — Stable-Baselines3 + Custom Multi-Agent Wrapper

Mevcut kodumuz SB3 üzerine kurulu. Multi-agent desteği yok; wrapper yazmak gerekir.

| Açı | Değerlendirme |
|---|---|
| **Yapılabilirlik** | ⚠️ Orta. Custom wrapper sağlam değil; bakım yükü yüksek |
| **Öğrenme eğrisi** | Düşük (SB3 biliniyor) |
| **Mimari sınır** | Attention / GNN custom policy ile entegrasyon zor |

#### Seçenek F2 — PettingZoo + RLlib

| Açı | Değerlendirme |
|---|---|
| **Yapılabilirlik** | ✅ **Önerilen.** PettingZoo API standardı, RLlib MAPPO/MADDPG/QMIX hazır |
| **Öğrenme eğrisi** | Orta. RLlib dokümantasyonu karmaşık ama topluluk destekli |
| **Mimari esneklik** | Yüksek. Custom policy ile attention/GNN entegre edilebilir |
| **Topluluk** | Büyük — 2024-2025'te aktif geliştiriliyor |

#### Seçenek F3 — PettingZoo + MARLlib veya Tianshou

| Açı | Değerlendirme |
|---|---|
| **Yapılabilirlik** | Orta. Daha yeni, daha az belgelenmiş |
| **Öğrenme eğrisi** | Yüksek |
| **Risk** | Framework bırakılırsa kod paslanır |

#### Seçenek F4 — FinRL-Meta Fork

FinRL-Meta finans odaklı ama MARL desteği sınırlı.

| Açı | Değerlendirme |
|---|---|
| **Yapılabilirlik** | Orta. Benchmark uyumluluğu avantaj |
| **Esneklik** | Düşük (finans-spesifik varsayımlar) |

**Tavsiye:** **F2 (PettingZoo + RLlib)**. Mevcut SB3 kodu baseline olarak kalır; yeni MARL mimarisi RLlib üzerine kurulur. Milestone 2'nin ilk haftası bu geçişe ayrılır.

---

## 4. Kesin Kapsam (Scope) ve Kapsam Dışı

Tez kapsamının net olması "dipsiz kuyu" riskini önler. Bu bölüm **savunma ve değerlendirmede** doğrudan atıf yapılacak referanstır.

### 4.1 In-Scope (Yapılacaklar)

**Veri katmanı:**
- BIST-30 günlük OHLCV (2018-2026)
- Makro: TCMB EVDS (faiz, enflasyon) + VIX + US10Y + DXY + USD/TRY + altın
- Fundamental: yfinance temel oranlar
- ICEEMDAN filtrelenmiş fiyat serileri

**Tahmin katmanı:**
- 5-model ensemble (XGB + LGBM + CatBoost + BiLSTM + TFT) + Ridge/XGB meta-learner
- TATS trend düzeltici
- SHAP açıklanabilirlik

**RL katmanı:**
- **Baseline:** Tek-ajan PPO/A2C/TD3 + PSR reward + ATR/Kelly pozisyon (mevcut)
- **Yeni:** Sektör-bazlı (~8 ajan) MARL
- **Yeni:** 3 iletişim mimarisi karşılaştırması: No-comm (IPPO) / Attention / GNN
- **Yeni:** Prediction-augmented communication (confidence-weighted messages)

**Değerlendirme:**
- Sharpe, Sortino, Calmar, Deflated Sharpe Ratio, Turnover, Max Drawdown
- Walk-forward backtest, purge gap + embargo
- Rejim-bazlı ablation (boğa/ayı/yatay)
- İstatistiksel anlamlılık (t-test, Wilcoxon, Diebold-Mariano)

**Reprodüktibilite:**
- Veri snapshot + random seed kilitleme
- Experiment tracker (JSON veya MLflow)
- LaTeX-ready tablolar + figürler

### 4.2 Out-of-Scope (Yapılmayacaklar)

- **Intraday / high-frequency trading** (sadece günlük)
- **Türev ürünler** (opsiyon, vadeli, future)
- **Short selling** (sadece long + flat)
- **Canlı para ile işlem** (sadece backtest)
- **Başka borsalar** (sadece BIST-30; S&P500 replikasyonu "future work")
- **LLM tabanlı haber/sentiment** (cazip ama kapsam dışı; başka tez / makale konusu)
- **Learned discrete communication protocols** (DIAL/RIAL) — risk yüksek, future work
- **Hiyerarşik derin MARL** (feudal RL) — karmaşıklık yüksek, kapsam dışı veya Milestone 2 sonrası kararı

### 4.3 Başarı Kriterleri

Tez "başarılı" sayılabilmek için aşağıdaki çıktıların **en az %80'i** gerçekleşmelidir:

- [ ] Ansari et al. (2024) replikasyonu BIST-30'da başarıyla gösterildi (baseline)
- [ ] 8-sektör MARL çerçevesi implement edildi; no-comm baseline çalışıyor
- [ ] En az 2 iletişim mimarisi (Attention + GNN) karşılaştırıldı
- [ ] Comm-enabled varyant baseline'a göre Sharpe'da **istatistiksel anlamlı (p<0.05)** iyileşme gösterdi
- [ ] Tahmin sistemi iletişime entegre edildi ve ablation yapıldı
- [ ] Rejim-bazlı (boğa/ayı/yatay) analiz rapor edildi
- [ ] Attention ağırlıkları / communication patterns nitel olarak yorumlandı
- [ ] Walk-forward out-of-sample testler tutarlı sonuç verdi
- [ ] LaTeX rapor + figürler tez formatında hazır
- [ ] En az 1 uluslararası makale submit / at least 1 ulusal makale yayında

---

## 5. Teknik Yapılabilirlik Analizi

### 5.1 Hesaplama Maliyeti

| Bileşen | Tahmini eğitim süresi (RTX 4060 8GB) | Risk |
|---|---|---|
| Mevcut tek-ajan baseline (50K step, PPO) | ~30 dk | Düşük |
| Ensemble prediction (5 model, walk-forward) | ~2 saat / fold | Orta (bellek) |
| MARL IPPO (8 ajan, 200K step) | ~1-2 saat (RLlib) | Orta |
| MARL + Attention comm | ~3-4 saat | Orta |
| MARL + GNN comm | ~4-6 saat | Orta-yüksek (PyG) |
| MARL + Prediction-augmented | ~4-8 saat | Yüksek (integration complexity) |

**Toplam**: Her iletişim mimarisi × 3 seed × 5 ablation config = ~150-300 saat net eğitim. Bu RTX 4060 için **yapılabilir** (2-4 haftaya dağıtılır).

### 5.2 Bellek ve Veri

- BIST-30 × 8 yıl × OHLCV + tüm feature'lar → ~200MB (yönetilebilir)
- Replay buffer (DDPG tipi) 8 ajan × 1M transition → ~3-5GB RAM (yönetilebilir)
- Model checkpoint'leri: 8 ajan × 3 mimari × 3 seed = 72 checkpoint → ~5-10GB disk

### 5.3 Framework Geçişi Maliyeti

SB3 → PettingZoo+RLlib geçişinde:
- Environment wrapper yeniden yazılmalı (mevcut `env/trading_env.py` tek-ajan)
- Reward sistemi ajan-başı düzenlenmeli
- Experiment tracking yeniden kurulmalı (RLlib kendi TensorBoard log formatını kullanır)

**Tahmini süre:** 1-2 hafta (Milestone 2'nin ilk iki haftası).

### 5.4 Risk Haritası

| Risk | Olasılık | Etkisi | Azaltma |
|---|---|---|---|
| MARL convergence sorunları | Yüksek | Yüksek | Parameter sharing, baseline'dan warm-start, küçük learning rate |
| PyG / RLlib version uyumsuzluğu | Orta | Orta | requirements.txt dondur; Docker opsiyonu |
| 8 sektör ajanı → kötü sonuç | Orta | Yüksek | 2. seçenek olarak 4-5 sektör grubu |
| Prediction-comm entegrasyonu kırılgan | Orta | Yüksek | Önce IBT feature olarak test, sonra comm'a taşı |
| Tez süresi aşımı | Yüksek | Orta | Milestone'larda "checkpoint kararı" — gecikirse kapsam daralt |
| Out-of-sample regime shift (2024-2026 makroekonomik) | Yüksek | Orta | Açıkça raporla, limitation olarak kabul et |

---

## 6. Akademik Çıktı Planı

### 6.1 Tez

**Başlık önerisi (çalışma):**
> *"Çoklu-Ajan Pekiştirmeli Öğrenmede Tahmin-Destekli İletişim: BIST-30 Portföy Yönetimi İçin Mimari Karşılaştırma"*

**Bölüm Taslakları:**
1. Giriş ve Motivasyon
2. Literatür Taraması (MARL + Portfolio + Comm + Prediction)
3. Metodoloji (Veri + Feature + Ensemble Prediction + RL)
4. Baseline: Ansari Replikasyonu + Faz 2/3 Sistem
5. Sektör-Bazlı MARL Çerçevesi
6. İletişim Mimarileri Karşılaştırması
7. Prediction-Augmented Communication
8. Rejim-Bazlı Ablation
9. Sonuçlar ve Gelecek Çalışmalar

### 6.2 Makale Hedefleri

Tezin modüler yapısı **3-4 bağımsız makale** çıkartma potansiyeline sahip:

**Makale 1 (Ana — flagship):**
- Başlık: *"Prediction-Augmented Communicating Agents for Portfolio Management in Emerging Markets"*
- Hedef: *Expert Systems with Applications* (Q1) veya *Applied Soft Computing* (Q1)
- Süre: Milestone 3 sonu taslak, Milestone 4'te submit

**Makale 2 (Erken-yayın):**
- Başlık: *"Ensemble Prediction with ICEEMDAN Denoising and TATS Correction for Reinforcement Learning Trading: A BIST-30 Study"*
- Hedef: *Neurocomputing* (Q1) veya *Knowledge-Based Systems* (Q1)
- Süre: Milestone 1 sonunda taslak hazır olabilir — **kısa yoldan ilk yayın**

**Makale 3 (Orta):**
- Başlık: *"Trust-Weighted Message Passing in Multi-Agent Portfolio Management"*
- Hedef: *Information Sciences* (Q1) veya *IEEE TNNLS* (Q1)
- Süre: Milestone 3 sonu

**Makale 4 (Ulusal, düşük risk):**
- Başlık: *"Çoklu-Ajan Pekiştirmeli Öğrenme ile BIST-30 Portföy Yönetimi: Karşılaştırmalı Bir Analiz"*
- Hedef: *Politeknik Dergisi* / *Gazi MMF* / *Hacettepe Ekonomi*
- Süre: Milestone 4

---

## 7. Milestone Haritası

### Milestone 0 — Zemin ve Baseline Dondurma *(0-3 hafta)*

**Amaç:** Mevcut kodu "kanonik baseline" olarak dondurmak.

- [ ] `docs/development/roadmap.md` dosyasını gerçek duruma güncelle
- [ ] Veri snapshot'ı üret (parquet, 2018-2026-Q1)
- [ ] Random seed + hyperparam dondurulmuş "baseline config" hazırla
- [ ] Faz 1/2/3 sistem tam BIST-30 üzerinde eğitilsin (30 hisse, 200K step, 3 seed)
- [ ] Sonuçları `results/baseline/` altına versiyonla
- [ ] Ablation: Ensemble ile ve ensemble olmadan (DT vs DTF)
- [ ] **Çıktı:** Tez Bölüm 4 taslağı + Makale 2 ham materyali

### Milestone 1 — Literatür + Tez Önerisi Resmileştirme *(3-5 hafta)*

**Amaç:** Akademik çerçeveyi formalize etmek, danışmanla hizalamak.

- [ ] Literatür taramasını genişlet (en az 60 makale özet tablosu)
- [ ] Tez önerisi resmi dokümanı (danışman/enstitü formatında) hazırla
- [ ] Bu belgeyi (`vision-and-scope.md`) danışmanla gözden geçir
- [ ] Makale 2 ilk taslağı (Faz 2/3 + baseline sonuçları) yaz
- [ ] **Çıktı:** Tez Bölüm 1-3 taslağı + Makale 2 submission-ready

### Milestone 2 — MARL Framework + Baseline Mimarisi *(5-11 hafta)*

**Amaç:** SB3 → PettingZoo+RLlib geçişi + no-comm baseline.

- [ ] PettingZoo uyumlu `env/marl_trading_env.py` yaz
- [ ] 8-sektör ajan tasarımı (AQ1 Seçenek B)
- [ ] IPPO (no-comm) baseline eğit (3 seed × 5 config)
- [ ] Sonuçları tek-ajan baseline ile karşılaştır
- [ ] Attention-based communication mimarisini implement et (TarMAC-benzeri)
- [ ] İlk comm vs no-comm karşılaştırması
- [ ] **Çıktı:** Tez Bölüm 5 taslağı + Makale 1 için çekirdek deneyler

### Milestone 3 — Prediction-Augmented Communication *(11-16 hafta)*

**Amaç:** Tezin özgün katkısı.

- [ ] GNN-based communication (PyG + GAT) implement et
- [ ] Prediction confidence → message weight entegrasyonu
- [ ] Meta-learner regime signal broadcast
- [ ] 4-yol karşılaştırma: no-comm / attention / GNN / prediction-augmented
- [ ] Rejim-bazlı ablation (boğa/ayı/yatay)
- [ ] Attention visualization ve yorumlama
- [ ] **Çıktı:** Tez Bölüm 6-8 taslağı + Makale 1 & 3 ham materyali

### Milestone 4 — Yazım, Submit, Savunma Hazırlık *(16-20 hafta)*

**Amaç:** Paketleme.

- [ ] Tez tüm bölümler tamamlandı
- [ ] Makale 1 submit
- [ ] Makale 3 taslak (mümkünse submit)
- [ ] Makale 4 (Türkçe, ulusal) taslak
- [ ] LaTeX + figure paketi final
- [ ] Savunma slaytları + demo
- [ ] **Çıktı:** Tez teslim + makale submissionlar

---

## 8. Belge Güncelleme Protokolü

Bu belge **yaşayan bir dokümandır**:

- Her milestone başında revize edilir.
- Seçenek dondurmaları burada belgelenir (örn: "Milestone 2 sonrası Seçenek B kesinleşti").
- Kapsam daralma / genişleme kararları burada işlenir.
- Versiyon geçmişi belgenin altına eklenir.

### Versiyon Geçmişi
- **v0.1 (2026-04-21):** İlk taslak. Literatür taraması, seçenek değerlendirmeleri, kapsam ve milestone haritası.

---

## 9. Kaynakça (Çalışma Sürümü)

### MARL-Portfolio
- ACM ICCBDA 2025 — *Collaborative Multi-Agent RL Model for Portfolio Management*
- Nature Sci Reports 2025 — *Graph Attention-Based Heterogeneous Multi-Agent DRL for Adaptive Portfolio*
- Complex & Intelligent Systems 2025 — *Hierarchical DRL Multi-Agent Portfolio Optimization*
- arXiv 2507.18560 — *HARLF: Hierarchical RL + LLM Sentiment*
- arXiv 2508.11856 — *RL-BHRP: Bayesian Hierarchical Risk Parity RL*
- Preprints 202511.0447 — *Data-Aware MARL + Adaptive Risk Control*
- MDPI Electronics 2025 — *DRL Multi-Source Sector Rotation Portfolio*

### MARL-Communication
- Sukhbaatar et al. 2016 — *Learning Multiagent Communication with Backpropagation* (CommNet)
- Foerster et al. 2016 — *Learning to Communicate with Deep Multi-Agent RL* (DIAL/RIAL)
- Das et al. 2019 — *TarMAC: Targeted Multi-Agent Communication*
- Iqbal & Sha 2019 — *Actor-Attention-Critic for Multi-Agent RL* (MAAC)
- arXiv 2508.13661 — *MACTAS: Self-Attention Module for Inter-Agent Communication*
- IJCAI 2023 — *Scalable Communication for Multi-Agent RL*
- Neurocomputing 2024 — *Explicit Teammate Modeling and Targeted Informative Communication*

### Prediction + RL
- Ansari et al. 2024 — *A Multifaceted Approach to Stock Market Trading Using RL* (ana referans)
- Nature Sci Reports 2025 — *DNN + RL for Exchange Rate Forecasting*
- arXiv 2511.12120 — *DRL for Automated Stock Trading: Ensemble Strategy*
- arXiv 2506.13981 — *HAELT: Hybrid Attentive Ensemble Learning Transformer*
- Yang et al. 2021 — *Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy*

### BIST / Emerging Markets
- Journal of Emerging Economies and Policy 2024 — *Stock Market Index Prediction: BIST Indices*
- Decision Science Letters 2025 — *Convolutional DRL for Emerging Stock Market*
- MDPI AI 2025 — *DRL Framework for NIFTY 50 Index Trading*

### Framework / Tools
- Terry et al. 2021 — *PettingZoo: Gym for Multi-Agent RL*
- AI4Finance Foundation — *FinRL: Financial Reinforcement Learning*
- RLlib Documentation (Ray Project)

*(Tam bibtex ve DOI listesi Milestone 1 içinde hazırlanacaktır.)*
