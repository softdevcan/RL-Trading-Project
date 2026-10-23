# Tez Vizyonu ve Kapsam Belgesi

**Proje:** RL Trading — BIST-30 Çoklu-Ajan Portföy Yönetim Sistemi
**Belge Türü:** Vizyon, kapsam ve teknik fizibilite analizi
**Durum:** Taslak v0.3 (2026-10-07 — seçilmiş kaynakçaya göre güncellendi: [literature-table.md](literature-table.md); bkz. §8 Versiyon Geçmişi)
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

Derin pekiştirmeli öğrenme (DRL) portföy yönetiminde model-serbest yaklaşım sunar: ajan, piyasa dinamiklerini önceden modellemeden, getiri-risk ödünleşimini doğrudan deneyim üzerinden öğrenir. Alan son yıllarda hızla olgunlaştı (derlemeler: Hambly vd., Mathematical Finance 2023; Pippas vd., ACM Computing Surveys 2025). FinRL gibi açık kaynak çatıları benchmark altyapısı sunuyor (Liu vd., ICAIF 2021).

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

> **Kaynak (v0.3):** Bu bölümün kaynak tabanı [literature-table.md](literature-table.md). Orada 169 kayıt değerlendirildi, **148'i seçildi** (85 SCIE/SSCI dergi, 41 üst düzey konferans). Seçim kuralı, kademeler, elenen kayıtlar ve tez/kod bileşenlerini kaynaklara bağlayan kapsam matrisi de o belgede. Künyeler Crossref / arXiv / OpenAlex / Semantic Scholar ile, dergi indeksleri Clarivate MJL ile doğrulandı. v0.1'deki dört bulunamayan kaynak ve kaynaksız bir sayısal iddia v0.2'de çıkarılmıştı. v0.3'te seçim kuralıyla elenen kayıtlar bu bölümden de çıkarıldı (ör. SAMP-HDRL, RL-BHRP, HARLF, MACTAS, ExpoComm ve hakemsiz derlemeler). İleri atıf taramasıyla 2025–2026'dan yeni öncüller eklendi.

MARL-finans çalışmalarını dört eksende gruplandırıyoruz. Her eksende literatürün doğruları, eksikleri ve **tezimizin boşluğunun bugünkü (daraltılmış) sınırı** açıklanmıştır. Boşluk iddiaları G1–G5 kodlarıyla izlenir (§2.5).

### 2.1 MARL Portföy Yönetimi (Doğrudan İlgili Alan)

**Ajan granülaritesine göre güncel örnekler:**

- **Varlık başına ajan:** MAPS (Lee vd., IJCAI 2020) ve MSPM (Huang & Tanaka, PLOS ONE 2022) temel örnekler. Hao vd. (ACM CIBDA 2025) her ajanı bir varlığın kararından sorumlu tutup bireysel ve küresel getiriyi karıştıran ödülü inceliyor. Bu, bizim `α · bireysel + (1−α) · portföy` ödül tasarımımızın **doğrudan öncülü**; ajanlar arası öğrenilmiş iletişim yok.
- **Rol tabanlı ajan:** Zhang (Scientific Reports 2025) risk, getiri tahmini ve piyasa algısı ajanlarını GAT ile birleştiriyor. Ajan güveni son tahmin isabetinin üstel hareketli ortalamasından hesaplanıp **nihai portföy önerilerinin konsensüsünde** kullanılıyor (tam metinle doğrulandı). Tohum sayısı ve iletişimsiz MAPPO karşılaştırması yok; kaynakçasında bir atıf yanlış yazarlarla verilmiş. Diğer örnekler: MASA (Li vd., AAMAS 2024) iki işbirlikçi ajan + piyasa gözlemcisi; MARS (Chen vd., AAAI 2026) güvenlik eleştirmeniyle ayrışan risk profilli heterojen ajan topluluğu + meta denetleyici.
- **Varlık sınıfı / grup tabanlı:** MAMA (Kim & Lee, IEEE Access 2025) varlık sınıfı başına ajan, GNN tabanlı durum temsili, faiz oranı ve kavram kayması.
- **Hiyerarşik / sektör düzeyli:** Sun vd. (Complex & Intelligent Systems 2025) hiyerarşik DRL ile çok-ajanlı dinamik portföy; MPLS (Khemlichi vd., MMEP 2025) **sektör dağıtan üst ajan** + sektör içi alt ajanlar + GNN; Yan vd. (Electronics 2025) DRL ile sektör rotasyonu.
- **2026'da yayımlanan yakın çalışmalar (tam metinleri henüz okunmadı):**
  - Hung & Huang (LNCS, IEA/AIE 2026): çok hisseli portföyde öz-dikkatli MARL.
  - Yuan vd. (*Engineering Applications of AI* 2026): uyarlamalı çok-ajanlı DRL hisse alım-satım karar destek sistemi.
  - Öztürk (SAUCIS 2026): PPO, A2C, DDPG, TD3 ve SAC'ı entropiden türetilen güven ve geçmiş performansla ağırlıklandıran bir topluluk. Burada güven **karar birleştirmede** kullanılıyor, ajanlar arası iletişimde değil (özetten).

  **G1 ve G2 ifadeleri Hung & Huang ile Yuan vd.'nin tam metni okunmadan kesinleşmemeli.**
- **Finansta iletişim karşılaştırması:** Fang vd. (KDD 2023) çoklu emir yürütmede CommNet, TarMAC ve iletişimsiz yöntemleri aynı düzenekte karşılaştırıp öğrenilmiş çok turlu niyet iletişimi öneriyor. Görev portföy değil emir yürütme; GNN kolu yok.
- **Derlemeler:** Rout vd. (Discover Computing 2026), tek ajandan çok ajana 2018–2025 dönemini PRISMA yöntemiyle derliyor. Pippas vd. (ACM Computing Surveys 2025), nicel finansta 167 RL yayınını çok-ajanlı çözümler dahil eleştirel olarak değerlendiriyor. Hambly vd. (Mathematical Finance 2023) finansta RL'in kuramsal ve uygulamalı genel çerçevesini veriyor.

**Literatürdeki ortak eksiklikler (doğrulanmış hâli):**

- **Portföy yönetiminde iletişim, deney değişkeni olarak izole edilmemiş.** Finanstaki tek sistematik karşılaştırma emir yürütmede (Fang vd. 2023). Hung & Huang (2026) portföyde öz-dikkatli MARL öneriyor; iletişimsiz bir kola karşı izole karşılaştırma yapıp yapmadığı tam metinden kontrol edilmeli.
- **Tahmin entegrasyonu çoğunlukla gözlem yoluyla.** Güven bilgisi kullanıldığında karar konsensüsüne (Zhang 2025) ya da topluluk ağırlığına (Öztürk 2026) giriyor, mesaj dikkatine değil.
- **Gelişmekte olan hisse piyasasında MARL-portföy çalışması bulunamadı.** Örneklerin büyük çoğunluğu ABD ve Çin A-hisse.
- **Metodolojik kalite düşük-orta:** çoklu tohum nadir, işlem maliyeti seviyesi çoğu çalışmada belirsiz, iletişimsiz baseline çoğu zaman yok.

**Tezin boşluğu (daraltılmış):** Hiçbir boşluk tek bir çalışmayla kapanmadı, ancak v0.1'deki geniş ifadeler savunulamaz hâle geldi. Savunulabilir çekirdek: *portföy yönetiminde iletişimsiz / dikkat / GAT iletişiminin eşit bütçe ve çoklu tohumla karşılaştırılması, harici kalibre tahmin güveninin mesaj dikkatine beslenmesi ve bunun bir gelişmekte olan piyasada (BIST-30) rejime koşullu ölçülmesi.* Ayrıntılı G1–G5 tablosu §2.5'te.

### 2.2 Multi-Agent Communication Mimarileri

MARL'ta iletişim tasarımı aktif bir araştırma alanıdır. Temel yaklaşımlar:

| Mimari | Temel Fikir | Kaynak | Güçlü Yanı | Zayıf Yanı |
|---|---|---|---|---|
| **CommNet** | Her ajan diğer ajanların gizli durumlarının ortalamasını mesaj olarak kullanır | Sukhbaatar vd., NIPS 2016 | Basit, diferansiyellenebilir | Seçici değil, gürültüye açık |
| **DIAL / RIAL** | Diferansiyellenebilir öğrenilmiş ayrık mesajlar | Foerster vd., NIPS 2016 | Öğrenilmiş protokol | Eğitimi zor, küçük ajan sayısı |
| **TarMAC** | İmza (anahtar) – sorgu – değer ile hedefli mesaj | Das vd., ICML 2019; dikkat mekanizması Vaswani vd., NeurIPS 2017 | Seçici, dikkat tabanlı | Transformer kadar zengin değil |
| **MAAC** | Dikkatli merkezi eleştirmen, seçici mesaj okuma | Iqbal & Sha, ICML 2019 | İyi ölçeklenir | Merkezi eleştirmen gerektirir |
| **GNN-tabanlı (GAT)** | Portföy bir graf; mesajlar düğümler arası akar | Veličković vd., ICLR 2018 | Yapısal bilgi (sektör) entegrasyonu | Graf kurulumu zor |
| **MAGIC** | Zamanlayıcı + GAT mesaj işleyici | Niu, Paleja & Gombolay, AAMAS 2021 | GAT kolumuzun MARL'daki doğrudan referansı | — |
| **CommFormer** | İletişim grafiği politika ile birlikte öğrenilir | Hu vd., ICLR 2024; dergi sürümü IEEE TKDE 2026 | TarMAC, MAGIC, CommNet'i geçiyor | Finans dışı |
| **Mean-Field MARL** | Büyük N için "ortalama alan" yaklaşımı | Yang vd., ICML 2018 | N>100 için idealdir | Bireysel etkileşim kaybolur — **kapsam dışı** |

**Güncel eğilimler:**
- Dikkat/transformer tabanlı iletişim ve öğrenilen iletişim grafikleri yaygınlaştı (CommFormer). Merkezi eğitim–dağıtık yürütme (CTDE; Lowe vd., NeurIPS 2017) standart paradigma.
- Kapsamlı bir derleme (Zhu, Dastani & Wang, *JAAMAS* 2024) iletişimli çok-ajanlı derin RL'i dokuz tasarım boyutunda sınıflandırıyor ve bu boyutların kombinasyonlarını açık tasarım alanı olarak öneriyor. AQ1'in tek bir "en iyi" mimariyi varsaymayan deneysel bir karşılaştırma olarak kurulması bu sınıflamayla uyumlu.
- 8–30 ajan ölçeğinde tam dikkatin maliyeti ihmal edilebilir; seyreltilmiş iletişim yöntemlerine gerek yok (tasarım değerlendirmesi).
- Öğrenilen ayrık iletişim (DIAL/RIAL) hâlâ zor; finansta denenmiş bir örneğe rastlanmadı.

**Belirsizlik ve güvenle ağırlıklı iletişim (G2'nin konumlandığı alt hat):**
- **IMMAC** (Sun vd., AAMAS 2021, genişletilmiş özet): dikkat, ajanın içsel belirsizlik/merak değerinden türetiliyor ("communicate what surprises you").
- **ADMAC** (Yu vd., AAAI 2024): alıcı, gelen mesajın güvenilirliğini öğrenilmiş bir kestiriciyle tahmin edip etkisini azaltıyor; amaç düşmanca mesajlara karşı savunma. Kestiricinin etiketleri ayrık eylem varsayıyor ("mesaj en iyi eylemi öneriyor mu"). Sürekli portföy eyleminde ancak **uyarlanarak** kullanılabilir ve "ADMAC'tan esinlenen" diye adlandırılmalı.
- **DebUnc** (Yoffe vd., Findings of EMNLP 2025): LLM tartışmasında softmax sonrası dikkat ağırlıklarını göndericinin ters belirsizliğiyle çarpıp yeniden normalize ediyor. **Bu işlem, dikkat logitine log-güven eklemekle matematiksel olarak eşdeğerdir:** softmax(ℓⱼ + log cⱼ) ∝ cⱼ·exp(ℓⱼ). G2'nin "log-güven önseli" kısmı bu yüzden tek başına yeni değil (Faz 9 planı Y7).
- **CAROC** (Lin & Zeng, *Int. J. General Systems* 2025): akranların güvenilirliğini ölçen metriklerle seçici iletişim, ödül tahmini ve bilgi entropisi düzenlileştirmesi (özetten). Güvenilirlik etkileşim geçmişinden türetiliyor; harici bir tahminci yok.
- **Finansta güven:** Akhmedov vd. (Mathematics 2026) RL işlem ajanında beş hafif güven kestirimiyle **ödülü** modüle ediyor. Öztürk (2026) güveni **topluluk ağırlığında** kullanıyor. İkisinde de güven iletişime girmiyor.
- **Sürüleşme riski:** Güven ağırlıklı birleştirme, yanlış ama kendinden emin bir ajanın diğerlerini sürüklemesine yol açabilir. Bu, tezde bir **tasarım varsayımı** olarak ele alınır: güven ağırlığında kırpma ya da entropi geçidi (Faz 9 M5).
- **Ortak nokta:** Güven ya ajanın kendi iç durumundan ya da alıcının yerel gözleminden veya etkileşim geçmişinden öğreniliyor. **Harici, denetimli ve kalibre bir tahmincinin güveni** bu çalışmaların hiçbirinde yok. Kalibrasyonun ölçümü için Guo vd. (ICML 2017) ve Naeini vd. (AAAI 2015, ECE) temel kaynaklar.

**Tezimiz için imalar:**
- Üç mimari karşılaştırması gerçekçi: **iletişimsiz baseline (IPPO; de Witt vd. 2020 / MAPPO; Yu vd., NeurIPS 2022), dikkat tabanlı (TarMAC benzeri), GNN tabanlı (GAT)**. Mean-field BIST-30 için gereksiz; öğrenilen ayrık protokol (DIAL) riskli.
- Dikkat ağırlıklarının görselleştirilmesi nicel bir yorumlanabilirlik dayanağı; ancak dikkatin açıklama sayılabilmesi için tek biçimli dikkat baseline'ı ve tohumlar arası varyans testleri gerekir (Jain & Wallace, NAACL 2019; Wiegreffe & Pinter, EMNLP 2019).

### 2.3 Prediction-Augmented RL (Hibrit Tahmin + RL)

**Güncel örnekler (tahminin RL'e giriş yoluna göre):**

- **Gözlem yolu (en yaygın):** Ansari vd. (IEEE Access 2024 — ana referans). Aynı ekibin devam çalışması Sattar vd. (IEEE Access 2025) kazanç raporlarını, maksimum düşüş ödülünü ve haber duyarlılığını PPO/A2C/DDPG ile birleştiriyor. Başlıktaki **RMS = Reports + Max drawdown + Sentiment**; tez-final.pdf'teki "Root Mean Square" yorumu yanlış. Diğer örnekler: StockFormer (Gao vd., IJCAI 2023, öngörü kodlama + aktör-eleştirmen); DeepClair (Choi vd., CIKM 2024, önceden eğitilmiş transformer tahminci RL portföy seçimine LoRA ile ince ayarla aktarılıyor).
- **Belirsizliğe duyarlı tahmin-ve-dağıt (RL değil):** RTS-PnO (Lyu vd., KDD 2025) tahmin belirsizliğini uyarlanabilir biçimde kalibre edip uçtan uca dağıtım hedefine bağlıyor.
- **Teorik dayanak:** Zhang, Zhao & Zomaya (NeurIPS 2025): tahminler hatalı olsa bile performans alt sınırını koruyan tahmin-destekli çevrimiçi portföy seçimi.
- **Konsensüs yolu:** Zhang (Sci Rep 2025) — getiri tahmin ajanının güven aralıkları ajan konsensüsüne giriyor, mesaj dikkatine değil.
- **RL tarafında topluluk:** Yang vd. (ICAIF 2020, DOI 10.1145/3383455.3422540) A2C + PPO + DDPG topluluğu ve onun çerçeveye dönüşmüş hâli FinRL (Liu vd., ICAIF 2021). arXiv 2511.12120, aynı ICAIF 2020 makalesinin 2025'te yüklenmiş hâlidir; 2020 olarak atıf yapılmalı.
- **Tek-ajan DRL'in finanstaki temel çalışmaları:** Moody & Saffell (IEEE TNN 2001, diferansiyel Sharpe); Deng vd. (IEEE TNNLS 2017); Jiang vd. (2017, EIIE); Zhang, Zohren & Roberts (J. Financial Data Science 2020, maliyet duyarlılığı); Théate & Ernst (ESWA 2021).

**Literatürdeki ortak eksiklikler:**
- Tahmin çıktısı çoğunlukla gözleme "ham özellik" olarak ekleniyor; tahmin güveninin **ajanlar arası mesaj dikkatine** girdiği bir çalışma bulunamadı.
- Tahmin katmanının RL performansına katkısı nadiren ayrı bir ablation ile raporlanıyor. AQ2 için dört koşullu ablation (tahminsiz / yalnız gözlem / yalnız mesaj / ikisi birden) bu boş hücreyi dolduruyor.
- Tahmin sistemlerinin açıklanabilirliği (SHAP; Lundberg & Lee, NeurIPS 2017; Lundberg vd., Nature Machine Intelligence 2020) RL kararlarına nadiren bağlanıyor.

**Tahmin katmanının kendisi (Makale 2 bağlamı):**
- **TATS özgün değil:** Trend-Adjusted Time Series modeli Kazemdehbashi (2026, arXiv 2601.12706) tarafından önerildi. Yöntem, ikili bir trend sınıflandırıcısıyla LSTM/Bi-LSTM tahminini düzeltiyor; makalede XGBoost dahil yedi sınıflandırıcı denenmiş. `prediction/tats.py` repoya 2026-03-27'de girdi; makale 2026-01-19'da yayımlandı. **Atıf zorunludur; TATS tezde ve makalede uyarlanan yöntem olarak sunulur, katkı olarak değil.**
- **Ayrıştırma sızıntısı yerleşik bir eleştiri:**
  - Finans verisinde: Carnelossi Furlaneto vd. (ESWA 2017), EEMD'nin kazancının sızıntı giderilince kaybolduğunu gösteriyor. Xiong vd. (*Applied Intelligence* 2025) başlığıyla, hisse tahmininde **tek seferlik** gürültü gidermenin tahminleri abarttığını söylüyor; tam metin okunmadı. **Makale 2'nin en yakın finans öncülü.**
  - Sızıntısız stratejiler: kayan pencereli CEEMDAN (Zhang, Peng & Song, *Computational Economics* 2024, finans) ve SW/STMP/MTMP-EMD (Yang vd., Sci Rep 2024 — veri su kalitesi, **yöntem kaynağı**).
  - Genel çerçeve: sızıntı taksonomisi (Kapoor & Narayanan, *Patterns* 2023) ve tahmin değerlendirme rehberi (Hewamalage vd., DMKD 2023).
- **Ayrıştırmanın kökeni:** EMD (Huang vd., Proc. R. Soc. A 1998) → EEMD (Wu & Huang 2009) → CEEMDAN (Torres vd., ICASSP 2011) → ICEEMDAN (Colominas vd., BSPC 2014).
- **Model seçimi:** Finansta ağaç tabanlı ve derin modellerin referans karşılaştırmaları Gu, Kelly & Xiu (RFS 2020) ve Fischer & Krauss (EJOR 2018). GBM + BiLSTM + TFT (Lim vd., IJF 2021) topluluğu ve stacking (Wolpert, Neural Networks 1992) bu çizgide savunulabilir.

**Tezimiz için imalar:**
- Mevcut topluluk (5 model + stacking + ICEEMDAN + TATS + SHAP) **güçlü bir ön çalışma**, ama Makale 2'nin katkı iddiası yeniden kurulmalı (bkz. §6.2): TATS uyarlanmış bir bileşen; ICEEMDAN'ın mevcut uygulaması sızıntılı (bkz. §5.4).
- Tahmin güveninin iletişime beslenmesi hâlâ **özgün katkı yolu**, ancak "literatürde karşılığı yok" denemez. Doğru ifade için bkz. §3.4 Seçenek II ve §2.5 G2.

### 2.4 Gelişen Piyasalar / BIST Özelinde DRL

**BIST'te RL/DRL (doğrulanmış):**

- **Candar & Üstündağ (2022)**, *Intelligent and Fuzzy Systems* (INFUS 2022), LNNS 505. Candar'ın İTÜ YL tezinin (2022) hakemli bildiri sürümü; evren BIST30, PPO/DDPG. **Tek-ajan baseline'ımızın doğrudan öncülü; jüri bilebilir.**
- **Beyhan, Ergin & Eren (2026)**, *Ekonomi Politika ve Finans Araştırmaları Dergisi* 11(1), DOI 10.30784/epfad.1811319. BIST30 bileşenlerinde DDPG kullanıyor; Markowitz, eşit ağırlık ve BIST30 endeksine karşı daha yüksek Sharpe raporluyor. **BIST'te DRL portföy için en güncel hakemli öncül.**
- Altuner & Kilimci (2022), *Turkish J. Electrical Eng. & Computer Sciences* 30(4) (SCIE): derin RL ile fiyat tahmini; topluluk duyarlılığı + bilgi grafiği.
- Karacam & Ulukuş (*Borsa Istanbul Review* 2026): aracı kurumların gizli işlem motivasyonlarını ters RL ile çıkarıyor.

**Türkiye makro bağlamı:** Taşkıran (*Borsa Istanbul Review* 2026), yüksek enflasyon döneminde enflasyon sürprizlerinin hisse getirilerine etkisini inceliyor. TÜFE'yi açıklanma tarihine hizalama gereğinin (S3; genel gerekçe Croushore & Stark, J. Econometrics 2001) Türkiye'ye özgü dayanağı bu.

**Diğer gelişmekte olan piyasalar:** gelişmekte olan piyasaların oynaklık yapısı Bekaert & Harvey (JFE 1997); SET50 örneği Thongkairat & Yamaka (*Mathematics* 2025).

**Ulusal yayınlar:** Eğe (2024, YL tezi), Akyapak & Tunç (ASYU 2025), Sümerli Sarıgül vd. (JOEEP 2024), Ergenç & Aktaş (2025) ve Erdağ vd. (2026) uluslararası metinlerde seçim kuralıyla elendi (literature-table Ek A). Makale 4'te (Türkçe, ulusal) kullanılabilirler.

**Literatürdeki ortak eksiklikler:**
- BIST'te DRL çalışmaları **var** (Candar & Üstündağ 2022; Beyhan vd. 2026); ancak **erişilebilen literatürde BIST üzerinde MARL çalışması bulunamadı**. YÖK Ulusal Tez Merkezi taraması henüz yapılmadı (dinamik arayüz nedeniyle otomatik taranamadı); bu tarama tamamlanmadan "yok" iddiası kullanılmamalı.
- Türkiye'ye özgü dinamikler (enflasyon, kur, TCMB faizi) bazı çalışmalarda girdi olarak kullanılıyor, ama getirilerin reelleştirilmesi ya da risksiz faizin enflasyona göre ayarlanması görülmedi.
- Kalite düşük: tek hisse, kısa test dönemi; bazı yüksek isabet iddiaları sızıntı şüphesi taşıyor.

**Tezimiz için imalar:**
- "EM-spesifik MARL portföy yönetimi" hâlâ bir katkı. Doğru ifade: *"Erişilebilen literatürde BIST-30 üzerinde ilk MARL uygulaması"*. DRL öncülleriyle farklar açıkça yazılmalı: makro ve tahmin özellikleri, walk-forward, çoklu algoritma, çok-ajanlı yapı.
- Bizim makro veri hattımız (EVDS + VIX + US10Y + DXY) Türkiye'ye özgü etkenleri hedefliyor. Ancak mevcut yerel makro önbellekte EVDS serileri fallback sabitlerinde (bkz. §5.4); sonuç üreten veri doğrulanmadan bu iddia kullanılmamalı.
- Tezin ikinci ana unsuru: **BIST'e özel rejime duyarlı davranış analizi** (rejim etiketinin kaynağı Hamilton, Econometrica 1989; portföyde uygulaması Nystrup vd., Quantitative Finance 2018).

### 2.5 Boşluk iddiaları — güncel durum (G1–G5)

| Kod | v0.1 iddiası | Hüküm (Ekim 2026, v0.3) | En yakın çalışmalar | Savunulabilir dar ifade |
|---|---|---|---|---|
| G1 | İletişim mimarilerinin karşılaştırması yok | **Kısmen kapandı** (finansta emir yürütmede yapılmış); portföy + GNN kolu için **açık** — Hung & Huang (2026) tam metni okununca kesinleşir | Fang vd. KDD 2023; Zhang Sci Rep 2025; Hung & Huang 2026 | Portföy yönetiminde iletişimsiz / dikkat / GAT'ın eşit bütçe, çoklu tohum ve aralık tahminiyle ilk karşılaştırması |
| G2 | Tahmin güveninin iletişime beslenmesinin karşılığı yok | **Kısmen kapandı**; v0.3'te **bir kez daha daraldı**: log-güven önseli DebUnc ölçeklemesiyle eşdeğer | Zhang 2025 (konsensüs ağırlığı); ADMAC AAAI 2024; IMMAC AAMAS 2021; DebUnc EMNLP-F 2025; CAROC IJGS 2025; Öztürk 2026 | Harici, denetimli ve **kalibre** bir topluluk tahmincisinin güvenini, sürekli eylemli MARL'da (a) gücü öğrenilen bir dikkat önseli olarak (β·log c; β=1 DebUnc'un özel hâli), (b) alıcı tarafında öz-güven geçidi olarak kullanmak. Kalibrasyonun katkısı ayrı bir kalibresiz-güven koluyla sınanır |
| G3 | Sektör-bazlı MARL az | **Kısmen kapandı** (hiyerarşik sektör dağıtıcılar var); eşdüzeyli + iletişimli sektör ajanları için **açık** | MPLS 2025; MAMA 2025; Sun vd. 2025; Yan vd. 2025 | Merkezi bir dağıtıcı ajan olmadan, öğrenilmiş iletişimle koordine olan eşdüzeyli sektör ajanları |
| G4 | BIST'te multi-agent DRL yok denecek kadar az | DRL için iddia **yanlış ifade edilmiş**; MARL için **açık** (YÖK taraması bekliyor) | Candar & Üstündağ 2022; Beyhan vd. 2026 | Erişilebilen literatürde BIST-30'da ilk MARL uygulaması |
| G5 | İletişimin rejime göre değeri araştırılmamış | **Kısmen kapandı** (LLM ajanlarında, kesitsel piyasa türüne göre) | Shi & Hollifield 2025 (ön baskı) | Öğrenilmiş sayısal iletişimin marjinal katkısının önceden tanımlı **zamansal** rejimlere koşullu ölçümü |

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
| **Akademik konum** | **Orta-yüksek özgünlük.** Hiyerarşik sektör dağıtıcıları var (MPLS 2025; Sun vd. 2025; Yan vd. 2025); eşdüzeyli ve öğrenilmiş iletişimle koordine olan sektör ajanlarına rastlanmadı (§2.5 G3) |
| **Finansal mantık** | Çok güçlü. Sektörler aynı makro şoka benzer tepki verir; doğal gruplama |
| **Teknik fizibilite** | ✅ **En uygun.** 8 ajan makul, CTDE critic bu boyutta sağlam eğitilir |
| **Eğitim maliyeti** | Orta |
| **Özgünlük** | Orta-yüksek — sektör iletişimi finansta anlamlı (bulaşma, rotasyon); MPLS tipi hiyerarşik sektör dağıtıcı **zorunlu baseline** olmalı |
| **Risk** | Her sektörün içindeki varlık dağılımı kararı alt-mekanizma gerektirir (allocator) |

#### Seçenek C — Hiyerarşik (Sektör allocator üstte + Hisse selector altta)

| Açı | Değerlendirme |
|---|---|
| **Akademik konum** | Feudal RL / hiyerarşik MARL; 2025'te popüler (Sun vd., CAIS 2025; MPLS 2025) |
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
| **Akademik konum** | 2025'te popüler (Zhang, Scientific Reports 2025; MPLS 2025; MAMA 2025 GNN kullanıyor); finansta iletişimsiz kola karşı GAT iletişiminin izole karşılaştırması bulunamadı (§2.5 G1) |
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
| **Akademik konum** | **Özgün ama dar ifadeyle (v0.3).** Bileşenleri ayrı ayrı var: güvenle ağırlıklı konsensüs (Zhang 2025), güvenle ağırlıklı DRL topluluğu (Öztürk 2026), alıcı tarafında öğrenilen güvenilirlik (ADMAC 2024), akran güvenilirliğiyle seçici iletişim (CAROC 2025), içsel belirsizlik tabanlı dikkat (IMMAC 2021), belirsizlikle dikkat ölçekleme (DebUnc, Findings of EMNLP 2025, LLM). **DebUnc'un ölçeklemesi dikkat logitine log-güven eklemekle matematiksel olarak eşdeğer**; "log-güven önseli" tek başına katkı değil. Bulunamayan: **harici, kalibre bir topluluk tahmincisinin** güveninin sürekli eylemli MARL'da gücü öğrenilen bir dikkat önseli (β·log c) olarak kullanılması + alıcının öz-güven geçidi + kalibrasyonun katkısının ayrıca sınanması (§2.5 G2) |
| **Yapılabilirlik** | Orta. Attention weight'i confidence ile modüle etmek standart attention üzerine ek mekanizma |
| **Zorunlu kontroller** | (i) Güvenin **kalibrasyonu** raporlanmalı (ECE, güvenilirlik diyagramı) — mekanizmanın değeri buna dayanır; mevcut topluluk güveni yön başlıklarının ortalama olasılığından ya da model oy oranından türetiliyor (`prediction/models/ensemble.py`); ECE fonksiyonu `prediction/evaluator.py::compute_calibration` içinde mevcut ama hiçbir akıştan çağrılmıyor. (ii) Ablation kolları: güvensiz standart dikkat (β=0) / DebUnc ölçeklemesi (β=1 sabit) / β öğrenilen / β öğrenilen + alıcı geçidi / ADMAC'tan esinlenen öğrenilmiş güvenilirlik (sürekli eyleme uyarlanmış) / karıştırılmış (shuffled) güven / **kalibresiz güven**. (iii) Sürüleşme riskine karşı güven ağırlığında kırpma veya entropi geçidi (tasarım varsayımı; Faz 9 planı M5) |
| **Yayın potansiyeli** | Yüksek — dar ve test edilebilir ifadeyle tek başına makale konusu (Makale 3) |

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

**Değerlendirme (v0.2 — literatür doğrulamasıyla güncellendi; uygulama takvimi henüz planlanmadı):**
- Sharpe, Sortino, Calmar, Deflated Sharpe Ratio, Turnover, Max Drawdown
- Walk-forward backtest, purge gap + embargo; **sızıntısız ön işleme** (ayrıştırma yalnızca geçmişle, makro veriler açıklanma tarihine hizalı, geriye doğru doldurma yok)
- Rejim-bazlı ablation (boğa/ayı/yatay). Değerlendirme etiketi önceden ve kural tabanlı tanımlanır; model girdisi olarak HMM kullanılırsa yalnızca filtrelenmiş olasılıklar (Viterbi/düzleştirme geleceği kullanır). Sonuç iki farklı rejim tanımıyla tekrarlanır; rejim başına gün sayısı raporlanır.
- **Tohum protokolü:** ön elemede 3–5, nihai yapılandırmalarda **≥10 tohum**; tohumlar arası dağılım IQM + katmanlı bootstrap güven aralığı + iyileşme olasılığıyla raporlanır (Agarwal vd. 2021; Henderson vd. 2018). Toplam deneme sayısı (HPO denemesi × yapılandırma × tohum) Deflated Sharpe için kaydedilir.
- **İstatistiksel testler soruya göre:**

  | Soru | Test |
  |---|---|
  | Tahmin modeli A, B'den isabetli mi? | Diebold-Mariano (HAC varyans) |
  | Strateji A'nın Sharpe'ı B'den yüksek mi? | Ledoit-Wolf (2008) studentize bootstrap |
  | Çok sayıda yapılandırmadan hangileri en iyi? | Model Confidence Set (Hansen vd. 2011) |
  | Seçilen Sharpe şans eseri mi? | Deflated Sharpe Ratio (Bailey & López de Prado 2014) |
  | Backtest aşırı uyumlu mu? | PBO / CSCV (Bailey vd. 2017) |
  | Çok sayıda karşılaştırma | Benjamini-Hochberg FDR |

  Eşleştirilmemiş t-testi getirilerin otokorelasyonu ve kalın kuyrukları nedeniyle tek başına kullanılmaz; Wilcoxon yalnızca eşleştirilmiş tohumlarda.
- **Baseline'lar:** Al-Tut, XU030 endeksi, periyodik yeniden dengelenen eşit ağırlık (1/N), minimum varyans (Ledoit-Wolf daraltmalı), HRP; öğrenen baseline olarak tek-ajan PPO (Ansari replikasyonu), IPPO, MAPPO ve hiyerarşik sektör dağıtıcı (MPLS tipi). 1/N'nin örneklem dışında optimize edilmiş portföyleri tutarlı biçimde geçmenin zor olduğu klasik bulgu (DeMiguel, Garlappi & Uppal, *Review of Financial Studies* 22:1915–1953, 2009) bu baseline setini zorunlu kılıyor.
- **İşlem maliyeti:** ana senaryo komisyon dahil gidiş-dönüş ~20 bps (mevcut ortam: tek yön %0,1); duyarlılık 0 / 10 / 25 / 50 bps.
- **Getiri birimi ve risksiz faiz:** nominal TL, TÜFE ile reel TL ve USD bazında; Sharpe'ta risksiz faiz politika faizi / TLREF zaman serisi (tek kaynak).
- **Evren:** tarihsel BIST-30 bileşenleri; kullanılamazsa hayatta kalma yanlılığı sınırlılık olarak açıkça beyan edilir.

**Reprodüktibilite:**
- Veri snapshot + random seed kilitleme; **her RL koşumunda tohum kaydı** (mevcut pano eğitim rotası RL modellerine tohum vermiyor — bkz. §5.4)
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
- [ ] Comm-enabled varyant baseline'a göre Sharpe'da **istatistiksel anlamlı (p<0.05)** iyileşme gösterdi — ≥10 tohum, Ledoit-Wolf / bootstrap testiyle ve Deflated Sharpe ile düzeltilmiş olarak
- [ ] Tahmin sistemi iletişime entegre edildi ve ablation yapıldı (güven kalibrasyonu ECE ile raporlandı)
- [ ] Tüm raporlanan sonuçlar sızıntısız veri hattıyla üretildi (§5.4 sızıntı maddeleri kapatıldı ya da sınırlılık olarak beyan edildi)
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

**Toplam (v0.1 tahmini)**: Her iletişim mimarisi × 3 seed × 5 ablation config = ~150-300 saat net eğitim. Bu RTX 4060 için **yapılabilir** (2-4 haftaya dağıtılır).

**v0.2 notu:** §4.1'deki tohum protokolü (nihai yapılandırmalarda ≥10 tohum) ve genişleyen baseline seti bu bütçeyi büyütür; nihai aşamanın tohum maliyeti kabaca 3 kattan fazlasına çıkar. Bütçe, ön elemede 3–5 tohumla yapılandırma sayısı daraltıldıktan sonra Milestone 2 başında yeniden hesaplanmalı (henüz yapılmadı).

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
| **Veri sızıntısı (v0.2'de kodda doğrulandı)** | **Gerçekleşti** | **Yüksek** — sonuçların geçerliliği | Ayrıntı ve kanıt: [literature-review-verification.md §6](literature-review-verification.md). Özet: (1) ICEEMDAN serinin tamamına uygulanıyor ve hedef filtrelenmiş fiyattan türetiliyor (opt-in grup; Makale 2 bu grubu kullanacak); (2) makro seride `bfill` gelecekteki değeri geçmişe yayıyor (varsayılan hatta etkin); (3) TÜFE/ÜFE açıklanma gecikmesi modellenmiyor (yerel önbellekte EVDS verisi yok, gizli); (4) temel oranlar tek anlık değer olarak tüm geçmişe yayılıyor (opt-in). Düzeltme işleri **henüz planlanmadı**; planlanana kadar bu hatla üretilen sonuçlar tez/makalede kullanılmamalı ya da sınırlılık olarak beyan edilmeli |
| **Değerlendirme protokolü açıkları (v0.2)** | Gerçekleşti | Orta-yüksek | RL eğitiminde tohum yok; DM testi HAC'siz ve t-testi eşleştirilmemiş, ikisi de akışa bağlı değil; risksiz faiz üç yerde üç farklı değer (0 / 0,15 / 0,02); 1/N, min-varyans ve HRP baseline'ları yok; sembol evreni bugünkü BIST-30 (hayatta kalma yanlılığı). Hedef protokol §4.1'de |
| **Özgünlük iddiasının daralması (v0.2)** | Gerçekleşti | Orta | İddialar §2.5'teki dar ifadelere çekildi; en yakın çalışmalara (Zhang 2025, Fang vd. 2023, Hung & Huang 2026, ADMAC, DebUnc, CAROC, MPLS, Candar & Üstündağ 2022, Beyhan vd. 2026) açıkça atıf |
| **Makro veri kalitesi (v0.2)** | Gerçekleşti (yerel önbellek) | Orta | Yerel `macro_data.csv`'de `policy_rate` fallback sabiti 50,0; TÜFE/ÜFE sabit 0. "Türkiye'ye özgü makro" iddiası, sonuç üreten verinin `data_quality` bayrağı kontrol edilmeden kullanılmamalı |

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
- Başlık (v0.1): *"Ensemble Prediction with ICEEMDAN Denoising and TATS Correction for Reinforcement Learning Trading: A BIST-30 Study"*
- Hedef: *Neurocomputing* (Q1) veya *Knowledge-Based Systems* (Q1)
- Süre: Milestone 1 sonunda taslak hazır olabilir — **kısa yoldan ilk yayın**
- **v0.2 yeniden konumlandırma (literatür doğrulaması sonrası):** v0.1 başlığındaki iki bileşen artık katkı olarak sunulamaz: TATS Kazemdehbashi (2026) tarafından yayımlanmış bir yöntem; seriyi bütününe ayrıştırmak bilinen bir sızıntıdır (Carnelossi Furlaneto vd., ESWA 2017; hisse tahmininde tek seferlik gürültü giderme için Xiong vd., Applied Intelligence 2025; sızıntısız yöntemler için Zhang, Peng & Song 2024 ve Yang vd. 2024) ve mevcut ICEEMDAN uygulaması bu biçimde. **v0.3:** Xiong vd. (2025) Makale 2'nin en yakın finans öncülü; tam metni okunup katkı farkı ("RL katmanına etkisi + TATS + BIST") buna göre yazılmalı. Önerilen ana soru: *"Sızıntısız (ileri yürüyen) ICEEMDAN altında ayrıştırmanın ve TATS düzeltmesinin tahmin ve RL performansına katkısı ne kalıyor?"* Aynı hat iki koşulda koşturulur (seri bütününe vs yalnızca geçmişle ayrıştırma); fark tahmin katmanında Diebold-Mariano (HAC), RL katmanında Deflated Sharpe ile raporlanır. İkincil katkı: TATS'ın çok varlıklı bir hisse evreninde ve stacking topluluğu üzerinde ilk uygulaması. **Ön koşul:** sızıntısız ayrıştırma ve hedefin ham fiyattan türetilmesi uygulanmış olmalı — bu iş henüz planlanmadı; Aralık 2026 gönderim hedefi bu planlamaya bağlı olarak yeniden değerlendirilmeli.

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

- [x] Literatür taramasını genişlet (en az 60 makale özet tablosu) — **2026-10-06:** 121 kaynaklı tarama (`docs/reference/Literatür Taraması ... (Ekim 2026).md`) + bağımsız doğrulama ([literature-review-verification.md](literature-review-verification.md)); dört uydurma vizyon kaynağı elendi, boşluk iddiaları daraltıldı (§2.5)
- [ ] YÖK Ulusal Tez Merkezi'nde BIST + RL/DRL/MARL tezleri elle taranmalı (otomatik taranamadı; G4 iddiası buna bağlı)
- [ ] Kaynakların Zotero'ya aktarılması ve nihai sayım
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
- **v0.2 (2026-10-06):** Ekim 2026 literatür taraması + bağımsız doğrulama ([literature-review-verification.md](literature-review-verification.md)) işlendi. §2 yeniden yazıldı (doğrulanmış kaynaklar; dört bulunamayan kaynak ve bir kaynaksız sayı çıkarıldı; yeni §2.5 G1–G5 boşluk tablosu). §3.1 B ve §3.4 II özgünlük ifadeleri daraltıldı. §4.1 değerlendirme protokolü (tohum, soruya göre testler, baseline seti, maliyet, risksiz faiz, evren) ve §4.3 başarı kriterleri güncellendi. §5.4'e kodda doğrulanan sızıntı ve protokol açıkları eklendi. §6.2 Makale 2 yeniden konumlandırıldı (TATS atfı, sızıntısız ayrıştırma). §9 kaynakça doğrulanmış künyelerle yeniden yazıldı. **Kod değişikliği yapılmadı; düzeltme işleri planlanmadı.**
- **v0.3 (2026-10-07):** Kaynak tabanı [literature-table.md](literature-table.md)'ye bağlandı (169 kayıt değerlendirildi, 148 seçildi; SCIE/SSCI + CORE A*/A ölçütü, Clarivate MJL kontrolü, ileri atıf taraması, kod/tez bileşenleri için kapsam matrisi). §2 yeniden yazıldı. Değişenler:
  - Elenen kaynaklar çıkarıldı.
  - 2026 öncülleri eklendi: Hung & Huang, Yuan vd., Öztürk, CAROC, Xiong vd., Taşkıran.
  - Künyeler düzeltildi: Candar → Candar & Üstündağ LNNS 2022; DebUnc → Findings of EMNLP 2025; Sattar vd. 2025'e göre "RMS" açılımı.
  - G2, DebUnc eşdeğerliği nedeniyle yeniden daraltıldı (§2.5, §3.4 II).
  - §6.2 Makale 2'ye Xiong vd. 2025 eklendi.
  - §9 seçilmiş kaynakçadan otomatik üretildi.

  Kod değişikliği yok.

---

## 9. Kaynakça (v0.3 — seçilmiş kaynakça)

> Bu liste [literature-table.md](literature-table.md)'deki **148 seçilmiş kayıttan** otomatik üretildi; tablo ile kaynakça aynı veriden gelir. Her girdinin sonundaki etiket kademesini gösterir: `K1` SCIE/SSCI dergi · `K2` üst düzey konferans · `K3` diğer hakemli · `K4` hakemsiz. Kademe K3/K4 olup tutulan her kaydın gerekçesi tablonun §2.2'sinde, elenen 21 kayıt ve yerlerine kullanılacaklar Ek A'da. Künyeler 2026-10-07'de Crossref / arXiv / OpenAlex / Semantic Scholar ile doğrulandı. Nihai liste Zotero'da tutulacak.

### 0. Ana referans

- Ansari, Y., Gillani, S. A., Bukhari, M., Lee, B., Maqsood, M. & Rho, S. (2024). A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning. *IEEE Access*. DOI 10.1109/access.2024.3418510 `[K1 SCIE/SSCI]`

### A. MARL ile portföy yönetimi ve alım-satım

- Chen, J., Li, J. & Wang, G. (2026). MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management. *AAAI 2026*. DOI 10.1609/aaai.v40i24.39095 `[K2 konf. A*/A]`
- Hao, J., Gao, M., Han, Y. & Gao, Q. (2025). Collaborative Multi-Agent Reinforcement Learning Model for Portfolio Management. *ACM CIBDA 2025*. DOI 10.1145/3746709.3746915 `[K3 hakemli]`
- Huang, Z. & Tanaka, F. (2022). MSPM: A modularized and scalable multi-agent reinforcement learning-based system for financial portfolio management. *PLoS ONE*. DOI 10.1371/journal.pone.0263689 `[K1 SCIE/SSCI]`
- Hung, Y. H. & Huang, K. (2026). A Multi-agent Reinforcement Learning Approach with Self-attention for Multi-stock Portfolio Trading. *LNCS — IEA/AIE 2026*. DOI 10.1007/978-981-92-2888-1_19 `[K3 hakemli]`
- Khemlichi, F., Khamlichi, Y. I. & Ali, S. E. B. (2025). MPLS: A Modular Portfolio Learning System for Adaptive Portfolio Optimization. *Mathematical Modelling and Engineering Problems*. DOI 10.18280/mmep.120612 `[K3 hakemli]`
- Kim, S. H. & Lee, K. H. (2025). Multi-Asset Multi-Agent Reinforcement Learning for Portfolio Management. *IEEE Access*. DOI 10.1109/access.2025.3632201 `[K1 SCIE/SSCI]`
- Lee, J., Kim, R., Yi, S. W. & Kang, J. (2020). MAPS: Multi-Agent reinforcement learning-based Portfolio management System. *IJCAI 2020*. DOI 10.24963/ijcai.2020/623 `[K2 konf. A*/A]`
- Li, Z., Tam, V. & Yeung, K. L. (2024). Developing a Multi-agent and Self-adaptive Framework with Deep Reinforcement Learning for Dynamic Portfolio Risk Management. *AAMAS 2024*. DOI 10.65109/bmry8759 `[K2 konf. A*/A]`
- Rout, A. K., Acharya, U. D., Aithal, P. K. & Sharma, A. (2026). Systematic review of reinforcement learning for automated equity portfolio management from single agent to multi agent systems. *Discover Computing*. DOI 10.1007/s10791-026-10336-1 `[K1 SCIE/SSCI]`
- Sattar, A., Sarwar, A., Gillani, S. A., Bukhari, M., Rho, S. & Faseeh, M. (2025). A Novel RMS-Driven Deep Reinforcement Learning for Optimized Portfolio Management in Stock Trading. *IEEE Access*. DOI 10.1109/access.2025.3546099 `[K1 SCIE/SSCI]`
- Sun, R., Xi, Y., Stefanidis, A., Jiang, Z. & Su, J. (2025). A novel multi-agent dynamic portfolio optimization learning system based on hierarchical deep reinforcement learning. *Complex & Intelligent Systems*. DOI 10.1007/s40747-025-01884-y `[K1 SCIE/SSCI]`
- Yan, Y., Zhang, C., An, Y. & Zhang, B. (2025). A Deep-Reinforcement-Learning-Based Multi-Source Information Fusion Portfolio Management Approach via Sector Rotation. *Electronics*. DOI 10.3390/electronics14051036 `[K1 SCIE/SSCI]`
- Yuan, X., Wang, J., Gu, S., Guo, Y., Qi, A. N. W., Li, S. vd. (2026). Adaptive multi-agent stock trading decision support system based on deep reinforcement learning. *Engineering Applications of Artificial Intelligence*. DOI 10.1016/j.engappai.2026.114130 `[K1 SCIE/SSCI]`
- Zhang, B. (2025). Graph attention-based heterogeneous multi-agent deep reinforcement learning for adaptive portfolio optimization. *Scientific Reports*. DOI 10.1038/s41598-025-32408-w `[K1 SCIE/SSCI]`
- Öztürk, C. (2026). Multi-Agent Deep Reinforcement Learning with Dynamic Portfolio Weighting: A Novel Approach to Algorithmic Trading. *Sakarya University Journal of Computer and Information Sciences*. DOI 10.35377/saucis...1825313 `[K3 hakemli]`

### B. MARL iletişimi ve güven/belirsizlik

- Das, A., Gervet, T., Romoff, J., Batra, D., Parikh, D., Rabbat, M. vd. (2019). TarMAC: Targeted Multi-Agent Communication. *ICML 2019*. DOI 10.48550/arxiv.1810.11187 `[K2 konf. A*/A]`
- Fang, Y., Tang, Z., Ren, K., Liu, W., Zhao, L., Bian, J. vd. (2023). Learning Multi-Agent Intention-Aware Communication for Optimal Multi-Order Execution in Finance. *KDD 2023*. DOI 10.1145/3580305.3599856 `[K2 konf. A*/A]`
- Feng, F., He, X., Wang, X., Luo, C., Liu, Y. & Chua, T. S. (2019). Temporal Relational Ranking for Stock Prediction. *ACM Transactions on Information Systems*. DOI 10.1145/3309547 `[K1 SCIE/SSCI]`
- Foerster, J., Assael, Y., de Freitas, N. & Whiteson, S. (2016). Learning to Communicate with Deep Multi-Agent Reinforcement Learning. *NIPS 2016*. DOI 10.48550/arxiv.1605.06676 `[K2 konf. A*/A]`
- Hu, S., Fan, Z., Shen, L., Zhang, Y. & Tao, D. (2026). Communication Learning in Multi-Agent Systems From Graph Modeling Perspective. *IEEE Transactions on Knowledge and Data Engineering*. DOI 10.1109/tkde.2026.3658366 `[K1 SCIE/SSCI]`
- Hu, S., Shen, L., Zhang, Y. & Tao, D. (2024). Learning Multi-Agent Communication from Graph Modeling Perspective. *ICLR 2024*. DOI 10.48550/arxiv.2405.08550 `[K2 konf. A*/A]`
- Iqbal, S. & Sha, F. (2019). Actor-Attention-Critic for Multi-Agent Reinforcement Learning. *ICML 2019*. DOI 10.48550/arxiv.1810.02912 `[K2 konf. A*/A]`
- Lin, Y. Y. & Zeng, X. (2025). Enhancing multi-agent communication through credibility and reward-based optimisation. *International Journal of General Systems*. DOI 10.1080/03081079.2025.2503785 `[K1 SCIE/SSCI]`
- Mantegna, R. N. (1999). Hierarchical structure in financial markets. *The European Physical Journal B*. DOI 10.1007/s100510050929 `[K1 SCIE/SSCI]`
- Niu, Y., Paleja, R. & Gombolay, M. (2021). Multi-Agent Graph-Attention Communication and Teaming. *AAMAS 2021*. DOI 10.65109/yhuo7761 `[K2 konf. A*/A]`
- Shi, J. & Hollifield, B. (2025). Market-Dependent Communication in Multi-Agent Alpha Generation. *arXiv*. DOI 10.48550/arxiv.2511.13614 `[K4 hakemsiz]`
- Sukhbaatar, S., Szlam, A. D. & Fergus, R. (2016). Learning Multiagent Communication with Backpropagation. *NIPS 2016*. DOI 10.48550/arxiv.1605.07736 `[K2 konf. A*/A]`
- Sun, C., Wu, B., Wang, R., Hu, X., Xiao-ya, Y. & Cong, C. (2021). Intrinsic Motivated Multi-Agent Communication. *AAMAS 2021 (genişletilmiş özet)*. DOI 10.65109/qzej9744 `[K3 hakemli]`
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N. vd. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762 `[K2 konf. A*/A]`
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P. & Bengio, Y. (2018). Graph Attention Networks. *ICLR 2018*. https://arxiv.org/abs/1710.10903 `[K2 konf. A*/A]`
- Yang, Y., Luo, R., Li, M., Zhou, M. Q., Zhang, W. & Wang, J. (2018). Mean Field Multi-Agent Reinforcement Learning. *ICML 2018*. DOI 10.48550/arxiv.1802.05438 `[K2 konf. A*/A]`
- Yoffe, L., Amayuelas, A. & Wang, W. Y. (2025). DebUnc: Improving Large Language Model Agent Communication With Uncertainty Metrics. *Findings of EMNLP 2025*. DOI 10.18653/v1/2025.findings-emnlp.1265 `[K2 konf. A*/A]`
- Yu, L., Qiu, Y., Yao, Q., Shen, Y., Zhang, X. & Wang, J. (2024). Robust Communicative Multi-Agent Reinforcement Learning with Active Defense. *AAAI 2024*. DOI 10.1609/aaai.v38i16.29708 `[K2 konf. A*/A]`
- Zhu, C., Dastani, M. M. & Wang, S. (2024). A survey of multi-agent deep reinforcement learning with communication. *Autonomous Agents and Multi-Agent Systems*. DOI 10.1007/s10458-023-09633-6 `[K1 SCIE/SSCI]`

### C. MARL algoritmaları, derlemeler ve araçlar

- Bettini, M., Prorok, A. & Moens, V. (2024). BenchMARL: Benchmarking Multi-Agent Reinforcement Learning. *Journal of Machine Learning Research 25*. DOI 10.48550/arxiv.2312.01472 `[K1 SCIE/SSCI]`
- Christianos, F., Papoudakis, G., Rahman, A. & Albrecht, S. V. (2021). Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing. *ICML 2021*. DOI 10.48550/arxiv.2102.07475 `[K2 konf. A*/A]`
- Fey, M. & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. *ICLR 2019 RLGM Workshop*. DOI 10.48550/arxiv.1903.02428 `[K3 hakemli]`
- Foerster, J., Farquhar, G., Afouras, T., Nardelli, N. & Whiteson, S. (2018). Counterfactual Multi-Agent Policy Gradients. *AAAI 2018*. DOI 10.1609/aaai.v32i1.11794 `[K2 konf. A*/A]`
- Gronauer, S. & Diepold, K. J. (2022). Multi-agent deep reinforcement learning: a survey. *Artificial Intelligence Review*. DOI 10.1007/s10462-021-09996-w `[K1 SCIE/SSCI]`
- Hernández-Leal, P., Kartal, B. & Taylor, M. E. (2019). A survey and critique of multiagent deep reinforcement learning. *Autonomous Agents and Multi-Agent Systems*. DOI 10.1007/s10458-019-09421-1 `[K1 SCIE/SSCI]`
- Liang, E., Liaw, R., Moritz, P., Nishihara, R., Fox, R., Goldberg, K. Y. vd. (2018). RLlib: Abstractions for Distributed Reinforcement Learning. *ICML 2018*. DOI 10.48550/arxiv.1712.09381 `[K2 konf. A*/A]`
- Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P. & Mordatch, I. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. *NeurIPS 2017*. DOI 10.48550/arxiv.1706.02275 `[K2 konf. A*/A]`
- Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J. & Whiteson, S. (2020). Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. *Journal of Machine Learning Research 21*. DOI 10.48550/arxiv.2003.08839 `[K1 SCIE/SSCI]`
- Schroeder de Witt, C., Gupta, T., Makoviichuk, D., Makoviychuk, V., Torr, P. H. S., Sun, M. vd. (2020). Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? *arXiv*. https://arxiv.org/abs/2011.09533 `[K4 hakemsiz]`
- Terry, J. K., Black, B., Grammel, N., Jayakumar, M., Hari, A., Sullivan, R. vd. (2021). PettingZoo: Gym for Multi-Agent Reinforcement Learning. *NeurIPS 2021*. DOI 10.48550/arxiv.2009.14471 `[K2 konf. A*/A]`
- Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A. M. vd. (2022). The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games. *NeurIPS 2022 (Datasets & Benchmarks)*. DOI 10.52202/068431-1787 `[K2 konf. A*/A]`

### D. Tek-ajan DRL: temel algoritmalar, finans uygulamaları, derlemeler

- Deng, Y. C., Bao, F., Kong, Y., Ren, Z. & Dai, Q. (2017). Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. *IEEE Transactions on Neural Networks and Learning Systems*. DOI 10.1109/tnnls.2016.2522401 `[K1 SCIE/SSCI]`
- Fujimoto, S., van Hoof, H. & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML 2018*. DOI 10.48550/arxiv.1802.09477 `[K2 konf. A*/A]`
- Hambly, B. M., Xu, R. & Yang, H. (2023). Recent advances in reinforcement learning in finance. *Mathematical Finance*. DOI 10.1111/mafi.12382 `[K1 SCIE/SSCI]`
- Jiang, Z., Xu, D. & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. *arXiv*. DOI 10.48550/arxiv.1706.10059 `[K4 hakemsiz]`
- Lillicrap, T., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y. vd. (2016). Continuous control with deep reinforcement learning. *ICLR 2016*. DOI 10.48550/arxiv.1509.02971 `[K2 konf. A*/A]`
- Liu, X. Y., Yang, H., Gao, J. & Wang, C. D. (2021). FinRL. *ACM ICAIF 2021*. DOI 10.1145/3490354.3494366 `[K3 hakemli]`
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T. vd. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML 2016*. DOI 10.48550/arxiv.1602.01783 `[K2 konf. A*/A]`
- Pippas, N., Ludvig, E. A. & Turkay, Ç. (2025). The Evolution of Reinforcement Learning in Quantitative Finance: A Survey. *ACM Computing Surveys*. DOI 10.1145/3733714 `[K1 SCIE/SSCI]`
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*. `[K1 SCIE/SSCI]`
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv*. https://arxiv.org/abs/1707.06347 `[K4 hakemsiz]`
- Théate, T. & Ernst, D. (2021). An application of deep reinforcement learning to algorithmic trading. *Expert Systems with Applications*. DOI 10.1016/j.eswa.2021.114632 `[K1 SCIE/SSCI]`
- Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T. vd. (2024). Gymnasium: A Standard Interface for Reinforcement Learning Environments. *arXiv*. DOI 10.48550/arxiv.2407.17032 `[K4 hakemsiz]`
- Wang, F., Li, S., Niu, S., Yang, H., Li, X. & Deng, X. (2025). A Survey on recent advances in reinforcement learning for intelligent investment decision-making optimization. *Expert Systems with Applications*. DOI 10.1016/j.eswa.2025.127540 `[K1 SCIE/SSCI]`

### E. Tahmin-destekli RL

- Choi, D., Kim, J., Gim, M., Lee, J. & Kang, J. (2024). DeepClair: Utilizing Market Forecasts for Effective Portfolio Selection. *CIKM 2024*. DOI 10.1145/3627673.3680008 `[K2 konf. A*/A]`
- Gao, S., Wang, Y. & Yang, X. (2023). StockFormer: Learning Hybrid Trading Machines with Predictive Coding. *IJCAI 2023*. DOI 10.24963/ijcai.2023/530 `[K2 konf. A*/A]`
- Lyu, F., Du, L., Weng, Y., Ying, Q., Xu, Z., Zou, W. vd. (2025). Timing is Important: Risk-aware Fund Allocation based on Time-Series Forecasting. *KDD 2025 (ADS)*. DOI 10.1145/3711896.3737268 `[K2 konf. A*/A]`
- Yang, H., Liu, X. Y., Zhong, S. & Walid, A. (2020). Deep reinforcement learning for automated stock trading. *ACM ICAIF 2020*. DOI 10.1145/3383455.3422540 `[K3 hakemli]`
- Zhang, Z., Zhao, T. & Zomaya, A. Y. (2025). Online Portfolio Selection with ML Predictions. *NeurIPS 2025*. DOI 10.52202/085713-2762 `[K2 konf. A*/A]`

### F. Tahmin modelleri, topluluk, özellik seçimi, HPO, açıklanabilirlik

- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna. *KDD 2019*. DOI 10.1145/3292500.3330701 `[K2 konf. A*/A]`
- Bergmeir, C. & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*. DOI 10.1016/j.ins.2011.12.028 `[K1 SCIE/SSCI]`
- Bergmeir, C., Hyndman, R. J. & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis*. DOI 10.1016/j.csda.2017.11.003 `[K1 SCIE/SSCI]`
- Bergstra, J., Bardenet, R., Bengio, Y. & Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. *NeurIPS 2011*. `[K2 konf. A*/A]`
- Breiman, L. (1996). Stacked regressions. *Machine Learning*. DOI 10.1007/bf00117832 `[K1 SCIE/SSCI]`
- Breiman, L. (2001). Random Forests. *Machine Learning*. DOI 10.1023/a:1010933404324 `[K1 SCIE/SSCI]`
- Chen, T. & Guestrin, C. (2016). XGBoost. *KDD 2016*. DOI 10.1145/2939672.2939785 `[K2 konf. A*/A]`
- Fischer, T. G. & Krauß, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*. DOI 10.1016/j.ejor.2017.11.054 `[K1 SCIE/SSCI]`
- Graves, A. & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. *Neural Networks*. DOI 10.1016/j.neunet.2005.06.042 `[K1 SCIE/SSCI]`
- Gu, S., Kelly, B. & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*. DOI 10.1093/rfs/hhaa009 `[K1 SCIE/SSCI]`
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*. DOI 10.1162/neco.1997.9.8.1735 `[K1 SCIE/SSCI]`
- Hoerl, A. E. & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*. DOI 10.1080/00401706.1970.10488634 `[K1 SCIE/SSCI]`
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W. vd. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*. `[K2 konf. A*/A]`
- Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*. DOI 10.1103/physreve.69.066138 `[K1 SCIE/SSCI]`
- Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*. DOI 10.48550/arxiv.1705.07874 `[K2 konf. A*/A]`
- Lundberg, S., Erion, G., Chen, H., DeGrave, A. J., Prutkin, J. M., Nair, B. G. vd. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*. DOI 10.1038/s42256-019-0138-9 `[K1 SCIE/SSCI]`
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 2018*. DOI 10.48550/arxiv.1706.09516 `[K2 konf. A*/A]`
- Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. *PLoS ONE*. DOI 10.1371/journal.pone.0087357 `[K1 SCIE/SSCI]`
- Schuster, M. & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*. DOI 10.1109/78.650093 `[K1 SCIE/SSCI]`
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*. DOI 10.1016/s0893-6080(05)80023-1 `[K1 SCIE/SSCI]`

### G. Tahmin katmanı, ayrıştırma sızıntısı, kalibrasyon

- Brier, G. W. (1950). VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF PROBABILITY. *Monthly Weather Review*. DOI 10.1175/1520-0493(1950)078<0001:vofeit>2.0.co;2 `[K1 SCIE/SSCI]`
- Carnelossi Furlaneto, D., Oliveira, L. S., Menotti, D. & Cavalcanti, G. D. C. (2017). Bias effect on predicting market trends with EMD. *Expert Systems with Applications*. DOI 10.1016/j.eswa.2017.03.053 `[K1 SCIE/SSCI]`
- Colominas, M. A., Schlotthauer, G. & Torres, M. E. (2014). Improved complete ensemble EMD: A suitable tool for biomedical signal processing. *Biomedical Signal Processing and Control*. DOI 10.1016/j.bspc.2014.06.009 `[K1 SCIE/SSCI]`
- Croushore, D. & Stark, T. (2001). A real-time data set for macroeconomists. *Journal of Econometrics*. DOI 10.1016/s0304-4076(01)00072-0 `[K1 SCIE/SSCI]`
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. DOI 10.48550/arxiv.1706.04599 `[K2 konf. A*/A]`
- Hewamalage, H., Ackermann, K. & Bergmeir, C. (2023). Forecast evaluation for data scientists: common pitfalls and best practices. *Data Mining and Knowledge Discovery 37*. DOI 10.1007/s10618-022-00894-5 `[K1 SCIE/SSCI]`
- Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng, Q. vd. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A Mathematical Physical and Engineering Sciences*. DOI 10.1098/rspa.1998.0193 `[K1 SCIE/SSCI]`
- Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*. DOI 10.1016/j.patter.2023.100804 `[K1 SCIE/SSCI]`
- Kazemdehbashi, S. (2026). Trend-Adjusted Time Series Models with an Application to Gold Price Forecasting. *arXiv*. DOI 10.48550/arxiv.2601.12706 `[K4 hakemsiz]`
- Lim, B. Y., Arık, S. Ö., Loeff, N. & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*. DOI 10.1016/j.ijforecast.2021.03.012 `[K1 SCIE/SSCI]`
- Lones, M. A. (2024). Avoiding common machine learning pitfalls. *Patterns*. DOI 10.1016/j.patter.2024.101046 `[K1 SCIE/SSCI]`
- López de Prado, M. (2018). Advances in Financial Machine Learning. *Kitap — Wiley, ISBN 978-1-119-48208-6*. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 `[K3 hakemli]`
- Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning. *AAAI 2015*. DOI 10.1609/aaai.v29i1.9602 `[K2 konf. A*/A]`
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML 2005*. DOI 10.1145/1102351.1102430 `[K2 konf. A*/A]`
- Pang, J. & Dong, S. X. (2025). Stop unrealistic data preprocessing in wind speed forecasting: approaches and discussions on preventing future data leakage. *Applied Mathematical Modelling*. DOI 10.1016/j.apm.2025.116376 `[K1 SCIE/SSCI]`
- Torres, M. E., Colominas, M. A., Schlotthauer, G. & Flandrin, P. (2011). A complete ensemble empirical mode decomposition with adaptive noise. *ICASSP 2011*. DOI 10.1109/icassp.2011.5947265 `[K3 hakemli]`
- Wu, Z. & Huang, N. E. (2009). ENSEMBLE EMPIRICAL MODE DECOMPOSITION: A NOISE-ASSISTED DATA ANALYSIS METHOD. *Advances in Adaptive Data Analysis*. DOI 10.1142/s1793536909000047 `[K3 hakemli]`
- Xiong, M., Xu, K., Yin, J. & Wang, W. (2025). Overestimated stock market forecasts resulting from the one-time denoising. *Applied Intelligence*. DOI 10.1007/s10489-025-06720-x `[K1 SCIE/SSCI]`
- Yang, X., Li, J. & Jiang, X. (2024). Research on information leakage in time series prediction based on empirical mode decomposition. *Scientific Reports*. DOI 10.1038/s41598-024-80018-9 `[K1 SCIE/SSCI]`
- Zhang, Y., Peng, Y. & Song, Y. (2024). Realized Volatility Forecasting for Stocks and Futures Indices with Rolling CEEMDAN and Machine Learning Models. *Computational Economics*. DOI 10.1007/s10614-024-10732-z `[K1 SCIE/SSCI]`

### H. Ödül, risk ve portföy baseline'ları

- Akhmedov, F., Cho, Y. I., Sattarov, O., Sodikovich, Y. S., Oybek, M., Khujamatov, H. vd. (2026). Confidence-Aware Reward Shaping for Crypto Trading: A Comparative Study of Lightweight Uncertainty Estimation Methods. *Mathematics*. DOI 10.3390/math14122075 `[K1 SCIE/SSCI]`
- Ang, A. & Bekaert, G. (2002). International Asset Allocation With Regime Shifts. *Review of Financial Studies*. DOI 10.1093/rfs/15.4.1137 `[K1 SCIE/SSCI]`
- Ayari, R. (2025). Feature configuration effects in DRL portfolio management: a risk-focused evaluation under market stress. *Quantitative Finance*. DOI 10.1080/14697688.2025.2592822 `[K1 SCIE/SSCI]`
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*. DOI 10.2307/1912559 `[K1 SCIE/SSCI]`
- Kelly, J. L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal*. DOI 10.1002/j.1538-7305.1956.tb03809.x `[K3 hakemli]`
- Kritzman, M. P. & Yuanzhen, L. (2010). Skulls, Financial Turbulence, and Risk Management. *Financial Analysts Journal*. DOI 10.2469/faj.v66.n5.3 `[K1 SCIE/SSCI]`
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis 88*. DOI 10.1016/s0047-259x(03)00096-4 `[K1 SCIE/SSCI]`
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*. DOI 10.2469/faj.v58.n4.2453 `[K1 SCIE/SSCI]`
- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*. DOI 10.3905/jpm.2016.42.4.059 `[K1 SCIE/SSCI]`
- MacLean, L. C., Thorp, E. O. & Ziemba, W. T. (2010). Long-term capital growth: the good and bad properties of the Kelly and fractional Kelly capital growth criteria. *Quantitative Finance*. DOI 10.1080/14697688.2010.506108 `[K1 SCIE/SSCI]`
- Markowitz, H. M. (1952). PORTFOLIO SELECTION*. *The Journal of Finance*. DOI 10.1111/j.1540-6261.1952.tb01525.x `[K1 SCIE/SSCI]`
- Millea, A. & Edalat, A. (2022). Using Deep Reinforcement Learning with Hierarchical Risk Parity for Portfolio Optimization. *International Journal of Financial Studies*. DOI 10.3390/ijfs11010010 `[K1 SCIE/SSCI]`
- Moody, J. E. & Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*. DOI 10.1109/72.935097 `[K1 SCIE/SSCI]`
- Nystrup, P., Madsen, H. & Lindström, E. (2018). Dynamic portfolio optimization across hidden market regimes. *Quantitative Finance*. DOI 10.1080/14697688.2017.1342857 `[K1 SCIE/SSCI]`
- Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*. DOI 10.1086/294846 `[K1 SCIE/SSCI]`
- Sortino, F. A. & Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. *The Journal of Investing*. DOI 10.3905/joi.3.3.59 `[K3 hakemli]`
- Zhang, Z., Zohren, S. & Roberts, S. (2020). Deep Reinforcement Learning for Trading. *The Journal of Financial Data Science*. DOI 10.3905/jfds.2020.1.030 `[K3 hakemli]`

### I. Değerlendirme metodolojisi ve istatistiksel testler

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C. & Bellemare, M. G. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. *NeurIPS 2021*. DOI 10.48550/arxiv.2108.13264 `[K2 konf. A*/A]`
- Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *The Journal of Portfolio Management*. DOI 10.3905/jpm.2014.40.5.094 `[K1 SCIE/SSCI]`
- Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance 20(4)*. DOI 10.21314/jcf.2016.322 `[K1 SCIE/SSCI]`
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society Series B (Statistical Methodology)*. DOI 10.1111/j.2517-6161.1995.tb02031.x `[K1 SCIE/SSCI]`
- Brown, S. J., Goetzmann, W. N., Ibbotson, R. G. & Ross, S. A. (1992). Survivorship Bias in Performance Studies. *Review of Financial Studies*. DOI 10.1093/rfs/5.4.553 `[K1 SCIE/SSCI]`
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient is the 1/ N Portfolio Strategy? *Review of Financial Studies 22*. DOI 10.1093/rfs/hhm075 `[K1 SCIE/SSCI]`
- Diebold, F. X. (2015). Comparing Predictive Accuracy, Twenty Years Later: A Personal Perspective on the Use and Abuse of Diebold–Mariano Tests. *Journal of Business and Economic Statistics*. DOI 10.1080/07350015.2014.983236 `[K1 SCIE/SSCI]`
- Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business and Economic Statistics*. DOI 10.1080/07350015.1995.10524599 `[K1 SCIE/SSCI]`
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business and Economic Statistics*. DOI 10.1198/073500105000000063 `[K1 SCIE/SSCI]`
- Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The Model Confidence Set. *Econometrica*. DOI 10.3982/ecta5771 `[K1 SCIE/SSCI]`
- Harvey, C. R., Liu, Y. & Zhu, C. (2016). … and the Cross-Section of Expected Returns. *Review of Financial Studies*. DOI 10.1093/rfs/hhv059 `[K1 SCIE/SSCI]`
- Harvey, D. I., Leybourne, S. J. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*. DOI 10.1016/s0169-2070(96)00719-4 `[K1 SCIE/SSCI]`
- Henderson, P. B., Islam, R., Bachman, P., Pineau, J., Precup, D. & Meger, D. (2018). Deep Reinforcement Learning That Matters. *AAAI 2018*. DOI 10.1609/aaai.v32i1.11694 `[K2 konf. A*/A]`
- Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*. DOI 10.1016/j.jempfin.2008.03.002 `[K1 SCIE/SSCI]`
- Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*. DOI 10.2307/1913610 `[K1 SCIE/SSCI]`
- Politis, D. N. & Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*. DOI 10.1080/01621459.1994.10476870 `[K1 SCIE/SSCI]`
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*. DOI 10.1111/1468-0262.00152 `[K1 SCIE/SSCI]`

### J. Açıklanabilirlik

- Jain, S. & Wallace, B. (2019). Attention is not Explanation. *NAACL 2019*. DOI 10.18653/v1/n19-1357 `[K2 konf. A*/A]`
- Wiegreffe, S. & Pinter, Y. (2019). Attention is not not Explanation. *EMNLP-IJCNLP 2019*. DOI 10.18653/v1/d19-1002 `[K2 konf. A*/A]`

### K. BIST, Türkiye ve gelişmekte olan piyasalar

- Altuner, A. B. & Kilimci, Z. H. (2022). A novel deep reinforcement learning based stock price prediction using knowledge graph and community aware sentiments. *TURKISH JOURNAL OF ELECTRICAL ENGINEERING & COMPUTER SCIENCES*. DOI 10.55730/1300-0632.3862 `[K1 SCIE/SSCI]`
- Bekaert, G. & Harvey, C. R. (1997). Emerging equity market volatility. *Journal of Financial Economics*. DOI 10.1016/s0304-405x(96)00889-6 `[K1 SCIE/SSCI]`
- Beyhan, H., Ergin, E. & EREN, B. S. (2026). Dynamic Portfolio Optimization with Deep Reinforcement Learning: Evidence from Borsa Istanbul. *Ekonomi Politika ve Finans Arastirmalari Dergisi*. DOI 10.30784/epfad.1811319 `[K3 hakemli]`
- Candar, M. & Üstündağ, A. (2022). Equity Portfolio Optimization Using Reinforcement Learning: Emerging Market Case. *LNNS 505 (INFUS 2022); ayrıca İTÜ YL tezi*. DOI 10.1007/978-3-031-09176-6_16 `[K3 hakemli]`
- Karacam, K. & Ulukuş, M. Y. (2026). Inferring latent trading motivations in Borsa Istanbul: A comparative inverse reinforcement learning approach. *Borsa Istanbul Review*. DOI 10.1016/j.bir.2026.100834 `[K1 SCIE/SSCI]`
- TAŞKIRAN, F. (2026). Inflation surprise and stock returns: Türkiye during a period of high inflation. *Borsa Istanbul Review*. DOI 10.1016/j.bir.2026.100917 `[K1 SCIE/SSCI]`
- Thongkairat, S. & Yamaka, W. (2025). A Combined Algorithm Approach for Optimizing Portfolio Performance in Automated Trading: A Study of SET50 Stocks. *Mathematics*. DOI 10.3390/math13030461 `[K1 SCIE/SSCI]`
