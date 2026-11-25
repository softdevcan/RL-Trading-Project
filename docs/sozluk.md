<!-- URETILDI: scripts/build_guide.py <- dashboard/content/glossary.yaml. ELLE DUZENLEME: YAML'i degistirip `python scripts/build_guide.py` kos. -->

# Sistem Sözlüğü

Projede geçen aşama adları, deney kodları, etiketler ve yöntem terimleri. Her terimde önce **sade** tanım (konuyu bilmeyen biri için), sonra **teknik** tanım (koddaki karşılığıyla) gelir. Sözlük deney sonucu içermez; sonuçlar ilgili tasarım belgelerindedir.

**Terim sayısı:** 181 · **Tek kaynak:** [`dashboard/content/glossary.yaml`](../dashboard/content/glossary.yaml) · **Plan:** [system-guide-plan.md](development/system-guide-plan.md)

## İçindekiler

- **[Sistem ve aşamalar](#kat-sistem)** — [Faz](#faz), [Milestone 0'](#milestone-0p), [M1–M8 (Faz 10 aşamaları)](#m-asamalari), [M8 (nihai koşumlar)](#m8), [İleri veri testi (sanal para simülasyonu)](#ileri-veri-testi), [M3a / M3b](#m3), [M4a / M4b](#m4), [M5](#m5), [K-kararları](#k-kararlari), [K12 yürütme ufku](#k12), [Ön kayıt](#on-kayit), [Hakem paneli](#hakem-paneli), [Geliştirme kapısı](#gelistirme-kapisi), [Deneme kaydı (N)](#deneme-kaydi), [Ansari replikasyonu](#ansari-replikasyonu), [Faz 9 paketleri (A–G, P, R)](#faz-9-paketleri), [Bilinen açıklar (Y kodları)](#bilinen-aciklar), [Protokol açık (protocol_open)](#protokol-acik), [Okuma kuralı (okuma tablosu)](#okuma-kurali), [Durma kuralı](#durma-kurali), [Yasak kurtarma yolları](#yasak-kurtarma), [Ölçüm arşivi](#olcum-arsivi)
- **[Deney kodları ve etiketler](#kat-deney)** — [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [Karıştırılmış kontrol](#karistirilmis-kontrol), [Komşu sinyali (σ)](#komsu-sinyali), [Kol](#kol), [A1 / S1 / A1c / S1c](#kol-kodlari-m3), [N / A / G / Z × X / S](#kol-kodlari-m4), [G0 / GL / GLS / GLM / GD](#kol-kodlari-m5), [GEÇTİ](#etiket-gecti), [KISMİ](#etiket-kismi), [GÖZLENMEDİ](#etiket-gozlenmedi), [PİLOT_İŞARETİ](#etiket-pilot), [Tohum](#tohum), [Gürültü tabanı](#gurultu-tabani), [Gerçek sinyal okumaları (O1–O4)](#okuma-o1-o4), [Keşifsel ve doğrulayıcı bulgu](#kesifsel), [R0-N / SR-N](#kol-kodlari-m6), [T0-N / SO / SO-S / SO-S2 / T0-G / SB / SB-S / SB-S2](#kol-kodlari-m5-gercek), [Sıralama tabanı](#siralama-tabani)
- **[Değerlendirme ve istatistik](#kat-degerlendirme)** — [Sharpe oranı](#sharpe), [Fazla getiri](#fazla-getiri), [Risksiz faiz (A8)](#risksiz-faiz), [D (Sharpe farkı)](#d-farki), [Ortak HAC](#hac), [Tek yönlü p](#tek-yonlu-p), [Holm düzeltmesi](#holm), [Benjamini–Hochberg (BH-FDR)](#bh-fdr), [Güç](#guc), [En küçük tespit edilebilir etki (MDE)](#mde), [Güven aralığı (GA)](#guven-araligi), [Eğitim / doğrulama / test penceresi](#pencereler), [1/N (eşit ağırlık)](#baseline-1n), [Al-tut](#al-tut), [Kural portföyleri (1/N, al-tut, min-varyans, HRP)](#kural-portfoyleri), [Devir](#devir), [İşlem maliyeti (bps)](#islem-maliyeti), [Deflated Sharpe (DSR)](#dsr), [Aşırı uyum olasılığı (PBO)](#pbo), [Model güven kümesi (MCS)](#mcs), [Ledoit–Wolf Sharpe farkı testi](#lw-bootstrap), [Blok bootstrap](#blok-bootstrap), [Diebold–Mariano testi](#diebold-mariano), [Sortino ve Calmar oranları](#sortino-calmar), [Getiri birimi (nominal / reel / USD)](#getiri-birimi), [Tohum gereksinimi](#tohum-gereksinimi), [Tohumlar arası özet (IQM, iyileşme olasılığı)](#tohum-ozetleri)
- **[Çok ajanlı öğrenme](#kat-marl)** — [S-reg (rejim yayın bloğu)](#s-reg), [Dikkat testleri (maskeleme, tek biçimli dikkat, karıştırılmış mesaj)](#dikkat-testleri), [Çok ajanlı pekiştirmeli öğrenme (MARL)](#marl), [Ajan](#ajan), [MAPPO](#mappo), [SARL (tek ajanlı kıyas kolları)](#sarl), [IPPO](#ippo), [Merkezi eğitim, merkezi olmayan yürütme (CTDE)](#ctde), [Eleştirmen](#elestirmen), [Parametre paylaşımı (K24)](#parametre-paylasimi), [İletişim kolları (none / attn / gat / self)](#iletisim-kollari), [Dikkat (attention)](#dikkat), [Graf (sektör + MST)](#graf), [α (ödül karışımı)](#alfa), [Logit sınırı (5 + ln N)](#logit-siniri), [S-obs](#s-obs), [S-msg](#s-msg), [β (güven ağırlığı)](#beta), [Güven geçidi](#guven-gecidi), [Kare (frame)](#kare), [Bölüm (episode)](#bolum), [BenchMARL / TorchRL](#benchmarl), [Muhasebe (kesirli adım)](#muhasebe), [Hedef ağırlık izdüşümü (risk projeksiyonu)](#izdusum), [Emir dönüşümü](#emir-donusumu), [zlogodds güven terimi](#zlogodds), [Başlangıç eylem gürültüsü (init_std)](#init-std), [Deterministik değerlendirme](#deterministik-degerlendirme)
- **[Tek ajan ve genel pekiştirmeli öğrenme](#kat-rl)** — [Politika](#politika), [Gözlem](#gozlem), [Ödül](#odul), [EMA Sharpe düzeyi](#ema-sharpe), [PPO](#ppo), [Eylem](#eylem), [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami), [Gymnasium](#gymnasium), [Stable-Baselines3](#sb3), [A2C](#a2c), [TD3](#td3), [SAC](#sac), [Hiperparametre optimizasyonu](#hpo), [Kayan pencereli bölme (wf-v1)](#walk-forward), [Kontrol noktası seçimi](#kontrol-noktasi-secimi), [Moody–Saffell Dₜ terimi](#moody-saffell), [PSR ödülü](#psr-odulu), [ATR ve Kelly pozisyon boyutlandırma](#atr-kelly), [Türbülans endeksi](#turbulans), [Gözlem sağlığı (obs_health)](#gozlem-sagligi), [Fiyat ölçeği (price_norm)](#fiyat-olcegi), [Eksik veri modu](#eksik-veri-modu)
- **[Tahmin hattı](#kat-tahmin)** — [AUC](#auc), [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [Bilgi kapısı (İ3)](#bilgi-kapisi), [Kalibrasyon (sıcaklık)](#kalibrasyon), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble), [Gradyan artırma modelleri (XGBoost, LightGBM, CatBoost)](#gbm-modelleri), [BiLSTM](#bilstm), [TFT](#tft), [Yön başı](#yon-basi), [Meta-öğrenici](#meta-ogrenici), [Örneklem dışı tahmin (OOF)](#oof), [Çapa (yeniden eğitim noktası)](#capa), [Arındırma ve ambargo (purge / embargo)](#arindirma), [Tahmin ufku](#tahmin-ufku), [p_up_cal (kalibre edilmiş yükseliş olasılığı)](#p-up-cal), [ECE](#ece), [Brier skoru ve BSS](#brier), [TATS](#tats), [Yön isabeti (TDA ve direction_accuracy)](#tda), [ICEEMDAN gürültü ayrıştırma](#iceemdan), [SHAP açıklaması](#shap), [Tahmin izleyici (tracker)](#tahmin-izleyici), [Maske nedeni (mask_reason)](#mask-reason)
- **[Veri](#kat-veri)** — [HMM rejimi (ikinci tanım)](#hmm-rejim), [BIST-30](#bist30), [Panel](#panel), [Listeleme kilidi](#listeleme-kilidi), [Satış engeli (taban açılış)](#satis-engeli), [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [OHLCV](#ohlcv), [yfinance](#yfinance), [Düzeltilmiş fiyat](#duzeltilmis-fiyat), [Teknik göstergeler (MACD, RSI, CCI, ADX)](#teknik-gostergeler), [ADX (Wilder)](#adx), [RL veri kapısı](#veri-kapisi), [Hayalet gün ve bayat hücre](#hayalet-gun), [Eksik hücre](#eksik-hucre), [EVDS (TCMB)](#evds), [Makro veri (yurt içi)](#makro-veri), [Küresel makro (VIX, US10Y, DXY)](#kuresel-makro), [Endeks serileri (XU100, XU030)](#endeks-serileri), [Piyasa rejimi](#rejim), [Sembol evreni (E0)](#evren), [Tarihsel üyelik (E1)](#tarihsel-uyelik), [Sektör grupları (E2)](#sektor-gruplari), [KAP](#kap), [Temel veri (fundamentals)](#temel-veri)
- **[Yazılım ve ürün](#kat-yazilim)** — [Pano (dashboard)](#pano), [Rehber sayfası](#rehber-sayfasi), [API](#api), [Roller (yönetici / kullanıcı / izleyici)](#roller), [Oturum ve çerezler](#oturum), [Çalışma alanı](#calisma-alani), [Denetim kaydı](#denetim-kaydi), [Uzun iş (arka plan görevi)](#uzun-is), [Eğitim süresi tahmini (ETA)](#eta), [Model meta dosyası](#model-meta), [Günlük karar](#gunluk-karar), [Kağıt portföy](#kagit-portfoy), [Tahmin golden testi](#golden-test), [Sürüm (SemVer)](#surum), [Sistem haritası](#sistem-haritasi), [venv-marl (ayrı ortam)](#venv-marl), [Konteyner (Docker)](#konteyner)

<a id="kat-sistem"></a>

## Sistem ve aşamalar

<a id="faz"></a>

### Faz

**Sade:** Projenin büyük geliştirme aşamaları. Her faz bir iş paketini (örneğin tahmin sistemi, kimlik, arayüz, araştırma geçerliliği) tamamlar.

**Teknik:** Tamamlananlar: Faz 1 (kavram kanıtı), 2 (tahmin sistemi), 3 (üretim iyileştirmeleri), 6 (performans), 7 (kimlik ve çok kullanıcı), 8 (arayüz), 9 (araştırma geçerliliği). Faz 10 (çok ajanlı öğrenme) sürüyor. Bileşen ve aşama durumu sistem haritasında tutulur.

- Belgede: [system-map.md](development/system-map.md), [README.md](README.md)
- İlgili: [M1–M8 (Faz 10 aşamaları)](#m-asamalari), [Milestone 0'](#milestone-0p)

<a id="milestone-0p"></a>

### Milestone 0'

**Sade:** Tek ajanlı modellerin, önceden dondurulmuş bir protokolle yeniden koşulduğu kıyas aşaması. Sonradan kural değiştirilemez.

**Teknik:** Protokol m0p-v2 dondurulmuş (git etiketi m0-protocol-v2); koşumlar scripts/run_milestone0.py, analiz scripts/analyze_milestone0.py. Protokol JSON'u ve belgesi değiştirilmez; değişiklik yeni sürüm (m0p-v3) ister.

- Kodda: [`scripts/run_milestone0.py`](../scripts/run_milestone0.py), [`scripts/analyze_milestone0.py`](../scripts/analyze_milestone0.py)
- Belgede: [milestone0-protocol-v2.md](development/milestone0-protocol-v2.md)
- İlgili: [Ön kayıt](#on-kayit), [Faz](#faz)

<a id="m-asamalari"></a>

### M1–M8 (Faz 10 aşamaları)

**Sade:** Çok ajanlı sistemin adım adım kurulduğu aşamalar. Her aşama bir öncekinin üstüne bir parça ekler ve kendi denetimiyle kapanır.

**Teknik:** M1 çerçeve denemesi (BenchMARL/TorchRL), M2 ortam ve muhasebe çekirdeği, M3 öğrenme ve pozitif kontrol, M4 iletişim kolları, M5 tahmin sinyali ve güvenle ağırlıklı iletişim, M6 rejim, M7 yorumlanabilirlik, M8 ön kayıtlı nihai koşumlar (2026-11-22'de okundu). Faz 11 M8 politikalarını dondurup ileri veride izler.

- Belgede: [phase-10-marl-plan.md](development/phase-10-marl-plan.md)
- İlgili: [M3a / M3b](#m3), [M4a / M4b](#m4), [M5](#m5), [M8 (nihai koşumlar)](#m8), [Geliştirme kapısı](#gelistirme-kapisi)

<a id="m8"></a>

### M8 (nihai koşumlar)

**Sade:** Çok ajanlı sistemin, kuralları önceden yazılıp dondurulmuş büyük son sınavı. Altı yılın her biri ayrı test edildi ve sonuç bir kez okundu.

**Teknik:** Ön kayıt v1.0 (etiket m8-prereg-v1): 6 kol × 12 tohum × 6 kat (2021–2026, wf-v1) = 432 koşu, 20 bps maliyet. H1 M-none ↔ S-esm, H2 M-none ↔ S-agr, H3 M-gat ↔ M-self, boş kol negatif kontrol; eşli işaret-çevirme + bootstrap, H2/H3 Holm. Kural portföyleri ölçüte girmez, her raporda yanında verilir. M8 en iyi kolu seçmez (§6.5).

- Kodda: [`scripts/marl/analyze_m8.py`](../scripts/marl/analyze_m8.py)
- Belgede: [phase-10-m8-design.md](development/phase-10-m8-design.md)
- İlgili: [Ön kayıt](#on-kayit), [MAPPO](#mappo), [SARL (tek ajanlı kıyas kolları)](#sarl), [Kural portföyleri (1/N, al-tut, min-varyans, HRP)](#kural-portfoyleri), [İleri veri testi (sanal para simülasyonu)](#ileri-veri-testi), [M1–M8 (Faz 10 aşamaları)](#m-asamalari)

**Uyarı:** M8 "MARL tek ajandan iyi mi" sorusunu yanıtlar, "MARL basit kurallardan iyi mi" sorusunu yanıtlamaz.

<a id="ileri-veri-testi"></a>

### İleri veri testi (sanal para simülasyonu)

**Sade:** Dondurulmuş modellerin, eğitimden sonra gelen gerçek günlerde sanal parayla izlenmesi. Sonuç, yeterli gün birikince bir kez okunur.

**Teknik:** Faz 11: M8'in 2026 katından seçilmiş 72 politika (kol başına 12) ve 4 kural portföyü, 2026-11-23'ten itibaren her akşam aynı defterle ilerletilir; tam 126 seansta ön kayıt ekine göre bir kez okunur. Pano ara görünümdür, hiçbir kural ona göre değişmez. Defterin tek yazarı sunucudur (WRITER).

- Kodda: [`marl/forward_job.py::status`](../marl/forward_job.py), [`marl/forward_views.py::view`](../marl/forward_views.py)
- Belgede: [phase-11-integration-plan.md](development/phase-11-integration-plan.md), [phase-11-m8-forward-annex.md](development/phase-11-m8-forward-annex.md)
- İlgili: [M8 (nihai koşumlar)](#m8), [Ön kayıt](#on-kayit), [Kağıt portföy](#kagit-portfoy), [Okuma kuralı (okuma tablosu)](#okuma-kurali)

<a id="m3"></a>

### M3a / M3b

**Sade:** Çok ajanlı sistemin, cevabı bilerek konmuş sahte bir ipucunu kullanmayı öğrenip öğrenemediğinin sınandığı aşama. M3a maliyetsiz, M3b işlem maliyetli koşulur.

**Teknik:** M3a: A1 (pozitif kontrol bloğu gözlemde) ↔ S1 (aynı bloğun karıştırılmış hâli), maliyetsiz, ön kayıtlı eşiklerle. M3b: aynı karşılaştırma 20 bps gidiş-dönüş maliyetle (A1c ↔ S1c).

- Belgede: [phase-10-m3-design.md](development/phase-10-m3-design.md)
- İlgili: [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [Karıştırılmış kontrol](#karistirilmis-kontrol), [A1 / S1 / A1c / S1c](#kol-kodlari-m3), [İşlem maliyeti (bps)](#islem-maliyeti), [Geliştirme kapısı](#gelistirme-kapisi)

<a id="m4"></a>

### M4a / M4b

**Sade:** İletişimin, bir ajanın elindeki bilgiyi ihtiyacı olan komşusuna taşıyıp taşımadığının sınandığı aşama.

**Teknik:** Komşu sinyali düzeninde. M4a: iletişimsiz (none), tam bağlı dikkat (attn) ve graf maskeli dikkat (gat) kolları. M4b: gat ↔ mimari-eş iletişimsiz kol (self) pilotu. İkisi de geliştirme kapısıdır.

- Kodda: [`marl/comm_models.py::CommPolicy`](../marl/comm_models.py)
- Belgede: [phase-10-m4-design.md](development/phase-10-m4-design.md)
- İlgili: [Komşu sinyali (σ)](#komsu-sinyali), [İletişim kolları (none / attn / gat / self)](#iletisim-kollari), [N / A / G / Z × X / S](#kol-kodlari-m4), [Geliştirme kapısı](#gelistirme-kapisi)

<a id="m5"></a>

### M5

**Sade:** Tahmin sinyalinin ajanlara gözlem ya da mesaj olarak verilmesinin ve mesajları tahminin güvenine göre ağırlıklandırmanın neyi değiştirdiğinin sınandığı aşama.

**Teknik:** Gerçek sinyal kolları (S-obs, S-msg, ikisi birden) sağlamlık sorusunu sınar. Güven mekanizması (güvenle ağırlıklı dikkat, karşılaştırılabilir güven) yalnız sentetik pozitif kontrolde, komşu sinyali düzeninde sınanır. Okuma kuralı sonuçtan önce yazılır.

- Kodda: [`marl/comm_models.py::confidence_term`](../marl/comm_models.py)
- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [S-obs](#s-obs), [S-msg](#s-msg), [β (güven ağırlığı)](#beta), [G0 / GL / GLS / GLM / GD](#kol-kodlari-m5), [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [Komşu sinyali (σ)](#komsu-sinyali)

<a id="k-kararlari"></a>

### K-kararları

**Sade:** Plan belgelerinde numaralanmış ve kullanıcının onayladığı tasarım kararları.

**Teknik:** Faz 9 ve Faz 10 planlarının karar tablolarında "K" ve bir sayıyla anılır (Faz 10 §6; ör. K12 yürütme ufku, K24 parametre paylaşımı). Aynı numara farklı tablolarda farklı bir karar olabilir; hangi tablo olduğu bağlamdan okunur.

- Belgede: [phase-10-marl-plan.md](development/phase-10-marl-plan.md), [phase-9-research-validity-plan.md](development/phase-9-research-validity-plan.md)
- İlgili: [K12 yürütme ufku](#k12), [Parametre paylaşımı (K24)](#parametre-paylasimi)

<a id="k12"></a>

### K12 yürütme ufku

**Sade:** Akşam kapanışta karar ver, ertesi sabah açılışta işlem yap, pozisyonu bir sonraki sabaha kadar tut.

**Teknik:** Karar t kapanışında, yürütme t+1 açılışında; kararın kazandığı getiri Oₜ₊₁ → Oₜ₊₂. Pozitif kontrolün etiketi aynı ufukta: 1[Oₜ₊₂ > Oₜ₊₁]. Eski tek ajan ortamı değişmedi (orada karar ve yürütme aynı kapanışta).

- Kodda: [`scripts/marl/pc_alignment_m2.py::k12_labels`](../scripts/marl/pc_alignment_m2.py)
- Belgede: [phase-10-marl-plan.md](development/phase-10-marl-plan.md)
- İlgili: [K-kararları](#k-kararlari), [Pozitif kontrol (PC060-K12)](#pozitif-kontrol)

<a id="on-kayit"></a>

### Ön kayıt

**Sade:** Deneyi koşmadan önce neyin başarı sayılacağını ve sonucun nasıl okunacağını yazıp dondurmak. Sonuca bakıp kural değiştirmeyi önler.

**Teknik:** Kollar, tohumlar, eşikler ve okuma tablosu koşudan önce commit'lenir ve DONDURULDU diye işaretlenir. Sonradan her değişiklik yeni bir ön kayıt ister ve deneme sayısına eklenir. Yasak kurtarma yolları (ek tohum, eşik değişimi, birincil karşılaştırmanın değişmesi) ön kayıtta listelenir.

- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md), [phase-10-m4-design.md](development/phase-10-m4-design.md)
- İlgili: [Deneme kaydı (N)](#deneme-kaydi), [Hakem paneli](#hakem-paneli), [Tohum](#tohum)

<a id="hakem-paneli"></a>

### Hakem paneli

**Sade:** Bir deney ya da önemli bir kod değişikliği koşulmadan önce onu eleştirel gözle okuyan yapay zekâ hakemleri.

**Teknik:** hakem-finans ve hakem-rl dar kapsamlı uzmanlardır ve pahalı bir işin girdisinde çağrılır; hakem-2 faz kapanışında son kapıdır. Ana oturum ~5 bin token'lık bir inceleme paketi yazar ve uzun ölçümleri kendisi yapar; ikinci okuma yalnız BÜYÜK bulguda yapılır. Maliyet kaydı review-panel.md'de.

- Belgede: [review-panel.md](development/review-panel.md)
- İlgili: [Ön kayıt](#on-kayit)

<a id="gelistirme-kapisi"></a>

### Geliştirme kapısı

**Sade:** Sistemin bir parçası beklendiği gibi çalışıyor mu diye bakılan ara denetim. Tezin bulgusu değildir.

**Teknik:** Faz 10 §3.2.5: gerçek sinyalli koşulardan önce sentetik bir pozitif kontrolle ve pilot büyüklükte (az tohum) koşulur. Sonuç bir sonraki aşamanın tasarımına girdi olur; uzun koşu sistem olgunlaşınca M8'den önce yapılır.

- Belgede: [phase-10-marl-plan.md](development/phase-10-marl-plan.md)
- İlgili: [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [PİLOT_İŞARETİ](#etiket-pilot), [M1–M8 (Faz 10 aşamaları)](#m-asamalari)

<a id="deneme-kaydi"></a>

### Deneme kaydı (N)

**Sade:** Yapılan her deneyin, sonucu ne olursa olsun kaydedildiği defter. Çok deneme yapıp yalnız iyi çıkanı göstermeyi engeller.

**Teknik:** Deneme sayısı N, çoklu deneme düzeltmelerine (ör. Deflated Sharpe) girer. Ön kayıt hangi karşılaştırmaların deneme sayıldığını yazar; değerlendirme protokolünün açık maddeleri kayıtta etiketlenir.

- Kodda: [`evaluation/trial_registry.py`](../evaluation/trial_registry.py)
- İlgili: [Deflated Sharpe (DSR)](#dsr), [Ön kayıt](#on-kayit)

<a id="ansari-replikasyonu"></a>

### Ansari replikasyonu

**Sade:** Projenin başlangıç noktası: yayımlanmış bir tek ajanlı alım-satım çalışmasının BIST-30'a uyarlanmış kopyası. Eski ortam bu çalışmaya sadık tutulur.

**Teknik:** Ansari vd. (2024) metodolojisi; tek ajan ortamı ve gözlem tanımı (MACD, türbülans, hacim) bu çalışmanın tanımında kalır. Bilinen açıklar düzeltilmez, beyan edilir (K26 = a); yeni tasarım MARL ortamındadır.

- Kodda: [`env/trading_env.py::TradingEnv`](../env/trading_env.py)
- Belgede: [vision-and-scope.md](thesis/vision-and-scope.md)
- İlgili: [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami), [Milestone 0'](#milestone-0p), [Faz 9 paketleri (A–G, P, R)](#faz-9-paketleri)
- Kaynak: Ansari vd. (2024)

<a id="faz-9-paketleri"></a>

### Faz 9 paketleri (A–G, P, R)

**Sade:** Araştırma geçerliliği aşamasında yapılan iş paketleri: istatistik, sızıntı kapatma, tekrar üretilebilirlik, baseline, evren, kalibrasyon, ödül tanımı ve tahmin tablosu gibi.

**Teknik:** A değerlendirme çekirdeği, B sızıntı kapatma, C tekrar üretilebilirlik, D baseline ve raporlama, E evren ve sektör, F tahmin güveni ve kalibrasyon, G ödül tanımı; P nokta-zamanlı tahmin tablosu, R1 tek bölme/panel/evren. Faz 9 kapandı (2026-11-15); durum planın §9 alt bölümlerinde.

- Belgede: [phase-9-research-validity-plan.md](development/phase-9-research-validity-plan.md)
- İlgili: [Faz](#faz), [Bilinen açıklar (Y kodları)](#bilinen-aciklar), [P tablosu (pit_v1 / pit_v2)](#pit-tablosu)

<a id="bilinen-aciklar"></a>

### Bilinen açıklar (Y kodları)

**Sade:** Kodda ya da yöntemde bulunmuş ve kayda geçmiş sorunların listesi. Bir açık kapanana kadar ilgili sonuçlar ön sonuç sayılır.

**Teknik:** Faz 9 planında Y ve bir sayıyla anılır (ör. Y13 test ölçeği, Y30 SHAP girdisi, Y31 altın kapanışı); roadmap'te "Bilinen Metodolojik Açıklar". Açık bir madde deneme kaydında protocol_open etiketine dönüşür.

- Belgede: [roadmap.md](development/roadmap.md), [phase-9-research-validity-plan.md](development/phase-9-research-validity-plan.md)
- İlgili: [Protokol açık (protocol_open)](#protokol-acik), [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [Faz 9 paketleri (A–G, P, R)](#faz-9-paketleri)

<a id="protokol-acik"></a>

### Protokol açık (protocol_open)

**Sade:** Bilinen bir açıktan etkilenen deneme kaydına konan etiket. Bu etiketi taşıyan sonuçlar kanıt olarak kullanılmaz.

**Teknik:** Deneme kaydı protokol maddelerini ölçümle taşır (test ölçeği, gösterge bfill, ADX, obs_health, eval_window_unexposed, fundamentals_pit). Açık madde varsa kayıt PROTOCOL_OPEN_TAG alır ve load_trials onu varsayılan dışlar; yine de N'ye sayılır.

- Kodda: [`evaluation/trial_registry.py::PROTOCOL_OPEN_TAG`](../evaluation/trial_registry.py), [`evaluation/trial_registry.py::EXPOSED_EVAL_WINDOWS`](../evaluation/trial_registry.py)
- Belgede: [CLAUDE.md](../evaluation/CLAUDE.md)
- İlgili: [Deneme kaydı (N)](#deneme-kaydi), [Bilinen açıklar (Y kodları)](#bilinen-aciklar), [Gözlem sağlığı (obs_health)](#gozlem-sagligi)

<a id="okuma-kurali"></a>

### Okuma kuralı (okuma tablosu)

**Sade:** Bir deneyin olası her sonucunun ne anlama geleceğini, koşudan önce yazan tablo. Sonuç görüldükten sonra değiştirilmez.

**Teknik:** Ön kaydın parçası; eşikler, test ve öncelik sırası okuma betiğinde (analyze_*.py) koddadır ve kendi kendini sınar (--self-test). Her okuma "yazılmayacak olan" ifadeyle birlikte verilir.

- Kodda: [`scripts/marl/analyze_m5.py`](../scripts/marl/analyze_m5.py)
- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [Ön kayıt](#on-kayit), [Yasak kurtarma yolları](#yasak-kurtarma), [Durma kuralı](#durma-kurali), [Gerçek sinyal okumaları (O1–O4)](#okuma-o1-o4)

<a id="durma-kurali"></a>

### Durma kuralı

**Sade:** Bir deney beklenen eşiği geçemezse sonraki adıma geçilmeyip kararın kullanıcıya bırakılmasını söyleyen kural.

**Teknik:** Ön kayıtta yazılıdır: hangi etikette sonraki aşamanın koşulacağı ya da koşulmayacağı önceden sabitlenir (M3b ve M4a aşama kuralları). Aynı kapı ek tohumla kurtarılmaz; yeni deney yeni ön kayıt ister.

- Belgede: [phase-10-m3-design.md](development/phase-10-m3-design.md), [phase-10-m4-design.md](development/phase-10-m4-design.md)
- İlgili: [Ön kayıt](#on-kayit), [Geliştirme kapısı](#gelistirme-kapisi), [Yasak kurtarma yolları](#yasak-kurtarma)

<a id="yasak-kurtarma"></a>

### Yasak kurtarma yolları

**Sade:** Bir sonuç beğenilmediğinde başvurulması yasaklanan değişiklikler: ek tohum, eşiği değiştirmek, iyi görünen alt kümeyi seçmek gibi.

**Teknik:** Ön kayıtta listelenir (M5 §2.5: hedef ufkunu değiştirmek, özellik kümesi, sembol alt kümesi, ek tohum, eşik/test/birincil karşılaştırma). Gerekirse yeni ön kayıt yazılır ve deneme sayısına eklenir.

- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [Ön kayıt](#on-kayit), [Okuma kuralı (okuma tablosu)](#okuma-kurali), [Deneme kaydı (N)](#deneme-kaydi), [Durma kuralı](#durma-kurali)

<a id="olcum-arsivi"></a>

### Ölçüm arşivi

**Sade:** Metne giren her sayının üretildiği JSON dosyalarının tutulduğu yer. Rakamlar metne elle kopyalanmaz, hangi dosyadan geldiği yazılır.

**Teknik:** docs/development/measurements/<aşama>/; her JSON betik SHA'sını, commit'i ve git_status'u taşır (kirli ağaç iz bırakır, bu yüzden ölçüm betiği commit'lenmeden koşulmaz). Rehber kartları sayılarını bu dosyalardan okur.

- Kodda: [`scripts/build_guide.py`](../scripts/build_guide.py)
- Belgede: [review-panel.md](development/review-panel.md)
- İlgili: [Rehber sayfası](#rehber-sayfasi), [Hakem paneli](#hakem-paneli), [Deneme kaydı (N)](#deneme-kaydi)

<a id="kat-deney"></a>

## Deney kodları ve etiketler

<a id="pozitif-kontrol"></a>

### Pozitif kontrol (PC060-K12)

**Sade:** Cevabı bilerek içine konmuş sahte bir ipucu. Sistemin gerçek bir ipucunu kullanabilip kullanamadığını sınamak içindir; gerçek bir tahmin değildir.

> *Benzetme:* Sınavdan önce öğrenciye kopya kâğıdı verip kopyayı okuyabiliyor mu diye bakmak.

**Teknik:** Etiketten (geleceği bilerek) üretilmiş, hedef AUC'si 0,60 olan sentetik p_up_cal; taban tablonun maskesi ve ölçeği korunur. Etiket K12 ufkunda 1[Oₜ₊₂ > Oₜ₊₁]. Üreteç tohumu bloğu belirler. Yalnız boru hattı kontrolüdür.

- Kodda: [`prediction/pit_table.py::positive_control_table`](../prediction/pit_table.py), [`scripts/marl/build_pred_m3.py`](../scripts/marl/build_pred_m3.py)
- Belgede: [phase-10-m3-design.md](development/phase-10-m3-design.md)
- İlgili: [K12 yürütme ufku](#k12), [AUC](#auc), [Karıştırılmış kontrol](#karistirilmis-kontrol)

**Uyarı:** Bilerek gelecekten bilgi taşır; performans sonucu ya da tez bulgusu olarak raporlanmaz.

<a id="karistirilmis-kontrol"></a>

### Karıştırılmış kontrol

**Sade:** Aynı ipucunun günleri karıştırılmış hâli. Biçimi aynıdır ama doğru güne ait değildir, yani bilgi taşımaz.

**Teknik:** Tahmin bloğunun satırları (günler) üreteç tohumuyla permüte edilir; üç alan ve maske birlikte. Kesit yapısı ve marjinal dağılım korunur, zaman hizası kopar. İpucunun bilgisini girdi boyutundan ve kapasiteden ayırır.

- Kodda: [`scripts/marl/build_pred_m3.py`](../scripts/marl/build_pred_m3.py)
- Belgede: [phase-10-m3-design.md](development/phase-10-m3-design.md)
- İlgili: [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [Kol](#kol)

<a id="komsu-sinyali"></a>

### Komşu sinyali (σ)

**Sade:** Her ajanın elindeki ipucu kendi hissesine değil, bir komşu hisseye aittir. O bilgi ancak iletişimle ihtiyacı olan ajana ulaşır.

**Teknik:** σ, gat grafında sabit noktasız bir komşu eşlemesidir (iki parçalı eşleme, Hopcroft–Karp); ajan j'nin gözlemine σ(j)'nin pozitif kontrol bloğu konur. σ M4a'da donduruldu (neighbor_map.json) ve sonraki üretimler onu okur. Komşular korele olduğundan iletişimsiz politika da komşu sinyalinden kısmi değer alabilir; bu sızıntı koşudan önce ölçülür.

- Kodda: [`scripts/marl/build_pred_m4.py::neighbor_bijection`](../scripts/marl/build_pred_m4.py)
- Belgede: [phase-10-m4-design.md](development/phase-10-m4-design.md)
- İlgili: [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [Graf (sektör + MST)](#graf), [İletişim kolları (none / attn / gat / self)](#iletisim-kollari)

<a id="kol"></a>

### Kol

**Sade:** Bir deneyde karşılaştırılan yapılandırmalardan her biri. İlaç denemesindeki "ilaç grubu" ve "plasebo grubu" gibi.

**Teknik:** Aynı bütçe, aynı tohumlar ve aynı değerlendirmeyle koşulan, yalnız sınanan öğede farklı yapılandırma. Kodlar aşamaya özgüdür (M3: A1/S1; M4: N/A/G/Z × X/S; M5: G0/GL/GLS/GLM/GD).

- İlgili: [A1 / S1 / A1c / S1c](#kol-kodlari-m3), [N / A / G / Z × X / S](#kol-kodlari-m4), [G0 / GL / GLS / GLM / GD](#kol-kodlari-m5), [Tohum](#tohum)

<a id="kol-kodlari-m3"></a>

### A1 / S1 / A1c / S1c

**Sade:** M3'ün kolları: A1 sahte ipucuyla, S1 aynı ipucunun karıştırılmış hâliyle eğitilir. Sondaki "c" işlem maliyetli sürümü belirtir.

**Teknik:** A1: pozitif kontrol bloğu gözlemde; S1: karıştırılmış blok; A0: tahminsiz. "c" son eki 20 bps gidiş-dönüş maliyetli kol (M3b).

- Belgede: [phase-10-m3-design.md](development/phase-10-m3-design.md)
- İlgili: [M3a / M3b](#m3), [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [Karıştırılmış kontrol](#karistirilmis-kontrol), [İşlem maliyeti (bps)](#islem-maliyeti)

<a id="kol-kodlari-m4"></a>

### N / A / G / Z × X / S

**Sade:** M4'ün kolları: ilk harf iletişim türü (N yok, A herkesle, G yalnız komşularla, Z yalnız kendine), ikinci harf ipucu (X gerçek komşu ipucu, S karıştırılmış).

**Teknik:** N = none (BenchMARL MLP), A = attn, G = gat (graf maskeli dikkat), Z = self (aynı katmanlar, dikkat yalnız kendine: mimari-eş iletişimsiz kontrol). X = komşu sinyali bloğu, S = karıştırılmış komşu bloğu.

- Kodda: [`marl/comm_models.py::CommPolicy`](../marl/comm_models.py)
- Belgede: [phase-10-m4-design.md](development/phase-10-m4-design.md)
- İlgili: [M4a / M4b](#m4), [İletişim kolları (none / attn / gat / self)](#iletisim-kollari), [Komşu sinyali (σ)](#komsu-sinyali), [Karıştırılmış kontrol](#karistirilmis-kontrol)

<a id="kol-kodlari-m5"></a>

### G0 / GL / GLS / GLM / GD

**Sade:** M5 mekanizma kolları: G0 güvensiz, GL güvenle ağırlıklı, GLS güveni karıştırılmış, GLM güveni göndericiler arasında bozulmuş, GD sabit güven ölçeklemeli iletişim. Hepsi komşularla sınırlı iletişim kullanır.

**Teknik:** Hepsi gat + komşu sinyali. GL: zlogodds güven terimi, β öğrenilen (0'dan); GLS: güven terimi sabit bir derangement ile göndericiler arasında karıştırılmış; GLM: yalnız güven teriminde gönderici başına sıcaklık; GD: prob terimi, β = 1 sabit (DebUnc ölçeklemesi).

- Kodda: [`marl/comm_models.py::confidence_term`](../marl/comm_models.py)
- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [M5](#m5), [S-msg](#s-msg), [β (güven ağırlığı)](#beta), [Komşu sinyali (σ)](#komsu-sinyali)

<a id="etiket-gecti"></a>

### GEÇTİ

**Sade:** Ön kayıttaki bütün koşullar sağlandı. Aşama bir sonrakine girdi verebilir.

**Teknik:** Ön kayıtlı kapının koşulları birlikte: etki yönü, tek yönlü p < 0,05, tohum sıralaması ve doğrulama penceresinde işaret. Geliştirme kapısı sonucudur, tez bulgusu değildir.

- İlgili: [Ön kayıt](#on-kayit), [Geliştirme kapısı](#gelistirme-kapisi), [Tek yönlü p](#tek-yonlu-p)

<a id="etiket-kismi"></a>

### KISMİ

**Sade:** Birincil koşul sağlandı ama diğerlerinden en az biri sağlanmadı. Kaynağı belirsiz bir işaret.

**Teknik:** Birincil fark ve testi var; kontrol, sıralama ya da doğrulama koşullarından biri yok. Aynı kapı ek tohumla kurtarılmaz; yeni deney yeni ön kayıt ister.

- İlgili: [Ön kayıt](#on-kayit), [Tohum](#tohum)

<a id="etiket-gozlenmedi"></a>

### GÖZLENMEDİ

**Sade:** Beklenen fark bu deneyde görülmedi. "Etki yok" demek değildir; deney küçük olabilir.

**Teknik:** Birincil koşul (fark > 0 ve tek yönlü p < 0,05) sağlanmadı. Güç düşükse okunuşu "bu güçle tespit edilmedi"dir.

- İlgili: [Güç](#guc), [Tek yönlü p](#tek-yonlu-p)

<a id="etiket-pilot"></a>

### PİLOT_İŞARETİ

**Sade:** Küçük bir ön denemede beklenen yönde bir işaret görüldü. Karar vermek için değil, büyük deneyi tasarlamak içindir.

**Teknik:** Pilot ön kayıtlarında (M4b, M5) tüm koşullar sağlandığında verilen etiket; karar kapısı değil, işarettir. Uzun koşu ayrı bir ön kayıt ister.

- İlgili: [Ön kayıt](#on-kayit), [Güç](#guc), [Tohum](#tohum)

<a id="tohum"></a>

### Tohum

**Sade:** Rastgeleliğin başlangıç numarası; aynı tohumla aynı deney aynı sonucu verir. Farklı tohumlar, sonucun şansa ne kadar bağlı olduğunu gösterir.

**Teknik:** Ağırlık başlangıcı, veri toplama ve sentetik sinyal üretecinin tohumu. Yeni bir ön kayıt, sonucu görülmüş tohumları yeniden kullanmaz (seçim yanlılığı). CPU'da bit-eşlik için OMP_NUM_THREADS ve MKL_NUM_THREADS = 1, torch import edilmeden önce.

- Kodda: [`marl/benchmarl_task.py`](../marl/benchmarl_task.py)
- İlgili: [Güç](#guc), [Kol](#kol)

<a id="gurultu-tabani"></a>

### Gürültü tabanı

**Sade:** Bilgi açısından aynı olması gereken iki kol arasında, yalnız şans ve donanım yüzünden ortaya çıkan fark. Gerçek bir etkinin bu farktan büyük olması beklenir.

**Teknik:** M5 mekanizma kollarında |D(GD, G0)|; GL ↔ G0 farkının bunu aşması H2'nin (v) koşuludur. Ortak HAC p'si üç tohumda tohum varyansını taşımadığı için taban, testin kalibrasyonunu da denetler.

- Kodda: [`scripts/marl/analyze_m5.py`](../scripts/marl/analyze_m5.py)
- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [D (Sharpe farkı)](#d-farki), [Ortak HAC](#hac), [G0 / GL / GLS / GLM / GD](#kol-kodlari-m5), [Karıştırılmış kontrol](#karistirilmis-kontrol)

<a id="okuma-o1-o4"></a>

### Gerçek sinyal okumaları (O1–O4)

**Sade:** Gerçek tahmin sinyaliyle yapılan koşuların dört olası sonucunun önceden yazılmış yorumu: fark yok, bozuyor, girdi etkisi ya da keşifsel işaret.

**Teknik:** O1 gerçek ↔ karıştırılmış fark yok (MDE ile raporlanır); O2 gerçek sinyal karıştırılmıştan ya da tahminsizden kötü (sağlamlık uyarısı); O3 gerçek ≈ karıştırılmış, ikisi de tahminsizden farklı (girdi boyutu ya da kapasite); O4 gerçek > karıştırılmış, yalnız keşifsel ve ek koşullarla.

- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [Okuma kuralı (okuma tablosu)](#okuma-kurali), [Karıştırılmış kontrol](#karistirilmis-kontrol), [En küçük tespit edilebilir etki (MDE)](#mde), [Keşifsel ve doğrulayıcı bulgu](#kesifsel)

<a id="kesifsel"></a>

### Keşifsel ve doğrulayıcı bulgu

**Sade:** Önceden yazılmış bir sınamadan geçen bulgu doğrulayıcı, sonuca bakarak fark edilen ama sınanmamış işaret keşifseldir. Keşifsel bir işaret tez bulgusu sayılmaz, yeni bir ön kayıtla sınanır.

**Teknik:** Geliştirme pilotları (M3–M5) keşifsel ya da geliştirme kapısıdır; doğrulayıcı sınama M8 ön kaydında ve dokunulmamış pencerede (m0p-v2 doğrulayıcı penceresi) yapılır. Rehber kartlarında "tez bulgusu değil" ve "ön kayıtlı nihai deney" rozetleri bu ayrımı gösterir.

- Belgede: [phase-10-marl-plan.md](development/phase-10-marl-plan.md)
- İlgili: [Ön kayıt](#on-kayit), [Geliştirme kapısı](#gelistirme-kapisi), [Gerçek sinyal okumaları (O1–O4)](#okuma-o1-o4), [Eğitim / doğrulama / test penceresi](#pencereler)

<a id="kol-kodlari-m6"></a>

### R0-N / SR-N

**Sade:** M6b pilotunun iki kolu: biri bugünkü gözlemle, öteki ayrıca bütün ajanlara aynı piyasa durumu bilgisini (eğilim ve oynaklık) vererek eğitilir. İkisi de iletişimsiz ve tahminsizdir.

**Teknik:** Tek etken S-reg yayın bloğu [z(sma_ratio), z(vol)]; maliyetsiz, F 600 bin, tohum 12–14; okuma BOZULMA / IYILESME_ISARETI / FARK_YOK (M6 tasarımı §6).

- Kodda: [`scripts/marl/run_m6sreg.py`](../scripts/marl/run_m6sreg.py), [`scripts/marl/analyze_m6sreg.py`](../scripts/marl/analyze_m6sreg.py)
- Belgede: [phase-10-m6-design.md](development/phase-10-m6-design.md)
- İlgili: [S-reg (rejim yayın bloğu)](#s-reg), [Kol](#kol), [Piyasa rejimi](#rejim)

<a id="kol-kodlari-m5-gercek"></a>

### T0-N / SO / SO-S / SO-S2 / T0-G / SB / SB-S / SB-S2

**Sade:** M5 gerçek sinyal kolları: SO gerçek tahmini gözleme ekler, SB gözleme ve güvenli mesaja ekler; -S ve -S2 sonekli kollar aynı sinyalin günleri karıştırılmış iki kopyasını, T0 kolları hiç tahmin görmeyen karşılıklarını kullanır. N iletişimsiz, G komşularla sınırlı iletişimli mimaridir.

**Teknik:** Sinyal pit_v2, komşu sinyali yok; karıştırma pencere içi (ısınma | eğitim | doğrulama), tohum s ve 1000 + s. SB = gat + GL (zlogodds, β öğrenilen). 20 bps, tohum 9–11. Birincil karşılaştırma gerçek ↔ karıştırılmış, tahminsiz karşılaştırmaları tabansız ikincildir.

- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [M5](#m5), [S-obs](#s-obs), [S-msg](#s-msg), [Karıştırılmış kontrol](#karistirilmis-kontrol), [Sıralama tabanı](#siralama-tabani), [Gerçek sinyal okumaları (O1–O4)](#okuma-o1-o4)

<a id="siralama-tabani"></a>

### Sıralama tabanı

**Sade:** Gerçek sinyalli kolun her tohumda iki karıştırılmış kolun ikisinin de üstünde (ya da altında) kalması koşulu. Sinyal bilgisizse bunun şansla olma olasılığı küçüktür.

**Teknik:** Aynı mimari ve tohumda gerçek, karıştırılmış ve ikinci karıştırma bilgi yokken değiştirilebilir sayılırsa, gerçeğin bir tohumda en üstte olma olasılığı 1/3; k = 3'te yön başına (1/3)³, iki yönlü yaklaşık 2/27. Bu bir alt sınırdır: gerçek blok tohumlar arasında ortaktır. Holm p ve doğrulama koşuluyla birlikte okunur.

- Kodda: [`scripts/marl/analyze_m5real.py`](../scripts/marl/analyze_m5real.py)
- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [T0-N / SO / SO-S / SO-S2 / T0-G / SB / SB-S / SB-S2](#kol-kodlari-m5-gercek), [Gürültü tabanı](#gurultu-tabani), [Karıştırılmış kontrol](#karistirilmis-kontrol), [Tohum](#tohum)

<a id="kat-degerlendirme"></a>

## Değerlendirme ve istatistik

<a id="sharpe"></a>

### Sharpe oranı · `SR`

**Sade:** Alınan her birim risk için ne kadar kazanıldığı. Yüksek olması, aynı dalgalanmayla daha çok kazanmak demektir.

**Teknik:** Fazla getirinin (X = R − rf) günlük ortalaması / standart sapması (ddof 1), √252 ile yıllıklandırılmış.

- Kodda: [`scripts/marl/analyze_m3.py::sr`](../scripts/marl/analyze_m3.py)
- İlgili: [Fazla getiri](#fazla-getiri), [D (Sharpe farkı)](#d-farki), [Risksiz faiz (A8)](#risksiz-faiz)
- Kaynak: Sharpe (1966)

<a id="fazla-getiri"></a>

### Fazla getiri

**Sade:** Portföyün getirisinin, para risksiz faizde dursaydı kazanacağının üstündeki kısmı.

**Teknik:** X = R − rf, R = g₁(1 − c·τ)g₂ − 1 (gece kayması, maliyet, seans içi). Ödül ve seçim metriği aynı seriyi kullanır.

- Kodda: [`marl/accounting.py::step_fractional`](../marl/accounting.py)
- İlgili: [Risksiz faiz (A8)](#risksiz-faiz), [İşlem maliyeti (bps)](#islem-maliyeti), [Devir](#devir), [Ödül](#odul)

<a id="risksiz-faiz"></a>

### Risksiz faiz (A8)

**Sade:** Parayı hiç risk almadan tutsaydık kazanacağımız faiz. Bir stratejinin gerçekten değer katıp katmadığı bunun üstünden ölçülür.

**Teknik:** Tek kaynak: TCMB ağırlıklı ortalama fonlama maliyeti (EVDS'ten data/macro/risk_free.csv), seans başına dönem başı getiriye çevrilir. Seri yoksa hata verir; 0 ya da sabit bir değere sessizce düşülmez.

- Kodda: [`evaluation/risk_free.py::risk_free_series`](../evaluation/risk_free.py), [`marl/panel.py::risk_free_for`](../marl/panel.py)
- İlgili: [Fazla getiri](#fazla-getiri), [Sharpe oranı](#sharpe)

<a id="d-farki"></a>

### D (Sharpe farkı)

**Sade:** İki kolun Sharpe oranları arasındaki farkın tohumlar üzerinden ortalaması. Pozitifse ilk kol daha iyidir.

**Teknik:** D = (1/K) Σₖ [SR(aₖ) − SR(bₖ)], eşli tohumlar ve aynı tarihler; standart hatası ortak HAC ile, yıllık. Eşleşme ve tarih uyumu okuma betiğinde denetlenir.

- Kodda: [`evaluation/sharpe_tests.py::sharpe_diff_hac_pooled`](../evaluation/sharpe_tests.py)
- İlgili: [Sharpe oranı](#sharpe), [Ortak HAC](#hac), [Tohum](#tohum), [Kol](#kol)

<a id="hac"></a>

### Ortak HAC

**Sade:** Günlük getirilerin birbirine bağımlı olmasını hesaba katan bir belirsizlik (standart hata) hesabı.

**Teknik:** Heteroskedasticity and autocorrelation consistent: her çiftin Ledoit–Wolf Sharpe farkı serileri birlikte (ortak) bir HAC kovaryansıyla; D'nin etki fonksiyonunun HAC varyansı. K = 1'de tek çift testiyle aynı sonucu verir.

- Kodda: [`evaluation/sharpe_tests.py::sharpe_diff_hac_pooled`](../evaluation/sharpe_tests.py)
- İlgili: [D (Sharpe farkı)](#d-farki), [Tek yönlü p](#tek-yonlu-p)
- Kaynak: Newey & West (1987); Ledoit & Wolf (2008)

**Uyarı:** p değeri piyasa günü gürültüsünü taşır, eğitim tohumları arasındaki değişkenliği taşımaz.

<a id="tek-yonlu-p"></a>

### Tek yönlü p

**Sade:** Gerçekte fark yokken, yalnız şansla en az bu kadar olumlu bir fark görme olasılığı. Küçükse sonucu şansa bağlamak zorlaşır.

**Teknik:** H0: D ≤ 0'a karşı tek taraflı; iki taraflı p'den D > 0 ise p/2, değilse 1 − p/2. Ön kayıtlarda eşik 0,05.

- Kodda: [`scripts/marl/analyze_m3.py::pooled`](../scripts/marl/analyze_m3.py)
- İlgili: [D (Sharpe farkı)](#d-farki), [Holm düzeltmesi](#holm), [Güç](#guc)

<a id="holm"></a>

### Holm düzeltmesi

**Sade:** Birden çok karşılaştırma yapınca şansla "bir şey bulma" riskini sınırlayan düzeltme.

**Teknik:** Holm–Bonferroni: p'ler küçükten büyüğe sıralanır, r'inci (0 tabanlı) p (m − r) ile çarpılır, kümülatif en büyük alınır ve 1'de kırpılır; aile hata oranını korur. Sabit sıralı testlerde (önce birincil, sonra ikincil) gerekmez.

- Kodda: [`scripts/marl/analyze_m3b.py::holm`](../scripts/marl/analyze_m3b.py)
- İlgili: [Tek yönlü p](#tek-yonlu-p), [Benjamini–Hochberg (BH-FDR)](#bh-fdr)

<a id="bh-fdr"></a>

### Benjamini–Hochberg (BH-FDR)

**Sade:** Çok sayıda test yapınca "bulduklarımın ne kadarı yanlış alarm" oranını sınırlayan yöntem.

**Teknik:** Yanlış keşif oranını (FDR) q düzeyinde tutan adım-yukarı yöntem. Tahmin bilgi kapısında 30 sembol tek aile.

- Kodda: [`evaluation/multiple_testing.py::benjamini_hochberg`](../evaluation/multiple_testing.py)
- İlgili: [Bilgi kapısı (İ3)](#bilgi-kapisi), [Holm düzeltmesi](#holm)
- Kaynak: Benjamini & Hochberg (1995)

<a id="guc"></a>

### Güç

**Sade:** Gerçekten bir fark varsa deneyin onu yakalama olasılığı. Az tohumlu küçük deneylerin gücü düşüktür.

**Teknik:** Tip II hata olasılığının tümleyeni. Ön kayıtlarda tek yönlü %5 ve %80 güç için gereken tohum sayısı betimsel verilir: k = 3·((1,645 + 0,842)·SE / etki)²; SE'nin tohumla 1/√k küçüldüğü varsayımı iyimserdir.

- Kodda: [`scripts/marl/analyze_m4b.py::seeds_needed`](../scripts/marl/analyze_m4b.py)
- İlgili: [En küçük tespit edilebilir etki (MDE)](#mde), [Tohum](#tohum), [GÖZLENMEDİ](#etiket-gozlenmedi)

<a id="mde"></a>

### En küçük tespit edilebilir etki (MDE)

**Sade:** Bu deneyin güvenle yakalayabileceği en küçük etki. Bundan küçük bir etki varsa bile görülmeyebilir.

**Teknik:** Bilgi kapısında AUC için 0,5 + (z₁₋α + z_güç)·SE_boot; tek test için α, aile için α/m (BH'de ilk reddin Bonferroni sınırı).

- Kodda: [`prediction/pit_table.py::evaluate_gate`](../prediction/pit_table.py)
- İlgili: [Güç](#guc), [AUC](#auc), [Bilgi kapısı (İ3)](#bilgi-kapisi)

<a id="guven-araligi"></a>

### Güven aralığı (GA)

**Sade:** Gerçek değerin büyük olasılıkla içinde olduğu aralık. Dar olması tahminin kesin olduğunu gösterir.

**Teknik:** %95 GA: tahmin ± 1,96·SE (normal yaklaşım) ya da blok bootstrap yüzdelikleri (bilgi kapısı, havuzlu AUC).

- İlgili: [Ortak HAC](#hac), [Güç](#guc)

<a id="pencereler"></a>

### Eğitim / doğrulama / test penceresi

**Sade:** Verinin üç dilimi: model eğitimde öğrenir, doğrulamada kontrol edilir. Son dilime (2021 ve sonrası) geliştirme sırasında bakılmaz.

**Teknik:** MARL: eğitim 2013-06 .. 2018-12-31, doğrulama 2019–2020; 2021+ m0p-v2 doğrulayıcı penceresidir ve M2–M5 geliştirmesinde başarımla okunmaz. Değerlendirme deterministik ve tam penceredir; normalizasyon yalnız eğitim satırlarından.

- Kodda: [`scripts/marl/train_m3.py::TRAIN_END`](../scripts/marl/train_m3.py)
- İlgili: [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [Milestone 0'](#milestone-0p)

<a id="baseline-1n"></a>

### 1/N (eşit ağırlık)

**Sade:** Parayı bütün hisselere eşit bölen en basit strateji. Her modelin en azından bunu geçmesi beklenir.

**Teknik:** MARL kıyasında her gün, tek ajan kıyasında aylık olarak geçerli sembollere eşit ağırlık; ortamın aynı muhasebesi, fiyat dolgusu, projeksiyonu ve maliyetiyle.

- Kodda: [`marl/rule_baselines.py::rule_series`](../marl/rule_baselines.py), [`evaluation/baselines.py::equal_weight`](../evaluation/baselines.py)
- İlgili: [Al-tut](#al-tut), [İşlem maliyeti (bps)](#islem-maliyeti)

<a id="al-tut"></a>

### Al-tut

**Sade:** İlk gün parayı eşit bölüp sonra hiç işlem yapmamak.

**Teknik:** İlk gün 1/N, sonra hedef = kaymış ağırlık (işlem yok; yalnız zorunlu dondurma), ortamın muhasebesiyle.

- Kodda: [`marl/rule_baselines.py::rule_series`](../marl/rule_baselines.py)
- İlgili: [1/N (eşit ağırlık)](#baseline-1n)

<a id="kural-portfoyleri"></a>

### Kural portföyleri (1/N, al-tut, min-varyans, HRP)

**Sade:** Öğrenme içermeyen, sabit bir kuralla ağırlık veren basit stratejiler. Öğrenen modellerin her raporda yanında gösterilir.

**Teknik:** 1/N her gün eşit ağırlık, al-tut ilk gün 1/N sonra işlem yok; min-varyans (uzun-yalnız, Ledoit-Wolf kovaryans) ve HRP ay başında 252 günlük pencereyle yeniden dengelenir. Ortamın muhasebesi ve aynı maliyetle (rule_series).

- Kodda: [`marl/rule_baselines.py::rule_series`](../marl/rule_baselines.py), [`marl/rule_baselines.py::hrp`](../marl/rule_baselines.py), [`marl/rule_baselines.py::minvar_long_only`](../marl/rule_baselines.py)
- İlgili: [1/N (eşit ağırlık)](#baseline-1n), [Al-tut](#al-tut), [İşlem maliyeti (bps)](#islem-maliyeti), [M8 (nihai koşumlar)](#m8)

<a id="devir"></a>

### Devir

**Sade:** Bir günde portföyün ne kadarının alınıp satıldığı. Yüksek devir, yüksek işlem maliyeti demektir.

**Teknik:** τ = Σᵢ |w*ᵢ − w̃ᵢ|, yalnız hisseler (nakit devre girmez); w* hedef, w̃ gece kaymış ağırlık. Maliyet bir kez sayılır.

- Kodda: [`marl/accounting.py::step_fractional`](../marl/accounting.py)
- İlgili: [İşlem maliyeti (bps)](#islem-maliyeti), [Fazla getiri](#fazla-getiri)

<a id="islem-maliyeti"></a>

### İşlem maliyeti (bps)

**Sade:** Her alım-satımda ödenen komisyon ve fiyat kayması. Baz puanla (bps, on binde bir) ölçülür.

**Teknik:** Gidiş-dönüş bps verilir; tek yön c = bps / 2 / 10⁴ (evaluation/baselines.py D4 ile aynı) ve devire uygulanır. Mekanizma kolları maliyetsiz, gerçek sinyal kolları 20 bps ile koşulur.

- Kodda: [`marl/accounting.py::one_way_cost`](../marl/accounting.py)
- İlgili: [Devir](#devir), [Fazla getiri](#fazla-getiri)

<a id="dsr"></a>

### Deflated Sharpe (DSR)

**Sade:** Çok sayıda deneme yapıldığında en iyi Sharpe oranının şansla ne kadar şişebileceğini düzelten ölçü.

**Teknik:** Deneme sayısı, getirinin çarpıklığı ve basıklığıyla düzeltilmiş Sharpe olasılığı.

- Kodda: [`evaluation/sharpe.py::deflated_sharpe_ratio`](../evaluation/sharpe.py)
- İlgili: [Sharpe oranı](#sharpe), [Deneme kaydı (N)](#deneme-kaydi)
- Kaynak: Bailey & López de Prado (2014)

**Uyarı:** Eski uygulama yanlıştı ve düzeltildi; eski raporlarda eski fonksiyonla verilmiş bir DSR varsa yeniden hesaplanmalıdır.

<a id="pbo"></a>

### Aşırı uyum olasılığı (PBO)

**Sade:** Geçmiş veride en iyi görünen yapılandırmanın, yeni veride ortalamanın altında kalma olasılığı. Yüksekse seçim şansa dayanıyor demektir.

**Teknik:** Kombinatoryal simetrik çapraz doğrulama (CSCV): (T, N) getiri matrisi S = 16 bloğa bölünür, C(16, 8) birleşimde örneklem içi en iyinin örneklem dışı sırası; PBO ve λ dağılımı.

- Kodda: [`evaluation/overfitting.py::probability_of_backtest_overfitting`](../evaluation/overfitting.py)
- İlgili: [Deflated Sharpe (DSR)](#dsr), [Model güven kümesi (MCS)](#mcs), [Deneme kaydı (N)](#deneme-kaydi)
- Kaynak: Bailey vd. (2017)

<a id="mcs"></a>

### Model güven kümesi (MCS)

**Sade:** Çok sayıda model arasından, istatistiksel olarak "en iyiden ayırt edilemeyen" modellerin kümesi. Tek bir kazanan yerine güvenilir bir küme verir.

**Teknik:** Hansen–Lunde–Nason; kayıp matrisi, durağan bootstrap (reps 2000), T_R istatistiği, varsayılan düzey %10.

- Kodda: [`evaluation/model_sets.py::model_confidence_set`](../evaluation/model_sets.py)
- İlgili: [Aşırı uyum olasılığı (PBO)](#pbo), [Blok bootstrap](#blok-bootstrap)
- Kaynak: Hansen vd. (2011)

<a id="lw-bootstrap"></a>

### Ledoit–Wolf Sharpe farkı testi

**Sade:** İki stratejinin Sharpe oranlarının gerçekten farklı olup olmadığını, getirilerin birbirine bağımlılığını hesaba katarak sınayan test.

**Teknik:** Sharpe farkının delta yöntemiyle HAC standart hatası (ön beyazlatmalı, QS çekirdeği, Andrews bant genişliği) ve studentize dairesel blok bootstrap sürümü (H0: SR1 = SR2, iki yönlü). Çok tohumlu okumada ortak (havuzlu) sürüm kullanılır.

- Kodda: [`evaluation/sharpe_tests.py::sharpe_diff_hac`](../evaluation/sharpe_tests.py), [`evaluation/sharpe_tests.py::sharpe_diff_bootstrap`](../evaluation/sharpe_tests.py)
- İlgili: [Ortak HAC](#hac), [Sharpe oranı](#sharpe), [D (Sharpe farkı)](#d-farki), [Blok bootstrap](#blok-bootstrap)
- Kaynak: Ledoit & Wolf (2008)

<a id="blok-bootstrap"></a>

### Blok bootstrap

**Sade:** Zaman serisini tek tek gün yerine ardışık bloklar halinde yeniden örnekleyerek belirsizliği ölçme yöntemi. Günler arası bağımlılık böylece korunur.

**Teknik:** Durağan bootstrap (rastgele blok uzunluğu, optimal_block_length) ortalama farkı için; bilgi kapısında 20 günlük tarih blokları. Güven aralığı ve tek yönlü p yüzdeliklerden.

- Kodda: [`evaluation/paired_tests.py::mean_diff_bootstrap`](../evaluation/paired_tests.py), [`evaluation/calibration.py::block_bootstrap_ci`](../evaluation/calibration.py)
- İlgili: [Güven aralığı (GA)](#guven-araligi), [Bilgi kapısı (İ3)](#bilgi-kapisi), [Model güven kümesi (MCS)](#mcs)
- Kaynak: Politis & Romano (1994)

<a id="diebold-mariano"></a>

### Diebold–Mariano testi · `DM`

**Sade:** İki tahmin modelinden hangisinin daha küçük hata yaptığını sınayan test.

**Teknik:** Kayıp farkının HAC (Newey–West, h − 1 gecikme) varyansı, Harvey–Leybourne–Newbold küçük örneklem düzeltmesi ve t dağılımı. prediction/evaluator.py'deki eski DM buraya yönlendirildi.

- Kodda: [`evaluation/forecast_tests.py::diebold_mariano`](../evaluation/forecast_tests.py)
- İlgili: [Ortak HAC](#hac), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble)
- Kaynak: Diebold & Mariano (1995)

<a id="sortino-calmar"></a>

### Sortino ve Calmar oranları

**Sade:** Sharpe'a benzer iki risk ayarlı getiri ölçüsü: Sortino yalnız aşağı yönlü dalgalanmayı, Calmar en büyük düşüşü risk sayar.

**Teknik:** compute_sortino_ratio (aşağı yönlü sapma), compute_calmar_ratio (yıllık getiri / maksimum düşüş). Tahmin değerlendiricisinde betimsel; tez çıkarımı Sharpe ve fazla getiri üstünden yapılır.

- Kodda: [`prediction/evaluator.py`](../prediction/evaluator.py)
- İlgili: [Sharpe oranı](#sharpe), [Fazla getiri](#fazla-getiri)
- Kaynak: Sortino & Price (1994)

<a id="getiri-birimi"></a>

### Getiri birimi (nominal / reel / USD)

**Sade:** Getirinin hangi parayla ölçüldüğü: Türk lirası, enflasyondan arındırılmış lira ya da dolar. Yüksek enflasyonda bu seçim sonucu çok değiştirir.

**Teknik:** Birincil Sharpe nominal TL, risksiz faiz (A8) üstü fazla getiri. Reel (TÜFE basamağı) ve USD (USDTRY_{t−1}) birimleri sharpe_rf0 ile; TÜFE düzeyi yalnız ex post raporlama birimidir, karar girdisi değildir.

- Kodda: [`evaluation/units.py::convert_values`](../evaluation/units.py), [`evaluation/units.py::fx_before`](../evaluation/units.py)
- İlgili: [Fazla getiri](#fazla-getiri), [Risksiz faiz (A8)](#risksiz-faiz), [Makro veri (yurt içi)](#makro-veri)

<a id="tohum-gereksinimi"></a>

### Tohum gereksinimi

**Sade:** Gözlenen bir farkın güvenle ayırt edilebilmesi için kaç tohumla koşmak gerektiğinin kaba tahmini. Uzun koşuların boyutu buna göre planlanır.

**Teknik:** seeds_needed(effect, se, k_now) = ⌈k_now · ((z₀,₉₅ + z₀,₈₀) · SE / etki)²⌉: standart hatanın √k ile küçüldüğü varsayımıyla, tek yönlü %5 düzeyde %80 güç için gereken tohum sayısı. Betimseldir; ön kayıttaki tohum sayısını sonradan değiştirmez.

- Kodda: [`scripts/marl/analyze_m4b.py::seeds_needed`](../scripts/marl/analyze_m4b.py)
- İlgili: [Tohum](#tohum), [Güç](#guc), [Güven aralığı (GA)](#guven-araligi), [En küçük tespit edilebilir etki (MDE)](#mde)

<a id="tohum-ozetleri"></a>

### Tohumlar arası özet (IQM, iyileşme olasılığı)

**Sade:** Birden fazla tohumla koşulan bir kolun sonucunu, tek bir şanslı tohuma kanmadan özetleme yolları.

**Teknik:** aggregate_with_ci (rliable): medyan, IQM (çeyrekler arası ortalama), ortalama, optimality gap ve katmanlı bootstrap GA; probability_of_improvement: rastgele bir tohumda bir kolun diğerini geçme olasılığı.

- Kodda: [`evaluation/seed_stats.py::aggregate_with_ci`](../evaluation/seed_stats.py), [`evaluation/seed_stats.py::probability_of_improvement`](../evaluation/seed_stats.py)
- İlgili: [Tohum](#tohum), [Güven aralığı (GA)](#guven-araligi)

<a id="kat-marl"></a>

## Çok ajanlı öğrenme

<a id="s-reg"></a>

### S-reg (rejim yayın bloğu)

**Sade:** Bütün ajanlara aynı anda verilen piyasa durumu bilgisi: endeksin eğilimi ve oynaklığı. Her ajan kendi hissesine bakarken piyasanın genel havasını da görür.

**Teknik:** Gözleme eklenen paylaşılan ek özellikler (extras): birincil rejim etiketinin sürekli özellikleri z(sma_ratio), z(vol); as-of, ölçek yalnız eğitim satırlarından; eleştirmene de girer. Bayrak kapalıyken gözlem v1 ile bit-eş.

- Kodda: [`scripts/marl/build_sreg_m6.py`](../scripts/marl/build_sreg_m6.py)
- Belgede: [phase-10-m6-design.md](development/phase-10-m6-design.md)
- İlgili: [Piyasa rejimi](#rejim), [S-obs](#s-obs), [Gözlem](#gozlem)

<a id="dikkat-testleri"></a>

### Dikkat testleri (maskeleme, tek biçimli dikkat, karıştırılmış mesaj)

**Sade:** Ajanların mesajlara ne kadar ve nasıl dayandığını, eğitimden sonra mesajları kapatarak, eşit ağırlıkla ortalayarak ya da göndericiler arasında karıştırarak sınayan tanılar. Kararın nedensel açıklaması değildir.

**Teknik:** Yalnız değerlendirme: `set_sender_mask` (M7a; mesaj kapalı), `set_uniform_attention` (izinli göndericilere eşit ağırlık), `set_sender_permutation` (yuva j, π(j)'nin mesajını taşır; dağılım korunur), `set_attention_record`. Kapalıyken bit-eş.

- Kodda: [`marl/comm_models.py::CommPolicy`](../marl/comm_models.py), [`scripts/marl/mask_m7.py`](../scripts/marl/mask_m7.py), [`scripts/marl/attn_m7b.py`](../scripts/marl/attn_m7b.py)
- Belgede: [phase-10-m6-design.md](development/phase-10-m6-design.md)
- İlgili: [Dikkat (attention)](#dikkat), [İletişim kolları (none / attn / gat / self)](#iletisim-kollari)
- Kaynak: Wiegreffe & Pinter (2019); Jain & Wallace (2019)

<a id="marl"></a>

### Çok ajanlı pekiştirmeli öğrenme (MARL)

**Sade:** Birden çok karar vericinin (ajanın) aynı ortamda deneme-yanılmayla birlikte öğrenmesi.

**Teknik:** Faz 10: hisse başına bir ajan (30 ajan, K6 seçenek A), ortak portföy. PettingZoo ortamı marl/env.py; eğitim BenchMARL/TorchRL ile ayrı sanal ortamda (venv-marl, K9).

- Kodda: [`marl/env.py`](../marl/env.py), [`marl/benchmarl_task.py`](../marl/benchmarl_task.py)
- Belgede: [phase-10-m2-design.md](development/phase-10-m2-design.md)
- İlgili: [Ajan](#ajan), [MAPPO](#mappo), [Merkezi eğitim, merkezi olmayan yürütme (CTDE)](#ctde)

<a id="ajan"></a>

### Ajan

**Sade:** Kendi gözlemine bakıp karar veren, öğrenen program. Burada her hisseye bir ajan düşer.

**Teknik:** Ajan i hisse i için bir logit üretir; logitler hedef ağırlığa softmax([0, kırp(z, −B, B)]) ile çevrilir (0 = nakit). Politika parametreleri ajanlar arasında paylaşılır; ajan kimliği gözlemde one-hot.

- Kodda: [`marl/weights.py::target_weights`](../marl/weights.py)
- İlgili: [Parametre paylaşımı (K24)](#parametre-paylasimi), [Logit sınırı (5 + ln N)](#logit-siniri), [Politika](#politika), [Gözlem](#gozlem)

<a id="mappo"></a>

### MAPPO

**Sade:** Ajanların eğitim sırasında ortak bir "eleştirmen"den yararlanarak öğrendiği yaygın bir çok ajanlı öğrenme yöntemi.

**Teknik:** Multi-Agent PPO: merkezi eleştirmen (varsayılan girdi tüm yerel gözlemler) + paylaşılan aktör. Ayarlar: mini-parti bölünmez, en fazla 10 epoch, clip 0,2, lr 5e-4 (M3a ön kaydı); BenchMARL 1.4 varsayılanları.

- Kodda: [`marl/m3_runner.py::build_experiment`](../marl/m3_runner.py)
- İlgili: [IPPO](#ippo), [Eleştirmen](#elestirmen), [Merkezi eğitim, merkezi olmayan yürütme (CTDE)](#ctde), [PPO](#ppo)
- Kaynak: Yu vd. (2022)

<a id="sarl"></a>

### SARL (tek ajanlı kıyas kolları)

**Sade:** Bütün portföy kararını tek bir ajanın verdiği kıyas düzeni. Çok ajanlı eğitimin bir şey katıp katmadığı buna göre ölçülür.

**Teknik:** M8 iki SARL kolu: S-esm (eşmimari; M-none ajan MLP'siyle aynı katmanlar, ortak oran kırpması ve tek değer) ve S-agr (düz gözlemden 2N çıkışlı tek MLP). Aynı gözlem, eylem temsili, maliyet ve tohumlar.

- Kodda: [`marl/sarl.py::SarlPortfolioEnv`](../marl/sarl.py), [`marl/sarl.py::RowSharedPolicy`](../marl/sarl.py)
- Belgede: [phase-10-m8-design.md](development/phase-10-m8-design.md)
- İlgili: [MAPPO](#mappo), [M8 (nihai koşumlar)](#m8), [Parametre paylaşımı (K24)](#parametre-paylasimi)

<a id="ippo"></a>

### IPPO

**Sade:** Her ajanın eleştirmeninin yalnız kendi gözlemine baktığı sürüm.

**Teknik:** Independent PPO: eleştirmen yerel gözlemle; eğitici seçeneği --algo ippo. Eleştirmen kapasitesi MAPPO ile eşit değildir ve kayda geçer.

- Kodda: [`scripts/marl/train_m3.py::critic_summary`](../scripts/marl/train_m3.py)
- İlgili: [MAPPO](#mappo), [Eleştirmen](#elestirmen)
- Kaynak: de Witt vd. (2020)

<a id="ctde"></a>

### Merkezi eğitim, merkezi olmayan yürütme (CTDE)

**Sade:** Eğitimde ajanlar her şeyi görebilir, ama işlem anında her ajan yalnız kendi gözlemiyle karar verir.

**Teknik:** Centralised training, decentralised execution: eleştirmen eğitimde merkezi girdi alır; politika yalnız yerel gözlemi (iletişim kolunda ayrıca mesajları) kullanır. Canlı karar eleştirmeni kullanmaz.

- İlgili: [Eleştirmen](#elestirmen), [MAPPO](#mappo)
- Kaynak: Lowe vd. (2017)

<a id="elestirmen"></a>

### Eleştirmen

**Sade:** Eğitim sırasında "bu durum ne kadar iyi" diye puan veren yardımcı ağ. İşlem anında kullanılmaz.

**Teknik:** Değer fonksiyonu; MAPPO'da girdi tüm yerel gözlemler (varsayılan) ya da ortamın state()'i (--use-state). Bütün kollarda aynı BenchMARL MLP; parametre sayısı ve girdi boyutu kayda geçer.

- Kodda: [`scripts/marl/train_m3.py::critic_summary`](../scripts/marl/train_m3.py)
- İlgili: [Merkezi eğitim, merkezi olmayan yürütme (CTDE)](#ctde), [MAPPO](#mappo)

<a id="parametre-paylasimi"></a>

### Parametre paylaşımı (K24)

**Sade:** Bütün ajanlar aynı "beyni" kullanır. Hangi hisseye baktığını gözlemindeki kimlikten bilir.

**Teknik:** Tam paylaşım + ajan kimliği one-hot (K24); share_policy_params = True. Sektör başına seçici paylaşım ablasyon olarak planda.

- Kodda: [`marl/m3_runner.py::build_experiment`](../marl/m3_runner.py)
- İlgili: [Ajan](#ajan), [K-kararları](#k-kararlari)

<a id="iletisim-kollari"></a>

### İletişim kolları (none / attn / gat / self)

**Sade:** Ajanların birbirine mesaj gönderip göndermediğini ve kime gönderdiğini belirleyen seçenekler.

**Teknik:** none: iletişimsiz MLP; attn: tam bağlı çok başlı öz-dikkat (TarMAC tipi); gat: aynı katmanlar, dikkat yalnız graf komşuları ve kendisiyle sınırlı; self: aynı katmanlar, dikkat yalnız kendine (mimari-eş iletişimsiz kontrol). attn, gat ve self'in parametre sayısı eşittir.

- Kodda: [`marl/comm_models.py::CommPolicy`](../marl/comm_models.py), [`marl/comm_models.py::graph_mask`](../marl/comm_models.py)
- Belgede: [phase-10-m4-design.md](development/phase-10-m4-design.md)
- İlgili: [Graf (sektör + MST)](#graf), [N / A / G / Z × X / S](#kol-kodlari-m4), [Dikkat (attention)](#dikkat)
- Kaynak: Das vd. (2019); Veličković vd. (2018)

<a id="dikkat"></a>

### Dikkat (attention)

**Sade:** Bir ajanın gelen mesajların hangisine ne kadar kulak vereceğini öğrendiği mekanizma.

**Teknik:** puanᵢⱼ = qᵢ·kⱼ/√d (+ β·bⱼ güven terimi); softmax ağırlıklarıyla mesajlar toplanır, artık bağlantı ve LayerNorm. Çıktı başı [kendi kod ‖ mesaj sonrası] alır.

- Kodda: [`marl/comm_models.py::CommPolicy`](../marl/comm_models.py)
- İlgili: [İletişim kolları (none / attn / gat / self)](#iletisim-kollari), [β (güven ağırlığı)](#beta), [S-msg](#s-msg)

<a id="graf"></a>

### Graf (sektör + MST)

**Sade:** Hangi hisselerin "komşu" sayılacağını gösteren bağlantı haritası: aynı sektördekiler ve fiyatları en çok birlikte hareket edenler.

**Teknik:** sector+mst kuralı: KAP sektör grupları + eğitim penceresi log getiri korelasyonundan (Pearson, çift bazında tam gözlemli günler) minimum yayılan ağaç. Kararlılık eğitim penceresinin iki yarısındaki grafların Jaccard'ıyla ölçülür.

- Kodda: [`scripts/marl/graph_m4.py::pairwise_corr`](../scripts/marl/graph_m4.py)
- Belgede: [phase-10-m4-design.md](development/phase-10-m4-design.md)
- İlgili: [İletişim kolları (none / attn / gat / self)](#iletisim-kollari), [Komşu sinyali (σ)](#komsu-sinyali)

<a id="alfa"></a>

### α (ödül karışımı)

**Sade:** Bir ajanın ödülünde kendi hissesinin katkısıyla bütün portföyün başarısının hangi oranda karıştırılacağı.

**Teknik:** rᵢ = ölçek·(α·kᵢ + (1 − α)·G); kᵢ ajanın fazla getiri katkısı, G portföy terimi, α ∈ [0, 1). Referans yapılandırma α = 0,25, ölçek 100. 'differential' ödülü yalnız α = 0 ablasyonunda.

- Kodda: [`marl/accounting.py::agent_rewards`](../marl/accounting.py)
- İlgili: [Ödül](#odul), [Fazla getiri](#fazla-getiri)

<a id="logit-siniri"></a>

### Logit sınırı (5 + ln N)

**Sade:** Bir ajanın tek bir hisseye ne kadar uç bir ağırlık isteyebileceğinin üst sınırı.

**Teknik:** Eylem logitleri [−B, B] aralığına kırpılır, B = 5 + ln N (kullanıcı kararı 2026-11-16); nakit logiti sabit 0.

- Kodda: [`marl/weights.py::logit_bound`](../marl/weights.py)
- İlgili: [Ajan](#ajan)

<a id="s-obs"></a>

### S-obs

**Sade:** Tahmin sinyalini ajanın kendi gözlemine eklemek.

**Teknik:** Gözlem bloğu [p_up_cal − 0,5, z(pred_logret), z(model_disagreement), pred_ok]; eksik tahmin maskelenir, 0,5'e düşülmez. Ölçek yalnız eğitim satırlarından.

- Kodda: [`marl/obs_spec.py::pred_block`](../marl/obs_spec.py)
- İlgili: [S-msg](#s-msg), [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [Gözlem](#gozlem)

<a id="s-msg"></a>

### S-msg

**Sade:** Tahminin güvenini mesajlaşmada kullanmak: kendinden emin olan göndericinin mesajına daha çok kulak verilir.

**Teknik:** puanᵢⱼ = qᵢ·kⱼ/√d + β·bⱼ; bⱼ göndericinin güven terimi, alıcıdan bağımsız: prob = log kırp(0,5 + |p − 0,5|), zlogodds = |logit p| / z_scale. Güven gözlemden okunur; gözlem tanımı değişmez (K2).

- Kodda: [`marl/comm_models.py::confidence_term`](../marl/comm_models.py)
- Belgede: [phase-10-m5-design.md](development/phase-10-m5-design.md)
- İlgili: [β (güven ağırlığı)](#beta), [Dikkat (attention)](#dikkat), [S-obs](#s-obs), [G0 / GL / GLS / GLM / GD](#kol-kodlari-m5)
- Kaynak: Yoffe vd. (2025)

<a id="beta"></a>

### β (güven ağırlığı)

**Sade:** Güvenin dikkate ne kadar karışacağını belirleyen ayar. Sıfırsa güven hiç kullanılmaz.

**Teknik:** Sabit (0 = güvensiz dikkat, 1 = DebUnc ölçeklemesi) ya da öğrenilen bir skaler (0'dan başlar). Eğitim sonu değeri kayda geçer ve mekanizmanın kullanılıp kullanılmadığının denetimidir.

- Kodda: [`marl/comm_models.py::CommPolicy`](../marl/comm_models.py)
- İlgili: [S-msg](#s-msg), [Dikkat (attention)](#dikkat)

<a id="guven-gecidi"></a>

### Güven geçidi

**Sade:** Ajan kendi tahmininden emin değilse gelen mesajlara daha çok ağırlık veren anahtar.

**Teknik:** gᵢ = σ(a·dᵢ + b), dᵢ alıcının kendi güvensizliği; mesaj gᵢ ile ölçeklenir. Başlangıç a = 0, b = 4,6 (g ≈ 0,99). Komşu sinyali düzeninde sınanamaz, çünkü orada ajanın kendi güveni başka bir hisseye aittir.

- Kodda: [`marl/comm_models.py::CommPolicy`](../marl/comm_models.py)
- İlgili: [S-msg](#s-msg), [β (güven ağırlığı)](#beta), [Komşu sinyali (σ)](#komsu-sinyali)

<a id="kare"></a>

### Kare (frame)

**Sade:** Eğitimde ortamla yapılan tek bir adım. Eğitim bütçesi kare sayısıyla ölçülür.

**Teknik:** BenchMARL max_n_frames; M3–M5 koşularında 600.000 kare, parti 6.000 kare, partide 10 epoch.

- Kodda: [`scripts/marl/train_m3.py`](../scripts/marl/train_m3.py)
- İlgili: [Bölüm (episode)](#bolum)

<a id="bolum"></a>

### Bölüm (episode)

**Sade:** Eğitimde baştan sona oynanan bir dönem. Burada yaklaşık bir yıl (252 işlem günü).

**Teknik:** episode_len 252; bölüm sonu terminated = False ve GAE bootstrap (kesik bölüm, gerçek son değil).

- Kodda: [`marl/benchmarl_task.py`](../marl/benchmarl_task.py)
- İlgili: [Kare (frame)](#kare)

<a id="benchmarl"></a>

### BenchMARL / TorchRL

**Sade:** Çok ajanlı modelleri eğitmek için kullanılan hazır kütüphaneler. Eğitim döngüsü, algoritmalar ve deney kaydı bunlardan gelir.

**Teknik:** MarlTask (BenchMARL görevi) MARL ortamını sarar; algoritma MAPPO/IPPO, ağlar TorchRL. Yalnız venv-marl'da (K9). Tohum bit-eşliği için OMP_NUM_THREADS = MKL_NUM_THREADS = 1 torch import'undan önce ayarlanır.

- Kodda: [`marl/benchmarl_task.py::MarlTask`](../marl/benchmarl_task.py), [`marl/m3_runner.py`](../marl/m3_runner.py)
- İlgili: [venv-marl (ayrı ortam)](#venv-marl), [MAPPO](#mappo), [IPPO](#ippo), [Kare (frame)](#kare)
- Kaynak: Bettini vd. (2024)

<a id="muhasebe"></a>

### Muhasebe (kesirli adım)

**Sade:** Ortamın her gün portföy ağırlıklarını, nakit faizini, işlem maliyetini ve getiriyi nasıl hesapladığını belirleyen kurallar. Ortam ve canlı karar aynı fonksiyonu kullanır.

**Teknik:** Karar t kapanışında, yürütme t+1 açılışında: gece hisseler Oₜ₊₁/Cₜ ile kayar, nakit rf kazanır; açılışta hedef izdüşürülür ve maliyet c·τ bir kez düşer; seans içi Cₜ₊₁/Oₜ₊₁ (nakit seans içinde faiz kazanmaz). R = g1(1 − cτ)g2 − 1, X = R − rf. rf zorunludur, sabite düşülmez.

- Kodda: [`marl/accounting.py::step_fractional`](../marl/accounting.py), [`marl/accounting.py::step_growth`](../marl/accounting.py)
- Belgede: [phase-10-m2-design.md](development/phase-10-m2-design.md)
- İlgili: [Devir](#devir), [İşlem maliyeti (bps)](#islem-maliyeti), [Risksiz faiz (A8)](#risksiz-faiz), [Hedef ağırlık izdüşümü (risk projeksiyonu)](#izdusum), [K12 yürütme ufku](#k12)

<a id="izdusum"></a>

### Hedef ağırlık izdüşümü (risk projeksiyonu)

**Sade:** Ajanların istediği ağırlıkların, işlem yapılamayan hisseler ve tavan kuralları gibi kısıtlara uyacak şekilde düzeltilmesi.

**Teknik:** project(u, w_drift, tradable, Constraints): işlem göremeyen hisse kaymış ağırlığında tutulur, satış engeli (no_sell) uygulanır; band (dönüşüm bandı), max_position, max_group parametreleri varsayılan kapalı. Parametreler model meta'sına risk_projection olarak yazılır.

- Kodda: [`marl/weights.py::project`](../marl/weights.py), [`marl/weights.py::Constraints`](../marl/weights.py)
- İlgili: [Eylem](#eylem), [Satış engeli (taban açılış)](#satis-engeli), [Listeleme kilidi](#listeleme-kilidi), [Muhasebe (kesirli adım)](#muhasebe)

<a id="emir-donusumu"></a>

### Emir dönüşümü

**Sade:** Hedef ağırlıkların gerçek alım-satım emirlerine, yani tam sayı lot adetlerine çevrilmesi.

**Teknik:** rebalance_orders(target_w, prices, holdings, cash, commission, ...): ortamın kesirli muhasebesiyle canlı kararın tam sayı emirleri aynı hedef ağırlıktan üretilir (K5); parite testleri bit-eşlik ister.

- Kodda: [`marl/orders.py::rebalance_orders`](../marl/orders.py), [`marl/orders.py::apply_to_holdings`](../marl/orders.py)
- İlgili: [Hedef ağırlık izdüşümü (risk projeksiyonu)](#izdusum), [Günlük karar](#gunluk-karar), [Muhasebe (kesirli adım)](#muhasebe)

<a id="zlogodds"></a>

### zlogodds güven terimi

**Sade:** Bir tahminin ne kadar emin olduğunu, olasılığın 0,5'ten uzaklığını ölçeklenmiş log-oran olarak ifade eden biçim.

**Teknik:** b_j = |log(p / (1 − p))| / z_scale; z_scale = eğitim satırlarında tahminli hücrelerin |logit p| standart sapması (conf_scale). Tahmini olmayan hücrede taban 0. Alternatif biçim 'prob': log(clip(0,5 + |p − 0,5|)).

- Kodda: [`marl/comm_models.py::confidence_term`](../marl/comm_models.py), [`marl/comm_models.py::conf_scale`](../marl/comm_models.py)
- İlgili: [β (güven ağırlığı)](#beta), [S-msg](#s-msg), [G0 / GL / GLS / GLM / GD](#kol-kodlari-m5), [p_up_cal (kalibre edilmiş yükseliş olasılığı)](#p-up-cal)

<a id="init-std"></a>

### Başlangıç eylem gürültüsü (init_std)

**Sade:** Eğitimin başında ajanların ne kadar rastgele davranacağını belirleyen ayar. Çok küçük olursa ajan yeterince denemez.

**Teknik:** TanhNormal politika, tanh öncesi başlangıç std'si init_std (biased_softplus); M3–M5 koşularında 0,12.

- Kodda: [`marl/m3_runner.py`](../marl/m3_runner.py), [`scripts/marl/train_m3.py`](../scripts/marl/train_m3.py)
- İlgili: [Politika](#politika), [Eylem](#eylem), [MAPPO](#mappo)

<a id="deterministik-degerlendirme"></a>

### Deterministik değerlendirme

**Sade:** Eğitilmiş politikanın rastgelelik olmadan, bütün pencere boyunca tek seferde sınanması.

**Teknik:** eval_m3.py kayıtlı politikayı deterministic_rollout ile, tam eğitim ya da doğrulama penceresinde koşar ve günlük seriyi (tarih, getiri, fazla getiri, devir, nakit ağırlığı) yazar; baseline serileri aynı muhasebeyle üretilir. Sharpe ve okuma analiz betiğinde.

- Kodda: [`scripts/marl/eval_m3.py`](../scripts/marl/eval_m3.py), [`marl/m3_runner.py::deterministic_rollout`](../marl/m3_runner.py)
- İlgili: [Eğitim / doğrulama / test penceresi](#pencereler), [1/N (eşit ağırlık)](#baseline-1n), [Muhasebe (kesirli adım)](#muhasebe)

<a id="kat-rl"></a>

## Tek ajan ve genel pekiştirmeli öğrenme

<a id="politika"></a>

### Politika

**Sade:** Ajanın gözlemden karara giden kuralı. Öğrenilen şey budur.

**Teknik:** MARL'da TanhNormal aktör; eğitimde başlangıç gürültüsü tanh öncesi std 0,12, değerlendirmede deterministik (mod).

- Kodda: [`marl/m3_runner.py::deterministic_rollout`](../marl/m3_runner.py)
- İlgili: [Ajan](#ajan), [Gözlem](#gozlem), [Ödül](#odul)

<a id="gozlem"></a>

### Gözlem

**Sade:** Ajanın karar anında gördüğü bilgiler.

**Teknik:** marl-obs-v1: 11 z-ölçekli özellik, feat_ok, kendi ağırlığı, nakit ağırlığı, ajan kimliği (N), isteğe bağlı tahmin bloğu. Normalizasyon yalnız eğitim satırlarından (norm_stats); t+1 geçerliliği gözleme konmaz.

- Kodda: [`marl/obs_spec.py::build_obs_at`](../marl/obs_spec.py)
- İlgili: [S-obs](#s-obs), [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [Politika](#politika)

<a id="odul"></a>

### Ödül

**Sade:** Ajana "ne kadar iyi yaptın" diye verilen sayı. Öğrenme bunu artırmaya çalışır.

**Teknik:** MARL birincil ödülü 'excess' (fazla getiri katkısı ve portföy terimi, α karışımı). Tek ajan ortamında varsayılan Sharpe terimi EMA Sharpe düzeyidir; ödül tanımının tek kaynağı reward-definition.md.

- Kodda: [`marl/accounting.py::agent_rewards`](../marl/accounting.py), [`env/reward_functions.py`](../env/reward_functions.py)
- Belgede: [reward-definition.md](thesis/reward-definition.md)
- İlgili: [Fazla getiri](#fazla-getiri), [α (ödül karışımı)](#alfa), [EMA Sharpe düzeyi](#ema-sharpe)

<a id="ema-sharpe"></a>

### EMA Sharpe düzeyi

**Sade:** Tek ajanlı modelin ödülünde, son dönem getirilerinin üstel ortalamasından hesaplanan Sharpe benzeri terim.

**Teknik:** Varsayılan sharpe_term = 'ema_level'. Moody–Saffell'in Dₜ türevi değildir; o seçimli olarak sharpe_term = 'differential' ile (Faz 9 G1). Komisyon ödülde iki kez sayılıyor (beyan).

- Kodda: [`env/reward_functions.py`](../env/reward_functions.py)
- Belgede: [reward-definition.md](thesis/reward-definition.md)
- İlgili: [Ödül](#odul)
- Kaynak: Moody & Saffell (2001)

**Uyarı:** Bu terime "diferansiyel Sharpe" denmez (tez kuralı, docs/thesis/CLAUDE.md).

<a id="ppo"></a>

### PPO

**Sade:** Politikayı küçük ve güvenli adımlarla güncelleyen yaygın bir öğrenme yöntemi.

**Teknik:** Proximal Policy Optimization: kırpılmış olasılık oranı amacı (clip ε = 0,2). MAPPO ve IPPO bunun çok ajanlı sürümleridir.

- İlgili: [MAPPO](#mappo), [IPPO](#ippo), [Politika](#politika)
- Kaynak: Schulman vd. (2017)

<a id="eylem"></a>

### Eylem

**Sade:** Ajanın her adımda verdiği karar. Bu sistemde eylem, paranın hisseler ve nakit arasında nasıl dağıtılacağını söyler.

**Teknik:** MARL: her ajan kendi hissesi için bir logit üretir; logitler logit sınırında kırpılır, nakit logiti 0 sabit, softmax hedef ağırlığı verir (target_weights), sonra kısıtlara izdüşürülür (project). Tek ajan ortamı: hisse başına [-1, +1] sürekli çıktı, lot sayısına ölçeklenir (max_shares_per_trade).

- Kodda: [`marl/weights.py::target_weights`](../marl/weights.py), [`marl/weights.py::project`](../marl/weights.py), [`env/trading_env.py::TradingEnv`](../env/trading_env.py)
- İlgili: [Politika](#politika), [Logit sınırı (5 + ln N)](#logit-siniri), [Gözlem](#gozlem)

<a id="tek-ajan-ortami"></a>

### Tek ajan ortamı (TradingEnv)

**Sade:** Bütün hisseleri tek bir ajanın yönettiği eski öğrenme ortamı. Ansari replikasyonu ve Milestone 0' bu ortamda koşulur.

**Teknik:** Gymnasium ortamı; gözlem bakiye + hisse adetleri + OHLCV + göstergeler (5 sembolde 56, 30 sembolde 331 boyut), faz 2 tahmin bloğuyla genişler. State space yapısı eğitilmiş modellerle uyum için değiştirilmez; yeni MARL ortamı ayrıdır.

- Kodda: [`env/trading_env.py::TradingEnv`](../env/trading_env.py), [`env/trading_env.py::make_env`](../env/trading_env.py)
- Belgede: [CLAUDE.md](../env/CLAUDE.md)
- İlgili: [Ansari replikasyonu](#ansari-replikasyonu), [Milestone 0'](#milestone-0p), [OHLCV](#ohlcv), [Gymnasium](#gymnasium)

<a id="gymnasium"></a>

### Gymnasium

**Sade:** Pekiştirmeli öğrenme ortamları için standart arayüz kütüphanesi. Ortam "gözlem ver, eylem al, ödül döndür" kalıbıyla yazılır.

**Teknik:** TradingEnv gym.Env alt sınıfıdır (reset/step, observation_space ve action_space Box). MARL ortamı aynı kalıbın çok ajanlı karşılığı PettingZoo ile yazılır ve yalnız venv-marl'da çalışır.

- Kodda: [`env/trading_env.py::TradingEnv`](../env/trading_env.py), [`marl/env.py`](../marl/env.py)
- İlgili: [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami), [venv-marl (ayrı ortam)](#venv-marl)
- Kaynak: Towers vd. (2024)

<a id="sb3"></a>

### Stable-Baselines3 · `SB3`

**Sade:** Tek ajanlı modelleri eğitmek için kullanılan hazır algoritma kütüphanesi.

**Teknik:** Eğitim rotası PPO, A2C, TD3 ve SAC'ı SB3'ten kurar; model .zip olarak kaydedilir, yanına evren ve ölçeği taşıyan model meta dosyası yazılır. MARL tarafı SB3 değil BenchMARL/TorchRL kullanır.

- Kodda: [`app/api/routes/trading.py`](../app/api/routes/trading.py)
- İlgili: [PPO](#ppo), [A2C](#a2c), [TD3](#td3), [SAC](#sac), [Model meta dosyası](#model-meta), [BenchMARL / TorchRL](#benchmarl)
- Kaynak: Raffin vd. (2021)

<a id="a2c"></a>

### A2C

**Sade:** Politikayı ve değer tahminini aynı anda öğrenen, eşzamanlı çalışan bir öğrenme yöntemi.

**Teknik:** Advantage Actor-Critic (A3C'nin eşzamanlı sürümü); eğitim rotasında algorithm = 'a2c' ile SB3'ten.

- Kodda: [`app/api/routes/trading.py`](../app/api/routes/trading.py)
- İlgili: [PPO](#ppo), [Stable-Baselines3](#sb3), [Eleştirmen](#elestirmen)
- Kaynak: Mnih vd. (2016)

<a id="td3"></a>

### TD3

**Sade:** Sürekli eylemler için, değeri olduğundan yüksek tahmin etme eğilimini azaltan bir öğrenme yöntemi.

**Teknik:** Twin Delayed DDPG: iki eleştirmenin küçüğü, gecikmeli politika güncellemesi, hedef politika yumuşatma; algorithm = 'td3' ile SB3'ten.

- Kodda: [`app/api/routes/trading.py`](../app/api/routes/trading.py)
- İlgili: [Stable-Baselines3](#sb3), [Eleştirmen](#elestirmen), [SAC](#sac)
- Kaynak: Fujimoto vd. (2018)

<a id="sac"></a>

### SAC

**Sade:** Ödülle birlikte davranıştaki çeşitliliği de ödüllendiren, sürekli eylemli bir öğrenme yöntemi.

**Teknik:** Soft Actor-Critic (entropi düzenlemeli, politika dışı); eğitim rotasında algorithm = 'sac' ile SB3'ten seçilebilir.

- Kodda: [`app/api/routes/trading.py`](../app/api/routes/trading.py)
- İlgili: [Stable-Baselines3](#sb3), [TD3](#td3)

<a id="hpo"></a>

### Hiperparametre optimizasyonu · `HPO`

**Sade:** Öğrenme hızı gibi ayarların, çok sayıda deneme yapılarak otomatik seçilmesi. Her deneme deneme kaydına sayılır.

**Teknik:** Optuna, varsayılan TPE örnekleyici (ayar tohumuyla kurulur) ve budayıcı. HPO rotanın yolunu kullanır: panel, kat ve ortam evaluation/rl_folds'tan, pencere wf-v1 katından; arama doğrulama bloğunda yapılır, test okunmaz.

- Kodda: [`hyperparameter_optimization/base_optimizer.py`](../hyperparameter_optimization/base_optimizer.py), [`evaluation/rl_folds.py::build_fold`](../evaluation/rl_folds.py)
- Belgede: [CLAUDE.md](../env/CLAUDE.md)
- İlgili: [Kayan pencereli bölme (wf-v1)](#walk-forward), [Deneme kaydı (N)](#deneme-kaydi), [Eğitim / doğrulama / test penceresi](#pencereler)
- Kaynak: Akiba vd. (2019); Bergstra vd. (2011)

<a id="walk-forward"></a>

### Kayan pencereli bölme (wf-v1)

**Sade:** Verinin zamanda ilerleyen katlara bölünmesi: her katta model geçmişle eğitilir, bir sonraki yılda sınanır.

**Teknik:** wf-v1: 6 kat, test = takvim yılı 2021..2026, doğrulama = bir önceki yıl, eğitim 2013-06'dan genişleyen; purge 0. Tanım değişmez, uzatmak yeni sürümdür (wf-v2). Seçim yalnız doğrulamada (10 kontrol noktası), test seçilen modelle bir kez okunur.

- Kodda: [`evaluation/splits.py::get_spec`](../evaluation/splits.py), [`evaluation/splits.py::fold_frames`](../evaluation/splits.py), [`evaluation/rl_folds.py::build_fold`](../evaluation/rl_folds.py)
- İlgili: [Eğitim / doğrulama / test penceresi](#pencereler), [Kontrol noktası seçimi](#kontrol-noktasi-secimi), [Hiperparametre optimizasyonu](#hpo)
- Kaynak: López de Prado (2018)

<a id="kontrol-noktasi-secimi"></a>

### Kontrol noktası seçimi

**Sade:** Eğitim boyunca kaydedilen ara modellerden hangisinin kullanılacağına karar verme kuralı.

**Teknik:** Tek ajan: eğitim 10 kontrol noktasına bölünür, doğrulamada fazla getiri Sharpe'ı en yüksek olan seçilir ('argmax_validation'); son nokta ('last_checkpoint') de ayrı kayıt olarak yazılır, ikisi de N'ye sayılır. MARL geliştirme koşularında son ağırlıklar kullanılır.

- Kodda: [`evaluation/rl_folds.py::select_checkpoint`](../evaluation/rl_folds.py), [`evaluation/rl_folds.py::checkpoint_grid`](../evaluation/rl_folds.py)
- İlgili: [Kayan pencereli bölme (wf-v1)](#walk-forward), [Deneme kaydı (N)](#deneme-kaydi), [Fazla getiri](#fazla-getiri)

<a id="moody-saffell"></a>

### Moody–Saffell Dₜ terimi

**Sade:** Sharpe oranının her adımdaki değişimini ödül olarak veren klasik yöntem. Bu projede yalnız seçimli bir kol olarak var.

**Teknik:** Moody–Saffell Dₜ türevi; tek ajan ortamında sharpe_term = 'differential' (yalnız reward_type = 'psr'). MARL'da 'differential' ödül terimi yalnız α = 0 ablasyonu içindir. Varsayılan tek ajan terimi EMA Sharpe düzeyidir.

- Kodda: [`env/reward_functions.py::RewardCalculator`](../env/reward_functions.py)
- Belgede: [reward-definition.md](thesis/reward-definition.md)
- İlgili: [EMA Sharpe düzeyi](#ema-sharpe), [Ödül](#odul), [Sharpe oranı](#sharpe), [α (ödül karışımı)](#alfa)
- Kaynak: Moody & Saffell (2001)

**Uyarı:** Varsayılan tek ajan terimi (EMA Sharpe düzeyi) bu terim değildir; ona "diferansiyel Sharpe" denmez (tez kuralı).

<a id="psr-odulu"></a>

### PSR ödülü

**Sade:** Tek ajanlı modelde getiriyi, riski ve işlem maliyetini birlikte tartan karma ödül.

**Teknik:** reward_type = 'psr' (faz 2); ağırlıklar w1–w5, Sharpe terimi sharpe_term ile seçilir. 'simple' ödül portföy değeri değişimidir. Tanımın tek kaynağı reward-definition.md; ödülü değiştiren commit onu da günceller.

- Kodda: [`env/reward_functions.py::RewardCalculator`](../env/reward_functions.py), [`env/reward_functions.py::SimpleRewardCalculator`](../env/reward_functions.py)
- Belgede: [reward-definition.md](thesis/reward-definition.md)
- İlgili: [Ödül](#odul), [EMA Sharpe düzeyi](#ema-sharpe), [Moody–Saffell Dₜ terimi](#moody-saffell)

<a id="atr-kelly"></a>

### ATR ve Kelly pozisyon boyutlandırma

**Sade:** Alınacak hisse miktarını oynaklığa ya da kazanma olasılığına göre ayarlayan seçimli kurallar. Varsayılan olarak kapalıdır.

**Teknik:** use_atr_sizing (ATR tabanlı lot) ve use_kelly (f* = (p·b − q·a)/(a·b), kesirli Kelly), ikisi de varsayılan False; eğitilmiş modeller bozulmasın diye açılmaz.

- Kodda: [`env/trading_env.py::TradingEnv`](../env/trading_env.py)
- İlgili: [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami)
- Kaynak: Kelly (1956)

<a id="turbulans"></a>

### Türbülans endeksi

**Sade:** Piyasanın olağandışı çalkantılı olup olmadığını gösteren gösterge. Eski ortamda gözleme girer ama neredeyse hep tavanda kalır.

**Teknik:** Çok sembollü hesapta kesitsel Mahalanobis uzaklığı, sembol düzeyinde yedek olarak 252 günlük kayan oynaklık × 100. Gözlemde tanh(turbulence/2); bu terim doygun olduğu için eski ortamın her kolunda obs_health açık kalır (K26 = a, beyan).

- Kodda: [`data/technical_indicators.py`](../data/technical_indicators.py), [`evaluation/observation_health.py::observation_health`](../evaluation/observation_health.py)
- İlgili: [Gözlem sağlığı (obs_health)](#gozlem-sagligi), [Teknik göstergeler (MACD, RSI, CCI, ADX)](#teknik-gostergeler), [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami)

<a id="gozlem-sagligi"></a>

### Gözlem sağlığı (obs_health)

**Sade:** Ajanın gördüğü sayıların kırpılıp kırpılmadığını, hep aynı değerde takılıp kalmadığını ve eğitimde görülen aralıkta olup olmadığını ölçen denetim.

**Teknik:** Kapılı bloklarda kırpma ≤ %1, sabit boyut yok, özellik başına doygunluk ≤ %90; features_in_support (havuzlu) ve dims_in_support (her sembol × özellik boyutu ≤ %1). Ölçülür ve kayda yazılır; geçmeyen kol protocol_open alır.

- Kodda: [`evaluation/observation_health.py::observation_health`](../evaluation/observation_health.py), [`evaluation/observation_health.py::dims_in_support`](../evaluation/observation_health.py)
- İlgili: [Türbülans endeksi](#turbulans), [Fiyat ölçeği (price_norm)](#fiyat-olcegi), [Protokol açık (protocol_open)](#protokol-acik), [Gözlem](#gozlem)

<a id="fiyat-olcegi"></a>

### Fiyat ölçeği (price_norm)

**Sade:** Fiyatların ajana verilmeden önce hangi ortalama ve yayılımla küçültüldüğü. Değerlendirme ortamı eğitim ortamının ölçeğini kullanmak zorundadır.

**Teknik:** 'stats': eğitim ortamından price_stats (held-out ortama verilmesi zorunlu, None reddedilir); 'causal': sembolün ≤ t son 252 kapanışının ölçeği (K25 b). Kökeni beyanla değil ölçümle etiketlenir (rl_eval_protocol).

- Kodda: [`env/trading_env.py::TradingEnv`](../env/trading_env.py), [`evaluation/trial_registry.py`](../evaluation/trial_registry.py)
- Belgede: [CLAUDE.md](../env/CLAUDE.md)
- İlgili: [Gözlem sağlığı (obs_health)](#gozlem-sagligi), [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami)

<a id="eksik-veri-modu"></a>

### Eksik veri modu

**Sade:** Bir hissenin o gün verisi yoksa ortamın ne yapacağını belirleyen ayar.

**Teknik:** missing_data_mode = 'legacy' (varsayılan, eğitilmiş modeller): eksik hücrede önceki sembolün bloğunu kopyalar ve fiyatı 0 sayar (bilinen hata, dondurulmuş). 'own_last': sembolü kendi son gözlemiyle taşır, o hücrede işlem yapmaz; yeni eğitim ve HPO bunu kullanır.

- Kodda: [`env/trading_env.py::TradingEnv`](../env/trading_env.py)
- İlgili: [Eksik hücre](#eksik-hucre), [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami), [Hiperparametre optimizasyonu](#hpo)

<a id="kat-tahmin"></a>

## Tahmin hattı

<a id="auc"></a>

### AUC

**Sade:** Bir tahminin yükselen ve düşen günleri ne kadar iyi ayırdığı. 0,5 yazı-tura, 1 kusursuz demektir.

**Teknik:** ROC eğrisi altındaki alan; p_up_cal ile yön etiketi arasında.

- Kodda: [`evaluation/calibration.py::auc`](../evaluation/calibration.py)
- İlgili: [Bilgi kapısı (İ3)](#bilgi-kapisi), [Pozitif kontrol (PC060-K12)](#pozitif-kontrol)

<a id="pit-tablosu"></a>

### P tablosu (pit_v1 / pit_v2)

**Sade:** Her gün için, yalnız o gün bilinebilecek veriyle üretilmiş tahminlerin tablosu.

**Teknik:** Nokta-zamanlı ve örneklem dışı tahmin tablosu. pit_v1 kapanıştan kapanışa (Cₜ → Cₜ₊₁) yönü, pit_v2 (P5) K12 ufkundaki açılıştan açılışa (Oₜ₊₁ → Oₜ₊₂) yönü tahmin eder.

- Kodda: [`prediction/pit_table.py`](../prediction/pit_table.py)
- İlgili: [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [K12 yürütme ufku](#k12), [Bilgi kapısı (İ3)](#bilgi-kapisi)

<a id="bilgi-kapisi"></a>

### Bilgi kapısı (İ3)

**Sade:** Bir tahmin sinyalinin gerçekten bilgi taşıyıp taşımadığını, kullanılmadan önce sınayan test.

**Teknik:** Sembol başına: AUC için tek taraflı yüzdelik blok bootstrap p (H0: AUC ≤ 0,5), 30 sembol tek aile BH; BSS (taban oranı test dışından) ve MDE raporlanır. Kapıyı geçmeyen sinyalin güveni "sıcaklık ölçekli, ayırt edici değil" diye anılır, "kalibre" denmez.

- Kodda: [`prediction/pit_table.py::evaluate_gate`](../prediction/pit_table.py)
- İlgili: [AUC](#auc), [Benjamini–Hochberg (BH-FDR)](#bh-fdr), [En küçük tespit edilebilir etki (MDE)](#mde), [Kalibrasyon (sıcaklık)](#kalibrasyon)

<a id="kalibrasyon"></a>

### Kalibrasyon (sıcaklık)

**Sade:** Tahminin söylediği olasılığın gerçekleşen sıklığa uyması: "%60" denen günlerin yaklaşık %60'ı gerçekten yükselmeli.

**Teknik:** Sıcaklık ölçekleme: tek parametre T > 0, doğrulama kümesinde NLL en küçüklenir, p = σ(z / T). ECE ve Brier ile ölçülür.

- Kodda: [`evaluation/calibration.py::fit_temperature`](../evaluation/calibration.py), [`evaluation/calibration.py::apply_temperature`](../evaluation/calibration.py)
- İlgili: [Bilgi kapısı (İ3)](#bilgi-kapisi), [AUC](#auc)

<a id="tahmin-ensemble"></a>

### Tahmin topluluğu (stacking ensemble)

**Sade:** Birkaç farklı tahmin modelinin çıktısını birleştirip tek bir tahmin üreten yapı. Canlı tahmin sayfası bunu kullanır.

**Teknik:** StackingEnsemble: taban modeller XGBoost, LightGBM, CatBoost, BiLSTM, TFT; meta-öğrenici bunların örneklem dışı (OOF) tahminleriyle eğitilir. Kronolojik 60/20/20 bölme. MARL ajanlarına giden sinyal bu sınıftan değil P tablosundan gelir.

- Kodda: [`prediction/models/ensemble.py::StackingEnsemble`](../prediction/models/ensemble.py)
- Belgede: [CLAUDE.md](../prediction/CLAUDE.md)
- İlgili: [Gradyan artırma modelleri (XGBoost, LightGBM, CatBoost)](#gbm-modelleri), [BiLSTM](#bilstm), [TFT](#tft), [Meta-öğrenici](#meta-ogrenici), [Örneklem dışı tahmin (OOF)](#oof), [P tablosu (pit_v1 / pit_v2)](#pit-tablosu)
- Kaynak: Wolpert (1992)

**Uyarı:** "Sızıntı yok" denmez: dağıtım öncesi %80 yeniden uyumda meta-öğrenici örneklem içi tahminlerle yeniden uyduruluyor (K19).

<a id="gbm-modelleri"></a>

### Gradyan artırma modelleri (XGBoost, LightGBM, CatBoost)

**Sade:** Çok sayıda küçük karar ağacını art arda ekleyerek tahmin yapan modeller. P tablosunun taban modelleri bunlardır.

**Teknik:** XGBoostModel, LightGBMModel, CatBoostModel (BasePredictionModel alt sınıfları). P tablosunda V1_MODELS = xgboost, lightgbm, catboost; yalnız fiyat/hacim özellikleri, her çapada yeniden eğitim.

- Kodda: [`prediction/models/xgboost_model.py::XGBoostModel`](../prediction/models/xgboost_model.py), [`prediction/models/lightgbm_model.py::LightGBMModel`](../prediction/models/lightgbm_model.py), [`prediction/models/catboost_model.py::CatBoostModel`](../prediction/models/catboost_model.py), [`prediction/pit_table.py::V1_MODELS`](../prediction/pit_table.py)
- İlgili: [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble), [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [Çapa (yeniden eğitim noktası)](#capa)
- Kaynak: Chen & Guestrin (2016); Ke vd. (2017); Prokhorenkova vd. (2018)

<a id="bilstm"></a>

### BiLSTM

**Sade:** Zaman serisini hem ileri hem geri yönde okuyan bir derin öğrenme modeli.

**Teknik:** Çift yönlü LSTM; getiri tahminine ek olarak bir yön başı (_predict_direction_raw) taşır. Eğitim penceresi satır i'yi içerir (B11); yön etiketi ham hedeften.

- Kodda: [`prediction/models/lstm_model.py::BiLSTMModel`](../prediction/models/lstm_model.py)
- İlgili: [TFT](#tft), [Yön başı](#yon-basi), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble)
- Kaynak: Hochreiter & Schmidhuber (1997); Schuster & Paliwal (1997)

<a id="tft"></a>

### TFT

**Sade:** Hangi girdinin ne zaman önemli olduğunu dikkat mekanizmasıyla seçen bir derin öğrenme tahmin modeli.

**Teknik:** Temporal Fusion Transformer: geçitli artık ağlar, değişken seçim ağı, dikkat; yön başı taşır. Gruplanmış VSN gibi hız ayarları golden'ı değiştirdiği için varsayılan kapalı.

- Kodda: [`prediction/models/tft_model.py::TFTModel`](../prediction/models/tft_model.py)
- İlgili: [BiLSTM](#bilstm), [Yön başı](#yon-basi), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble), [Dikkat (attention)](#dikkat)
- Kaynak: Lim vd. (2021)

<a id="yon-basi"></a>

### Yön başı

**Sade:** Derin öğrenme modellerinin, fiyat tahminine ek olarak "yükselir mi düşer mi" sorusuna verdiği ayrı çıktı.

**Teknik:** BiLSTM/TFT _predict_direction_raw. confidence_mode = 'legacy' (varsayılan): yön fiyat işaretinden, güven yön başlarının ortalamasından max(p, 1 − p); ikisi zıt yönü gösterebilir (Y4, head_vs_price_conflict_rate ile ölçülür).

- Kodda: [`prediction/models/base.py::BasePredictionModel`](../prediction/models/base.py), [`prediction/models/ensemble.py::CONFIDENCE_MODES`](../prediction/models/ensemble.py)
- İlgili: [BiLSTM](#bilstm), [TFT](#tft), [p_up_cal (kalibre edilmiş yükseliş olasılığı)](#p-up-cal)

<a id="meta-ogrenici"></a>

### Meta-öğrenici

**Sade:** Taban modellerin tahminlerini girdi olarak alıp hangisine ne kadar güvenileceğini öğrenen üst model.

**Teknik:** Topluluk: Ridge ya da XGBoost (meta_learner_type). P tablosu: lojistik meta, yalnız geçmiş blokların örneklem dışı tahminlerinde uydurulur (K19: %80 yeniden uyum ve örneklem içi meta yok).

- Kodda: [`prediction/models/ensemble.py::StackingEnsemble`](../prediction/models/ensemble.py), [`prediction/pit_table.py::meta_layer`](../prediction/pit_table.py)
- İlgili: [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble), [Örneklem dışı tahmin (OOF)](#oof), [P tablosu (pit_v1 / pit_v2)](#pit-tablosu)
- Kaynak: Wolpert (1992); Hoerl & Kennard (1970)

<a id="oof"></a>

### Örneklem dışı tahmin (OOF)

**Sade:** Bir modelin, eğitimde görmediği veriler için ürettiği tahmin. Üst model yalnız bunlarla eğitilirse kendini kandırmaz.

**Teknik:** Out-of-fold: topluluğun 60/20/20 bölmesinde orta dilim meta-eğitim için ayrılır. P tablosunda taban eğitimi label_date < çapa ile sınırlanır; meta ve sıcaklık yalnız geçmiş blokların OOF tahminlerinde.

- Kodda: [`prediction/pit_table.py::base_predictions`](../prediction/pit_table.py), [`prediction/pit_table.py::meta_layer`](../prediction/pit_table.py)
- İlgili: [Meta-öğrenici](#meta-ogrenici), [Çapa (yeniden eğitim noktası)](#capa), [Arındırma ve ambargo (purge / embargo)](#arindirma)

<a id="capa"></a>

### Çapa (yeniden eğitim noktası)

**Sade:** P tablosunda modellerin yeniden eğitildiği tarihler. Her çapada yalnız o tarihe kadar gerçekleşmiş etiketler kullanılır.

**Teknik:** FIRST_ANCHOR = 2012-01-01, RETRAIN_EVERY = 63 seans; satırın bloğu kendinden önceki son çapadır. Sıcaklık çapa başına 30 sembolde havuzlu; informative_asof çapa anındaki bilgi kapısı.

- Kodda: [`prediction/pit_table.py::make_anchors`](../prediction/pit_table.py), [`prediction/pit_table.py::RETRAIN_EVERY`](../prediction/pit_table.py)
- İlgili: [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [Örneklem dışı tahmin (OOF)](#oof), [Bilgi kapısı (İ3)](#bilgi-kapisi), [Kalibrasyon (sıcaklık)](#kalibrasyon)

<a id="arindirma"></a>

### Arındırma ve ambargo (purge / embargo)

**Sade:** Eğitim ile sınama verisi arasına boşluk bırakmak. Etiketi sınama dönemine taşan eğitim satırları böylece dışarıda kalır.

**Teknik:** Tahmin eğiticisi purge_days = 5, embargo_days = 3 (WalkForwardTrainer). P tablosunda arındırma ufku label_steps: pit_v1'de h, pit_v2'de h + 1 seans (taban erken durdurma ve meta yarıları dahil).

- Kodda: [`prediction/trainer.py::WalkForwardTrainer`](../prediction/trainer.py), [`prediction/pit_table.py::label_steps`](../prediction/pit_table.py)
- İlgili: [Örneklem dışı tahmin (OOF)](#oof), [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [K12 yürütme ufku](#k12), [Kayan pencereli bölme (wf-v1)](#walk-forward)
- Kaynak: López de Prado (2018)

<a id="tahmin-ufku"></a>

### Tahmin ufku

**Sade:** Tahminin ne kadar ileriyi hedeflediği: bir gün sonrası ya da bir hafta sonrası gibi.

**Teknik:** horizon 'daily' (h = 1) ya da 'weekly' (h = 5 seans). pit_v1 hedefi Cₜ₊ₕ / Cₜ, pit_v2 (K12) hedefi Oₜ₊₁₊ₕ / Oₜ₊₁; etiket tarihi pit_v2'de t + h + 1.

- Kodda: [`prediction/pit_table.py::horizon_steps`](../prediction/pit_table.py), [`prediction/pit_table.py::TARGETS`](../prediction/pit_table.py)
- İlgili: [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [K12 yürütme ufku](#k12), [Arındırma ve ambargo (purge / embargo)](#arindirma)

<a id="p-up-cal"></a>

### p_up_cal (kalibre edilmiş yükseliş olasılığı)

**Sade:** Tahminin "fiyat yükselecek" olasılığı, sıcaklık ölçeklemesinden geçirilmiş haliyle. Ajanların gördüğü tahmin sinyalinin ana parçasıdır.

**Teknik:** Meta lojistik p(UP), sıcaklık T ile ölçeklenir; yön = 1[p_cal ≥ 0,5]. P tablosunda PRED_FIELDS'in ilk alanı. Bilgi kapısını geçmeyen bir p_up_cal için "kalibre" denmez, "sıcaklık ölçekli" denir.

- Kodda: [`marl/obs_spec.py::PRED_FIELDS`](../marl/obs_spec.py), [`prediction/models/ensemble.py::StackingEnsemble`](../prediction/models/ensemble.py)
- İlgili: [Kalibrasyon (sıcaklık)](#kalibrasyon), [Bilgi kapısı (İ3)](#bilgi-kapisi), [S-obs](#s-obs), [Yön başı](#yon-basi)

<a id="ece"></a>

### ECE

**Sade:** Söylenen olasılıklarla gerçekleşen sıklıklar arasındaki ortalama uyumsuzluk. Küçük olması iyi kalibrasyon demektir.

**Teknik:** Expected Calibration Error: 15 kutu, kutu ağırlıklı |ortalama p − gerçekleşen oran|. Tracker'daki 5 kutulu compute_calibration bununla karıştırılmaz.

- Kodda: [`evaluation/calibration.py::ece`](../evaluation/calibration.py), [`evaluation/calibration.py::reliability`](../evaluation/calibration.py)
- İlgili: [Kalibrasyon (sıcaklık)](#kalibrasyon), [Brier skoru ve BSS](#brier)
- Kaynak: Naeini vd. (2015); Guo vd. (2017)

<a id="brier"></a>

### Brier skoru ve BSS

**Sade:** Olasılık tahminlerinin ortalama kare hatası ve bunun "hep ortalamayı söylemek"e göre ne kadar iyi olduğu. BSS sıfırın üstündeyse tahmin bir şey katıyordur.

**Teknik:** Brier = ortalama (p − y)²; BSS = 1 − Brier / Brier(taban oranı), taban oranı test dışından. Bilgi kapısının üç koşulundan biri BSS > 0'dır.

- Kodda: [`evaluation/calibration.py::brier`](../evaluation/calibration.py), [`evaluation/calibration.py::brier_skill`](../evaluation/calibration.py)
- İlgili: [ECE](#ece), [Bilgi kapısı (İ3)](#bilgi-kapisi), [Kalibrasyon (sıcaklık)](#kalibrasyon)
- Kaynak: Brier (1950)

<a id="tats"></a>

### TATS

**Sade:** Bir eğilim sınıflandırıcısının çıktısıyla getiri tahminini düzelten yöntem. Bu projede uyarlanmış haliyle kullanılır.

**Teknik:** TATSCorrector (correction_factor, flat_threshold); uyum girdileri ham kapanıştan. Uyarlama işareti korur, makaledeki gibi yönü çevirmez; test metriğinde TATS'lı ve TATS'sız sonuç ayrı tutulur.

- Kodda: [`prediction/tats.py::TATSCorrector`](../prediction/tats.py)
- Belgede: [CLAUDE.md](../prediction/CLAUDE.md)
- İlgili: [Yön isabeti (TDA ve direction_accuracy)](#tda), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble)
- Kaynak: Kazemdehbashi (2026)

<a id="tda"></a>

### Yön isabeti (TDA ve direction_accuracy)

**Sade:** Tahminin yönü doğru bilme oranı. Kodda iki farklı tanım olduğu için hangisinin kullanıldığı her zaman yazılır.

**Teknik:** compute_metrics'in direction_accuracy'si fark tabanlıdır (np.diff) ve makaledeki TDA değildir; TATS karşılaştırmasında ensemble_test_metrics['tats'][*]['tda'] kullanılır.

- Kodda: [`prediction/models/base.py::BasePredictionModel`](../prediction/models/base.py)
- Belgede: [CLAUDE.md](../evaluation/CLAUDE.md)
- İlgili: [TATS](#tats)

<a id="iceemdan"></a>

### ICEEMDAN gürültü ayrıştırma

**Sade:** Fiyat serisini farklı hızlarda dalgalanan bileşenlere ayırıp gürültüyü süzen yöntem.

**Teknik:** ICEEMDANProcessor.decompose. Mevcut uygulama seriyi bütünüyle ayrıştırdığı için ileriye bakış taşır; tezde katkı olarak sunulmaz (vision-and-scope §5.4).

- Kodda: [`prediction/iceemdan_processor.py::ICEEMDANProcessor`](../prediction/iceemdan_processor.py)
- Belgede: [vision-and-scope.md](thesis/vision-and-scope.md)
- İlgili: [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble), [Nokta-zamanlılık (B0)](#nokta-zamanlilik)
- Kaynak: Colominas vd. (2014)

<a id="shap"></a>

### SHAP açıklaması

**Sade:** Bir tahminde hangi girdinin ne kadar pay sahibi olduğunu gösteren açıklama yöntemi.

**Teknik:** ModelExplainer (explain_prediction, explain_global). SHAP ucu predict_next'ten farklı bir girdi matrisi kuruyor (Y30 açık); açıklama canlı tahminle birebir aynı girdiye dayanmayabilir.

- Kodda: [`prediction/explainability.py::ModelExplainer`](../prediction/explainability.py)
- İlgili: [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble)

<a id="tahmin-izleyici"></a>

### Tahmin izleyici (tracker)

**Sade:** Canlı tahminleri kaydedip, gerçekleşen fiyat geldikçe ne kadar isabetli olduklarını izleyen bileşen.

**Teknik:** PredictionTracker; kurallar tek yerde (tracker_rules). prediction_date = feature_date + h iş günü; gerçekleşen kapanış feature_date'ten sonraki h. gerçek seans. Geçersiz kayıt (invalid_reason) ve bayat özellikli kayıt değerlendirilmez.

- Kodda: [`prediction/tracker.py::PredictionTracker`](../prediction/tracker.py), [`prediction/tracker_rules.py::invalid_reason`](../prediction/tracker_rules.py), [`prediction/tracker_rules.py::actual_after_sessions`](../prediction/tracker_rules.py)
- Belgede: [CLAUDE.md](../prediction/CLAUDE.md)
- İlgili: [Tahmin ufku](#tahmin-ufku), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble)

**Uyarı:** GOLD_GRAM_TRY'de gerçekleşen kapanış GC=F'ten (ons, USD) çekiliyor; o sembolde isabet anlamsız (Y31 açık).

<a id="mask-reason"></a>

### Maske nedeni (mask_reason)

**Sade:** Tahmin tablosunda bir hücre için tahmin olmadığında, nedeninin açıkça yazıldığı alan. Eksik tahmin sessizce yazı-turaya çevrilmez.

**Teknik:** MASK_REASONS = ok, no_data, no_features, insufficient_history, model_failed, meta_insufficient. MARL gözleminde pred_ok bayrağına dönüşür; tek ajan ortamında env_prediction_features maskeli hücreyi nötr (0) yapar.

- Kodda: [`prediction/pit_table.py::MASK_REASONS`](../prediction/pit_table.py), [`prediction/pit_table.py::env_prediction_features`](../prediction/pit_table.py)
- İlgili: [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [Eksik hücre](#eksik-hucre), [S-obs](#s-obs)

<a id="kat-veri"></a>

## Veri

<a id="hmm-rejim"></a>

### HMM rejimi (ikinci tanım)

**Sade:** Piyasanın sakin ya da çalkantılı döneminin, getirilerden istatistiksel olarak çıkarılan ikinci bir etiketi. Kural tabanlı etiketin tekrarı değil, daha çok oynaklık durumunu ölçer.

**Teknik:** Saklı Markov modeli (rejime göre değişen ortalama + varyans), USD bazlı XU100 günlük getiri; 63 seansta bir genişleyen yeniden uyum, yalnız filtrelenmiş olasılık; durum sayısı BIC ile (K = 3). İki hücre: en sakin durum ≥ 0,5 ↔ geri kalanlar (`calm_vs_rest`, K-M6-5).

- Kodda: [`evaluation/regimes_hmm.py::PRIMARY_TWO_CELL`](../evaluation/regimes_hmm.py)
- Belgede: [phase-10-m6-design.md](development/phase-10-m6-design.md)
- İlgili: [Piyasa rejimi](#rejim), [Nokta-zamanlılık (B0)](#nokta-zamanlilik)
- Kaynak: Hamilton (1989); Nystrup vd. (2018)

<a id="bist30"></a>

### BIST-30

**Sade:** Borsa İstanbul'un en büyük 30 şirketinin hissesinden oluşan endeks. Sistem bu hisselerde işlem yapar.

**Teknik:** Sembol evreni tek kaynaktan okunur; panel sırası bu kaynağın sırasıdır (sorted() uygulanmaz). Üyelik değişimleri ve KAP sektör grupları ayrıca izlenir.

- Kodda: [`data/bist30_symbols.py`](../data/bist30_symbols.py)
- İlgili: [Panel](#panel)

<a id="panel"></a>

### Panel

**Sade:** Bütün hisselerin günlük fiyat ve göstergelerinin tarih × hisse tablosu.

**Teknik:** data/marl/panel_m2.npz (SHA sabit); build_arrays [T, N] diziler + risksiz faiz. Eksik hücre atılmaz: fiyat taşınır ve hisse o gün dondurulur. Fiyatlar yfinance düzeltilmiş.

- Kodda: [`marl/panel.py::build_arrays`](../marl/panel.py), [`marl/accounting.py::fill_prices`](../marl/accounting.py)
- İlgili: [Risksiz faiz (A8)](#risksiz-faiz), [Listeleme kilidi](#listeleme-kilidi), [BIST-30](#bist30)

<a id="listeleme-kilidi"></a>

### Listeleme kilidi

**Sade:** Borsaya yeni giren bir hissede ilk haftalarda işlem yapılmaması kuralı.

**Teknik:** LISTING_LOCKOUT = 20 seans; ortam ve canlı karar aynı işlem yapılabilirlik maskesini kullanır.

- Kodda: [`marl/accounting.py::LISTING_LOCKOUT`](../marl/accounting.py)
- İlgili: [Panel](#panel), [Satış engeli (taban açılış)](#satis-engeli)

<a id="satis-engeli"></a>

### Satış engeli (taban açılış)

**Sade:** Bir hisse güne en düşük fiyat sınırından (taban) açılırsa o gün satılamayacağını varsayan kural.

**Teknik:** limit_down_gaps(dates): dönem bağımlı fiyat marjı − 0,005. 2020-03-13'ten beri marj %10 (Borsa İstanbul 2020/20 sayılı duyuru; 2021+ için yeni giriş yok). Takvim 2026-11-09'a kadar doğrulandı; sonrası için hata verir (önce kaynakla doğrulanmalı). Tavan alım engeli yoktur.

- Kodda: [`marl/accounting.py::limit_down_gaps`](../marl/accounting.py)
- Belgede: [phase-10-m2-design.md](development/phase-10-m2-design.md)
- İlgili: [Listeleme kilidi](#listeleme-kilidi), [Panel](#panel)

<a id="nokta-zamanlilik"></a>

### Nokta-zamanlılık (B0)

**Sade:** Geçmişteki bir gün için yalnız o gün gerçekten bilinebilecek veriyi kullanmak.

**Teknik:** B0 kesme değişmezliği: t'den sonraki veri bozulduğunda t'deki özellik ya da tahmin değişmemelidir; testlerde mutasyonla denetlenir. Bilinen metodolojik açıklar roadmap'te listelidir.

- Belgede: [roadmap.md](development/roadmap.md)
- İlgili: [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [Eğitim / doğrulama / test penceresi](#pencereler)

**Uyarı:** "Sızıntı yok" denmez; doğrulanmış sızıntı kaynakları var (roadmap "Bilinen Metodolojik Açıklar").

<a id="ohlcv"></a>

### OHLCV

**Sade:** Bir hissenin bir gündeki açılış, en yüksek, en düşük ve kapanış fiyatı ile işlem hacmi.

**Teknik:** Open, High, Low, Close, Volume; yfinance'ten düzeltilmiş (auto_adjust) olarak çekilir. MARL ortamı açılışta yürütür (K12), değerlemeyi kapanışla yapar.

- Kodda: [`data/data_fetcher.py::DataFetcher`](../data/data_fetcher.py)
- İlgili: [yfinance](#yfinance), [Düzeltilmiş fiyat](#duzeltilmis-fiyat), [K12 yürütme ufku](#k12), [Panel](#panel)

<a id="yfinance"></a>

### yfinance

**Sade:** Hisse fiyatlarının ve bazı küresel göstergelerin indirildiği açık veri kaynağı.

**Teknik:** DataFetcher.fetch_stock_data (yeniden deneme, önbellek, %80 kapsam denetimi) ve fetch_incremental (yalnız eksik günler). Seans kapanmadan dönen OHLC'si boş taslak satır yalnız yeni veride süzülür; diske yazılırsa o seans bir daha çekilmez.

- Kodda: [`data/data_fetcher.py::DataFetcher`](../data/data_fetcher.py), [`data/data_fetcher.py::fetch_incremental`](../data/data_fetcher.py)
- Belgede: [CLAUDE.md](../app/CLAUDE.md)
- İlgili: [OHLCV](#ohlcv), [Düzeltilmiş fiyat](#duzeltilmis-fiyat), [Küresel makro (VIX, US10Y, DXY)](#kuresel-makro)

<a id="duzeltilmis-fiyat"></a>

### Düzeltilmiş fiyat

**Sade:** Temettü ve bölünmelerin etkisi geçmişe yansıtılmış fiyat. Yeni bir temettü geldiğinde geçmiş fiyatların hepsi yeniden ölçeklenebilir.

**Teknik:** yfinance auto_adjust. Tam yeniden indirme geçmişi değiştirir; bir sonucu yeniden üretirken veri yeniden indirilmez, panel SHA-256'sı ve anlık görüntüyle çalışılır. Düzeltilmiş fiyatla lot ya da fiyat seviyesi iddiası kurulmaz.

- Belgede: [CLAUDE.md](../env/CLAUDE.md)
- İlgili: [OHLCV](#ohlcv), [yfinance](#yfinance), [Panel](#panel)

<a id="teknik-gostergeler"></a>

### Teknik göstergeler (MACD, RSI, CCI, ADX)

**Sade:** Fiyat geçmişinden hesaplanan, eğilim ve momentum özetleyen sayılar. Ajanın gözleminde fiyatın yanında yer alır.

**Teknik:** macd (+ sinyal, histogram), rsi, cci, adx ve turbulence. Isınma satırları: eski hat bfill ile doldurur, RL veri kapısı bfill imzalı baş bloğu atar (B7). ADX için ayrı terime bakın.

- Kodda: [`data/technical_indicators.py`](../data/technical_indicators.py), [`data/integrity.py::INDICATOR_COLS`](../data/integrity.py)
- İlgili: [ADX (Wilder)](#adx), [Türbülans endeksi](#turbulans), [Gözlem](#gozlem), [RL veri kapısı](#veri-kapisi)

<a id="adx"></a>

### ADX (Wilder)

**Sade:** Bir eğilimin ne kadar güçlü olduğunu ölçen teknik gösterge.

**Teknik:** Kayıtlı CSV'deki ADX eski hesabın sabit 0'ıdır (indeks hizalama hatası; eski modeller böyle eğitildi). Rota ve MARL paneli ADX'i with_adx(df, 'wilder') ile yeniden hesaplar ve meta'ya indicator_spec yazar; eski ADX build_arrays'te reddedilir (K18).

- Kodda: [`data/technical_indicators.py::with_adx`](../data/technical_indicators.py), [`marl/panel.py::build_arrays`](../marl/panel.py)
- İlgili: [Teknik göstergeler (MACD, RSI, CCI, ADX)](#teknik-gostergeler), [Model meta dosyası](#model-meta)

<a id="veri-kapisi"></a>

### RL veri kapısı

**Sade:** Fiyat verisi ajana gitmeden önce bozuk satırları yakalayan ve öğrenme penceresini sabitleyen denetim.

**Teknik:** prepare_rl_panel: pencere RL_DATA_START = 2013-06-01 (K17); pencere içinde fiyat artefaktı varsa DataIntegrityError (sessizce temizlenmez); hayalet gün ve bayat hücre atılır, hücre eksik sayılır; gösterge ısınması atılır.

- Kodda: [`data/integrity.py::prepare_rl_panel`](../data/integrity.py), [`data/integrity.py::RL_DATA_START`](../data/integrity.py)
- Belgede: [CLAUDE.md](../env/CLAUDE.md)
- İlgili: [Hayalet gün ve bayat hücre](#hayalet-gun), [Eksik hücre](#eksik-hucre), [Teknik göstergeler (MACD, RSI, CCI, ADX)](#teknik-gostergeler), [Eğitim / doğrulama / test penceresi](#pencereler)

<a id="hayalet-gun"></a>

### Hayalet gün ve bayat hücre

**Sade:** Borsanın aslında kapalı olduğu halde veride işlem günü gibi görünen günler ve tek bir hissede donmuş kalmış satırlar.

**Teknik:** Hayalet gün: aktif satırların ≥ %90'ı düz (volume = 0 ve O = H = L = C). Bayat hücre: tek sembolün düz satırı; atılır ve eksik sayılır. "Önceki kapanışa eşit" şartı aranmaz, çünkü yfinance kapalı günde düzeltilmiş fiyatı farklı yazabiliyor.

- Kodda: [`data/integrity.py::find_ghost_days`](../data/integrity.py), [`data/integrity.py::find_stale_cells`](../data/integrity.py)
- İlgili: [RL veri kapısı](#veri-kapisi), [Eksik hücre](#eksik-hucre)

<a id="eksik-hucre"></a>

### Eksik hücre

**Sade:** Bir hissenin bir gün için verisinin olmaması. Hisse borsaya sonradan girdiyse bu yapısaldır, hata değildir.

**Teknik:** MARL: hücre atılmaz, fill_prices fiyatı taşır ve hisse o gün dondurulur; gözlemde feat_ok bayrağı. Tahmin tablosunda mask_reason ile işaretlenir (sessiz 0,5 yok). Tek ajan ortamında davranışı eksik veri modu belirler.

- Kodda: [`marl/accounting.py::fill_prices`](../marl/accounting.py), [`prediction/pit_table.py::MASK_REASONS`](../prediction/pit_table.py)
- İlgili: [Panel](#panel), [Eksik veri modu](#eksik-veri-modu), [Listeleme kilidi](#listeleme-kilidi), [RL veri kapısı](#veri-kapisi)

<a id="evds"></a>

### EVDS (TCMB)

**Sade:** Türkiye Cumhuriyet Merkez Bankası'nın faiz, enflasyon gibi seriler sunduğu veri sistemi.

**Teknik:** Politika faizi TP.BISPOLFAIZ.TUR, TÜFE TP.TUKFIY2025.GENEL, ÜFE TP.TUFE1YI.T1, risksiz faiz TP.APIFON4. Tek istekte en fazla 1000 satır (parçalanır). Eski kodlar (TP.YSSK.A01, TP.FG.G0) EVDS'de yoktur.

- Kodda: [`data/macro_fetcher.py::MacroDataFetcher`](../data/macro_fetcher.py)
- Belgede: [CLAUDE.md](../data/CLAUDE.md)
- İlgili: [Makro veri (yurt içi)](#makro-veri), [Risksiz faiz (A8)](#risksiz-faiz)

<a id="makro-veri"></a>

### Makro veri (yurt içi)

**Sade:** Faiz ve enflasyon gibi ekonominin genel durumunu gösteren seriler. Yalnız açıklandıkları günden sonra kullanılır.

**Teknik:** Nokta-zamanlı (B3): ay m TÜFE/ÜFE'si m+1'in 6'sından, politika faizi m+1'in 1'inden geçerli; bfill yok (B1). Olay zamanlıdır, vintage değil. Makro seri 2018'de başlar; tek ajan faz 2 ortamı 2013–2017'de makro gözlemi 0 görür (bilinen açık).

- Kodda: [`data/macro_fetcher.py::MacroDataFetcher`](../data/macro_fetcher.py)
- Belgede: [CLAUDE.md](../data/CLAUDE.md)
- İlgili: [EVDS (TCMB)](#evds), [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [Küresel makro (VIX, US10Y, DXY)](#kuresel-makro)

<a id="kuresel-makro"></a>

### Küresel makro (VIX, US10Y, DXY)

**Sade:** Dünya piyasalarındaki korku, ABD faizi ve dolar gücünü gösteren üç seri. Yalnız tahmin hattına girer, ajanın gözlemine girmez.

**Teknik:** yfinance'ten (vix, us10y, dxy; ayrıca usd_try, eur_try, bist100_index). RL state space'e eklenmez: eğitilmiş modellerin gözlem boyutu bozulmasın diye macro_features sabit kodlanmaz.

- Kodda: [`data/macro_fetcher.py::MacroDataFetcher`](../data/macro_fetcher.py)
- İlgili: [Makro veri (yurt içi)](#makro-veri), [yfinance](#yfinance), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble)

<a id="endeks-serileri"></a>

### Endeks serileri (XU100, XU030)

**Sade:** Borsa İstanbul'un 100 ve 30 hisselik endekslerinin günlük kapanışları. Karşılaştırma ve piyasa dönemi etiketi için kullanılır.

**Teknik:** XU100 uzun seri data/macro/xu100.csv (2005..; |r| > %15 sıçramada yazılmaz); XU030 fiyat endeksi (temettüsüz) baseline için. İkisi de kaynak JSON ve SHA ile.

- Kodda: [`data/index_fetcher.py::fetch_index_close`](../data/index_fetcher.py), [`data/index_fetcher.py::check_index_series`](../data/index_fetcher.py), [`evaluation/baselines.py::load_xu030`](../evaluation/baselines.py)
- İlgili: [Piyasa rejimi](#rejim), [1/N (eşit ağırlık)](#baseline-1n), [Al-tut](#al-tut)

<a id="rejim"></a>

### Piyasa rejimi

**Sade:** Piyasanın yükselişte, düşüşte ya da yatay olduğu dönemlerin etiketi. Etiket yalnız o güne kadarki veriyle verilir.

**Teknik:** XU100 200 günlük eğilim × 60 günlük oynaklık, genişleyen tercil, ≤ t (B0 testli). Birincil etiket USD bazlı XU100 (XU100 / USDTRY_{t−1}); nominal TL ve 5 yıllık kayan tercil duyarlılık kolları. Gerçekleşen getiriye koşullu rejim yazılmaz.

- Kodda: [`evaluation/regimes.py::regime_labels`](../evaluation/regimes.py), [`evaluation/regimes.py::load_primary_labels`](../evaluation/regimes.py)
- İlgili: [Endeks serileri (XU100, XU030)](#endeks-serileri), [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [Getiri birimi (nominal / reel / USD)](#getiri-birimi)

<a id="evren"></a>

### Sembol evreni (E0)

**Sade:** Sistemin işlem yaptığı 30 hissenin dondurulmuş listesi. Bugünkü gerçek BIST-30 üyeliğiyle aynı değildir.

**Teknik:** Tek kaynak data/universe/bist30_panel.json (RL panelinin 30 hissesi, panel sırasıyla). Elle liste yazılmaz; evren değişirse scripts/freeze_universe.py yeni sürüm adıyla. Tarihsel üyelik (E1) ve sektör grupları (E2) ayrı terimlerdir.

- Kodda: [`data/bist30_symbols.py::get_symbols`](../data/bist30_symbols.py), [`scripts/freeze_universe.py`](../scripts/freeze_universe.py)
- Belgede: [CLAUDE.md](../data/CLAUDE.md)
- İlgili: [BIST-30](#bist30), [Tarihsel üyelik (E1)](#tarihsel-uyelik), [Sektör grupları (E2)](#sektor-gruplari)

<a id="tarihsel-uyelik"></a>

### Tarihsel üyelik (E1)

**Sade:** Geçmişte hangi hissenin hangi dönemde BIST-30'da olduğu bilgisi. Bugünkü listeyi geçmişe uygulamak hayatta kalma yanlılığı yaratır.

**Teknik:** scripts/build_membership.py çeyreklik Borsa İstanbul duyurularından; universe_mask(date, mode) 'strict' (bilinmeyen gün None), 'count_checked' (sezgisel) ya da 'raw'. Dönem içi olaylar KAP'tan izlenir; iş sürüyor.

- Kodda: [`data/bist30_symbols.py::universe_mask`](../data/bist30_symbols.py), [`scripts/build_membership.py`](../scripts/build_membership.py)
- Belgede: [CLAUDE.md](../data/CLAUDE.md)
- İlgili: [Sembol evreni (E0)](#evren), [BIST-30](#bist30), [KAP](#kap)

<a id="sektor-gruplari"></a>

### Sektör grupları (E2)

**Sade:** Hisselerin faaliyet alanına göre gruplanması. Ajanlar arası iletişim grafiği bu gruplardan kurulur.

**Teknik:** data/universe/sectors.csv, KAP alt sektörlerinden (satır başına URL, erişim tarihi, ham sayfa SHA-256); evrende 2'den az üyesi olanlar DİĞER grubuna düşer. load_sector_groups() ile okunur.

- Kodda: [`data/bist30_symbols.py::load_sector_groups`](../data/bist30_symbols.py), [`scripts/build_sectors.py`](../scripts/build_sectors.py)
- İlgili: [Graf (sektör + MST)](#graf), [KAP](#kap), [Sembol evreni (E0)](#evren)

<a id="kap"></a>

### KAP

**Sade:** Şirketlerin resmi duyurularını yaptığı Kamuyu Aydınlatma Platformu. Sektör bilgisi ve endeks olayları buradan okunur.

**Teknik:** Sektörler KAP gömülü verisinden; dönem içi endeks olayları VYK API'sinden (data/kap_api.py, .env anahtarı, dakikada 6 çağrı). Test geçidinde veri yalnız 2023'ü kapsar.

- Kodda: [`data/kap_api.py`](../data/kap_api.py)
- Belgede: [CLAUDE.md](../data/CLAUDE.md)
- İlgili: [Sektör grupları (E2)](#sektor-gruplari), [Tarihsel üyelik (E1)](#tarihsel-uyelik)

<a id="temel-veri"></a>

### Temel veri (fundamentals)

**Sade:** Şirketlerin bilanço oranları gibi finansal tablolardan gelen bilgiler. Bu sistemde tek bir anlık görüntü olarak tutulur.

**Teknik:** FundamentalDataFetcher (yfinance); tek ajan faz 2 gözlemine girer. Nokta-zamanlı değildir (tek anlık oran), deneme kaydında fundamentals_pit açık maddesi olarak etiketlenir.

- Kodda: [`data/fundamental_fetcher.py::FundamentalDataFetcher`](../data/fundamental_fetcher.py)
- İlgili: [Nokta-zamanlılık (B0)](#nokta-zamanlilik), [Protokol açık (protocol_open)](#protokol-acik), [Tek ajan ortamı (TradingEnv)](#tek-ajan-ortami)

<a id="kat-yazilim"></a>

## Yazılım ve ürün

<a id="pano"></a>

### Pano (dashboard)

**Sade:** Sistemin tarayıcıda açılan arayüzü. Veri indirme, eğitim, tahmin, günlük karar ve analiz sayfaları buradadır.

**Teknik:** Dash uygulaması, FastAPI sunucusunun /dash/ yolunda (python run_server.py → http://localhost:8000). Sayfalar dashboard/pages; menü modeli tek yerde (kenar çubuğu), üst çubuk kırıntıyı ve aramayı ondan üretir.

- Kodda: [`dashboard/app.py`](../dashboard/app.py), [`dashboard/components/sidebar.py`](../dashboard/components/sidebar.py), [`dashboard/components/topbar.py`](../dashboard/components/topbar.py)
- Belgede: [CLAUDE.md](../dashboard/CLAUDE.md)
- İlgili: [API](#api), [Rehber sayfası](#rehber-sayfasi), [Uzun iş (arka plan görevi)](#uzun-is)

<a id="rehber-sayfasi"></a>

### Rehber sayfası

**Sade:** Panoda sistemin ne yaptığını, nerede olduğumuzu, deneylerin ne bulduğunu ve bu sözlüğü sade dille anlatan sayfa.

**Teknik:** /dash/guide; içerik dashboard/content (guide.yaml, glossary.yaml elle yazılır; snapshot.json scripts/build_guide.py ile sistem haritası ve ölçüm JSON'larından üretilir). Üst çubuk araması sözlük terimlerine #terim-<id> çapasıyla gider.

- Kodda: [`dashboard/pages/guide.py`](../dashboard/pages/guide.py), [`scripts/build_guide.py`](../scripts/build_guide.py)
- Belgede: [system-guide-plan.md](development/system-guide-plan.md)
- İlgili: [Pano (dashboard)](#pano), [Sistem haritası](#sistem-haritasi)

<a id="api"></a>

### API

**Sade:** Panonun ve betiklerin sunucuya istek gönderdiği programlama arayüzü.

**Teknik:** FastAPI, /api/* altında (trading, prediction, hyperopt, account, admin, config). Yazma uçları RequireWriter ya da RequireAdmin bağımlılığı taşır; /api/* yazma isteklerinde CSRF doğrulanır. Etkileşimli belge /docs (oturumlu).

- Kodda: [`app/main.py`](../app/main.py), [`app/api/routes/trading.py`](../app/api/routes/trading.py), [`app/auth/deps.py::RequireWriter`](../app/auth/deps.py)
- İlgili: [Roller (yönetici / kullanıcı / izleyici)](#roller), [Oturum ve çerezler](#oturum), [Pano (dashboard)](#pano)

<a id="roller"></a>

### Roller (yönetici / kullanıcı / izleyici)

**Sade:** Kimin neyi yapabileceğini belirleyen üç yetki düzeyi. İzleyici yalnız bakar, kullanıcı kendi alanında çalışır, yönetici ayrıca hesapları yönetir.

**Teknik:** Role.VIEWER / USER / ADMIN; require_role, admin'i her zaman geçirir. Yazma uçları RequireWriter (user + admin), yönetim uçları RequireAdmin. Kayıt yoktur: hesapları yönetici açar. Yeni yazma ucunda bağımlılık atlanmaz.

- Kodda: [`app/auth/models.py::Role`](../app/auth/models.py), [`app/auth/deps.py::require_role`](../app/auth/deps.py), [`app/auth/deps.py::RequireAdmin`](../app/auth/deps.py)
- Belgede: [CLAUDE.md](../app/CLAUDE.md)
- İlgili: [Çalışma alanı](#calisma-alani), [Oturum ve çerezler](#oturum), [API](#api), [Denetim kaydı](#denetim-kaydi)

<a id="oturum"></a>

### Oturum ve çerezler

**Sade:** Giriş yapan kişinin tarayıcıda tanınmasını sağlayan kısa ve uzun ömürlü anahtarlar.

**Teknik:** Çerez tabanlı oturum (Dash WSGI başlık taşıyamaz): erişim ve yenileme JWT'leri, CSRF için double-submit çerezi (X-CSRF-Token başlığıyla eşleşmeli). Kapı middleware /dash/*, /api/* ve /docs'u korur; AUTH_ENABLED=False ile dışarı açılmaz.

- Kodda: [`app/auth/security.py::create_access_token`](../app/auth/security.py), [`app/auth/security.py::csrf_matches`](../app/auth/security.py), [`app/auth/middleware.py`](../app/auth/middleware.py), [`app/auth/cookies.py::set_auth_cookies`](../app/auth/cookies.py)
- Belgede: [phase-7-auth.md](development/phase-7-auth.md)
- İlgili: [Roller (yönetici / kullanıcı / izleyici)](#roller), [API](#api)

<a id="calisma-alani"></a>

### Çalışma alanı

**Sade:** Her kullanıcının modellerini, sonuçlarını ve kararlarını ayrı tutan kendi klasörü. Piyasa verisi herkes için ortaktır.

**Teknik:** workspaces/<user_id>/; yollar app/auth/workspace.py çözümleyicisinden (models_dir, results_dir, use_workspace). Arka plan görevi kullanıcı kimliğini use_workspace ile taşır, aksi halde dosyalar yanlış alana yazılır. Yollar koda sabitlenmez.

- Kodda: [`app/auth/workspace.py::use_workspace`](../app/auth/workspace.py), [`app/auth/workspace.py::models_dir`](../app/auth/workspace.py)
- Belgede: [CLAUDE.md](../app/CLAUDE.md)
- İlgili: [Roller (yönetici / kullanıcı / izleyici)](#roller), [Uzun iş (arka plan görevi)](#uzun-is), [Model meta dosyası](#model-meta)

**Uyarı:** OPTUNA_STORAGE çalışma alanına göre çözülmüyor: optimizasyon çalışmaları kullanıcılar arasında ortak (bilinen gedik).

<a id="denetim-kaydi"></a>

### Denetim kaydı

**Sade:** Hesap açma, model silme gibi yönetimsel işlemlerin kim tarafından ne zaman yapıldığını tutan kayıt.

**Teknik:** service.audit; ortak model silme model.delete_shared kaydı düşer. Kullanıcı kendi etkinliğini yalnız user_id ile görür (yöneticinin işlemi yöneticinin kimliğini taşır, sızdırılmaz).

- Kodda: [`app/auth/service.py::audit`](../app/auth/service.py), [`app/api/routes/admin.py`](../app/api/routes/admin.py)
- İlgili: [Roller (yönetici / kullanıcı / izleyici)](#roller)

<a id="uzun-is"></a>

### Uzun iş (arka plan görevi)

**Sade:** Eğitim gibi dakikalar süren ve sayfayı kilitlememek için arka planda çalışan iş. İlerleme panoda ve bildirim zilinde görünür.

**Teknik:** background_jobs.spawn / spawn_sync (iş parçacığı, adla izlenir); kullanıcı kimliği use_workspace ile taşınır. Bildirimler kalıcı tablo değil, bellekteki çalışma durumlarının özetidir (/api/account/notifications).

- Kodda: [`app/services/background_jobs.py::spawn`](../app/services/background_jobs.py), [`app/services/background_jobs.py::is_running`](../app/services/background_jobs.py)
- Belgede: [CLAUDE.md](../dashboard/CLAUDE.md)
- İlgili: [Çalışma alanı](#calisma-alani), [Eğitim süresi tahmini (ETA)](#eta), [Pano (dashboard)](#pano)

<a id="eta"></a>

### Eğitim süresi tahmini (ETA)

**Sade:** Bir eğitimin ne kadar süreceğine dair başlamadan önce ve çalışırken verilen tahmin.

**Teknik:** toplam = hazırlık + total_timesteps × adım maliyeti(algoritma, sembol) + değerlendirme; geçmiş koşuların medyanı (kullanıcı başına training_eta_history.json), geçmiş yoksa yerleşik katsayı. Sembol sayısı yüklenen panelden alınır.

- Kodda: [`app/services/training_eta.py::estimate`](../app/services/training_eta.py), [`app/services/training_eta.py::live_eta`](../app/services/training_eta.py), [`app/services/training_eta.py::record_run`](../app/services/training_eta.py)
- İlgili: [Uzun iş (arka plan görevi)](#uzun-is)

<a id="model-meta"></a>

### Model meta dosyası

**Sade:** Kaydedilen her modelin yanında duran ve modelin hangi hisselerle, hangi ölçekle eğitildiğini söyleyen dosya.

**Teknik:** <model>.meta.json (write_model_meta): sembol evreni ve sırası, price_stats, indicator_spec, tohum. Modelin evreni adından çözülmez; tek gerçek kaynak modelin action_space boyutu ve bu dosyadır. Eski modeller için backfill_model_meta.py.

- Kodda: [`app/services/daily_trading.py::write_model_meta`](../app/services/daily_trading.py), [`app/services/daily_trading.py::resolve_trade_universe`](../app/services/daily_trading.py), [`scripts/backfill_model_meta.py`](../scripts/backfill_model_meta.py)
- Belgede: [CLAUDE.md](../app/CLAUDE.md)
- İlgili: [Günlük karar](#gunluk-karar), [Fiyat ölçeği (price_norm)](#fiyat-olcegi), [ADX (Wilder)](#adx)

<a id="gunluk-karar"></a>

### Günlük karar

**Sade:** Eğitilmiş bir modelin akşam verisine bakıp ertesi gün için alım-satım önerisi üretmesi. Öneri kağıt portföye uygulanabilir.

**Teknik:** POST /trading/daily-decision; durum build_live_state ile kurulur, tazelik ölçütü takvim günü değil kapanmış olması beklenen son seanstır (last_expected_session). Canlı karar henüz model meta'sındaki ölçeği ve indicator_spec'i okumuyor (K13 açık iş).

- Kodda: [`app/services/daily_trading.py::build_live_state`](../app/services/daily_trading.py), [`app/services/daily_trading.py::last_expected_session`](../app/services/daily_trading.py), [`app/api/routes/trading.py`](../app/api/routes/trading.py)
- İlgili: [Kağıt portföy](#kagit-portfoy), [Model meta dosyası](#model-meta)

<a id="kagit-portfoy"></a>

### Kağıt portföy

**Sade:** Gerçek para kullanmadan, önerilen işlemlerin sonucunu izlemek için tutulan sanal portföy.

**Teknik:** Tek kaynak workspaces/<uid>/data/live_trading/portfolio.json. Kâr/zarar value_portfolio ile başka günün fiyatıyla değerlenir; summary.daily_return_pct kâr/zarar değildir (aynı gün fiyatıyla yalnız komisyonu gösterir). Aynı tarih iki kez uygulanmaz (applied_dates).

- Kodda: [`app/services/portfolio.py::value_portfolio`](../app/services/portfolio.py), [`app/services/portfolio.py::portfolio_path`](../app/services/portfolio.py)
- Belgede: [CLAUDE.md](../app/CLAUDE.md)
- İlgili: [Günlük karar](#gunluk-karar), [Çalışma alanı](#calisma-alani), [İşlem maliyeti (bps)](#islem-maliyeti)

<a id="golden-test"></a>

### Tahmin golden testi

**Sade:** Tahmin hattının çıktısını kayıtlı bir referansla karşılaştıran test. Kırmızı olması davranışın gerçekten değiştiği anlamına gelir.

**Teknik:** tests/test_prediction_regression.py, CPU'ya sabit (GPU sonucu farklı). K1 gereği 2026-11-15'te bir kez CPU'da yenilendi (v2.0.0). Yalnız bilinçli bir davranış değişikliğinde yenilenir ve nedeni plana yazılır; --update --gpu reddedilir.

- Kodda: [`tests/test_prediction_regression.py`](../tests/test_prediction_regression.py)
- Belgede: [CLAUDE.md](../tests/CLAUDE.md)
- İlgili: [K-kararları](#k-kararlari), [Tahmin topluluğu (stacking ensemble)](#tahmin-ensemble), [Sürüm (SemVer)](#surum)

<a id="surum"></a>

### Sürüm (SemVer)

**Sade:** Sistemin her yayına verdiği numara. Büyük değişiklikte ilk, yeni iş paketinde ortadaki, düzeltmede son hane artar.

**Teknik:** Tek kaynak app/__init__.py (__version__); paket = MINOR. main korumalı (yalnız PR); birleşmeden sonra tag + gh release --generate-notes. Sürüm /health, RL ortam kaydı ve tahmin manifestinde görünür.

- Kodda: [`app/__init__.py::__version__`](../app/__init__.py)
- Belgede: [versioning.md](development/versioning.md)
- İlgili: [Sistem haritası](#sistem-haritasi)

<a id="sistem-haritasi"></a>

### Sistem haritası

**Sade:** Sistemin bileşenlerini ve her birinin şu anki durumunu gösteren tek belge.

**Teknik:** docs/development/system-map.md: düğüm sınıfı + durum tablosu + 'Son güncelleme' satırı. Bir bileşenin durumunu değiştiren PR haritayı aynı PR'da günceller; Rehber'in aşama çizelgesi bu tablodan üretilir.

- Kodda: [`scripts/build_guide.py`](../scripts/build_guide.py)
- Belgede: [system-map.md](development/system-map.md)
- İlgili: [Rehber sayfası](#rehber-sayfasi), [Sürüm (SemVer)](#surum), [Faz](#faz)

<a id="venv-marl"></a>

### venv-marl (ayrı ortam)

**Sade:** Çok ajanlı öğrenme kütüphaneleri için kurulmuş ayrı Python ortamı. Üretim kodu ona bağımlı değildir.

**Teknik:** K9: BenchMARL/TorchRL/PettingZoo yalnız venv-marl'da. marl/ içindeki saf çekirdek (weights, accounting, orders, obs_spec, panel) iki ortamda da çalışır; marl/env.py ve comm_models.py yalnız venv-marl'da. app/ ve dashboard/ bunları import etmez.

- Kodda: [`marl/env.py`](../marl/env.py), [`marl/comm_models.py`](../marl/comm_models.py)
- Belgede: [CLAUDE.md](../marl/CLAUDE.md)
- İlgili: [BenchMARL / TorchRL](#benchmarl), [Gymnasium](#gymnasium), [Çok ajanlı pekiştirmeli öğrenme (MARL)](#marl)

<a id="konteyner"></a>

### Konteyner (Docker)

**Sade:** Sistemin bütün bağımlılıklarıyla paketlenip tek komutla çalıştırılabilen kopyası. Kod imaja kopyalandığı için değişiklik yeniden derleme ister.

**Teknik:** Dockerfile + docker-compose.yml. İmaj docs/ ve *.md almaz (.dockerignore); bu yüzden panonun okuduğu içerik dashboard/content altında tutulur.

- Kodda: [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml)
- İlgili: [Pano (dashboard)](#pano), [Rehber sayfası](#rehber-sayfasi)
