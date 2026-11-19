<!-- URETILDI: scripts/build_guide.py <- dashboard/content/glossary.yaml. ELLE DUZENLEME: YAML'i degistirip `python scripts/build_guide.py` kos. -->

# Sistem Sözlüğü

Projede geçen aşama adları, deney kodları, etiketler ve yöntem terimleri. Her terimde önce **sade** tanım (konuyu bilmeyen biri için), sonra **teknik** tanım (koddaki karşılığıyla) gelir. Sözlük deney sonucu içermez; sonuçlar ilgili tasarım belgelerindedir.

**Terim sayısı:** 73 · **Tek kaynak:** [`dashboard/content/glossary.yaml`](../dashboard/content/glossary.yaml) · **Plan:** [system-guide-plan.md](development/system-guide-plan.md)

## İçindekiler

- **[Sistem ve aşamalar](#kat-sistem)** — [Faz](#faz), [Milestone 0'](#milestone-0p), [M1–M8 (Faz 10 aşamaları)](#m-asamalari), [M3a / M3b](#m3), [M4a / M4b](#m4), [M5](#m5), [K-kararları](#k-kararlari), [K12 yürütme ufku](#k12), [Ön kayıt](#on-kayit), [Hakem paneli](#hakem-paneli), [Geliştirme kapısı](#gelistirme-kapisi), [Deneme kaydı (N)](#deneme-kaydi)
- **[Deney kodları ve etiketler](#kat-deney)** — [Pozitif kontrol (PC060-K12)](#pozitif-kontrol), [Karıştırılmış kontrol](#karistirilmis-kontrol), [Komşu sinyali (σ)](#komsu-sinyali), [Kol](#kol), [A1 / S1 / A1c / S1c](#kol-kodlari-m3), [N / A / G / Z × X / S](#kol-kodlari-m4), [G0 / GL / GLS / GLM / GD](#kol-kodlari-m5), [GEÇTİ](#etiket-gecti), [KISMİ](#etiket-kismi), [GÖZLENMEDİ](#etiket-gozlenmedi), [PİLOT_İŞARETİ](#etiket-pilot), [Tohum](#tohum)
- **[Değerlendirme ve istatistik](#kat-degerlendirme)** — [Sharpe oranı](#sharpe), [Fazla getiri](#fazla-getiri), [Risksiz faiz (A8)](#risksiz-faiz), [D (Sharpe farkı)](#d-farki), [Ortak HAC](#hac), [Tek yönlü p](#tek-yonlu-p), [Holm düzeltmesi](#holm), [Benjamini–Hochberg (BH-FDR)](#bh-fdr), [Güç](#guc), [En küçük tespit edilebilir etki (MDE)](#mde), [Güven aralığı (GA)](#guven-araligi), [Eğitim / doğrulama / test penceresi](#pencereler), [1/N (eşit ağırlık)](#baseline-1n), [Al-tut](#al-tut), [Devir](#devir), [İşlem maliyeti (bps)](#islem-maliyeti), [Deflated Sharpe (DSR)](#dsr)
- **[Çok ajanlı öğrenme](#kat-marl)** — [Çok ajanlı pekiştirmeli öğrenme (MARL)](#marl), [Ajan](#ajan), [MAPPO](#mappo), [IPPO](#ippo), [Merkezi eğitim, merkezi olmayan yürütme (CTDE)](#ctde), [Eleştirmen](#elestirmen), [Parametre paylaşımı (K24)](#parametre-paylasimi), [İletişim kolları (none / attn / gat / self)](#iletisim-kollari), [Dikkat (attention)](#dikkat), [Graf (sektör + MST)](#graf), [α (ödül karışımı)](#alfa), [Logit sınırı (5 + ln N)](#logit-siniri), [S-obs](#s-obs), [S-msg](#s-msg), [β (güven ağırlığı)](#beta), [Güven geçidi](#guven-gecidi), [Kare (frame)](#kare), [Bölüm (episode)](#bolum)
- **[Tek ajan ve genel pekiştirmeli öğrenme](#kat-rl)** — [Politika](#politika), [Gözlem](#gozlem), [Ödül](#odul), [EMA Sharpe düzeyi](#ema-sharpe), [PPO](#ppo)
- **[Tahmin hattı](#kat-tahmin)** — [AUC](#auc), [P tablosu (pit_v1 / pit_v2)](#pit-tablosu), [Bilgi kapısı (İ3)](#bilgi-kapisi), [Kalibrasyon (sıcaklık)](#kalibrasyon)
- **[Veri](#kat-veri)** — [BIST-30](#bist30), [Panel](#panel), [Listeleme kilidi](#listeleme-kilidi), [Satış engeli (taban açılış)](#satis-engeli), [Nokta-zamanlılık (B0)](#nokta-zamanlilik)

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

**Teknik:** M1 çerçeve denemesi (BenchMARL/TorchRL), M2 ortam ve muhasebe çekirdeği, M3 öğrenme ve pozitif kontrol, M4 iletişim kolları, M5 tahmin sinyali ve güvenle ağırlıklı iletişim. M6–M8 Faz 10 planında tanımlı sonraki aşamalardır; uzun koşular M8'den önce yapılır.

- Belgede: [phase-10-marl-plan.md](development/phase-10-marl-plan.md)
- İlgili: [M3a / M3b](#m3), [M4a / M4b](#m4), [M5](#m5), [Geliştirme kapısı](#gelistirme-kapisi)

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

- Kodda: [`marl/m3_runner.py::rule_series`](../marl/m3_runner.py), [`evaluation/baselines.py::equal_weight`](../evaluation/baselines.py)
- İlgili: [Al-tut](#al-tut), [İşlem maliyeti (bps)](#islem-maliyeti)

<a id="al-tut"></a>

### Al-tut

**Sade:** İlk gün parayı eşit bölüp sonra hiç işlem yapmamak.

**Teknik:** İlk gün 1/N, sonra hedef = kaymış ağırlık (işlem yok; yalnız zorunlu dondurma), ortamın muhasebesiyle.

- Kodda: [`marl/m3_runner.py::rule_series`](../marl/m3_runner.py)
- İlgili: [1/N (eşit ağırlık)](#baseline-1n)

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

<a id="kat-marl"></a>

## Çok ajanlı öğrenme

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

<a id="kat-veri"></a>

## Veri

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

**Teknik:** limit_down_gaps(dates): dönem bağımlı fiyat marjı − 0,005. Marj takvimi 2020-12-31'e kadar doğrulandı; sonrası için hata verir (önce kaynakla doğrulanmalı). Tavan alım engeli yoktur.

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
