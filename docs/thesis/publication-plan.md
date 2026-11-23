# Yayın planı (makale, bildiri, derleme)

**Sürüm:** v0.1 · 2026-11-23 · **TASLAK**: kullanıcı kararları (§6) bekleniyor
**Yerini aldığı:** [vision-and-scope.md](vision-and-scope.md) §6.2 "Makale Hedefleri" (v0.3). Vision §6.2 artık bu belgeye işaret eder.
**Dayanak:** [Faz 10 planı](../development/phase-10-marl-plan.md) §9 (araştırma kapanışı), [M8 tasarımı](../development/phase-10-m8-design.md) §7, [Faz 11 planı](../development/phase-11-integration-plan.md) §0 ve §3.6, [Faz 9 planı](../development/phase-9-research-validity-plan.md) "Makale 2 deney tasarımı".
**Yazım kuralları:** [CLAUDE.md](CLAUDE.md). Her yayın taslağı tez metniyle aynı kurallara bağlıdır:
- İddia ya kaynağa ya da ölçüm dosyasına bağlanır.
- "Sızıntı yok" denmez.
- Özgünlük dili temkinlidir; tarama kapsamı ve tarihi yazılır.
- Bitmiş taslak gönderimden önce hakem paneline okutulur ([review-panel.md](../development/review-panel.md)).

---

## 0. Özet

1. **Vision v0.3'teki makale planı M8 kanıtıyla artık uyuşmuyor.**
   - **Makale 1** ("Prediction-Augmented Communicating Agents") iki dayanağını kaybetti:
     - Tahmin güveninin iletişime beslenmesi (G2) M8'de sınanmadı; M5'te fark tespit edilmedi.
     - İletişim karşıtlığında (gat ↔ self) fark gözlenmedi.
   - **Makale 3** (güven ağırlıklı mesaj) aynı nedenle yeni deney gerektiriyor. Faz 11 yeni hipotez açmıyor (Faz 11 §4).
2. **Elde yayımlanabilir, hazır bir ana sonuç var:** ön kayıtlı, 432 koşulu bir MARL ↔ SARL ↔ kural portföyü karşılaştırması (P1).
   - Çerçeve "yöntem önerisi" değil, **ön kayıtlı karşılaştırma ve dürüst bir sonuç**:
     - öğrenen kollar 20 bps'de kural portföylerini geçemiyor;
     - MARL ↔ SARL farkının bir kısmı maliyetten bağımsız.
3. **Portföy:**
   - 3 dergi makalesi: P1 hazır; P2 ileri veri, ~2027 yazı; P3 Makale 2, ön koşullu.
   - 2 derleme: P4a uluslararası; P4b Türkçe, BIST odaklı.
   - 2 bildiri: B1, B2. P1 ile örtüşmeyen konular seçildi.
4. **Türkiye merkezli en uygun dergi Borsa Istanbul Review.** Konferanslarda en uygunları SIU, ASYU ve INISTA (IEEE Xplore). Tarama sonucu §3'te; sayıların çoğu ikincil kaynaktan geliyor, **gönderimden önce resmi kaynaktan doğrulanmalı**.

## 1. Neye dayanılabilir (kanıt kaynakları)

| Bulgu | Durum | Kaynak (sayılar metne buradan alınır) |
|---|---|---|
| H1 MAPPO-none ↔ SARL-eşmimari: FARK_VAR_MARL (yila_bagli, secime_dayanikli), Δ̄ +0,47; 0 bps'de +0,24 [0,08, 0,40] | Ön kayıtlı, okundu | `docs/development/measurements/phase10-m8/m8_reading.json`, `m8_robustness.json`; M8 §7 |
| H2 MAPPO-none ↔ SARL-ağırlık: FARK_VAR_M-none; fark maliyet sürüklemesiyle açıklanıyor | Ön kayıtlı + betimsel | Aynı; M8 §7.3 |
| H3 gat ↔ self: FARK_GOZLENMEDI, tohum kümeli %95 GA [−0,14, 0,12] | Ön kayıtlı | Aynı |
| Kural portföyleri (1/N, al-tut, min-var, HRP) 20 bps'de bütün öğrenen kolların üstünde | Betimsel | M8 §7.3 b |
| M4b GAT pozitif kontrolü: PİLOT_İŞARETİ (G 0,26, z 3,05) | Pilot, dikilmiş sinyal | [M4 tasarımı](../development/phase-10-m4-design.md) §6.4 |
| M3a boru hattı kapısı GEÇTİ (maliyetsiz, gelecekten türetilmiş pozitif kontrol) | Tez bulgusu değil, yöntem kanıtı | [M3 tasarımı](../development/phase-10-m3-design.md) §5.2 |
| İleri veri (126 seans, 72 politika + 4 kural) | **Henüz yok**, ~2027-05/06 | [Faz 11 ileri veri eki](../development/phase-11-m8-forward-annex.md) (dondurulmalı) |
| Sızıntısız ayrıştırma + TATS (Makale 2) | **Planlı**, B5 + B10 ön koşul | Faz 9 planı "Makale 2 deney tasarımı"; Faz 10 planı §3 (B10 notu) |

**Kullanılmayacaklar:**
- Faz 9 öncesi tek ajanlı RL sonuçları (`protocol_open`; [CLAUDE.md](CLAUDE.md) "RL sonuçları henüz tez sayısı değil").
- Bugünkü haliyle ICEEMDAN hattı.
- "BIST'te ilk MARL" ifadesi. G4 YÖK taraması bitmeden kullanılmaz.

## 2. Yayın portföyü

### P1 — Ana makale: ön kayıtlı MARL ↔ SARL ↔ kural portföyleri (BIST-30)

- **Soru:** Eşit bütçe ve ileri yürüyen 6 yıllık bölmede, hisse başına ajanlı MARL tek ajanlı karşılıklarından ve basit kural portföylerinden farklı mı? Fark işlem maliyetinden ne kadar etkileniyor?
- **Katkı (dar ifade):**
  1. Ön kayıtlı (etiket `m8-prereg-v1`) tasarım: tohum kümeli çıkarım, eşmimari SARL kontrolü, boş kol negatif kontrolü.
  2. Maliyet ayrıştırması: 0 bps ↔ 20 bps.
  3. Gelişen bir piyasada kural portföyü baseline'larıyla dürüst bir kıyas.
  4. Ön kayıtlı ileri veri testi süreci (P2'ye bağlanır).
- **Hazırlık:** Sonuç hazır. Yazım + hakem paneli (hakem-finans + hakem-rl → hakem-2).
- **Hedef:**
  - Birincil: **Borsa Istanbul Review**.
  - Yedek: *Emerging Markets Review*, *Expert Systems with Applications*.
  - Seçim §6 K-Y1'e bağlı.
- **Not:** Konferans sürümü yapılmaz (ICAIF dahil). Arşivlenen bir bildiri dergi makalesiyle çakışır.

### P2 — İleri veri okuması (126 seans)

- **Soru:** M8 §6.7 ön kayıt ölçütü. Dondurulmuş 72 politika ve 4 kural portföyü, eğitimde hiç görülmemiş ilk 126 seansta.
- **Ön koşul:** İleri veri ekinin ilk ileri günden önce dondurulması (etiket `m8-forward-annex-v1`). Okuma `scripts/marl/analyze_forward_m8.py` ile **bir kez** yapılır.
- **Biçim:**
  - P1 hakem sürecindeyse revizyona ek olarak girer.
  - P1 yayımlandıysa kısa makale olur (*Finance Research Letters*, ~2.500 kelime sınırı).
- **Takvim:** ~2027-05/06 (Faz 11 §0).

### P3 — Makale 2: sızıntısız ayrıştırma ve TATS düzeltmesinin katkısı

- **Soru:** Vision v0.2 §6.2: "Sızıntısız (ileri yürüyen) ICEEMDAN altında ayrıştırmanın ve TATS düzeltmesinin tahmin ve RL performansına katkısı ne kalıyor?"
- **Ön koşul:** B5 (nedensel ICEEMDAN) ve B10, sonra K0–K4 kolları (Faz 10 planı §3).
- **Hedef:**
  - Birincil: *Neurocomputing* / *Knowledge-Based Systems* (vision v0.3).
  - Türkiye merkezli seçenek: **Turkish Journal of Electrical Engineering and Computer Sciences** (TÜBİTAK, SCIE).
- **Takvim:** Ön koşullardan sonra. Tarih yok (Faz 10 K5).

### P4a — Uluslararası derleme: portföy yönetiminde (MA)RL ve değerlendirme protokolü

- **Fark yaratan açı:** Yöntem kataloğu değil, **değerlendirme denetimi**. Seçilmiş kaynakların şu açılardan kodlanması:
  - tohum sayısı,
  - işlem maliyeti,
  - 1/N / min-var / HRP baseline'ları,
  - çoklu test düzeltmesi (DSR/PBO),
  - ileri yürüyen bölme,
  - sızıntı denetimi.

  Kontrol listesi buradan çıkar.
- **Kaynak tabanı:** [literature-table.md](literature-table.md), 148 seçilmiş kayıt (2026-10-07). 2026-10 sonrası için güncelleme taraması gerekir (`/kaynak-ekle` hattı).
- **Hedef:**
  - Birincil: *Artificial Intelligence Review*. Tam açık erişim olduğu için APC gerekir (§6 K-Y6).
  - Alternatif: *ESWA* (derleme kabul ediyor).
- **Takvim:** 2027 Q1–Q2.

### P4b — Türkçe derleme: BIST'te makine öğrenmesi / DRL ile alım-satım

- **Soru:** BIST üzerinde ML/DRL alım-satım çalışmaları hangi protokolle değerlendirilmiş? Kodlama şeması P4a ile aynı.
- **Ek getiri:** G4 için gereken YÖK Tez taraması bu işin parçası olur. "İlk" ifadesi ancak bundan sonra (ya da hiç) kurulur.
- **Hedef:**
  - **Gazi Üniv. MMF Dergisi** (SCIE).
  - Alternatif: **Pamukkale Üniv. Mühendislik Bilimleri Dergisi** (ESCI), **Istanbul Business Research** (ESCI; finans okuru).
- **Örtüşme:** P4a ile kaynak kümesi ve soru farklı (BIST yazını ↔ uluslararası MARL), örtüşme düşük. Ortak kodlama şeması iki metinde de beyan edilir.
- **Vision'daki "Makale 4"ün yerini alır.** Eski Makale 4 P1'in Türkçe sürümüydü; bu çift yayın riski taşıyordu.

### B1 — Bildiri: iletişim kanalı bilgi taşıyor mu? Dikilmiş sinyalle pozitif kontrol

- **İçerik:** M3a boru hattı kapısı ve M4b GAT pozitif kontrol tasarımı (G metriği), ayrıca negatif kontrol mantığı. MARL iletişim çalışmalarında "iletişim işe yaradı" iddiasından önce yapılması gereken test olarak sunulur.
- **P1 ile ilişki:** P1 gerçek veride Sharpe'ı ölçer; B1 yöntem kontrolünü anlatır. Sayılar P1'de yer almaz, ya da yalnız bir cümleyle atıf yapılır.
- **Hedef:** SIU 2027 (4 sayfa) ya da ASYU / INISTA 2027 (6 sayfa).

### B2 — Bildiri: tekrar üretilebilir finansal RL hattı ve ileri veri simülasyonu

- **İçerik:**
  - ön kayıt + etiket,
  - SHA ile dondurulmuş politika arşivi,
  - BenchMARL politikalarının bağımlılıksız ileri geçişe bit-eş aktarımı (72/72),
  - defter ↔ değerlendirme serisi paritesi,
  - idempotent günlük iş ve boşluk doldurmanın bit-eşliği.

  Kaynak: [Faz 11 planı](../development/phase-11-integration-plan.md) §3.1–3.5.
- **P1 ile ilişki:** Mühendislik katkısı; araştırma sonucu içermez.
- **Hedef:** UBMK ya da ASYU 2027.

### Askıya alınanlar

- **Vision Makale 1 başlığı ve çerçevesi:** kanıtla desteklenmiyor (§0). Konu P1'e dönüştü.
- **Vision Makale 3** (güven ağırlıklı mesaj): G2 için yeni deney gerekir. Faz 11 kapsamı dışı; yeni bir faz kararı ister.

## 3. Mekân taraması (2026-11-23)

**Yöntem:** Web araması (ikincil kaynaklar: yayıncı sayfaları, DergiPark, wos-journal.info, toplayıcı siteler).
- Etki faktörleri **kaynaklar arasında tutarsız**. Tabloda hangi kaynaktan geldiği yazılı; hiçbiri Clarivate JCR'den doğrudan doğrulanmadı.
- Gönderimden önce [Clarivate Master Journal List](https://mjl.clarivate.com/) ve JCR'den doğrulanmalı.

### 3.1 Dergiler

| Dergi | Yayıncı | İndeks (doğrulama) | Etki (kaynak) | Uygun iş | Not |
|---|---|---|---|---|---|
| **Borsa Istanbul Review** | Borsa İstanbul / Elsevier | WoS, Scopus, DOAJ (toplayıcı); SSCI resmi listeden doğrulanmadı | JCR 2023 IF 5,2 (Ali, Aysan & Gokirmak 2025); SJR Q1 Finans | **P1**, P2 | Gelişen piyasa, BIST odağı; çift kör; açık erişim |
| Central Bank Review | TCMB / Elsevier | ESCI (Ekonomi), Scopus, DOAJ | JIF 1,7 (wos-journal.info) | P1 yedeği (zayıf uyum) | Makro / para politikası ağırlıklı |
| **Turkish J. of Electrical Eng. & Computer Sciences** | TÜBİTAK | SCIE (2010'dan beri; dergi sayfası) | IF 2,1, Q3 (dergi sayfası / Wikipedia; diğer kaynaklar 1,5–2,5) | **P3**, B1'in genişletilmiş hali | TÜBİTAK dergisi, ücretsiz |
| **Gazi Üniv. MMF Dergisi** | Gazi Üniv. | SCIE (Engineering, Multidisciplinary) | JIF ~1,0 (wos-journal.info); dergi 2023 için 1,274 diyor | **P4b** | Türkçe/İngilizce |
| Pamukkale Üniv. Müh. Bil. Dergisi (PAJES) | Pamukkale Üniv. | ESCI (2018'den beri), TR Dizin | 5 yıllık IF 0,7 | P4b yedeği | |
| Politeknik Dergisi | Gazi Üniv. | ESCI (yalnız toplayıcı; **doğrulanmadı**) | — | P4b yedeği | Türkçe/İngilizce |
| Istanbul Business Research | İstanbul Üniv. | ESCI (Business, Finance), TR Dizin | 2024 JIF 0,6 | P4b (finans okuru) | |
| Business and Economics Research Journal, Ege Academic Review | — | **Doğrulanmadı** | — | — | Taramada indeks bilgisi bulunamadı |
| *Emerging Markets Review* | Elsevier | SSCI **doğrulanmadı** | ~5 (toplayıcılar) | P1 yedeği | Gelişen piyasa finansı |
| *Emerging Markets Finance and Trade* | Taylor & Francis | — | ~3,3–3,9 (toplayıcılar) | P1 yedeği | Türk yazarların sık kullandığı |
| *Finance Research Letters* | Elsevier | — | ~7 (toplayıcılar) | **P2** | ~2.500 kelime sınırı |
| *Artificial Intelligence Review* | Springer | — | 2024 JIF 13,9 (Springer) | **P4a** | Derleme ve öğretici makale dergisi; 2024'ten beri tam açık erişim → APC |

### 3.2 Konferanslar

2027 duyuruları taramada bulunamadı. Tarihler **2026 edisyonundan tahmin**.

| Konferans | Yayın | 2026 edisyonu (kaynak) | 2027 tahmini son tarih | Uygun iş |
|---|---|---|---|---|
| **SIU** (Sinyal İşleme ve İletişim Uyg. Kurultayı) | IEEE Xplore; 4 sayfa, çift kör | 7–10 Tem 2026, Piri Reis Üniv.; gönderim 13 Mart (CFP), siteye göre 29 Mart | ~Mart 2027 | B1 |
| **ASYU** (Akıllı Sistemlerde Yenilikler ve Uyg.) | IEEE Xplore; 6 sayfa, çift kör, ≥3 hakem | 9–11 Eyl 2026, Özyeğin Üniv.; gönderim 1 Haz | ~Haziran 2027 | B1, B2 |
| **INISTA** | IEEE Xplore + WoS CC; 6 sayfa | 17–19 Eyl 2026, Guimarães (YTÜ ortak); gönderim 30 May | ~Mayıs 2027 | B1 |
| UBMK | IEEE (bilgimize göre) | Taramada **bulunamadı** | doğrulanmadı | B2 |
| YAEM (YA/EM Ulusal Kongresi) | Özet / bildiri | 29 Haz–1 Tem 2026, İTÜ | ~bahar 2027 | P1'in Türkçe sunumu (yalnız özet; §4) |
| EYİ (Ekonometri, YA ve İstatistik Semp.) | Özet + tam metin kitabı | 17–19 Eyl 2026, Manisa CBÜ; özet 13 Tem | ~Temmuz 2027 | P1'in Türkçe sunumu (yalnız özet) |
| ICFT (Financial Technology Int. Conf.) | — | 19–21 Ağu 2026, İstanbul; özet 19 Tem | ~Temmuz 2027 | P1 / P2 sunumu |
| *ACM ICAIF* (uluslararası, Türkiye dışı) | ACM, arşivli | 14–17 Kas 2026, Milano; gönderim 2→9 Ağu | ~Ağustos 2027 | P1 ile çakışır, kullanılmaz; çalıştayları B1 için olabilir |
| "Finans Sempozyumu" (ulusal) | — | Taramada **bulunamadı** | — | — |

**Kaynaklar:**
[BIR – SciJournal](https://www.scijournal.org/borsa-istanbul-review),
[BIR bibliyometri (İZÜ)](https://openaccess.izu.edu.tr/bitstreams/b3d1b4d8-56f2-4447-9858-24b675fd0eb8/download),
[Central Bank Review – WoS](https://wos-journal.info/journalid/10915),
[TJEECS – TÜBİTAK](https://journals.tubitak.gov.tr/elektrik),
[TJEECS – Wikipedia](https://en.wikipedia.org/wiki/Turkish_Journal_of_Electrical_Engineering_and_Computer_Sciences),
[Gazi MMF – WoS](https://wos-journal.info/journalid/860),
[PAJES ESCI haberi](https://haber.pau.edu.tr/Haber/pau-engineering-sciences-journal-started-to-be-scanned-at-esci),
[Politeknik – WoS](https://wos-journal.info/journalid/6007),
[IBR – DergiPark](https://dergipark.org.tr/en/pub/ibr/about-journal),
[EMR – Elsevier](https://shop.elsevier.com/journals/emerging-markets-review/1566-0141),
[EMF&T – SciJournal](https://scijournal.org/emerg-mark-financ-trade),
[FRL – CASRAI](https://casrai.org/guides/finance-research-letters),
[AIR – Springer](https://www.springer.com/10462),
[SIU 2026](https://siu2026.pirireis.edu.tr/en/?p=106),
[SIU 2026 CFP](https://siu2026.pirireis.edu.tr/en/wp-content/uploads/2026/03/SIU_2026_CFP_EN_v1.4-1.pdf),
[ASYU 2026 CFP](https://asyu2026.ozyegin.edu.tr/documents/CFP-IEEE-ASYU-2026-EN.pdf),
[INISTA 2026 CFP](https://inista2026.di.uminho.pt/cfp.html),
[YAEM 2026](https://anadolu.edu.tr/kurum-disi-duyurular/45-yoneylem-arastirmasi-ve-endustri-muhendisligi-ulusal-kongresi-yaem-2026-1773058939),
[EYİ 2026](https://anadolu.edu.tr/kurum-disi-duyurular/23-uluslararasi-ekonometri-yoneylem-arastirmasi-ve-istatistik-sempozyumu-1782478360),
[ICFT 2026](https://easychair.org/cfp/icft2026),
[ICAIF 2026 CFP](https://groups.google.com/g/aifin-worldwide/c/zh7y9B-Gw-4).

## 4. Kurallar

1. **Çift yayın ve dilimleme.**
   - Aynı sonuç (P1) iki dergide yayımlanmaz, Türkçe-İngilizce dahil.
   - Bildiri sonra dergiye genişletilecekse yeni içerik dergi politikasının istediği oranda olur ve önceki yayın beyan edilir.
   - **Yalnız özet basan** ulusal kongrelerde (YAEM, EYİ ve benzeri) sunum genelde önceki yayın sayılmaz. Hedef derginin politikası yine de kontrol edilir; tam metin kitabına metin verilmez.
2. **Ön kayıtın dışarıdan doğrulanabilirliği.**
   - Hakem, `m8-prereg-v1` ve `m8-forward-annex-v1` etiketlerinin tarihine güvenebilmeli. Depo özel ise bu tarih dışarıdan görülemez.
   - Seçenek: etiketlenmiş belgelerin OSF ya da Zenodo'da zaman damgalı kaydı (§6 K-Y3).
3. **Ön baskı.** Elsevier ve Springer dergileri genelde arXiv/SSRN ön baskısına izin veriyor; derginin kendi politikası gönderimden önce kontrol edilir. Çift kör dergide ön baskı anonimliği bozabilir.
4. **Veri ve kod paylaşımı.**
   - yfinance ve İş Yatırım verisi yeniden dağıtılamayabilir. Paylaşım "kod + veri çekme betiği + SHA manifesti" biçiminde olur.
   - `workspaces/`, `models/`, `results/` git dışı kalır (kök CLAUDE.md).
5. **Sayılar.** Metne elle kopyalanmaz; hangi ölçüm dosyasından geldiği yazılır (§1 tablosu).

## 5. Takvim (öneri)

| Dönem | İş |
|---|---|
| 2026-11-23 | **İleri veri ekinin dondurulması** (P2'nin ön koşulu; Faz 11 §3.6) |
| 2026-12 | P1 taslağı + hakem paneli; K-Y1..K-Y3 kararları |
| 2027-01/02 | P1 gönderimi; P4a/P4b kodlama şeması ve güncelleme taraması; YÖK taraması |
| 2027-03 | SIU 2027 (tahmini son tarih) → B1 |
| 2027-05/06 | 126. seans → P2 okuması (bir kez) → P1 revizyonu ya da FRL kısa makalesi; ASYU/INISTA 2027 → B2 (ya da B1) |
| 2027 Q2 | P4a / P4b gönderimleri |
| Tarihsiz | P3: B5 + B10 + K0–K4 bittikten sonra |

## 6. Karar bekleyen konular (kullanıcı)

| # | Konu | Öneri |
|---|---|---|
| K-Y1 | P1 birincil dergi | Borsa Istanbul Review: BIST odağı, gelişen piyasa ve finans okuru tam uyuyor. Yedek: Emerging Markets Review |
| K-Y2 | ÜAK temel alanı (mühendislik ↔ sosyal / finans) | Mekân ağırlığını belirler. İkincil kaynağa göre mühendislikte Q1–Q3 dergi belirleyici, bu durumda ESCI dergilerin değeri düşer. Resmi ÜAK tablosundan doğrulanmalı |
| K-Y3 | Ön kayıt için kamusal zaman damgası | Etiketlenmiş ön kayıt belgeleri OSF ya da Zenodo'ya. Maliyeti düşük, hakem itirazını önler |
| K-Y4 | Tez bölüm planı (vision §6.1) | Bölüm 7 ("Prediction-Augmented Communication") ve 8 (rejim ablasyonu) M8 kanıtıyla yeniden çerçevelenmeli; ayrı bir karar |
| K-Y5 | Vision Makale 3'ün askıya alınması | Askıya alınsın; G2 için yeni bir faz kararı gerekir |
| K-Y6 | APC bütçesi (AIR tam açık erişim) | Bütçe yoksa P4a için ESWA |

## Sürüm geçmişi

- **v0.1 (2026-11-23):** İlk taslak. Vision §6.2'nin yerini alır. M8 kanıtına göre portföy P1–P4b, B1–B2; mekân taraması (web, ikincil kaynaklar); karar bekleyen konular K-Y1..K-Y6.
