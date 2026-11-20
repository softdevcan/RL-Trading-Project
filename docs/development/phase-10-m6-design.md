# Faz 10 M6/M7 — Rejim ve yorumlanabilirlik: tasarım

**Sürüm:** v0.8 · 2026-11-20 · dal `feat/phase10-m6` · **Durum:** kararlar K-M6-1..4 KAPANDI (2026-11-20, kullanıcı; dördü de öneri yönünde, §5). Sıradaki: §2.1 2021+ etiket sayımı, M6a HMM, M6c okuma aracı.
**Üst belgeler:**
- [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.1.3 sinyal yolları (S-reg), §4 sıra (M5 → M6/M7 → M8), §6 K10, §3.2.8 ortam hızı
- [phase-9-research-validity-plan.md](phase-9-research-validity-plan.md) M6 (rejim, AQ3, G5), M7 (yorumlanabilirlik), M8 (H4, "Rejim etiketi" maddesi)
- [vision-and-scope.md](../thesis/vision-and-scope.md) AQ3, G5
- [phase-10-m5-design.md](phase-10-m5-design.md) §4.11, §5.10 (M5 sonuçları; M6 bunların üstüne kurulur)

---

## 1. Amaç

Planın üç M6 maddesi ve bir M7 maddesi var:

1. **Rejim etiketi (birincil):** kural tabanlı, nedensel etiket (B6, `evaluation/regimes.py`): XU100 / USDTRY_{t−1}, 200 günlük eğilim × 60 günlük oynaklık, genişleyen tercil. Duyarlılık kolları nominal TL ve 5 yıllık kayan tercil (Faz 9 M8 maddesi, kullanıcı kararı 2026-11-15). **Kod hazır.**
2. **İkinci tanım:** HMM'in **yalnızca filtrelenmiş** olasılıkları (Hamilton 1989; Nystrup vd. 2018); durum sayısı BIC ile. Rejim başına gün ve bağımsız dönem sayısı raporlanır. **Kod yok.**
3. **S-reg (K10):** tüm ajanlara ve eleştirmene yayınlanan nedensel rejim özellikleri. Plan "S-reg ana tasarımda sabit tutulur; katkısı ayrı ve tek bir ablasyonla ölçülür" diyor (Faz 10 §3.1.3). **Kod yok; M3–M5 pilotlarının hiçbirinde S-reg yok.**
4. **M7:** Wiegreffe & Pinter testleri (tek biçimli dikkat baseline'ı, tohumlar arası dikkat varyansı) ve karşı-olgusal mesaj maskeleme (bir ajanın mesajı sıfırlanınca portföy Sharpe'ındaki değişim). Jain & Wallace sınırlılık olarak anılır.

Tez sorusu AQ3/H4: "iletişimin marjinal katkısı önceden tanımlı zamansal rejimlere göre farklı mı?" Kesin sınama M8 ön kaydında yapılır. M6'nın işi M8'e **uygulanabilir** bir rejim tanımı ve çıkarım düzeni vermek.

## 2. Ön ölçüm: geliştirme pencerelerinde rejim başına gün ve dönem

Kaynak: `measurements/phase10-m6/regime_windows.json` (`scripts/marl/regime_windows_m6.py`, temiz ağaç `b16f837`; panel, XU100 ve USDTRY SHA-256'ları kayıtlı). Tarihler MARL panelinin seansları. Pencereler `eval_m3` ile aynı: eğitim 2013-08-27..2018-12-31 (1.343 seans), doğrulama 2019-01-02..2020-12-31 (500 seans). Hücre = gün sayısı / en az 20 seans süren dönem sayısı (`episodes_min20`). Performans sayısı yoktur.

**Altı hücre (eğilim × oynaklık), birincil etiket:**

| Hücre | Eğitim gün / ≥20 g dönem | Doğrulama gün / ≥20 g dönem |
|---|---|---|
| bear_low | 290 / 4 | 69 / 1 |
| bear_mid | 392 / 6 | 126 / 1 |
| bear_high | 158 / 1 | 77 / 1 |
| bull_low | 463 / 6 | 112 / 2 |
| bull_mid | 31 / 0 | 94 / 1 |
| bull_high | 9 / 0 | 22 / 0 |

**Yalnız eğilim (iki hücre), birincil etiket:** eğitim bear 840 gün / 4 dönem, bull 503 / 6; doğrulama bear 272 / 3, bull 228 / 4. Eğilim etiketi 5 yıllık kayan tercil kolunda birincille aynıdır (kayan tercil yalnız oynaklık düzeyini değiştirir). Nominal TL kolunda eğitim bear 599 / 5, bull 744 / 5.

**Kollar arası uyuşma (altı hücre, aynı gün aynı hücre oranı):** birincil ↔ nominal eğitim 0,64, doğrulama 0,51; birincil ↔ 5 yıllık kayan eğitim 0,67, doğrulama 0,52.

**Okuma (tasarım girdisi, sonuç değil):**
- Altı hücreli etiket geliştirme pencerelerinde **sınanamaz.** İki hücrede (bull_mid, bull_high) eğitimde 20 günü aşan tek bir dönem yok. Doğrulamada her hücre en fazla 2 dönem taşıyor. Hücre başına bir iletişim katkısı kestirimi bu dönem sayısıyla blok bootstrap'a dayanamaz.
- Eğilim (iki hücre) en azından her pencerede 3–6 bağımsız sayılabilecek dönem veriyor. Bu da az; geliştirme pilotunda H4 ancak **betimsel** okunabilir.
- Kollar arası uyuşma altı hücrede düşük (0,51–0,67): "rejim" sonucu etiket seçimine duyarlı olacaktır. Bu, duyarlılık kollarının M8'de neden zorunlu olduğunu gösteriyor.
- H4'ün kesin sınaması M8'in test pencerelerine (wf-v1, 2021..2026) kalır. Bu pencerelerdeki rejim sayımı **başarım okunmadan** yapılabilir, ama 2021+ m0p-v2 doğrulayıcı penceresi olduğu için bu belgede yapılmadı (karar K-M6-4).

### 2.1 wf-v1 test yılları: yalnız etiket sayımı (K-M6-4)

Kaynak: `measurements/phase10-m6/regime_test_years.json` (`regime_windows_m6.py --test-years`, temiz ağaç). Seanslar XU100 işlem günleri (MARL paneli 2020'de biter); 2021-01-04..2026-10-30, 1.454 seans. **Yalnız endeks etiketi sayıldı; hiçbir politika, getiri ya da başarım okunmadı.** Bu yüzden pencere "başarımla açılmış" sayılmaz; `EXPOSED_EVAL_WINDOWS`'a eklenmedi (beyan).

| Etiket | Birleşik test: bear gün / ≥20 g dönem | bull gün / ≥20 g dönem | Kat (yıl) başına ≥20 g dönem |
|---|---|---|---|
| Birincil (USD bazlı) eğilim | 685 / 8 | 769 / 9 | 1–3 |
| Nominal TL eğilim | 219 / 4 | 1.235 / 4 | 0–2 |

- Altı hücrede birleşik testte hücre başına 2–6 dönem var; kat başına çoğu hücre 0–1.
- Kollar arası uyuşma (altı hücre, birleşik): birincil ↔ 5 yıllık kayan 0,86, birincil ↔ nominal 0,52. Nominal TL etiketi enflasyonla yükselişe kayıyor (2022'de hiç düşüş günü yok), birincilin USD bazlı seçilme gerekçesiyle tutarlı (evaluation/CLAUDE.md, rejim maddesi).

**Okuma (tasarım girdisi):** M8'de H4 tek tek katlarda sınanamaz (kat başına 1–3 dönem). Çıkarım planın dediği gibi katlar birleştirilerek ve blok bootstrap ile yapılmalı; birleşik testte iki eğilim hücresinin her birinde 8–9 dönem var. Bu da küçük bir sayı: H4 için güç M8 ön kaydında ayrıca hesaplanmalı.

## 3. İş paketleri (öneri)

| # | İş | Neye hizmet eder | Kaynak | Hesap |
|---|---|---|---|---|
| M6a | **HMM ikinci tanımı:** `evaluation/regimes_hmm.py`. `statsmodels` MarkovRegression (kurulu, 0.14.5), USD bazlı XU100 günlük log getirisi, rejime göre değişen ortalama + varyans. Parametreler **nedensel**: çapalarda genişleyen pencereyle yeniden uyum (P tablosu gibi her 63 seans), olasılıklar yalnız filtrelenmiş (Hamilton ileri filtresi, düzeltilmiş değil). Durum sayısı {2, 3} BIC ile, ilk çapada sabitlenir. Test: kesme değişmezliği (B0), filtrelenmiş ↔ düzeltilmiş ayrımı (mutasyon), BIC seçimi | M6 madde 2; AQ3'ün ikinci tanımla tekrarı (K10) | Hamilton (1989), Nystrup vd. (2018) | CPU, dakikalar |
| M6b | **S-reg gözlem bloğu:** yeni gözlem sürümü `marl-obs-v2` = v1 + yayın bloğu [sma_ratio z, vol z, (HMM bull olasılığı − 0,5)], tüm ajanlarda aynı, eleştirmene de girer. v1 bit-eş kalır (bayrak kapalıyken). Normalizasyon yalnız eğitim satırlarından | Plan §3.1.3 S-reg, K10 | — | CPU; parite testi |
| M6c | **Rejime koşullu okuma aracı:** `evaluation/regimes.by_regime` üstüne, iki kolun günlük fark serisinin rejim başına Sharpe farkı + blok bootstrap GA; katlar birleştirilir (M8 maddesi). Self-test sentetik veriyle | M8 H4 analiz planı | Politis & Romano (1994) | CPU |
| M7a | **Karşı-olgusal mesaj maskeleme:** kayıtlı gat politikalarında (M4b, M5) ajan j'nin mesajı sıfırlanınca deterministik değerlendirmede portföy Sharpe'ındaki değişim; ajan başına ve sektör başına | M7; plan §3.3 K4 (dikkat ve geçit kaydı) | Foerster vd. (2018) karşı-olgusal fikir (planda) | venv-marl, değerlendirme yalnız (eğitim yok) |
| M7b | **Dikkat testleri:** tek biçimli dikkat baseline'ı (dikkat ağırlıkları komşular arasında eşitlenince fark), tohumlar arası dikkat varyansı | M7 | Wiegreffe & Pinter (2019); Jain & Wallace (2019) sınırlılık | venv-marl, değerlendirme yalnız |

Sıra önerisi: M6a ve M6c (ana venv, GPU gerektirmez) → M7a/M7b (eğitim yok; kayıtlı politikalarla) → M6b (gözlem değişikliği; parite testi) → §3.2.8 ortam hızı profili → M8 ön kaydı.

### 3.1 M6a durumu (2026-11-20): kod + ölçüm

Kod: `evaluation/regimes_hmm.py` (test `tests/test_regimes_hmm.py` 4/4: B0 kesme değişmezliği + düzeltilmiş olasılık mutasyonu, çapa öncesi uyum, deterministik uyum, BIC, kanonik sıra). Ölçüm: `measurements/phase10-m6/hmm_regimes.json` (geliştirme pencereleri, temiz ağaç `d9a9b4f`) ve `hmm_regimes_test_years.json` (wf-v1 test yılları, yalnız etiket, temiz ağaç `9304b3a`). Başarım okunmadı.

- **Tekrar üretim hatası (test yakaladı, düzeltildi):** statsmodels `search_reps` rastgele başlangıçları NumPy'ın küresel üreticisinden çekiyor; tohumsuz uyum aynı veride farklı parametre veriyordu (kesme değişmezliği ~1e-5 ile kırıldı). Uyum artık sabit tohumla (`FIT_SEED`), küresel durum geri yüklenerek yapılıyor. İlk çapa varsayılan 100 iterasyonda yakınsamıyordu; `MAXITER = 500` ile 31 + 54 uyumun hepsi yakınsadı.
- **Durum sayısı:** ilk çapada (2013-06-03, MARL panelinin ilk seansı) BIC K = 3'ü seçti (K = 2: 8.774, K = 3: 8.738). **Beyan:** ilk çapanın tarihi, scratchpad'deki bir denemede BIC'in çapaya bağlı olduğu (2010 çapasında K = 2) görüldükten sonra "olasılığın gerektiği ilk gün" ilkesine bağlandı; başarım okunmadı.
- **Kanonik sıra tutarlı:** 85 uyumun hepsinde σ'ya göre artan sıra ortalamaya göre azalan sırayla aynı (`mean_desc_with_sd`): en sakin durumun ortalama getirisi en yüksek, en çalkantılı durumunki en düşük.
- **İki hücreli eşleme seçimi (KARAR K-M6-5, 2026-11-20, kullanıcı: `calm_vs_rest`; §5):** "en çalkantılı durum ≥ 0,5 → düşüş benzeri" (`two_cell`) K = 3'te dejenere: o durum nadir bir kriz durumu (eğitim 35 gün, ≥ 20 günlük dönem yok; birleşik test 69 gün). **"En sakin durum ≥ 0,5 ↔ geri kalanlar" (`calm_vs_rest`)** dengeli: eğitim 570 / 773 gün (11 / 15 dönem), doğrulama 253 / 247 (5 / 6), birleşik test 670 / 784 (9 / 11).
- **İkinci tanım birincilin tekrarı değil:** `calm_vs_rest` ile kural tabanlı birincil eğilim etiketinin gün bazında uyuşması geliştirme pencerelerinde 0,74 / 0,70, birleşik testte **0,52** (yıllara göre 0,31–0,66). Test döneminde HMM ikinci tanımı eğilimi değil oynaklık durumunu ölçüyor. Sonuç: M8'de AQ3'ün HMM ile tekrarı "aynı sorunun ikinci etiketle sınanması" diye değil, "oynaklık durumuna koşullu ayrı bir okuma" diye yazılmalı (K10 döngüselliğine ek beyan).

### 3.2 M7a durumu (2026-11-20): karşı-olgusal mesaj maskeleme (betimsel tanı)

Kod: `CommPolicy.set_sender_mask` (yalnız değerlendirme; kalıcı değil; None iken bit-eş, `test_marl_m4` 19/19) ve `scripts/marl/mask_m7.py`. Ölçüm: `measurements/phase10-m7/mask_train.json`, `mask_val.json` (temiz ağaç `fc675ef`). 21 kayıtlı iletişimli politika (M4b G-X ve G-S; M5 mekanizma G0 ve GL; M5 gerçek sinyal T0-G, SB, SB-S), her biri için maskesiz, bütün göndericiler kapalı ve 8 sektör grubunun her biri ayrı kapalı. Eğitim yapılmadı. **42 maskesiz değerlendirmenin hepsi kayıtlı `eval_*.npz` serisiyle bit-eş** (doğru politika, doğru kurulum).

Bütün mesajlar kapalıyken eğitim penceresinde ΔSR (yıllık, fazla getiri; tohum sırasıyla):

| Kol | Tahmin / ipucu | ΔSR, tüm mesajlar kapalı | Sektör maskesi ΔSR aralığı |
|---|---|---|---|
| M4b G-X | gerçek komşu ipucu | −0,60 / −0,09 / −0,23 | −0,12 .. +0,03 |
| M4b G-S | karıştırılmış ipucu | −0,15 / −0,16 / −0,27 | −0,09 .. +0,01 |
| M5 G0 | sentetik, güvensiz | −0,32 / −0,49 / −0,72 | −0,23 .. +0,04 |
| M5 GL | sentetik, güvenli | −0,48 / −0,63 / −0,85 | −0,32 .. +0,02 |
| M5 gerçek T0-G | tahmin yok | −0,37 / −0,39 / −0,39 | −0,25 .. +0,03 |
| M5 gerçek SB | gerçek sinyal | +0,03 / −0,20 / −0,05 | −0,08 .. +0,03 |
| M5 gerçek SB-S | karıştırılmış | −0,05 / −0,19 / −0,17 | −0,14 .. +0,01 |

**Okuma (betimsel tanı, tez bulgusu değil):**
- Mesajları kapatmak, tahmin sinyali hiç olmayan kolda (T0-G) bile Sharpe'ı düşürüyor; gerçek ipuçlu G-X ile karıştırılmış G-S arasında belirgin bir fark yok. **Maskeleme, politikanın mesaj kanalına bağımlılığını ölçüyor, mesajın bilgi taşıyıp taşımadığını değil:** kapalı mesaj, politikanın eğitimde görmediği bir girdi dağılımıdır. Bu, Jain & Wallace (2019) sınırlılığının bu düzende ölçülmüş bir örneğidir.
- Sonuç: M7'de "iletişim bilgi taşıyor" okuması maskelemeyle kurulmaz. Dağılımı koruyan bir kontrol gerekir: göndericilerin mesajlarını ajanlar arasında karıştırmak (M7b ile birlikte önerilir). Bilgi sorusunun asıl sınaması zaten eğitim zamanı kontrolleridir (M4b karıştırılmış ipucu, M5 karıştırılmış güven ve sinyal).
- Doğrulama penceresinde etkiler daha küçük ve işaret karışık (dosyada).

**M8'e giden yolda bilinen engeller (bu belgenin kapsamı dışında, beyan):**
- MARL paneli (`panel_m2.npz`) 2020-12-31'de bitiyor; 2021+ test pencereleri için panel uzatılmalı.
- Satış engeli marj takvimi 2020-12-31'den sonra doğrulanmamış; kod o tarihten sonra hata veriyor (`PRICE_MARGIN_VERIFIED_UNTIL`). 2021+ için marj geçmişi kaynakla doğrulanmalı (M2 tasarımı §2.5).
- PC060-K12 pozitif kontrol etiketi ve pit_v2 blokları yalnız panel aralığında kuruldu.
- Ortam hızı (§3.2.8): M8 tohum gereksinimiyle (~16) uzun koşular ~8 saat mertebesinde.

### 3.3 M7b durumu (2026-11-20): dikkat testleri (betimsel tanı)

**Sade özet.** Ajanların "kime ne kadar kulak verdiği" (dikkat ağırlıkları) neredeyse eşit dağılmış. Ağırlıkları bilerek tamamen eşitlediğimizde karar kalitesi çok az değişti. Yani öğrenilen dikkat ağırlıkları kararların açıklaması olarak okunamaz. Mesajları göndericiler arasında karıştırmak ise (her ajan yine gerçek bir mesaj görüyor, ama yanlış kişiden) gerçek komşu ipuçlu kollarda kararı belirgin bozdu. Hiç tahmin taşımayan kolda bu bozulma, mesajı tamamen kapatmanın yarısı kadardı. Demek ki M7a'daki düşüşün bir kısmı "mesajın bilgisi" değil, "eğitimde görülmemiş girdi" etkisiydi. Tez bulgusu değildir.

**Kod ve ölçüm:**
- Değerlendirme kipleri (yalnız değerlendirme, kaydedilmez, kapalıyken bit-eş): `CommPolicy.set_uniform_attention`, `set_sender_permutation`, `set_attention_record` (`test_marl_m4` 20/20, `test_marl_m5_msg` 20/20).
- Ölçüm `scripts/marl/attn_m7b.py`, M7a'nın aynı 21 politikası. Kayıt `measurements/phase10-m7/m7b_train.json` (temiz ağaç `03c52db`) ve `m7b_val.json` (temiz ağaç `8a9e85c`). Tohum benzerliği aynı kayıtlı matrislerden `--rescore` ile yeniden hesaplandı (`cross_seed_rescore`; ölçüm ve provenance değişmedi).
- **42 temel değerlendirmenin hepsi kayıtlı `eval_*.npz` ile bit-eş.** Eğitim yapılmadı.
- Karıştırma: 3 sabit derangement (tohum 7000–7002), ortalaması verilir.

**Eğitim penceresi, ΔSR (yıllık, fazla getiri; tohum sırasıyla):**

| Kol | Tek biçimli dikkat | Karıştırılmış mesaj (3 permütasyon ort.) | M7a: tüm mesajlar kapalı | Normalize dikkat entropisi |
|---|---|---|---|---|
| M4b G-X (gerçek komşu ipucu) | −0,14 / −0,01 / −0,06 | −0,51 / −0,08 / −0,35 | −0,60 / −0,09 / −0,23 | 0,990–0,994 |
| M4b G-S (karıştırılmış ipucu) | −0,04 / −0,01 / −0,09 | −0,16 / −0,00 / −0,16 | −0,15 / −0,16 / −0,27 | 0,992–0,996 |
| M5 G0 (sentetik, güvensiz) | −0,05 / −0,02 / −0,11 | −0,23 / −0,49 / −0,68 | −0,32 / −0,49 / −0,72 | 0,990–0,999 |
| M5 GL (sentetik, güvenli) | −0,08 / −0,09 / −0,11 | −0,40 / −0,59 / −0,74 | −0,48 / −0,63 / −0,85 | 0,990–0,997 |
| M5 gerçek T0-G (tahmin yok) | −0,07 / −0,11 / −0,20 | −0,11 / −0,22 / −0,17 | −0,37 / −0,39 / −0,39 | 0,981–0,994 |
| M5 gerçek SB (gerçek sinyal) | −0,03 / −0,03 / −0,01 | −0,12 / −0,00 / −0,06 | +0,03 / −0,20 / −0,05 | 0,995–0,998 |
| M5 gerçek SB-S (karıştırılmış) | −0,05 / −0,05 / −0,04 | −0,07 / +0,00 / −0,15 | −0,05 / −0,19 / −0,17 | 0,995–0,998 |

Doğrulama penceresinde etkiler daha küçük. Tek biçimli dikkat −0,10..+0,03. Karıştırılmış mesaj G-X'te −0,42 / −0,07 / −0,07, G0/GL'de −0,22..−0,34, G-S'te ~0. Ayrıntı dosyada.

**Tohumlar arası dikkat benzerliği:** Ham ortalama dikkat matrislerinin tohumlar arası korelasyonu her kolda ≈ 1,00 çıktı. Bu bir ölçüm hatası değil, ama anlamsız: dikkat neredeyse tek biçimli olduğu için matris satır başına ≈ 1/komşu sayısıdır ve korelasyon yalnız graf yapısını ölçer. Anlamlı ölçüt, tek biçimliden sapmanın (Wᵢⱼ − 1/kᵢ) korelasyonudur. Bu sapma çok küçüktür (girdi başına ortalama |sapma| 0,001–0,01, ağırlıklar ~0,1–0,3). Ama tohumlar arasında tutarlıdır: eğitimde kol ortalaması r 0,63 (SB-S) – 0,95 (G-S), doğrulamada 0,69–0,91.

**Okuma (betimsel tanı, tez bulgusu değil):**
- **Wiegreffe & Pinter tek biçimli baseline testi:** Öğrenilen dikkat ağırlıkları neredeyse tek biçimli (entropi ≥ 0,98). Eşitlenince SR değişimi küçük (eğitimde en fazla −0,20, çoğunlukla −0,1'in altında). Dikkat ağırlıkları bu politikalarda kararın açıklaması olarak okunmaz. Bilgi, ağırlıklardan çok mesajın (değer izdüşümünün) içeriğinden taşınıyor.
- **Karıştırılmış mesaj, M7a'nın kontrolü:**
  - Tahmin taşımayan T0-G'de karıştırma kapatmanın yaklaşık yarısı kadar etki yapıyor (−0,11..−0,22 ↔ −0,37..−0,39). M7a'daki düşüşün bir kısmı dağılım dışı girdi etkisidir.
  - Gerçek komşu ipuçlu G-X'te karıştırma, karıştırılmış ipuçlu G-S'ten belirgin daha çok bozuyor (−0,51 / −0,08 / −0,35 ↔ −0,16 / −0,00 / −0,16). Sentetik komşu sinyalli G0/GL'de karıştırma kapatmaya yakın.
  - Bu, mesajların kimden geldiğinin bilgi taşıdığı kollarda önemli olduğuyla tutarlıdır. İstatistik testi yoktur; 3 tohum.
- **Tutarlı ama küçük sapma:** Tek biçimliden sapma tohumlar arasında tekrarlanıyor. Olası bir açıklama, sapmanın öğrenilmiş bir ilgiden değil girdi yapısından gelmesidir; örneğin anahtar izdüşümleri, bazı hisselerin özellik düzeyinden etkilenebilir. Bu ölçümle ayrılmaz.
- **Sınırlar:** Dikkat nedensel bir açıklama değildir (Jain & Wallace 2019). Karıştırma alıcının kendi yuvasını da değiştirir; alıcının kendi kodu çıkış başına ayrıca girer. Üç permütasyonun ortalaması bir dağılım değildir.

## 4. Sınırlar

- Rejim etiketi bir **koşullama** değişkenidir; S-reg açıkken aynı etiket ajanın girdisidir (K10 döngüselliği). AQ3 sonucu HMM tanımıyla da tekrarlanır.
- Geliştirme pencerelerindeki rejime koşullu her okuma betimseldir. Pilot sonuçlarına bakarak rejim tanımı ya da hücre sayısı seçmek **yasak kurtarma** sayılır; tanım bu belgede, sonuçtan önce sabitlenir.
- M7 analizleri açıklama değil tanıdır: dikkat ağırlığı nedensel bir açıklama sayılmaz (Jain & Wallace 2019).

## 5. Kullanıcı kararları (2026-11-20: dördü de öneri yönünde KAPANDI)

**Kapanan kararlar:** K-M6-1 = iki hücre (eğilim) birincil, altı hücre betimsel; K-M6-2 = 63 seansta bir genişleyen yeniden uyum, yalnız filtrelenmiş, durum sayısı {2, 3} BIC ile ilk çapada sabit; K-M6-3 = S-reg ana tasarımda (planı izle) + 3 tohumlu maliyetsiz kısa pilot; K-M6-4 = 2021+ test yıllarında yalnız rejim etiketi sayımı (başarım okunmaz, beyan edilir).

- **K-M6-1 — H4'ün rejim çözünürlüğü.** Öneri: **birincil çözünürlük iki hücre (eğilim)**, altı hücre yalnız betimsel. Gerekçe §2: altı hücrede dönem sayısı sıfıra iniyor. Alternatif: üç oynaklık düzeyi (eğilimden bağımsız).
- **K-M6-2 — HMM uyum protokolü.** Öneri: çapalarda (63 seans) genişleyen pencereyle yeniden uyum, yalnız filtrelenmiş olasılık, durum sayısı {2, 3} BIC ile ilk çapada sabit, girdi USD bazlı XU100 günlük log getirisi. Alternatif: yalnız eğitim penceresinde tek uyum (daha basit, ama test penceresinde parametre bayatlar).
- **K-M6-3 — S-reg ana tasarımda mı?** Plan "ana tasarımda sabit, katkısı tek ablasyonla" diyor. Bu, M8'in temel gözleminin M3–M5 pilotlarından farklı olacağı (yeni gözlem sürümü) anlamına gelir. Öneri: planı izle (S-reg açık, tek ablasyon "S-reg kapalı"); M6b'yi bir kısa pilotla (3 tohum, maliyetsiz, ~1 sa) "S-reg açıkken öğrenme bozulmuyor mu" diye sına, sonuç tez bulgusu değil.
- **K-M6-5 — HMM iki hücreli eşleme (2026-11-20, kullanıcı; öneri yönünde KAPANDI): `calm_vs_rest`.** "En sakin durum olasılığı ≥ 0,5" ↔ geri kalanlar. `two_cell` (en çalkantılı ↔ geri kalan) K = 3'te dejenere olduğu için birincil değil, yalnız betimsel. Kodda tek kaynak `evaluation/regimes_hmm.py::PRIMARY_TWO_CELL`. M8'de bu tanım eğilimin tekrarı değil, "oynaklık durumuna koşullu ayrı okuma" olarak yazılır (§3.1).
- **K-M6-4 — 2021+ rejim sayımı.** Öneri: M8 ön kaydından önce, wf-v1 test yıllarında **yalnız rejim etiketinin** gün/dönem sayımı (piyasa endeksi verisi; hiçbir politika ya da başarım okunmaz). Bu, M8'de H4 için kaç dönem olacağını önceden söyler. Pencere dokunulmamışlık kuralının (Faz 9 R1 `EXPOSED_EVAL_WINDOWS`) kapsamına girip girmediği kullanıcı kararıdır.

## 6. M6b kısa pilot ön kaydı (K-M6-3; DONDURULDU 2026-11-20, koşudan önce)

**Soru (sağlamlık, tez bulgusu değil):** S-reg yayın bloğu gözleme eklenince öğrenme bozuluyor mu? Plan S-reg'i ana tasarımda açık tutuyor (Faz 10 §3.1.3); bu pilot onun bir sakıncasını erken yakalamak için. S-reg'in değerini ölçmez; o M8'deki tek ablasyondur.

**Düzen:**
- Blok: `data/marl/sreg_m6.npz` (`build_sreg_m6.py`; kayıt `measurements/phase10-m6/sreg_block.json`): [z(sma_ratio), z(vol)], birincil (USD bazlı) etiketin sürekli özellikleri, as-of, ölçek yalnız eğitim satırlarından; kırpma 0, B0 5/5.
- Kollar: **R0-N** (none, tahminsiz, S-reg yok) ve **SR-N** (none, tahminsiz, S-reg açık). Tek etken S-reg; mimari, tohum ve diğer her şey aynı. İletişimsiz mimari, en ucuz ve S-reg'in iletişimden bağımsız etkisini verir.
- Sabitler: **maliyetsiz** (K-M6-3), α 0,25, MAPPO, eleştirmen MLP (tüm yerel gözlemler; S-reg eleştirmene de girer), F = 600.000, lr 5e-4, `init_std` 0,12, son ağırlıklar, deterministik tam pencere değerlendirme. Eğitim 2013-08..2018-12, doğrulama 2019–2020.
- **Tohumlar 12, 13, 14** (yeni; 0–11 önceki pilotlarda kullanıldı ve sonuçları görüldü).
- Süre tahmini: 6 koşu, iki eş zamanlı, ~25–30 dk (none kolu ~6–7 dk/koşu, M5 gerçek sinyal kayıtları).

**Okuma kuralı (önceden sabit; `scripts/marl/analyze_m6sreg.py`, self-test):** D = D(SR-N, R0-N), yıllık Sharpe farkı, fazla getiri, eşli tohum, ortak HAC.
| Okuma | Koşul (hepsi) | Sonuç |
|---|---|---|
| `BOZULMA` | eğitimde D < 0, iki yönlü p < 0,05, **üç tohumun üçünde** fark negatif, doğrulamada D < 0 ve en az 2/3 tohumda negatif | S-reg ana tasarıma bu haliyle girmez; karar kullanıcıda |
| `IYILESME_ISARETI` | aynı koşullar pozitif yönde | Keşifsel; S-reg plan gereği ana tasarımda, değeri M8 ablasyonunda sınanır. Tez bulgusu değil |
| `FARK_YOK` | ikisi de değil | Öğrenme bozulmuyor; plan izlenir (S-reg ana tasarımda) |

- Betimsel (kararı değiştirmez): tohum başına Sharpe, devir, nakit ağırlığı, eğitim eğrileri (`curves_m4a.py`), "her koşuda son 10 güncelleme ilk 10'dan yüksek" denetimi.
- **Sınırlar:** 3 tohumda ortak HAC tohum varyansını taşımaz (M5 §4.11); bu yüzden üç tohumda aynı işaret ve doğrulama koşulu birlikte aranır. Maliyetsiz düzen 20 bps'deki davranışı söylemez. Pilot tek bir S-reg tanımını (iki sürekli özellik) sınar.
- **Yasak kurtarma:** ek tohum, özellik kümesini, maliyeti ya da eşiği sonuca bakarak değiştirmek yapılmaz; gerekirse yeni ön kayıt.
- Hakem okuması yapılmadı (kademeli kural: kısa ve ucuz pilot, tez bulgusu üretmez; beyan).

### 6.1 Sonuç (2026-11-20): FARK_YOK — S-reg açıkken öğrenme bozulmuyor

**Sade özet.** Bütün ajanlara aynı piyasa durumu bilgisini (endeksin eğilimi ve oynaklığı) ek gözlem olarak verdik ve vermediğimiz kolla karşılaştırdık. Eğitimde ek bilgili kol üç tohumda da biraz öndeydi. Doğrulamada iki tohumda gerideydi. Fark iki pencerede de istatistik olarak ayırt edilemedi ve işareti pencereler arasında değişti. Ön kaydın sorusu "öğrenmeyi bozuyor mu" idi; bozduğuna dair işaret yok. Plan gereği S-reg ana tasarımda kalır, değeri M8'deki tek ablasyonla sınanır. Bu bir tez bulgusu değildir.

**Kaynak:** `measurements/phase10-m6/m6sreg_results.json` (`analyze_m6sreg.py`), eğriler `m6sreg_curves.json` (`curves_m4a.py`, her koşunun `run.log`'u). Sayılar bu dosyalardan yuvarlanmıştır.

**Koşu izi:** 6/6 koşu, F = 600.000, maliyetsiz, tohum 12–14. Hepsi commit `a207e67`'den, temiz ağaçla. Koşu başına eğitim 531–542 sn (iki eş zamanlı). Parametre sayısı R0-N 77.826, SR-N 78.338. S-reg bloğunun SHA'sı üç SR-N koşusunda aynı (`dde75389…`).

| | Eğitim | Doğrulama |
|---|---|---|
| D = D(SR-N, R0-N), yıllık (SE) | +0,19 (0,23) | −0,28 (0,29) |
| İki yönlü p | 0,41 | 0,33 |
| Tohum farkları (12, 13, 14) | +0,01 / +0,30 / +0,25 | +0,16 / −0,54 / −0,45 |
| SR SR-N / R0-N | 1,21 / 1,01 / 1,07 ↔ 1,20 / 0,71 / 0,82 | 0,81 / 0,44 / 0,30 ↔ 0,65 / 0,98 / 0,76 |

**Okuma (§6 tablosu):** `BOZULMA` koşulları sağlanmadı: eğitimde D > 0, p 0,41. `IYILESME_ISARETI` koşulları da sağlanmadı: p 0,41 ve doğrulamada D < 0. Okuma **FARK_YOK**: öğrenme bozulmuyor; plan izlenir (S-reg ana tasarımda).

**Betimsel (kararı değiştirmez):**
- **Öğrenme denetimi:** altı koşunun hepsinde son 10 güncellemenin ortalama ödülü ilk 10'unkinden yüksek. R0-N tohum 14 sınırda (14,0 → 14,5).
- Eğitim ödül eğrileri SR-N'de daha yüksek bitiyor: son 10 ortalaması 26–30, R0-N'de 14,5–21,5. Bu eğitim içi bir ölçüdür; doğrulamada tutmadı.
- **İşaret dönüşü:** eğitimde üç tohumda pozitif, doğrulamada iki tohumda negatif. Bu, ek yayın bloğunun eğitim penceresine uyum sağlıyor olabileceğine dair bir uyarıdır. 3 tohumla bu ayrılamaz. M8 ablasyonunda doğrulama ve test pencereleri bu yüzden ayrıca raporlanmalı.
- Devir: SR-N eğitim 0,31 / doğrulama 0,32, R0-N 0,30 / 0,28. Nakit ağırlığı iki kolda da ~%1–4.

**Sınırlar:** 3 tohum, maliyetsiz, iletişimsiz mimari, tek bir S-reg tanımı (iki sürekli özellik). Ortak HAC tohum varyansını taşımaz (M5 §4.11). Pilot S-reg'in değerini ölçmez.

**Yasak kurtarma (§6):** ek tohum, özellik kümesi, maliyet ya da eşik sonuca bakılarak değiştirilmez.

---

## Versiyon geçmişi

- **v0.8 (2026-11-20):** §3.3 M7b ölçümü (betimsel): dikkat neredeyse tek biçimli; tek biçimli baseline'a göre SR değişimi küçük; karıştırılmış mesaj, M7a maskelemenin dağılım dışı payını ayırıyor; tohum benzerliği tek biçimliden sapma üzerinden.
- **v0.7 (2026-11-20):** §6.1 M6b pilot sonucu: FARK_YOK (eğitim D +0,19, p 0,41; doğrulama −0,28); S-reg ana tasarımda kalır. Ön kayıt değişmedi.
- **v0.6 (2026-11-20):** K-M6-5 kapandı (HMM iki hücre = `calm_vs_rest`, `PRIMARY_TWO_CELL`). M7b kodu (tek biçimli dikkat, karıştırılmış mesaj, dikkat kaydı; `attn_m7b.py`) yazıldı; ölçüm koşulmadı.
- **v0.5 (2026-11-20):** §6 M6b kısa pilot ön kaydı DONDURULDU (R0-N ↔ SR-N, tohum 12–14, maliyetsiz; okuma BOZULMA / IYILESME_ISARETI / FARK_YOK). M6b kodu: `build_sreg_m6.py`, `extras_path`.
- **v0.4 (2026-11-20):** §3.2 M7a ölçümü: maskeleme mesaj kanalına bağımlılığı ölçüyor, bilgiyi değil (tahminsiz kolda da düşüş); dağılımı koruyan karıştırma kontrolü önerisi.
- **v0.3 (2026-11-20):** §3.1 M6a kod + ölçüm: K = 3, `calm_vs_rest` önerisi, test yıllarında birincil eğilimle uyuşma 0,52 (ikinci tanım oynaklık durumunu ölçüyor). M6c kodu (`evaluation/regime_contrast.py`, test 3/3).
- **v0.2 (2026-11-20):** K-M6-1..4 kullanıcı tarafından öneri yönünde kapatıldı. §2.1: wf-v1 test yıllarında yalnız etiket sayımı (birleşik testte eğilim hücresi başına 8–9 dönem, kat başına 1–3).
- **v0.1 (2026-11-20):** İlk taslak. Ön ölçüm (§2): altı hücreli rejim geliştirme pencerelerinde sınanamaz, eğilim (iki hücre) 3–6 dönem. İş paketleri M6a–c, M7a–b; M8 engelleri beyan; kararlar K-M6-1..4. Kod: yalnız `regime_windows_m6.py`.
