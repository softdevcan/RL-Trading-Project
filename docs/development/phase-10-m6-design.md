# Faz 10 M6/M7 — Rejim ve yorumlanabilirlik: tasarım

**Sürüm:** v0.3 · 2026-11-20 · dal `feat/phase10-m6` · **Durum:** kararlar K-M6-1..4 KAPANDI (2026-11-20, kullanıcı; dördü de öneri yönünde, §5). Sıradaki: §2.1 2021+ etiket sayımı, M6a HMM, M6c okuma aracı.
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
- **İki hücreli eşleme seçimi (öneri, kullanıcı onayı bekliyor):** "en çalkantılı durum ≥ 0,5 → düşüş benzeri" (`two_cell`) K = 3'te dejenere: o durum nadir bir kriz durumu (eğitim 35 gün, ≥ 20 günlük dönem yok; birleşik test 69 gün). **"En sakin durum ≥ 0,5 ↔ geri kalanlar" (`calm_vs_rest`)** dengeli: eğitim 570 / 773 gün (11 / 15 dönem), doğrulama 253 / 247 (5 / 6), birleşik test 670 / 784 (9 / 11).
- **İkinci tanım birincilin tekrarı değil:** `calm_vs_rest` ile kural tabanlı birincil eğilim etiketinin gün bazında uyuşması geliştirme pencerelerinde 0,74 / 0,70, birleşik testte **0,52** (yıllara göre 0,31–0,66). Test döneminde HMM ikinci tanımı eğilimi değil oynaklık durumunu ölçüyor. Sonuç: M8'de AQ3'ün HMM ile tekrarı "aynı sorunun ikinci etiketle sınanması" diye değil, "oynaklık durumuna koşullu ayrı bir okuma" diye yazılmalı (K10 döngüselliğine ek beyan).

**M8'e giden yolda bilinen engeller (bu belgenin kapsamı dışında, beyan):**
- MARL paneli (`panel_m2.npz`) 2020-12-31'de bitiyor; 2021+ test pencereleri için panel uzatılmalı.
- Satış engeli marj takvimi 2020-12-31'den sonra doğrulanmamış; kod o tarihten sonra hata veriyor (`PRICE_MARGIN_VERIFIED_UNTIL`). 2021+ için marj geçmişi kaynakla doğrulanmalı (M2 tasarımı §2.5).
- PC060-K12 pozitif kontrol etiketi ve pit_v2 blokları yalnız panel aralığında kuruldu.
- Ortam hızı (§3.2.8): M8 tohum gereksinimiyle (~16) uzun koşular ~8 saat mertebesinde.

## 4. Sınırlar

- Rejim etiketi bir **koşullama** değişkenidir; S-reg açıkken aynı etiket ajanın girdisidir (K10 döngüselliği). AQ3 sonucu HMM tanımıyla da tekrarlanır.
- Geliştirme pencerelerindeki rejime koşullu her okuma betimseldir. Pilot sonuçlarına bakarak rejim tanımı ya da hücre sayısı seçmek **yasak kurtarma** sayılır; tanım bu belgede, sonuçtan önce sabitlenir.
- M7 analizleri açıklama değil tanıdır: dikkat ağırlığı nedensel bir açıklama sayılmaz (Jain & Wallace 2019).

## 5. Kullanıcı kararları (2026-11-20: dördü de öneri yönünde KAPANDI)

**Kapanan kararlar:** K-M6-1 = iki hücre (eğilim) birincil, altı hücre betimsel; K-M6-2 = 63 seansta bir genişleyen yeniden uyum, yalnız filtrelenmiş, durum sayısı {2, 3} BIC ile ilk çapada sabit; K-M6-3 = S-reg ana tasarımda (planı izle) + 3 tohumlu maliyetsiz kısa pilot; K-M6-4 = 2021+ test yıllarında yalnız rejim etiketi sayımı (başarım okunmaz, beyan edilir).

- **K-M6-1 — H4'ün rejim çözünürlüğü.** Öneri: **birincil çözünürlük iki hücre (eğilim)**, altı hücre yalnız betimsel. Gerekçe §2: altı hücrede dönem sayısı sıfıra iniyor. Alternatif: üç oynaklık düzeyi (eğilimden bağımsız).
- **K-M6-2 — HMM uyum protokolü.** Öneri: çapalarda (63 seans) genişleyen pencereyle yeniden uyum, yalnız filtrelenmiş olasılık, durum sayısı {2, 3} BIC ile ilk çapada sabit, girdi USD bazlı XU100 günlük log getirisi. Alternatif: yalnız eğitim penceresinde tek uyum (daha basit, ama test penceresinde parametre bayatlar).
- **K-M6-3 — S-reg ana tasarımda mı?** Plan "ana tasarımda sabit, katkısı tek ablasyonla" diyor. Bu, M8'in temel gözleminin M3–M5 pilotlarından farklı olacağı (yeni gözlem sürümü) anlamına gelir. Öneri: planı izle (S-reg açık, tek ablasyon "S-reg kapalı"); M6b'yi bir kısa pilotla (3 tohum, maliyetsiz, ~1 sa) "S-reg açıkken öğrenme bozulmuyor mu" diye sına, sonuç tez bulgusu değil.
- **K-M6-4 — 2021+ rejim sayımı.** Öneri: M8 ön kaydından önce, wf-v1 test yıllarında **yalnız rejim etiketinin** gün/dönem sayımı (piyasa endeksi verisi; hiçbir politika ya da başarım okunmaz). Bu, M8'de H4 için kaç dönem olacağını önceden söyler. Pencere dokunulmamışlık kuralının (Faz 9 R1 `EXPOSED_EVAL_WINDOWS`) kapsamına girip girmediği kullanıcı kararıdır.

---

## Versiyon geçmişi

- **v0.3 (2026-11-20):** §3.1 M6a kod + ölçüm: K = 3, `calm_vs_rest` önerisi, test yıllarında birincil eğilimle uyuşma 0,52 (ikinci tanım oynaklık durumunu ölçüyor). M6c kodu (`evaluation/regime_contrast.py`, test 3/3).
- **v0.2 (2026-11-20):** K-M6-1..4 kullanıcı tarafından öneri yönünde kapatıldı. §2.1: wf-v1 test yıllarında yalnız etiket sayımı (birleşik testte eğilim hücresi başına 8–9 dönem, kat başına 1–3).
- **v0.1 (2026-11-20):** İlk taslak. Ön ölçüm (§2): altı hücreli rejim geliştirme pencerelerinde sınanamaz, eğilim (iki hücre) 3–6 dönem. İş paketleri M6a–c, M7a–b; M8 engelleri beyan; kararlar K-M6-1..4. Kod: yalnız `regime_windows_m6.py`.
