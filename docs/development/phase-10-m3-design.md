# Faz 10 M3 — Algoritma katmanı: tasarım ve M3a ön kaydı

**Sürüm:** v0.6 · 2026-11-19 · dal `feat/phase10-m3b` (M3a ön kaydı v0.2 ve M3b ön kaydı v1.1 değişmedi; v0.6 yalnız Aşama 1 sonucunu ekler, §6.7)
**Üst belgeler:**
- [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.2, §4
- Faz 9 planı §M3 (IPPO, MAPPO, parametre paylaşımı + ajan kimliği, MAPPO önerileri)
- [phase-10-m2-design.md](phase-10-m2-design.md): §9 #4, #10, #11 ve §9.1 (pozitif kontrol PC060-K12)

**Durum:** **M3a koşuldu, etiket GEÇTİ (2026-11-18, §5.2).** Bu bir boru hattı kapısıdır, tez bulgusu değildir; sinyal bilerek gelecekten türetilmiştir. Ön kayıt v0.2 ilk eğitim koşusundan önce commit'lendi ve sonuçtan sonra değiştirilmedi. **M3b Aşama 1 koşuldu, etiket KISMİ (2026-11-19, §6.7); ön kayıt gereği Aşama 2 koşulmadı.**

Bu belge M3'ün yapısını tanımlar. Kod ile belge farklıysa kod hatalıdır.

---

## 0. Kapsam

| Madde | Burada | Kod |
|---|---|---|
| BenchMARL görevi (M2 §9 #4) | §1 | `marl/benchmarl_task.py` |
| Bölüm sonu: truncation + bootstrap (M2 §9 #10, hakem-rl R6) | §1 | `test_marl_m3` [2] |
| Başlangıç gürültüsü (R7), deterministik değerlendirme | §2 | `marl/m3_runner.py`, `test_marl_m3` [4] |
| Baseline'lar aynı muhasebeyle (M2 §9 #11) | §3 | `m3_runner.rule_series` |
| Pozitif kontrol PC060-K12 + karıştırılmış kontrol (M2 §9.1; hakem-rl M3a #2) | §4, §5 | `scripts/marl/build_pred_m3.py` (`--shuffle`) |
| **M3a ön kaydı: boru hattı kapısı** | §5 | `train_m3.py`, `eval_m3.py`, `analyze_m3.py` (ön kayıttan sonra) |
| **M3b ön kaydı:** 20 bps kapısı ve M4 temel yapılandırması (IPPO, R5, α) | §6 | `run_m3b.py`, `analyze_m3b.py` |

**M3'ün sorusu.** M0' ve Faz 9 §9.37–9.39'daki boş sonucun en güçlü açıklaması şuydu: öğrenme, bilinen bir sinyali bile ayıklayamıyor. Yeni ortam bu açıklamanın kaynaklarını değiştirdi:
- yerel, normalize gözlem
- K12'ye hizalı sinyal ufku
- tek kez sayılan maliyet
- bootstrap'lı bölüm sonu

**M3a**, aynı soruyu yeni boru hattında, gerçek sinyalli hiçbir MARL koşusundan önce sınar. Sonuç tez iddiası değildir; bir geliştirme kapısıdır.

---

## 1. Görev ve eğitim çerçevesi

- **Görev:**
  - BenchMARL 1.4 + TorchRL 0.7.2 (K9).
  - Tek grup `agent` altında 30 ajan; ajan ekseni tensör boyutudur.
  - Tam parametre paylaşımı ve gözlemde ajan kimliği (K24).
- **Bölüm sonu:**
  - Ortam bölüm sonunda `terminated=False`, `truncated=True` verir.
  - MAPPO/IPPO kaybı grup `terminated` anahtarına bağlıdır.
  - TorchRL GAE bu durumda V(s_T)'yi bootstrap eder.
  - Üç halka da test edildi (`test_marl_m3` [2]).
- **Tohum:** `OMP_NUM_THREADS=1` ve `MKL_NUM_THREADS=1` torch import'undan önce ayarlanınca aynı tohumla iki CPU koşusu bit-eş. `torch.set_num_threads` süreç ortasında yetmiyor; 2 güncellemede 1,9e-9 fark kaldı (`test_marl_m3` [3]). GPU'da bit-eşlik iddia edilmez (M1 ile aynı).

## 2. Hiperparametreler (sabit; M3a boyunca değişmez)

| Ad | Değer | Kaynak / gerekçe |
|---|---|---|
| Algoritma | MAPPO | Faz 9 §M3; M3a'da tek algoritma |
| Eleştirmen girdisi | tüm ajanların yerel gözlemleri (`use_state=False`) | M1 ile aynı; `state()` kolu M3b'de (R5) |
| Paylaşım | tam + ajan kimliği | K24 |
| Epoch / mini-parti | 10 epoch, parti bölünmez (`minibatch_size = frames_per_batch` = 6000) | MAPPO önerisi (Faz 9 §M3, Öneri 3) |
| Clip ε | 0,2 | MAPPO önerisi (Öneri 4); BenchMARL varsayılanı |
| Değer normalizasyonu | **yok** | BenchMARL 1.4 MAPPO'da seçenek yok → **sapma, beyan** (Öneri 1) |
| **lr** | **5e-4** | v0.2 (hakem-rl M3a #4). Varsayılan 5e-5 ile parti başına 10 gradyan adımı ve ~1 saatlik bütçede ağırlıklar ancak çok az kayar; KALDI bir lr artefaktı olabilirdi. Önceden, sonuç görülmeden sabitlendi |
| γ, λ, grad kırpma | 0,99, 0,9, 5 | BenchMARL 1.4 varsayılanları |
| Entropi katsayısı | 0 | BenchMARL varsayılanı |
| Dağılım | TanhNormal, eylem [−B, B], B = 5 + ln 30 | M2 §3.1 |
| **Başlangıç std** (tanh öncesi) | **0,12** (`biased_softplus_0.12`) | R7: ölçüldü, varsayılan 1,0 örneklemde ~4 logit gürültü (her gün ~tam devir), 0,12 ~0,77 logit (`test_marl_m3` [4]) |
| Ödül | `'excess'`, α = 0,25, ölçek 100 | M2 §2.2 kararı; α ön elemesi M3a dışında |
| Bölüm | 252 adım, eğitim penceresinde rastgele başlangıç, 10 ortam, partide 6000 kare | M1/M2 ile aynı |
| Gözlem | `marl-obs-v1`; S-obs kolunda +4 (p_up_cal − 0,5, z(pred_logret), z(disagreement), pred_ok) | M2 §5 |

## 3. Değerlendirme ve baseline'lar

- **Politika:** deterministik eylem (TanhNormal modu), pencerenin tamamı, kesirli mod, eğitim ortamının `norm_stats`'ı (held-out'ta `'fit'` reddedilir). Seri: günlük R, X = R − rf, devir, nakit ağırlığı (`eval_m3.py`).
- **Baseline'lar** ortamın aynı muhasebesiyle koşar (`m3_runner.rule_series`): aynı `fill_prices`, yani kilit 20 ve v0.7 marj takvimi; aynı `project`, maliyet ve A8 rf.
  - 1/N: her gün tam yatırımlı
  - al-tut: ilk gün 1/N, sonra işlem yok
  - nakit: X = 0

  Sınırsız 1/N ayrı duyarlılık satırıdır (M2 §9 #11).
- **Pencereler:**
  - eğitim: 2013-08-27..2018-12-31, 1343 gün
  - doğrulama: 2019-01-02..2020-12-31, 500 gün
  - **2021 ve sonrası okunmaz.**

## 4. Pozitif kontrol PC060-K12

- **Üretim:**
  - `positive_control_table` (AUC 0,60), taban tablo `pit_v1_seed42` (maske ve ölçek).
  - Etiket 1[Ofₜ₊₂ > Ofₜ₊₁].
  - Beş üreteç tohumu: {20261115, 1, 2, 3, 4}.
  - Bloklar `data/marl/pred_pc060_k12_g{tohum}.npz`; SHA ve gerçekleşen AUC (0,595–0,602) `measurements/phase10-m3/pred_blocks.json`'da.
- **Sinyal bilerek gelecekten türetildi.** Yalnız boru hattı kontrolü için kullanılır; tez sonucu değildir.
- **RL'siz referans** (M2 §9.1, `pc_alignment_seeds.json`; S_yön − 1/N, E1 muhasebesi, 20 bps): eğitim penceresinde 0,08–0,30, doğrulamada 0,25–0,67 (9 tohum). Eğitim penceresinin SE'si ~0,12. Bu yüzden 20 bps'de RL'nin sinyali alıp almadığını ayırt etmek zor; **birincil kapı maliyetsiz kurulur** (§5). Maliyetsizde RL'siz brüt değer ~1,5 (M2 §9.1 E1 brüt, kayıtlı tohum).

---

## 5. ÖN KAYIT v0.2 — M3a boru hattı kapısı (2026-11-16, ilk eğitim koşusundan ÖNCE)

v0.1 (`331730e`) hiç koşulmadan değiştirildi. Gerekçeler:
- hakem-rl M3a okuması (§8)
- kullanıcı kararı: toplam süre ~1 saat; v0.1'in 4 kol × 5 tohum, ~8 saatlik planı reddedildi

**Soru:** Yeni MARL boru hattında MAPPO, gözlemine verilen bilinen bir sinyali (PC060-K12) eyleme dönüştürüp değer üretebiliyor mu?

**Kollar.** İkisi de maliyetsiz (0 bps), 3 eğitim tohumu s ∈ {0, 1, 2}. Tohum s, üreteç tohumu g_s ∈ {20261115, 1, 2} ile eşlenir.

| Kol | Gözlemdeki tahmin bloğu | Rol |
|---|---|---|
| A1 | PC060-K12 (g_s); etiketle AUC 0,595–0,600 | sinyalli |
| S1 | **aynı blok, satırları (günleri) tohum g_s ile permüte**: boyut, maske, kesit yapısı ve marjinal dağılım aynı, zaman hizası yok. Etiketle AUC 0,500–0,504 | **kontrol** |

AUC değerleri `measurements/phase10-m3/pred_blocks*.json` → `auc_block_vs_label`.

**Neden S1:** hakem-rl M3a #2. Tahminsiz bir kol (A0) gözlem boyutunu ve kapasiteyi eşitlemez. Günlük ve tarihle hizalı bir blok, eğitim penceresinde ezbere yardım eden bir parmak izi olabilir. S1 bu parmak izini taşır ama sinyali taşımaz; fark yalnız sinyalden gelebilir. Tahminsiz kol M3a'da yok (M3b).

**Eğitim:**
- Her koşuda aynı F kare (§5.1).
- **lr 5e-4** (§2; hakem-rl M3a #4).
- Diğer her şey §2'deki gibi.
- **Her koşunun son parti sonrası ağırlıkları değerlendirilir; kontrol noktası seçimi ve erken durdurma yok** (hakem-rl #6).

**Değerlendirme:** deterministik, tam pencere, maliyetsiz. Eğitim penceresi 1343 gün, doğrulama penceresi 500 gün. 2021+ yok.

**Okuma kuralı (önceden; `scripts/marl/analyze_m3.py`, `--self-test` 3/3).** SR yıllık, X üzerinde. Δ_s = SR(A1_s) − SR(S1_s), D = ort_s Δ_s.

1. **GEÇTİ = üçü birden:**
   - (i) Eğitim penceresi: D ≥ **0,3** ve ortak HAC (`sharpe_diff_hac_pooled`) tek yönlü p < 0,05 (hakem-rl #5: büyüklük eşiği).
   - (ii) Eğitim penceresi, **tanımsal sıralama kuralı:** min_s SR(A1_s) > max_s SR(S1_s). 3'e 3 kesin permütasyon testinde bu sıralamanın tek yönlü p'si tam 1/20 = 0,05; kural için ≤ 0,05 kabul edilir (hakem-rl #1 ve ikinci okuma #1: tohum varyansı; ortak HAC yalnız gün gürültüsünü ölçer). 3 tohumla ulaşılabilecek en küçük p bu olduğundan bir **güç sınırlamasıdır** (§8).
   - (iii) Doğrulama penceresi, **işaret kontrolü (yetersiz güç):** D > 0 ve en az 2/3 tohumda Δ_s > 0. Bu bir genelleme kanıtı değildir; H0 altında da ~0,5 olasılıkla sağlanır. Yalnız çelişkiyi dışlar (hakem-rl #3, ikinci okuma #2).

   Okuma: "Eğitim penceresinde, aynı kapasite ve parmak izine sahip kontrole karşı, maliyetsiz koşulda sinyal aktarımı gözlendi; doğrulamada işaret çelişmiyor." M3b ve M4 açılır.
2. **KISMİ:** eğitimde D > 0 ve p < 0,05 var, ama GEÇTİ koşullarından biri eksik. Gerçek sinyalli MARL koşusu yapılmaz.
3. **GÖZLENMEDİ:** "Bu bütçe ve lr'de aktarım gözlenmedi." Bu bir kanıt yokluğudur; "aktaramıyor" denmez (hakem-rl #4). Keşifsel tanı, ön kayıtsız ve tez dışı:
   - tek tohumla 4F
   - eleştirmen `state()`
   - değer ölçeği

   Gerçek sinyalli koşu ancak yeni bir ön kayıttan sonra yapılır.
4. **Betimsel,** kararı değiştirmez:
   - tohum başına SR ve Δ
   - devir ve nakit ağırlığı
   - kolların tohum ortalamalı serisinin 1/N ve al-tut'a karşı ΔSR'si (aynı muhasebe, maliyetsiz)
   - RL'siz brüt referans (~1,5, M2 §9.1) ile büyüklük karşılaştırması
5. Hiçbir sonuç "RL piyasayı yener" ya da "tahmin katkı verir" cümlesine dönüştürülmez. Sinyal gelecekten türetilmiştir. M3 sayıları tez bulgusu değildir.

**Ertelenen (v0.1'de vardı):** 20 bps çifti ve tahminsiz kol → M3b. Maliyet altında değerin RL'siz referansı (0,08–0,30) bu pencerede MDE'nin altında kalabilir; ayrı ve küçük bir soru.

### 5.1 Bütçe (kullanıcı kararı: toplam ~1 saat)

- 6 koşu.
- Eğitim F kare: koşu başına ≤ **8 dakika** eğitim duvar süresi. Her koşuya ~1 dk değerlendirme eklenir.
- F = min(2.000.000, ⌊8 × 60 / saniye_kare⌋), 60.000'e (10 parti) aşağı yuvarlanır.
- **Alt sınır F_min = 600.000** (~100 parti ≈ 1000 Adam adımı; hakem-rl ikinci okuma #3, ölçümden ÖNCE sabitlendi). F < F_min çıkarsa M3a **koşulmaz**; süre kararı kullanıcıya döner. Tohum sayısı 3'ün altına inmez, S1 kontrolü korunur. `run_m3a.py` F < F_min'i reddeder.
- Saniye/kare: `train_m3.py --frames 30000 --train-device cuda --parallel` ölçümü. GPU başka işle meşgul değilken; `nvidia-smi` kontrolü, `run_m3a.py` de her koşudan önce bakar.
- **Ölçüm (2026-11-16, eğitim sonucu yok):** `measurements/phase10-m3/timing_gpu.json`
  - 30.000 kare 34,4 sn → 1,147 ms/kare (229,4 sn / 200 bin)
  - RTX 4060, paralel toplama, 10 ortam, MAPPO, parti 6000 × 10 epoch
  - Ölçüm sırasında GPU'da `cs2.exe` açıktı (%36 kullanım), yani süre kötümser olabilir.
- **Kural uygulaması:** 8 dk sınırıyla F = ⌊480 / 0,001147⌋ = 418.604 → 360.000. Bu, **F_min = 600.000'in altında → M3a bu bütçeyle koşulmaz;** süre kararı kullanıcıya döndü.
  - F_min için koşu başına ~11,5 dk eğitim gerekir; 6 koşu ~69 dk eğitim + ~10 dk değerlendirme.
- **F:** — **Ertelendi (kullanıcı kararı, 2026-11-16).** M3a GPU boşken F = 600.000 ile koşulacak: ~80 dk; koşu başına ~11,5 dk eğitim. Hazırlık tamam: `venv-marl\Scripts\python scripts/marl/run_m3a.py --frames 600000`. Başlatmadan önce toplam süre kullanıcıya yeniden sunulur.
- Toplam süre ön kayda yazılmadan önce kullanıcıya sunulur.
- Sıra: A1 s0..s2, sonra S1 s0..s2 (`run_m3a.py`). Yarıda kalırsa eksik kol okunmaz.

### 5.2 Sonuç (2026-11-18): GEÇTİ

**Kaynak:** `measurements/phase10-m3/m3a_results.json`; komut `python scripts/marl/analyze_m3.py --root results/marl_m3 --out docs/development/measurements/phase10-m3/m3a_results.json`. Aşağıdaki sayılar bu dosyadan yuvarlanmıştır.

**Koşu izi:** 6/6 koşu, F = 600.000, commit `b20b7f3`, temiz çalışma ağacı (`results/marl_m3/*/s*/train.json`). Koşu başına eğitim 455–495 sn; RTX 4060. S1 s1 eğitiminin son dakikalarında GPU'da `cs2` açıldı; süre 495 sn'ye çıktı, sonuca etkisi ayrıca ölçülmedi.

| Tohum | Eğitim SR A1 | Eğitim SR S1 | Eğitim Δ | Doğrulama SR A1 | Doğrulama SR S1 | Doğrulama Δ |
|---|---|---|---|---|---|---|
| 0 | 1,84 | 0,95 | +0,88 | 2,10 | 1,30 | +0,80 |
| 1 | 1,13 | 0,89 | +0,24 | 1,32 | 0,75 | +0,57 |
| 2 | 1,86 | 0,70 | +1,16 | 2,22 | 1,14 | +1,08 |

**Ön kayıt koşulları:**

| Koşul | Eşik | Ölçüm | Sonuç |
|---|---|---|---|
| (i) Eğitim büyüklüğü ve ortak HAC | D ≥ 0,3 ve tek yönlü p < 0,05 | D = 0,76 (SE 0,08), p ≈ 9e-24 | sağlandı |
| (ii) Eğitim sıralaması | min A1 > max S1 | 1,13 > 0,95 (kesin permütasyon p = 1/20) | sağlandı |
| (iii) Doğrulama işareti | D > 0 ve en az 2/3 tohumda Δ > 0 | D = 0,82, 3/3 tohum | sağlandı |

**Okuma (ön kayıt cümlesi):** Eğitim penceresinde, aynı kapasite ve parmak izine sahip kontrole karşı, maliyetsiz koşulda sinyal aktarımı gözlendi; doğrulamada işaret çelişmiyor. M3b ve M4 açılır.

**Betimsel (kararı değiştirmez):**
- Ortalama günlük devir: A1 0,66, S1 0,33 (eğitim); A1 0,67, S1 0,34 (doğrulama). Nakit ağırlığı iki kolda da ~%2–3.
- Tohum ortalamalı serinin 1/N'e karşı ΔSR'si: A1 +1,30 (eğitim, SE 0,11), +1,11 (doğrulama, SE 0,21). S1 +0,47 (eğitim, SE 0,12), +0,23 (doğrulama, SE 0,18; p = 0,20).
- RL'siz brüt referans: PC060-K12 E1 brüt +1,58 (M2 §9.1). A1'in 1/N'e karşı farkı bu büyüklüğün altında kalıyor.

**Sınırlar:**
- Sinyal gelecekten türetilmiştir ve koşu maliyetsizdir. Sonuç "RL piyasayı yener" ya da "tahmin katkı verir" anlamına gelmez.
- Ortak HAC p'si yalnız gün gürültüsünü ölçer. Tohum varyansına karşı kanıt (ii) kuralıdır ve 3 tohumla ulaşılabilecek en küçük p = 0,05'tir.
- Tohum 1'de eğitim Δ'sı 0,24; 0,3 eşiği tohum başına değil ortalamaya uygulanır.
- Doğrulama yalnız işaret kontrolüdür, genelleme kanıtı değildir.
- Sinyalsiz S1 eğitim penceresinde 1/N'in üstünde kaldı, doğrulamada fark belirsiz. Kaynağı (pencereye uyum ya da ezber) bu koşuda ayrıştırılmadı.
- A1'in devri S1'in iki katıdır; değerin maliyet altında kalıp kalmadığı M3b'nin sorusudur (§6).

---

## 6. ÖN KAYIT v1.1 — M3b maliyet kapısı ve M4 temel yapılandırması (2026-11-18, ilk M3b eğitiminden ÖNCE)

Kapsam ve toplam süre kullanıcıya sunuldu ve onaylandı (2026-11-18). M3a'nın okuma kuralı ve eşikleri değiştirilmedi. v1.0 (`2462853`) hiç koşulmadan hakem-rl okumasıyla v1.1'e güncellendi (§8).

### 6.1 Sorular

1. **Aşama 1:** MAPPO, 20 bps maliyetle yeniden eğitildiğinde bilinen sinyalden kontrole göre değer üretebiliyor mu?
2. **Aşama 2:** M4'ün temel yapılandırması ne olmalı: referans mı, tek etkeni değiştirilmiş bir aday mı?

Sinyal bilerek gelecekten türetilmiştir (§4). M3b sayıları **tez bulgusu değildir**; geliştirme kararı üretir.

### 6.2 Ortak yapılandırma

- §2'deki hiperparametreler, aşağıda değişen etken dışında aynen. lr 5e-4, F = 600.000 kare, son ağırlıklar; kontrol noktası seçimi ve erken durdurma yok.
- **Maliyet:** 20 bps gidiş-dönüş hem eğitimde hem değerlendirmede (`--cost-bps 20`).
- **Tohumlar:** s ∈ {0, 1, 2}, üreteç g_s ∈ {20261115, 1, 2} (§5 ile aynı eşleme).
- **Değerlendirme:** deterministik, tam pencere. Eğitim 1343 gün, doğrulama 500 gün; 2021+ yok.
- **Süre:** koşu başına ~8 dk (M3a'da eğitim 455–495 sn, RTX 4060). Aşama 1 ~50 dk, Aşama 2 ~1,6 saat, toplam ~2,4 saat.
- **Kod:** `scripts/marl/run_m3b.py`, `scripts/marl/analyze_m3b.py` (`--self-test` 16/16).

### 6.3 Aşama 1: maliyet kapısı (6 koşu)

| Kol | Gözlemdeki tahmin bloğu | Rol |
|---|---|---|
| A1c | PC060-K12 (g_s) | sinyalli, **referans yapılandırma** (MAPPO, eleştirmen = tüm yerel gözlemler, α = 0,25) |
| S1c | aynı blok, günleri permüte (§5) | kontrol |

SR yıllık, 20 bps sonrası X üzerinde. Δ_s = SR(A1c_s) − SR(S1c_s), D = ort_s Δ_s.

- **GEÇTİ = üçü birden:**
  - (i) Eğitim: D ≥ **0,3** ve ortak HAC tek yönlü p < 0,05.
  - (ii) Eğitim: min_s SR(A1c_s) > max_s SR(S1c_s).
  - (iii) Doğrulama, işaret kontrolü: D > 0 ve en az 2/3 tohumda Δ_s > 0.

  Okuma: "Eğitim penceresinde, aynı kapasite ve parmak izine sahip kontrole karşı, 20 bps ile yeniden eğitilen politikada sinyal aktarımı gözlendi; doğrulamada işaret çelişmiyor."
- **KISMİ:** eğitimde D > 0 ve p < 0,05 var, ama GEÇTİ koşullarından biri eksik.
- **GÖZLENMEDİ:** "Bu bütçe, lr ve 20 bps'de maliyet sonrası aktarım gözlenmedi." Kanıt yokluğudur; "aktaramıyor" denmez.
- **Durma kuralı:** GEÇTİ dışındaki her etikette **Aşama 2 koşulmaz** (`run_m3b.py` reddeder), M4'e geçilip geçilmeyeceği kullanıcı kararına döner. M4'e yine de geçilirse temel referanstır (MAPPO, tüm yerel gözlemler, α = 0,25); bu seçim kanıta dayanmaz, varsayılandır.

### 6.4 Aşama 2: temel yapılandırma (12 koşu; yalnız Aşama 1 GEÇTİ ise)

Her aday referanstan (A1c) **tek yapılandırma seçimiyle** ayrılır; sinyal, maliyet ve tohumlar aynıdır. IPPO ve STATE'te bu seçim eleştirmenin bilgi kümesini ve **kapasitesini birlikte** değiştirir, kapasite eşlenmez. Sonuç "merkezileştirmenin etkisi" diye değil, "bu eleştirmen yapılandırmasının etkisi" diye okunur. Eleştirmen boyutu her koşunun `train.json` → `critic` alanına yazılır.

| Aday | Değişen seçim | Eleştirmen girişi / parametre | Seçilebilir mi |
|---|---|---|---|
| Referans (A1c) | — | 1440 / 434.945 | — |
| IPPO | Algoritma: ajan başına yerel eleştirmen (plan §2.3, §3.2.4) | 48 / 78.593 | evet |
| STATE | Eleştirmen girdisi: ortam `state()` (R5) | 43 / 77.313 | evet |
| A05 | α = 0,5 (M2 §2.2) | referansla aynı | evet |
| A00 | α = 0 (M2 §2.2) | referansla aynı | **hayır:** yalnız ablasyon. Plan §2.3 α > 0 seçimini şart koşar |

D_j = ort_s [SR(aday_j,s) − SR(A1c_s)].

**Aday UYGUN = beşi birden:**
- (a) Doğrulama: D_j ≥ **0,3**.
- (b) Doğrulama: ortak HAC tek yönlü p, seçilebilir 3 aday üzerinde **Holm** düzeltmesiyle < 0,05.
- (c) Doğrulama: **3/3 tohumda** fark > 0.
- (d) Eğitim: D_j > 0 (işaret tutarlılığı).
- (e) Doğrulama: D_j > referansın tohumlar arası SR aralığı (max_s − min_s SR(A1c_s)). (b) yalnız gün gürültüsünü ölçtüğü için tohum gürültüsüne karşı koşul budur.

**Seçim:** Uygun adaylar içinde doğrulama D_j'si en büyük olan seçilir. Uygun aday yoksa **referans kalır**. Etkenler birleştirilmez; seçilen yapılandırma her zaman fiilen koşulmuş bir yapılandırmadır.

M3b'nin seçimi yalnız M4'ün temelini belirler. Plan §2.3 gereği IPPO ve MAPPO nihai raporda ikisi birden yer alır.

**Betimsel, kararı değiştirmez:** tohum başına SR ve fark, devir, nakit ağırlığı, A00 ablasyonu, aynı maliyetli 1/N ve al-tut SR'si.

### 6.5 §6'nın önceki taslağından sapmalar (kullanıcı kararı, 2026-11-18)

| Madde | Karar | Gerekçe |
|---|---|---|
| Tahminsiz kol (A0) | M5'e | Sinyal sorusunu aynı kapasiteli S1c yanıtlıyor; tahminin katkısı gerçek sinyalle M5'in sorusu |
| Yerel gözleme piyasa bağlamı (R4) | Ayrı paket | `marl-obs-v1` gözlem sürümünü ve canlı karar paritesini (K2) değiştirir |
| Tek ajan ve SARL-ağırlık baseline'ları | M8'den önce ayrı paket | Kodları yok; birincil MAPPO ↔ SARL-eşmimari karşıtlığı M8 ön kaydında (plan §2.3) |

### 6.6 Beyanlar

- **Ön kayıttan önce görülen ölçüm.** M3a'nın maliyetsiz eğitilmiş politikalarına 20 bps sonradan uygulandı (`measurements/phase10-m3/m3a_expost_cost.json`, `scripts/marl/expost_cost_m3a.py`): net D eğitimde 0,41, doğrulamada 0,52; tohum 1'in eğitim farkı 0,07. Bu ölçüm yeniden eğitim sonucu değildir. Eşikler buna göre ayarlanmadı, M3a'dan aynen alındı. Net fark eşiğe yakın ve tohum 1'in farkı küçük olduğu için Aşama 1'in GEÇTİ dışında bir etiket vermesi kurala uygun, olası bir sonuçtur.
- **Doğrulama penceresi yeniden kullanılıyor.** Pencere M3a'da maliyetsiz okundu ve Aşama 2'de seçim için kullanılıyor. Seçilen adayın doğrulama SR'si seçim nedeniyle iyimserdir ve başarım sayısı olarak raporlanmaz.
- **Güç.** 3 tohumla sıralama kuralının ulaşabileceği en küçük p 1/20'dir. Ortak HAC ve Holm yalnız gün gürültüsünü denetler, tohum varyansını içermez. Aşama 2 kuralı bilerek tutucudur: gerçek ama küçük bir üstünlük seçilmeyebilir, bu durumda referans kalır.
- **Tek seçim tasarımı.** Etkileşimler ölçülmez.
- **Aktarılabilirlik.** Yapılandırma sentetik PC060-K12 sinyaliyle seçilir. Gerçek `pit_v2` sinyali kapıyı geçmedi (0/30, havuzlu AUC 0,5044; plan §3.1.3); seçimin gerçek sinyale taşınacağı varsayılmaz.
- **Süren sapmalar.** Değer normalizasyonu yok (§2). GPU'da bit-eşlik iddia edilmez (§1).

### 6.7 Aşama 1 sonucu (2026-11-19): KISMİ, Aşama 2 koşulmadı

**Kaynak:** `measurements/phase10-m3/m3b_stage1.json`; komut `python scripts/marl/analyze_m3b.py --stage 1 --out docs/development/measurements/phase10-m3/m3b_stage1.json`. Aşağıdaki sayılar bu dosyadan yuvarlanmıştır.

**Koşu izi:** 6/6 koşu, F = 600.000, 20 bps, commit `2f43f62`, temiz çalışma ağacı (`results/marl_m3b/*/s*/train.json`). Koşu başına eğitim 454–473 sn.

| Tohum | Eğitim SR A1c | Eğitim SR S1c | Eğitim Δ | Doğrulama SR A1c | Doğrulama SR S1c | Doğrulama Δ |
|---|---|---|---|---|---|---|
| 0 | 1,43 | 0,78 | +0,65 | 1,64 | 0,82 | +0,82 |
| 1 | 0,79 | 0,70 | +0,08 | 0,78 | 0,73 | +0,05 |
| 2 | 0,76 | 0,30 | +0,46 | 1,34 | 0,75 | +0,58 |

| Koşul | Eşik | Ölçüm | Sonuç |
|---|---|---|---|
| (i) Eğitim büyüklüğü ve ortak HAC | D ≥ 0,3 ve tek yönlü p < 0,05 | D = 0,40 (SE 0,09), p ≈ 1e-5 | sağlandı |
| (ii) Eğitim sıralaması | min A1c > max S1c | 0,76 > 0,78 değil | **sağlanmadı** |
| (iii) Doğrulama işareti | D > 0 ve en az 2/3 tohumda Δ > 0 | D = 0,49 (SE 0,12), 3/3 tohum | sağlandı |

**Okuma (ön kayıt):** Etiket KISMİ. 20 bps ile yeniden eğitilen politikada eğitim penceresinde anlamlı pozitif fark var, ama tohum varyansına karşı konan sıralama koşulu sağlanmadı. §6.3 durma kuralı gereği **Aşama 2 koşulmadı** ve M4 temel yapılandırması için kanıt üretilmedi. Eşikler sonuçtan sonra değiştirilmedi; (ii)'deki farkın küçük olması etiketi değiştirmez.

**Betimsel (kararı değiştirmez):**
- Ortalama günlük devir: A1c 0,48, S1c 0,31 (eğitim); A1c 0,50, S1c 0,31 (doğrulama). M3a'nın maliyetsiz A1'inde eğitim devri 0,66 idi.
- Aynı maliyetli baseline SR'leri: 1/N 0,38 (eğitim), 0,83 (doğrulama); al-tut 0,66 (eğitim), 0,81 (doğrulama).
- Tohum 1'de fark iki pencerede de küçük (+0,08 ve +0,05); bu tohumda A1c'nin doğrulama SR'si (0,78) 1/N'in altında.

**Sınırlar:** Sinyal gelecekten türetilmiştir; sonuç tez bulgusu değildir. Ortak HAC p'si tohum varyansını içermez. Aynı kapı ek tohumlarla yeniden denenmez; yeni bir deney ayrı ön kayıt gerektirir.

---

## 7. Testler

| Test | Ortam | Kontrol | Kapsam |
|---|---|---|---|
| `venv-marl\Scripts\python tests/test_marl_m3.py` | `venv-marl/` | 18 | görev, bölüm sonu + GAE bootstrap, MAPPO anahtarları, kısa MAPPO (iki eleştirmen), tohum bit-eş, kaydet/yükle, deterministik eval, aynı muhasebeyle baseline, R7 |

## 8. Hakem incelemesi

**hakem-rl, M3a ön kaydı v0.1 (2026-11-16, uç `331730e`).** Paket ~1,3 bin kelime; Python koşmadı. ~28 bin token / 6 araç / ~0,9 dk. **Küçük revizyon.**

| # | Cid. | Bulgu | Sonuç (v0.2) |
|---|---|---|---|
| 1 | BÜYÜK | Ortak HAC tohum varyansını SE'ye katmıyor | Kural (ii): min A1 > max S1 (3'e 3 kesin permütasyon, p = 1/20) |
| 2 | BÜYÜK | Eğitim penceresinde ezber ile sinyal ayrılamıyor; tahminsiz kol boyutu eşitlemiyor | Kontrol = karıştırılmış PC (S1); A0 M3b'ye |
| 3 | BÜYÜK | "Aktarıyor" cümlesi genelleme ister | Kural (iii): doğrulamada D > 0 ve ≥ 2/3 tohum |
| 4 | BÜYÜK | lr 5e-5 ile KALDI hiperparametre artefaktı olabilir; "aktaramıyor" cümlesi yanlış | lr 5e-4 önceden sabit; GÖZLENMEDİ etiketi (kanıt yokluğu) |
| 5 | KÜÇÜK | D > 0 zayıf eşik | Kural (i): D ≥ 0,3; KISMİ etiketi |
| 6 | KÜÇÜK | Kontrol noktası politikası yazılı değil | Son ağırlıklar, seçim yok (§5) |
| 7 | KÜÇÜK | 20 bps çiftinin beklenen sonucu boş | Çift M3b'ye ertelendi (kullanıcının bütçe kararı) |

**İkinci okuma** (SendMessage, yalnız v0.1 → v0.2 farkı; ~40 bin token önceki bağlamla / 2 araç / ~0,4 dk): Küçük revizyon, 7/7 karşılandı. Açık kalan 3 madde v0.2'de işlendi: (ii) tanımsal sıralama kuralı ve p = 0,05 eşitliği; (iii) "işaret kontrolü" etiketi, GEÇTİ cümlesi daraltıldı; **F_min = 600.000** (BÜYÜK: adım tabanı yoksa GÖZLENMEDİ yine lr/bütçe artefaktı olabilir). Hakem: "3 tohumla tutucu bir kapı olarak savunulabilir; GÖZLENMEDİ zayıf bilgi taşır."

Hakem 25 koşu önermişti (5 tohum). Kullanıcının ~1 saatlik sınırı yüzünden 3 tohum ile #1'in kesin permütasyon ölçütü kullanıldı; 3'e 3'te en küçük p tam 0,05. Bu bir **güç sınırlamasıdır**, beyan edilir. İkinci okuma: SendMessage ile, yalnız v0.1 → v0.2 farkı.

**hakem-rl, M3b ön kaydı v1.0 (2026-11-18, uç `2462853`).** ~51 bin token / 17 araç / ~1,3 dk. **Küçük revizyon.**

| # | Cid. | Bulgu | Sonuç (v1.1) |
|---|---|---|---|
| 1 | BÜYÜK | Aşama 2 kuralı tohum gürültüsüne karşı kalibresiz; boş hipotezde "≥ 2/3 tohum" olasılığı 0,5 | (c) 3/3 tohum; yeni (e) D_j > referansın tohumlar arası SR aralığı |
| 2 | KÜÇÜK | IPPO ve STATE'te eleştirmen kapasitesi de değişiyor; "tek etken" fazla güçlü | Metin düzeltildi; eleştirmen boyutu `train.json`'a yazılıyor |
| 3 | KÜÇÜK | Aşama 1 geçmezse M4 temeli yazılı değil | §6.3: referans, kanıta dayanmayan varsayılan |
| 4 | KÜÇÜK | Öz test koşulları tek tek eletmiyor | Beş koşul için tek tek eleme testi, 16/16 |
| 5 | KÜÇÜK | Aşama 2 kapısı JSON'un ön kaydını denetlemiyor | `preregistration` ve `stage` denetimi eklendi |

**İkinci okuma** (SendMessage, yalnız `2462853..3dff3a5` farkı): **Kabul**, 5/5 karşılandı, yeni BÜYÜK bulgu yok. Hakemin notu: (e) koşulundaki 3 tohumluk aralık kendisi gürültülü bir tahmindir; (a) ve (c) ile birlikte uygulandığı için kabul edilebilir.

## Versiyon geçmişi

- **v0.6 (2026-11-19):** M3b Aşama 1 koşuldu (6/6 koşu); etiket KISMİ, (ii) sıralama koşulu sağlanmadı (§6.7). Aşama 2 koşulmadı. Ön kayıt metni ve eşikleri değişmedi.

- **v0.5 (2026-11-18):** M3b ön kaydı v1.1: hakem-rl bulguları işlendi (§8). Koşu yok.

- **v0.4 (2026-11-18):** M3b ön kaydı v1.0 (§6): Aşama 1 20 bps'de A1c ↔ S1c, Aşama 2 referansa karşı IPPO, STATE, A05 ve A00 ablasyonu. A0, R4 ve SARL baseline'ları §6.5'teki kararla kapsam dışına alındı. Koşu yok.

- **v0.3 (2026-11-18):** M3a koşuldu (F = 600.000, 6/6 koşu); etiket GEÇTİ (§5.2). Ön kayıt metni ve eşikleri değişmedi.

- **v0.2 (2026-11-16):** Ön kayıt eğitimden önce yeniden yazıldı (§5): A1 ↔ karıştırılmış PC (S1), 3 tohum, maliyetsiz, lr 5e-4. GEÇTİ = (i) D ≥ 0,3 ve p < 0,05, (ii) min A1 > max S1, (iii) doğrulamada işaret. Bütçe toplam ~1 saat (kullanıcı kararı); 20 bps çifti ve tahminsiz kol M3b'ye. Hakem-rl tablosu §8.

- **v0.1 (2026-11-16):** İlk sürüm; M3a ön kaydı (§5). Bütçe (§5.1) süre ölçümünü bekliyor. GPU ölçümü 2026-11-16'da yapılamadı: GPU başka bir uygulamayla %100 doluydu, koşu durduruldu.
