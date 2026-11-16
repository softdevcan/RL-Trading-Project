# Faz 10 M3 — Algoritma katmanı: tasarım ve M3a ön kaydı

**Sürüm:** v0.1 · 2026-11-16 · dal `feat/phase10-m3` (taban: M2, PR #34)
**Üst belgeler:**
- [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.2, §4
- Faz 9 planı §M3 (IPPO, MAPPO, parametre paylaşımı + ajan kimliği, MAPPO önerileri)
- [phase-10-m2-design.md](phase-10-m2-design.md): §9 #4, #10, #11 ve §9.1 (pozitif kontrol PC060-K12)

**Durum:** kod ve testler hazır. **M3a ön kaydı bu belgeyle commit'lenir, ilk eğitim koşusundan ÖNCE.** Tek açık alan eğitim bütçesi (kare sayısı, §5). Bütçe süre ölçümünden sonra, hiçbir eğitim sonucu görülmeden ayrı bir commit'le doldurulur. Performans sayısı yok.

Bu belge M3'ün yapısını tanımlar. Kod ile belge farklıysa kod hatalıdır.

---

## 0. Kapsam

| Madde | Burada | Kod |
|---|---|---|
| BenchMARL görevi (M2 §9 #4) | §1 | `marl/benchmarl_task.py` |
| Bölüm sonu: truncation + bootstrap (M2 §9 #10, hakem-rl R6) | §1 | `test_marl_m3` [2] |
| Başlangıç gürültüsü (R7), deterministik değerlendirme | §2 | `marl/m3_runner.py`, `test_marl_m3` [4] |
| Baseline'lar aynı muhasebeyle (M2 §9 #11) | §3 | `m3_runner.rule_series` |
| Pozitif kontrol PC060-K12 (M2 §9.1) | §4 | `scripts/marl/build_pred_m3.py` |
| **M3a ön kaydı: boru hattı kapısı** | §5 | `train_m3.py`, `eval_m3.py`, `analyze_m3.py` (ön kayıttan sonra) |
| M3b: algoritma ve eleştirmen seçimleri (IPPO ↔ MAPPO, R4, R5) | §6 | ayrı ön kayıt, M3a geçince |

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
| lr, γ, λ, grad kırpma | 5e-5, 0,99, 0,9, 5 | BenchMARL 1.4 varsayılanları, değiştirilmedi |
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

## 5. ÖN KAYIT — M3a boru hattı kapısı (2026-11-16, ilk eğitim koşusundan ÖNCE)

**Soru:** Yeni MARL boru hattında MAPPO, gözlemine verilen bilinen bir sinyali (PC060-K12) eyleme dönüştürüp değer üretebiliyor mu?

**Kollar** (her biri 5 eğitim tohumu, s ∈ {0, 1, 2, 3, 4}; tohum s'in PC kolu üreteç tohumu g_s ∈ {20261115, 1, 2, 3, 4} ile eşlenir):

| Kol | Tahmin | Maliyet (eğitim ve değerlendirme) | Rol |
|---|---|---|---|
| A0 | yok | 0 bps | birincil karşılaştırma |
| A1 | PC060-K12 (g_s) | 0 bps | birincil |
| B0 | yok | 20 bps | ikincil |
| B1 | PC060-K12 (g_s) | 20 bps | ikincil |

Kare bütçesi F her koşuda aynıdır (§5.1). Diğer her şey §2'deki gibi.

**Birincil test.**
- Eğitim penceresinde (1343 gün) deterministik değerlendirme.
- D = (1/5) Σ_s [SR(A1_s) − SR(A0_s)]; X üzerinde, yıllık.
- Ortak HAC (`sharpe_diff_hac_pooled`, m0p-v2 birleştirme kuralı), tek yönlü.

**Pencere seçimi.** Pozitif kontrolün sorusu öğrenilebilirlik olduğu için birincil pencere eğitim penceresidir. İki kolun kapasitesi ve verisi aynı olduğundan, tarihleri ezberleme etkisi farkta birbirini götürür; fark yalnız sinyalden gelir. Bu tercih **beyan edilir**. Doğrulama penceresi ikincildir.

**Okuma kuralı (önceden):**
1. **D > 0 ve tek yönlü p < 0,05 → GEÇTİ:** "Boru hattı bilinen bir sinyali maliyetsiz koşulda eyleme aktarıyor." M3b ve M4 açılır.
2. **Aksi hâlde → KALDI:** "Bu bütçe ve yapıda boru hattı bilinen sinyali aktaramıyor." Gerçek sinyalli (P5) hiçbir MARL koşusu yapılmaz. Keşifsel tanı sırası, ön kayıtsız ve tez dışı:
   - tek tohumla 4F kare
   - lr 5e-4
   - eleştirmen `state()`

   Sonra yeni bir ön kayıt yazılır.
3. **İkincil ve betimsel,** kapı kararını değiştirmez:
   - Doğrulama penceresinde aynı D.
   - 20 bps çiftinde (B1 − B0) aynı D, eğitim ve doğrulama.
   - Yakalama oranı = D(A1 − A0) / RL'siz brüt ΔSR. Referans S_yön − 1/N olduğu için oran yalnız büyüklük fikri verir.
   - Tohum başına ΔSR dağılımı.
   - Her kolun 1/N, al-tut ve nakde karşı ΔSR'si.
   - Devir ve nakit ağırlığı.
4. Hiçbir sonuç "RL piyasayı yener" ya da "tahmin katkı verir" cümlesine dönüştürülmez. Sinyal gelecekten türetilmiştir. M3 sayıları tezin bulgusu değildir (Faz 10 §4 S1 ruhu).

**Güç (önceden, kaba).**
- RL'siz brüt değer ~1,5 olduğundan birincil test için güç sorunu beklenmez.
- 20 bps çifti için RL'siz değer (0,08–0,30) testin MDE'sinin (~0,3) altında kalabilir; bu çiftin boş çıkması beklenen sonuçlardan biridir.
- RL ↔ RL farkının SE'si ölçülmeden bilinmez; sonuçta raporlanır.

### 5.1 Bütçe (AÇIK — süre ölçümünden sonra, eğitim sonucu görülmeden doldurulur)

- **Kural:** M3a toplamı 20 koşu.
- **Hedef:** toplam GPU duvar süresi ≤ ~8 saat ve koşu başına F ≤ 2.000.000 kare.
- F = min(2.000.000, ⌊8 sa × 3600 / 20 / (saniye/kare)⌋), en yakın 60.000'e (10 parti) aşağı yuvarlanır.
- Saniye/kare değeri: `train_m3.py --frames 30000 --train-device cuda --parallel` ölçümü. GPU başka bir işle meşgul olmamalı; `nvidia-smi` ile kontrol edilir.
- **Ölçüm:** — *(doldurulacak)*
- **F:** — *(doldurulacak)*
- Koşu sırası: A0, A1, B0, B1, her biri tohum 0..4. Yarıda kalırsa tamamlanan kollar raporlanır; eksik kol okunmaz.

---

## 6. M3b (M3a geçerse; ayrı ön kayıt)

M3a geçerse açılır, ayrı bir ön kayıtla koşulur.

- **Seçimler:**
  - IPPO ↔ MAPPO
  - eleştirmen: gözlemler ↔ `state()` (R5)
  - yerel gözleme piyasa bağlamı: kesitsel ortalama z, geçerli oran (R4)
  - α ∈ {0; 0,25; 0,5} ön elemesi (M2 §2.2)
- **Seçim kuralı ve pencere:** doğrulama penceresi; hepsi M3b ön kaydında.
- **Zorunlu öğrenen baseline'lar** (Faz 9 §M3): tek ajan ve SARL-ağırlık kolu (§3.2.4) aynı ortam ve sınırlarla.

---

## 7. Testler

| Test | Ortam | Kontrol | Kapsam |
|---|---|---|---|
| `venv-marl\Scripts\python tests/test_marl_m3.py` | `venv-marl/` | 18 | görev, bölüm sonu + GAE bootstrap, MAPPO anahtarları, kısa MAPPO (iki eleştirmen), tohum bit-eş, kaydet/yükle, deterministik eval, aynı muhasebeyle baseline, R7 |

## Versiyon geçmişi

- **v0.1 (2026-11-16):** İlk sürüm; M3a ön kaydı (§5). Bütçe (§5.1) süre ölçümünü bekliyor. GPU ölçümü 2026-11-16'da yapılamadı: GPU başka bir uygulamayla %100 doluydu, koşu durduruldu.
