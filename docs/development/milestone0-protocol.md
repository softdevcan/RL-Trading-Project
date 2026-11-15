# Milestone 0' protokolü (m0p-v1)

**Durum: TASLAK r2, 2026-11-15.** Kapanış kapısının (hakem-2) Büyük revizyonu işlendi (§12). Kullanıcı onayı ve hakem-2 ikinci okumasıyla dondurulur. Bağlayıcı kuralların tümü [milestone0/protocol-v1.json](milestone0/protocol-v1.json)'da. Koşum betiği JSON'un ve bu belgenin SHA-256'sını deftere bağlar; biri değişirse koşmayı reddeder ([scripts/run_milestone0.py](../../scripts/run_milestone0.py)). Değişiklik yeni bir sürümdür (m0p-v2).

**Dayanak:** Faz 9 planı §4 "Milestone 0'", §5 M8 ve §9.22–§9.34 · Faz 10 planı §6 (K4, K7, K17, K20, K25, K26) · [vision-and-scope.md](../thesis/vision-and-scope.md) §7 Milestone 0.

## 1. Amaç ve yeri

Milestone 0' eski tek ajanlı ortamda (`env/trading_env.py`) Faz 9 altyapısıyla kurulan **baseline**'dır. Ürettiği şeyler:
- Ansari vd. (2024) uyarlamasının BIST-30 replikasyonu: **sınırlılıkları beyanlı** sadık replikasyon (K26 = a).
- Vision §7 ablasyonu: **DT ↔ DTF** (tahminsiz ↔ nokta-zamanlı tahminli) ve bunun pozitif kontrolü.
- K25 fiyat ölçeği kıyası (`stats` ↔ `causal`), K4 Sharpe terimi kıyası, M8 seçim kuralı etkeni (argmax ↔ son nokta).

**Tezin birincil kanıtı değildir.** Birincil karşıtlık yeni ortamda: MAPPO ↔ SARL-eşmimari (Faz 10 §2.3, K15). H1–H4 MARL hipotezleri (M8) bu protokolün konusu değil.

## 2. Dondurulan girdiler (JSON `prediction_levels`, `expected`, `hyperparameters`)

| Girdi | Değer | Denetim |
|---|---|---|
| Evren | Panelin 30 hissesi, dondurulmuş, hayatta kalan (K7). Tarihsel üyelik maskesi kullanılmaz | E1 beyanı §9.32 |
| Bölme | `wf-v1`, özet `33a196c602976de1`, 6 kat (test 2021..2026), doğrulama bir önceki yıl, purge 0 | `test_rl_walk_forward` |
| Panel | Rotanın eğitim paneli, SHA-256 `008e3b11…9a89` (iki fiyat ölçeğinde aynı) | Betik koşumdan önce ve her hücreden sonra karşılaştırır; veri tazelenirse durur (yeni sürüm) |
| P tablosu (DTF) | `pit_v1_seed42`, SHA `747bd39a…eae9`, eşleme P4 v1 | Dosya SHA'sı koşumdan önce, kayıttaki SHA her hücreden sonra |
| Pozitif kontrol | `pc_auc055` (ölçülen AUC 0,5519), `pc_auc060` (0,6019) | Aynı |
| Hiperparametreler | Rota sabitleri, algoritma başına JSON'da (PPO `ent_coef` 0,15; A2C `vf_coef` 0,25 + RMSprop; TD3 hedef gürültüsü 0,3, eylem gürültüsü σ 0,2). **SB3 varsayılanları değil.** 2025-11-16'da (d3d2fce) R1 öncesi açık pencerelerle seçildi; bir araştırmacı serbestlik derecesi, beyan (kullanıcı kararı 2026-11-15: mevcut değerler) | Kayıtta `config.effective_hyperparameters`; betik her hücrede karşılaştırır |
| Cihaz | CPU (C1). GPU görülürse betik koşmaz | `device_guard`, defter |
| Kod | Dondurma commit'i araştırma etiketi `m0-protocol-v1` alır | [versioning.md](versioning.md) |

## 3. Etkenler (ön eleme: 3 × 3 × 2 × 2 = 36 yapılandırma)

| Etken | Seviyeler | Neden (plan) |
|---|---|---|
| Algoritma | PPO, A2C, TD3 | Faz 9 §4 Milestone 0' tanımı |
| Ödül | `simple` (faz 1) · `psr_ema_level` (faz 2, w₂ 0,30) · `psr_differential` (faz 2, w₂ 0,0496) | Ansari uyarlaması; K4 |
| Fiyat ölçeği | `stats` (B9) · `causal` (W = 252) | K25 = (b) |
| Tahmin | DT · DTF (P4) | Vision §7; §9.32 |

- **`simple` ↔ `psr` kıyaslanmaz.** Durum uzayı farklı: faz 2 temel oran ve makro gözlemi ekler, temel oranlar nokta-zamanlı değil (S4). Ödül etkeninin kıyası yalnız `psr_ema_level` ↔ `psr_differential` arasındadır.
- **K4 ölçek eşitlemesi** (kullanıcı kararı 2026-11-15; M8): `differential` kolunda w₂ = 0,30 · k.
  - k = σ(ema_level terimi) / σ(differential terimi) = **0,1652**.
  - Ölçüm: eşit ağırlıklı tam yatırımlı referans, 2013-06..2019-12. Bu aralık her doğrulama ve test penceresinden önce biter.
  - Kaynak: [k4_scale.json](measurements/phase9-m0/k4_scale.json). Tek seviye; N artmaz. Kıyas bu yüzden terim kıyasıdır, ödül paketi kıyası değil.

**Sabitler:**
- 200.000 adım (vision §7).
- Başlangıç bakiyesi 1.000.000 TL.
- Komisyon 0,001 tek yön (gidiş-dönüş 20 bps).
- En fazla 100 pay/işlem; `missing_data_mode='own_last'`.
- 10 doğrulama kontrol noktası (R1).

## 4. Seçim kuralı ve iki okuma (M8, R1)

- Her eğitim 10 kontrol noktasında doğrulamada ölçülür: fazla getiri Sharpe'ı (A8).
- **Aynı eğitimden iki test kaydı:** `argmax_validation` ve `last_checkpoint`. İkisi de N'ye sayılır ve `paired_trial_id` ile bağlıdır. `same_policy=True` ise getiriler aynıdır.
- `picks_agree` ve negatif fazla getiri rejimindeki seçim davranışı (§9.12) raporlanır.

## 5. Aşamalar ve budama

**Ön eleme:**
- 36 yapılandırma × **4 kat (2023–2026)** × 3 tohum = **432 eğitim**.
- Doğrulama yılları 2022–2025. **2021 ne testte ne doğrulamada** (hakem-2 #2: 2022 katının doğrulaması 2021 olurdu).
- **Puan:** son kontrol noktasının doğrulama fazla getiri Sharpe'ı, kat × tohum medyanı. Argmax puanı 10 noktanın maksimumu olduğu için yukarı yanlı ve oynak algoritmayı kayırır; ayrıca raporlanır ama budamaya girmez.
- Test metrikleri seçime girmez.

**Budama (KARAR 1, hakem-2 destekledi):**
- Algoritma başına puan, yapılandırmalarının puan medyanı. En yüksek **tek** algoritma nihaiye geçer; eşitlikte PPO.
- Ödül, fiyat ölçeği ve tahmin etkenleri tümüyle taşınır.
- Çıktı mekanik olarak `protocol-v1-final.json`'a yazılır ve ayrıca SHA'lanır.

**Nihai:**
- 1 × 3 × 2 × 2 = 12 yapılandırma × 6 kat × 10 tohum (1001–1010) = **720 eğitim**.

**Maliyet duyarlılığı (hakem-rl İ3):**
- DT ve DTF'nin nihai doğrulamada en iyi yapılandırması, 0 ve 50 bps ile yeniden eğitilir: 2 × 2 × 6 × 10 = **240 eğitim**.
- Ex post 0–50 bps ızgarası (D4) yalnız raporlama içindir.

**Pozitif kontrol (kullanıcı kararı 2026-11-15; hakem-2 #7 / soru 2):**
- DT'nin nihai doğrulamada en iyi yapılandırması, tahmin seviyesi PC055 ve PC060: 2 × 6 × 10 = **120 eğitim**.
- Sinyal **bilerek gelecekten** (gerçekleşen yön) türetildi. Aynı P4 eşlemesi ve gerçek tablonun ölçeği kullanılıyor (`positive_control_table`, `test_pit_table` 33).
- Doğrulayıcı aileye girmez. Soruyu ayırır: "sinyal yok" mu, "RL bilgili bir sinyali bu eşlemeyle kullanamıyor" mu?

## 6. Çıkarım

- **Birincil metrik (M8):** A8 risksiz faiz üstü, maliyet sonrası, nominal TL günlük getiriden yıllık Sharpe. İkincil: Sortino, Calmar, MDD, devir. Reel (TÜFE) ve USD birimlerinde aylık Sharpe (D3).
- **Çıkarım birimi (K20):** yapılandırma × tohum başına katların örneklem dışı günlük getirilerinin birleştirilmiş serisi.
  - Tohumlar piyasa gürültüsüne karşı bağımsız gözlem eklemez, çünkü günler aynıdır. Tohum ortalaması yalnız politika gürültüsünü azaltır.

**Doğrulayıcı aile** (hakem-2 #5; JSON `inference.confirmatory_family`):
1. **DT ↔ DTF:** A3 HAC Sharpe farkı, iki taraflı. Ayrıca eşdeğerlik TOST'u; sınır, ölçülen eşleşik korelasyondaki MDE.
2. **`stats` ↔ `causal`:** A3 HAC Sharpe farkı, iki taraflı.
- α = 0,05, aile içinde BH-FDR. **Pencere:** 2021 katı ve ön kayıt sonrası ileri veri.
- Geri kalan her kıyas **keşifseldir**: K4 terimi, seçim kuralı, algoritma, maliyet, rejim, pozitif kontrol.

**Güç (MDE)** ([mde.json](measurements/phase9-m0/mde.json); A3 testiyle Monte Carlo, %80 güç, α 0,05 iki taraflı, iid normal, günlük σ %1,5):

| Pencere | ρ 0,5 | ρ 0,8 | ρ 0,9 | ρ 0,95 |
|---|---|---|---|---|
| 2021 (248 seans) | 2,9 | 1,8 | 1,4 | 1,0 |
| 2023–2026 (~1000) | 1,5 | 0,9 | 0,7 | 0,5 |
| 6 kat (~1460) | 1,2 | 0,8 | 0,6 | 0,4 |

ρ, eşleşik kolların günlük getiri korelasyonudur; ölçülmedi, sonuçlarla raporlanır. Değerler yıllık Sharpe farkıdır.

**Bu yüzden:**
- 2021 tek başına ancak yıllık Sharpe farkı ≳ 1 olan bir etkiyi yakalayabilir. **"2021, bir duman okuması dışında okunmamış, düşük güçlü pencere"** olarak etiketlenir.
- Bu pencerede boş bir sonuç "etki yok" değil, "bu güçle tespit edilmedi (MDE = x)" diye yazılır.
- TOST sınırı aynı MDE ile kurulduğu için eşdeğerlik ancak dar bir GA'yla iddia edilebilir.

**Diğer çıkarım kuralları:**
- **DSR'nin N'i:** `load_trials(exclude_tags=('spike',))`. `protocol_open` kayıtlar dahildir; eski ortamın her kaydı açıktır ve varsayılan yol bunları dışlardı (hakem-2 #11). Duman kayıtları `spike` ile dışlanır.
- **Rejim (B6):** birincil USD bazlı XU100 genişleyen tercil; duyarlılık kolları nominal ve 5 yıllık kayan. Blok bootstrap; `episodes_min20`.
- **Baseline (D):** 1/N, al-tut, XU030. Evren hayatta kalan olduğu için **üstünlük çıkarımı yapılmaz**; bu, aynı evrende "RL > 1/N" için de geçerlidir (hakem-2 soru 3). XU030 "kıyas değil" dipnotuyla verilir.
- **Ödül kollarında ek tablo:** işlem sayısı, devir hızı, komisyon payı (komisyon ödülde iki kez sayılıyor).

## 7. Önceden yazılan beklentiler (hipotez değil)

- **DT ≈ DTF beklenir.** Geliştirme penceresinde (2013-06..2019) yön bilgisi tespit edilmedi (blok içi AUC 0,501, §9.31). Açık pencerede (2022-07-18..2026-08-28, 30.835 satır) de yok: havuzlu AUC 0,497, blok içi 0,502 (çapa kümesiyle iki taraflı p 0,45), bilgi kapısı 0/30 ([gate_v2_exposed.json](measurements/phase9-p/gate_v2_exposed.json)). 2021 ölçülmedi.
  - DTF yön özelliği ikili ve tam ölçeklidir; bilgisiz bir p'den 30 boyut gürültü ekler.
  - Ablasyon bu yüzden pratikte bir **gürültü girdisi ablasyonudur**. Pozitif kontrol bunu ayırır.
- Pozitif kontrolde DT'ye göre fark beklenir; büyüklüğü önceden yazılmaz.

## 8. Kurulamayacak cümleler (hakem-2 soru 2–3)

- Pozitif kontrol olmadan "tahmin RL'ye katkı vermez" ya da "RL sinyali kullanamaz" yazılmaz.
- RL'nin piyasayı, BIST-30'u ya da XU030'u yendiği, ya da mutlak Sharpe veya getirinin gerçekleştirilebilir olduğu yazılmaz.
- "RL > 1/N" yazılmaz, aynı evrende de. Hayatta kalan evren, kazananlara ağırlık veren politikayı 1/N'den daha çok şişirebilir.
- Sonuçlar BIST-30 endeks bileşenlerine genellenmez.
- Kurulabilenler eşleşik iç kıyaslardır (DT↔DTF, `stats`↔`causal`, terim), "aynı dondurulmuş, hayatta kalan evrende" sınırlamasıyla.

## 9. Beyan listesi

- **Eski ortam her kolda protokolde açık:** `obs_health` (türbülans), `features_in_support`/`dims_in_support` (K26 = a).
- **Faz 2:** temel oranlar tek anlık görüntü (`fundamentals_pit`); makro 2018 öncesi 0; eski ödülde r_f 0,15 (G2 yeni ortamda).
- **Ödül:** komisyon iki kez; `ema_level` düşük maruziyette −1'de doyar (§9.23).
- **Hiperparametreler** R1 öncesi açık pencerelerle seçildi (§2).
- **Evren hayatta kalan;** tarihsel üyelik geçici (E1).
- **2021 duman okuması:** 2026-11-15'te bir kez okundu (PPO, faz 1, `stats`, DT, tohum 101, 20.000 adım). Metriklere ana oturumda bakılmadı; kapanış hakemi kaydı açtı ama değerleri rapora yazmadı. Kayıt `spike` etiketiyle ana deneme kaydında (§9.34).
- **C1:** CPU; bit-eşlik yalnız CPU'da iddia edilir. **Golden kırmızı** (K1).

## 10. Bütçe (ölçüm, 2026-11-15)

**Ölçüm koşulları:**
- Duman koşumları CPU'da, 30 sembol, 20.000 adım, 10 doğrulama noktası dahil.
- 2024 katı; PPO faz 1 için 2021 katı.
- Deneme kaydına yazılmadı. Protokol denetimi (`verify_trial`) her koşumda boş döndü.
- 200.000 adıma doğrusal izdüşüm aşağıda.

| Algoritma | Faz 1 (`simple`) | Faz 2 (`psr`) |
|---|---|---|
| PPO | 307 sn | 537 sn |
| A2C | 222 sn | 462 sn |
| TD3 | 1.378 sn | 1.941 sn |

**Ön eleme:**
- 432 eğitim: algoritma başına 48 faz 1 + 96 faz 2.
- Tek süreçte PPO 18 sa + A2C 15 sa + TD3 70 sa = **~104 sa**.

**Sonrası budamaya bağlı** (nihai 720: 240 faz 1 + 480 faz 2; maliyet + pozitif kontrol 360, faz 2 varsayımıyla):

| Kazanan algoritma | Nihai | Maliyet + kontrol | Toplam, tek süreç | 4 paralel süreçle |
|---|---|---|---|---|
| A2C | ~76 sa | ~46 sa | **~226 sa** | ~57 sa |
| PPO | ~92 sa | ~54 sa | **~250 sa** | ~63 sa |
| TD3 | ~351 sa | ~194 sa | **~649 sa** | ~162 sa |

**KARAR (bütçe, kullanıcı):** TD3'ün kazanma olasılığı bütçeyi ~2,6 katına çıkarıyor. Seçenekler:
- (a) Protokol olduğu gibi kalır.
- (b) Bir bütçe tavanı önceden yazılır. Örneğin TD3 kazanırsa nihaide ikinci algoritma da koşar ve ikisi ayrı raporlanır, ya da paralellik artırılır.
- (c) TD3 Milestone 0' etkenlerinden çıkarılır; Faz 9 §4 tanımından sapma olur, beyan edilir.

## 11. Koşum

```bash
python scripts/run_milestone0.py --dry-run [--stage prescreen]
python scripts/run_milestone0.py --print-input-shas           # JSON 'expected' ile karşılaştır
CUDA_VISIBLE_DEVICES=-1 python scripts/run_milestone0.py --smoke [--algorithm TD3 --reward psr_differential]
CUDA_VISIBLE_DEVICES=-1 python scripts/run_milestone0.py --stage prescreen [--limit N] [--user <id>]
```

Defter: `results_dir()/milestone0/m0p-v1/ledger.jsonl`. İlk satırda protokol JSON'u ve belgesinin SHA'sı ile cihaz var. Kayıt protokolle uyuşmazsa hücre `mismatch` yazılır ve koşum durur.

## 12. Kapanış kapısı (hakem-2, 2026-11-15): Büyük revizyon → bu sürüm

| # | Bulgu | Yapılan |
|---|---|---|
| 1 | 2021 duman koşusuyla okundu | Beyan (§9), `spike` kaydı, geçici dizin silindi; duman artık yalnız `smoke.fold` = 2024'te, okunmamış katta koşmayı reddeder |
| 2 | Ön eleme 2021'i doğrulama olarak kullanıyordu; argmax yanlılığı | Ön eleme katları 2023–2026; puan son nokta doğrulaması |
| 3 | "SB3 varsayılanları" yanlış | Değerler ve kökeni JSON'da; kayıtta `effective_hyperparameters`; betik denetler |
| 4 | MDE yok | Güç simülasyonu (§6 tablo) |
| 5 | Doğrulayıcı kıyas tanımsız | Doğrulayıcı aile: DT↔DTF (+TOST), `stats`↔`causal`; geri kalan keşifsel |
| 6 | K4 ↔ M8 çelişkisi | Tek seviyeli ölçek eşitlemesi (k = 0,1652) |
| 7 | Pozitif kontrol ve hakem-rl önerileri | Pozitif kontrol 120 eğitim; hakem-rl tablosu aşağıda; §9.31 makro kolu MARL M8'e (§9.34) |
| 8 | Veri dondurulmuyordu | Panel ve tablo SHA'ları betikte zorlanıyor |
| 9 | Bilgisizlik yalnız 2013–2019'da | Açık pencerede kapı ölçüldü (§9.34) |
| 10–15 | `simple`↔`psr`, DSR N yolu, kurallar JSON dışı, CPU, alt sınır, ifadeler | §3 dipnotu; §6; JSON'a taşındı; `device_guard`; §10 "alt sınır"; tek taraflı p ve kesin pencere §9.34'te |

**hakem-rl İ3 Milestone 0' önerileri (§9.30):**

| Öneri | Karar | Gerekçe |
|---|---|---|
| Gözleme `p_cal` yerine `logit` | Yeni ortam (M2 `obs_spec`) | Eski env'in 4 alanlı yapısı değişmez (davranış dondurma); P4 v1 ikili yön (hakem-2 soru 2: ikili kalsın) |
| `meta_ok` bayrağı, maskeli hücrede S-obs 0 | Eski env: maskeli hücre nötr (4 alan 0, P4). Bayrak yeni ortamda | Durum uzayı yapısı korunur |
| `model_disagreement` kayan z-skoru | Yeni ortam | Eski env bu alanı almıyor (uyum = GBM işaret payı) |
| 0 ve 50 bps ile yeniden eğitim | **Milestone 0'** (§5 maliyet duyarlılığı) | — |
| Makro/çapraz kol 2018+ alt deneyi | MARL M8 (§9.34) | Kullanıcı kararı H3/M5 içindi (MARL hipotezleri) |
