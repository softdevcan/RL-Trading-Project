# Faz 10 M8 — Nihai koşumlar: tasarım ve ön kayıt

**Sürüm:** v0.5 TASLAK · 2026-11-21 · dal `feat/phase10-m8-prereg`
**Üst belge:** [phase-10-marl-plan.md](phase-10-marl-plan.md) §2.3 (birincil karşıtlık), §3.2.4 (kollar), §3.2.8 (toplama düzeni), §6 K15 (SARL kolları), K20 (walk-forward, çıkarım birimi, güç).
**Durum:** TASLAK. Koşu yok. Bu belge dondurulmadan (ön kayıt v1.0) hiçbir M8 koşusu başlamaz. Bu belge yeni bir hipotez eklemez; planın M8'e bıraktığı maddeleri tek yerde toplar. Karar bekleyen maddeler §4'te.

---

## 1. Sabit girdiler (karar verilmiş)

| Girdi | Değer | Kaynak |
|---|---|---|
| Panel | `data/marl/panel_m8.npz` (`m8-v1`), 2013-06-03..2026-11-09, 30 sembol | [M2 tasarımı](phase-10-m2-design.md) §6.1 |
| Satış engeli | dönem bağımlı marj; 2020-03-13'ten %10, kapsam 2026-11-09 | M2 §2.5 |
| Bölme | **wf-v1**: 6 kat, test = takvim yılı 2021..2026 (2026 katı 2026-11-01'de biter), doğrulama = önceki yıl, eğitim 2013-06'dan genişleyen; purge 0 | `evaluation/splits.py` (kullanıcı onayı 2026-11-10) |
| Kontrol noktası kuralı | eğitim bütçesi boyunca eşit aralıklı 10 nokta; doğrulamada belirlenimci tek geçiş; `val_excess_sharpe` en yüksek olan (eşitlikte en erken). Erken durdurma ve yeniden uyum yok | `evaluation/rl_folds.py` (Faz 9 R1) |
| Birincil metrik | A8 risksiz faiz üstü günlük getiriden yıllık Sharpe, nominal TL; maliyet 20 bps ana, D4 ızgarası (0/10/20/25/50) raporlanır | `rl_folds.excess_sharpe`, `evaluation/units.py` |
| Ortam | ödül `'excess'`, logit sınırı 5 + ln N, listeleme kilidi 20, nakit A8 faizi, karar t kapanışı / yürütme t+1 açılışı | M2 §2–§3, `marl/CLAUDE.md` |
| Referans yapılandırma | MAPPO, eleştirmen = tüm yerel gözlemler, α = 0,25, lr 5e-4, başlangıç std 0,12, bölüm 252, `frames_per_batch` 6000, 10 epoch. **Kanıta değil varsayılana dayanır** (beyan) | [M3 tasarımı](phase-10-m3-design.md) §6.8 |
| Toplama düzeni | toplu yerel ortam (`marl/batched_env.py`), `n_envs` 10; ön kayıtta sabitlenir, çünkü eğitim dinamiğini değiştirir | plan §3.2.8 |
| Kollar | MAPPO (ve IPPO), SARL-ağırlık, SARL-eşmimari (K15); iletişim kolları (M4; taban gat, kullanıcı kararı); klasik baseline'lar 1/N, min-varyans, HRP, al-tut, XU030 | plan §3.2.4 |
| Birincil karşıtlık | her iletişim kolu için **MAPPO ↔ SARL-eşmimari**; aynı α, aynı gözlem, aynı tohumlar | plan §2.3 |

## 2. M8'den önce gereken altyapı (kodu yok)

- **A1 — Kat koşucusu. BİTTİ (2026-11-21).**
  - `train_m3.py` yeni seçenekler: `--panel`, `--train-end` ve `--checkpoints K` (varsayılanlar M3–M7 değerleri).
  - Kontrol noktaları `m3_runner` callback'iyle eğitim bütçesi boyunca eşit aralıklı yazılıyor; yalnız politika ağırlıkları kaydediliyor, `ckpt/index.json` iterasyon, kare ve ağırlık özetini tutuyor.
  - `scripts/marl/eval_fold_m8.py`: noktalar doğrulama yılında belirlenimci tek geçişle ölçülüyor; seçim `marl/selection.py` ile yapılıyor (`rl_folds`'un saf numpy eşi, `test_marl_core` [8] parite testi). Test yalnız seçilen noktayla **bir kez** okunuyor. Çıktı varsa betik duruyor; `--also-last` son noktayı ayrı kayıtla okuyor (K-M8-4).
  - Testler: `test_marl_m3` [4]. Noktalar doğru iterasyonda yazılıyor, son nokta son politikayla aynı, kontrol noktalı eğitim kontrol noktasız eğitimle bit-eş. `test_marl_batched_env`, `test_marl_m4`, `test_marl_m5_msg` ve `test_marl_env` yeşil.
  - Duman koşusu `panel_m8`'de, 2021+'a dokunmadan yapıldı: eğitim ..2017, doğrulama 2018, test 2019. Pencereler ve satır kuralı doğru, ikinci test okuması reddediliyor.
  - Kalan: deneme kaydına yazım ve wf-v1 kat tarihlerinin `evaluation/splits.py` ile denetimi. İkisi ana venv'deki okuma adımında yapılacak (A3).
- **A2 — SARL kolları (K15).** SARL-ağırlık: tek merkezi PPO aktörü, 30 boyutlu logit, nakit z₀ = 0 sabit (MARL'la aynı eylem temsili). SARL-eşmimari: MARL ile aynı paylaşımlı mimari, tek-ajan eğitimi (§2.3'teki itirazın kontrolü). Tasarım (2026-11-21):
  - **Tek-ajan görünümü** `marl/sarl.py::SarlPortfolioEnv`: `MarlPortfolioEnv`'in üstünde ince bir katman. Ortamın kendisi değişmez.
    - Tek ajan `agent_0`, grup adı MARL'la aynı (`agent`).
    - Gözlem: 30 yerel gözlem panel sırasıyla yan yana. Her satır MARL ajanının gözlemiyle aynı, kimlik one-hot dahil.
    - Eylem: 30 logit, i. bileşen ajan i'nin logiti. Hedef ağırlık, projeksiyon ve muhasebe aynı çekirdekte (iç ortam) yapılır.
    - Ödül: ajan ödüllerinin ortalaması = ölçek · (α · ort(kᵢ) + (1 − α) · G), yani takım ödülü. α = 0'da MARL ajanlarının ödülüne eşit.
    - Bölüm sonu iç ortamla aynı.
  - **Eğitim:** BenchMARL MAPPO, tek üyeli grupla; bu PPO'ya eşittir. Ortak log-olasılık satır log-olasılıklarının toplamıdır, bu yüzden oran kırpması ortaktır (§2.3'teki farkın kendisi).
  - **Eleştirmen:** aynı MLP yapılandırması, girdi düz gözlem. Bu, MAPPO eleştirmeninin gördüğü bilgiyle aynıdır (tüm yerel gözlemler). Çıkış 1 değer; MAPPO'da ajan başına değer. Fark yalnız çıkış katmanında (beyan).
  - **SARL-ağırlık politikası:** BenchMARL MLP, düz gözlemden (N·d) 2N çıkışa; MARL ile aynı hücreler.
  - **SARL-eşmimari politikası** `RowSharedPolicy`: düz gözlemi N satıra böler. Her satıra aynı MLP uygulanır (d → hücreler → 2; MARL paylaşımlı ajan MLP'siyle aynı katmanlar). Çıkış `[loc₁..loc_N, ölçek₁..ölçek_N]` (`NormalParamExtractor` yarıları). Aynı ağırlıklarla eylem dağılımı MARL ajanlarının ortak dağılımına eşittir.
  - **Beyan:** α > 0'da SARL kollarında ajan başına kredi yoktur; bu tasarım gereğidir ve karşıtlığın konusudur. Ön kayıt paketiyle hakem-rl okur.
  - **Testler** (`tests/test_marl_sarl.py`, venv-marl):
    - aynı eylem dizisiyle SARL ve MARL ortamının muhasebesi bit-eş, SARL ödülü MARL ödüllerinin ortalaması;
    - toplu yol tek ortamla bit-eş;
    - aynı ağırlıklarla `RowSharedPolicy` çıktısı MARL MLP'sinin ajan başına çıktısına eşit;
    - iki SARL kolunda kısa MAPPO koşusu ve tohum belirlenimciliği; kaydet/yükle ve belirlenimci değerlendirme.
  - **BİTTİ (2026-11-21).** `marl/sarl.py` (yalnız venv-marl), `make_env` `single_agent` anahtarı, `m3_runner` `rowshared` modeli, `train_m3.py --sarl {agirlik, esmimari}`. Testler `test_marl_sarl` 4/4; diğer venv-marl testleri ve `test_marl_core` yeşil. Komut satırı denemesi: `panel_m8`, toplu toplayıcı, kontrol noktaları ve `eval_fold_m8` çalışıyor. SARL-eşmimari politikası 77.826 parametre; MARL paylaşımlı politikasıyla aynı.
- **A3 — Okuma kodu.** Çıkarım birimi ve gürültü tabanı kuralı (§3) koşudan önce yazılır ve sentetik kayıtla sınanır (`--self-test`, M5 kalıbı).
  - Boyut ölçümü okuması `scripts/marl/analyze_noise_m8.py` koşudan önce yazıldı (2026-11-21), `--self-test` 12/12. Betik Sharpe tanımını, σ_tohum'u, 10 ayrık 3/3 bölmeyi ve MDE formülünü sınıyor. Kayıt denetimleri de var: git izi, pencereler 2019/2020, panel SHA, ön kayıt yapılandırması; eksik tohum ya da işlemsiz seri olursa etiket düşüyor. M8 birincil karşıtlık okuması ön kayıt v1.0 ile yazılır.

## 3. Gürültü tabanı dağılımı (engel 2)

**Neden.** M5 §4.11: neredeyse etkisiz kol GD ile G0 arasındaki fark eğitimde +0,18 Sharpe çıktı ve ortak HAC testinde p ≈ 0,001 verdi. Doğrulamada aynı fark +0,04 (p 0,33). Ortak HAC p'si piyasa günü gürültüsünü taşır, eğitim tohumu varyansını taşımaz (M5 §4.4 "p'nin sınırı"). Hakem-rl M5 ikinci okuma #1 de bunu söylüyor: gürültü tabanı tek bir gerçekleşme değil, bir dağılım olmalı.

**Soru.** Etkisi olmayan iki kol arasında, M8'in birincil istatistiği (k tohum × 6 kat) hangi dağılımı izler? Birincil karşıtlıkta gözlenen fark bu dağılımın neresinde?

**Tasarım önerisi (karar §4 K-M8-1).**
- Boş kol: referans yapılandırmanın aynısı, yalnız tohumu farklı. 2k tohum koşulur; rastgele k / k bölmelerinde birincil istatistik hesaplanır; bu değerler boş dağılımı verir.
- Ölçüm yeri: wf-v1 **2021 katının** eğitim (2013-06..2019) ve doğrulama (2020) pencereleri. Test yılları (2021+) okunmaz.
- Önce **boyut ölçümü** (15–20 dk, GPU): 6 tohum, toplu ortam, 3 eş zamanlı koşu. Çıktı: tohum başına doğrulama Sharpe'ının yayılımı (σ_tohum). Bundan asıl ölçüm için gereken tohum sayısı ve M8 karşıtlığının en küçük saptanabilir etkisi (MDE) hesaplanır. Asıl ölçüm ayrı bir ön kayıtla koşulur.
- Kullanım: birincil karşıtlığın etkisi boş dağılımın %95 üst ucunu aşmadıkça "fark var" yazılmaz.

### 3.1 Boyut ölçümü — ÖN KAYIT v1.0 (2026-11-21, sonuçtan önce)

**Soru.** Referans MAPPO yapılandırmasının M8 protokolündeki (kontrol noktası seçimi, örneklem dışı yıl) tohumdan tohuma yayılımı ne kadar? Bu ölçüm bir kol karşılaştırması değildir. Çıktısı asıl gürültü tabanı ölçümünün ve M8 tohum sayısının boyutudur (K-M8-3, K-M8-5).

**Kurgu.**
- Panel `panel_m8`. Sözde kat "2020": eğitim 2013-06..2018-12-31, doğrulama 2019, örneklem dışı yıl 2020. Hepsi M2–M7 geliştirme penceresinde; 2021+ okunmaz.
- Yapılandırma §1'deki referans MAPPO: F = 600.000, 10 kontrol noktası, toplu toplayıcı (`n_envs` 10), 20 bps, α = 0,25, lr 5e-4.
- Tohumlar 101–106 (daha önce kullanılmadı). GPU'da 3 eş zamanlı koşu.
- Düzenleyici `scripts/marl/run_noise_m8.py`, okuma `scripts/marl/analyze_noise_m8.py`. İkisi koşudan önce commit'lenir; koşular temiz ağaçtan yapılır.

**Okuma (betimsel; karar yok).**
- Tohum başına seçilen noktanın doğrulama Sharpe'ı ve 2020 fazla getiri Sharpe'ı; seçimi tanımsız (degenerate) tohum sayısı.
- σ_tohum: 2020 Sharpe'larının örnek standart sapması.
- Boş dağılım ilk bakış: 6 tohumun 3 / 3 bölmelerinde (10 ayrık bölme) grup ortalamaları farkı D. En büyük |D| ve çeyrekleri raporlanır.
- MDE tablosu: M8 eşli tasarımında k ∈ {6, 8, 12, 16, 24} tohum × 6 kat, tek yönlü %5, güç %80, σ_fark = √2 · σ_tohum. Bağımsız kollar ve katlar arası aynı varyans varsayılır (beyan).

**Sınırlar (önceden).** Tek bir örneklem dışı yıl (2020, COVID oynaklığı) yayılımı tipik olmayan biçimde büyütebilir; okuma bunu beyan eder. Sonuç M8 karar eşiğini değiştirmez, yalnız boyutu belirler. Ek tohum ya da başka bir pencere ancak yeni bir ön kayıtla.

### 3.2 Boyut ölçümü — okuma (2026-11-21)

Kaynak: `measurements/phase10-m8/noise_size.json`. Okuyan `analyze_noise_m8.py` @ 3bb10ed; koşular aynı commit'ten, temiz ağaçtan. Etiket **TAMAM**: 6/6 tohum, seçimi tanımsız tohum yok, her tohumda 10 ayrı ağırlık. Git izi, pencereler (2019 / 2020, 251 gün), panel SHA ve yapılandırma denetimleri doğru. Süre: 3 eş zamanlı koşu ~4 dk 47 sn; koşu başına etkin ~1,6 dk (eğitim 4,5 dk).

| Tohum | Seçilen adım | Doğrulama 2019 Sharpe (seçilen) | 2020 fazla getiri Sharpe |
|---|---|---|---|
| 101 | 540.000 | −0,03 | 0,59 |
| 102 | 420.000 | −0,08 | 0,68 |
| 103 | 300.000 | 0,75 | 0,43 |
| 104 | 360.000 | 0,39 | 0,83 |
| 105 | 60.000 | 0,53 | 0,55 |
| 106 | 360.000 | 0,38 | 0,43 |

- **σ_tohum = 0,155** (2020; ortalama 0,59, aralık 0,43..0,83). Seçilen noktanın doğrulama Sharpe'ı daha geniş yayılıyor (sd 0,32).
- **Boş dağılım ilk bakış:** 10 ayrık 3 / 3 bölme. En büyük |D| = 0,23; çeyrekler 0,05 / 0,09 / 0,14. Tek kat, 3'er tohum: aynı yapılandırmanın iki grubu arasında 0,2 Sharpe fark kendiliğinden çıkabiliyor.
- **MDE** (eşli, k × 6 kat, tek yönlü %5, güç %80, σ_fark = √2 · σ_tohum; varsayımlar §3.1):

  | k | 6 | 8 | 12 | 16 | 24 |
  |---|---|---|---|---|---|
  | MDE (Sharpe) | 0,091 | 0,079 | 0,064 | 0,056 | 0,045 |

- **Beyan (betimsel; ön kayıtta yoktu, karar değiştirmez):**
  - σ 6 tohumdan kestirildi ve belirsizliği büyük. χ²(5) güven aralığı %95'te 0,10..0,38; tek yönlü %80 üst sınır 0,23. Bu üst sınırla MDE 1,46 katına çıkar (k 12 → 0,094).
  - Seçilen noktanın doğrulama Sharpe'ı ile 2020 Sharpe'ı arasındaki ilişki 6 tohumda negatif (r = −0,42). Doğrulamada seçim bu katta örneklem dışını iyi öngörmüyor; 6 noktayla bundan sonuç çıkmaz. K-M8-4 (son nokta etkeni) bu nedenle ön kayıtta tutulmalı.
  - 2020 tek bir yıl (COVID oynaklığı).

## 4. Karar bekleyen maddeler (kullanıcı)

| # | Madde | Öneri |
|---|---|---|
| K-M8-1 | **Çıkarım birimi.** K20 "kat × tohum üzerinden birleştirilmiş örneklem dışı günlük getiri serisi" diyor; M5 §4.11 bu serinin ortak HAC p'sinin tohum varyansını taşımadığını gösterdi | **KARAR (2026-11-21, kullanıcı: öneri yönünde).** Birincil çıkarım **tohum × kat düzeyinde**: eşli (aynı tohum, aynı kat) Sharpe farkları, tabakalı bootstrap güven aralığı. Ortak HAC p betimsel kalır. Karar eşiği §3'teki boş dağılımla. K20'nin "birleştirilmiş seri" maddesi bu kararla değişir (plan kaydı) |
| K-M8-2 | **Doğrulayıcı kanıt** (Faz 9 plan: "yalnız 2021 katı + ön kayıt sonrası ileri veri mi") | **KARAR (2026-11-21, kullanıcı: öneri yönünde)** (§6.7): birincil okuma 6 katta; 2021 katı (hiç okunmamış tek kat) ayrıca raporlanır; ileri veri ayrı bir ekle |
| K-M8-3 | **Kollar ve bütçe**: IPPO dahil mi, hangi iletişim kolları, sinyal kolları (M5 gerçek sinyal `O1_fark_yok`), kol budama kuralı (K20) | **KARAR (2026-11-21, kullanıcı: öneri yönünde).** Kollar: MAPPO-`none`, MAPPO-`gat`, SARL-eşmimari, SARL-ağırlık ve boş kol (MAPPO-`none`, ayrı 12 tohum). IPPO yok, sinyal kolu yok, budama yok. K-M8-7 ile `self` kolu eklendi (§6.2) |
| K-M8-4 | **Seçim kuralı etkeni** (argmax ↔ son nokta; Faz 9 §9.12) | **Ön kayıtta sabitlendi** (§6.5): argmax birincil; son nokta ayrı kayıtla bir kez okunur, betimsel, deneme sayısına girer |
| K-M8-5 | **Eğitim bütçesi** (kare sayısı) ve tohum sayısı | **KARAR (2026-11-21, kullanıcı: öneri yönünde).** k = 12 tohum, F = 600.000 her katta |
| K-M8-6 | **Tahmin hedefi**: SARL kolunun `pit_v1` mi `pit_v2` mi kullanacağı (plan §3.1.3) | **Konusuz** (§6.2): M8'de sinyal kolu yok |
| K-M8-7 | **gat kolunun kontrolü** (yeni, §6.2) | **KARAR (2026-11-21, kullanıcı: öneri yönünde).** Mimari-eş iletişimsiz `self` kolu eklenir (+72 koşu, ~+2,4 sa); H3 = gat ↔ self |

## 5. Sıra

A1 kat koşucusu → A2 SARL kolları → gürültü tabanı boyut ölçümü (GPU boşken) → ön kayıt v1.0 (§6) + hakem-rl ve hakem-finans (paralel) → A4 kodu (koşucu, okuma, kural baseline'ları; self-test) → dondurma → M8 koşuları → faz kapanışında hakem-2.

## 6. M8 ön kaydı — v1.0 TASLAK (2026-11-21, sonuçtan önce)

**Durum:** TASLAK. Kullanıcı kararları tamam (K-M8-1..7). hakem-rl + hakem-finans okuması ve A4 kodu (§6.9) tamamlanınca **DONDURULUR**. Dondurulduktan sonra bu bölüm değişmez; değişiklik yeni sürümdür (v2.0) ve deneme sayısına eklenir. 2021+ test yılları dondurmadan önce hiçbir M8 koşusunda okunmaz.

### 6.1 Soru

**H1 (birincil; plan §2.3):** Aynı gözlem, aynı eylem temsili, aynı paylaşımlı mimari, aynı α ve aynı tohumlarla, çok-ajanlı eğitim (MAPPO: ajan başına oran kırpması ve ajan başına değer) tek-ajanlı eğitimden (SARL-eşmimari: ortak oran kırpması, tek değer) örneklem dışında farklı bir maliyet sonrası fazla getiri Sharpe'ı üretir mi? İki yönlü.

**İkincil (Holm, aile = 2):**
- **H2:** MAPPO-`none` ↔ SARL-ağırlık. Aynı eylem temsili, farklı mimari (paylaşımsız düz MLP). Soru: paylaşımlı yerel mimari, merkezi tek aktöre göre fark yaratır mı?
- **H3 (G1, iletişim):** MAPPO-`gat` ↔ MAPPO-`self` (K-M8-7). `self` aynı katmanlara ve parametre sayısına sahip, yalnız kendine dikkat ediyor (M4b). Böylece fark mimariden değil, ajanlar arası bilgi yolundan gelir (M4a §3.7 karışması kapanır).

### 6.2 Kollar

| Kol | Tanım | Tohumlar |
|---|---|---|
| **M-none** | MAPPO, paylaşımlı ajan MLP'si, iletişim yok (§1 referans yapılandırma) | 201–212 |
| **S-esm** | SARL-eşmimari: tek-ajan PPO, `RowSharedPolicy` (M-none ajan MLP'siyle aynı katmanlar, 77.826 parametre) | 201–212 |
| **S-agr** | SARL-ağırlık: tek-ajan PPO, düz gözlemden 2N çıkışlı MLP, z₀ = 0 | 201–212 |
| **M-gat** | MAPPO + `gat` (graf maskeli dikkat, `sector+mst`, 78.982 parametre; M4) | 201–212 |
| **M-self** | MAPPO + `self` (gat ile aynı katmanlar, birim maske; M4b) | 201–212 |
| **Boş kol** | M-none'ın aynısı, yalnız tohum farklı | 213–224 |

- Tohumlar daha önce kullanılmadı (M3–M7: 0–14; boyut ölçümü: 101–106).
- **Hiperparametreler** §1 referans yapılandırma: lr 5e-4, 10 epoch, `frames_per_batch` 6000, başlangıç std 0,12, α = 0,25, ödül `'excess'`, ölçek 100, bölüm 252, eleştirmen = tüm yerel gözlemler. HPO yok; her kolun deneme sayısı 0.
- **gat grafı:** M4'teki dondurulmuş `sector+mst` grafı (`measurements/phase10-m4/graph_rules.json`). M2 panelinin 2018-12-31'e kadarki eğitim satırlarından kurulmuştur. Her M8 katının eğitim penceresinin içinde kaldığı için ileriye bakmaz. Katlara göre yeniden kurulmaz; plan §3.2.5'teki "her eğitim penceresinde sabit" kuralının sıkı (eski veriyle) hâli. Graf `m8-v1` düzeltmesinden önceki panelden kuruldu (beyan); SHA okuma kaydında denetlenir.
- **Plandan sapmalar (beyan, kullanıcı kararı 2026-11-21):**
  - IPPO koşulmaz. Plan §2.3'teki "IPPO ve MAPPO'nun ikisinin birden raporlanması" azaltması M8'de yapılmaz.
  - Plan §2.3 "her iletişim kolu için MAPPO ↔ SARL-eşmimari" diyor; H1 yalnız `none`'da kurulur. gat'lı bir SARL-eşmimari kolu yazılmadı.
  - Sinyal kolu yok: gerçek sinyal kapıdan geçmedi (P5c) ve M5 gerçek sinyal okuması `O1_fark_yok`. Bu yüzden M8 G2'yi sınamaz; K-M8-6 konusuz.
  - K20 kol budama kuralı uygulanmaz: kol sayısı baştan bütçeye sığdırıldı.
- **Klasik baseline'lar** (betimsel; aynı yeni ortam muhasebesi, aynı maliyet, aynı test pencereleri):
  - 1/N günlük, al-tut (`m3_runner.rule_series`);
  - min-varyans (Ledoit-Wolf daraltmalı) ve HRP: aylık yeniden dengeleme, yalnız geçmiş 252 seans; kodu A4;
  - XU030 fiyat endeksi: aynı evren değil, temettüsüz, maliyetsiz. Kıyas değil, dipnotlu referans satırı (Faz 9 D1 beyanı).

### 6.3 Bölme, eğitim ve seçim

- **Katlar** wf-v1 (`evaluation/splits.py`, 2026-11-10 onayı): test yılı 2021..2026, doğrulama önceki yıl, eğitim 2013-06'dan doğrulama başlangıcına kadar genişleyen; 2026 katının testi 2026-11-01'de biter. Purge 0. Karar t kapanışı, yürütme t+1 açılışı.
- **Eğitim:** her (kol, tohum, kat) için ayrı koşu, F = 600.000 kare. Kare bütçesi katlar arasında sabittir; geç katlarda eğitim günü başına düşen kare azalır (beyan). Toplu toplayıcı `n_envs` 10, normalizasyon istatistiği kat eğitim penceresinden. Panel `panel_m8` (`m8-v1`, SHA 2b7ea84c…).
- **Kontrol noktası:** eşit aralıklı 10 nokta. Doğrulama yılında belirlenimci tek geçiş, 20 bps. `val_excess_sharpe` en yüksek olan seçilir, eşitlikte en erken (`marl.selection`; `rl_folds` ile parite testli). Hiçbiri tanımlı değilse son nokta seçilir ve `degenerate` sayılır.
- **Test:** seçilen noktayla bir kez (`eval_fold_m8.py`; çıktı varsa durur). `--also-last` ile son nokta ayrı kayıtla okunur (§6.5).
- **Koşu sayısı:** 6 kol (boş kol dahil) × 12 tohum × 6 kat = 432 koşu. Süre tahmini (boyut ölçümü ve M4 süre oranı): `none` türü ~1,6 dk etkin, gat/self ~2,0 dk; toplam ~12,5 sa, iki geceye bölünebilir. Süre tahmini ilk 3 koşuluk dalgada ölçülüp kayda yazılır. Koşucu kaldığı yerden sürer. Sıra kat → tohum → kol iç içedir; böylece yarıda kalan bir koşum her kolda dengeli kalır. Kısmi sonuç okunmaz.
- **Hata kuralı:** çöken bir koşu aynı yapılandırma ve tohumla bir kez yeniden başlatılır; olay kayda yazılır. GPU'da bit-eşlik iddia edilmez (M3 §1). İkinci çöküşte hücre eksik sayılır (§6.6).

### 6.4 Birincil istatistik ve karar kuralı (K-M8-1)

- **Hücre metriği:** SR_a(s, f) = kol a, tohum s, kat f'nin test yılı fazla getiri Sharpe'ı. Tanımı A8 risksiz faiz üstü, 20 bps, yıllık (`marl.selection.excess_sharpe(R, rf=R − X)`; seçim metriğiyle aynı). Tanımsızsa 0 alınır; sabit fazla getiri, yani tümüyle nakit demektir. Sayısı raporlanır.
- **Etki:** Δ̄(a, b) = (1 / 6k) Σ_f Σ_s [SR_a(s, f) − SR_b(s, f)]. Eşleme aynı tohum ve aynı kat üzerinden.
- **Belirsizlik (p_boot):** kat içinde tabakalı bootstrap; her katta 12 eşli fark yerine koyarak çekilir. B = 10.000, rastgele tohum 0. %95 yüzdelik güven aralığı; iki yönlü p_boot = 2 · min(P(Δ̄* ≤ 0), P(Δ̄* ≥ 0)).
- **Gürültü tabanı (p_null; §3):** her katta M-none (201–212) ve boş kol (213–224) koşuları, yani aynı yapılandırmanın 24 koşusu birleştirilir. Her katta bağımsız rastgele 12 / 12 bölme yapılır ve Δ̄⁰ hesaplanır. R = 10.000, tohum 1. p_null = P(|Δ̄⁰| ≥ |Δ̄|).
  - Önceden tanımlı bölme (201–212 ↔ 213–224) ayrıca raporlanır: kendi dağılımının içinde kalmalıdır (sağlık denetimi, karar değil).
  - Beyan: aynı tohumun iki kolda paylaşılması eşli farkları birbirine bağlayabilir. Bağımsız koşulardan kurulan boş dağılım bu yüzden tutucu yanda kalır.
- **H1 etiketi:**
  - p_boot ≤ 0,05 ve p_null ≤ 0,05 → Δ̄ > 0 ise **FARK_VAR_MARL**, Δ̄ < 0 ise **FARK_VAR_SARL**;
  - aksi hâlde **FARK_GÖZLENMEDİ**. Bu "eşdeğer" demek değildir. Güç beyanı: σ_tohum 0,155 ile iki yönlü %5'te MDE ≈ 0,072 Sharpe; σ'nın %80 üst sınırıyla ≈ 0,106 (§3.2). Bundan küçük farklar bu tasarımla ayırt edilemez.
- **H2, H3:** p = max(p_boot, p_null); iki p'ye Holm (α 0,05). H3'ün boş dağılımı da `none` koşularından gelir. gat ↔ self gibi mimari-eş kolların eşli farkı daha az gürültülü olabilir (M4b'de SE yarıya indi), bu yüzden H3'te p_null tutucu yandadır (beyan). Etiketler H1 ile aynı adlarla: MARL yerine birinci kol, SARL yerine ikinci kol.

### 6.5 Betimsel (kararları değiştirmez)

- Kat başına Δ̄_f ve işaret sayısı; kollar için ortalama ± std ve rliable tipi aralıklar (`evaluation/seed_stats`).
- **Son nokta kuralı** (K-M8-4): `test_last` aynı istatistiklerle ayrı tabloda verilir. Seçilen nokta son noktayla aynıysa ikinci okuma aynı seridir; bu oran raporlanır.
- **Maliyet ızgarası D4** (0 / 10 / 25 / 50 bps): ex post, aynı politika. Ağırlık yolu maliyetten bağımsız olduğu için (`accounting.step_fractional`) seri tam hesaplanır. Politika 20 bps ile eğitildi; başka maliyetle yeniden eğitilmiş politikanın sonucu değildir (beyan).
- K20'deki birleştirilmiş günlük seri: ortak HAC p (`sharpe_diff_hac_pooled`), yalnız betimsel (K-M8-1).
- DSR: N = okunan M8 yapılandırmaları (kol × seçim kuralı; boş kol ayrı yapılandırma sayılmaz). V denemeler arası SR varyansından. M2–M7 geliştirme koşuları 2021+'ı okumadığı için N'ye girmez (beyan).
- Devir, nakit ağırlığı, satış engeli sayısı; klasik baseline tablosu (§6.2).
- Kat başına doğrulama ↔ test ilişkisi: boyut ölçümünde r = −0,42 çıkmıştı.

### 6.6 Okuma öncesi denetimler (okuma JSON'unda; biri bozuksa etiket `UYARILI`)

- Tüm eğitim ve değerlendirme kayıtları aynı commit'ten ve temiz ağaçtan.
- Panel SHA `panel_m8.json` ile aynı; graf SHA M4 kaydıyla aynı.
- Kat pencereleri `evaluation/splits.py` wf-v1 ile aynı (A1'de kalan madde).
- Yapılandırma ön kayıtla aynı: F, lr, maliyet, α, `n_envs`, kontrol noktası, kol ve tohum.
- Eksik hücre: her kolda 72 hücre beklenir. Herhangi bir kolda %10'dan fazla eksik hücre varsa (8 ve üstü) o kolun karşıtlıkları **EKSİK** olur. Daha azsa eşleşmeyen hücreler düşer ve sayısı raporlanır.
- Deneme kaydı (C2): her kol ve seçim kuralı `evaluation/trial_registry` ile yazılır (A1'de kalan madde; ana venv okuma adımında).

### 6.7 Doğrulayıcı kanıt ve pencere açıklığı (K-M8-2, kullanıcı kararı)

- Faz 9 §9.11'e göre 2022-07-18..2026-08-28 penceresi eski ortamda (tek ajan, Faz 9 öncesi ve R1) okundu. Hiç okunmamış tek kat 2021. Yeni ortam ve M2–M7 geliştirmesi 2021+'ı hiç okumadı.
- **Karar (2026-11-21, kullanıcı):**
  - Birincil okuma 6 katın hepsiyle yapılır; her katın açıklık bayrağı tabloda yer alır.
  - **2021 katı ayrıca raporlanır.** Bu yalnız Δ̄ ve güven aralığıdır, etiket yoktur. Tek katın MDE'si ~0,18 (iki yönlü, k 12).
  - **İleri veri** (2026-11-10 ve sonrası): ön kayıttan sonra oluşan veride 2026 katının seçilmiş politikaları bir kez okunur. Bu, ayrı bir ön kayıt ekiyle ve en az 6 aylık veri birikince yapılır (wf-v2 değildir; yalnız tutma). Sonuç doğrulayıcı aile olarak ayrı raporlanır.
- Gerekçe: açık pencereler başka bir model sınıfıyla okundu. M8 tasarımı bu pencerelerin sonucuna göre ayarlanmadı, ama araştırmacının bilgisi beyan edilir.

### 6.8 Sonuç görüldükten sonra yapılmayanlar

Ek tohum ya da kol eklemek; kol, kat ya da hücre çıkarmak (§6.6 dışında); eşiği, testi, B ya da R'yi, seçim kuralını, birincil karşıtlığı, maliyeti ya da F'yi değiştirmek; seçilmeyen bir kontrol noktasını test etmek; kat yeniden tanımlamak. Gerekirse yeni bir ön kayıt yazılır ve deneme sayısına eklenir.

### 6.9 Dondurmadan önce gereken kod (A4)

- `scripts/marl/run_m8.py` (venv-marl): kat × tohum × kol koşucusu (§6.3 sırası, `wait_gpu`, kaldığı yerden sürme, `--dry-run`). Her koşu `train_m3.py --checkpoints 10` → `eval_fold_m8.py --also-last`.
- `scripts/marl/analyze_m8.py` (ana venv): §6.4–§6.6 ve `--self-test` (sentetik kayıtla etiketler, Holm, eksik hücre, denetimler).
- Min-varyans (LW) ve HRP kural baseline'ları, yeni ortam muhasebesiyle. Testi `test_marl_core`'a ya da ayrı bir teste eklenir.
- Hepsi koşudan önce commit'lenir; koşular tek commit'ten, temiz ağaçtan yapılır.

## Versiyon geçmişi

- **v0.5 (2026-11-21):** M8 ön kaydı v1.0 TASLAK (§6). K-M8-3/5 KARAR (kullanıcı: öneri yönünde): k = 12, F = 600.000; kollar M-none, M-gat, S-esm, S-agr + boş kol; IPPO ve sinyal kolu yok. K-M8-4 sabitlendi, K-M8-6 konusuz. K-M8-7 KARAR: `self` kolu eklenir (H3 = gat ↔ self, 432 koşu). K-M8-2 KARAR: 6 kat birincil, 2021 ayrı, ileri veri ekle. Koşu yok.

- **v0.4 (2026-11-21):** Okuma betiği `analyze_noise_m8.py` koşudan önce yazıldı (self-test 12/12). Gürültü tabanı boyut ölçümü koşuldu ve okundu (§3.2): σ_tohum 0,155, MDE k 12'de 0,064. K-M8-3/5 kullanıcı kararı bekliyor.

- **v0.3 (2026-11-21):** A2 SARL kolları bitti (`marl/sarl.py`: tek-ajan görünümü + `RowSharedPolicy`; `train_m3.py --sarl`); K-M8-1 KARAR (kullanıcı): birincil çıkarım tohum × kat düzeyinde. `test_marl_sarl` 4.

- **v0.2 (2026-11-21):** A1 kat koşucusu bitti (`train_m3.py --panel/--train-end/--checkpoints`, `eval_fold_m8.py`, `marl/selection.py`); `test_marl_core` 63, `test_marl_m3` 20.

- **v0.1 (2026-11-20):** Taslak: sabit girdiler, eksik altyapı (A1 kat koşucusu, A2 SARL kolları, A3 okuma kodu), gürültü tabanı tasarımı, karar bekleyen maddeler K-M8-1..6. Koşu yok.
