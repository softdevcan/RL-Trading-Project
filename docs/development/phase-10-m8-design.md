# Faz 10 M8 — Nihai koşumlar: tasarım ve ön kayıt

**Sürüm:** v0.3 TASLAK · 2026-11-21 · dal `feat/phase10-m8-prereg`
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

## 4. Karar bekleyen maddeler (kullanıcı)

| # | Madde | Öneri |
|---|---|---|
| K-M8-1 | **Çıkarım birimi.** K20 "kat × tohum üzerinden birleştirilmiş örneklem dışı günlük getiri serisi" diyor; M5 §4.11 bu serinin ortak HAC p'sinin tohum varyansını taşımadığını gösterdi | **KARAR (2026-11-21, kullanıcı: öneri yönünde).** Birincil çıkarım **tohum × kat düzeyinde**: eşli (aynı tohum, aynı kat) Sharpe farkları, tabakalı bootstrap güven aralığı. Ortak HAC p betimsel kalır. Karar eşiği §3'teki boş dağılımla. K20'nin "birleştirilmiş seri" maddesi bu kararla değişir (plan kaydı) |
| K-M8-2 | **Doğrulayıcı kanıt** (Faz 9 plan: "yalnız 2021 katı + ön kayıt sonrası ileri veri mi") | Karar A1–A2 bittikten sonra, ön kayıt v1.0'da |
| K-M8-3 | **Kollar ve bütçe**: IPPO dahil mi, hangi iletişim kolları, sinyal kolları (M5 gerçek sinyal `O1_fark_yok`), kol budama kuralı (K20) | Boyut ölçümünden (§3) sonra, bütçeyle birlikte |
| K-M8-4 | **Seçim kuralı etkeni** (argmax ↔ son nokta; Faz 9 §9.12) | Ön kayıtta önceden tanımlı etken olarak; her kural kendi kaydıyla bir kez okunur ve deneme sayısına (N) eklenir |
| K-M8-5 | **Eğitim bütçesi** (kare sayısı) ve tohum sayısı | Boyut ölçümünden (§3) sonra |
| K-M8-6 | **Tahmin hedefi**: SARL kolunun `pit_v1` mi `pit_v2` mi kullanacağı (plan §3.1.3) | Sinyal kolu M8'de kalırsa ön kayıtta |

## 5. Sıra

A1 kat koşucusu → A2 SARL kolları → gürültü tabanı boyut ölçümü (GPU boşken) → ön kayıt v1.0 + hakem-rl ve hakem-finans (paralel) → dondurma → M8 koşuları → faz kapanışında hakem-2.

## Versiyon geçmişi

- **v0.3 (2026-11-21):** A2 SARL kolları bitti (`marl/sarl.py`: tek-ajan görünümü + `RowSharedPolicy`; `train_m3.py --sarl`); K-M8-1 KARAR (kullanıcı): birincil çıkarım tohum × kat düzeyinde. `test_marl_sarl` 4.

- **v0.2 (2026-11-21):** A1 kat koşucusu bitti (`train_m3.py --panel/--train-end/--checkpoints`, `eval_fold_m8.py`, `marl/selection.py`); `test_marl_core` 63, `test_marl_m3` 20.

- **v0.1 (2026-11-20):** Taslak: sabit girdiler, eksik altyapı (A1 kat koşucusu, A2 SARL kolları, A3 okuma kodu), gürültü tabanı tasarımı, karar bekleyen maddeler K-M8-1..6. Koşu yok.
