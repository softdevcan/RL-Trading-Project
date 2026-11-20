# Faz 10 M8 — Nihai koşumlar: tasarım ve ön kayıt

**Sürüm:** v0.1 TASLAK · 2026-11-20 · dal `feat/phase10-m8-prereg`
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

- **A1 — Kat koşucusu.** `scripts/marl/train_m3.py` bugün `panel_m2.npz`'ye ve 2018-12-31 eğitim sonuna sabit; kontrol noktası seçimi yok (`m3_runner`: `checkpoint_at_end=False`). Gereken: panel yolu ve wf-v1 katı parametresi (satırlar `evaluation/splits.py`'den); eğitim bütçesi boyunca 10 kontrol noktası; doğrulamada `rl_folds` kuralıyla seçim; seçilen modelle testin **bir kez** okunması; deneme kaydına (`evaluation/trial_registry.py`) yazım. Varsayılan davranış M3–M7 koşularıyla bit-eş kalır.
- **A2 — SARL kolları (K15).** SARL-ağırlık: tek merkezi PPO aktörü, 30 boyutlu logit, nakit z₀ = 0 sabit (MARL'la aynı eylem temsili). SARL-eşmimari: MARL ile aynı paylaşımlı mimari, tek-ajan eğitimi (§2.3'teki itirazın kontrolü). Tasarım bu belgede, kod M8 koşularından önce ayrı commit'te ve testli.
- **A3 — Okuma kodu.** Çıkarım birimi ve gürültü tabanı kuralı (§3) koşudan önce yazılır ve sentetik kayıtla sınanır (`--self-test`, M5 kalıbı).

## 3. Gürültü tabanı dağılımı (engel 2)

**Neden.** M5 §4.11: neredeyse etkisiz kol GD ile G0 arasındaki fark eğitimde +0,18 Sharpe çıktı ve ortak HAC testinde p ≈ 0,001 verdi. Doğrulamada aynı fark +0,04 (p 0,33). Ortak HAC p'si piyasa günü gürültüsünü taşır, eğitim tohumu varyansını taşımaz (M5 §4.4 "p'nin sınırı"). Hakem-rl M5 ikinci okuma #1 de bunu söylüyor: gürültü tabanı tek bir gerçekleşme değil, bir dağılım olmalı.

**Soru.** Etkisi olmayan iki kol arasında, M8'in birincil istatistiği (k tohum × 6 kat) hangi dağılımı izler? Birincil karşıtlıkta gözlenen fark bu dağılımın neresinde?

**Tasarım önerisi (karar §4 K-M8-1).**
- Boş kol: referans yapılandırmanın aynısı, yalnız tohumu farklı. 2k tohum koşulur; rastgele k / k bölmelerinde birincil istatistik hesaplanır; bu değerler boş dağılımı verir.
- Ölçüm yeri: wf-v1 **2021 katının** eğitim (2013-06..2019) ve doğrulama (2020) pencereleri. Test yılları (2021+) okunmaz.
- Önce **boyut ölçümü** (15–20 dk, GPU): 6 tohum, toplu ortam, 3 eş zamanlı koşu. Çıktı: tohum başına doğrulama Sharpe'ının yayılımı (σ_tohum). Bundan asıl ölçüm için gereken tohum sayısı ve M8 karşıtlığının en küçük saptanabilir etkisi (MDE) hesaplanır. Asıl ölçüm ayrı bir ön kayıtla koşulur.
- Kullanım: birincil karşıtlığın etkisi boş dağılımın %95 üst ucunu aşmadıkça "fark var" yazılmaz.

## 4. Karar bekleyen maddeler (kullanıcı)

| # | Madde | Öneri |
|---|---|---|
| K-M8-1 | **Çıkarım birimi.** K20 "kat × tohum üzerinden birleştirilmiş örneklem dışı günlük getiri serisi" diyor; M5 §4.11 bu serinin ortak HAC p'sinin tohum varyansını taşımadığını gösterdi | Birincil çıkarım **tohum × kat düzeyinde**: eşli (aynı tohum, aynı kat) Sharpe farkları, tabakalı bootstrap güven aralığı. Ortak HAC p betimsel kalır. Karar eşiği §3'teki boş dağılımla |
| K-M8-2 | **Doğrulayıcı kanıt** (Faz 9 plan: "yalnız 2021 katı + ön kayıt sonrası ileri veri mi") | Karar A1–A2 bittikten sonra, ön kayıt v1.0'da |
| K-M8-3 | **Kollar ve bütçe**: IPPO dahil mi, hangi iletişim kolları, sinyal kolları (M5 gerçek sinyal `O1_fark_yok`), kol budama kuralı (K20) | Boyut ölçümünden (§3) sonra, bütçeyle birlikte |
| K-M8-4 | **Seçim kuralı etkeni** (argmax ↔ son nokta; Faz 9 §9.12) | Ön kayıtta önceden tanımlı etken olarak; her kural kendi kaydıyla bir kez okunur ve deneme sayısına (N) eklenir |
| K-M8-5 | **Eğitim bütçesi** (kare sayısı) ve tohum sayısı | Boyut ölçümünden (§3) sonra |
| K-M8-6 | **Tahmin hedefi**: SARL kolunun `pit_v1` mi `pit_v2` mi kullanacağı (plan §3.1.3) | Sinyal kolu M8'de kalırsa ön kayıtta |

## 5. Sıra

A1 kat koşucusu → A2 SARL kolları → gürültü tabanı boyut ölçümü (GPU boşken) → ön kayıt v1.0 + hakem-rl ve hakem-finans (paralel) → dondurma → M8 koşuları → faz kapanışında hakem-2.

## Versiyon geçmişi

- **v0.1 (2026-11-20):** Taslak: sabit girdiler, eksik altyapı (A1 kat koşucusu, A2 SARL kolları, A3 okuma kodu), gürültü tabanı tasarımı, karar bekleyen maddeler K-M8-1..6. Koşu yok.
