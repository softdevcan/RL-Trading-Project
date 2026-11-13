# Faz 9 — Araştırma Geçerliliği ve Tez Deney Altyapısı (Plan)

**Durum:** **Onaylandı (2026-11-06)**, Faz 10 planındaki eklemelerle (B7–B10, F4, P, V, R1). İlk kapı (V + A + C) uygulandı → [§9 Uygulama durumu](#9-uygulama-durumu). Plan ilk kez 2026-10-07'de yazıldı.
**Dayanak:** [vision-and-scope.md v0.2](../thesis/vision-and-scope.md), [literature-review-verification.md](../thesis/literature-review-verification.md), [roadmap.md → Bilinen Metodolojik Açıklar](roadmap.md#bilinen-metodolojik-açıklar-planlanmadı), indirilen tam metinler ([../reference/papers/README.md](../reference/papers/README.md))
**Kapsam:** Literatür doğrulamasında bulunan açıkların kapatılması, tez deneylerinin istatistiksel altyapısı ve MARL aşamasının (Milestone 2–3) literatüre dayalı tasarım kararları.

---

## 0. Özet

1. **Sıra: önce geçerlilik, sonra yenilik.** Değerlendirme araçları ve veri hattı sızıntılıyken üretilecek MARL sonuçları tezde savunulamaz. Bu yüzden MARL'dan önce yedi iş paketi var (A–G); MARL (M1–M8) onların üzerine kurulur.
2. **Bu turda tam metin okunarak 10 yeni bulgu çıktı** (§2). En ağırları: **Deflated Sharpe hesabı yanlış** (makalenin kendi sayısal örneğine 0,9004 yerine 0,4031 veriyor; tek denemede her zaman 1,0) ve **G2 mekanizmasının yarısı DebUnc'un dikkat ölçeklemesiyle matematiksel olarak eşdeğer** — özgünlük iddiası bir kez daha daralmalı.
3. **Her iş maddesi üç şeye bağlı:** bir literatür kaynağı (arşivdeki PDF yolu), bir kod konumu ve bir kabul ölçütü. Mümkün olan her yerde kabul ölçütü **makalenin kendi sayısal örneği** (ör. DSR = 0,9004).
4. **Davranış dondurma korunur:** varsayılan çıktıyı değiştiren her düzeltme bayrakla gelir; varsayılanın ne zaman çevrileceği ve golden'ın ne zaman yenileneceği ayrı karardır (§7).
5. **Nihai deneylerden önce analiz planı dondurulur** (ön kayıt, M8): hipotezler, birincil metrik, testler, tohum sayısı, çoklu karşılaştırma ailesi.
6. **Kullanıcı kararı gereken 5 açık konu** §7'de (6. konu, G2 metni, 2026-10-07'de işlendi).
7. **Kaynak tabanı:** planın atıfları [literature-table.md](../thesis/literature-table.md)'deki seçilmiş kaynakçayla hizalandı (2026-10-07).

---

## 1. Planlama ilkeleri

| İlke | Uygulama |
|---|---|
| Geçerlilik önce | A–G bitmeden üretilen sayılar tez/makalede yalnızca "ön sonuç" olarak kullanılır |
| Kaynağa bağlılık | Her iş maddesi en az bir doğrulanmış kaynağa atıf yapar; atıfsız yöntem kararı "tasarım varsayımı" diye işaretlenir |
| Altın değerli test | İstatistik fonksiyonları makaledeki sayısal örneklerle ya da H₀ altında boyut (size) simülasyonuyla test edilir |
| Hazır ve denetlenmiş kütüphane | MCS, stationary bootstrap, HAC: `arch`; IQM/katmanlı bootstrap: `rliable`; FDR: `statsmodels`. Elle yazılan yalnızca kütüphanesi olmayanlar (DSR, PBO/CSCV, LW Sharpe testi) |
| Davranış dondurma | Yeni davranış bayrakla (`opt-in`); eski modellerin yüklenmesi ve state space bozulmaz (CLAUDE.md) |
| Ön kayıt | Nihai deney koşmadan önce `docs/thesis/analysis-plan.md` dondurulur; sonrasında yapılan her analiz "keşifsel" etiketi taşır |

---

## 2. Bu planlama turunda doğrulanan yeni bulgular

Kod `main` @ b82fedb (çalışma ağacında yalnızca belge değişiklikleri var). "Kaynak" sütunundaki yollar `docs/reference/papers/` altındadır.

| # | Bulgu | Kanıt | Kaynak | Etki |
|---|---|---|---|---|
| **Y1** | **Deflated Sharpe hesabı üç yerden yanlış:** (a) eşik SR₀'da denemeler arası varyansın karekökü √V[{SRₙ}] yok; eşik standart normal biriminde kalıyor ve **yıllıklandırılmış** SR ile kıyaslanıyor; (b) σ(SR) formülüne yıllıklandırılmış SR giriyor (formül dönem başı SR ister); (c) `(γ₄−1)/4` yerine **fazla** basıklık/4 kullanılıyor. Makalenin s.10 örneği (N=100, V=½, T=1250, γ₃=−3, γ₄=10, SR=2,5) için doğru değer **0,9004**; repo **0,4031** veriyor. `n_trials=1` iken SR*=0 olduğundan sonuç **1,0000** — her strateji "anlamlı". Fonksiyon hiçbir akıştan çağrılmıyor (yalnızca kullanılmayan `compute_comprehensive_metrics` içinden) | [evaluator.py:495](../../prediction/evaluator.py#L495) | `G_evaluation/2014_Bailey_DeflatedSharpe.pdf` s.9–10 | Faz 3.4'te "DSR metriği eklendi" denmişti; tezde DSR ile raporlanmış bir sayı varsa geçersiz |
| **Y2** | **Ödüldeki "Differential Sharpe Ratio" türev değil.** Moody & Saffell'in Dₜ'si Sharpe'ın η'ya göre türevidir: Dₜ = (Bₜ₋₁ΔAₜ − ½Aₜ₋₁ΔBₜ) / (Bₜ₋₁ − Aₜ₋₁²)^{3/2}. Kod EMA Sharpe'ın **düzeyini** (Aₜ − r_f)/√(Bₜ − Aₜ²) hesaplayıp `tanh(x/2)` ile sıkıştırıyor | [reward_functions.py:155](../../env/reward_functions.py#L155) | Moody & Saffell (IEEE TNN 2001) — asıl kaynak, elle getirilecek; türetim şimdilik ikincil kaynaktan (Nawathe vd. 2024 §3.4, kaynakçada değil) | Tezde "diferansiyel Sharpe" denemez; ya düzeltilir ya "EMA Sharpe düzeyi" diye tanımlanır |
| **Y3** | **Rejim metriği sonuca koşullanıyor:** `compute_regime_metrics` "boğa"yı **gerçekleşen** getirisi pozitif örnekler diye tanımlıyor. Bu bir rejim değil, hedef değişkene göre alt kümeleme; boğa kümesinde yön isabeti tanım gereği yanlı | [evaluator.py:208](../../prediction/evaluator.py#L208) | lit. taraması §7.5 (önceden ve kural tabanlı etiket) | Bu fonksiyonla raporlanan "rejim isabeti" kullanılamaz |
| **Y4** | **Güven tanımı tahmin edilen yönün olasılığı değil:** yön fiyat tahmininden (`change_pct`), güven yön başlıklarının ortalamasından `max(p, 1−p)` ile geliyor. İkisi zıt yönü gösterebilir; o durumda "güven" diğer yönün olasılığıdır. Kalibrasyon ölçülmüyor | [ensemble.py:542](../../prediction/models/ensemble.py#L542) | `E_forecast-leakage/2017_Guo_Calibration_ICML.pdf` | G2'nin "kalibre güven" iddiasının ön koşulu |
| **Y5** | **Sektör eşlemesi 30 sembolün yalnızca 7'sinde var** | [bist30_symbols.py:51](../../data/bist30_symbols.py#L51) | — | Seçenek B (sektör ajanları) için kaynaklı bir sınıflama gerekiyor |
| **Y6** | **Eylem uzayı hisse başına [−1, 1] alım-satım sinyali**, portföy ağırlığı değil | [trading_env.py:159](../../env/trading_env.py#L159) | Jiang vd. (2017, EIIE) — softmax ağırlık vektörü politikası; DRL portföy literatürünün yerleşik eylem temsili | MARL'da sermayenin ajanlara bölünmesi bir tasarım kararı (M2) |
| **Y7** | **G2(a) ≡ DebUnc "Attention-All".** Dikkat logitine log c eklemek softmax sonrası ağırlığı c ile çarpıp yeniden normalize etmekle aynıdır: softmax(ℓⱼ + log cⱼ) ∝ cⱼ·exp(ℓⱼ). DebUnc tam olarak bunu yapıyor (ağırlığı mⱼ = 1/belirsizlik ile çarpıp ölçeklenenleri yeniden normalize ediyor) | — | `B_marl-comm/2024_Yoffe_DebUnc.pdf` "Attention Scaling" bölümü | §2.5 G2 ifadesi yeniden yazılmalı (bkz. M5) |
| **Y8** | **ADMAC ayrık eylem varsayıyor:** güvenilirlik kestiricisinin etiketi "mesaj en iyi eylemi (K seçenekten) öneriyor mu" kuralından geliyor. Sürekli portföy eyleminde doğrudan uygulanamaz | — | `B_marl-comm/2024_Yu_ADMAC.pdf` "Reliability Estimator" | Ablation kolu "ADMAC'tan esinlenen" diye tanımlanmalı, replikasyon denmemeli |
| **Y9** | **Yang vd. (2024) su kalitesi verisinde.** Sızıntısız ayrıştırma için yöntem kaynağı; finans kanıtı değil. Finans kanıtı Carnelossi Furlaneto (2017), Xiong vd. (*Applied Intelligence* 2025; ileri atıf taramasında bulundu) ve Zhang, Peng & Song (2024) — üçü de elle getirilecek | — | `E_forecast-leakage/2024_Yang_EMD-Leakage_SciRep.pdf` | Makale 2 literatür paragrafında kaynakların rolü ayrılmalı |
| **Y10** | **IMF özellikleri zaten filtrelenmiş seriden çıkarılıyor** (çift ayrıştırma); `imf_energy_ratio` gürültüsü önceden atılmış seride ölçülüyor | [feature_engineer.py:89](../../prediction/feature_engineer.py#L89) | — | Küçük; B5 ile birlikte düzelir |

---

## 3. Bağımlılık ve sıra

```
A Değerlendirme çekirdeği ──┐
C Tekrar üretilebilirlik ───┼──► Milestone 0' (baseline yeniden dondurma) ──► Makale 2 deneyleri
B Sızıntı kapatma ──────────┤                                        │
D Baseline + raporlama ─────┤                                        ▼
E Evren + sektör ───────────┤                    M1 çerçeve spike'ı ──► M2–M7 MARL ──► M8 ön kayıt ──► nihai koşumlar
F Kalibrasyon ──────────────┤                                        ▲
G Ödül düzeltmesi ──────────┘────────────────────────────────────────┘
```

A ve C bağımsız ve en ucuz; ilk ikisi olmalı (sonraki her iş onların ölçümünü kullanır). B'nin S2 maddesi varsayılan hatta etkin olduğu için en acil sızıntı. E ve F, MARL'ın (M2, M5) ön koşulu.

**Efor tahminleri kabadır** (tek geliştirici, gün); M1 spike'ından sonra yeniden hesaplanacak.

---

## 4. İş paketleri (geçerlilik)

### A — Değerlendirme çekirdeği (istatistik)

**Amaç:** vision-and-scope §4.1'deki "soruya göre test" tablosunu çalışan, test edilmiş koda dönüştürmek. **Efor:** ~6 gün.

Önerilen konum: yeni `evaluation/` paketi (tahmine ve RL'e ortak; `prediction/evaluator.py` tahmine özgü kalır, oradaki hatalı fonksiyonlar bunlara yönlendirilir).

| İş | Kaynak | Kabul ölçütü |
|---|---|---|
| A1 PSR + DSR (dönem başı SR, √V[{SRₙ}], γ₄ ham basıklık; N ve V deneme kaydından) | Bailey & López de Prado 2014 | s.10'daki üç sayı: SR₀ = 0,1132 (dönem başı) ve **DSR = 0,9004** (N=100); N=46 → 0,9505; normal getiri (γ₃=0, γ₄=3), N=88 → 0,9505 (±1e-4). Üçü de doğru formülle 2026-10-07'de yeniden üretildi |
| A2 Diebold-Mariano: HAC (Newey-West, h−1 gecikme) + Harvey-Leybourne-Newbold küçük örneklem düzeltmesi + t dağılımı | Diebold & Mariano 1995; Harvey vd. 1997; Diebold 2015 | H₀ altında simülasyonda reddetme oranı %5 ± 1 (AR(1) hata farkıyla) — eski fonksiyon aynı testte belirgin fazla reddetmeli |
| A3 Ledoit-Wolf Sharpe farkı testi: HAC (prewhitened QS çekirdeği) ve studentize dairesel blok bootstrap | Ledoit & Wolf 2008 (§3.1, §3.2) | H₀ (eşit Sharpe, korelasyonlu GARCH getiri) simülasyonunda boyut ≈ %5; blok boyu seçimi makaledeki kalibrasyon yöntemiyle ya da sabit b ile belgelenir |
| A4 Model Confidence Set | Hansen, Lunde, Nason 2011 → `arch.bootstrap.MCS` | `arch` testleriyle tutarlılık; açıkça kötü bir modelin kümeden düşmesi |
| A5 Tohumlar arası raporlama: IQM, katmanlı bootstrap GA, iyileşme olasılığı, performans profili | Agarwal vd. 2021 → `rliable` | Kütüphane çıktısıyla birebir; tablo + grafik üreticisi |
| A6 PBO / CSCV | Bailey vd. 2017 | Saf gürültü stratejilerde PBO ≈ 0,5; bir stratejiye gerçek sinyal eklenince belirgin düşüş |
| A7 Benjamini-Hochberg FDR | Benjamini & Hochberg 1995 → `statsmodels` | Aile tanımı M8'de; fonksiyon sarmalayıcı |
| A8 **Tek kaynak risksiz faiz** (D3): günlük politika faizi / TLREF serisi; `evaluator`, `reward_functions`, `model_analysis` aynı seriyi okur | vision §4.1 | Üç sabit (0 / 0,15 / 0,02) kodda kalmaz; yoksa açık hata |
| A9 Eski çağrılar: `model_analysis` `ttest_ind` → eşleştirilmiş/bootstrap; hatalı `diebold_mariano_test` ve `compute_deflated_sharpe_ratio` yeni pakete yönlenir | doğrulama §6 #8 | `tests/test_eval_stats.py` |

**Golden etkisi:** yok (bu fonksiyonlar hiçbir akıştan çağrılmıyor). **Tez etkisi:** tez-final.pdf'te DSR/DM/t-testi sonucu raporlandıysa yeniden hesaplanmalı.

### B — Sızıntı kapatma (S1–S4 + Y3, Y10)

**Amaç:** "leakage yok" ifadesini yeniden kurulabilir kılmak. **Efor:** ~8 gün + ICEEMDAN hesaplama süresi (B5'te ölçülecek).

| İş | Açık | Kaynak | Kabul ölçütü |
|---|---|---|---|
| **B0 Genel bekçi: kesme değişmezliği testi** — her özellik grubu için özellikleri tam seride ve t'de kesilmiş seride hesapla; t anındaki değerler **bit-eş** olmalı | Tümü | Hewamalage vd. 2023; Lones 2024 | `tests/test_no_lookahead.py`; mevcut hatta S1/S2/S4 açıkken testin **kaldığı** (sızıntıyı sayısal olarak gösterdiği) kaydedilir — bu çıktı Makale 2'nin motivasyon şekli olur |
| B1 Makro `bfill` kaldır; başlangıç boşluğu ya düşürülür ya eksik bayrağıyla sabit doldurulur | S2 (**varsayılan hatta etkin**) | doğrulama §6 #2 | [macro_fetcher.py:275](../../data/macro_fetcher.py#L275), [:454](../../data/macro_fetcher.py#L454); B0 geçer |
| B2 EVDS verisinin gerçekten gelmesi (faiz sabit 50, TÜFE/ÜFE sabit 0) — anahtar/seri kodu teşhisi | Veri kalitesi | doğrulama §6 ek bulgu | Manifest `data_quality` bayrağı `fallback_*` değil |
| B3 TÜFE/ÜFE'yi **açıklanma tarihine** hizala (TÜİK yayın takvimi; takvim alınamazsa muhafazakâr sabit gecikme — tasarım varsayımı olarak beyan) | S3 | lit. taraması §7.4 | B0 geçer; açıklanmadan önceki günlerde değer değişmez |
| B4 Temel oranlar: grup kapalı kalır, tezde sınırlılık olarak beyan; point-in-time kaynak (KAP mali tablolar) **kapsam dışı** | S4 | doğrulama §6 #4 | Grup seçilirse uyarı + manifest bayrağı |
| B5 **Nedensel ICEEMDAN:** her t için yalnızca [t−W+1, t] penceresi ayrıştırılır, son noktanın filtrelenmiş değeri ve IMF özellikleri alınır; **hedef ham fiyattan**; IMF özellikleri ham seriden (Y10). Disk önbelleği (sembol, tarih, W, parametreler) + sembol paralelliği | S1, Y10 | Yang vd. 2024 (SW-EMD, yöntem); Zhang, Peng & Song 2024 (kayan CEEMDAN, finans); Xiong vd. 2025 (hisse tahmininde tek seferlik gürültü giderme) | Önce **maliyet ölçümü** (30 sembol × ~2000 gün × W∈{128, 256} × trials∈{50, 100}); B0 geçer; kenar etkisi için W duyarlılık analizi raporlanır |
| B6 Rejim metriğini önceden tanımlı etikete taşı (XU100 200g trend + 60g gerçekleşmiş oynaklık tercili); eski fonksiyon kaldırılır | Y3 | lit. taraması §7.5 | Etiket t anında yalnızca ≤t veriyle hesaplanır (B0) |

**Golden etkisi:** B1 varsayılan hatta etkin olduğundan **tahmin golden'ı değişir** → §7 Karar 1.

### C — Tekrar üretilebilirlik

**Efor:** ~3 gün.

| İş | Kaynak | Kabul ölçütü |
|---|---|---|
| C1 RL eğitimine tohum (SB3 `seed=`, `set_random_seed`, env `reset(seed=)`); tohum `<model>.meta.json` ve manifeste yazılır (D1) | Henderson vd. 2018 | Aynı tohumla iki CPU koşumu aynı ödül eğrisi; GPU'da sapma ölçülüp belgelenir |
| C2 **Deneme kaydı** (`trial_registry.jsonl`, kullanıcı çalışma alanında): her değerlendirilen yapılandırma × tohum × HPO denemesi + günlük getiri serisi yolu | Bailey & LdP 2014 (N, V); Bailey vd. 2017 (PBO matrisi) | A1'in N ve V'si, A6'nın matrisi bu kayıttan okunur |
| C3 Veri anlık görüntüsü: parquet + SHA-256, manifestte | vision §4.1 | Hash değişince manifest uyarır |
| C4 Ortam kaydı: paket sürümleri, GPU, `torch.backends` belirlenimcilik bayrakları manifestte | Henderson vd. 2018 | — |

### D — Baseline'lar ve raporlama

**Efor:** ~5 gün (D3 için B2'deki TÜFE verisi gerekir).

| İş | Kaynak | Kabul ölçütü |
|---|---|---|
| D1 Aylık yeniden dengelenen 1/N; al-tut; gerçek XU030 serisi (sentetik yedek yalnızca etiketli) | DeMiguel vd. 2009 | Mevcut [benchmark_strategies.py](../../scripts/benchmarking/benchmark_strategies.py) genişler |
| D2 Uzun-yalnız minimum varyans (Ledoit-Wolf 2004 daraltmalı kovaryans), HRP | Ledoit & Wolf 2004; López de Prado 2016; Millea & Edalat 2022 | Pencere yalnızca geçmiş veri (B0 mantığı) |
| D3 Getiri birimi: nominal TL, TÜFE-reel TL, USD | vision §4.1; lit. taraması §7.6 | Tek tablo üreticisi üç birimi birlikte basar |
| D4 Maliyet duyarlılığı: 0 / 10 / 20 (ana) / 25 / 50 bps gidiş-dönüş | Zhang, Zohren & Roberts 2020 | Maliyet eğrisi grafiği |
| D5 LaTeX tablo/şekil üreticisi (A5 çıktıları + D1–D4) | — | `results/latex/` |

### E — Evren ve sektör sınıflaması

**Efor:** veri bulunmasına bağlı (~2 gün kod).

| İş | Kaynak | Kabul ölçütü |
|---|---|---|
| E1 Tarihsel BIST-30 bileşenleri (Borsa İstanbul dönemsel endeks bileşenleri) → tarihe göre evren maskesi (D5) | lit. taraması §7.4 | Bulunamazsa hayatta kalma yanlılığı sınırlılık olarak beyan (§7 Karar 3) |
| E2 30 sembolün kaynaklı sektör sınıflaması (BIST sektör endeksi üyeliği ya da KAP sektörü); kaynak ve tarih dosyada; ~6–8 ajan grubu | Y5 | Her sembol tam bir grupta; grup başına ≥2 sembol |

### F — Tahmin güveni ve kalibrasyon

**Efor:** ~3 gün. **G2'nin ön koşulu.**

| İş | Kaynak | Kabul ölçütü |
|---|---|---|
| F1 Güveni yeniden tanımla: p(UP) tek olasılık; yön = argmax; güven = tahmin edilen yönün olasılığı (Y4). Fiyat tahmini ile yön çelişkisi ayrı bayrak | Guo vd. 2017 | Çelişki oranı manifestte raporlanır |
| F2 Sıcaklık ölçekleme (ve karşılaştırma için izotonik) — **yalnızca OOF meta bölümünde** fit; test bölümünde değerlendirme | Guo vd. 2017 (§4) | Test ECE'si kalibrasyon sonrası düşer ya da düşmediği raporlanır |
| F3 ECE (eşit genişlikli 15 kutu, Guo vd. ile aynı), Brier skoru, güvenilirlik diyagramı; `compute_calibration` eğitim akışına bağlanır | Guo vd. 2017 | Her eğitim manifestinde ECE |

### G — Ödül tanımı

**Efor:** ~2 gün.

| İş | Kaynak | Kabul ölçütü |
|---|---|---|
| G1 Gerçek Moody-Saffell Dₜ'yi seçenek olarak ekle (`sharpe_term = 'ema_level' \| 'differential'`); varsayılan değişmez | Moody & Saffell 2001 (elle getirilecek); Nawathe vd. §3.4 | Sabit getiri dizisinde ΣDₜ ≈ Sharpe yaklaşımı sayısal olarak doğrulanır |
| G2 Risksiz faiz A8'den | — | — |
| G3 Tezdeki ödül tanımı kodla aynı terimlerle yazılır (Y2) | — | Metin–kod denetimi |

### Milestone 0' — Baseline'ı yeniden dondurma

A–G bittikten sonra mevcut Milestone 0 bu altyapıyla koşulur: 30 sembol, walk-forward, ön elemede 3–5, nihaide ≥10 tohum; tek-ajan PPO/A2C/TD3 + D1–D2 baseline'ları. **tez-final.pdf sonuçlarının hangi veri durumuyla üretildiği** bu aşamada karşılaştırılır (açık madde).

### Makale 2 deney tasarımı (Milestone 0' üzerine)

| Kol | ICEEMDAN | TATS |
|---|---|---|
| K0 | yok | yok |
| K1 | tüm seri (sızıntılı — yalnızca karşılaştırma için) | yok |
| K2 | nedensel (B5) | yok |
| K3 | nedensel (B5) | var |
| K4 | tüm seri | var |

Tahmin katmanı: DM-HAC (A2) ikili, MCS (A4) beşli. RL katmanı: DSR (A1, N deneme kaydından), LW Sharpe farkı (A3), tohum dağılımı (A5). Beklenti literatürle uyumlu biçimde K1/K4'ün avantajının K2/K3'te küçülmesi; **beklenti hipotez değildir**, sonuç ne çıkarsa raporlanır. TATS atfı: Kazemdehbashi (2026).

---

## 5. MARL tasarım kararları (Milestone 2–3)

Bunlar literatürden türetilen **tasarım kararlarıdır**; uygulaması A–G'den sonra.

### M1 — Çerçeve spike'ı (~1 hafta, karar noktası)

Aday: PettingZoo Parallel + RLlib `MultiRLModule` (vision §3.5 F2) vs BenchMARL + TorchRL (Bettini vd. 2024; hazır IPPO/MAPPO + GNN). Ölçütler: (1) değiştirilebilir iletişim katmanlı özel modül yazılabiliyor mu, (2) tohumla tekrar üretilebilirlik (C1 testi), (3) **Windows'ta çalışma** (Ray'in Windows desteği spike'ta doğrulanmalı), (4) RTX 4060'ta 8 ajan × 200K adım süresi. Çıktı: seçim + gerekçe + ölçülen bütçe → §5.1 hesap tablosu güncellenir.

### M2 — Ortam

- PettingZoo Parallel API; ajan = sektör grubu (E2).
- **Sermaye bölüşümü (Y6):** her ajan kendi grubu için bir bütçe logiti + grup içi ağırlıklar (softmax, Jiang vd. 2017 EIIE; ya da Dirichlet politika — tasarım seçeneği) üretir; grup bütçeleri **öğrenilmeyen** bir softmax ile normalize edilir. Bu, G3'ün "merkezi dağıtıcı olmadan" ifadesiyle tutarlıdır ama metinde açıkça yazılmalı: normalizasyon sabit bir işlemdir, öğrenen bir üst ajan değildir.
- Gözlem, eylem ve maliyet tek-ajan ortamıyla aynı veri hattından (B sonrası); maliyet D4 ızgarası.
- Ödül: vision §3.3 γ (α·bireysel + (1−α)·portföy), δ ablation; öncül Hao vd. 2025.

### M3 — Algoritmalar

- IPPO (de Witt vd. 2020) ve MAPPO (Yu vd. 2022), parametre paylaşımı + ajan kimliği.
- MAPPO makalesinin önerileri varsayılan alınır: değer normalizasyonu (Öneri 1), eleştirmene yerel + küresel özellik (Öneri 2), ≤10 epoch ve mini-batch bölmeme (Öneri 3), ε ≤ 0,2 (Öneri 4). Farklı seçilen her değer gerekçelendirilir.
- Zorunlu öğrenen baseline'lar: tek-ajan PPO (Milestone 0'), IPPO, MAPPO, MPLS tipi hiyerarşik sektör dağıtıcı (G3).

### M4 — İletişim kolları (AQ1, G1)

| Kol | Mekanizma | Kaynak |
|---|---|---|
| none | iletişimsiz | IPPO/MAPPO |
| attn | TarMAC tipi imza-sorgu-değer dikkati | Das vd. 2019 |
| gat | GAT mesaj işleyici (sektör + korelasyon grafı) | Veličković vd. 2018; MAGIC (Niu vd. 2021) |

**Eşit bütçe** (G1'in dar ifadesinin şartı): parametre sayısı kollar arasında ±%10; aynı HPO deneme sayısı; aynı tohumlar; aynı ortam adımı. Fang vd. (2023) finanstaki tek öncül olarak atıf alır.

### M5 — Güvenle ağırlıklı iletişim (AQ2, G2) — **yeniden çerçeveleme (Y7)**

Y7 nedeniyle "log-güven önseli" tek başına katkı değildir; DebUnc'un Attention-All ölçeklemesinin β=1 özel hâlidir. Savunulabilir katkı üç parçaya dayanır:

1. **Bağlam:** harici, denetimli ve **kalibre** (F) bir tahmin topluluğunun güveni; RL ile eğitilen sürekli mesajlarda (DebUnc: LLM tartışması, eğitim yok).
2. **Öğrenilen güç:** softmax(ℓⱼ + β·log cⱼ), β öğrenilir (β=1 DebUnc, β=0 güvensiz dikkat) — iki literatür mekanizması bu ailenin uç noktaları.
3. **Alıcı tarafı öz-güven geçidi:** alıcının kendi düşük güveni gelen mesajlara verilen toplam ağırlığı artırır.

Ablation kolları:

| Kol | Neyi sınar | Kaynak |
|---|---|---|
| attn (β=0) | güvensiz dikkat | Das vd. 2019 |
| β=1 sabit | DebUnc ölçeklemesi | Yoffe vd. 2024 |
| β öğrenilen | parça 2 | — |
| β öğrenilen + geçit | parça 2 + 3 (tam önerilen) | — |
| öğrenilen güvenilirlik geçidi | ADMAC'tan **esinlenen**, sürekli eyleme uyarlanmış (etiketsiz, uçtan uca) — Y8 | Yu vd. 2024 |
| karıştırılmış güven | güven sinyali bilgi taşıyor mu | lit. taraması §7.2 |
| **kalibresiz güven** | "kalibre" sıfatının değeri (F2 öncesi ham güven) | Guo vd. 2017 |

Sürüleşme bekçisi: c ∈ [c_min, c_max] kırpma ya da entropi geçidi. Bu bir **tasarım varsayımıdır**; dayanak olarak gösterilebilecek hakemli bir RL kaynağı bulunamadı (Farr vd. 2026 ön baskısı seçim kuralıyla elendi).

### M6 — Rejim (AQ3, G5)

Değerlendirme etiketi B6'daki kural tabanlı tanım; ikinci tanım HMM'nin **yalnızca filtrelenmiş** olasılıkları (Hamilton, Econometrica 1989; portföyde Nystrup vd., Quantitative Finance 2018; durum sayısı BIC ile). Rejim başına gün ve bağımsız dönem sayısı raporlanır. Shi & Hollifield (2025) motivasyon; iddia "zamansal rejimlere koşullu marjinal iletişim katkısı".

### M7 — Yorumlanabilirlik

Wiegreffe & Pinter (2019) testleri: tek biçimli dikkat baseline'ı, tohumlar arası dikkat varyansı. Karşı-olgusal mesaj maskeleme (bir ajanın mesajı sıfırlanınca portföy Sharpe'ındaki değişim). Jain & Wallace (2019) sınırlılık olarak anılır.

### M8 — Ön kayıt (analiz planı)

Nihai koşumlardan önce `docs/thesis/analysis-plan.md` dondurulur:

- **Hipotezler:** H1 iletişim > iletişimsiz; H2 güvenle ağırlıklı > güvensiz dikkat; H3 kalibre > kalibresiz güven; H4 iletişim katkısı rejime göre farklı.
- **Birincil metrik:** risksiz faiz (A8) üzeri, 20 bps maliyet sonrası, nominal TL günlük getiriden yıllık Sharpe. İkincil: Sortino, Calmar, MDD, dönüşüm.
- **Testler:** A3 (ikili Sharpe), A4 (MCS), A1 (DSR, N deneme kaydından), A5 (tohum dağılımı); aile başına BH-FDR (A7) — aile = bir hipotezin tüm karşılaştırmaları.
- **Tohum:** ön eleme 3–5, nihai ≥10; tohumlar önceden listelenir.
- **Durma kuralı:** ön elemeden sonra nihaiye taşınan yapılandırma sayısı önceden sabitlenir (Bailey & LdP 2014 "when should we stop testing").
- **Model seçim kuralı (R1; kullanıcı kararı 2026-11-14):** kontrol noktası seçiminin değeri önceden tanımlı bir etkenle ölçülür: argmax (doğrulamada fazla getiri Sharpe'ı) ↔ son nokta. Her kural kendi kaydıyla testte bir kez okunur ve N'ye sayılır. Negatif fazla getiri rejiminde seçim metriğinin davranışı (Faz 9 §9.12) bu karşılaştırmada raporlanır; yüksek faizli doğrulama yıllarında negatif ortalama olağan, `picks_agree` raporlanır.
- **Doğrulayıcı katlar (R1; kullanıcı kararı 2026-11-14):** 2021–2023. 2024–2026 R1 öncesi başarımla okundu (`eval_window_unexposed`, Faz 9 §9.11). Ön kayıttan sonra gelen veri gerçek sınama olarak ayrılır.

---

## 6. Literatürün plana etkisi — özet eşleme

> **2026-10-07:** Planın tüm bileşenlerinin kaynak eşlemesi artık [literature-table.md §2.1 kapsam matrisi](../thesis/literature-table.md)'nde (29 bileşen, K1/K2 kaynağı olmayan bileşen yok). Aşağıdaki tablo planın ilk sürümündeki eşlemedir. Seçim kuralıyla elenen üç satırın yerine geçen kaynaklar şöyle: Farr → tasarım varsayımı; Xue & Ye → Jiang vd. 2017; Nawathe → Moody & Saffell 2001.

| Kaynak (arşiv yolu) | Plana etkisi |
|---|---|
| `G_evaluation/2014_Bailey_DeflatedSharpe.pdf` | A1 altın değeri; Y1 |
| `G_evaluation/2008_LedoitWolf_SharpeTest_WP.pdf` | A3 yöntemi (HAC + studentize dairesel blok bootstrap) |
| `G_evaluation/2017_Bailey_PBO.pdf` | A6; C2 deneme kaydı |
| `G_evaluation/2021_Agarwal_StatisticalPrecipice.pdf` | A5; tohum protokolü |
| `E_forecast-leakage/2015_Diebold_DM-TwentyYearsLater.pdf` | A2: DM model değil tahmin karşılaştırır; kullanım sınırları |
| `E_forecast-leakage/2024_Yang_EMD-Leakage_SciRep.pdf` | B5 (SW yaklaşımı); Y9 |
| `E_forecast-leakage/2023_Hewamalage_ForecastPitfalls_DMKD.pdf` | B0 |
| `E_forecast-leakage/2017_Guo_Calibration_ICML.pdf` | F1–F3; M5 "kalibresiz" kolu |
| `C_marl-algo/2022_Yu_MAPPO.pdf` | M3 varsayılanları |
| `C_marl-algo/2020_deWitt_IPPO.pdf` | M3 IPPO baseline gerekçesi |
| `C_marl-algo/2024_Bettini_BenchMARL_JMLR.pdf` | M1 alternatif |
| `B_marl-comm/2024_Yoffe_DebUnc.pdf` | **Y7**, M5 yeniden çerçeveleme |
| `B_marl-comm/2024_Yu_ADMAC.pdf` | Y8, M5 kolu |
| `B_marl-comm/2019_Das_TarMAC.pdf`, `2018_Velickovic_GAT.pdf`, `2021_Niu_MAGIC_AAMAS.pdf` | M4 |
| `B_marl-comm/2023_Fang_IntentionComm_KDD.pdf` | M4 eşit bütçe tasarımı, G1 sınırı |
| ~~`B_marl-comm/2026_Farr_EpistemicHerding.pdf`~~ → tasarım varsayımı | M5 sürüleşme bekçisi |
| `J_single-drl/2017_Jiang_DRLPortfolio.pdf` (önceden Xue & Ye 2025) | M2 ağırlık politikası |
| Moody & Saffell 2001 (önceden Nawathe vd. 2024 §3.4) | Y2 (diferansiyel Sharpe) |
| `B_marl-comm/2024_Zhu_CommSurvey_JAAMAS.pdf` | M4 konumlandırma (iletişim tasarım boyutları) |
| `C_marl-algo/2017_Lowe_MADDPG.pdf`, `2018_Foerster_COMA.pdf`, `2021_Christianos_SePS.pdf` | M3: CTDE, kredi ataması, parametre paylaşımı |
| `E_forecast-leakage/2023_Kapoor_Leakage_Patterns.pdf` | B0 sızıntı taksonomisi |
| `H_xai/*` | M7 |

---

## 7. Kullanıcı kararı gereken konular

| # | Karar | Seçenekler | Öneri |
|---|---|---|---|
| 1 | **Tahmin golden'ı** — B1 (bfill) varsayılan hatta etkin; düzeltme golden'ı değiştirir | (a) B1 bayrakla, varsayılan eski; (b) varsayılanı çevir + golden'ı yenile | **Karar (2026-11-06): (b)** — varsayılan çevrilir, golden Milestone 0' ile birlikte bir kez yenilenir (Faz 10 K1; B1 ve B10 birlikte) |
| 2 | EVDS erişimi (B2) | API anahtarı mevcut mu / yeniden alınacak mı | **Kapandı (2026-11-06):** anahtar geçerli; sorun yanlış seri kodları ve ayrıştırma hatalarıydı (§9.3) |
| 3 | Tarihsel BIST-30 bileşenleri (E1) | Borsa İstanbul'dan elle temin / sınırlılık beyanı | Önce temin denenir; 2 hafta içinde bulunamazsa beyan |
| 4 | Ödül terimi (G1) | Yeni deneylerde `differential` / `ema_level` kalır, adı düzeltilir | Milestone 0' ön elemesinde ikisi denenir, seçim deneme kaydına girer |
| 5 | Makale 2 hedef tarihi (Aralık 2026) | Korunur / ertelenir | A + B + C ≈ 3–4 hafta + B5 hesaplama süresi; tarih B5 maliyet ölçümünden sonra yeniden değerlendirilir |
| 6 | ~~G2 metni (Y7)~~ | — | **Yapıldı (2026-10-07):** vision-and-scope v0.3 §2.5 G2 ve §3.4 II, DebUnc'u β=1 özel hâli olarak anıyor |

---

## 8. Bu turda doğrulanan yeni kaynaklar

arXiv API (2026-10-07) ya da Crossref ile doğrulandı. **İşlendi:** aşağıdakilerin hepsi [literature-table.md](../thesis/literature-table.md)'de seçildi ve vision-and-scope §9'a girdi.

- de Witt, C. S. vd. (2020). Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? arXiv 2011.09533
- Yu, C. vd. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. *NeurIPS 2022 Datasets & Benchmarks*. arXiv 2103.01955
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. arXiv 1706.04599
- Veličković, P. vd. (2018). Graph Attention Networks. *ICLR 2018*. arXiv 1710.10903
- Bettini, M. vd. (2024). BenchMARL: Benchmarking Multi-Agent Reinforcement Learning. *JMLR* 25
- Diebold, F. X., Mariano, R. S. (1995). Comparing Predictive Accuracy. *JBES* 13:253–263. DOI 10.1080/07350015.1995.10524599
- Diebold, F. X. (2015). Comparing Predictive Accuracy, Twenty Years Later. *JBES* 33:1. DOI 10.1080/07350015.2014.983236
- Harvey, D., Leybourne, S., Newbold, P. (1997). Testing the equality of prediction mean squared errors. *IJF* 13:281–291. DOI 10.1016/S0169-2070(96)00719-4
- Newey, W. K., West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55:703. DOI 10.2307/1913610
- Politis, D. N., Romano, J. P. (1994). The Stationary Bootstrap. *JASA* 89:1303–1313. DOI 10.1080/01621459.1994.10476870
- Benjamini, Y., Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS-B* 57:289–300. DOI 10.1111/j.2517-6161.1995.tb02031.x
- Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA* 88:365–411. DOI 10.1016/S0047-259X(03)00096-4
- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *JPM* 42(4):59–69. DOI 10.3905/jpm.2016.42.4.059
- Lim, B. vd. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *IJF* 37:1748–1764. DOI 10.1016/j.ijforecast.2021.03.012 (kodda kullanılan TFT'nin atfı)

---

## 9. Uygulama durumu

### 9.1 İlk kapı: V + A + C (2026-11-06, dal `feat/phase9-gate1-v-a-c`)

Faz 10 §8'e göre ilk kapı **V (veri bütünlüğü), A ve C**'dir. Gözlem sağlığı (Faz 10 K18: ADX ≡ 0, türbülans doygunluğu) aynı kapıda anılıyor, ama bu turda **yapılmadı**.

**Bu kapı RL'de tez sayısı üretmez.** hakem-2 incelemesinin sonucu (§9.2) şu: istatistik çekirdeği formül düzeyinde doğru, ama bu kapıdan sonra üretilen RL sonuçları da üç açık protokol maddesi taşıyor:
- **Y13 / B9:** test ölçeği test penceresinden kuruluyor. **Kapandı (§9.5), yalnız bu daldan sonraki koşumlar için.** Metriğe etkisi tanımlanamadı. Bedeli: eski ortamda test fiyat değerlerinin %98,9'u eğitim desteği dışında; deneme kaydı bunu `price_in_support` ile açık tutuyor.
- **Y15 / B7:** gösterge ısınma boşluğu `bfill` ile dolduruluyor. **RL'de kapandı (§9.7)**: kapı bfill bloğunu atıyor ve protokol bunu ölçüyor. Tahmin hattı ve CSV'nin yeniden üretimi K18'e bağlı.
- **K18:** ADX ≡ 0.

Bu yüzden bu kapıdan sonra üretilen RL sonuçları da "ön sonuç"tur. Deneme kaydı bu kayıtları `protocol_open` olarak işaretliyor ve kanıt olarak dışlıyor.

Aşağıdaki tüm sayılar şu komutların çıktısıdır (2026-11-06, panel son tarihi 2026-07-27):

```bash
python tests/test_eval_stats.py           # A — 50 kontrol (~45 sn)
python tests/test_data_integrity.py       # V — 27 kontrol
python tests/test_env_missing_mode.py     # V — 17 kontrol
python tests/test_rl_reproducibility.py   # C — 39 kontrol
python tests/test_prediction_regression.py  # golden: "[REGRESYON] GECTI (OK) - davranis korunmus." (2026-11-06)
```

| İş | Durum | Konum | Kabul ölçütü → ölçülen |
|---|---|---|---|
| A1 PSR + DSR | **Tamam** | `evaluation/sharpe.py` | s.10 örneği: SR₀ = 0,1132; DSR = 0,9004 (N=100), 0,9505 (N=46), 0,9505 (normal, N=88). Üçü de ±1e-4 içinde. N=1'de DSR tanımsız döner (eski kod PSR'yi DSR diye veriyordu) |
| A2 Diebold-Mariano | **Kısmen** | `evaluation/forecast_tests.py` | H₀, h=4, MA(3) hata, T=250, 2.000 sim: yeni **%5,95** (ölçüt %5 ± 1'in sınırında), eski formül %25,5. **AR(1) kayıp farkında ölçüt karşılanmıyor:** φ=0,5, T=500, 1.000 sim, `lag='auto'` ile %7,6 (MC SE ≈ %0,7); eski formül %13,5. Varsayılan `lag='h-1'`, h=1'de iid varyans demektir; otokorelasyonlu d_t'de eskisi kadar fazla reddeder. Bartlett+HLN birleşimi literatürde türetilmiş değil, ad hoc (hakem-2 #5e). Açık iş: uzun dönem varyans seçimini simülasyonla belirlemek. Sarmalayıcı artık `h` alıyor (haftalık hedef h=5) |
| A3 Ledoit-Wolf | **Tamam** (güç sınırlı) | `evaluation/sharpe_tests.py` | GARCH(1,1), eşit Sharpe. ρ=0,5, T=1.000: HAC **%5,75** (400 sim), bootstrap **%4,33** (600 sim, M=299, b=5). Gerçek kullanım koşulu ρ=0,9, T=500, 300 sim: HAC %5,00; bootstrap b=2 / 5 / 10 için %3,67 / %3,67 / %3,33 (hafif temkinli). b=5 bir **tasarım varsayımı**. **Güç düşük** (hakem-2 #4, hakemin ölçümü, T=495): ρ=0,5'te yıllık ΔSR 0,5 / 1,0 / 2,0 için %15 / %35 / %76. Tek test penceresinde "fark yok" sonucu çıkarılamaz → R1 + M8 MDE |
| A4 MCS | **Tamam** | `evaluation/model_sets.py` (`arch`) | Açıkça kötü model kümeden düşüyor, eşdeğer üç model kalıyor |
| A5 rliable | **Tamam** | `evaluation/seed_stats.py` | Kütüphane çıktısıyla birebir. **Not:** rliable 1.2.0 `random_state`'i kullanmıyor (yeniden örnekleme küresel `np.random.choice` ile). Sarmalayıcı küresel durumu geçici olarak tohumluyor ve geri yüklüyor |
| A6 PBO/CSCV | **Tamam** | `evaluation/overfitting.py` | 40 strateji, T=1.600, S=12. Saf gürültü: PBO 0,475. Orta sinyal (yıllık SR ≈ 1): 0,235. Güçlü sinyal (yıllık SR ≈ 4,8): 0,000 |
| A7 BH-FDR | **Tamam** | `evaluation/multiple_testing.py` (`statsmodels`) | El hesabıyla aynı |
| A8 Risksiz faiz | **Kısmen** | `evaluation/risk_free.py` | İlk kapıda tek kaynak modülü yazıldı ve yedek değerli seride açık hata verdi. **§9.3'te (B2) gerçek seriye bağlandı:** `risk_free.csv`, `funding_rate`. Eski sabitler (0,15 / 0,02 / 0) **bağlanmadı**: biri eğitilmiş RL modellerinin ödülünü, diğerleri pano sayılarını değiştirirdi. Bağlama D paketinde yapılacak |
| A9 Eski çağrılar | **Tamam** | `prediction/evaluator.py`, `app/services/model_analysis.py`, `evaluation/paired_tests.py` | `diebold_mariano_test` ve `compute_deflated_sharpe_ratio` yeni pakete yönlendi. `ttest_ind` yerine eşleştirilmiş t-testi geldi; HAC, durağan bootstrap ve LW seçenekleri de var |
| C1 RL tohumu | **Tamam** (GPU açık) | `app/api/routes/trading.py`, `evaluation/reproducibility.py` | Aynı tohumla iki CPU koşumu PPO ve A2C'de ödül eğrisi ve ağırlıklar açısından bit-eş. RTX 4060'ta küçük MLP ile iki GPU koşumu da bit-eş çıktı; bu tek ölçümdür, genel bir garanti değildir. **B9'da gözlenen sapma (§9.5):** GPU görünürken rota kabul kontrolü 5 koşumda 1 kez kaldı; aynı tohumla 10 GPU rota koşumu bit-eş; neden doğrulanmadı. Tez koşumları için deterministik algoritma zorlaması açık iş (§9.6 #5). Tohum meta dosyasına, manifeste ve deneme kaydına yazılıyor; istekte yoksa üretiliyor |
| C2 Deneme kaydı | **Kısmen** | `evaluation/trial_registry.py` | `results_dir()/trial_registry.jsonl`, getiriler `trial_returns/*.parquet`. Eğitim rotası `rl_train` (test bölümü), HPO `rl_hpo` (doğrulama bölümü) olarak yazıyor. Budanan ve başarısız denemeler de `status` ile kaydediliyor ve N'ye sayılıyor. N varsayılan olarak deneme başına (yapılandırma × tohum). V yalnızca getirisi olanlardan geliyor. Farklı kind / split / pencere karıştırılırsa hata. Her kayıt `protocol` ve `protocol_open` taşıyor; açık protokol varsayılan olarak dışlanıyor. Veri hash'i yapılandırma özetinin dışında. **Açık:** (1) rota ve HPO tek bir bölme tanımı paylaşmıyor ve rota her koşumda testi okuyor; seçim/sınama ayrımı R1'de (hakem-2 #3). (2) Tohum kuralı (`by='trial'` / `'config'`) M8'de sabitlenecek. (3) Kayıttaki SR'ler risksiz faiz düşülmeden; A8'e bağlı |
| C3 Veri anlık görüntüsü | **Tamam** | `evaluation/reproducibility.py` | Kanonik CSV baytlarının SHA-256'sı + `data/snapshots/*.parquet`. Manifest ve meta özet kaydını taşıyor; `verify_manifest_snapshots` veri değişince uyarıyor |
| C4 Ortam kaydı | **Tamam** | `prediction/manifest.py` | Her manifestte paket sürümleri, GPU ve torch belirlenimcilik bayrakları |
| V Veri bütünlüğü (Faz 10 K17) | **Kısmen** | `data/integrity.py`, `env/trading_env.py` | Fiyat, hayalet gün, bayat hücre ve pencere tamam. Göstergelerin `bfill` ısınması (B7) ve ADX (K18) yalnızca **raporlanıyor**, düzeltilmedi. Ayrıntı aşağıda |

**V — ölçülen panel durumu** (`stock_data_with_indicators.csv`, eğitimin okuduğu dosya):

| | Tam panel 2000-05-10..2026-07-27 | RL penceresi 2013-06-01.. |
|---|---|---|
| Fiyat artefaktı satırı | 235 (ohlc 170, seviye sıçraması 55, aşırı getiri 9, YTL geçişi 1) | **0** |
| Hayalet gün (≥ %90 satır hacim 0 ve O=H=L=C) | 251 | 72 → atılıyor |
| Bayat hücre (tek sembol hacim 0 ve O=H=L=C) | 781 | 39 → atılıyor |
| Eksik hücre oranı | %14,19 | %0,076 |
| Sabit sütun | `adx` | `adx` (K18, açık) |
| Gösterge ısınma bloğu (`bfill`) | — | ODAS CCI 10 seans, PGSUS CCI 17 seans (B7, açık) |
| \|r\| > %20 hücre (artefakt değil) | — | 13 (ör. 2023-02-15 deprem sonrası açılış); sınıflandırılmadı |

**Hayalet gün tanımı inceleme sonrası değişti.** İlk sürüm "= önceki kapanış" şartı da arıyordu. Bu yüzden 2013-10-15 (Kurban Bayramı) kaçmıştı: o gün 30 sembolün hepsinde hacim 0, ama yfinance 19 sembolde düzeltilmiş fiyatı farklı yazmış. Gün panelde kısmi bir gün olarak kalıyor ve geniş tablo sırasını bozuyordu (hakem-2 #11'in kökü). İlk sürümün sayıları 222 / 69 / 1.035 / 60 idi ve geçersizdir.

Hakem-2'nin plan incelemesinde ölçtüğü 88 hayalet gün ile buradaki 251 arasındaki farkın nedeni **doğrulanmadı**: hakemin komutu ve eşiği kayıtlı değil. Raporda bu modülün tanımı kullanılır.

Tarayıcı, **sürekli yüksek ama sıçramayan** seviyeleri yakalamıyor (TUPRS'un 2000–2003 arası > 10.000 TL satırları). Bu satırlar pencerenin dışında kalıyor.

**Davranış değişiklikleri** (Faz 10 §5 tablosuna göre):
- Eğitim rotası ve HPO artık 2013-06'dan başlayan pencere ve `missing_data_mode='own_last'` ile çalışıyor.
- `TradingEnv`'in varsayılanı `legacy`. Eski davranış, eksik hücrede komşu sembol kopyası dahil bit-eş korunuyor; bu `test_env_missing_mode.py` ve env bekçi testleriyle sabitlendi.
- Bu kapıdan önce üretilmiş RL sonuçları (`results/*.json`) 2000–2018 penceresi, komşu sembol kopyası ve tohumsuz eğitimle üretildi. Bu kapıdan **sonra** üretilenler de Y13/Y15/K18 açık maddelerini taşıyor. İkisi de tezde ancak sınırlılık ya da ön sonuç olarak anılabilir.
- Tahmin golden'ı etkilenmedi: `python tests/test_prediction_regression.py` → "[REGRESYON] GECTI (OK) - davranis korunmus." (2026-11-06). Değişen fonksiyonları (DM, DSR) hiçbir tahmin akışı çağırmıyor (grep: yalnızca `build_prediction_chart_data`, `build_accuracy_trend_data` ve `compute_rolling_metrics` kullanılıyor).

### 9.2 hakem-2 incelemesi (kod, 2026-11-06) — karar: Büyük revizyon

Bu incelemede formül düzeyinde sorun bulunmadı (DSR elle yeniden hesaplandı; LW gradyanı, QS, Andrews ve HLN kontrol edildi). İtirazlar sonuçların hangi koşulda üretildiğine ve kayıt protokolüne yönelik.

| # | Cid. | İtiraz | Yanıt |
|---|---|---|---|
| 1 | RED | Bu kapıdan sonraki RL test metrikleri de test-penceresi ölçeğiyle (Y13) üretiliyor; kayıt bunu işaretlemiyor | **Kabul.** Kayıtlara `protocol` / `protocol_open` eklendi; açık protokol varsayılan olarak dışlanıyor. §9.1'e "RL'de tez sayısı yok" notu girdi. Asıl düzeltme B9 |
| 2 | BÜYÜK | DSR N/V türetmesi yanlı: budananlar yok, tohum ortalaması V'yi küçültüyor, split'ler karışıyor, veri hash'i config'te | **Kabul.** Budanan ve başarısız denemeler N'ye giriyor; varsayılan `by='trial'`; kind / split / pencere karışımı hata veriyor; hash config dışına alındı. Tohum kuralı M8'e, risksiz faiz A8'e bağlı. Durum "Kısmen" |
| 3 | BÜYÜK | Tek bölme tanımı yok; rota testi her koşumda okuyor; DSR hangi örneklemde uygulanacak belirsiz | **Kabul, R1'e** (Faz 10 K20): bölme tek kaynaktan; seçim doğrulamada; test yapılandırma başına bir kez |
| 4 | BÜYÜK | Tek 495 seanslık pencerede güç düşük | **Kabul, R1 + M8.** Birleştirilmiş örneklem dışı seri, MDE ön kayıtta; p yerine güven aralığı |
| 5 | BÜYÜK | A2 ölçütü karşılanmadı ama "Tamam" yazıyor; h=1 sabit; HLN+Bartlett ad hoc | **Kabul.** Durum "Kısmen"; test ölçüt yerine "ilerleme kontrolü" diye adlandırıldı; sarmalayıcıya `h` eklendi |
| 6 | BÜYÜK | Tek pencerede tohum güven aralığı piyasa belirsizliğini ölçmez | **Kabul.** `seed_stats` docstring'i; tezde ayrı raporlama (M8) |
| 7 | BÜYÜK | Ortamda nakde faiz yok, ama A8 politika faizini risksiz faiz alıyor | **Kabul, açık iş:** nakit faiz tahakkuku ya da açık varsayım beyanı; risksiz faiz serisi seçiminin kaynakla gerekçelendirilmesi (G2 / A8 ile birlikte) |
| 8 | KÜÇÜK | Blok boyu duyarlılığı ve gerçek koşul testte yok; doğrulanmamış "LW HAC liberal" ifadesi | **Kabul.** ρ=0,9, T=500, b ∈ {2, 5, 10} teste eklendi; ifade kaldırıldı; arşiv yolu notu eklendi |
| 9 | KÜÇÜK | Pencere içinde gösterge `bfill` ileriye bakışı (ODAS CCI) | **Kabul.** `indicator_warmup_blocks` raporlıyor; düzeltme B7 |
| 10 | KÜÇÜK | \|r\| eşiği gevşek; uç hareketler açıklanmamış | **Kısmen.** `large_moves` 13 hücreyi listeliyor; eşik artefakt için %50'de kalıyor; sınıflandırma (sermaye artırımı vb.) açık |
| 11 | KÜÇÜK | Temizlenmiş panelde `unstack` tarih sırası bozuk | **Kabul.** Kök neden kaçan hayalet gündü (yukarıda); ayrıca `remove_unused_levels`; teste eklendi |
| 12 | KÜÇÜK | Aynı kapanışta karar ve yürütme | **Kabul, K12** (Faz 10; yeni ortamda t+1 açılış) |
| 13 | KÜÇÜK | Küresel RNG tohumlaması iş parçacığı güvenli değil | **Kabul.** Kilit eklendi; sınırı docstring'de |
| 14 | KÜÇÜK | PBO sinyal testi önemsiz derecede güçlü | **Kabul.** Orta sinyal (yıllık SR ≈ 1) senaryosu eklendi: 0,235 |
| 15 | KÜÇÜK | Golden iddiasının kanıtı yok | **Kabul.** Komut ve çıktı §9.1'de |

### 9.3 B2 + B3 + B1 + A8 — makro veri ve risksiz faiz (2026-11-06, dal `feat/phase9-b2-evds`)

**Teşhis (B2).** Kullanıcı EVDS'de politika faizini 37 olarak görüyordu; repodaki 50 hiç EVDS'den gelmemişti. Anahtar geçerliydi: TÜFE endeksi aynı anahtarla geldi. Sorun dört ayrı hataydı:

| Seri | Hata | Sonuç (son çekim 2026-07-28) | Düzeltme |
|---|---|---|---|
| Politika faizi | Kod `TP.YSSK.A01` EVDS'de yok; API bağlantı hatası döndürüyor | `macro_fetcher` her çekimde sabit **50,0**'a düşüyordu (`fallback_constant_50.0`) | `TP.BISPOLFAIZ.TUR` (BIS, aylık; `get_series('bie_bispolfaiz')`). Son değer 37,0 (2026-09) |
| TÜFE | (a) EVDS sütunu `TP_FG_J0` (alt çizgili) döner, kod `TP.FG.J0` arıyordu; (b) aylık `Tarih` 'YYYY-M', kod `%d-%m-%Y` bekliyordu; (c) `TP.FG.J0` (2003=100) 2026-01'de **bitiyor** | `fallback_zero` | `TP.TUKFIY2025.GENEL` (2025=100, 2005'ten); ayrıştırma `_evds_series` / `_parse_evds_dates` |
| ÜFE | Kod `TP.FG.G0` EVDS'de yok | `fallback_zero` | `TP.TUFE1YI.T1` (Yurt İçi ÜFE) |
| Günlük seriler | EVDS tek istekte ≤ 1000 satır veriyor ve fazlasını aralığın **başından** kesiyor (2013-01..2016-12 isteği 2013-03-04'ten başladı) | — | 730 günlük parçalar; 1000 satır dönen parça ikiye bölünüyor |

Seri kodları EVDS kataloğundan (`get_sub_categories`, `get_series`) bulundu. Değerler gözle kontrol edildi; aşağıdakiler `fetch_macro_data()` çıktısının yıl sonu ve ay değerleri (2026-11-06):
- politika faizi Aralık 2018 = 24, Aralık 2022 = 9, Kasım 2024 = 50, **Aralık 2024 = 47,5**, Eylül 2026 = 37;
- Kasım 2022 TÜFE yıllık %84,39 ve ÜFE yıllık %136,02.

Resmî kaynaklarla tek tek karşılaştırma yapılmadı. Hakem-2 ise TÜFE için iki seriyi ölçtü: `TP.TUKFIY2025.GENEL / TP.FG.J0` oranı sabit (std/ortalama 3,6e-6). Yıllık değişim farkı 2010 sonrası 0,000 puan, Kasım 2022'de ikisi de 84,39. 2026-02 sonrası eski seri bittiği için karşılaştırılamıyor.

**B3 — açıklanma gecikmesi.** Aylık bir değer, açıklanmadan önceki günlere yazılmıyor. Eski kod ay *m*'nin TÜFE'sini ayın 1'ine yerleştirip ileri dolduruyordu; yani değer açıklanmasından yaklaşık 5 hafta önce görünüyordu.

| Seri | Ay *m* değeri ne zaman görünür | Dayanak |
|---|---|---|
| TÜFE / ÜFE yıllık % | ay *m+1*'in **6'sından** itibaren | **Tasarım varsayımı.** TÜİK genelde ayın 3'ünde açıklıyor; hafta sonu kaymasına pay bırakıldı. Resmî yayın takvimi kullanılmadı |
| Politika faizi (BIS aylık) | ay *m+1*'in **1'inden** itibaren | **Tasarım varsayımı.** Aylık seri ay içindeki karar tarihini taşımıyor, bu yüzden ay içindeki değişiklik bir ay gecikmeli görünüyor |

**B1 — `bfill` kaldırıldı.** Hem `fetch_macro_data`'dan hem `fetch_incremental`'dan çıktı.
- Zorunlu sütunların hepsi dolmadan önceki baş satırlar **atılıyor**. İnceleme sonrası gerçek çekimde 1 satır atıldı; ilk satır 2018-01-02.
- Yıllık enflasyon için endeks **14 ay** erken çekiliyor. İlk sürümde bu 13 aydı ve bir aylık kayma yüzünden ilk 5 gün boş kalıyordu (hakem-2 B2 #7).
- Dosya bitiş tarihinden sonrasını içermiyor (hakem-2 B2 #4).

**A8.** `data/macro/risk_free.csv` (`fetch_risk_free`) ayrı bir dosya:
- `funding_rate`: TP.APIFON4, TCMB Ağırlıklı Ortalama Fonlama Maliyeti; iş günü, 2011-01-03..2026-11-06, 3.983 satır, 7 günden uzun boşluk yok.
- `policy_rate`: B3 kuralıyla günlüğe yayılmış aylık seri.

`evaluation.risk_free` varsayılan olarak `funding_rate`'i kullanıyor. Seçim bir **tasarım varsayımıdır**: seri günlük ve RL penceresini kapsıyor. Bunun kaynakla gerekçelendirilmesi ve nakit faiz tahakkuku açık iş olarak duruyor (§9.2 #7). Veri sayfasındaki makro güncellemesi bu dosyayı da yeniliyor; yedek değer kullanılırsa panoda kırmızı uyarı çıkıyor.

**Kabul** (`python tests/test_macro_pit.py`, ağ yok, **33/33**). Sahte EVDS, gerçek API'nin ölçülen davranışlarını taklit ediyor:
- alt çizgili sütun ve 'YYYY-M' tarih biçimi;
- baştan kesen 1000 satır sınırı;
- **BIS yayın gecikmesi** (son 65 gün yok).

Sahte TÜFE/ÜFE'nin aylık değişimi her ay farklı. Kontroller:
- **B3:** hiçbir ayın TÜFE'si açıklanmadan önce görünmüyor. Bir **öz-test**, "ay *m* değeri ay *m*'nin 1'inde" mutasyonunun bu kontrolü kırdığını gösteriyor (hakem-2 B2 #2).
- **B1 — doğrudan kontrol:** ilk satırın TÜFE'si başlangıçtan *önce* bilinen değer; *sonra* gelecek değer değil. İlk sürümdeki kesme değişmezliği testi `bfill`'i **yakalamıyordu**, çünkü baştaki `bfill` bitiş tarihinden bağımsız (hakem-2 B2 #3). Kesme değişmezliği kontrolü yerinde duruyor, ama B1'in kanıtı o değil.
- **Bitiş tarihi:** dosya bitiş tarihini aşmıyor.
- **Artımlı çekim:** BIS gecikmesi altında yedeğe düşmüyor, gün kaybetmiyor. Yedek değerli delta dosyaya yazılmıyor (hakem-2 B2 #1).
- **Hata yolu:** kalite bayrağı ve `strict_data`. Bayraklı dosya tam çekimle yenileniyor.
- **Risksiz faiz:** t−1 oranından hesaplanıyor.

Gerçek çekimde kalite bayrağı boş, NaN yok. THYAO için özellik üretimi gerçek makroyla çalıştı.

Tahmin golden testi geçti, ama **bu değişiklik için kanıt sayılmaz**: golden sentetik makro kullanıyor. K1'deki "B1 golden'ı değiştirir" öngörüsü gerçek veri hattı için geçerli; sentetik golden için değil (hakem-2 B2 #13). Golden yenilemesi yine Milestone 0' ile yapılacak.

**Etkileri ve açık kalanlar:**
- **Bu tarihten önce üretilmiş her şey sabit faiz 50 ve sıfır enflasyonla üretildi:** tahmin modelleri, faz 2 RL modelleri, `macro_data.csv`'ye dayalı her sonuç. Makro özellikler bu sonuçlarda bilgi taşımıyor; hepsi sınırlılıktır.
- **Faz 2 RL modelleri** gözlem vektöründe sabit `tanh(50/50)` görerek eğitildi. Yeni makro dosyasıyla canlı kararları **değişir** (dağılım dışı gözlem).
- **Makro 2018'den başlıyor, RL penceresi 2013-06'dan.** Faz 2 ortamı 2013–2017 arasında makroyu 0 görüyor (önceden de böyleydi).
- Resmî TÜİK yayın takvimi kullanılmadı; politika faizinin karar tarihli günlük serisi yok. İkisi de tasarım varsayımı.
- Eski risksiz faiz sabitleri (0,15 / 0,02 / 0) bağlanmadı; bağlama D paketinde yapılacak.

### 9.4 hakem-2 incelemesi (B2/B3/B1/A8, 2026-11-06) — karar: Büyük revizyon

| # | Cid. | İtiraz | Yanıt |
|---|---|---|---|
| 1 | BÜYÜK (birleştirmeyi engeller) | Artımlı çekim gerçek EVDS'de politika faizini yine 50'ye düşürüyor (BIS ~5 hafta gecikmeli; kısa pencere boş dönüyor) ve gün kaybediyor | **Kabul, düzeltildi.** EVDS 120 gün geriye bakıyor; yedek değerli delta yazılmıyor; gecikmeli sahte EVDS testi eklendi |
| 2 | BÜYÜK | B3 testi içi boş (geometrik sahte TÜFE → YoY sabit) | **Kabul, düzeltildi.** Değişken sahte seri + mutasyon öz-testi |
| 3 | BÜYÜK | Kesme değişmezliği `bfill`'i yakalamıyor; CLAUDE.md iddiası yanlış | **Kabul, düzeltildi.** Doğrudan ilk değer kontrolü eklendi; belgeler düzeltildi |
| 4 | BÜYÜK | Olay zamanlı ama vintage değil: BIS değeri geriye yazılıyor, eğitimde ay *m−1*, canlıda *m−2* görünüyor; bitiş sonrasına satır | **Kısmen.** Bitiş kesimi eklendi. **Açık:** vintage / son gözlenen ay kaydı; canlıda eksik ay uyarısı |
| 5 | BÜYÜK | Aylık ay sonu politika faizi ölçüm hatası taşıyor; 2018'in 1. yarısında 1 haftalık repo fiili fonlamayı temsil etmiyor (8 ↔ 12,75–16,5); 471 günde fark > 3 puan | **Kabul, açık iş:** duruş özelliği için günlük TP.APIFON4 (t−1) seçeneği + duyarlılık analizi; o zamana kadar sınırlılık |
| 6 | BÜYÜK | TP.APIFON4 risksiz faiz olarak gerekçesiz (TCMB fonlama maliyeti, yatırılabilir getiri değil) | **Kabul, açık iş:** TLREF (`TP.BISTTLREF.ORAN`, 2018-12'den) ile duyarlılık; kaynakla gerekçe (`/kaynak-dogrula`); nakit faiz (§9.2 #7) |
| 7 | KÜÇÜK | YoY erken çekimde bir aylık kayma (13 → 14) | **Kabul, düzeltildi** |
| 8 | KÜÇÜK | 2025=100 serisinin varsayımı ölçülmemiş | **Kabul.** Hakemin ölçümü §9.3'e yazıldı; 2026-02 sonrası TÜİK bülteniyle karşılaştırma açık |
| 9 | KÜÇÜK | `CPI_RELEASE_DAY=6` ölçülmemiş | **Kabul, açık iş:** 2018–2026 TÜİK yayın tarihleri |
| 10 | KÜÇÜK | "2024 sonu 50" yanlış; risk_free'de `policy_rate`'in ilk 21 satırı NaN | **Kabul, düzeltildi** (Aralık 2024 = 47,5; politika bir ay erken çekiliyor) |
| 11 | KÜÇÜK | Çapraz varlık verisinde `ffill().bfill()` duruyor; "KAPANDI" kapsamı aşıyor | **Kabul, düzeltildi** (`prediction_service._fetch_cross_asset`); ifade daraltıldı |
| 12 | KÜÇÜK | RL ortamında makro `<= current_date`: aynı günün TRY=X kapanışı (BIST kapanışından sonra olabilir; doğrulanmadı) | **Kabul, açık iş** (env değişikliği → davranış dondurma kararı) |
| 13 | KÜÇÜK | Golden'ın geçmesi kanıt değil; K1 gerekçesiyle çelişki | **Kabul.** §9.3'te açıklandı |
| 14 | KÜÇÜK | Eski dosya yalnızca kalite bayrağından tanınıyor; `pit_version` yok | **Kabul, açık iş** |

**Sıradaki:** B (+B7–B10, B9 bu kapının açık protokolünü kapatır; K1: golden Milestone 0' ile bir kez), K18 gözlem sağlığı, R1 (tek bölme, walk-forward), E, F, G (+ §9.2 #7 nakit faizi), D; ardından P ve Milestone 0'.

### 9.5 B9 — val/test ortamının fiyat ölçeği (2026-11-06, dal `feat/phase9-b9-price-stats`)

**Sorun (Faz 10 Y13, §8 #16).** Test ortamı fiyat z-skorunu test penceresinin kendi ortalama ve std'siyle kuruyordu. Bu yüzden t gözlemi t sonrası test fiyatlarına bağlıydı. HPO'nun doğrulama ortamı ve beş betik de aynı şeyi yapıyordu.

**Durum: kapandı, ama yalnızca bu daldan sonraki koşumlar için.** Sızıntının test metriğine etkisi **tanımlanamadı** (aşağıda "Etkisi"). B9 kolunun kendisi de dağılım dışı: eski ortamda test fiyat değerlerinin %98,9'u eğitim desteğinin dışında.

**Değişiklik.**
- **Env:** `TradingEnv(price_stats=None)` opt-in parametre.
  - `None` eski davranıştır: ölçek ortamın kendi df'inden hesaplanır. Bu tanım eğitim ortamı için doğrudur.
  - Verilirse denetlenir ve kopyalanır. Ortamdaki her sembol için sonlu bir ortalama ve pozitif std gerekir; yoksa `ValueError` atılır. Sessizce kendi penceresine ya da 50/50'ye düşmez.
  - Ortam ölçeğin nereden geldiğini `price_stats_source` (`own_window` / `supplied`) ile taşır.
  - Üretici ortam, ölçeği yedeğe düşen sembolü işaretler (`price_stats_fallbacks`: std tabanı 1,0 ya da 50/50). Değerler değişmez; işaret meta dosyasına yazılır.
- **Çağıranlar:**
  - Eğitim rotası: test ortamı `temp_env.get_price_stats()`'i alır; bu, meta dosyasına yazılan ölçeğin aynısıdır.
  - HPO: doğrulama ortamı eğitim ortamının ölçeğini alır.
  - Betikler: `train_a2c_phase1.py`, `optimize_reward_weights.py`, `ab_test_rewards.py`, `debug_model_actions.py`.
  - Eski test betikleri: `tests/test_all_algorithms.py`, `test_ppo.py`, `test_model_manual.py`, `test_env_pytest.py`.
  - Sarmalayıcılarda (`create_env`, `create_env_with_weights`, HPO `create_env`) `price_stats` **zorunlu anahtar**. Eğitim çağrısı `None`'ı açıkça yazıyor, böylece varsayılan sızıntılı yol olamıyor.
- **Ölçek uydurulmaz (K13).** Meta'sında ölçek olmayan eski model raporda, `debug_model_actions.py`'de ve `test_model_manual.py`'de **atlanır ya da durdurulur**.
- **Akademik rapor D5'e kadar varsayılan olarak duruyor** (çıkış kodu 2). Gerekçeleri ve atlanan modeller ayrı dosyaya yazılır; `--allow-invalid` ile çalışır (aşağıda Gözlemler).
- **Protokol (rota ve HPO ortak: `rl_eval_protocol`)** ölçeğin **kökenini** ölçer. `train_window` için üç koşul var:
  - eğitim ortamı ölçeği kendi penceresinden hesaplamış olmalı;
  - o pencere değerlendirme başlamadan bitmeli;
  - değerler birebir aynı olmalı ve sembol yedeğe düşmemiş olmalı.

  Aksi hâlde etiket `own_window`, `overlapping_window`, `supplied_other`, `supplied_unverified` ya da `train_window_fallback` olur; hepsi açık madde.
- **Yeni protokol maddeleri:**
  - `price_in_support`: değerlendirme fiyat değerlerinin en fazla %1'i eğitim desteği dışında olabilir (tasarım varsayımı). Ölçüm `evaluation/observation_health.py` ile yapılır.
  - `hyperparams_clean`: hiperparametreler, kaydı açık ya da bulunamayan bir HPO çalışmasından mı geliyor. Bilgi alanları `hyperparams_source` ve `hyperparams_open_items`.
  - `indicators_present`: gösterge sütunları veride mi (Y21).
  - Ayrıca ADX sütunu yoksa `adx_constant=True` sayılıyor; eski hesap bu durumda "sabit değil" diyordu.
- **Gözlem sağlığı** (`evaluation/observation_health.py`):
  - Piyasa blokları (fiyat, hacim, göstergeler) politikadan bağımsızdır; env'in kendi gözlem fonksiyonuyla eğitim ve değerlendirme penceresinde ayrı ölçülür.
  - Blok başına kırpılan oran, sabit boyut oranı ve eğitim desteği dışı oran raporlanır.
  - Pozisyon bloğu modelin gördüğü gözlemden ayrı ölçülür.
  - Rota bunu manifeste, deneme kaydına ve `_metrics.json`'a protokolle birlikte yazar. `price_in_support` açıksa panoda uyarı çıkar.

**Kabul** (`python tests/test_price_stats_window.py`, ağ yok, CPU, **103/103**):
- **Env, t ∈ {0, 15, T−2}:** test penceresinde t sonrası fiyatlar değiştirildi (günlük %2 sürükleme). t'ye kadar (dahil) **tüm gözlem vektörü bit-eş** kaldı; t sonrası gözlem değişti, yani ölçüt boş değil. Eski davranışta t=0 gözlemi bile değişiyor (max |fark| 1,34): ileriye bakış sayısal olarak gösterildi.
- **Rota uçtan uca** (`_run_training_inner`, sentetik panel, sabit tohum), t ∈ {0, 6}:
  - Piyasa bloğu t'ye kadar bit-eş. Bu kıyas politikadan bağımsız olduğu için GPU belirlenimsizliğinden etkilenmez.
  - Tüm gözlem ve kayıttaki test getirileri de t'ye kadar bit-eş (CPU). Fark çıkarsa ilk sapmanın adımı ve bloğu raporlanıyor.
  - Kayıt, manifest, meta ve metrik dosyası aynı protokolü ve sağlığı taşıyor.
- **Hiperparametre kökeni:** açık bir HPO çalışmasından gelen hiperparametreyle koşan rotanın kaydı açık kalıyor ve maddeleri listeleniyor. Kaydı bulunamayan çalışmada `None` (açık) yazılıyor.
- **HPO `objective`:** doğrulama ortamı eğitim ölçeğinde; kayıt `train_window`. Kayıt destek ölçümünü taşıyor ve ham OHLCV yüzünden açık.
- **Mutasyon öz-testleri** (ölçüt kırılmalı):
  - ölçeği yok sayan env: her t'de kırılıyor;
  - ölçeği değerlendirme penceresinin ilk 16 gününden kuran env (hakem-2'nin mutantı): t=15'i geçiyor, **t=0'da yakalanıyor**; etikette `supplied_other`;
  - ölçeği kopyalamayan env;
  - ölçeği test ortamına geçirmeyen rota (B9 öncesi hâli): kaydı kendiliğinden `own_window` + açık.
- **Köken:** üç kurgu da temiz çıkmıyor:
  - test penceresinin kendi ölçeği "eğitim ölçeği" diye verildiğinde (hakem-2 K7);
  - eğitim penceresi değerlendirmeyle kesiştiğinde;
  - ölçek yedeğe düştüğünde.
- **Varsayılan yol değişmedi.** main @ 78edf06'daki env `git show` ile yüklendi ve aynı kütüphanelerle 11 yapılandırmada karşılaştırıldı; hepsi bit-eş. Yapılandırmalar: tam ve delikli panel × `legacy`/`own_last`, faz 2 PSR + temel/makro, faz 3, ATR, ATR + Kelly, önbelleksiz `.loc` yolu, ham OHLCV, komisyon/lot. Git yoksa dört dondurulmuş SHA-256 özeti yedek.
- **AST bekçisi:**
  - Kapsam: adı held-out olan df ile kurulan `make_env`, `TradingEnv` ve adında "env" geçen sarmalayıcı çağrıları. Held-out ad biçimleri: `test_df`, `splits.test`, `data['test']`, `df.loc[test_mask]`.
  - Kural: her çağrıda `price_stats` olmalı ve değeri `None` sabiti olmamalı.
  - Sonuç: 31 çağrı bulundu, 17'si test dosyasının kendisinde, **14'ü dışında**. Hepsi uyumlu.
  - Bekçi örnek kaynakta ölçeksiz, `None`'lı, sarmalayıcılı, nitelikli ve maskeli çağrıları yakalıyor.
- **Sarmalayıcıyla env kuran betikler** davranışla sınanıyor (`train_a2c_phase1`, `ab_test_rewards`, `optimize_reward_weights`, rapor). Zorunlu anahtar eksikse `TypeError` alınıyor.

**Bedeli — doygunluk ve destek dışılık.** Komut: `python scripts/analysis/measure_price_scale.py --mode saturation`. Çıktı: [measurements/phase9-b9/price_scale_saturation.json](measurements/phase9-b9/price_scale_saturation.json) (panel SHA-256 `73dee836…`; commit 96f2d8b, temiz ağaçta — `git_dirty=false`; iki ölçüm de önceki koşumla bit-eş). Panel 2013-06-03..2026-07-27, 30 sembol. Bölmeler: eğitim 2013-06-03..2022-08-11 (2.303 gün), doğrulama 2022-08-12..2024-08-05 (494), test 2024-08-06..2026-07-27 (494).

| Test penceresi | Eğitim ölçeğiyle (B9) | Kendi ölçeğiyle (eski) |
|---|---|---|
| O/H/L/C z'nin \|z\| ≥ 10 oranı | **%74,7** (medyan z 25,1) | %0,0 |
| Fiyat değerlerinin eğitim [min, maks] desteği dışında kalan oranı | **%98,9** | %10,8 |
| Pencere boyunca sabit fiyat boyutu | %63,3 (19/30 sembolün dört fiyat özelliği her gün +10) | %0 |
| Sabit gösterge boyutu | %20 (ADX) | %20 (ADX) |

Doğrulama penceresinde eğitim ölçeğiyle kırpılan oran %53,1, destek dışı oran %92,9.

Neden: TL fiyatları eğitim penceresi ortalamasının onlarca std üstünde. Faz 10 K18 bunu öngörmüştü ("B9 eski ortamda doygunluk yaratır; beyan edilir"). Eski ölçekte de destek dışı oran %10,8; yani bu kol da dağılım içinde değil.

**Etkisi — test metrikleri: tanımlanamadı.** Komut: `python scripts/analysis/measure_price_scale.py --mode effect --algos A2C,PPO --seeds 1,2,3 --steps 50000`. Çıktı: [measurements/phase9-b9/price_scale_effect.json](measurements/phase9-b9/price_scale_effect.json). Rota varsayılanlarıyla, CPU'da, tohum başına bir model eğitildi. **Aynı model** test penceresinde iki ölçekle değerlendirildi.

| | Fazla getiri Sharpe'ı, eski ölçek (± std, n=3) | Fazla getiri Sharpe'ı, eğitim ölçeği (± std, n=3) | Eşleştirilmiş fark, %95 GA | İşlem sayısı, eski → eğitim |
|---|---|---|---|---|
| A2C | −0,78 ± 0,12 | −0,71 ± 0,26 | +0,07 [−0,87; 1,01] | 610 → 2.632 |
| PPO | −0,76 ± 0,01 | −0,64 ± 0,22 | +0,12 [−0,42; 0,66] | 5.564 → 905 |
| Havuz (n=6) | −0,77 ± 0,08 | −0,67 ± 0,22 | +0,10 [−0,20; 0,39] | |

- **Risksiz faiz.** Fazla getiri Sharpe'ı `evaluation/risk_free.py` ile hesaplandı (funding_rate, TP.APIFON4, t−1 oranı). Kolların σ'sı farklı olduğu için rf=0 Sharpe farkı rf'den bağımsız değil (hakem-2 #11).
- **Bağlam:**
  - Aynı pencerede risksiz birikim +%104,6.
  - Komisyonsuz eşit ağırlıklı al-tut +%56,4 (fazla getiri Sharpe'ı −0,40).
  - 12 değerlendirmenin 12'sinde fazla getiri Sharpe'ı negatif.
- **Okuma:**
  - Bu tasarım sızıntının metriğe etkisini **ayırt edemez.** İki kol iki yönden farklı: biri gelecek bilgisini kaldırıyor, öteki girdiyi eğitim desteğinin dışına çıkarıyor. Ölçülen fark bu ikisinin toplamı.
  - Politikalar 50k adımla (~22 bölüm) eğitildi. Sızıntıyı kullanacak kadar öğrenmemiş bir politikada şişirme beklenmez; ölçüm "etki yok" yönüne yanlı.
  - GA'lar sıfırı içeriyor; tablo ne yön ne büyüklük belirliyor. Bu tablo "eski sonuçlar şişirilmemiş" diye **okunamaz**.
  - Sızıntının kanıtı kabul testindeki t=0 gözlem farkı.
  - Davranış ölçeğe duyarlı: PPO'nun işlem sayısı eğitim ölçeğinde düşüyor (fark GA'sı [−6.076; −3.242], n=3).
  - Eski kolda tüm vektörde kırpılan değerler fiyat bloğundan değil, pozisyon bloğundan geliyor (≥ 1.000 lot; pozisyon özelliklerinin %14–26'sı).
- **Eski ortamda ölçeğe bağlı fiyat özelliğiyle üretilen B9 sonrası test sayısı tez sayısı olamaz.** K18 bu ölçeği değiştirmiyor; eski ortama yalnızca ADX bayrağı getiriyor. Sınırlama artık **makine tarafından uygulanıyor**: `price_in_support` açık kaldıkça kayıt kanıt sayılmaz, B7 ve K18 kapansa bile. Destek içinde kalan, sızıntısız bir üçüncü kol (yalnızca ≤ t veriyle kayan ya da genişleyen z) karar maddesi olarak önerildi: Faz 10 §6 **K25**.

**Yeni bulgu Y21 — HPO başka bir gözlem fonksiyonunda ve başka bir evrende ayar arıyor.**
- `BaseHyperparameterOptimizer.create_env` `raw_stock_data.csv`'yi okuyor; bu dosyada yalnızca OHLCV var. Env eksik gösterge sütunlarını sabit varsayılanla dolduruyor (macd 0, rsi 50, cci 0, adx 0, türbülans 0); sembol başına 10 özelliğin 5'i sabit.
- HPO API'sinin varsayılan evreni `PHASE1_SYMBOLS`, yani 5 sembol (gözlem 56 boyut). Rota 30 sembolle eğitiyor (331). Pencereler de istekten geliyor, rotanın bölmesinden değil (hakem-2 #8).
- B9 ile HPO'nun seçim hedefi de dağılım dışına çıkıyor. Hakem-2 varsayılan pencerelerde doğrulama kırpmasını %13,6 ölçtü; bu sayı benim tarafımdan yeniden üretilmedi.
- Eğitim rotası `stock_data_with_indicators.csv` ile eğitiyor. HPO hiperparametreleri başka bir gözlem fonksiyonunda seçilip göstergeli ortamda kullanılıyor; etkisi ölçülmedi.
- **Kayıt artık kökeni taşıyor:** HPO kayıtları `indicators_present=False` ile açık. HPO hiperparametresiyle koşan rota kaydı `hyperparams_clean` üzerinden açık kalıyor. Düzeltme **R1**'de: rota ve HPO tek bölme tanımını, tek paneli ve tek evreni paylaşacak.

**Gözlemler:**
- **GPU.** Testin ilk koşumunda GPU görünürdü ve rota kabul kontrolü bir kez kaldı (5 GPU koşumunda 1). Kalan 4 koşum geçti; aynı panel ve tohumla 10 GPU rota koşumu da birbirine bit-eş çıktı. Nedeni doğrulanmadı; ilk sapmanın adımı ve bloğu o koşumda kaydedilmemişti, artık raporlanıyor.
  - Test CPU'ya zorlanıyor (`CUDA_VISIBLE_DEVICES=-1`; Windows'ta boş dize değişkeni siliyor ve GPU görünür kalıyor). Bu belirtiyi siliyor, nedeni değil (hakem-2 #5).
  - Rota kabulünün piyasa bloğu kıyası belirlenimsizlikten etkilenmez.
  - Tez koşumları GPU'da olacağı için `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` ya da CPU açık iş (§9.6 #5).
- **`tests/test_env.py`** ilk import satırında kırılıyor: `tests/data/` dizini `data` paketini gölgeliyor. B9'dan bağımsız, önceden vardı.
- **`generate_academic_report.py`** D5'e kadar duruyor. Bilinen sorunlar:
  - ham paneli temizlemeden ve RL penceresinin dışında bölüyor; hakem-2'ye göre test penceresi yeni rota modellerinin eğitimiyle **18 gün kesişiyor**;
  - ortam `legacy` eksik hücre modunda;
  - modelin meta'sındaki `test_range`'i kullanmıyor;
  - risksiz faiz 0,02 sabit;
  - SELL'de `trade['total']` KeyError.

  Belgelenen komut (`python scripts/analysis/...`) `app` importunda kırılıyordu; yol eklendi. Panodaki "rapor üret" ucu betiği yanlış yoldan çağırıyor (`scripts/generate_academic_report.py`); bu da önceden vardı.
- **Canlı karar** (`/daily-decision`) hâlâ ölçek geçirmiyor (Y12). Bu K paketinin işi (K9, K13).
- **Faz 2 ortamı** temel ve makro tablo verilmeden kurulursa 61 boyutlu uzay ilan edip 34 boyutlu gözlem üretiyor; SB3 eğitimin başında kırılıyor. Rota ve HPO faz 2'de iki tabloyu da yüklediği için varsayılan hatta görülmüyor. Betiklerin duman testinde bulundu; B9'dan bağımsız, önceden vardı.

**Etkileri:**
- Bu daldan sonra eğitilen modellerin test metrikleri farklı çıkar. Eski metrikler `TradingEnv(price_stats=None)` ile yeniden üretilebilir; env varsayılanı bit-eş.
- Rota kayıtlarında `price_stats_source='train_window'`. Gerçek panelde açık kalan maddeler:
  - `price_in_support` (B9'un eski ortamdaki bedeli);
  - `indicator_bfill` (B7);
  - `adx_constant` (K18).

  HPO kayıtlarında ek olarak `indicators_present` (Y21).
- **RL test sayıları ön sonuç.** B7 ve K18 kapansa da eski ortamın fiyat özelliği doygun kaldığı sürece kayıt açık kalır (K25).

### 9.6 hakem-2 incelemesi (B9, 2026-11-06) — karar: Büyük revizyon

Hakem kodda B9 sızıntısını geri getiren bir yol bulamadı. Kendi ölçümleri:
- test 65/65;
- doygunluk sayıları aynı;
- main ile 4/4 özet ve ek 10 yapılandırmada bit-eşlik.

Hakem bu ölçümleri `python -I` betikleriyle yaptı; çıktıları oturumun geçici dizininde kaldı, depoya alınmadı.

**Depo koduyla yeniden ürettiklerim:**
- doygunluk sayıları;
- 11 yapılandırmada main ile bit-eşlik (artık testte);
- etki farklarının %95 GA'ları (A2C [−0,93; 1,09] Sharpe farkı, aynı);
- sabit özellik oranı %35,3 (76 fiyat + 30 ADX boyutu / 300; blok ölçümü bileşenleri veriyor);
- eski koldaki kırpmanın pozisyon bloğundan geldiği (ölçüm: %14–26);
- "ilk 16 gün" mutantının t=15'te geçip t=0'da kaldığı.

**Farklı çıkan:** test penceresinde risksiz birikim. Hakem ortalama orandan yaklaşık +%137,5 hesapladı; depo fonksiyonu (`evaluation/risk_free.py`, seans başına t−1 oranı, ACT/365) **+%104,6** veriyor. Rapor ikincisini kullanır.

**Yeniden üretmediklerim:** raporun test penceresinin rota eğitimiyle 18 gün kesişmesi ve HPO varsayılan penceresinde %13,6 kırpma.

Üç büyük itiraz birleştirmeyi engelliyordu; üçü de düzeltildi.

| # | Cid. | İtiraz | Yanıt |
|---|---|---|---|
| 1 | BÜYÜK | Protokol doygunluğu taşımıyor. B7 ve K18 kapanınca %74,7'si kırpık eski ortam kayıtları "temiz" sayılır; §9.5'in "K18'e kadar" cümlesi K18'in kapsamıyla çelişiyor | **Kabul, düzeltildi.** Ortamdan ölçülen `price_in_support` maddesi eklendi (destek dışı oran ≤ %1). Cümle düzeltildi: K18 eski ortamın ölçeğini değiştirmiyor. Destek içinde kalan kol için karar K25 |
| 2 | BÜYÜK | Etki tasarımı sızıntının etkisini ayırt edemiyor; "şişirme gösterilemedi" ve "ölçüldü" ifadeleri kanıtı aşıyor; %80,3 yeniden üretilmemiş bir sayı | **Kabul, düzeltildi (a).** Şişirme cümlesi kaldırıldı. Y13 "kapandı (bu daldan sonraki koşumlar); metriğe etkisi tanımlanamadı" oldu. Tablo ± std (n=3) + fark GA'sıyla yazıldı ve fazla getiri Sharpe'ına geçti (#11). Az eğitim yanlılığı ve iki farkın toplamı yazıldı. %80,3 çıkarıldı. (b) üçüncü kol (≥ 10 tohum) K25'e bağlandı |
| 3 | BÜYÜK | Etiket tutarlılığı ölçüyor, kökeni değil: test penceresinin kendi ölçeği "eğitim ölçeği" diye verilince temiz çıkıyor | **Kabul, düzeltildi.** `price_stats_source` artık eğitim ortamının kendisini alıyor. Üç koşul denetleniyor: eğitim ölçeği kendi penceresinden mi, pencere değerlendirmeden önce mi bitiyor, değerler eşit mi. Ayrıca yedek ölçek kontrolü var. K7 kurgusu, çakışan pencere ve yedek ölçek testte açık çıkıyor |
| 4 | KÜÇÜK | Ölçüt tek t'de sınanıyor | **Kabul, düzeltildi.** Env t ∈ {0, 15, T−2}, rota t ∈ {0, 6}. Hakemin "ilk 16 gün" mutantı teste eklendi; t=0'da yakalanıyor |
| 5 | KÜÇÜK | Rota kabulü politikaya bağlı gözlemi kıyaslıyor; GPU belirlenimsizliği sızıntıyla karışıyor; CPU'ya zorlamak nedeni silmiyor | **Kısmen.** Piyasa bloğu ayrıca kıyaslanıyor (politikadan bağımsız). İlk sapmanın adımı ve bloğu raporlanıyor. C1 satırına gözlenen sapma eklendi. **Açık iş:** tez koşumları için `use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG` ya da CPU (C1'in uzantısı, Milestone 0' öncesi) |
| 6 | KÜÇÜK | Bekçi anahtar kelimenin varlığına bakıyor, değerine değil; yardımcıların varsayılanı sızıntılı yol; 24 isabetin 12'si testin kendisinde | **Kabul, düzeltildi.** Yardımcılarda `price_stats` zorunlu anahtar. Bekçi `None` sabitini, sarmalayıcıları, niteliği ve maske dilimini yakalıyor. Sayım ayrı yazılıyor (31; 14'ü test dışı). `partial(...)` ve `**dict(...)` yolları hâlâ görülmüyor; sarmalayıcılar S bölümünde davranışla sınanıyor |
| 7 | KÜÇÜK | K13 ilkesi `test_model_manual.py` ve `debug_model_actions.py`'de çiğneniyor | **Kabul, düzeltildi.** İkisi de meta ölçeği yoksa duruyor |
| 8 | KÜÇÜK | Y21'in kapsamı eksik; HPO kökeni rota kaydına geçmiyor | **Kabul, düzeltildi.** Y21 metni genişletildi (5 sembol, ayrı pencereler, seçim hedefi de dağılım dışı). Rota kaydında `hyperparams_source` / `hyperparams_clean` / `hyperparams_open_items` var. HPO kaydı sağlık ölçümünü taşıyor. Asıl düzeltme R1'de |
| 9 | KÜÇÜK | `observation_health` K18 tanımını karşılamıyor; uyarı eşiği gerekçesiz; `_metrics.json` protokol taşımıyor | **Kısmen.** Blok başına kırpılan ve sabit oran, eğitim ve test ayrı ölçülüyor; pozisyon bloğu fiyattan ayrıldı. Uyarı keyfi %10 yerine protokol maddesine bağlandı. `_metrics.json` protokolü, açık maddeleri ve sağlığı taşıyor. Ölçütün nihai tanımı (eşikler, MARL `obs_spec`) K18'de |
| 10 | KÜÇÜK | Rapor yalnız yeni modelleri değerlendiriyor; test penceresi eğitimle 18 gün kesişiyor; legacy mod; atlanan modeller yazılmıyor; rf 0,02 | **Kabul, düzeltildi (koruma).** Rapor D5'e kadar varsayılan olarak duruyor; gerekçeler ekrana ve ayrı dosyaya yazılıyor. Asıl düzeltme D5 |
| 11 | KÜÇÜK | Sharpe rf=0 ile hesaplanmış | **Kabul, düzeltildi.** Ölçüm fazla getiri Sharpe'ı, al-tut ve risksiz birikimi raporluyor |
| 12 | KÜÇÜK | "Sessiz yedek yok" yalnız tüketici ortam için doğru | **Kabul, düzeltildi.** Üretici `price_stats_fallbacks`; meta dosyasında; etiket `train_window_fallback` (açık) |
| 13 | KÜÇÜK | Sayıların izi geçici dosyada; veri SHA'sı yok | **Kabul, düzeltildi.** Ölçüm JSON'ları `docs/development/measurements/phase9-b9/` altında. İçlerinde panel SHA-256, commit ve komut var |
| 14 | KÜÇÜK | HPO araması hiçbir DSR N'sine girmiyor (önceden var olan C2 sorunu) | **Kabul, açık iş:** M8'de HPO → rota seçim soyu bağlanacak. Rota kaydı artık hangi HPO çalışmasından geldiğini taşıyor (#8) |

**Sıradaki:** B7 (gösterge `bfill`), K18 (ADX + gözlem sağlığı; K25 kararıyla birlikte), R1 (tek bölme, tek panel, tek evren — Y21 dahil), ardından B8, B10, E, F, G, D, P ve Milestone 0'.

---

### 9.7 B7 — gösterge ısınma boşluğu (2026-11-06, dal `feat/phase9-b7-indicator-warmup`)

**Sorun (Faz 10 Y15).**
- `technical_indicators.add_all_indicators` ısınma boşluğunu ffill → **bfill** → 0 ile dolduruyordu: sembolün ilk satırlarına göstergenin ilk geçerli (gelecekteki) değeri yazılıyordu.
- `DataFetcher.clean_data` da sembolün ilk geçerli satırından önceki NaN satırları bfill'liyordu.

**Değişiklik** (hepsi bayrakla; varsayılanlar eski davranış):
- `add_indicators_to_multi_symbol_df(warmup='nan')`: yalnızca ffill uygulanır (nedensel); ısınma satırları NaN kalır.
- `clean_data(leading='drop')`: baştaki NaN satırlar atılır, gelecek değer geriye yazılmaz.
- `prepare_rl_panel(drop_indicator_warmup=True)`: bfill imzası taşıyan baş blok pencere içinde atılır.
  - İmza: sembolün **tam** geçmişinde ilk değerin birebir tekrarı, en az 2 satır (`WARMUP_MIN_LEN`; kapı, protokol ve rapor aynı eşiği kullanıyor).
  - Yalnız bfill'lenen sütunlara bakılır (`BFILL_WARMUP_COLS`: macd, rsi, cci). Türbülansın 252 günlük 0 başı nedensel bir yer tutucudur ve sayılmaz; ADX sabittir (K18).
  - `nan` moduyla üretilmiş panelde NaN ısınma satırları da atılır.
  - Sembol başına 30 satırdan fazla atım uyarı verir. Atılan hücre eksik sayılır (`own_last`: işlem yok).
  - Rota ve HPO bu seçenekle çağırıyor.
- Protokoldeki `indicator_bfill` artık **ölçülüyor**: eğitim ya da değerlendirme verisinde bfill bloğu ya da NaN ısınma satırı kalmış mı. Rota ve HPO beyan göndermiyor.

**Kabul** (`python tests/test_indicator_warmup.py`, **28/28**):
- **B0, göstergeler:** `warmup='nan'` ile t ∈ {3, 10, 18, 25, 60, 260, 300} için t'de kesilen seride t'ye kadar bit-eş (NaN = NaN). Fikstür 330 gün, böylece türbülans da sınanıyor. Eski `bfill` ısınma içindeki t'de kalıyor; sızıntı sayısal olarak gösteriliyor.
- **B0, OHLCV:** `clean_data(leading='drop')` t ∈ {1, 5, 40, 200} için kesme değişmez. Eski `bfill` t=1'de kalıyor.
- **Varsayılan yol:** `add_indicators` ve `clean_data` varsayılanları, main @ 90d50f3'ten `git show` ile yüklenen modülle bit-eş.
- **Kapı:** atılan satırlar `nan` modunun ısınma satırlarını kapsıyor. Fark en fazla 1 satır: bloğun içinde ilk gerçek değer de var, yani fazladan atılan gerçek bir satır; sızıntı yaratmaz.
  - Blok pencereden önce bitiyorsa hiçbir satır atılmıyor.
  - Panel pencereden 252 seanstan az önce başlıyorsa, yani türbülansın 0 başı pencerenin içindeyse, yalnız geç listelenen sembol etkileniyor.
  - `nan` moduyla üretilmiş panelde NaN satırlar atılıyor.
- **Protokol:** kapı bloğu atınca `indicator_bfill=False`, atmayınca `True`; kapısız `nan` panelde de `True`. İki mutasyon da yakalanıyor: dedektörü kör bir kapı ve protokol tarafında kör bir dedektör.
- **Gerçek panel** (RL penceresi 2013-06):
  - Kapı ODAS'ta **pencere içi ilk 10**, PGSUS'ta **ilk 17 satırı** atıyor; §9.1'deki ısınma bloklarıyla aynı. Kapıdan sonra pencerede bfill bloğu kalmıyor.
  - Bağımsız `nan` referansı (sembol başına yeniden hesap) NaN ısınma satırlarının atılanların içinde olduğunu gösteriyor: ODAS 9 ⊆ 10, PGSUS 16 ⊆ 17. Hakem-2 de aynı sayıyı buldu.
  - Ham panelde NaN kapanış yok; test R bunu bekçi olarak denetliyor. Yani `clean_data`'nın bfill'i bugünkü veride hiçbir satırı etkilemiyor.
  - PGSUS'un ham panelde IPO günü (2013-04-26) tek satır, sonra 2013-07-23'e kadar satır yok. Isınma bu boşlukta 3 aya yayılıyor; EMA'lar sıçramayı tohum olarak taşıyor. Bu nedensel, sızıntı değil.

**K18 için bulgu:** `warmup='nan'` modunda **ADX bütün satırlarda NaN**. "ADX ≡ 0"ın kökü hesabın NaN üretmesi; eski yol bunu `fillna(0)` ile sabit 0'a çeviriyordu. Testte kayıt olarak sabitlendi.

**Açık kalanlar:**
- Tahmin hattı (`prediction_service.py:89`) ve canlı karar (`daily_trading.py:419`) hâlâ varsayılan `bfill` kullanıyor. `nan` moduna geçiş K18'e bağlı: ADX tümüyle NaN olduğu için özellik satırları düşebilir.
- Canlı kararda yalnız son satır kullanıldığı için ısınma bfill'i karara ulaşmıyor; 30 seanslık pencere CCI'nin 20 seanslık penceresinden uzun. Bu kod okumasından çıkan bir çıkarım, ölçülmedi.
- Önceden hesaplanmış `stock_data_with_indicators.csv` hâlâ bfill'li; RL kapısı blokları atıyor. CSV'yi `nan` moduyla yeniden üretmek de K18'e bağlı (aynı ADX nedeni).

**Etkileri:**
- Rota kayıtlarında `indicator_bfill=False`. Gerçek panelde açık kalan maddeler `adx_constant` (K18) ve `price_in_support` (K25). **Düzeltme (K18, §9.9):** bu cümle gerçek panel için doğru değildi; B7 protokol kuralı test penceresinin başındaki doğal RSI tekrarını (VAKBN) blok sayıyordu.
- Bu daldan sonra eğitilen modellerin eğitim verisinde ODAS ve PGSUS'un pencere içi ilk 10 ve 17 satırı yok.
- HPO kayıtları `indicators_present=False` ile açık kalıyor (Y21).

### 9.8 hakem-2 incelemesi (B7, 2026-11-06) — karar: Büyük revizyon

İnceleme paketiyle yapılan ilk inceleme: **~144 bin token, 25 araç çağrısı, ~12 dk** (B9'da paket yokken 301 bin / 66 / 23 dk; paket küçük olduğu için kıyas tam değil; `review-panel.md`). Sızıntı bulgusu yok.

| # | Cid. | İtiraz | Yanıt |
|---|---|---|---|
| 1 | BÜYÜK | Dedektör türbülansın 252 günlük nedensel 0 başını bfill bloğu sanıyor. Panel pencereden 252 seanstan az önce başlıyorsa kapı her sembolden ~1 yıl atıyor. Hakemin ölçümü: 2018 başlangıçlı gerçek OHLCV'de 7.279 / 64.958 satır. Bugünkü CSV (2000 başlangıçlı) etkilenmiyor | **Kabul, düzeltildi.** Dedektör yalnız bfill'lenen sütunlara bakıyor (`BFILL_WARMUP_COLS`). 330 günlük senaryoda yalnız geç listelenen sembol etkileniyor. 30 satırı aşan atım uyarı veriyor |
| 2 | KÜÇÜK | B0'da türbülans ve ADX boşuna geçiyor; OHLCV için t-kesme testi yok; "bugün etkisiz" ölçümünün bekçisi yok. CSV üreten yollar (`trading.py:531, 828, 838, 1174`) hâlâ varsayılan `bfill` | **Kısmen.** Fikstür 330 gün; türbülansın değiştiği ve ADX'in NaN olduğu kontrol ediliyor. `clean_data` için t-kesme testi eklendi; ham panelde NaN kapanış yokluğu test R'de bekçi. **Açık:** CSV üreten yolların `nan`/`drop` moduna geçmesi K18'e bağlı (ADX NaN) |
| 3 | KÜÇÜK | Protokol kapıyla aynı dedektörü kullanıyor (ortak mod); NaN baş `{}` dönüyor; doğrulama ortamı ölçülmüyor | **Kabul, düzeltildi.** Test R'de bağımsız `nan` referansı var (ODAS 9 ⊆ 10, PGSUS 16 ⊆ 17). Protokol tarafı mutasyonu eklendi. Protokol ve kapı NaN başı da sayıyor. Rotada doğrulama ortamı yok; bu R1'de gelecek |
| 4 | KÜÇÜK | Rapor `min_len=5`, kapı ve protokol 2 kullanıyor | **Kabul, düzeltildi.** Tek sabit: `WARMUP_MIN_LEN = 2` |
| 5 | KÜÇÜK | PGSUS'un IPO günü tek satır, ardından 3 ay boşluk; "ilk n seans" ifadesi yanlış | **Kabul.** İfade "pencere içi ilk n satır" oldu, boşluk §9.7'ye yazıldı. **Açık:** tek satırlık IPO günü ve boşluk eksik hücre kaydında ayrıca işaretlenmeli (V) |
| 6 | KÜÇÜK | Belge izi: `integrity.py` docstring'i, `nan` modu docstring'i, CLAUDE.md'de `clean_data` tarifi, "ölçüldü" ibaresinin komutu | **Kabul, düzeltildi** |

### 9.9 K18 + K25 (b) — ADX, nedensel fiyat ölçeği, gözlem sağlığı (2026-11-09, dal `feat/phase9-k18-adx-obs-health`)

**Karar.** K25 için (b) seçildi (kullanıcı, 2026-11-09) ve K18 dalında uygulandı (Faz 10 §6). Karar yalnız **fiyat** ölçeğini kapsıyor. MACD, türbülans ve hacmin ölçeği ayrı bir karar: K26.

**Sorun.**
- **ADX ≡ 0 (Faz 10 §8 #2).** B7'de `warmup='nan'` modunda ADX tümüyle NaN çıkmıştı (§9.7). Kök neden bir indeks hizalama hatası:
  - `calculate_adx`'te +DM/−DM RangeIndex'li seri olarak kuruluyor, ATR tarih indeksli.
  - Bölmede pandas indeksleri hizalıyor; DI'lar tümüyle NaN oluyor ve `dx = 0` çıkıyor.
  - RangeIndex'li sonuç tarih indeksli df'e atanınca yine tümüyle NaN oluyor; eski yol bunu `fillna(0)` ile sabit 0'a çeviriyordu.
  - Mevcut RL modelleri ve tahmin hattı ADX = 0 ile eğitildi. Aynı fonksiyon RangeIndex'li bir df'te doğru çalışırdı.
- **K25.** B9 sonrası eski ortamda test fiyat özelliklerinin %98,9'u eğitim desteğinin dışında (§9.5).
- **§9.6 #9.** Gözlem sağlığı ölçütünün nihai tanımı (eşikler) K18'e bırakılmıştı.

**Değişiklik.** Fonksiyon ve env varsayılanları eski davranış (main ile bit-eş, test D). **Rotanın ADX'i bilinçli olarak değişti** (`RL_ADX_MODE='wilder'`, istekle geri alınamıyor; hakem-2 #7).
- **ADX** (`data/technical_indicators.py`):
  - `adx='legacy'|'wilder'` bayrağı eklendi; varsayılan `legacy`.
  - `wilder` kipi indeks hizalı ve klasik Wilder (1978) yumuşatması: ilk n değerin ortalamasıyla tohumlanıyor, sonrası A_t = A_{t−1} + (x_t − A_{t−1})/n.
  - İlk 2n−1 = 27 satır NaN ısınma. Tanımsız bölmeler 0: ATR = 0 ise DI = 0; DI⁺ + DI⁻ = 0 ise DX = 0.
  - İlk sürüm ilk değerle tohumlanan bir EMA kullanıyordu; hakem-2 (#6) bunun ilk satırlarda klasik tanımdan ayrıştığını gösterdi.
  - Rota ADX'i `with_adx(df, 'wilder')` ile OHLCV'den yeniden hesaplıyor. CSV'ye yazılmıyor, çünkü eski modeller ve canlı karar aynı dosyayı okuyor. Meta, manifest ve deneme kaydının `config` alanı `indicator_spec` taşıyor.
- **Veri kapısı:** `adx` artık `BFILL_WARMUP_COLS`'ta. Legacy ADX sabit (ya da tümüyle NaN) olduğu için dedektör onu zaten atlıyor; legacy panelde kapı main ile aynı.
- **K25 (b), nedensel fiyat ölçeği** (`env/trading_env.py`):
  - `price_norm='stats'|'causal'` bayrağı; varsayılan `stats`.
  - `causal` kipinde t günündeki O/H/L/C ölçeği, sembolün t'ye kadarki (dahil) son W kapanışının ortalaması ve std'si (ddof=1). W = 252 (`CAUSAL_WINDOW`); `None` genişleyen pencere demek.
  - En az 20 gözlem gerekiyor (`CAUSAL_MIN_PERIODS`); yoksa ya da std ≈ 0 ise o günün fiyat özellikleri 0 (nötr). W ve 20 tasarım varsayımı; literatür dayanağı yok.
  - Pencerenin başı `price_context` ile dolduruluyor; bağlamın ortamın ilk tarihinden önce bitmesi zorunlu, aksi `ValueError`. Lookup cache zorunlu (`.loc` yolu desteklenmiyor).
  - **Eğitim bağlamı** (hakem-rl #2, hakem-2 #8): `data.integrity.rl_context_before`, RL penceresinden önceki son 504 panel tarihinden hayalet günleri ve bayat hücreleri atıyor. Sonra sembol başına son 252 geçerli satırı seçiyor, yani atım bağlamı kısaltmıyor. Bu kadar geçmişi olmayan sembol seri başı olarak işaretleniyor (`attrs['series_start_symbols']`). Artefakt taraması bağlam + pencerenin ilk tarihi üzerinde yapılıyor, böylece sınır geçişi de taranıyor. Kapanışı etkileyen artefakt `DataIntegrityError` veriyor; `ohlc_inconsistent` satırları kalıyor ama sayılıyor.
  - Rotada test ortamının bağlamı = eğitim bağlamı + eğitim + doğrulama (hepsi testten önce). İlk sürümde eğitim bağlamı bu listede yoktu; kısa panelde pencere dolmuyordu. Yeni `causal_short_context` etiketi rota testinde bunu yakaladı.
  - Rotada `TrainingRequest.price_norm` alanı; varsayılan `stats`. Meta ve manifest `price_norm` taşıyor.
  - Pencere kullanımı kayıtta: sağlık kaydında `price_norm_warmup` (pencere dolmadan geçen ve nötr satır, bağlam satırı, seri başı semboller).
- **MACD, opt-in** (hakem-2 #2): `price_norm_macd=False` varsayılan; MACD eski tanımında, `tanh(macd/0,1)`. `True` ile fiyatla aynı nedensel ölçeğe geçiyor: `tanh(macd/std_t)`, yeni sabit yok. İlk sürümde bu varsayılandı ve K25'in KARAR metnine yazılmıştı; kullanıcı kararını aştığı için geri alındı ve karar K26'ya bağlandı.
- **Protokol** (`evaluation/trial_registry.py`, `evaluation/observation_health.py`):
  - **`price_stats_source='causal'`** temiz sayılıyor. Koşullar:
    - iki ortam da nedensel; pencere ve MACD tanımı aynı;
    - eğitim penceresi değerlendirmeden önce bitiyor;
    - **ölçek gerçekten dolu pencereyle kurulmuş:** değerlendirmede pencere dolmadan geçen satır yok. Eğitimde böyle satır yalnız bağlamın işaretlediği seri başı sembollerde olabilir (hakem-2 #3).
    
    Aksi `mixed_norm`, `overlapping_window`, `causal_short_context` ya da `causal_unverified`; hepsi açık.
  - **`obs_health`** (ölçülüyor): eğitim ve değerlendirme penceresinde fiyat, hacim ve gösterge bloklarında üç koşul aranıyor (hakem-rl #1):
    - kırpılan oran (\|x\| ≥ 10) en fazla %1;
    - pencere boyunca sabit gözlem boyutu yok;
    - özellik başına doygunluk (`sat_frac`: \|x\| ≥ %99 × sınır) en fazla %90.
    
    Doymuş bir özellik gözlemde neredeyse sabit bir boyuttur, yani ADX = 0 hatasının RL karşılığı. Faz 2'nin temel oranları sembol başına sabit bir tablodan geldiği için sabit-boyut kuralına girmiyor.
  - **`features_in_support`** (ölçülüyor): fiyat dışı her piyasa özelliğinde, 30 sembolün havuzlu değerlerinin en fazla %1'i eğitim [min, maks] aralığı dışında.
  - **`dims_in_support`** (ölçülüyor; hakem-2 #1): kapılı blokların **her gözlem boyutunda** (sembol × özellik) aynı ölçü en fazla %1. Politika bir MLP ve her boyut ayrı bir girdi; havuzlu oran tek bir sembolü gizliyor (gerçek panelde SASA hacmi testte %23,3 destek dışıyken havuz %0,87). Sağlık kaydı her özellik için en büyük boyutu, sembolünü ve eşiği aşan boyutları taşıyor.
  - `adx_constant` artık eğitim ortamını da ölçüyor.
  - Bu maddeleri taşımayan (K18 öncesi) kayıt açık sayılıyor.
  - Eşikler (%1 kırpma, %1 destek, %90 doygunluk) tasarım varsayımı.
- **B7 ölçütünün yanlış pozitifi** (yeni bulgu, K18 ölçümünde):
  - Protokolün bfill dedektörü değerlendirme penceresinin başını sembolün seri başı sanıyordu.
  - Gerçek panelde VAKBN'nin RSI'ı test penceresinin ilk iki gününde birebir aynı; kapanış değişmeyince RSI aynı kalıyor. Eski kural bu pencerede `{VAKBN: rsi 2}` döndürüyor (JSON `b7_old_rule_eval_blocks`); doğrulama penceresinde blok yok. hakem-2 bunu bağımsız olarak yeniden üretti.
  - Yani B7 ölçütü bu panelde her rota test kaydına `indicator_bfill=True` (açık) yazardı. Depoda gerçek panelle yazılmış bir rota kaydı yok (`trial_registry.jsonl` bulunamadı); bu gözlenmiş bir kayıt değil, ölçülmüş bir koşullu. §9.7'deki "rota kayıtlarında `indicator_bfill=False`" cümlesi bu panel için doğru değildi. Hata güvenli yöndeydi.
  - Düzeltme: eğitimde en az `WARMUP_MAX_ROWS` = 28 satırı olan sembolün değerlendirme penceresi başı sayılmıyor. 28, en uzun ısınma: `wilder` ADX'in 27 NaN satırı + ilk geçerli değer. Eğitimde daha kısa geçmişi olan sembolün değerlendirmeye taşan bloğu yakalanıyor (hakem-rl #3), değerlendirmede listelenen sembolün bloğu da.

**Kabul** (`python tests/test_k18_obs_health.py`, ağ yok, CPU, **109/109**; hakem düzeltmelerinden önce 91):
- **A, ADX:**
  - Legacy kök neden kayıt olarak sabit: RangeIndex'li sıfır serisi; `nan` modunda tümüyle NaN, `bfill` modunda 0.
  - `wilder`: df indeksini taşıyor, ilk 27 satır NaN, değerler [0, 100].
  - **Bağımsız referans 1:** testte yazılmış SMA tohumlu klasik Wilder döngüsü; ilk tanımlı satırdan (27) itibaren \|fark\| < 1e-9.
  - **Bağımsız referans 2:** `pandas_ta.adx` 0.4.71b0; RMA ilk değerle tohumlu, satır ≥ 400'de \|fark\| < 1e-6.
  - Düz trendde ADX > 90, yatay seride 0.
  - **B0:** t ∈ {3, 26, 27, 60, 300} için `nan` + `wilder` kesme değişmez. `bfill` + `wilder` ısınma içindeki t'de kalıyor.
- **D, dondurma:** main @ 01aabdb (`git show`) ile bit-eş:
  - göstergeler (`bfill` ve `nan`);
  - veri kapısı, legacy panel, `drop_indicator_warmup` her iki değeri;
  - env'de 8 yapılandırma.
- **G, kapı:** her sembolden 27 satır atılıyor; kalan satırlar bağımsız `nan` referansına göre nedensel.
  - **Mutasyon:** dedektör `adx`'i bilmezse geriye doldurulmuş ADX satırı kalıyor ve test bunu yakalıyor. İlk sürümde bu mutasyon boştu: varsayılan argüman tanım anında bağlanıyor.
- **N, K25:**
  - z, bağımsız numpy hesabıyla aynı; genişleyen pencere ve opt-in `tanh(macd/std_t)` de.
  - Varsayılanda MACD eski tanımında.
  - **Mekanizma:** fiyat düzeyi ×7 olunca nedensel z ve `tanh(macd/std_t)` değişmiyor, B9 z'si değişiyor.
  - Bağlamsız ortamda ilk 19 gün nötr; pencere kullanımı kayıtta.
  - **B0, t ∈ {0, 60, T−2}:** t sonrası fiyat değişince t'ye kadar tüm gözlem vektörü bit-eş, t+1 değişiyor.
  - **Mutasyon:** merkezli (gelecekli) pencere t=0'da yakalanıyor.
  - Penceresine taşan bağlam, `stats` kipinde verilen bağlam, `close`'suz bağlam ve önbelleksiz panel `ValueError` veriyor.
  - `own_last` eksik günde sembolün son satırının istatistiği kullanılıyor.
  - Bu fikstürde (tohum 5, 1.500 günlük eğitim) nedensel ölçek destek içinde, B9 dışında. **Destek içinde kalmak garanti değil:** sentetik rastgele yürüyüşte tohuma bağlı (ölçüm arşivlenmedi), gerçek panelde ölçülüyor.
- **C, bağlam:**
  - Sembol başına son 252 geçerli satır alınıyor; yeni listelenen sembol seri başı olarak işaretleniyor.
  - Bağlam–pencere sınırındaki sıçrama, negatif kapanış: `DataIntegrityError`.
  - `ohlc_inconsistent` kalıyor ve sayılıyor.
  - Bayat hücre atılıyor ama bağlam kısalmıyor.
- **P, protokol:**
  - Etiketler sınanıyor: `causal`, `mixed_norm`, `overlapping_window`, `causal_unverified`. `causal_short_context` hem bağlamsız değerlendirmede hem bağlamsız eğitimde çıkıyor.
  - Seri başı istisnası: işaretli sembol temiz, işaretsiz açık.
  - Kapılar ayrı ayrı yakalıyor: `obs_health` kırpma, sabit ADX ve doygunluk; `features_in_support` tek bir özelliği; `dims_in_support` tek bir sembol boyutunu.
  - B7 yanlış pozitifi ve iki kontrol grubu.
- **R, rota uçtan uca** (sentetik, CPU):
  - Eğitim ortamı rotanın `wilder` ADX'ini görüyor.
  - Meta, kayıt ve config `indicator_spec` ile `price_norm` taşıyor.
  - Nedensel kolda eğitim ortamı pencere öncesi 252 seansı bağlam alıyor ve penceresi ilk günden dolu (panel 2012'de başlıyor); etiket `causal`.
  - **t ∈ {0, 6}:** test piyasa bloğu t'ye kadar bit-eş, t+1'de değişiyor.
- **X, gerçek panel:**
  - Her sembolde yalnız ilk 27 satır NaN; ADX [0, 100] aralığında ve sabit değil.
  - Kapı ODAS'tan 17, PGSUS'tan 24 pencere içi satır atıyor (B7'de 10 ve 17); kalan satırlar bağımsız referansla nedensel.
- **Uyarlanan eski testler:**
  - B9 rota testi test penceresini rotanın veri yolundan türetiyor.
  - C testinin temiz fikstürü yeni maddeleri taşıyor.
  - Rotanın yeni uyarı metni B9'un alt dizge kontrolüyle çakışıyordu; ayrıştırıldı.
- **Diğer takımlar yeşil** (son düzeltmelerden sonra): B9 103, C 39, B7 28, V 27, eksik hücre 17, lookup 41, fiyat bekçisi 26, A 51, eğitim durumu, günlük karar 18, evren 22, manifest. Tahmin regresyon golden'ı korunuyor (`test_prediction_regression`: davranış korunmuş).

**Ölçüm — gerçek panel.**
- Komut: `python scripts/analysis/measure_obs_health.py --out docs/development/measurements/phase9-k18/obs_health.json`.
- Çıktı: [measurements/phase9-k18/obs_health.json](measurements/phase9-k18/obs_health.json). Commit 69c3a66, `git_dirty=false` (5f1c2d4 ile birebir aynı sayılar); CSV SHA-256 `5ca2fe3f…`, K18 paneli `9439764f…`.
- Veri yolu rotanınki; eğitim yapılmıyor. Bölmeler: eğitim 2013-06-03..2022-08-11 (2.303 gün), doğrulama 2022-08-12..2024-08-05 (494), test 2024-08-06..2026-07-27 (494), 30 sembol.
- Nedensel kollarda eğitim bağlamı `rl_context_before`, test bağlamı bağlam + eğitim + doğrulama.
- **Kırpma:** K18 öncesi ve B9 kollarında %74,7; nedensel kollarda %0.

| Yapılandırma (test) | Fiyat destek dışı (havuz) | MACD destek dışı | MACD doygun | Açık maddeler (test) |
|---|---|---|---|---|
| K18 öncesi rota (legacy ADX, B9) | %98,9 | %28,7 | %75,0 | `adx_constant`, `price_in_support`, `obs_health`, `features_in_support`, `dims_in_support` |
| K18 rota varsayılanı (`wilder`, B9) | %98,9 | %28,7 | %75,0 | `price_in_support`, `obs_health`, `features_in_support`, `dims_in_support` |
| **Nedensel W=252 (K25, MACD eski)** | **%0,03** | %28,7 | %75,0 | `obs_health`, `features_in_support`, `dims_in_support` |
| Nedensel W=252 + MACD (opt-in) | %0,03 | %0,27 | %0 | `obs_health`, `dims_in_support` |
| Nedensel W=63 (duyarlılık) | %0,08 | %28,7 | %75,0 | `obs_health`, `features_in_support`, `dims_in_support` |
| Nedensel, genişleyen | %8,7 | %28,7 | %75,0 | `price_in_support`, `obs_health`, `features_in_support`, `dims_in_support` |
| Nedensel W=252, eğitim bağlamsız | %0,03 | %28,7 | %75,0 | `price_stats_source` (`causal_short_context`), `obs_health`, `features_in_support`, `dims_in_support` |

- **Sembol düzeyinde** (`dims_in_support`; JSON `eval_out_of_train_support_dims`; nedensel W=252):
  - **Test:** hiçbir fiyat boyutu %1'i aşmıyor. Hacimde 2 sembol aşıyor, en büyüğü SASA %23,3. MACD eski tanımda 20 sembol aşıyor (en büyük YKBNK %83,2); opt-in nedensel MACD'de 3 sembol (en büyük FROTO %3,4).
  - **Doğrulama:** fiyatta 8 sembol aşıyor (en büyük GARAN kapanış %7,3). ADX'te 8 sembol (SKBNK %10,5), hacimde 5 (ENKAI %10,1), RSI'da 5 (SKBNK %2,8).
  - W=63'te testte 1 sembol aşıyor (TKFEN, O/H/L/C'nin dördü, yani 4 fiyat boyutu; en çok %1,8); W=252'de 0. Fiyat boyutlarında W=252 daha iyi.
- **Okuma.** Nedensel ölçek fiyatın **düzey** bağımlılığını kaldırıyor; mekanizma ×7 testinde gösterildi. Ama destek içinde kalmayı garanti etmiyor. Havuzlu %0,03 sembol düzeyindeki aşımları gizliyordu; doğrulamada sekiz sembolün fiyat boyutu eğitim aralığının dışına taşıyor. "Nedensel kol fiyatı desteğe alıyor" ifadesi yalnız test penceresi ve fiyat boyutları için doğru.
- **Neden kayan pencere:** genişleyen ortalama TL enflasyonunun gerisinde kalıyor (test %8,7, doğrulama %22,8 destek dışı). W=63 ile W=252 aynı protokol sonucunu veriyor, çünkü ikisi de hacim ve MACD yüzünden açık; fiyat boyutlarında W=252 daha iyi (hakem-2 N2).
- **Eğitim bağlamı:**
  - Aralık 2012-05-22..2013-05-31; 18 hayalet gün ve 12 bayat hücre atıldı, 7 `ohlc_inconsistent` satırı kaldı. Seri başı: ODAS, PGSUS (JSON `train_context`).
  - Eğitimde pencere dolmadan geçen satır bağlamsız 7.530'dan (%10,9) 492'ye indi; bunların tamamı ODAS (242) ve PGSUS (250), seri başı. Nötr satır 570'ten 28'e indi.
  - Test desteğini değiştirmiyor (fiyat %0,03 iki durumda da).
- **Türbülans** (hakem-rl #1): gözlemde eğitimde %99,7, testte %100 doygun (`tanh(turb/2)`). 30 sembollü Mahalanobis uzaklığının eğitim medyanı 32,0, %1'lik dilimi 8,9. **Eski ortamın hiçbir kolu `obs_health`'ten geçmiyor.**
  - Seri düzeyinde seçenekler (eğitim/test doygunluk; JSON `turbulence_scaling`):
    - `tanh(t/2)`: %99,8 / %100;
    - `tanh(t/30)` (n = sembol sayısı): %6,7 / %3,2;
    - `tanh(log1p(t)/5)`: %0 / %0, ama std 0,07'ye sıkışıyor.
  - Üçünde de test destek dışı 0. K26 kararı (a): eski ortamda yalnız beyan; adaylar M2 `obs_spec`'ine taşınır (Faz 10 §6).
  - CCI %1,0–1,4 doygun.
- **ADX dağılımı** (pencere, `wilder`): medyan 24,0, %1–%99 aralığı 10,2–60,0, gözlemlerin %46,3'ü > 25, sembol ortalamaları 23,5–28,1. Kaynak: JSON `adx_profile_wilder`.

**Okuma ve sınırlamalar.**
- **Eski ortamın hiçbir kolu protokolde temiz değil** (türbülans: `obs_health`; hacim ve doğrulamada fiyat/ADX: `dims_in_support`; MACD eski tanımda: `features_in_support`). K26 = (a) olduğu için bu kalıcı: eski ortam yalnız sadık replikasyon olarak, sınırlamaları beyanlı raporlanır; tezin kanıtı yeni ortamdan (MAPPO ↔ SARL-eşmimari) gelir. Protokole girmeyen açıklar sürüyor: tek bölme, doğrulamasız seçim ve göstergesiz HPO (R1, Y21). Kol karşılaştırması ve ≥ 10 tohum Milestone 0''da (K25).
- **Milestone 0' kolları yalnız fiyat ölçeğinde ayrışıyor** (K25: `stats` ↔ `causal`). K26 = (a) olduğu için MACD, türbülans ve hacim iki kolda da Ansari tanımında; Y13 kıyası bunlarla karışmıyor.
- **Eşikler gürültüye yakın.** Havuzlu hacim %0,87 (test) ve %0,96 (doğrulama). Eşikler tasarım varsayımı ve tek bir pencereyle ölçülüyor. R1'in walk-forward katlarında kat başına raporlanmalı. Doğrulamada seçim yapan tek yol HPO; onun kaydı da `rl_eval_protocol`'den geçiyor (hakem-rl #4).
- **Göstergeler ham panelde, hayalet günler atılmadan hesaplanıyor.** Bu tüm göstergeler için önceden vardı; ADX de aynı yoldan geçiyor.
- **Canlı karar** (`build_live_state`) hâlâ legacy ADX'le, nedensel ölçeksiz ve 50/50 fiyat ölçeğiyle durum kuruyor. Meta'daki `indicator_spec` ve `price_norm` alanlarını okuması K13'ün işi (Y12). Meta hâlâ `price_stats`'ı da taşıyor; nedensel modelde canlı karar bunu kullanırsa gözlem bozulur (hakem-rl notu).
- **HPO** hâlâ ham OHLCV ve `stats` kipinde (Y21); nedensel kol R1'de HPO'ya taşınacak.
- **Tahmin hattı** legacy ADX'te kalıyor. `nan` + `wilder` modlarına geçiş Milestone 0' golden yenilemesiyle (K1).

**Etkileri:**
- Bu daldan sonra eğitilen rota modellerinin ADX'i gerçek değerler taşıyor. ODAS'ın 17, PGSUS'un 24 pencere içi satırı eksik hücre.
- K18 öncesi deneme kayıtları yeni maddeleri taşımadığı için açık sayılıyor (ihtiyatlı).

**Açık kalanlar:**
- **K26 kapandı (2026-11-09): (a), yalnız beyan** (Faz 10 §6). Ölçeksiz özellik tasarımı M2 `obs_spec`'inde.
- Canlı karar paritesi (K13).
- HPO'da nedensel kol ve göstergeler (R1).
- MARL `obs_spec` (M2; `obs_health` ve `dims_in_support` aynı ölçütü uygular).
- Milestone 0' etken tasarımı (M8).
- Eşik duyarlılığı (M8, R1).

### 9.10 Hakem paneli (K18, 2026-11-09)

| Hakem (model) | Okuma | Karar | Token | Araç | Süre |
|---|---|---|---|---|---|
| hakem-rl (sonnet) | ilk | Büyük revizyon | ~50 bin | 7 | ~1,5 dk |
| hakem-rl (sonnet) | ikinci (SendMessage) | Küçük revizyon | ~57 bin | 3 | ~0,5 dk |
| hakem-2 (opus) | son kapı | Büyük revizyon (RED yok) | ~143 bin | 39 | ~8,6 dk |
| hakem-2 (opus) | ikinci (SendMessage) | Kabul | ~165 bin (önceki bağlamla) | 8 | ~1,7 dk |

İlk okumalar ~250 bin token; ikinci okumalarla birlikte bildirimlerin toplamı ~415 bin. İkinci okuma bildirimi önceki bağlamı da sayıyor olabilir. Tek başına B9 hakem-2 ~301 bin harcamıştı; paket ve kapsam farklı olduğu için kıyas tam değil.

| # | Kaynak | Cid. | İtiraz | Yanıt |
|---|---|---|---|---|
| R1 | hakem-rl | BÜYÜK | Kapı yalnız tam sabiti arıyor; doygun türbülans (%99,7) temiz geçiyor | **Kabul, düzeltildi.** `obs_health` doygunluğu ölçüyor (≤ %90). Eski ortamın her kolu açık; karar K26 |
| R2 | hakem-rl | BÜYÜK | Eğitim ortamı bağlamsız (ilk 252 gün genişleyen pencere), test tam pencere | **Kabul, farklı çözümle.** Hakem bağlamı mümkün görmedi; CSV'de pencere öncesi veri var. Eğitim bağlamı `rl_context_before`. Pencere dolmadan geçen eğitim satırı %10,9 → yalnız seri başı (ODAS, PGSUS) |
| R3 | hakem-rl | BÜYÜK | B7 düzeltmesi yanlış negatif açıyor (eğitimde 1–2 satırı olan sembol) | **Kabul, düzeltildi.** Atlama yalnız ≥ 28 eğitim satırı olan sembolde; kontrol grubu testi |
| R4 | hakem-rl | KÜÇÜK | Doğrulamada `features_in_support` açık; yalnız test "temiz" | **Kabul.** Doğrulama sonuçları tabloda; doğrulamada seçim yapan tek yol (HPO) aynı protokolden geçiyor |
| R5 | hakem-rl | KÜÇÜK | MACD kolları iki etkende ayırıyor | **Kabul**, ara kol eklendi; hakem-2 #2 ile varsayılan da değişti |
| R6 | hakem-rl (2. okuma) | KÜÇÜK | Bağlamda bayat hücre atılmıyor | **Kabul, düzeltildi** |
| H1 | hakem-2 | BÜYÜK | `features_in_support` / `price_in_support` 30 sembolü havuzluyor; "blok ortalaması gizler" gerekçesi sembol düzeyinde de geçerli. SASA hacmi %23 destek dışıyken kapıdan geçiyor | **Kabul, düzeltildi.** Yeni madde `dims_in_support` (boyut başına ≤ %1). Boyut düzeyi sağlık kaydında ve JSON'da. "Fiyatı desteğe alıyor" ifadesi koşullandı |
| H2 | hakem-2 | BÜYÜK | MACD varsayılanı kullanıcının K25 kararını aşıyor; yorum KARAR metnine yazılmış | **Kabul, düzeltildi (b).** `price_norm_macd=False` varsayılan; K25 satırında karar ve uygulama notu ayrıldı; MACD K26'ya |
| H3 | hakem-2 | KÜÇÜK | `causal` etiketi ölçeğin gerçekte hangi pencereyle kurulduğunu denetlemiyor | **Kabul, düzeltildi.** `causal_short_context` (seri başı istisnası açık). Rota testi bununla gerçek bir açığı yakaladı: test bağlamı eğitim bağlamını içermiyordu |
| H4 | hakem-2 | KÜÇÜK | Pano uyarısı doygunluk kapısından önceki metinde | **Kabul, düzeltildi**; boyut düzeyi uyarısı da eklendi |
| H5 | hakem-2 | KÜÇÜK | §9.7 cümlesi düzeltilmemiş; eski dedektörün çıktısı JSON'da yok | **Kabul, düzeltildi.** JSON `b7_old_rule_eval_blocks`; iddia koşullu yazıldı (depoda gerçek panel kaydı yok); §9.7'ye not |
| H6 | hakem-2 | KÜÇÜK | `wilder` ilk değerle tohumlu; klasik Wilder'dan ilk satırlarda ayrışıyor | **Kabul, kodda düzeltildi.** SMA tohumu; referansla 27. satırdan itibaren < 1e-9 |
| H7 | hakem-2 | KÜÇÜK | "Varsayılan yollar main ile bit-eş" rota için doğru değil | **Kabul.** Kabul cümlesi "fonksiyon ve env varsayılanları; rotanın ADX'i bilinçli olarak değişti" oldu |
| H8 | hakem-2 | KÜÇÜK | Bağlam 252 tarih seçip sonra atıyor (etkin 242); sınır taranmıyor | **Kabul, düzeltildi.** Atımdan sonra sembol başına 252 satır; tarama bağlam + pencerenin ilk tarihi; ilk satırın öncülü dahil |

**hakem-2 ikinci okuma (SendMessage): Kabul.** H1–H8 karşılandı. hakem-2 JSON'daki boyut değerlerini kendi sembol başına ölçümüyle birebir buldu, H6'yı bağımsız referansla yeniden üretti (2,8e-14). İki yeni KÜÇÜK not düzeltildi:
- **N1:** pencere içinde listelenen sembol, bağlamda satırı olmadığı için seri başı listesine girmiyordu. Hata güvenli yöndeydi; gerçek panelde etkisi yok. Düzeltildi, test C'ye eklendi.
- **N2:** W=63 ifadeleri düzeltildi.

### 9.11 R1 — RL walk-forward bölmesi, kontrol noktası seçimi, ortak HPO yolu (2026-11-13, dal `feat/phase9-r1-walk-forward`)

**Kapsam.**
- Faz 10 §8 #8 → K20: RL katları, purge, doğrulamada kontrol noktası kuralı.
- Y21: HPO göstergesiz bir ortamda arıyordu.
- §9.6 #14: HPO → rota seçim soyu.
- Tasarım 2026-11-10'da kullanıcıyla onaylandı. Çıkarım birimi, MDE ve kol budama K20'ye göre M8'de.

**Tasarım kararları.** "Tasarım varsayımı" yazanlar kaynağa değil gerekçeye dayanıyor.

| Karar | Değer | Gerekçe ve kanıt |
|---|---|---|
| Bölme | `wf-v1` (özet `33a196c602976de1`): 6 kat, test = takvim yılı 2021..2026 (2026: 2026-11-01'e kadar, hariç), doğrulama = bir önceki yıl, eğitim 2013-06-01'den genişleyen (K17) | Sabit tarihli ve sürümlü ([evaluation/splits.py](../../evaluation/splits.py)). Eskiden 70/15/15 oranla bölünüyordu; panel uzadıkça sınırlar kayıyordu (Y17). Tanım değişmez, uzatma yeni sürümdür (wf-v2). Rota ve HPO pencereyi yalnız buradan alır |
| 2013-06 öncesi | Alınmadı | Pencereyi geri çekmek panelin en geç başlayan sembollerini düşürür: PGSUS 2013-04-26, ODAS 2013-05-21 (Faz 10 Ek A komut 1). Eski ortam etkin olmayan sembolü maskeleyemiyor. Kullanıcı notu: eski veri performansı iyileştirirse alınabilir. M2 aktiflik maskesiyle denenebilir |
| Purge | 0 seans | Eski ortamda ödül ufku bir adım; ortam kendi penceresinin son gününde bitiyor; genişleyen ve yalnız ileri giden walk-forward'da embargo sorusu doğmuyor. **Varsayılmadı, ölçüldü — kapsamıyla:** faz 1, basit ödül, eski ortam, iki kol; fiyat ve hacim kanalları (ortak ve sembol başına farklı sürüklenme); sentetik panel (test B) ve gerçek panelde 2026 katı (`real_b0.json`); düzeltilmiş panel sabit. Etiket örtüşmesi yok. **Ölçülmeyenler:** faz 2 (`fundamentals_pit` ile açık) ve düzeltme çarpanı kanalı (analitik; purge kapatmaz, aşağıda). Ufuk büyürse (Faz 10 K12, t+1 açılış) test kırılır ve değer yükseltilir |
| Kontrol noktası | Bütçe boyunca 10 eşit aralıklı nokta, sonuncusu bütçe sonu. Her noktada doğrulama bloğunda belirlenimci tek geçiş. Seçim: risksiz faiz üstü yıllık Sharpe'ın argmax'ı, eşitlikte en erken adım. Erken durdurma ve yeniden uyum yok | Tasarım varsayımı (N = 10; bir yıllık doğrulamada yıllık Sharpe'ın standart hatası kabaca 1). Kollar eşit ortam adımı alır. Test yalnız seçilen modelle, bir kez okunur. Seçimde kullanılan tahminin seçilen model için yanlı olması: Varma & Simon 2006; Cawley & Talbot 2010 (özetten) |
| Seçim metriği | `val_excess_sharpe`; risksiz faiz `evaluation/risk_free.py`'den (A8) | Seri yoksa rf = 0 ve kayıt açık (`selection_metric='val_sharpe_rf0'`); sabite sessizce düşülmez. Ortalama fazla getiri negatifken davranışı ölçüldü; tanı kayda giriyor (§9.12, kullanıcı kararı 2026-11-14) |
| HPO | (a) yalnız ilk katta: 2021 katının doğrulaması (2020) bütün testlerden önce bitiyor. Panel, kat ve ortam rotayla aynı. Amaç = en iyi kontrol noktasının doğrulama metriği. Budama aynı metrikten; bütçenin ilk yarısında budama yok | Y21 kapandı: HPO göstergeli CSV'yi, `wilder` ADX'i, veri kapısını ve nedensel kolu rotayla aynı yoldan alıyor. Ayar tohumu kayda giriyor; rota soyu ölçüyor (`hpo_lineage`). Ayar ve sınama tohumlarının ayrılması: Eimer vd. 2023 (özetten) |

**Deneme kaydı ([evaluation/trial_registry.py](../../evaluation/trial_registry.py)).**
- Yeni protokol maddeleri:
  - `split_from_spec`: pencereler kanonik tanımdan mı?
  - `selection_metric`: R1 öncesi kayıtlar bu maddeleri taşımadığı için açık sayılır.
  - `eval_window_unexposed`: değerlendirme penceresi R1 öncesi okunmuş bir pencereyle kesişmiyor mu (hakem-2 R1 #1).
  - `fundamentals_pit`: faz 2 temel oranları tek anlık görüntü (S4) → açık; faz 1 temiz (hakem-2 R1 #5, B4'ün RL tarafı).
- `test_reread` etiketi: aynı yapılandırma × tohum × kat × bölme × veri ikinci kez okununca konur. Kayıt N'ye sayılır ama kanıt sayılmaz; varsayılan okuma dışlar.
- Farklı katlar ya da bölme tanımları tek N/V'de karışmaz (`sharpe_trial_stats` hata verir). Katlar arası birleştirme kuralı M8'de.
- `hpo_lineage` maddeleri:
  - `hpo_window_after_test_start`
  - `hpo_seed_reused`, `hpo_seed_unknown`
  - `hpo_universe_mismatch`, `hpo_split_mismatch`, `hpo_obs_mismatch`
  - Ortam kurulmadan düşen bir HPO denemesinin `price_norm=None`'u soyu açar. Hata güvenli yönde; beyan edilir.

**Kabul.** `python tests/test_rl_walk_forward.py` → 90 kontrol, < 1 dk, CPU. Her grupta testi gerçekten kırdığı gösterilen bir mutasyon var.

| Grup | Ölçüt | Mutasyon |
|---|---|---|
| A bölme | wf-v1 sabit (özet). [eğitim başlangıcı, test bitişi) içindeki her seans tam bir blokta: örtüşme ve boşluk yok. Tamamlanmamış test ve eksik eğitim başlangıcı hata verir. `fold_coverage` 2018'de başlayan paneli "tamam" saymaz | Sınır bir gün kayınca özet değişir |
| B kat sınırı B0 | Sentetik OHLCV → gerçek gösterge hattı → `load_rl_panel` → `build_fold` → `make_fold_env`. Sınırdan sonraki fiyatlar değişince önceki ortamın gözlem + ödül + portföy izi bit-eş (`stats`, `causal`); sonraki ortamın izi değişir. Bozma iki aile: tüm sembollere ortak düzgün sürüklenme (O/H/L/C) ve sembol başına farklı sürüklenme + hacim (hakem-2 R1 #4) | (i) Eğitime doğrulamanın ilk seansı eklenir; (ii) ADX bir gün ileriden okunur. İkisi de yakalanır |
| C test seçime sızmıyor | Rota uçtan uca: test bloğu değişince seçim (10 noktanın metrikleri) ve seçilen ağırlıklar bit-eş, test metriği değişir. Aynı koşum tekrarı bit-eş ve `test_reread` alır. Seçim son nokta değilse kaydedilen model seçilen noktanınki | Doğrulama ortamı test bloğundan kurulursa seçim değişir |
| D geri çağrı | Geri çağrı açık/kapalı son ağırlıklar bit-eş (A2C, PPO, TD3, SAC; PPO'da ölçüm rollout ortasına düşer) | Stokastik ölçüm |
| E ortak yol | Aynı kat ve kolda HPO'nun eğitim ve doğrulama ortamları rotanınkilerle bit-eş iz üretir (`stats`, `causal`) | HPO ADX'i yeniden hesaplamazsa iz ayrışır |
| F AST | Rota eğitim fonksiyonu, HPO optimizer'ı ve HPO rotasında `split_data` ve `raw_stock_data` yok; `trading.py`'nin tamamında `split_data` yok | Bekçi öz-testi |
| G kayıt | Kat karışımı hatası, `test_reread`, `hpo_lineage` maddeleri. wf-v1 rota kaydında `split_from_spec=True`; rf yoksa `selection_metric` açık | Her madde tek değişiklikle tetiklenir |
| H uçlar | HPO'da eski tarih alanı → 400, bilinmeyen kat → 400. `/train`'de bilinmeyen kat → 400 ve görev başlamaz | — |

Faz 9 test seti (12 test) ve `test_auth` aynı sayılarla yeşil.

**Ölçüm (eğitimsiz).**
- Komut: `python scripts/analysis/measure_r1_folds.py --old-panel data/bist/backup_2026-07-27/raw_stock_data.csv --dividends --old-indicator-panel data/bist/backup_2026-07-27/stock_data_with_indicators.csv --out docs/development/measurements/phase9-r1/folds.json`
- Ölçüm commit `27c4887` üzerinde, temiz ağaçta koşuldu.
- Gösterge CSV'si 2005-01-03..2026-11-09; frame SHA-256 `7a9f5d4d…`, dosya `3aec450d…`. RL paneli `008e3b11…`.
- Ortam ayarları eğitim isteğinin varsayılanları (faz 1, basit ödül).
- Model eğitilmediği için `selection_metric` ölçülemez; tablodaki açık maddeler JSON'daki `open_items_env`.

| Kat | Seans (eğitim / doğrulama / test) | `stats`: fiyat destek dışı (doğr. / test) | `causal`: fiyat destek dışı (doğr. / test) | `causal`: açık maddeler (doğr.; test) |
|---|---|---|---|---|
| 2021 | 1.651 / 251 / 249 | %20,0 / %51,0 | %1,96 / %2,69 | price, obs, features, dims; price, obs, features, dims |
| 2022 | 1.902 / 249 / 252 | %35,6 / %81,7 | %2,11 / %2,26 | price, obs, features, dims; price, obs, features, dims |
| 2023 | 2.151 / 252 / 249 | %57,3 / %96,7 | %1,63 / %0,28 | price, obs, features, dims; obs, features, dims |
| 2024 | 2.403 / 249 / 249 | %49,1 / %83,9 | %0,01 / %0,00 | obs, features, dims; obs, features, dims |
| 2025 | 2.652 / 249 / 251 | %60,0 / %71,1 | %0,00 / %0,00 | obs, features, dims; obs, features, dims |
| 2026 | 2.901 / 251 / 207 | %22,2 / %48,3 | %0,00 / %0,04 | obs, dims; obs, features, dims |

*Kısaltmalar: price = `price_in_support`, obs = `obs_health`, features = `features_in_support`, dims = `dims_in_support`.*

**Okuma.**
- Her kat ve kolda 30 sembol var. Düşen sembol ve ölçek yedeği yok; 6 katın hepsi panelde tamam (`fold_coverage`).
- **Hiçbir kat ve kol protokolde temiz değil.**
  - Türbülans eğitimde ve değerlendirmede %99,6–100 doygun (`obs_health`). K26 = (a) olduğu için bu kalıcı.
  - MACD (eski tanım) her katta sembol düzeyinde destek dışı; en kötü sembol %15 (EKGYO, 2026 doğrulama) ile %90 (SASA, 2021 test) arasında.
- **Kollar yalnız fiyatta ayrışıyor.** Hacim, MACD, RSI, CCI, ADX ve türbülansın destek dışılığı her katta iki kolda birebir aynı (JSON). Bu K25/K26 tasarımıyla tutarlı.
- **`stats` kolu:** test fiyatlarının %48–97'si eğitim desteği dışında (B9'un bedeli, §9.5). Oran kattan kata değişiyor.
- **`causal` kolu:** destek erken katlarda dar.
  - 2024–2026'da fiyat destek içinde (≤ %0,04).
  - 2021–2022'de (doğrulama ve test) ve 2023 doğrulamada %1,6–2,7 destek dışı; `price_in_support` açık.
  - K18'deki tek bölmede (test 2024-08-06..2026-07-27) bu oran %0,03'tü.
  - Eğitim penceresi kısaldıkça nedensel z'nin desteği daralıyor. §9.9'daki "destek içinde kalması garanti değil" uyarısı burada kat başına ölçülmüş oldu.
- **Eşik duyarlılığı.** Havuzlu hacim destek dışılığı kat başına %0,01 (2025 doğrulama) ile %6,6 (2021 doğrulama) arasında. Sembol düzeyinde SASA hacmi 2026 testinde %60,9.
  - Havuzlu eşik kat seçimine duyarlı. §9.9'daki "eşikler gürültüye yakın" notu kat başına doğrulandı.
  - Eşik duyarlılığı M8'de.
- **Nedensel eğitim bağlamı** (2012-05-22..2013-05-31) K18'le birebir aynı: 18 hayalet gün, 12 bayat hücre. Pencere dolmadan geçen eğitim satırı 492, nötr satır 28; seri başı yalnız ODAS ve PGSUS.

**Okunmuş test pencereleri (hakem-2 R1 #1; kullanıcı kararı 2026-11-14).**
- 2024-08-06..2026-07-27, R1'den önce başarım metrikleriyle okundu: §9.5, B9 etki ölçümü (A2C/PPO × 3 tohum, fazla getiri Sharpe'ı, al-tut). Okunan sonuçlar K25 ve K26 kararlarına girdi.
- Kayıt bunu ölçüyor: yeni madde `eval_window_unexposed` (`EXPOSED_EVAL_WINDOWS` ile kesişim). 2024, 2025 ve 2026 katlarının test kayıtları bu yüzden açık; 2025 ve 2026 katlarının doğrulama yılları da bu pencerede.
- R1 öncesi oranlı bölmeyle yapılan eğitimlerin (23 metrik dosyası) test aralığı kayıtlı değil; okunan dönemler ölçülemiyor, büyük olasılıkla panelin son %15'i, yani 2023 sonrası. Bu yüzden 2021–2023 katları "kanıtlanmış okunmamış" değil, "bilinen en az okunmuş" katlar.
- Okunmadığı kesin tek gerçek veri 2026-07-28 sonrası: eski panel 2026-07-27'de bitiyordu ve R1 ölçümleri bu dönemde başarım okumadı.
- M8: doğrulayıcı katlar 2021–2023; ön kayıttan sonra gelen veri gerçek sınama olarak ayrılır.

**Veri tazelemesi (2026-11-11, kullanıcı: tam yeniden indirme).**
- **Panel.**
  - Ham ve göstergeli CSV 2005-01-03..2026-11-09. yfinance 2005 öncesini artık vermiyor; eski panel 2000-05-10'da başlıyordu.
  - Eski panelin yedeği `data/bist/backup_2026-07-27/`; yenisinin yedeği `data/bist/backup_2026-11-11/` (SHA256SUMS ile).
  - Fiyatı ≤ 0 satır: eskide 8.155, yenide 0.
  - Tam panelde (JSON `gate.full_panel`): hayalet gün 117, bayat hücre 585, eksik hücre %5,33, artefakt 10 (8 `ohlc_inconsistent`, 2 `extreme_return`; hepsi pencere dışında), büyük hareket 27.
- **RL penceresi** (JSON `gate.window`):
  - 102.900 satırdan 100.834'e; 3.365 tarih.
  - Hayalet gün 67, bayat hücre 39, eksik hücre %0,115, artefakt 0, büyük hareket 14.
  - Isınma atımı ODAS 17, PGSUS 24.
- **Fark veriden mi koddan mı** (JSON `gate_same_code`: aynı güncel kapı, eski panel ↔ yeni panelin 2026-07-27'ye kesilmişi):
  - Satır (98.614), tarih (3.291), eksik hücre (%0,117), bayat hücre, ısınma atımı ve büyük hareket (13) **birebir aynı**.
  - Tek fark hayalet gün: 72 ↔ 67. Eksilen beş gün 2026 tatilleri (23 Nisan, 1 Mayıs, 27–29 Mayıs Kurban Bayramı). Eski panelde bu günlere ait 92 düz satır vardı; yfinance bunları artık vermiyor.
  - Önceki notlardaki "eksik hücre %0,076 → %0,115" farkı **koddan**. %0,076 §9.1'de, gösterge ısınma atımları (B7, K18) eklenmeden önce ölçülmüştü: 116 − (ODAS 17 + PGSUS 24) = 75 hücre = %0,0760.
- **Fiyatlar** (JSON `panel_refresh`; pencere 2013-06-01..2026-07-27, 100.680 ortak satır):
  - 26 sembolde kapanış oranı 1 ve getiri farkı ≤ 1,1e-6, yani yuvarlama.
  - BIMAS (0,99399), TAVHL (0,99354) ve TUPRS (0,98240) sabit çarpanla ölçeklenmiş. Üçü de eski indirmeden sonra temettü almış: 2026-09-16, 09-22, 09-30 (yfinance, JSON `panel_refresh.dividends`).
  - SKBNK'nin oranı 1,00955'ten 1'e dört basamakta iniyor: 2023-04-10, 2024-05-13, 2025-06-24, 2026-06-08. Bunlar yfinance'teki SKBNK temettü tarihleri; aynı geçmiş temettülere iki indirmede farklı düzeltme çarpanı uygulanmış. Sağlayıcının neden revize ettiği doğrulanamadı.
  - Günlük log-getiri yalnız bu 4 günde farklı (en fazla 0,0041).
- **Ders:** auto_adjust edilmiş yfinance paneli değişmez değil; tam indirme geçmişi geriye dönük değiştirebiliyor. Bir sonuç yeniden üretilecekse veri yeniden indirilmez, anlık görüntü (C3) kullanılır. Veri kapısı tek paneldeki bozukluğu yakalar; iki indirme arasındaki farkı panel SHA-256'sı ayırır.
- B9 ve K18 ölçüm JSON'ları eski panelin SHA'sına ait ve tarihsel kayıt olarak değişmez.
- **Düzeltilmiş fiyat ve nokta-zamanlılık (analitik, ölçülmedi).** `Ticker.history` varsayılanı `auto_adjust=True` ([data_fetcher.py:164](../../data/data_fetcher.py#L164); yfinance 0.2.66). Çarpımsal düzeltmede p̃ₜ = pₜ · Πⱼ﹥ₜ fⱼ = C · TRₜ; burada TRₜ = pₜ / Πⱼ≤ₜ fⱼ, t'ye kadarki bilgiyle kurulan toplam getiri endeksi, C = Πⱼ fⱼ ise indirme tarihine kadarki tüm temettü ve bölünme çarpanları (sembol başına tek sabit). Türetim **bir indirme içinde ve fⱼ'ler sabitken** geçerli: sağlayıcı geçmiş fⱼ'leri sonradan revize edebiliyor (SKBNK, yukarıda; hakem-finans R1 #3). TRₜ temettüyü ex-tarihinde, brüt (stopajsız) ve gecikmesiz yeniden yatırıyor; getiride küçük bir yukarı yanlılık olarak beyan edilir (#5).
  - Fiyatta sıfırıncı dereceden homojen özellikler (getiri, aynı seriden kurulan `stats` z'si, nedensel kayan z, RSI, CCI, ADX) C'yi taşımaz: gelecek temettü bilgisi sadeleşir. MACD birinci dereceden homojen; bu yüzden ayrı.
  - Ölçeğe bağlı olanlar C'yi taşır: TL cinsinden MACD (`tanh(macd/0,1)`), pay adediyle işlem büyüklüğü (`max_shares_per_trade`) ve portföydeki pay sayıları. Bunlar K26'da zaten ölçeğe bağlı diye beyanlı; burada eklenen, ölçeğin gelecekteki temettüleri de içermesi. Pay sınırının TL karşılığı bu yüzden sembole, tarihe ve indirmeye göre değişiyor; D1 baseline'ları (1/N, al-tut) bu sınırla kısıtlanmıyorsa karşılaştırma kayar ve bu D1'de beyan edilir (#4). Ölçeksiz gözlem ve ağırlık tabanlı eylem M2'de (K26, Y6).
  - **Kat sınırıyla ilişkisi (hakem-2 R1 #6):** C, sınırdan sonra (doğrulama ve test döneminde) ödenen temettüleri de içerir; ölçeğe bağlı kanallar bu bilgiyi eğitim ortamına taşır. B0 düzeltilmiş paneli sabit tuttuğu için bu kanalı kurgu gereği göremez, purge da kapatmaz. Kapanışı M2'nin ölçeksiz gözlemi. Öncülün (yfinance'in geriye dönük çarpımsal çarpanı) ucuz bir ölçümü mümkün; yapılmadı.

**Bu paketin bulguları.**
1. Panoda HPO doğrulama penceresi CSV'nin sonuna uzanıyordu; rotanın test dönemini içeriyordu. **Kapandı:** kat seçici.
2. HPO budaması fiilen ölüydü: rapor adımı 0, 1, 2…, `n_warmup_steps=10000`. **Kapandı:** rapor adımı kontrol noktası sırası, değer doğrulama metriği.
3. HPO'nun Optuna sampler'ı tohumsuz. API `seed=None` veriyor; ayar tohumu `optimize()`'da üretilip modellere geçiyor, ama sampler kurucuda tohumsuz kuruluyor. Arama tekrar üretilemez. **Açık.**
4. `/train` bilinmeyen katı arka plan görevinde fark ediyordu. **Kapandı:** 400.
5. Veri uçlarının kat kapsamı yalnız test bitişine bakıyordu; 2018'de başlayan paneli her katta "tamam" gösterirdi. **Kapandı:** `fold_coverage` eğitim başlangıcını da denetliyor.
6. **Plan dışı, veri kaybı riski.** Panodaki "legacy veri üret" düğmesi tarihleri JSON gövdesinde gönderiyor, `/data/generate` ise sorgu parametresi bekliyor. Her basış faz 1'in 5 sembolüyle 2018'den başlayan bir panel üretip ham ve göstergeli CSV'nin ikisinin de üzerine yazıyor; sonrasında wf-v1'in hiçbir katı kurulamaz. Yedek alındı. **Öneri (kullanıcı onayı, R1 sonrası ayrı iş):** düğme ve uç kaldırılır; doğru yol Veri sayfası "Yeniden İndir (tam)".
7. Literatür hattı: `resolve.py` toptan yeniden çözümde tablonun atıf sayılarını değiştiriyor. R1 kaynaklarından yalnız üç kayıt birleştirildi (literature-review-verification §8.5); toptan yenileme ayrı iş.

**Kapanan açıklar.** §9.9 "HPO'da nedensel kol ve göstergeler (R1)"; §9.5 Y21; §9.6 #14 (soy kayıt tarafı: `hpo_lineage`); §9.2 #3 (tek bölme, test yapılandırma başına bir kez: `test_reread`).

**Etkileri.**
- Rota artık son adımdaki modeli değil, doğrulamada seçileni kaydedip sınıyor. Meta, manifest ve kayıt `split_spec`, `fold`, `selected_step` ve `selection_metric` taşıyor.
- Varsayılan kat, panelin tamamladığı son kat (bugün 2026).
- HPO API'si eski tarih alanlarını 400 ile reddediyor. CLI'da evren varsayılanı rotanın evreni (panelin tüm sembolleri).
- R1 öncesi deneme kayıtları açık sayılıyor (ihtiyatlı).
- K18 ölçüm betiği (`measure_obs_health.py`) bilerek `split_data`'da kaldı: §9.9 sayılarını yeniden üreten arşiv ölçümü.

**Açık kalanlar.**
- HPO sampler tohumu (bulgu 3).
- `/data/generate` (bulgu 6), R1 sonrası.
- Çıkarım birimi, katlar arası birleştirme, MDE, kol budama (M8, K20).
- Kat başına HPO yok (karar (a)); maliyeti kat sayısı kadar.
- Eşik duyarlılığı (M8).
- GPU belirlenimciliği (§9.6 #5; testler CPU'da).
- 2013-06 öncesi veri (M2 aktiflik maskesi).
- Canlı karar paritesi (K13): meta `fold` ve `selected_step` taşıyor, canlı karar okumuyor.
- Seçim protokolü kaynaklarının tam metni (verification §8.4).
- Seçim kuralının test karşılaştırması (argmax ↔ son nokta) → M8 ön kaydı (§9.12).
- Okunmuş pencere listesi R1 öncesi oranlı bölmeleri içermiyor (aralıkları kayıtlı değil, ölçülemiyor).
- Faz 2 RL `fundamentals_pit` ile açık (B4).
- Nakit getirisi: ortamda nakit 0 kazanıyor, seçim metriği rf düşüyor → §9.2 #7, G paketi (§9.12 F2).

### 9.12 Hakem paneli (R1, 2026-11-13/14)

| Hakem (model) | Okuma | Karar | Token | Araç | Süre |
|---|---|---|---|---|---|
| hakem-rl (sonnet) | ilk | Küçük revizyon | ~57 bin | 9 | ~1 dk |
| hakem-finans (sonnet) | ilk | Küçük revizyon | ~66 bin | 9 | ~2 dk |
| hakem-2 (opus) | son kapı | Büyük revizyon (RED yok; yeniden tasarım gerekmiyor) | ~123 bin | 33 | ~7,9 dk |

**Paneli izleyen ölçüm.** Komut: `python scripts/analysis/measure_r1_selection.py --folds 2021,2025 --timesteps 50000 --b0-fold 2026 --out docs/development/measurements/phase9-r1/selection.json`. Koşum commit `7ab7a33` üzerinde, temiz ağaçta yapıldı.
- Kurulum: gerçek panel, rotanın A2C varsayılanları, CPU, tek tohum (7). Deneme kaydına yazılmadı; test bloğu kurulmadı.
- **Seçim metriği.**
  - 2021 katında (doğrulama 2020, rf %10,0) 10 noktanın 0'ının ortalama fazla getirisi negatif. rf burada 252 × seans ortalaması (≈ sürekli bileşik); yıllık basit karşılığı ≈ %10,5.
  - 2025 katında (doğrulama 2024, rf %40,5; yıllık basit ≈ %49,9) 10 noktanın **10'u** negatif.
  - 2025'te dört sıralama da (fazla getiri Sharpe'ı, rf = 0 Sharpe, ortalama fazla getiri, işarete duyarlı sıralama) aynı noktayı, yani son noktayı seçti. **Ama bu kat sorunu ayırt edemez:** 10 noktanın 9'unun oynaklığı %22,8–25,4 (nokta 0: %15,4). Oynaklıklar bu kadar yakınken Sharpe ile ortalama sıralaması zaten çakışır (hakem-2 R1 #2).
  - 2021'de dört sıralama üç farklı nokta seçiyor: fazla getiri Sharpe'ı 8, rf = 0 Sharpe 4, ortalama fazla getiri 0, işarete duyarlı 8. rf = 0 Sharpe'ın seçtiği 4. nokta düşük oynaklıklı (%4,3, büyük ölçüde nakit): F2 asimetrisi bu katta seçimi fiilen değiştiriyor.
  - Doğrulama tarafında seçilen − son: 2021'de 1,423 − 1,384 = 0,039; en iyi üç nokta (0, 8, 9) 0,07 içinde. 2025'te seçilen nokta son noktanın kendisi (fark 0). Bir yıllık doğrulamada Sharpe'ın standart hatası ~1 iken argmax gürültü düzeyindeki noktalar arasından seçiyor.
  - İşarete duyarlı sıralama yalnız sıra olarak kullanıldı; değeri yıllıklaştırılmış bir metrik değil.
  - Her katta 10 noktanın ağırlıkları farklı. Callback'in metriği bağımsız yeniden hesapla birebir.
  - Tek tohum ve tek algoritma olduğu için bu bir tanıdır, genelleme değil.
- **Gerçek panel B0** (2026 katı; ham fiyat doğrulama ya da test başından itibaren değiştirildi, CSV'yi üreten gösterge hattı yeniden koşuldu): iki kolda ve iki sınırda sınırdan önceki ortamın izi bit-eş, sonraki ortamın izi değişiyor (JSON `real_b0`). İkinci ölçüm geniş aileyle (sembol başına farklı sürüklenme + hacim): `docs/development/measurements/phase9-r1/real_b0.json`, sonuç aynı. Komut: `python scripts/analysis/measure_r1_selection.py --folds "" --b0-fold 2026 --out docs/development/measurements/phase9-r1/real_b0.json` (kod `4c2b995`). O koşumda ağaçta yalnız iki belgenin yeniden adlandırılması vardı (kullanıcı, kod dışı); JSON bu yüzden `git_dirty=true`.

**Kullanıcı kararları (2026-11-14).**
- **Seçim metriği korunur** (M8 birincil metriğiyle tutarlı). Her kontrol noktasına yıllık ortalama fazla getiri (`mean_excess`) yazılır; seçim raporu negatif ortalamalı nokta sayısını (`n_negative_mean_excess`) ve son noktanın doğrulama metriğini (`last_metric`) taşır. Nakit getirisi (ortamda 0) zaten §9.2 #7'de açık iş: G paketinde nakit faiz tahakkuku ya da varsayım beyanı (M2 ortamı aynı kararı devralır).
- **Son kontrol noktasının test karşılaştırması M8'e.** "Test bir kez" kuralı korunur. "Seçim kuralı: argmax ↔ son nokta" M8 ön kaydında önceden tanımlı bir etken olur; her kural kendi kaydıyla bir kez okunur ve N'ye sayılır. R1'de yalnız doğrulama tarafındaki fark raporlanır.

| # | Kaynak | Cid. | İtiraz | Yanıt |
|---|---|---|---|---|
| R1 | hakem-rl | BÜYÜK | Seçimin değeri ölçülmüyor; son nokta referansı yok | **Kısmen.** Test bir kez okunur (Varma & Simon 2006; Cawley & Talbot 2010). Doğrulama tarafındaki fark `last_metric` ile raporlanıyor; test tarafı M8'de (karar yukarıda) |
| R2 | hakem-rl | BÜYÜK | Pasif referans (al-tut, nakit) yok | **Plan D1** (1/N, al-tut, XU030); R1 kapsamı dışında. Paketin bilinen açıklar listesinde eksikti |
| R3 | hakem-rl | BÜYÜK | `stats` kolunda seçim destek dışı doğrulama gözlemleriyle yapılıyor | **Kabul, beyan.** Bu seçimi zayıflatır ama testin ölçümünü yanlı kılmaz; `protocol_val` bilgi alanı olarak kalır (K18 #4). `stats` kayıtları zaten `price_in_support` ile açık. Kol karşılaştırmasında referans `causal` (Milestone 0'). Nedensel kol da 2021–2023 doğrulamasında `price_in_support` ile açık (%1,63–2,11); bu referans erken katlarda sınırlı (hakem-2) |
| R4 | hakem-rl | KÜÇÜK | Güncelleme arasına düşen noktalar aynı modeli ölçüyor | **Kabul, düzeltildi:** `params_sha` + `n_distinct` (test C, I) |
| R5 | hakem-rl | KÜÇÜK | 2021 katında HPO ve kontrol noktası seçimi aynı doğrulama yılını kullanıyor | **Beyan.** Test dokunulmuyor; doğrulama metriği kanıt olarak kullanılmıyor ve çift seçimle iyimser olduğu biliniyor |
| R6 | hakem-rl | KÜÇÜK | Tek tohum; seçim kararlılığı ölçülmedi | **Beyan.** R1 performans sayısı üretmiyor; ≥ 10 tohum Milestone 0''da |
| R7 | hakem-rl | KÜÇÜK | Katlar arasında bütçe/seans oranı sabit değil (1.651 → 2.901 seans) | **Beyan,** M8 (bütçe ölçekleme ya da kol) |
| R8 | hakem-rl | KÜÇÜK | TD3/SAC bit-eşliği test edilmemiş; `learning_starts` öncesi noktalar eğitilmemiş politikayı ölçüyor | **Kabul.** Test D TD3 ve SAC'ı da kapsıyor, ikisi de bit-eş. `learning_starts` (rota varsayılanı 1000) öncesine düşen noktalar beyan edilir |
| F1 | hakem-finans | BÜYÜK | Ortalama fazla getiri negatifken Sharpe oynaklığı ödüllendirir | **Ölçüldü, ayırt edici değil** (hakem-2 #2): 2025'te 10/10 negatif ama oynaklıklar dar (%22,8–25,4), kat sorunu ayıramıyor. 2021'de rf = 0 Sharpe başka nokta seçiyor (F2). Karar: metrik korunur; tanı kayda girer (`mean_excess`, `n_negative_mean_excess`, `pick_mean_excess_step`, `picks_agree`) |
| F2 | hakem-finans | BÜYÜK | Nakit 0 kazanırken metrik rf düşüyor (asimetri) | **Kabul, beyan.** Ortam değişmez (davranış dondurma); `sharpe_rf0` her noktada kayıtlı. Nakit faizi §9.2 #7'de zaten açık (G paketi) |
| F3 | hakem-finans | KÜÇÜK | Düzeltilmiş fiyat türetimi fⱼ'nin sonradan değişmemesine bağlı | **Kabul,** §9.11 metni "bir indirme içinde ve fⱼ sabitken" diye sınırlandı |
| F4 | hakem-finans | KÜÇÜK | Pay sınırının TL değeri değişken; baseline'larla karşılaştırma kayar | **Kabul, beyan** (§9.11); D1'de |
| F5 | hakem-finans | KÜÇÜK | Temettü brüt ve gecikmesiz yeniden yatırılıyor | **Kabul, beyan** (§9.11) |
| F6 | hakem-finans | KÜÇÜK | Tek tohum; seçilen ve son noktanın testi | R1 ve R6 ile aynı (M8) |
| F7 | hakem-finans | KÜÇÜK | Takvim yılı blokları rejime özgü; arife yarım günleri hacmi yanlı yapar | **Beyan,** M8 raporu. Yarım gün işareti M2 `obs_spec`'inde |
| FS1 | hakem-finans (soru 1) | — | Kat sınırı B0'ın gerçek panelde de doğrulanması | **Yapıldı:** `real_b0` bit-eş |
| FS5 | hakem-finans (soru 5) | — | SHA tek başına yetmez; dosyanın kendisi korunmalı | **Kabul.** Yedekler (`data/bist/backup_*`) yerel ve git dışında. Makine dışı kopya kullanıcı işi (açık) |
| H1 | hakem-2 | BÜYÜK | 2024–2026 katlarının testi R1 öncesi başarımla okundu (B9) ve sonuçlar K25/K26'ya girdi; `test_reread` protokoller arası okumayı yakalamıyor | **Kabul (kullanıcı kararı 2026-11-14: ölçülen bayrak + beyan).** Yeni madde `eval_window_unexposed` (`EXPOSED_EVAL_WINDOWS`); 2024–2026 kayıtları açık; R1 öncesi oranlı okumalar ölçülemiyor, beyan (§9.11). M8: doğrulayıcı katlar 2021–2023 |
| H2 | hakem-2 | BÜYÜK | F1 yanıtı kanıttan fazlasını söylüyor: 2025'te oynaklıklar dar, oynaklık aralığı yanlış yazılmış, 2021 seçimleri eksik, F2 bağı kurulmamış | **Kabul, düzeltildi.** Metin "ayırt edici değil"; aralık %22,8–25,4 (nokta 0: %15,4); 2021 seçimleri 8/4/0/8; 4. nokta F2'nin gözlenen etkisi. Rapora `pick_mean_excess_step` ve `picks_agree` (test C, I) |
| H3 | hakem-2 | KÜÇÜK | rf etiketi: 252 × seans ortalaması ≈ sürekli bileşik, yıllık basit değil | **Kabul, düzeltildi** (yıllık basit ≈ %10,5 / %49,9) |
| H4 | hakem-2 | KÜÇÜK | B0 bozması dar ailede: ortak düzgün sürüklenme, hacim yok | **Kabul, düzeltildi.** Test B ve `real_b0.json`'da sembol başına farklı sürüklenme + hacim; iki kol, iki sınır bit-eş |
| H5 | hakem-2 | KÜÇÜK | "Purge 0 ölçüldü" yalnız faz 1 / basit ödül için; faz 2'nin tek anlık temel oranları protokolde değil | **Kabul.** Purge iddiasının kapsamı yazıldı; yeni madde `fundamentals_pit` (B4'ün RL tarafı) |
| H6 | hakem-2 | KÜÇÜK | Düzeltme çarpanı kanalı B0'ın kapsamı dışında; metin sınırla ilişkilendirmiyor | **Kabul.** §9.11'de "etiket örtüşmesi yok (ölçüldü)" ile "düzeltme çarpanı kanalı (analitik; purge kapatmaz; M2)" ayrıldı |
| H7 | hakem-2 | KÜÇÜK | §9.11 eskimiş (75 kontrol, D grubu yalnız A2C/PPO) | **Kabul, düzeltildi** (90 kontrol; A2C, PPO, TD3, SAC) |
| H8 | hakem-2 | KÜÇÜK | Doğrulama tarafındaki fark sayı olarak yazılmamış | **Kabul.** 2021: 1,423 − 1,384 = 0,039, en iyi üç nokta 0,07 içinde; 2025: fark 0. Argmax gürültü düzeyinde seçiyor (beyan) |
| H9 | hakem-2 | KÜÇÜK | İşarete duyarlı Sharpe kaynaksız, yıllıklaştırması tutarsız | **Kabul.** Yalnız sıra olarak raporlanıyor (sıra yıllıklaştırma sabitinden bağımsız); kural değil, tanı. Kaynak eklenmedi |

---

## Versiyon geçmişi

- **2026-10-07:** İlk plan. 66 tam metin arşivlendi; 10 yeni bulgu (Y1–Y10) kod ve makaleyle doğrulandı; A–G geçerlilik paketleri, Makale 2 deney tasarımı, M1–M8 MARL kararları, 6 açık karar. **Kod değişikliği yok.**
- **2026-11-06:** Plan kullanıcı tarafından onaylandı (Faz 10 eklemeleriyle). §7 Karar 1 ve 2 işlendi. İlk kapı V + A + C uygulandı (§9.1).
- **2026-11-06 (kod incelemesi):** hakem-2 "Büyük revizyon" verdi; 15 itiraz §9.2'de (14 kabul, 1 kısmen).
  - Kayda protokol işareti eklendi.
  - N/V türetmesi düzeltildi.
  - Hayalet gün tanımı düzeltildi; ilk sayılar geçersiz.
  - A2, C2 ve V "Kısmen"e indi.
- **2026-11-06 (B2/B3/B1/A8):** EVDS teşhisi — dört hata (yanlış seri kodları, alt çizgili sütun, 'YYYY-M' tarih, 1000 satır sınırı). Makro artık nokta-zamanlı ve `bfill`'siz. Risksiz faiz TP.APIFON4'ten (§9.3).
- **2026-11-06 (hakem-2 B2):** "Büyük revizyon"; 14 itiraz §9.4'te.
  - Artımlı çekimdeki sabit 50 geri dönüşü (birleştirmeyi engelliyordu) düzeltildi.
  - B1/B3 testlerinin içi boştu; mutasyonla doğrulanır hale getirildi.
  - Bitiş kesimi, 14 aylık erken çekim ve çapraz varlık `bfill`'i düzeltildi.
  - Açık: vintage kaydı, politika faizi ölçüm hatası, risksiz faiz gerekçesi, TÜİK takvimi, RL'de aynı gün FX, `pit_version`.
- **2026-11-06 (B9):** val/test ortamı eğitim penceresinin fiyat ölçeğini kullanıyor (§9.5).
  - Kapsam: rota, HPO, akademik rapor ve tüm held-out çağıranlar.
  - Protokol etiketi artık ölçülüyor; ölçek kayda geçmezse kayıt kendiliğinden açık kalıyor.
  - Bedeli ölçüldü: test fiyat özelliklerinin %74,7'si kırpılıyor.
  - Yeni bulgu Y21: HPO göstergesiz bir ortamda ayar arıyor; düzeltme R1'de.
- **2026-11-06 (hakem-2 B9):** "Büyük revizyon", 14 itiraz §9.6'da.
  - Etiket artık ölçeğin kökenini ölçüyor.
  - Doygunluk `price_in_support` ile makine tarafından uygulanıyor.
  - Etki iddiası geri çekildi ("tanımlanamadı"); tablo fazla getiri Sharpe'ına geçti.
  - Kabul t ∈ {0, orta, T−2}; main ile 11 yapılandırmada bit-eşlik.
  - Hiperparametre kökeni kayda geçiyor; rapor D5'e kadar duruyor.
  - Ölçüm JSON'ları depoda arşivlendi.
  - Açık: GPU belirlenimciliği (#5), HPO → rota seçim soyu (#14); destek içi üçüncü kol için karar K25.
- **2026-11-06 (B7):** gösterge ısınma `bfill`'i bayrakla kalktı; RL kapısı bfill bloğunu atıyor (gerçek panelde ODAS 10, PGSUS 17 satır). Protokolde `indicator_bfill` artık ölçülüyor. K18 için bulgu: ADX hesabı tümüyle NaN üretiyor (§9.7). hakem-2 (§9.8): "Büyük revizyon"; dedektör türbülansın nedensel 0 başını blok sanıyordu, düzeltildi. Paketli inceleme ~144 bin token.
- **2026-11-09 (K18 + K25 b):** K25 kararı (b). ADX'in kök nedeni indeks hizalaması; `wilder` bayrağı rota varsayılanı oldu (iki bağımsız referans). Eski ortama opt-in nedensel fiyat ölçeği eklendi (`price_norm='causal'`, W=252; MACD aynı ölçekle). Yeni ölçülen protokol maddeleri: `obs_health` ve `features_in_support`. Gerçek panelde nedensel kol test fiyatlarının havuzlu destek dışılığını %98,9'dan %0,03'e indiriyor. Doğrulamada sembol düzeyinde aşım sürüyor (`dims_in_support`). Türbülans doygunluğu yüzünden eski ortamın hiçbir kolu `obs_health`'ten geçmiyor; MACD, türbülans ve hacim için karar K26 önerildi. B7 ölçütünün yanlış pozitifi bulunup düzeltildi. Panel (§9.10): hakem-rl iki okuma (Büyük → Küçük revizyon), hakem-2 Büyük revizyon (RED yok). Düzeltmeler: boyut düzeyi kapı, MACD varsayılanı geri alındı, `causal_short_context`, klasik Wilder tohumu, sembol başına bağlam.
- **2026-11-09 (K26):** kullanıcı kararı (a), yalnız beyan. Eski ortam sadık Ansari replikasyonu olarak kalıyor; ölçeksiz özellik adayları (`tanh(macd/σ_t)`, `tanh(turb/n)`) M2 `obs_spec`'ine taşınıyor. Gerekçe Faz 10 §6 K26.
- **2026-11-13 (R1):** RL walk-forward bölmesi `wf-v1` (6 kat, test 2021..2026, purge 0 kat sınırında B0 ile ölçüldü), doğrulamada kontrol noktası seçimi (10 nokta, fazla getiri Sharpe'ı) ve HPO'nun rotanın yolunda çalışması (Y21 kapandı) — §9.11. Kabul: `tests/test_rl_walk_forward.py` (75). Kat ölçümü 6 × 2: hiçbir kat/kol temiz değil (türbülans, K26); nedensel kolda erken katlar fiyatta %1,6–2,7 destek dışı. Veri tam yeniden indirildi; aynı kapıyla fark yalnız 5 tatil satırı (hayalet gün 72 → 67). Yeni bulgular: pano HPO penceresi test dönemini içeriyordu ve budama ölüydü (kapandı); HPO sampler tohumsuz (açık); `/data/generate` veri kaybı riski (plan dışı, ayrı iş). Literatür: Varma & Simon 2006, Cawley & Talbot 2010, Eimer vd. 2023.
- **2026-11-14 (R1 paneli):** hakem-rl ve hakem-finans Küçük revizyon, hakem-2 Büyük revizyon (RED yok) — §9.12. Kullanıcı kararları: seçim metriği korunur, tanı kayda; argmax ↔ son nokta M8'e; okunmuş test pencereleri ölçülen bayrakla (`eval_window_unexposed`; 2024–2026 doğrulayıcı değil). Yeni madde `fundamentals_pit`. Gerçek panelde seçim metriği tanısı (2025'te 10/10 negatif, ayırt edici değil) ve geniş B0 ailesi bit-eş. `test_rl_walk_forward` 90.
