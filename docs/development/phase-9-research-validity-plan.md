# Faz 9 — Araştırma Geçerliliği ve Tez Deney Altyapısı (Plan)

**Durum:** Plan — **onay bekliyor, kod değişikliği yapılmadı** (2026-10-07)
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
| 1 | **Tahmin golden'ı** — B1 (bfill) varsayılan hatta etkin; düzeltme golden'ı değiştirir | (a) B1 bayrakla, varsayılan eski; (b) varsayılanı çevir + golden'ı yenile | (b): sızıntılı varsayılanı korumanın akademik değeri yok; golden yenilemesi Milestone 0' ile birlikte |
| 2 | EVDS erişimi (B2) | API anahtarı mevcut mu / yeniden alınacak mı | B2 başlamadan teyit |
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

## Versiyon geçmişi

- **2026-10-07:** İlk plan. 66 tam metin arşivlendi; 10 yeni bulgu (Y1–Y10) kod ve makaleyle doğrulandı; A–G geçerlilik paketleri, Makale 2 deney tasarımı, M1–M8 MARL kararları, 6 açık karar. **Kod değişikliği yok.**
