# Faz 10 — Çoklu-Ajan Yenileme Planı: Tahmin, RL ve Karar Katmanı

**Durum:** **Onaylandı (2026-11-06)**, §8 hakem-2 eklemeleriyle. Kullanıcının açıkça verdiği kararlar K1, K17 ve K23 (§6). Diğer K maddelerinde plandaki öneri geçerli; ilgili paket başlamadan teyit edilir. İlk kapı (V + A + C) uygulandı → [Faz 9 §9](phase-9-research-validity-plan.md#9-uygulama-durumu). hakem-2 incelemesi (plan): **Büyük revizyon**; itirazlar ve plana etkileri §8'de.
**Dayanak:** [vision-and-scope.md v0.3](../thesis/vision-and-scope.md) (§1.4, §2.5, §3, §4, §7) · [phase-9-research-validity-plan.md](phase-9-research-validity-plan.md) (Y1–Y10, A–G, M1–M8, §7) · [literature-table.md](../thesis/literature-table.md) (2026-10-07, 148 seçilmiş kayıt) · [roadmap.md → Bilinen Metodolojik Açıklar](roadmap.md#bilinen-metodolojik-açıklar-planlanmadı)
**Kod durumu:** `chore/claude-academic-setup` @ 7817d79. Faz 9'un dayandığı b82fedb'den bu yana üretim kodu değişmedi: `git diff --stat b82fedb..HEAD -- '*.py' ':!scripts/literature'` yalnızca `.claude/hooks/guard.py` gösteriyor.
**Faz 9 ile ilişki:** Faz 10, Faz 9'un yerine geçmez, üstüne kurulur. Faz 9 ve Faz 10 2026-11-06'da birlikte onaylandı. Faz 9 paketlerine eklenen maddeler (B7–B9, F4 vb.) Faz 9 numaralandırmasını sürdürür.

---

## 0. Özet

1. **Y1–Y10'un hepsi koddaki mevcut hâlde geçerli.** Satırlar kaymamış; tek fark Y4'te bir satırlık kayma (§1.1).
2. **10 yeni bulgu (Y11–Y20).** En ağırları şunlar:
   - **Y20:** varsayılan hatta açık olan TATS (`use_tats=True`), trend sınıflandırıcısını yanlış zaman penceresine ve yanlış birime göre üretilmiş etiketlerle eğitiyor. Makale 2'nin K3/K4 kollarını doğrudan etkiliyor.
   - **Y14:** tahmin → RL bağlantısı hiçbir eğitim akışında kurulmamış. Vision §3.4 Seçenek I'deki "zaten yapılmış" ifadesi yalnızca arayüz düzeyinde doğru.
   - **Y12, Y13:** canlı kararda ve test ortamında durum vektörü eğitimdekinden farklı ölçekleniyor. Test ortamı ölçeği test penceresinin kendi (gelecek dahil) istatistiğinden kuruyor.
   - **Y11:** eğitim panelindeki 30 sembol koddaki `BIST30_SYMBOLS` ile aynı değil; altışar sembol farklı.
3. **Ajan yapısı önerisi: Seçenek A (hisse başına 30 ajan).** Ajanlar parametre paylaşır ve ajan kimliği taşır; eğitim CTDE/MAPPO ile. Sektör karar veren bir ajan olarak değil, şu üç yerde girer: GAT iletişim grafının yapısı, sektör gömmesi ve karar katmanındaki sektör maruziyet sınırı (§2).
   - Bu öneri vision §3.1'deki geçici "B" tavsiyesinden sapıyor ve G3 iddiasını erteliyor. Karar senin (§6 K6).
4. **Tahmin katmanına yeni bir iş paketi (P) eklendi: nokta-zamanlı, örneklem dışı tahmin tablosu.** Bu tablo olmadan RL eğitim dönemine tahmin beslemek sızıntı yaratır. Vision §7 Milestone 0'daki "DT vs DTF" ablasyonu da bu tabloya bağlı (§3.1.3).
5. **Karar katmanında tek gerçek kaynak ilkesi.** Gözlem kurucu, emir üretimi ve risk projeksiyonu ortamda (backtest) ve canlı kararda aynı fonksiyondan çalışır. Y12'deki türden farklar ancak K2'nin bit-eşlik testi geçince kapanmış sayılır; o test, canlı gözlemin eğitimdekiyle aynı ısınma uzunluğuyla hesaplanmasını da kapsar (§8 #9). Kağıt portföy (`apply_decisions`) değişmeden kullanılır; yürütme fiyatı farkı için bkz. §8 #18 (§3.3).
8. **hakem-2: Büyük revizyon (§8).** Doğrulanan iki RED kapı var:
   - **Veri bütünlüğü:** fiyat artefaktları, hayalet günler, eksik hücrede komşu sembolün kopyalanması; RL eğitim penceresi 2000–2018.
   - **Gözlem sağlığı:** ADX tüm panelde 0, türbülans doygun.

   İkisi de MARL'dan önce, A–G ile aynı kapıda kapanmalı. Yeni karar maddeleri K17–K23 (§6).
6. **Sıra korunuyor: önce geçerlilik (A–G), sonra MARL (M1–M8).** Önerilen üç sapmanın gerekçesi §4'te:
   - M1 çerçeve spike'ı A–G ile paralel yürür; rapora girecek sayı üretmez.
   - P paketi Milestone 0''dan önce gelir.
   - A seçilirse E2 kritik yoldan çıkar.
7. **Karar maddeleri (§6):** 24 madde; 5'i Faz 9 §7'den, 19'u yeni, 8'i hakem-2 incelemesinden. 2026-11-06'da işlendi:
   - **K1:** varsayılan çevrilir, golden Milestone 0' ile bir kez yenilenir.
   - **K17:** RL penceresi 2013-06'dan başlar.
   - **K23:** anahtar koddan kaldırıldı (PR #3), yenilenmeyecek.

---

## 1. Durum doğrulaması

### 1.1 Faz 9 bulguları (Y1–Y10)

Her satır 2026-11-06'da kod okunarak kontrol edildi. "Geçerli" = bulgu kodda aynen duruyor.

| # | Faz 9 bulgusu (kısa) | Bugünkü kanıt | Durum |
|---|---|---|---|
| Y1 | Deflated Sharpe üç yerden yanlış | Fonksiyon [evaluator.py:495](../../prediction/evaluator.py#L495). Hatalı noktalar: yıllıklandırılmış SR :517; fazla basıklık `kurtosis()` :520; SR₀'da √V[{SRₙ}] yok :524–526; σ(SR) formülü :529. Tek çağıran `compute_comprehensive_metrics` (:605); onu çağıran yok (repo genelinde grep) | **Geçerli**, satırlar aynı. 0,9004 / 0,4031 sayıları bu turda yeniden koşulmadı; kod değişmediği için Faz 9 kaydı geçerli sayıldı |
| Y2 | Ödüldeki "Differential Sharpe" türev değil | [reward_functions.py:155](../../env/reward_functions.py#L155) yöntem; EMA düzeyi :188; `tanh(x/2)` :195 | **Geçerli** |
| Y3 | Rejim metriği gerçekleşen getiriye koşullu | [evaluator.py:208](../../prediction/evaluator.py#L208); `bull_mask = actual_returns > 0` :228 | **Geçerli** |
| Y4 | Güven, tahmin edilen yönün olasılığı değil | Yön fiyat tahmininden: [ensemble.py:543](../../prediction/models/ensemble.py#L543). Güven yön başlıklarından: :547–550. Yön başlığı yoksa oy oranı: :552–554. Bu yolda ayrıca **birim hatası** var: `p > current_close` log-getiriyi fiyatla karşılaştırıyor (varsayılan hedef `log_return`), bu yüzden `up_votes` = 0 ve güven her zaman 1,0 (§8 #12) | **Geçerli**, 1 satır kaymış (542 → 543). Kapsamı Y18 ile genişliyor |
| Y5 | Sektör eşlemesi 7/30 | [bist30_symbols.py:50](../../data/bist30_symbols.py#L50) `STOCK_INFO` (7 kayıt) | **Geçerli**, ama eğitim panelinde **6/30**: `ALARK.IS` panelde yok (Y11) |
| Y6 | Eylem hisse başına [−1, 1] sinyal, ağırlık değil | [trading_env.py:159](../../env/trading_env.py#L159) | **Geçerli** |
| Y7 | G2(a) ≡ DebUnc β=1 | Matematiksel; koda bağlı değil | **Geçerli**. Vision v0.3 §2.5 G2 ve §3.4 II'de işlendi |
| Y8 | ADMAC ayrık eylem varsayıyor | Literatür bulgusu | **Geçerli**. M5'te "ADMAC'tan esinlenen" adlandırması korunuyor |
| Y9 | Yang vd. 2024 su kalitesi verisinde | Literatür bulgusu | **Geçerli** |
| Y10 | IMF özellikleri filtrelenmiş seriden | [feature_engineer.py:87–89](../../prediction/feature_engineer.py#L87) | **Geçerli** |

Faz 9'un dayandığı roadmap açıkları da aynı durumda:

| Açık | Kanıt |
|---|---|
| S2 makro `bfill` | [macro_fetcher.py:275](../../data/macro_fetcher.py#L275), [:454](../../data/macro_fetcher.py#L454) |
| D1 RL tohumsuz | `app/api/routes/trading.py`'de `seed` için 0 eşleşme |
| D2 eşleştirilmemiş t-testi | [model_analysis.py:219](../../app/services/model_analysis.py#L219) |
| D3 risksiz faiz | 0,15 [reward_functions.py:37](../../env/reward_functions.py#L37); 0,0 [evaluator.py:450](../../prediction/evaluator.py#L450); 0,02 [model_analysis.py:40](../../app/services/model_analysis.py#L40). Ayrıca dördüncü bir yer: `scripts/benchmarking/benchmark_strategies.py:127, :312` (0,02) |
| B2 makro kalite | `data/macro/macro_data_quality.json` (2026-07-28): `policy_rate: fallback_constant_50.0`, `cpi_inflation` ve `ppi_inflation: fallback_zero`. `macro_data.csv`'nin 3.131 satırında bu üç sütun tek değerli (§Ek A, komut 2) |

### 1.2 Yeni bulgular (Y11–Y20)

Ciddiyet ölçeği:
- **Y** = sonuç geçerliliğini doğrudan etkiler.
- **O** = belirli kollarda ya da sınırlı dönemde etkiler.
- **D** = tasarım varsayımı; beyan ve duyarlılık analizi gerektirir.

| # | Bulgu | Kanıt | Cid. | Etki | Paket |
|---|---|---|---|---|---|
| **Y11** | **Eğitim panelinin evreni koddaki listeyle aynı değil.** Ayrıca listedeki bir sembol "BIST30'da değil" diye işaretli | `data/bist/raw_stock_data.csv`: 30 sembol (komut 1). `BIST30_SYMBOLS` ([bist30_symbols.py:7](../../data/bist30_symbols.py#L7)) ile farklar:<br>• yalnızca panelde: HALKB, MGROS, ODAS, SASA, SKBNK, TSKB<br>• yalnızca listede: ALARK, HEKTS, KOZA, KOZAA, KOZAL, TTKOM<br>• `KOZA.IS # ... (Not in BIST30 - placeholder)` ([:24](../../data/bist30_symbols.py#L24))<br>Eğitim paneli bütünüyle kullanıyor (CLAUDE.md "Sembol evreni") | Y | "BIST-30" tanımı belirsiz. E2 sektör sınıflamasının hangi 30 sembol için yapılacağı karar ister (§6 K7). Sembollerin yaşları da farklı: 19 sembol 2000-05-10'dan, PGSUS/ODAS 2013'ten | E |
| **Y12** | **Canlı karar durum vektörü eğitim normalizasyonunu kullanmıyor** | Fiyat ölçeği:<br>• Ortam fiyatı sembol bazlı z-skorla ölçekliyor ([trading_env.py:198–210](../../env/trading_env.py#L198) hesap, :627–629 kullanım)<br>• `/daily-decision`, `build_live_state`'i `price_stats` vermeden çağırıyor ([trading.py:1701–1707](../../app/api/routes/trading.py#L1701)); sonuç `(p−50)/50` yedeği ([daily_trading.py:589–593](../../app/services/daily_trading.py#L589))<br>• İstatistik model meta dosyasına yazılmıyor ([trading.py:1408–1420](../../app/api/routes/trading.py#L1408))<br>Bakiye ölçeği:<br>• `initial_balance` canlıda varsayılan 1.000.000 ([daily_trading.py:498](../../app/services/daily_trading.py#L498)); kağıt portföy 100.000 ile açılıyor ([portfolio.py:49](../../app/services/portfolio.py#L49))<br>• Boş portföyde `balance_norm` = ln(1,1) ≈ 0,095; eğitimin ilk adımında ln(2) ≈ 0,693 ([trading_env.py:591–593](../../env/trading_env.py#L591))<br>Gösterge değerleri (§8 #9):<br>• Canlı veri penceresi `lookback_days=30` ([trading.py:1697](../../app/api/routes/trading.py#L1697)); 252 seanslık Mahalanobis türbülansı bu pencerede hesaplanamıyor, canlıda türbülans 0 çıkıyor<br>• `build_live_state` ±10 kırpması yapmıyor; eğitim ortamı yapıyor | Y | Mevcut tek-ajan modellerin canlı kararları, eğitimde görülmemiş ölçekte bir durumdan üretiliyor. Kağıt portföy sonuçları bu modellerin değerlendirmesi olarak kullanılamaz | K (§3.3) |
| **Y13** | **Test ortamı ölçeği test penceresinin kendi istatistiğinden kuruyor** | `make_test_env(df=test_df)` ([trading.py:1423–1433](../../app/api/routes/trading.py#L1423)); `_compute_price_stats` aldığı df'in tamamının ortalama/std'sini kullanıyor ([trading_env.py:198–210](../../env/trading_env.py#L198)) | Y | Test gözlemlerinin ölçeği test penceresindeki **gelecek** fiyatları içeriyor. RL test metrikleri (`results/*.json`) bu bilgiyi taşıyor; etkinin büyüklüğü ölçülmedi. **B9 ile kapandı (yalnız bu daldan sonraki koşumlar); metriğe etkisi tanımlanamadı, B9 kolu dağılım dışı (Faz 9 §9.5)** | B (B9) |
| **Y14** | **Tahmin → RL bağlantısı hiçbir akışta kurulmamış** | `TradingEnv(prediction_features=...)` parametresi var ([trading_env.py:51](../../env/trading_env.py#L51), :100), ama repo genelinde `prediction_features=` ile tek çağrı yok (grep, venv hariç). `/daily-decision` `prediction_data` vermiyor ([trading.py:1701–1707](../../app/api/routes/trading.py#L1701)). Walk-forward kat tahminleri saklanmıyor; `cross_validate` yalnızca metrik özeti döndürüyor ([trainer.py:82](../../prediction/trainer.py#L82), :309) | Y | Vision §3.4 Seçenek I'deki "zaten yapılmış" ifadesi yalnızca ortam arayüzü için doğru. AQ2'nin "yalnız gözlem" kolu ve Milestone 0'daki "DT vs DTF" ablasyonu için RL eğitim dönemine **örneklem dışı** tahmin üreten bir yol yok. tez-final.pdf'te tahmin destekli bir RL sonucu varsa hangi akışla üretildiği doğrulanmadı | P (yeni) |
| **Y15** | **Teknik göstergelerin ısınma boşluğu geriye doğru dolduruluyor** | Göstergelerde `ffill → bfill → 0` ([technical_indicators.py:169–172](../../data/technical_indicators.py#L169)). `clean_data` OHLCV'de de `bfill` yapıyor ([data_fetcher.py:342](../../data/data_fetcher.py#L342)). İkisi de varsayılan RL hattında | O | Her sembolün ilk birkaç on günü, göstergenin ilk geçerli (gelecekteki) değerini taşıyor. Etki eğitimin başıyla sınırlı, ama B0 bekçisinin yakalaması gereken türden | B (B7) |
| **Y16** | **Walk-forward CV'de özellik seçimi katlara bölmeden önce tüm seride yapılıyor** | `FeatureSelector.fit_select` tüm `X, y` üzerinde ([trainer.py:125–142](../../prediction/trainer.py#L125)); katlar :144'ten sonra. Akış API'den çağrılıyor ([prediction.py:287](../../app/api/routes/prediction.py#L287) → [prediction_service.py:918](../../app/services/prediction_service.py#L918)) | Y | CV metrikleri, test katlarının hedeflerini görmüş özellik kümesiyle hesaplanıyor. Kapoor & Narayanan (2023) taksonomisi B0 gerekçesi olarak tabloda; bu alt türün taksonomideki etiketi tam metinden teyit edilmedi | B (B8) |
| **Y17** | **RL protokolü tek bir kronolojik bölme; eğitim penceresi yeniden kurulamıyor** | `split_data` 70/15/15 oranlı ([data_fetcher.py:481–509](../../data/data_fetcher.py#L481)). Val bölümü yalnızca hizalanıyor ([trading.py:1163](../../app/api/routes/trading.py#L1163)), seçimde kullanılmıyor. Ölçülen panelde eğitim penceresi 2000-05-10..2018-07-20; vision §4.1 "2018-2026" diyor (§8 #1). Oranlar o günkü panel uzunluğuna göre hesaplandığı için panel büyüdükçe pencere kayıyor. Meta dosyasında eğitim tarih aralığı, tohum ve istatistik yok ([trading.py:1408–1420](../../app/api/routes/trading.py#L1408)) | Y | Vision §4.1'in walk-forward şartı RL katmanında karşılanmıyor. Eski bir modelin hangi pencereyle eğitildiği bugün kesin olarak kurulamaz; bu, Y12 için geriye dönük düzeltmeyi de yaklaşık kılar | C, D |
| **Y18** | **Güven yalnızca iki derin modelin yön başlığından geliyor; ECE varsayılanı 5 kutu** | `_predict_direction_raw` taban sınıfta `None` ([base.py:120–125](../../prediction/models/base.py#L120)); yalnızca `lstm_model.py` ve `tft_model.py` override ediyor. Üç GBM güvene katılmıyor. `compute_calibration(n_bins=5)` ([evaluator.py:158–160](../../prediction/evaluator.py#L158)), Guo vd. (2017) 15 kutu kullanıyor (Faz 9 F3) | O | Y4'ü genişletiyor: güveni tek bir olasılık kaynağına bağlamak gerekiyor (F1) | F |
| **Y19** | **Backtest kararı ve yürütmesi aynı kapanış fiyatında** | `step()` önce işlemleri yapıyor ([trading_env.py:419–426](../../env/trading_env.py#L419)), sonra `current_step += 1` (:435). Fiyat o günün kapanışı ([trading_env.py:866–904](../../env/trading_env.py#L866)), yani gözlenen kapanışla aynı. Canlıda karar seans kapandıktan sonra veriliyor, işlem en erken sonraki seansta | D | İyimser bir yürütme varsayımı olabilir; büyüklüğü ölçülmedi. D4 maliyet ızgarasına bir "t+1 açılış" duyarlılığı eklenmeli (§6 K12) | D (D4) |
| **Y20** | **TATS trend sınıflandırıcısı yanlış hizalanmış etiketlerle eğitiliyor (varsayılan hatta açık)** | `use_tats=True` ([ensemble.py:75](../../prediction/models/ensemble.py#L75)), `target_type='log_return'` (:78). Uyum çağrısı: `current_prices_meta = y_base[-tats_min_len:]` ([ensemble.py:330–334](../../prediction/models/ensemble.py#L330)) | Y | Makale 2'nin K3/K4 kolları etkileniyor ve bu kolların ölçüm yolu kodda yok; ayrıntı aşağıda | B (B10) |

**Y20 ayrıntısı.** TATS etiketleri iki ayrı hatayla üretiliyor:

1. **Yanlış zaman penceresi.** "Mevcut fiyatlar", meta bölümünün kendi kapanışları değil; **taban bölümünün son k hedefi**.
2. **Yanlış birim.** `TATSCorrector.fit` girdisinin fiyat olmasını bekliyor ([tats.py:56–80](../../prediction/tats.py#L56) docstring). Varsayılan hedef `log_return` olduğu için ([feature_engineer.py:579](../../prediction/feature_engineer.py#L579)) her iki dizi de log-getiri; etiket "log-getirilerin oranı"ndan çıkıyor.

**Yönü değiştirmiyor** (ilk sürümdeki "yönü kaydırabiliyor" ifadesi yanlıştı; hakem-2 #6, kodda doğrulandı). Düzeltme gücü `trend_confidence × correction_factor`; `correction_factor` varsayılanı 0,5 ([tats.py:45](../../prediction/tats.py#L45)). Çatışmada düzeltilmiş tahmin cari fiyat ile ham tahmin arasında kalıyor ([tats.py:150–156](../../prediction/tats.py#L150)): işaret korunuyor, yalnızca büyüklük en fazla yarıya iniyor.

Hakem-2'nin eklediği iki sorun:
- **Değerlendirilmiyor:** test metrikleri TATS uyumundan önce ve TATS'sız hesaplanıyor ([ensemble.py:318](../../prediction/models/ensemble.py#L318) ↔ :328–335).
- **Diske yazılmıyor:** diskten yüklenen modelde TATS hiç çalışmıyor (hakem-2 #6, `load` :649–689; bu turda ayrıca okunmadı).

Uygulama Kazemdehbashi (2026) yöntemine karşılık gelmiyor; etkinin büyüklüğü ölçülmedi.

**Diğer gözlemler (bulgu değil, planı etkileyen durum):**
- **EVDS anahtarı.** `.env`'de `EVDS_API_KEY` dolu (10 karakter; değer okunmadı, yazılmadı), ama son çekimde üç seri yedeğe düştü. B2'nin sorusu "anahtar var mı" değil, "neden düşüyor" (§6 K2).
- **Panel güncelliği.** Panelin son tarihi 2026-07-27 (komut 1). Milestone 0' öncesi tazeleme ve anlık görüntü (C3) gerekiyor. **Güncelleme (2026-11-13):** panel 2026-11-11'de tam yeniden indirildi (2005-01-03..2026-11-09; yfinance 2005 öncesini artık vermiyor). Aynı kapıyla pencerede tek fark 5 tatil satırı; SKBNK'de geçmiş temettü çarpanları revize edilmiş (Faz 9 §9.11). Tam indirme geçmişi değiştirebildiği için sonuçlar panel SHA-256'sına bağlanır, yeniden indirmeyle yeniden üretilmez.

---

## 2. Ajan yapısı önerisi

### 2.1 Öneri

**Seçenek A: hisse başına ajan (N = panelin 30 sembolü), tam parametre paylaşımı + ajan kimliği, CTDE.**

- **Eğitim:** IPPO ve MAPPO, merkezi eleştirmenle; yürütme dağıtık (Lowe vd. 2017; de Witt vd. 2020; Yu vd. 2022).
- **Sektörün yeri:** karar veren bir ajan olarak değil, üç yapısal rolde:
  1. GAT kolunda graf kenarı (E2 sektör kenarları ∪ korelasyon kenarları; Mantegna 1999)
  2. Ajan gözlemine eklenen sektör gömmesi
  3. Karar katmanında sektör maruziyet sınırı (§3.3 K6)

Hedefindeki "hisse ve sektör düzeyi", hisse ajanlarının karar verdiği, sektörün iletişim yapısını ve risk sınırını belirlediği bir yapı olarak karşılanır.

### 2.2 Gerekçe

| # | Gerekçe | Dayanak |
|---|---|---|
| 1 | **G2'nin ön koşuluyla birebir örtüşür.** Tahmin topluluğu sembol başına eğitiliyor (model kimliği sembolden: [ensemble.py:595](../../prediction/models/ensemble.py#L595)). F paketinde kalibre edilecek c_j doğrudan gönderen ajanın güveni olur. B'de bir sektör ajanının güveni ya sembol güvenlerinin bir birleşimidir (o zaman birleşik güvenin hangi olayın olasılığı olduğu tanımlanmalı ve ayrıca kalibre edilmeli) ya da sektör sepeti için yeni bir tahmin hedefi gerektirir. G2'nin dar ifadesi "harici, denetimli, **kalibre** topluluk güveni" diyor | vision §2.5 G2; Faz 9 F, M5; Guo vd. 2017 |
| 2 | **Ajan tanımı sektör eşlemesine bağlı değil.** Eşleme eğitim panelinde 6/30 (Y5, Y11). A'da E2 gecikirse `none` ve `attn` kolları etkilenmez; `gat` kolu korelasyon grafıyla başlayabilir | Y5, Y11; Mantegna 1999 |
| 3 | **G1 karşılaştırması daha ayırt edici olabilir.** `attn` (tam bağlı dikkat) ile `gat` (yapısal graf) arasındaki fark graf öncülünden gelir; 6–8 sektör düğümünde iki kol birbirine yaklaşır.<br>**Düzeltme (hakem-2 #14):** seyreklik eşik kuralıyla kendiliğinden gelmiyor. 2013-06..2018-07'de ortalama çiftli korelasyon 0,398; eşik 0,3'te kenar yoğunluğu %77,5. Kenar kuralı (MST ya da top-k) önceden sabitlenmeli, yoğunluk raporlanmalı | Das vd. 2019; Veličković vd. 2018; Niu vd. 2021; Mantegna 1999 |
| 4 | **Tek GPU'da maliyet sınırlı kalır.** Parametre paylaşımında 30 ajan, tek bir ağın 30 satırlık toplu ileri geçişidir. MAPPO eleştirmeni ortak eylemi değil yerel + küresel özellikleri alır (Faz 9 M3, "Öneri 2"). Vision §3.1 A'daki "joint action=30 → critic şişer" kaygısı MADDPG tipi merkezi Q-eleştirmenler için geçerli, MAPPO için değil. Gerçek süre M1 spike'ında ölçülür | Yu vd. 2022; Lowe vd. 2017; Christianos vd. 2021 |
| 5 | **Eylem homojen.** Her ajan tek bir skaler logit üretir. B'de sektör ajanları grup büyüklüğüne göre değişen boyutlu grup içi ağırlık vektörü üretir; maskeleme/doldurma gerekir ve paylaşım karmaşıklaşır | Faz 9 M2; Christianos vd. 2021 |
| 6 | **Kapsam küçülür.** B seçilirse MPLS tipi hiyerarşik sektör dağıtıcı **zorunlu baseline** olur (vision §3.1 B; Faz 9 M3) ve ayrı bir hiyerarşik ajan olarak yazılması gerekir. A'da G3 iddia edilmediği için bu baseline zorunlu değil. C (hiyerarşik) vision §4.2'de kapsam dışı; §3.1'de "bitirilememe riski belirgin" | vision §3.1, §4.2; Khemlichi vd. 2025 |
| 7 | **İletişimsiz kolun literatürde bir karşılığı olur.** A + parametre paylaşımı + softmax ağırlık, Jiang vd. (2017) EIIE topolojisine (varlık başına özdeş değerlendirici + softmax ağırlık vektörü) yakın. `none` kolu "uydurma" değil, literatürde tanımlı bir topolojinin çok-ajanlı eğitilmiş hâli olur | Jiang vd. 2017 (Y6'nın kaynağı) |
| 8 | ~~Varlık başına ajanın doğrudan öncülleri var~~ — **geri çekildi (hakem-2 #4).** Neden:<br>• MAPS'te ajan bir varlık değil, kendi portföyünü kuran bağımsız bir "yatırımcı".<br>• MSPM'de varlık düzeyindeki modül yalnızca sinyal üretiyor; dağıtımı portföy düzeyindeki bir modül yapıyor.<br>• Hao vd. 2025 için "varlık başına" niteliği tablodaki rolde yok, yalnızca vision §2.1'de geçiyor; tam metin kaydı da yok.<br>Bu kaynaklar A'nın gerekçesi olarak kullanılmaz. Literatür tablosu (meta.py ROLE) ve vision §2.1 düzeltmesi §6 K22'de | MAPS ve MSPM tam metinleri hakem-2 tarafından web'de okundu (§8 #4) |

### 2.3 Bedeli (açıkça)

- **G3 ertelenir.** "Merkezi dağıtıcı olmadan, öğrenilmiş iletişimle koordine olan eşdüzeyli sektör ajanları" (vision §2.5 G3) bu tasarımda iddia edilmez. Vision §2.1'in son paragrafındaki savunulabilir çekirdek G1, G2, G4 ve G5'i içeriyor; G3'ü içermiyor. Çekirdek bu yüzden etkilenmez. Etkilenen yerler:
  - vision §3.1 tavsiyesi
  - §4.1 RL katmanı ("Sektör-bazlı ~8 ajan")
  - §4.3 başarı kriteri ("8-sektör MARL çerçevesi")
  - §7 Milestone 2 ("8-sektör ajan tasarımı")
  - Faz 9 M2 (ajan = sektör grubu) ve M3 (MPLS baseline zorunlu)
- **Özgünlük granülaritede değil.** Vision §3.1 A'da "Düşük — literatürde çok örnek var" deniyor. Katkı iddiaları G1, G2, G4 ve G5'te durur. Her biri vision §2.5'teki dar ifadesiyle ve "bildiğimiz kadarıyla (tarama: 2026-10, kapsam: literature-table.md §1.1)" kalıbıyla yazılır. **G4, YÖK Ulusal Tez Merkezi taraması tamamlanmadan kullanılmaz** (vision §2.5). Y7 daralması (G2(a) ≡ DebUnc β=1) dikkate alınınca G2'nin savunulabilir parçaları kalibre harici güven, öğrenilen β ve alıcı geçidi. Kalibresiz-güven kolu bir katkı değil, bu parçaları sınayan bir ablasyon (Faz 9 M5).
- **Beklenen hakem itirazı:** "Parametre paylaşımı + sabit softmax + ortak ödül = çarpanlara ayrılmış tek-ajan politika." Bu itiraz **biçimsel olarak doğru** ve metinde açıkça kabul edilir (hakem-2 #3). Softmax her wᵢ'yi tüm zⱼ'lere bağlar; EIIE de tek-ajan bir politikadır. Doğru tanım: "yerel gözlemli politika + öğrenmeyen merkezi normalizasyon". Tezin "MARL mı, tek-ajan mı" sorusunu yanıtlayan **birincil karşıtlık** M8 ön kaydına yazılır:
  - Her iletişim kolu için **MAPPO ile SARL-eşmimari** karşılaştırılır; aynı α, aynı gözlem, aynı tohumlar.
  - α > 0 ön elemede seçilir. α=0'da iki yöntemin farkı esas olarak oran kırpmasının ajan başına mı ortak mı yapıldığına iner.
  - SARL-ağırlıkta da z₀ sabit tutulur, yani aynı eylem temsili (§3.2.4).
- **30 ajanda durağan olmama** (vision §3.1 A riski; Hernández-Leal vd. 2019). Azaltma: parametre paylaşımı, MAPPO'nun merkezi değer fonksiyonu, IPPO ve MAPPO'nun ikisinin birden raporlanması (de Witt vd. 2020; Yu vd. 2022).
- **B ne zaman yeniden açılır:** Milestone 3 sonunda zaman kalırsa, sabit grup-içi eşit ağırlıklı B, **keşifsel** bir granülarite ablasyonu olarak eklenebilir. M8 ön kaydında hipotez olarak yer almaz.
- **Açık literatür riski:** Hung & Huang (2026), portföyde öz-dikkatli MARL; tam metni okunmadı (vision §2.1). Varlık başına öz-dikkat kullanıyorsa G1'in "ilk karşılaştırma" ifadesi daha da daralabilir. Milestone 2'den önce tam metin okunmalı.

---

## 3. Üç katmanın yenileme planı

### 3.1 Tahmin katmanı

#### 3.1.1 Sızıntı kapatma

Faz 9 B paketi olduğu gibi kalır:
- **B0** kesme değişmezliği testi
- **B1** makro `bfill`
- **B2** EVDS
- **B3** TÜFE/ÜFE açıklanma tarihi
- **B4** temel oranlar kapalı + sınırlılık beyanı
- **B5** nedensel ICEEMDAN
- **B6** önceden tanımlı rejim etiketi

Yeni bulgulardan dört madde eklenir:

| İş | Açık | Kaynak | Kabul ölçütü | Varsayılan / golden etkisi |
|---|---|---|---|---|
| **B7** Gösterge ısınma satırları `bfill` yerine maskelenir ya da atılır; `clean_data`'daki `bfill` yalnızca sembolün ilk geçerli satırından sonrasına uygulanır | Y15 | Kapoor & Narayanan 2023; Hewamalage vd. 2023 | B0 göstergeler ve OHLCV için geçer | **Davranış değişikliği** (RL gözlem değerleri, ilk günler) → bayrak. Durum vektörünün boyutu değişmez |
| **B8** Özellik seçimi her katın yalnızca eğitim bölümünde; önbellek anahtarına kat sınırı girer | Y16 | Bergmeir & Benítez 2012; Kapoor & Narayanan 2023 | Test katının `y`'si karıştırıldığında seçilen küme **değişmez** | CV metrikleri değişir; CV hiçbir modeli diske yazmaz (`train_final_models` ayrı) |
| **B9** RL val/test ortamına eğitim penceresinin `price_stats`'ı verilir (`TradingEnv(..., price_stats=None)` opt-in; verilmezse eski davranış) | Y13 | B0 mantığı | Test penceresinde t sonrası fiyatlar değiştirildiğinde t gözlemi bit-eş kalır | `env/trading_env.py` değişir → env bekçi testleri (hook). Durum boyutu değişmez. Eski sonuçlar eski bayrakla yeniden üretilebilir |
| **B10** TATS uyumunda "mevcut fiyat" meta bölümünün kendi kapanışından alınır; `log_return` hedefinde etiket getiriden üretilir | Y20 | Kazemdehbashi 2026 (yöntem) | Sentetik testte (bilinen trend) etiket isabeti ≈ 1. Uyum girdileri ile meta bölümü tarihleri birebir hizalı | **Tahmin golden'ı değişir** (`use_tats=True` varsayılan) → §6 K1 kapsamına girer |

**Makale 2 için not.** Faz 9'daki K0–K4 tasarımı B10 olmadan kurulursa TATS kolları (K3, K4) yayımlanmış yöntemi değil hatalı bir uyarlamayı ölçer. Makale 2 deneyleri B5 ve **B10**'dan sonra başlar.

#### 3.1.2 Kalibre güven (Faz 9 F + Y4, Y18)

| İş | Ne | Kaynak | Kabul ölçütü |
|---|---|---|---|
| **F1** | Ayrıntı tablonun altında | Wolpert 1992; Niculescu-Mizil & Caruana 2005 | Çelişki oranı manifestte. Güven tanım gereği ≥ 0,5 |
| **F2** | Ayrıntı tablonun altında | Guo vd. 2017; Niculescu-Mizil & Caruana 2005 | Test ECE'si kalibrasyondan önce ve sonra raporlanır; düşmezse bu da raporlanır |
| **F3** | ECE (**15** eşit genişlikli kutu), Brier skoru, güvenilirlik diyagramı; her eğitim manifestinde. `compute_calibration`'ın 5 kutu varsayılanı kullanılmaz (Y18) | Naeini vd. 2015; Brier 1950; Guo vd. 2017 | Manifest alanı |
| **F4** (yeni) | Kalibrasyon öncesi ham olasılık p_raw da saklanır. M5'in "kalibresiz güven" kolu bu sütunu kullanır | Faz 9 M5 | P tablosunda `p_up_raw` ve `p_up_cal` birlikte |

**F1 — tek olasılık kaynağı.**
- p(UP), stacking meta düzeyinde bir lojistik sınıflandırıcıdan gelir. Girdileri taban modellerin getiri tahminleri ve yön başlıklarıdır; böylece **üç GBM de güvene katılır** (Y18).
- Yön = 1[p ≥ 0,5]; güven c = max(p, 1−p), yani tahmin edilen yönün olasılığı.
- Fiyat tahmininin yönü ile p'nin yönü çelişirse bu ayrı bir bayrak olarak kaydedilir (Faz 9 F1).

**F2 — kalibrasyon verisi.**
- Sıcaklık ölçekleme yapılır; karşılaştırma için izotonik regresyon da denenir.
- Kalibrasyon verisi, meta-öğreniciyi uydurmak için kullanılan veriden ayrılır. OOF bölümü ([ensemble.py:221–227](../../prediction/models/ensemble.py#L221), %60–80 aralığı) kronolojik olarak ikiye bölünür: önce meta uyumu, sonra kalibrasyon. Test bölümünde yalnızca değerlendirme yapılır.
- **Tasarım kararı:** sıcaklık T 30 sembol için ortak (havuzlanmış) tutulur, ECE sembol başına raporlanır. Gerekçe örnek büyüklüğü: OOF'un yarısı sembol başına yalnızca ~%10 veri.

#### 3.1.3 Ajanlara giden sinyal (vision §3.4)

**P — nokta-zamanlı örneklem dışı tahmin tablosu (yeni paket).** Y14'ü kapatır; Seçenek I ile II'nin ortak girdisidir.

| İş | Ne | Kaynak | Kabul ölçütü |
|---|---|---|---|
| **P1** | Ayrıntı tablonun altında | López de Prado 2018 (purge/embargo); Bergmeir & Benítez 2012; Wolpert 1992 (OOF) | B0: t sonrası veri değiştirilince t satırı bit-eş. Eksik hücre açık maskeyle işaretlenir (bkz. P2) |
| **P2** | Kapsam raporu: sembol × gün doluluk oranı, maske oranı, yeniden eğitim tarihleri | — | Manifestte |
| **P3** | Maliyet ölçümü önce yapılır: sembol başına bir eğitim süresi × yeniden eğitim sayısı × 30. Referans: Faz 6 kapanış koşumu 5 sembol için 157,6 sn (perf ayarları açık; CLAUDE.md Faz 6) | — | Ölçüm sonucuyla model kapsamı kararı (§6 K11) |

**P1 — tablonun içeriği.**
- Her sembol × işlem günü için, yalnızca o günden önce (purge + embargo ile) biten veriyle eğitilmiş bir modelin tahmini.
- Pencere genişleyen; yeniden eğitim periyodik (öneri her 63 işlem günü — *tasarım varsayımı*, P3 maliyet ölçümüyle kesinleşir).
- Sütunlar: `date, symbol, p_up_raw, p_up_cal, pred_logret, model_disagreement, model_version, train_end`. `model_disagreement` taban model tahminlerinin standart sapması.
- Konum: kullanıcı çalışma alanı, `results_dir()/predictions_pit/<run_id>.parquet`, manifest ile (`prediction/manifest.py`).
- Sembolün tahmini yoksa bu açık bir maskeyle işaretlenir; sessiz bir varsayılana düşülmez. Bugünkü kodda böyle sessiz varsayılanlar var: `.get('confidence', 0.5)` ([trading_env.py:694](../../env/trading_env.py#L694); [daily_trading.py:622](../../app/services/daily_trading.py#L622)).

**Sinyal yolları:**

| Yol | Vision karşılığı | Tasarım |
|---|---|---|
| **S-obs** | Seçenek I | Ajan i'nin yerel gözlemine kendi `[p_up_cal, pred_logret, model_disagreement]` değerleri eklenir |
| **S-msg** | Seçenek II (Faz 9 M5) | Mesaj dikkati softmax(ℓᵢⱼ + β·log cⱼ). Burada cⱼ ∈ [c_min, c_max] arasına kırpılmış kalibre güvendir (sürüleşme bekçisi; *tasarım varsayımı*). β öğrenilir; β=0 güvensiz dikkat, β=1 DebUnc ölçeklemesi (Yoffe vd. 2025). Alıcı tarafında öz-güven geçidi gᵢ = σ(a·(1−cᵢ)+b) mesajların toplam ağırlığını ölçekler |
| **S-reg** | Seçenek III — **yeniden tanım** | Vision "meta-learner çıktısı → rejim" diyor, ama meta-öğrenici bir getiri tahmini üretiyor, rejim değil ([ensemble.py:505–518](../../prediction/models/ensemble.py#L505)). Öneri: tüm ajanlara ve eleştirmene yayınlanan sinyal B6'nın **nedensel** kural tabanlı rejim özellikleri (XU100 200 günlük trend + 60 günlük gerçekleşmiş oynaklık tercili, yalnızca ≤ t veri). İkinci tanım olarak HMM'in yalnızca filtrelenmiş olasılıkları (Hamilton 1989; Nystrup vd. 2018). Aynı etiket AQ3'te koşullama değişkeni olduğu için bu döngüsellik metinde beyan edilir; AQ3 sonucu ikinci tanımla da tekrarlanır (Faz 9 M6) → §6 K10 |

**AQ2 ablasyonu.** Vision §2.3'teki dört koşul kullanılır: tahminsiz / yalnız S-obs / yalnız S-msg / ikisi birden. Kombinatoryal büyümeyi önlemek için S-reg ana tasarımda **sabit** tutulur (açık); katkısı ayrı ve tek bir ablasyonla ölçülür.

### 3.2 RL katmanı

Tüm MARL işleri **yeni** bir ortamda yapılır (`env/marl_trading_env.py`, PettingZoo Parallel; Terry vd. 2021). `env/trading_env.py` ve mevcut state space değişmez; bu dosyadaki tek değişiklik B9'un opt-in parametresi. Eski modeller yüklenmeye ve çalışmaya devam eder.

#### 3.2.1 Ödül (Faz 9 G + Y2)

- **G1–G3** Faz 9'daki gibi: `sharpe_term='ema_level'|'differential'`, gerçek Moody-Saffell Dₜ; risksiz faiz A8'den; metin-kod denetimi. Yeni ortamda risksiz faiz serisi **zorunlu girdi**. Eski ortamdaki 0,15 sabiti ([reward_functions.py:37](../../env/reward_functions.py#L37)) davranış dondurma gereği varsayılan olarak kalır.
- **MARL ödülü** (vision §3.3 γ; öncül Hao vd. 2025):
  - rᵢ,ₜ = α·kᵢ,ₜ + (1−α)·Rₜ
  - kᵢ,ₜ = wᵢ,ₜ₋·rᵢ,ₜ − maliyetᵢ,ₜ (ajanın basit getiri katkısı). Katkıların toplamı Rₜ'nin riskli kısmına eşit; nakit getirisi yalnızca portföy terimine girer.
  - Rₜ = portföy getirisi − maliyet, ya da G1 sonucuna göre diferansiyel Sharpe.
  - α ön elemede {0; 0,25; 0,5}. α=1 kullanılmaz (vision §3.3 β, "zayıf").
  - δ (risk-ayarlı bireysel + dönüşüm cezası) bir ablasyon varyantı.
- **Kredi ataması:** MAPPO'nun merkezi değer fonksiyonu. Karşı-olgusal yaklaşım (Foerster vd. 2018) eğitimde değil, M7'deki mesaj maskeleme analizinde kullanılır.

#### 3.2.2 Eylem uzayı (Y6 → M2, Seçenek A için)

- **Politika:** ajan i bir skaler logit zᵢ üretir (Gauss politikası, PPO). Nakdin logiti sabit z₀ = 0 (*tasarım varsayımı*). Hedef ağırlık w = softmax([z₀, z₁…z₃₀]).
- **Uzun-yalnız + nakit,** çünkü açığa satış kapsam dışı (vision §4.2).
- **Normalizasyon öğrenmeyen sabit bir işlemdir; merkezi bir dağıtıcı ajan değildir.** Faz 9 M2'deki uyarı burada da geçerli ve metinde açıkça yazılır.
- **Yürütme:** ortam, fiyat değişimiyle kaymış ağırlık wₜ₋'yi hedef wₜ'ye taşır. Maliyet c·Σ|wₜ − wₜ₋|, c D4 ızgarasından (0/10/20/25/50 bps). Ana senaryo **t kapanışında karar, t+1 açılışında yürütme**; t kapanışı duyarlılık analizi olarak koşulur (Y19 → §6 K12).
- **Risk projeksiyonu:** tek pozisyon üst sınırı, sektör maruziyet üst sınırı ve dönüşüm bandı (|Δwᵢ| < b ise işlem yok). Projeksiyon **ortamda ve canlı kararda aynı fonksiyonla** uygulanır (§3.3 K5–K6).
- **Kesirli/tam sayı:** eğitim kesirli ağırlıkla (hızlı), nihai backtest'ler tam sayı hisse modunda yapılır; bu mod canlı kararın emir üretimiyle aynı fonksiyondur. Yuvarlama etkisi raporlanır.
- **Dirichlet politika** (Faz 9 M2) ön elemeye girmez; Gauss logit yakınsamazsa alternatif olarak denenir.

#### 3.2.3 Gözlem ve eleştirmen

- **Yerel gözlem (aktör):**
  - kendi sembolünün 10 fiyat/gösterge özelliği (mevcut dönüşümlerle: [trading_env.py:631–641](../../env/trading_env.py#L631))
  - kendi ağırlığı wᵢ ve nakit oranı
  - S-obs açıksa tahmin sinyali
  - ajan kimliği gömmesi, E2 hazırsa sektör gömmesi
  - S-reg rejim özellikleri
- **Normalizasyon:** istatistikler yalnızca eğitim penceresinden hesaplanır ve model paketine yazılır (Y12 ve Y13'ün dersi).
- **Küresel gözlem (eleştirmen):** tüm ağırlık vektörü, XU100, makro (B1–B3 sonrası), rejim özellikleri. MAPPO'da eleştirmene yerel ve küresel özellik birlikte verilir (Faz 9 M3, "Öneri 2").

#### 3.2.4 Algoritmalar ve kollar

- **MARL:** IPPO (de Witt vd. 2020) ve MAPPO (Yu vd. 2022), parametre paylaşımı + ajan kimliği. Hiperparametreler Faz 9 M3'teki MAPPO önerilerinden; farklı seçilen her değerin gerekçesi yazılır.
- **Kontrol kolları** (aynı yeni ortam, aynı maliyet ve gözlem verisi):
  - **SARL-ağırlık:** tek merkezi PPO aktörü, 30 boyutlu logit (nakit z₀ = 0 sabit, MARL'la aynı). Tek-ajan ile çok-ajanlıyı **aynı eylem temsiliyle** karşılaştırır.
  - **SARL-eşmimari:** paylaşımlı mimari, tek-ajan eğitimi (§2.3'teki itirazın kontrolü).
- **Replikasyon baseline'ı:** Ansari replikasyonu eski ortamda (Milestone 0'). Eylem temsili farklı olduğu için MARL ile **doğrudan** kıyaslanmaz; ayrı bir tabloda raporlanır.
- **Klasik baseline'lar:** 1/N, min-varyans (Ledoit-Wolf daraltmalı), HRP, al-tut, XU030 (Faz 9 D1–D2; DeMiguel vd. 2009).
- **MPLS tipi hiyerarşik dağıtıcı:** yalnızca §6 K6'da B seçilirse zorunlu.

#### 3.2.5 İletişim kolları (M4)

- **Kollar:** `none` / `attn` (TarMAC tipi, Das vd. 2019) / `gat` (Veličković vd. 2018; MAGIC, Niu vd. 2021).
- **Eşit bütçe:** parametre sayısı ±%10, aynı HPO deneme sayısı, aynı tohumlar, aynı ortam adımı (Faz 9 M4; Fang vd. 2023 finanstaki tek öncül).
- **GAT grafı:** E2 sektör kenarları ∪ yalnızca eğitim penceresinden kurulan korelasyon kenarları (Mantegna 1999). Graf da B0 mantığına tabi. Varsayılan olarak her eğitim penceresinde sabit; dinamik graf kapsam dışı.

#### 3.2.6 Çerçeve seçimi (M1 spike'ı)

**Ölçütler.** Faz 9 M1'deki dört ölçüte üç ek:

| # | Ölçüt | Kaynak |
|---|---|---|
| 1 | Değiştirilebilir iletişim katmanlı özel modül yazılabiliyor mu | Faz 9 M1 |
| 2 | Tohumla tekrar üretilebilirlik (C1 testi) | Faz 9 M1 |
| 3 | Windows'ta çalışma | Faz 9 M1 |
| 4 | RTX 4060'ta süre | Faz 9 M1 |
| 5 | **Ajan ekseni bir tensör boyutu mu?** İletişim tüm ajanlar için tek ileri geçişte hesaplanabilmeli; A + parametre paylaşımı için kritik | yeni |
| 6 | Birden çok ortamın paralel koşturulması | yeni |
| 7 | Mevcut torch CUDA kurulumuyla çakışmama | yeni |

**Aday sırası (öneri):**
1. **BenchMARL + TorchRL + PettingZoo** (Bettini vd. 2024). Saf PyTorch olduğu ve Ray'e bağımlı olmadığı için Windows riskinin daha düşük olması bekleniyor; bu **doğrulanmadı**, 3. ölçütle ölçülür.
2. **RLlib + PettingZoo** (Liang vd. 2018). Ray'in Windows desteği spike'ta doğrulanmadan seçilmez.
3. **Yedek:** PettingZoo Parallel üzerinde yerel, paylaşımlı parametreli MAPPO. Bir oyuncak ortamda 1. ya da 2. adayın eğrisiyle doğrulanmadan kullanılmaz.

Vision §3.5'in F2 (RLlib) tavsiyesi bu yüzden spike sonucuna bağlanır (§6 K9).

**Spike çıktısı:** seçim + gerekçe; 30 ajan × `attn` × 200K adım süre ölçümü; aynı tohumla iki CPU koşumunun eşitliği. Bu sonuçlarla vision §5.1 hesap tablosu ve §5.1'deki "≥10 tohum" bütçesi yeniden hesaplanır (vision §5.1 v0.2 notu bunu açık iş olarak bırakmıştı).

### 3.3 Karar mekanizması

**Bugünkü akış:**

1. `POST /daily-decision` ([trading.py:1590](../../app/api/routes/trading.py#L1590))
2. `resolve_trade_universe` ([daily_trading.py:117](../../app/services/daily_trading.py#L117))
3. `fetch_latest_market_data` (:261)
4. `build_live_state` (:493)
5. `model.predict` ([trading.py:1728](../../app/api/routes/trading.py#L1728))
6. `interpret_actions_with_risk` (:718)
7. `save_daily_decision` (:956)
8. `POST /apply-decision` (:1901) → `portfolio.apply_decisions` ([portfolio.py:132](../../app/services/portfolio.py#L132)), `applied_dates` bekçisiyle; değerleme `value_portfolio` (:229)

**Hedef akış.** Opt-in; model meta dosyasındaki `kind`/`action_type` alanıyla seçilir. Bu alanları taşımayan her model bugünkü yoldan geçer.

| # | Bileşen | Tasarım | Uyum notu |
|---|---|---|---|
| K1 | **Model paketi + meta** | Ayrıntı tablonun altında | Eski modellerde bu alanlar yok → eski yol (davranış dondurma) |
| K2 | **Ortak gözlem kurucu** | MARL ortamı ve canlı karar aynı saf fonksiyonu çağırır | Kabul: aynı tarih için ortam gözlemi ile canlı gözlem **bit-eş** (yeni test). Eski `build_live_state` ve `TradingEnv`'e dokunulmaz |
| K3 | **Canlı tahmin sinyali** | Karar anında P tablosunu üreten hattın aynısıyla, aynı kalibrasyon parametreleriyle tahmin üretilir. Tahmin eksikse maske + `missing_predictions` raporu | Sessiz 0,5'e düşülmez (§3.1.3) |
| K4 | **Çıkarım ve açıklama** | Logitler → softmax → hedef ağırlık. Dikkat ağırlıkları, β·log c terimleri ve geçit değerleri karar kaydına yazılır (M7; pano) | `trade_decisions.json` şemasına ek alan, mevcut alanlar korunur |
| K5 | **Emir üretimi** | Tek fonksiyon: `rebalance_orders(target_w, prices, holdings, cash, commission, constraints)`. Ortamın tam sayı modu ile canlı karar aynı fonksiyonu kullanır | Nakit tanımları `interpret_actions_with_risk` ile aynı (BUY maliyeti komisyon dahil, SELL geliri komisyon düşülmüş; CLAUDE.md "Kağıt portföy"). Çıktı mevcut karar sözlüğü şemasında ([daily_trading.py:770–781](../../app/services/daily_trading.py#L770)); `apply_decisions` **değişmeden** çalışır |
| K6 | **Risk modları** | Ayrıntı tablonun altında | Pano risk modu seçicisi korunur, anlamı model türüne göre gösterilir |
| K7 | **Yürütme varsayımı** | Backtest ana senaryosu ile canlı uygulama aynı fiyat kuralını kullanır (§3.2.2; Y19) | §6 K12 |
| K8 | **RBAC / çalışma alanı** | Ayrıntı tablonun altında | CLAUDE.md "Do NOT" maddeleri |
| K9 | **Mevcut tek-ajan yolu (Y12)** | Ayrıntı tablonun altında | **Davranış değişikliği** → §6 K13 |
| K10 | **Kağıt portföy** | Tek gerçek kaynak `portfolio.json` kalır; MARL kararları aynı dosyaya, aynı `applied_dates` kuralıyla | Strateji başına ayrı portföy bir ürün kararı (§6 K14). Tez deneyleri backtest'le yapılır, kağıt portföy tez sonucu değildir |

**K1 — meta dosyasına eklenecek alanlar.** Mevcut `write_model_meta` ([daily_trading.py:73](../../app/services/daily_trading.py#L73)) genişler:
- `kind: "marl"`, `action_type: "target_weights"`
- `symbols` (sıralı, CLAUDE.md "Sembol SIRASI önemli"), `agent_ids`, `comm_arm`
- `obs_spec` (sürüm), `norm_stats` (eğitim penceresi), `train_range`, `seed`
- `prediction_source` (P tablosunun `run_id`'si + kalibrasyon parametreleri)
- `risk_projection` (eğitimde kullanılan kısıtlar), `run_id` (manifest)

**K6 — risk modları.** Bugünkü üç mod ([daily_trading.py:27–58](../../app/services/daily_trading.py#L27)) sinyal eşiği, günlük işlem sayısı ve pozisyon sınırı içeriyor. Ağırlık eyleminde "sinyal eşiği"nin anlamı yok; MARL modellerindeki karşılıkları:
- dönüşüm bandı b
- tek pozisyon sınırı
- **sektör maruziyet sınırı** (E2)

Varsayılan değerler **eğitimde kullanılan projeksiyonla aynı**. Kullanıcı farklı bir değer seçerse karar `off_policy_constraint` bayrağı taşır; aynı kısıtla bir backtest yeniden üretilebilir.

**K8 — RBAC ve çalışma alanı.**
- Yeni uçlar (ör. `POST /api/marl/train`, `POST /api/marl/daily-decision`) `RequireWriter` taşır.
- Arka plan eğitimi `ws.use_workspace(user_id)` içinde koşar; model ve manifest `models_dir()` / `training_runs_dir()` altına yazılır.
- Uzun işin sayfası açılışta durumu sorar, iş bitince yoklamayı kapatır.
- `OPTUNA_STORAGE` hâlâ tüm kullanıcılar arasında ortak (CLAUDE.md Faz 8/I). MARL HPO'su başlamadan bu gedik kapatılmalı ya da sınırlılık olarak beyan edilmeli.

**K9 — mevcut tek-ajan yolu.**
- **Yeni** eğitilen tek-ajan modeller `price_stats`, `train_range` ve `seed`'i meta dosyasına yazar; canlı karar bunları kullanır.
- **Eski** modeller için istatistik geriye dönük kesin olarak kurulamaz (Y17). Öneri: canlı kararda "normalizasyon istatistiği yok" uyarısı göster, yaklaşık bir backfill yapma.

**"Sektör düzeyi" karar katmanında üç yerde görünür:** sektör maruziyet sınırı (K6), sektör toplu dikkat/bulaşma haritası (K4 → pano, M7) ve sektör bazında getiri katkısı raporu.

---

## 4. Sıra ve bağımlılık

```
Faz 9:  A (değerlendirme) ─┐
        C (tekrar üretim) ─┤
        B (+B7–B10) ───────┤
        E (evren; +Y11) ───┼──► P (nokta-zamanlı tahmin) ──► Milestone 0' (tek-ajan + DT/DTF) ──► Makale 2
        F (+F4) ───────────┤                                   │
        G (ödül) ──────────┤                                   ▼
        D (+Y19 duyarlılık)┘          M2 ortam + K (karar katmanı) ──► M3/M4 ──► M5 ──► M6/M7 ──► M8 ön kayıt ──► nihai koşumlar
                                                 ▲
        M1 spike ────────────────────────────────┘   (A–G ile paralel; sapma S1)
```

**Faz 9'un "önce geçerlilik, sonra MARL" sırası korunuyor.** Sapmalar ve gerekçeleri:

| # | Sapma | Gerekçe | Koşul |
|---|---|---|---|
| S1 | **M1 spike'ı A–G ile paralel** (Faz 9 diyagramında Milestone 0''dan sonra) | Spike tez/makalede kullanılacak bir sayı üretmez, yalnızca fizibilite ve süre ölçer. Takvimin en büyük bilinmeyeni çerçeve/Windows uyumu ve hesaplama bütçesi; Makale 2 tarihi (§6 K5) ve vision §5.1 bütçesi bu ölçüme bağlı | Spike'ta üretilen hiçbir performans sayısı rapora girmez; deneme kaydına (C2) "spike" etiketiyle yazılır |
| S2 | **P paketi Milestone 0''dan önce** (Faz 9'da P yok) | Vision §7 Milestone 0'daki "Ensemble ile ve ensemble olmadan (DT vs DTF)" ablasyonu ve M5, RL eğitim dönemine örneklem dışı tahmin gerektiriyor (Y14) | F'den sonra (güven tanımı ve kalibrasyon P tablosuna yazılıyor) |
| S3 | **A seçilirse E2 kritik yoldan çıkar** | A'da sektör eşlemesi yalnızca `gat` kolunu ve sektör sınırını etkiliyor; `none` ve `attn` kolları onu beklemez. E1 (evren) kritik yolda kalır | B seçilirse E2 M2'nin ön koşuludur |

**Kaba efor tahmini** (tek geliştirici, gün; ölçüm değil, M1 spike'ından sonra yeniden hesaplanır):

| Grup | Paketler | Gün |
|---|---|---|
| Faz 9 | A 6 · B 8 · C 3 · D 5 · E 2 · F 3 · G 2 (Faz 9 §4) | ≈ 29 + B5 hesaplama süresi |
| Yeni geçerlilik | B7 0,5 · B8 1 · B9 1 · B10 1 · F4 0,5 | ≈ 4 |
| Tahmin sinyali | P1–P3 | ≈ 4–6 + hesaplama |
| Karar katmanı | K1–K10 | ≈ 5–7 |
| MARL | M1 5 · M2 5–7 · M3 3–5 · M4 5–7 · M5 4–6 · M6–M7 4 · M8 2 | ≈ 28–36 |

---

## 5. Kısıtların korunması

| Kısıt | Bu planda nasıl korunuyor | **Kıran ya da değiştiren öneri** |
|---|---|---|
| State space uyumluluğu | MARL yeni ortamda. Eski `TradingEnv`'de boyut değişmiyor; B9 yalnızca opt-in parametre | Yok |
| Varsayılan davranış dondurma | Yeni davranışlar bayrakla geliyor; MARL yolu meta alanıyla seçiliyor | **B1** (Faz 9; golden değişir), **B10** (TATS varsayılan açık; golden değişir), **B7** (RL gözlem değerleri, ilk günler; bayrakla), **B8** (CV metrikleri), **K9** (Y12 düzeltmesi yeni modellerde canlı kararı değiştirir) → hepsi §6 K1 ve K13 kapsamında |
| Kullanıcı bazlı çalışma alanı | P tablosu, MARL modeli, manifest ve kararlar çalışma alanı çözücüsünden; arka plan görevi `use_workspace` ile | `OPTUNA_STORAGE` ortak (mevcut gedik, bu plan yaratmıyor) |
| RBAC | Yeni yazma uçları `RequireWriter`; okuma uçları `CurrentUser` | Yok |
| Vision belgesi | — | **K6** (A seçimi: vision §3.1, §4.1, §4.3, §7 ve Faz 9 M2/M3 metinleri değişir), **K9** (vision §3.5 F2 tavsiyesi spike'a bağlanır), **K10** (vision §3.4 Seçenek III yeniden tanımlanır) |

---

## 6. Senden karar bekleyen konular

Faz 9 §7'deki açık kararlar (1–5) ve bu turda çıkanlar (6–16):

| # | Karar | Önerim | Neyi etkiler |
|---|---|---|---|
| **K1** | **Tahmin golden'ı** (Faz 9 #1; B1, şimdi B10 da) | **KARAR (2026-11-06): kabul.** Varsayılanı çevir, golden'ı Milestone 0' ile birlikte bir kez yenile. B1 ve B10'un ikisi de varsayılan hatta etkin; ikisini ayrı ayrı yenilemenin akademik değeri yok. Golden yenilemesi GPU'lu makinede: `python tests/test_prediction_regression.py --update` (CLAUDE.md) | `tests/test_prediction_regression.py`; tüm tahmin çıktıları; mevcut tahmin modelleri yeniden eğitilmeli |
| **K2** | **EVDS erişimi** (Faz 9 #2; B2) | **KAPANDI (2026-11-06):** anahtar geçerli; neden yanlış seri kodları + ayrıştırma hatalarıydı (Faz 9 §9.3). Eski metin: Anahtar `.env`'de tanımlı, ama son çekimde üç seri yedeğe düştü (§1.2). **Senin işin:** EVDS hesabında anahtarın geçerli olduğunu teyit etmek. Teşhis (seri kodları / uç) B2'de yapılır | B2, B3, D3 (TÜFE-reel getiri) **ve A8**: risksiz faiz politika faizi serisinden gelecekse EVDS çalışmadan A8 tamamlanamaz |
| **K3** | **Tarihsel BIST-30 bileşenleri** (Faz 9 #3; E1) | Faz 9'daki gibi: önce temin dene, 2 haftada bulunamazsa hayatta kalma yanlılığını sınırlılık olarak beyan et (Brown vd. 1992). K7'ye bağlı | E1, D5 evren maskesi, Makale 1–2 sınırlılık paragrafı |
| **K4** | **Ödül terimi** (Faz 9 #4; G1) | Faz 9'daki gibi: Milestone 0' ön elemesinde `differential` ve `ema_level` ikisi de denensin, seçim deneme kaydına girsin | G paketi, MARL portföy terimi Rₜ |
| **K5** | **Makale 2 hedef tarihi** (Faz 9 #5) | **Aralık 2026 bırakılsın.** Bugün 2026-11-06. Faz 9'un A+B+C'si tek başına ≈ 17 gün; buna B5 hesaplama süresi, B10 (yeni, K3/K4 kollarının ön koşulu), Milestone 0' koşumları (≥10 tohum) ve yazım ekleniyor. Yeni tarih B5 maliyet ölçümünden ve M1 spike'ından sonra konsun | Makale 2 takvimi, vision §6.2 |
| **K6** | **Ajan granülaritesi** (yeni) | **A, hisse başına 30 ajan** (§2). G3 ertelenir; B, Milestone 3'te keşifsel ablasyon olabilir | vision §3.1, §4.1, §4.3 ("8-sektör"), §7 Milestone 2; Faz 9 M2 (ajan = sektör) ve M3 (MPLS zorunlu); G3 |
| **K7** | **"BIST-30" evren tanımı** (yeni; Y11) | **Milestone 0' için panelin bugünkü 30 sembolü C3 anlık görüntüsüyle dondurulsun** (mevcut modellerle karşılaştırılabilirlik). `bist30_symbols.py`'deki listenin panelle aynı olmadığı belgelensin. Nihai evren E1'e göre | E1, E2 (sektör sınıflaması hangi 30 için), mevcut model meta dosyaları, P tablosu |
| **K8** | **M1 spike'ının öne çekilmesi** (sapma S1) | **Evet**; A–G ile paralel, sonuçları rapora girmez | Takvim; K5 ve vision §5.1 bütçesi |
| **K9** | **Çerçeve önceliği** (yeni) | BenchMARL/TorchRL'i önce dene, RLlib'i ikinci (§3.2.6). Vision §3.5 F2 tavsiyesi spike sonucuna bağlansın | M1, M2–M5 kodu, vision §3.5 ve §5.3 |
| **K10** | **Seçenek III'ün yeniden tanımı** (yeni) | "Meta-learner → rejim" yerine **nedensel kural tabanlı rejim özellikleri** (B6), HMM filtrelenmiş olasılıklar ikinci tanım. AQ3'teki döngüsellik beyan edilsin | vision §3.4 III, M6, AQ3 analizi |
| **K11** | **P tablosunun model kapsamı** (yeni) | P3 maliyet ölçümünü bekle. Ölçüm ağır çıkarsa **v1 = üç GBM + lojistik meta** (her 63 günde yeniden eğitim); BiLSTM/TFT daha seyrek yeniden eğitimle ya da yalnızca Makale 2 kollarında | P1, F1 (güvene hangi modeller katılıyor), AQ2 sonuçlarının yorumu |
| **K12** | **Backtest yürütme fiyatı** (yeni; Y19) | **Yeni ortamda ana senaryo: t kapanışında karar, t+1 açılışında yürütme**; t kapanışı duyarlılık. Eski ortam değişmez | M2, D4, K7, tüm MARL sonuçları |
| **K13** | **Eski tek-ajan modellerin canlı normalizasyonu** (yeni; Y12) | Eski modellerde **uyarı göster, yaklaşık backfill yapma** (eğitim penceresi kurulamıyor, Y17). Yeni modeller istatistiği meta dosyasına yazar | `/daily-decision` davranışı, kağıt portföy sonuçlarının yorumu |
| **K14** | **Strateji başına kağıt portföy** (yeni; ürün) | Düşük öncelik; MARL karar yolu çalıştıktan sonra. Tez sonucu değil | `portfolio.py`, pano |
| **K15** | **SARL kontrol kolları** (yeni) | SARL-ağırlık ve SARL-eşmimari ikisi de nihai tasarımda olsun (§3.2.4) | Hesaplama bütçesi (+2 yapılandırma × tohum), H1'in yorumu |
| **K16** | **Faz 9'un onayı** | Faz 10 Faz 9'a bağlı; önce Faz 9 (bu plandaki B7–B10 ve F4 eklemeleriyle) onaylanmalı | Tüm plan |

**hakem-2 incelemesinden çıkan kararlar** (§8). K23 güvenlik maddesiydi; 2026-11-06'da kapandı (aşağıda).

| # | Karar | Önerim | Neyi etkiler |
|---|---|---|---|
| **K23** | **EVDS API anahtarı kaynak kodda ve herkese açık depoda** (§8 #17) | **KARAR (2026-11-06): anahtar koddan kaldırıldı, `.env`'den okunuyor (PR #3, 8a5cdb8); anahtar yenilenmeyecek** (kullanıcı kararı). Git geçmişi yeniden yazılmadı; eski değer c87cc38..8a5cdb8 arası commit'lerde ve herkese açık depoda görünür kalıyor. Bu, kabul edilmiş bir risk olarak kayıtlı | Güvenlik (kabul edilen risk); B2 teşhisi mevcut anahtarla yapılır |
| **K17** | **Veri bütünlüğü kapısı (yeni paket V) ve RL veri penceresi** (§8 #1) | V paketi A–G ile aynı kapıda, MARL'dan önce. İçeriği:<br>• artefakt tarayıcı: seviye sıçraması, \|r\| eşiği, 2005 YTL geçişi<br>• hayalet gün filtresi<br>• sembol aktiflik maskesi: softmax yalnızca aktif ajanlar üzerinde; komşu sembol kopyalanmaz<br>• B0'a bir veri sağlığı testi<br>**KARAR (2026-11-06): kabul; uygulandı** (`data/integrity.py`, `RL_DATA_START`; Faz 9 §9). **RL penceresi 2013-06'dan başlasın:** panelde en geç başlayan sembol ODAS (2013-05-21, Ek A komut 1), yani bu tarihten sonra 30 sembolün hepsi panelde. 2000–2005 artefaktları ve 2009 sıçraması pencere dışında kalır; tarayıcı yine de çalışır. Vision §4.1'deki "2018-2026"ya göre örneklem dışı gün sayısı artar (#8) | vision §4.1; tüm RL sonuçları; mevcut `results/*.json` sınırlılık olarak işaretlenir |
| **K18** | **Gözlem yeniden tasarımı ve ADX düzeltmesi** (§8 #2) | **Eski ortam kısmı uygulandı (2026-11-09, Faz 9 §9.9):** ADX `wilder` bayrağı (rota varsayılanı), `obs_health` + `features_in_support` protokol maddeleri; MARL `obs_spec` M2'de bu ölçütle. MARL ortamında ölçekten bağımsız özellik seti (log-getiri, kayan z, oran göstergeleri). Kabul ölçütüne "gözlem sağlığı" eklenir: kırpılan ve sabit özellik oranı eğitim ve testte ayrı ölçülür. Eski ortamda ADX düzeltmesi bayrakla gelir (davranış değişikliği; mevcut RL modelleri ADX ≡ 0 ile eğitildi). B9 eski ortamda doygunluk yaratıyor; bu beyan edilir (ölçüldü: test fiyat özelliklerinin %74,7'si kırpılıyor, %98,9'u eğitim desteği dışında; Faz 9 §9.5). K18 eski ortamın fiyat ölçeğini DEĞİŞTİRMİYOR → K25 | `data/technical_indicators.py`, eski RL modelleri, M2 `obs_spec`; state space boyutu değişmez |
| **K19** | **Topluluğun dağıtım öncesi yeniden uyumu** (§8 #7) | P tablosu ve canlı karar için meta-öğrenici ve kalibrasyon **yalnızca örneklem dışı** tahminlerle uydurulur. Yani %80 yeniden uyum + örneklem içi meta yeniden uyumu ([ensemble.py:345–384](../../prediction/models/ensemble.py#L345)) kapatılır (bayrak) ya da çapraz uydurmayla yapılır. P1 kabul ölçütüne iki madde eklenir: "meta/kalibrasyon girdileri örneklem dışı" ve "t için son eğitim tarihi kayıtlı" | Tahmin golden'ı (K1 kapsamı), P1, F2, K3 |
| **K20** | **RL walk-forward, çıkarım birimi ve güç** (§8 #8) | Yeni iş maddesi **R1:** RL katları, purge, doğrulamada erken durdurma / kontrol noktası kuralı. M8'de sabitlenecekler:<br>• çıkarım birimi: kat × tohum üzerinden birleştirilmiş örneklem dışı günlük getiri serisi<br>• A3 boyut/güç simülasyonuyla MDE<br>• M1 bütçesine göre kol budama kuralı<br>Hakemin kaba hesabıyla 1.000 günde yıllık Sharpe farkı MDE'si ≈ 0,4–0,6 (yalnızca büyüklük mertebesi). Vision §4.3'teki p<0,05 başarı ölçütü bu güçle yeniden değerlendirilmeli | vision §4.3; M8; hesaplama bütçesi |
| **K21** | **TATS kapsamı** (§8 #6) | B10 genişler: TATS test bölümünde TATS'lı ve TATS'sız birlikte değerlendirilir; diske yazılır ve yüklenir; makaleyle yöntem farkları tablosu çıkarılır (3 sınıf ↔ ikili sınıflandırıcı; arXiv 2601.12706 tam metni). Makale 2'nin K3/K4 kolları bunlar bitmeden koşulmaz | Makale 2, K5 |
| **K22** | **Literatür düzeltmesi: MAPS / MSPM / Hao** (§8 #4) | Hao vd. 2025 tam metni okunup `literature-review-verification.md`'ye kaydedilsin. MAPS ve MSPM rol metinleri `/kaynak-ekle` hattıyla düzeltilsin (meta.py ROLE → build.py; tablo elle düzenlenmez); vision §2.1:76 güncellensin | literature-table.md, vision §2.1 |
| **K24** | **Parametre paylaşımının kapsamı** (§8 #5) | Ana tasarım tam paylaşım + ajan kimliği. **Sektör başına seçici paylaşım** (Christianos vd. 2021) Milestone 3'te ablasyon olarak eklenir (E2 gerekir). Bu, sektörü ajan yapmadan "sektör düzeyi"ni ve vision §1.2(2)/§1.3'teki heterojenlik motivasyonunu geri getirir. G3 ertelemesinin etkilediği yerler listesine roadmap.md:135, :160 ve vision §6.1 Bölüm 5 başlığı eklenir | vision §1.2–1.3, §6.1; roadmap; M3 |
| **K25** | **Eski ortamda nedensel, destek içi fiyat ölçeği** (Faz 9 §9.6 #1, #2; 2026-11-06) | **KARAR (2026-11-09, kullanıcı): (b).** K18 dalında uygulandı (Faz 9 §9.9): `price_norm='causal'`, W=252, yalnız O/H/L/C. Öneri (b) idi. *Uygulama notu (karar metni değil):* MACD de fiyat cinsinden (TL); nedensel ölçeğe alınması `price_norm_macd=True` ile opt-in, varsayılan kapalı — karar K26'da (hakem-2 K18 #2). (a) Yalnız beyan: eski ortamın B9 sonrası test sayıları tez dışı; Milestone 0' Ansari replikasyonu dağılım dışı girdiyle raporlanır. (b) Eski ortama opt-in nedensel ölçek: `price_norm='causal'`, yalnız ≤ t veriyle kayan ya da genişleyen z, B0'a tabi. Milestone 0'da iki kol koşulur: B9 ölçeği (sadık, `price_in_support` açık, beyanlı) ve nedensel ölçek (destek içi). Hakem-2'nin istediği üçüncü kol bu ikincisidir; ≥ 10 tohum ve GA ile Y13'ün metriğe etkisini ayırt etmeye yarar. Gerekçe: ölçeğe bağlı fiyat özelliğiyle eski ortamın her B9 sonrası test sayısı dağılım dışı (Faz 9 §9.5) | `env/trading_env.py` (davranış dondurma: yeni bayrak, varsayılan değişmez), K18 kapsamı, Milestone 0', Makale 1 replikasyon tablosu |
| **K26** | **Eski ortamda ölçeğe bağlı fiyat dışı özellikler: MACD, türbülans, hacim** (hakem-rl K18 #1, hakem-2 K18 #1–#2; 2026-11-09) | **KARAR (2026-11-09, kullanıcı): (a), yalnız beyan.** Gerekçe:<br>(1) Tezin birincil tek ajan ↔ çok ajan karşıtlığı MAPPO ↔ SARL-eşmimari, "aynı α, aynı gözlem, aynı tohumlar" (§3; K15). Yani yeni ortamın gözlemiyle yapılıyor, eski ortamla değil; ölçeksiz özellik seti M2 `obs_spec`'inde bir kez tasarlanır.<br>(2) K25'in amacı (Y13'ün etkisini ayırmak) bu özelliklere bağlı değil; MACD, türbülans ve hacim iki kolda da aynı.<br>(3) Eski ortam Ansari replikasyonu olarak sadık kalır; sınırlamalar ölçülmüş sayılarla beyan edilir ve protokolde açık maddeler olarak makine tarafından uygulanır.<br>(4) Ek kollar hesaplama ve DSR deneme sayısı (N) maliyeti getirir, plandaki hiçbir hipoteze hizmet etmez.<br>Sonuç: eski ortam kayıtları `obs_health` (türbülans), `features_in_support` ve `dims_in_support` ile açık kalır. Aşağıdaki ölçülmüş adaylar M2 `obs_spec` tasarımına girdi olarak taşınır. `price_norm_macd` bayrağı kodda opt-in kalır; M8 ön kaydı gerektirmedikçe Milestone 0''da kullanılmaz. Eski ortamdan temiz bir tek ajan sonucu ancak plan dışı bir amaç (örn. MARL öncesi ayrı yayın) çıkarsa yeniden açılır. *Karardan önceki öneri (b) idi:* Ölçüm (Faz 9 §9.9, JSON `docs/development/measurements/phase9-k18/obs_health.json`): MACD `tanh(macd/0,1)` TL cinsinden, testte %75 doygun, değerlerin %28,7'si eğitim desteği dışında (`features_in_support` açık). Türbülans `tanh(turb/2)` eğitimde %99,7, testte %100 doygun; 30 sembollü Mahalanobis uzaklığı 2'nin çok üstünde, özellik pratikte sabit +1. Bu yüzden eski ortamın HER kolu `obs_health` ile açık. Hacim pay sayısı cinsinden; sermaye artırımları seviyeyi kaydırıyor. Sembol düzeyinde destek dışı (`dims_in_support`), sayılar §9.9'da. (a) Yalnız beyan: eski ortam kayıtları açık kalır; Milestone 0' replikasyonu bu girdilerle raporlanır. (b) Opt-in ölçeksiz sürümler, varsayılan değişmez. MACD için `tanh(macd/std_t)` mevcut (`price_norm_macd=True`). Türbülans için `tanh(turb/n)` önerisi: n = sembol sayısı; çok değişkenli normal altında Mahalanobis uzaklığının karesi χ²(n), beklenen değeri n (tasarım gerekçesi). Hacim için nedensel kayan z adayı ölçülmedi. Milestone 0'da bunlar ayrı etkenler olarak koşulur (M8 ön kaydı); kollar her özellikte ayrışır | M2 `obs_spec` (ölçeksiz özellik adayları), Milestone 0' ve Makale 1 replikasyon raporlaması (beyan); eski ortam kodu değişmez |

---

## 7. Kaynak kullanımı notu

- Bu planda atıf yapılan her literatür kaynağı [literature-table.md](../thesis/literature-table.md)'deki seçilmiş kayıtlardandır. Yeni kaynak eklenmedi; literatür tablosu düzenlenmedi.
- Aşağıdaki tasarım kararları için tabloda bir dayanak yok; metinde **tasarım varsayımı** olarak beyan edilir:
  - nakit logitinin sabit 0 olması
  - P tablosunda 63 günlük yeniden eğitim periyodu
  - güven kırpma aralığı [c_min, c_max] (Faz 9 M5 ile aynı durum)
  - dönüşüm bandı b
  - α ızgarası
- **Tam metni okunmamış kaynaklar.** Hung & Huang (2026), Yuan vd. (2026) ve Xiong vd. (2025) tam metni okunmamış olarak işaretli (vision §2.1, §2.3). Bu planda bunlara özetlerinde olmayan bir iddia yüklenmedi.
- **Tam metin arşivi.** Arşiv (`docs/reference/papers/`) bu makinede yok. Faz 9'dan aktarılan tam metin gözlemleri (ör. MAPPO "Öneri 2", DebUnc "Attention Scaling") **Faz 9 planına** atıf olarak verildi; bu turda yeniden okunmadı.

---

## 8. Hakem-2 itirazları

**İnceleme:** 2026-11-06, `hakem-2` alt ajanı (salt okunur). **Karar: Büyük revizyon.**

**Kanıt durumu:**
- Hakemin dosya:satır ve ölçüm iddialarından şunları bu turda kodda/veride **ayrıca doğruladım**: #1 (ADX hariç veri ölçümleri: TUPRS/FROTO artefakt satırları, eğitim penceresi bitişi 2018-07-20, trading_env.py:665–678 komşu sembol kopyası), #2 (ADX: 173.558 satırın hepsinde 0), #6 (`correction_factor` 0,5, işaret korunuyor), #7 (ensemble.py:345–384), #12, #16 (çağıran satırlar), #17 (anahtar satırı, git'te izleniyor, `origin/main`'de, depo PUBLIC).
- Hakemin kendi ölçümü olup **benim yeniden üretmediklerim:** hayalet gün sayısı (88), eksik hücre oranı (%20,2), doygunluk ve z-skor yüzdeleri, korelasyon yoğunlukları, MDE hesabı. Bunlar hakemin `python -I` tek satırlıklarından geliyor; rapora girmeden V/K18 işlerinde yeniden ölçülmeli.
- MAPS ve MSPM tam metin alıntıları hakem tarafından web'de doğrulandı. URL'ler aşağıda; ben ayrıca okumadım.

Yanıt değerleri: **Kabul** = plana işlendi · **Kısmen** = gerekçeli sınırlama · **Ret** = gerekçeli.

| # | Cid. | İtiraz (kısa) | Hakemin kanıtı (özet) | Yanıt | Plana etkisi |
|---|---|---|---|---|---|
| 1 | **RED** | Veri bütünlüğü hiçbir pakette yok; RL eğitim penceresi vision ile çelişiyor | `split_data` 70/15/15 → eğitim 2000-05-10..2018-07-20. TUPRS 762, FROTO 1.051 satır > 10.000 TL; −%100 / +%98.113 / +%304 getiriler; 88 hayalet gün; eğitim hücrelerinin %20,2'si eksik; eksik sembolde önceki sembolün özellikleri kopyalanıyor (trading_env.py:676) | **Kabul** | Yeni V paketi + RL penceresi kararı → §6 K17. §0 madde 8 |
| 2 | **RED** | MARL yerel gözlemi "mevcut dönüşümlerle" dejenere; B9 kırpma yaratıyor | ADX ≡ 0 (technical_indicators.py:116–121, indeks hizalaması); türbülans tanh'ı %98,8 doygun; B9 ile test fiyat z'lerinin %80,3'ü \|z\| > 10 | **Kabul** | §3.2.3'teki "mevcut dönüşümlerle" yerine ölçekten bağımsız özellik seti + gözlem sağlığı ölçütü → §6 K18 |
| 3 | BÜYÜK | "Çarpanlara ayrılmış tek-ajan" itirazına karşılık yetersiz; birincil karşıtlık tanımsız | Softmax tüm zⱼ'leri bağlıyor; EIIE tek-ajan; α=0'da MAPPO ↔ SARL-eşmimari farkı oran kırpmasında; SARL-ağırlık 31 ↔ MARL z₀ sabit | **Kabul** | §2.3 yeniden yazıldı (biçimsel denklik kabul; birincil karşıtlık MAPPO ↔ SARL-eşmimari, aynı α, M8'de). §3.2.4 SARL-ağırlık 30 logit + z₀ sabit. A önerisi korunuyor: aynı itiraz B'deki bütçe softmax'ı için de geçerli |
| 4 | BÜYÜK | Varlık başına ajan öncülleri yanlış atfedilmiş | MAPS: "each agent is an independent 'investor'" (doğrulandı: https://www.ijcai.org/proceedings/2020/0623.pdf). MSPM: EAM sinyal, SAM dağıtım (doğrulandı: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263689). Hao 2025: nitelik yalnızca vision §2.1:76'da | **Kabul** | Gerekçe 8 geri çekildi. Tablo ve vision düzeltmesi → §6 K22 |
| 5 | BÜYÜK | G3 ertelemesinin etkileri eksik; tam paylaşım heterojenlik motivasyonuyla çelişiyor | roadmap.md:135, :160; vision §6.1 Bölüm 5; vision §1.2(2), §1.3; SePS rolü "sektör ajanlarında paylaşım kararı" | **Kabul** | Sektör başına seçici paylaşım ablasyonu ve etkilenen yerlerin tam listesi → §6 K24 |
| 6 | BÜYÜK | Y20'nin yön etkisi abartılmış; değerlendirme ve kalıcılık eksik | `correction_factor` 0,5 → işaret korunur (tats.py:45, :150–156); test metrikleri TATS'sız (ensemble.py:318); TATS diske yazılmıyor; 3 sınıf ↔ makalede ikili | **Kabul** | Y20 gövdede düzeltildi. B10 genişliyor → §6 K21 |
| 7 | BÜYÜK | Dağıtımda meta-öğrenici örneklem içi tahminlerle yeniden uyduruluyor; F/P bunun üzerine kuruluyor | ensemble.py:345–349 (%0–80 ile eğitim), :366 (X_meta üzerinde örneklem içi tahmin), :380–384 (meta yeniden uyumu) | **Kabul** | → §6 K19. CLAUDE.md'deki "data leakage yok" ifadesi (ensemble.py özellikleri) bu nedenle de doğru değil; düzeltmesi Faz 9/10 onayından sonra belge işi |
| 8 | BÜYÜK | RL walk-forward, model seçimi, çıkarım birimi ve güç tanımsız | Y17'nin C/D'ye atanması RL katlarını tanımlamıyor; kaba MDE 1.000 günde 0,44–0,63; kol sayısı bütçeyle eşlenmemiş | **Kabul** | Yeni R1 iş maddesi, M8 çıkarım birimi, MDE ve kol budama → §6 K20 |
| 9 | BÜYÜK | Y12 eksik; "yapısal olarak kapanır" kanıtsız | Canlı pencere 30 gün (trading.py:1697) → türbülans 0; EMA ısınması; canlıda ±10 kırpma yok | **Kabul** | Y12 gövdede genişletildi; §0 madde 5 düzeltildi. K2 kabul ölçütüne "canlı gözlem eğitimdekiyle aynı ısınma uzunluğuyla (≥ 252 seans + gösterge penceresi)" eklenir |
| 10 | BÜYÜK | Kalibre güvenin dinamik aralığı dar; H3 ölçek farkıyla karışabilir | c ≥ 0,5 → log c ∈ [−0,69, 0]. Tek ölçümde yön isabeti %50,66: `workspaces/dbe145fbf9624a77831ee7aba2d54854/models/prediction/AKBNK_IS_daily_ensemble_meta.json` (tek sembol, tek koşum; ben okumadım) | **Kabul** | M5'te önsel ölçekten bağımsız tanımlanır (standartlaştırılmış log-odds), kalibre ve kalibresiz kollar aynı standartlaştırmayla karşılaştırılır. Karıştırılmış-güven kolu zorunlu. P2/F3'te c dağılımı sembol başına raporlanır |
| 11 | BÜYÜK | Kritik kararlar §6'da yok | #1, #2, #5, #6, #7, #8, #13, #17 | **Kabul** | §6'ya K17–K24 eklendi. #13 aşağıda; M2'de muhasebe denklemiyle kapanır |
| 12 | KÜÇÜK | Oy oranı yedeğinde birim hatası | ensemble.py:552 `p > current_close`, p log-getiri → güven hep 1,0 | **Kabul** | Y4 gövdede düzeltildi. F1 bitmeden K11 v1 (yalnız GBM) çalıştırılmaz: yön başlığı olmayan yol tam bu hataya düşer |
| 13 | KÜÇÜK | Ödül/maliyet zamanlaması ve birimi belirsiz | kᵢ,ₜ zaman indeksi; gece/seans içi bölünmesi; "katkı toplamı = Rₜ" diferansiyel Sharpe'ta tutmaz; c·Σ\|Δw\| ile gidiş-dönüş bps çift sayım; nakit | **Kabul** | M2'de zaman indeksli tam muhasebe denklemi yazılır. c **tek yön** tanımlanır (D4 ızgarası gidiş-dönüş/2). Nakit maliyet toplamına girmez. Katkı ayrıştırması yalnızca basit getiri terimi için iddia edilir |
| 14 | KÜÇÜK | "Graf seyrek kalabilir" ölçülmemiş, ölçüm tersini gösteriyor | Ortalama korelasyon 0,398; eşik 0,3 → yoğunluk %77,5 | **Kabul** | Gerekçe 3 düzeltildi. Kenar kuralı (MST ya da top-k) M4 öncesi sabitlenir |
| 15 | KÜÇÜK | Sektör gömmesi sonradan eklenirse kollar farklı gözlem görür | M4 eşit bütçe aynı gözlemi gerektirir | **Kabul** | `obs_spec` sabitlenmeden raporlanacak koşum başlamaz. **S3 daralıyor:** E2, M4 nihai koşumlarının ön koşulu (ön eleme için değil) |
| 16 | KÜÇÜK | B9 başka çağıranları kapsamıyor | scripts/analysis/generate_academic_report.py:57; hyperparameter_optimization/base_optimizer.py:349 | **Kabul** | B9 kapsamı tüm `TradingEnv` çağıranlarına genişler. **Uygulandı** (Faz 9 §9.5; AST bekçisi) |
| 17 | KÜÇÜK → **güvenlik** | EVDS anahtarı kaynak kodda | base_optimizer.py:341, git'te izleniyor | **Kabul, ciddiyet yükseltildi:** depo PUBLIC ve satır `origin/main`'de | → §6 **K23** (en önce) |
| 18 | KÜÇÜK | Kağıt portföyün yürütme fiyatı backtest'le çelişiyor | `apply_decisions` karar anındaki t kapanışı maliyetini kullanıyor (portfolio.py:170, :194) | **Kabul** | K7/K12: t+1 açılış yalnızca backtest için. Kağıt portföyde ya ertesi gün açılışla yeniden fiyatlama adımı eklenir (K14 ile birlikte) ya da fark kağıt portföy için sınırlılık olarak yazılır. Tez sonucu backtest'ten gelir |
| 19 | KÜÇÜK | Rejim etiketinin nedensellik ayrıntıları eksik | Tercil eşiklerinin örneklemi; HMM parametre penceresi; ikinci tanım aynı XU100'den | **Kabul** | S-reg: eşikler genişleyen pencereyle, HMM her kat penceresinde yeniden uydurulur, ikisi B0'a girer. İkinci tanımın bağımsız bir tekrar olmadığı beyan edilir |
| 20 | KÜÇÜK | Atıf ve kısıt ayrıntıları | Y17 ":1163" trading.py'de; `HYPEROPT_DIR` (prediction/hyperopt.py:24) çalışma alanı çözücüsünü kullanmıyor; kullanıcılar arası GPU yuvası yok | **Kabul** | Y17 atfı düzeltildi. §5 çalışma alanı satırına `HYPEROPT_DIR` ve GPU yuvası eklenir: K8, MARL eğitimini mevcut `DL_GPU_SLOTS` semaforuna bağlar ya da tek iş kuyruğu kurar |
| 21 | KÜÇÜK | Özgünlük dili | G4 YÖK koşulu yok; kalibresiz kol "katkı" sayılmış; "ilk karşılaştırma" kapsamsız | **Kabul** | §2.3 düzeltildi ("bildiğimiz kadarıyla" kalıbı, G4 koşulu, ablasyon ≠ katkı) |
| 22 | KÜÇÜK | K13 gerekçesi abartılı | CSV yalnızca eklemeyle büyüdüyse pencere `created_at` + oranla kurulabilir | **Kısmen** | Gerekçe "koşullu olarak kurulabilir, doğrulanamaz" olur; öneri değişmez (uyarı göster, eski modeller tez dışı). Y17 tablosu bu şekilde okunmalı |

**Ret edilen itiraz yok.** A önerisi korunuyor; #3 ve #5 onu çürütmüyor, tanımı ve kontrollerini sıkılaştırıyor. Gerekçe: aynı biçimsel denklik B'de de var (grup bütçesi softmax'ı), ve G2'nin kalibre güven gerekçesi (§2.2 #1) itirazlardan etkilenmiyor.

**Sıraya etkisi (§4).** V paketi (#1) ve gözlem sağlığı (#2), A ve C ile birlikte **ilk kapıya** girer. RL'de üretilen her sayı bu iki kapıdan sonra geçerli sayılır. S3 daralır (#15).

---

## Ek A. Ölçüm komutları

Bu planda geçen sayısal durum bilgileri aşağıdaki komutlardan alındı (2026-11-06, depo kökünde):

1. **Panel evreni ve tarih aralığı (Y11, §1.2):**
   ```bash
   python -I -c "import pandas as pd; df=pd.read_csv('data/bist/raw_stock_data.csv',usecols=['date','symbol']); g=df.groupby('symbol')['date'].agg(['min','max','count']); print(len(g)); print(g.sort_values('min'))"
   ```
   Sonuç: 30 sembol; son tarih 2026-07-27; en geç başlayanlar PGSUS (2013-04-26) ve ODAS (2013-05-21).
2. **Panel ↔ liste farkı ve makro sabitleri (Y11, B2):** aynı CSV ile `BIST30_SYMBOLS` kümesinin farkı; `data/macro/macro_data.csv`'de `policy_rate`, `cpi_inflation` ve `ppi_inflation` için `nunique()` → 1, 1, 1 (değerler 50,0 / 0,0 / 0,0).
3. **Tahmin özelliğinin RL'e bağlanmaması (Y14):** repo genelinde (venv hariç) `prediction_features\s*=` araması → yalnızca `env/trading_env.py:100`.
4. **Faz 9'dan bu yana kod değişikliği:**
   ```bash
   git diff --stat b82fedb..HEAD -- '*.py' ':!scripts/literature'
   ```
   Sonuç: yalnızca `.claude/hooks/guard.py`.

---

## Versiyon geçmişi

- **2026-11-06:** İlk plan.
  - Y1–Y10 kodda yeniden doğrulandı (hepsi geçerli).
  - 10 yeni bulgu (Y11–Y20).
  - Ajan yapısı önerisi A.
  - Üç katmanın planı; yeni paketler B7–B10, F4, P, K.
  - Sıra sapmaları S1–S3; 16 karar maddesi.
  - **Kod değişikliği yok.**
- **2026-11-06 (hakem-2 sonrası):** hakem-2 "Büyük revizyon" verdi; 22 itiraz §8'e işlendi (21 kabul, 1 kısmen).
  - Gövdede düzeltilen olgu hataları:
    - Y20: TATS yönü değiştirmiyor
    - §2.2 gerekçe 8: MAPS/MSPM yanlış atıf, geri çekildi
    - gerekçe 3: graf seyrekliği ölçümle çelişiyordu
    - Y4: oy yedeğinde birim hatası
    - Y12: canlı gösterge ısınması
    - Y17: satır atfı ve eğitim penceresi
    - §0 madde 5 ve §2.3: itiraza karşılık ve özgünlük dili
  - Yeni karar maddeleri K17–K24; K23 (herkese açık depoda EVDS anahtarı) en öncelikli.
  - **Kod değişikliği yok.**
- **2026-11-06 (onay):** Plan Faz 9 ile birlikte onaylandı. K1, K17 ve K23 kararları §6'ya işlendi. İlk kapı V + A + C uygulandı (Faz 9 §9). K18 (gözlem sağlığı) bu kapıda yapılmadı.
