# Faz 6 — Backend Performans & Eğitim Throughput Sprint'i

**Durum:** Devam ediyor (2026-07-23, 2. oturum — **iş bilgisayarı, GPU yok**) — ölçüm + dayanıklılık katmanı **+ CPU-güvenli hız/altyapı işleri** bitti; DL/GPU işleri (Epic 3, 2.2) GPU'lu makineye ertelendi. Sıradaki: **2.1 warm-start plumbing**.
**Branch:** `fix/dl-models-ensemble-integration`
**Önceki iş:** `perf: speed up dashboard...` (frontend/serving perf) + `devops: dockerize...` (single-VPS serving)
**Bu fazın odağı:** **Backend** — veri işleme + model eğitimi (frontend/serving değil)
**İkili hedef:** (1) **Hız** — eğitim throughput'u ↑; (2) **Dayanıklılık** — "sorunsuz eğitim süreci": sessiz hata yok, gözlemlenebilir, tekrar üretilebilir. Kullanıcı ikisini de kritik olarak işaretledi (2026-07-22).
**Hedef ortam:** Eğitim geliştirici makinesinde (CUDA GPU, ör. RTX 4060 8GB) çalışır; Docker/VPS yalnızca serving. Bu plandaki paralellik/GPU kararları bu varsayıma dayanır.

> **Bu plan bizi "sorunsuz eğitim"e geçirir mi?** Hız epic'leri (0-3) eğitimi *hızlandırır* ama tek başına *güvenilir* kılmaz — kod tabanında sessiz yutulan hatalar, yanlış fallback verisi ve gözlemlenebilirlik eksikliği var (aşağıda R1-R5, kod-kanıtlı). Bu yüzden **Epic G (Güvenilirlik)** eklendi. "Sorunsuz eğitim" ≙ Epic 0-3 (hız) **+** Epic G (dayanıklılık) birlikte.

---

## ⏱️ Uygulama Durumu (son güncelleme 2026-07-23)

> Bu bölüm sprintin **canlı ilerlemesini** izler. Bir sonraki oturum buradan devam eder.
> 1. oturum: 2026-07-22 (GPU'lu geliştirici makinesi). 2. oturum: 2026-07-23 (iş bilgisayarı, GPU yok).

### 🔴 En kritik keşif — ensemble aslında 3 model çalışıyormuş

"Önce çalıştır, sonra ölç" ilkesi uygulanırken (henüz benchmark bile yazılmadan), **ensemble'ın hiçbir zaman 5 modelle eğitilmediği** ortaya çıktı. BiLSTM ve TFT her koşumda `except ... continue` bloğunda sessizce düşürülüp sistem 3 ağaç modeliyle "başarılı" tamamlanıyormuş. Bu, planın **R1 (sessiz model düşürme)** kırılganlığının canlı ve aktif kanıtı. CLAUDE.md "5-model ensemble tamamlandı" diyordu; pratikte değilmiş.

**⚠️ Sonuç:** Bu commit'lerden önce eğitilmiş "5 model" sanılan modeller / geçmiş tahminler / akademik sonuçlar aslında 3 modelliydi. Yeniden eğitim gerekebilir.

### Düzeltilen 5 gizli bug (hepsi DL sessiz düşmesiyle maskeliydi)

| # | Bug | Konum | Commit |
|---|-----|-------|--------|
| 1 | `compute_metrics` IndexError (DL lookback-kısa çıktı) | base.py | `9dcb277` |
| 2 | BiLSTM/TFT `__init__` `source` kwarg reddi (TypeError) | lstm/tft_model.py | `9dcb277` |
| 3 | OOF meta `column_stack` uzunluk uyumsuzluğu (4 yer) | ensemble.py `_stack_aligned` | `9dcb277` |
| 4 | `predict_next` `total_models` tanımsız | ensemble.py | `9dcb277` |
| 5 | TATS XGBoost class-encoding ({0,2} eksik sınıf) | tats.py | `bc31b37` |

### Tamamlanan epic'ler ✅

| Epic | İçerik | Durum |
|------|--------|-------|
| **0** — Ölçüm | `scripts/benchmarking/profile_training.py` + `results/benchmarks/phase6_baseline.md` | ✅ commit `ee7a0f1` |
| **1.1** — Paralel veri çekme | `data_fetcher.py` ThreadPoolExecutor (`max_workers=8`, 1=seri). **Byte-eş doğrulandı** | ✅ commit `ee7a0f1` |
| **G.1** — Sessiz hata görünürlüğü | ensemble `strict`/`degraded` mod; `status`, `missing_models`, `failed_models`. Default `strict=False` | ✅ commit `ee7a0f1` |
| **G.2** — Fallback veri işareti | `macro_fetcher.py` `data_quality` bayrağı (sabit 50.0 / zero-fill), `df.attrs`'e iliştirilir | ✅ commit `ee7a0f1` |
| **G.3** — Eğitim manifesti | `prediction/manifest.py` `TrainingManifest` → `results/training_runs/<run_id>.json` | ✅ commit `ee7a0f1` |
| **Güvenlik ağı** | `tests/test_prediction_regression.py` + `tests/golden/`. **GEÇTİ, deterministik** | ✅ commit `ee7a0f1` |
| **2.3** — Feature-eng fragmentasyon (B4) | Her `_add_*` → dict biriktir + tek `_assign()`. **BIT-EŞ doğrulandı** (daily+weekly+macro+fund+cross+iceemdan); `PerformanceWarning` 106→0; medyan build daily 68→62ms, weekly 76→60ms | ✅ commit `32b49f6` (2. oturum) |
| **G.5** — Determinizm/seed | `prediction/seeding.py` `GLOBAL_SEED` (env `PREDICTION_SEED`, def 42) + `seed_everything()`; hardcode `42` 11 yerden merkeze; trainer başında çağrılır | ✅ commit `1e4a10b` |
| **Config knobs** | `PREDICTION_SEED`, `DATA_FETCH_WORKERS`, `DATA_CACHE_MAXSIZE`, `ENSEMBLE_WARM_START` — hepsi env-override, hardcode yok | ✅ commit `1e4a10b` |
| **1.2** — Cache LRU sınırı (B8) | `DataFetcher._cache` → `OrderedDict` LRU (`DATA_CACHE_MAXSIZE`, 0=sınırsız). Byte-eş; unit test geçti. (Parquet disk-cache bilinçli atlandı — düşük ROI + CSV kontratı) | ✅ commit `e87e3df` |
| **G.3+G.4** — Batch orchestrator + resume | `PredictionService.train_batch()` manifest'i gerçekten kullanır (önce hiçbir yerde kullanılmıyordu) + sembol-seviyesi checkpoint/resume; `scripts/training/train_prediction_batch.py` (`--resume`/`--resume-latest`/`--strict`). Mock'lu test geçti (ok/degraded/failed, resume-skip, strict-stop) | ✅ commit `053cfec` |
| **R2** — CV fold hata görünürlüğü | `WalkForwardTrainer` `strict` + `failed_folds` + `status='degraded'`; G.1'in eksik kalan CV ayağı (önce sadece R1/ensemble yapılmıştı). Service passthrough. | ✅ commit `fda30a0` (3. oturum) |
| **2.4** — Feature-selection cache (B5) | `FeatureSelector(cache_dir=...)` içerik-hash'li disk cache (MI+permutation tekrarını önler); opt-in `FEATURE_SELECTION_CACHE_DIR` (boş=kapalı). HIT==MISS, invalidation doğrulandı | ✅ commit `9578891` |
| **2.1 plumbing** — Warm-start iskeleti (B2) | `base.train(warm_start_from=)` + `_supports_warm_start()`; 5 model (ağaç `xgb_model`/`init_model`, DL `state_dict`); ensemble `warm_start` (config `ENSEMBLE_WARM_START`, %80 turunda %60-modeli kaynak). **Default OFF = bit-eş doğrulandı** (xgb/lgbm/bilstm). Asıl A/B GPU'da | ✅ commit `8203afe` |
| **1.3** — Paralel makro çekme (B1) | `macro_fetcher` yfinance sembol döngüsü ThreadPool (deterministik sıra, log/quality korunur) | ✅ commit `52a1b44` |
| **G.2-strict** — Fallback → hata modu (R3) | `MacroDataFetcher(strict_data=True)` sabit-50/zero-fill fallback'inde patlar (default False = eski davranış + bayrak) | ✅ commit `52a1b44` |
| **Teknik borç** | `catboost_info/` gitignore + untrack | ✅ commit `17711eb`+`98f4a83` |

### 📊 Baseline ölçüm bulguları (RTX 4060, 3 cached sembol)

- **Eğitim = wall-clock'un %99.8'i** — veri pipeline (çekme+temizle+indikatör+feature) sadece %0.2. Darboğaz kesinlikle eğitim, ölçümle kesinleşti.
- **Tüm semboller artık 5 model** ile eğitiliyor (P0 fix gerçek veride doğrulandı).
- **Sembol başına süre 2-4x değişken:** 140s–665s (DL early-stopping farklı epoch'larda tetikleniyor) → warm-start (2.1) + DL ince ayar (3.1) potansiyeli yüksek.
- **Ekstrapolasyon:** ~440s/sembol × 30 ≈ **3.7 saat** tam BIST-30 (seri). Optimize edilecek gerçek sayı bu.

### ⏭️ KALDIĞIMIZ YER — CPU işleri bitti, sıradakiler **GPU makinesinde**

Bu iş bilgisayarında (GPU yok) yapılabilecek/doğrulanabilecek **tüm** CPU-güvenli kod işleri tamamlandı (2. + 3. oturum). Geriye kalan işler doğaları gereği GPU gerektiriyor:

- **2.1 A/B ölçüm** — warm-start plumbing yazıldı ve **default OFF bit-eş doğrulandı**; ama "hız ≥%30 azalır + MAPE ±%2" kabul kriteri DL'in gerçek eğitim süresini gerektirir → GPU. Açmak için: `ENSEMBLE_WARM_START=True` (env) veya `StackingEnsemble(warm_start=True)`, sonra baseline metrik dosyalarıyla A/B.
- **3.1** DL AMP + DataLoader ([lstm_model.py](../../prediction/models/lstm_model.py), [tft_model.py](../../prediction/models/tft_model.py)) — GPU'ya özgü.
- **2.2** semboller arası paralellik (VRAM semaphore) — GPU + `batch_train` üstüne (`train_batch` orchestrator hazır, paralellik eklenecek).
- **3.2** HPO sqlite resume — CPU'da yapılabilir ama HPO nadiren kullanılıyor, P2; istenirse buraya alınabilir.
- **Golden:** `python tests/test_prediction_regression.py --update` **GPU makinesinde** çalıştırılıp golden o donanımda yenilenmeli.

### ✅ 2. + 3. oturumda tamamlananlar (bu iş bilgisayarında, CPU)

CPU'da güvenle yapılıp **kod-doğruluğu** hafif testlerle doğrulanan işler (ağır DL eğitimi/golden GPU makinesine bırakıldı):

**2. oturum (2026-07-23, öğle):**
- **2.3** feature-eng fragmentasyon — bit-eş, `PerformanceWarning` 106→0 (`32b49f6`)
- **G.5** merkezi seed politikası + config knobs (`1e4a10b`)
- **1.2** cache LRU sınırı (`e87e3df`)
- **G.3+G.4** batch orchestrator + resume — manifest ilk kez gerçekten kullanılıyor (`053cfec`)
- teknik borç: `catboost_info/` gitignore (`17711eb`, `98f4a83`)

**3. oturum (2026-07-23, akşam) — "testleri sona sakla, tüm kodu tamamla":**
- **R2** CV fold hata görünürlüğü (`fda30a0`) — G.1'in eksik ayağı
- **2.4** feature-selection disk cache (`9578891`)
- **2.1 plumbing** warm-start iskeleti, default OFF bit-eş (`8203afe`)
- **1.3** paralel makro çekme + **G.2-strict** (`52a1b44`)

> **3. oturum doğrulama notu:** Tüm değişiklikler default OFF/geriye-uyumlu tasarlandı; kritik özellik **"warm-start OFF = bit-eş"** ve **"cache disabled = değişmez"** — ikisi de aynı-seed karşılaştırmasıyla doğrulandı. Ağır bit-eş/mock testleri (import smoke, tree warm OFF==None, DL state_dict crash-yok, R2 degraded, 2.4 HIT==MISS) geçti. Gerçek DL A/B GPU'da.

### 🧹 Sprint kapanışta yapılacak teknik borç

- `catboost_info/` yanlışlıkla git-tracked; her CatBoost eğitiminde kirleniyor → `.gitignore`'a eklenmeli.
- `lightgbm`/`catboost` venv'de eksikti (requirements'ta vardı) → kuruldu; ortam kurulum notu.

### 🖥️ Ortam notları (bu sprint bu geliştirici makinesinde yürütüldü — iş bilgisayarında değişebilir)

- **Bağımlılık:** `lightgbm` ve `catboost` `requirements.txt`'te tanımlı ama bu venv'de kurulu değildi (ortam bug'ı). Yeni bir makinede: `pip install -r requirements.txt` ile tam kurulum yapılmalı, yoksa ensemble yine 3 modele düşer.
- **GPU:** Bu makinede CUDA RTX 4060 (8.6 GB) doğrulandı. DL eğitimleri GPU'da; iş bilgisayarında GPU yoksa BiLSTM/TFT CPU'da çok yavaş olur — Epic 3.1 (AMP) ve 2.2 (paralellik) kararları GPU varlığına göre yeniden değerlendirilmeli.
- **Baseline karşılaştırması:** `phase6_baseline.md` bu makinenin donanımına özgü. İş bilgisayarında optimizasyon ölçmeden önce o makinede yeni bir baseline alınmalı (`python scripts/benchmarking/profile_training.py --stage all --symbols 5 --cached --out results/benchmarks/phase6_baseline_<makine>.md`).
- **Terminal:** Bu makinede cp1254 (unicode ✓/❌ yazdıramıyor) → test/script çıktıları ASCII tutuldu.
- **Uzun eğitim koşumları:** Windows'ta `nohup &` + CUDA arka plan süreçleri güvenilmez (bir baseline koşumu sessizce öldü). Uzun eğitimleri senkron veya harness'in kendi background task mekanizmasıyla çalıştır.

### 🖥️ İş bilgisayarı (2. oturum) — kesin bulgular

- **GPU YOK:** Bu makinede NVIDIA ekran kartı yok → `torch.cuda.is_available() == False`. DL modelleri (BiLSTM/TFT) CPU'da çalışır ve **çok yavaştır**. Sonuç: Epic 3 (AMP/GPU) ve 2.2 (VRAM semaphore paralelliği) burada **anlamlı ölçülemez** → GPU'lu makineye ertelendi.
- **Python yorumlayıcısı:** Sistem `python`'ı (global) yalnızca numpy/pandas/yfinance içeriyor — **proje bağımlılıkları `venv/`'de**. Bu makinede tüm komutları `venv/Scripts/python.exe ...` ile çalıştır (yoksa `ModuleNotFoundError: sklearn`). venv'de: scikit-learn, xgboost, lightgbm, catboost, torch(CPU), optuna, shap, EMD-signal — hepsi kurulu.
- **Golden regresyon testi burada rebaseline EDİLMEDİ:** `test_prediction_regression.py` feature matrisini bit-eş üretiyor (deterministik kısım ✓), ama DL kaynaklı ensemble metrikleri GPU golden'ından sapıyor (beklenen CPU-vs-GPU non-determinizmi). **Golden GPU makinesinin donanımına ait**; bu makinede `--update` ile ezmey**in** — GPU baseline'ı bozulur. Bu oturumdaki tüm feature-eng değişiklikleri bunun yerine **kendi CPU-yerel bit-eş karşılaştırmasıyla** (git HEAD'e karşı) doğrulandı.
- **2.3 doğrulama yöntemi (tekrar için):** pre-refactor `feature_engineer.py`'ı `git show HEAD:...`'dan çekip aynı sentetik veri + macro/fund/cross/iceemdan ile `build_features` çıktısını NaN-farkında bit-eş karşılaştır; `PerformanceWarning`'i `warnings.simplefilter('error')` ile yakala.

### 🔁 Yarın devam etmek için

**Bu iş bilgisayarında (CPU-güvenli işler):**
1. `venv/Scripts/python.exe` kullan (global python değil).
2. Epic **2.1 warm-start plumbing** — kod iskeleti: `warm_start` bayrağını `StackingEnsemble.__init__`'e ekle (config `ENSEMBLE_WARM_START`'tan oku), %80 turunda DL için `state_dict` yükleme + ağaçlar için `xgb_model=`/`init_model=` yolunu hazırla. Davranış default (False) sıfırdan-eğitim; bit-eş kalmalı.
3. 2.4 (feature-sel cache) da CPU-OK.

**GPU'lu makinede (ölçüm + kalite):**
1. `git checkout fix/dl-models-ensemble-integration && pip install -r requirements.txt`
2. `python tests/test_prediction_regression.py --update` — golden'ı O makinede yeniden üret (donanım farkı DL non-determinizmini etkiler).
3. 2.1 A/B ölçümü (hız + MAPE regresyonu), sonra 3.1 (AMP), 2.2 (paralellik).

---

## Context

Son iki merge frontend + serving tarafını hızlandırdı (in-process ASGI, model load cache, dir-mtime cache). Ancak bu iyileştirmeler **inference/serving** yolundaydı. Bu sprint **write-path**'i ele alıyor: veri çekme (`data/`) ve model eğitimi (`prediction/`) — yani günün sonunda "30 hisse × 5 base model × ensemble'ı yeniden eğitmek ne kadar sürüyor" ve "günlük veriyi tazelemek ne kadar sağlam/hızlı" sorularını.

**Yaklaşım:** Ölçüm-öncelikli. Önce baseline profil çıkarılır, sonra en yüksek ROI'li darboğazlar sırayla giderilir. Her iş kaleminin **kabul kriteri ölçülebilir** (X saniye → Y saniye). Optimizasyonlar davranışı (tahmin çıktısını, data-leakage garantilerini, kaydedilen model formatını) **değiştirmemeli** — regresyon testi her epic'in Definition of Done'ı.

**Kapsam dışı:** Frontend/Dash callback'leri (tamamlandı), serving cache'leri (tamamlandı), yeni feature/model ekleme (bu bir perf sprinti, ürün sprinti değil).

---

## Tespit Edilen Darboğazlar (kod kanıtlı)

Sprint planı bu somut bulgulara dayanıyor:

| # | Darboğaz | Konum | Etki |
|---|----------|-------|------|
| B1 | **Seri veri çekme** — semboller `for` döngüsünde tek tek, retry gecikmeleriyle serileşmiş | [data/data_fetcher.py:97-111](../../data/data_fetcher.py#L97-L111) | 30 hisse × ağ RTT; wall-clock'un büyük kısmı boşta bekleme |
| B2 | **Her sembol 2× tam eğitim** — ensemble önce %60 (base+meta), sonra %80 (deployment) ile tüm base modelleri sıfırdan yeniden eğitiyor | [ensemble.py:218-241](../../prediction/models/ensemble.py#L218-L241) + [ensemble.py:324-341](../../prediction/models/ensemble.py#L324-L341) | 5 model × 2 tur; BiLSTM+TFT dahil → eğitim süresinin aslan payı |
| B3 | **Semboller arası paralellik yok** — her sembol seri; GPU ve CPU çekirdekleri aynı anda tam doymuyor | trainer/ensemble orkestrasyonu | 30 sembol × tek-sembol süresi, örtüşme yok |
| B4 | **DataFrame fragmentasyonu** — feature engineering yüzlerce kolonu tek tek `data[col] = ...shift()` ile yazıyor | [feature_engineer.py:160-655](../../prediction/feature_engineer.py#L160-L655) | Kolon-bazlı realloc + `PerformanceWarning`; cache'lenmeyen tekrar hesap |
| B5 | **Feature selection maliyeti** — CV'de her fold + final'de tekrar; MI (`n_neighbors=5`) + korelasyon matrisi pahalı | [feature_selector.py:148](../../prediction/feature_selector.py#L148), [trainer.py:118-127](../../prediction/trainer.py#L118-L127) | Fold sayısı × MI/korelasyon; büyük ölçüde tekrar eden iş |
| B6 | **HPO her trial'da sıfırdan** — `TimeSeriesSplit` × N trial, BiLSTM/TFT dahil, checkpoint/warm-start yok | [hyperopt.py:133-159](../../prediction/hyperopt.py#L133-L159) | En pahalı tek işlem; kullanıldığında saatler |
| B7 | **DL eğitim döngüsü ayarsız** — no AMP (mixed precision), `DataLoader(num_workers=0)`, CPU↔GPU her batch transfer, `torch.set_num_threads` ayarsız | [lstm_model.py:126-239](../../prediction/models/lstm_model.py#L126-L239), [tft_model.py:206-308](../../prediction/models/tft_model.py#L206-L308) | GPU %100 doymuyor; her BiLSTM/TFT eğitimi olması gerekenden yavaş |
| B8 | **Sınırsız in-process cache** — `DataFetcher._cache` class-level dict, sınır/TTL yok; batch eğitimde bellek şişer | [data_fetcher.py:24](../../data/data_fetcher.py#L24) | Uzun batch job'larda RAM baskısı |

---

## Tespit Edilen Güvenilirlik Kırılganlıkları (kod-kanıtlı)

"Sorunsuz eğitim" için hız yetmez — eğitim **doğru** ve **gözlemlenebilir** de olmalı. Kod tabanında bulunanlar:

| # | Kırılganlık | Konum | Neden "sorunsuz değil" |
|---|-------------|-------|------------------------|
| R1 | **Sessiz model düşürme** — bir base model eğitimde patlarsa `except ... continue`; ensemble 5 yerine 2-3 modelle "başarılı" biter | [ensemble.py:238-240](../../prediction/models/ensemble.py#L238-L240), [ensemble.py:340-341](../../prediction/models/ensemble.py#L340-L341) | Eksik ensemble sessizce production'a gider; kalite düşer, kimse fark etmez |
| R2 | **CV fold hataları yutuluyor** — fold patlarsa `logger.error + continue`; özet yine de üretilir | [trainer.py:182-183](../../prediction/trainer.py#L182-L183) | "Başarılı" CV aslında yarım fold'la hesaplanmış olabilir |
| R3 | **Sabit fallback verisi** — politika faizi çekilemezse `pd.Series(50.0)` sabiti kullanılıyor | [macro_fetcher.py:196-202](../../data/macro_fetcher.py#L196-L202) | Eğitim "sorunsuz" biter ama makro feature yanlış → sessiz model bozulması |
| R4 | **Yapısal çıktı / manifest yok** — batch eğitim sonucu sadece log satırları; ne başarılı/başarısız oldu, hangi veri sürümüyle eğitildi bilinmiyor | genel (trainer/ensemble) | Gece batch'inin yarısı başarısız olsa sabah teşhis edilemez |
| R5 | **Checkpoint/resume yok** — uzun batch ortada kesilirse baştan başlar | orkestrasyon | Uzun eğitimlerde kesinti = tam kayıp |

---

## Sprint Yapısı

Beş epic. **Epic 0 önce gelmeli** — ölçüm olmadan diğer epic'lerin kazancı doğrulanamaz. **Epic G (Güvenilirlik)** hız epic'lerine paralel yürür ve "sorunsuz eğitim" iddiasının asıl taşıyıcısıdır.

```
Epic 0 (Ölçüm)  ──►  Epic 1 (Veri I/O)      ──┐
                ──►  Epic 2 (Eğitim mimari) ──┼──►  Epic 3 (DL/GPU ince ayar)
                └──►  (paralel çalışabilir)  ──┘

Epic G (Güvenilirlik & Gözlemlenebilirlik)  ──►  tüm hız epic'lerine paralel;
   R1/R2 (sessiz hata) Epic 2'den ÖNCE gelmeli — paralelleştirmeden önce
   hataların görünür olması şart, yoksa paralel job'da teşhis imkânsızlaşır.
```

> **Sıralama nüansı:** Epic G.1 (sessiz hataları görünür kıl) **Epic 2.2'den (paralellik) önce** yapılmalı. Sebep: process-paralel eğitimde yutulan bir hata, tek-process'e göre çok daha zor teşhis edilir. Önce görünürlük, sonra paralellik.

---

## Epic 0 — Profilleme & Baseline (önkoşul, ~0.5 gün)

Kör optimizasyon yapmamak için. Bu epic biter bitmez elimizde "neye ne kadar süre gidiyor" tablosu olur.

### 0.1 — Uçtan uca eğitim benchmark script'i
- **Yeni dosya:** `scripts/benchmarking/profile_training.py`
- **İçerik:** Tek sembol + N sembol için `WalkForwardTrainer.train_final_models()` süresini ölç; model-tipi bazında breakdown (XGB / LGBM / CatBoost / BiLSTM / TFT / meta / feature-eng / feature-select) logla. `time.perf_counter()` + basit bir context-manager timer.
- **Kabul kriteri:** `python scripts/benchmarking/profile_training.py --symbols 5` çalışır, her aşamanın saniyesini tablo olarak basar.

### 0.2 — Veri pipeline profili
- Aynı script'e `--stage data` modu: `fetch_stock_data`, `macro`, `fundamental`, `clean_data`, `build_features` sürelerini ayrı ölç.

### 0.3 — Baseline dokümanı
- **Çıktı:** `results/benchmarks/phase6_baseline.md` (gitignore'da değilse `docs/` altına özet) — 1/5/30 sembol için wall-clock, model breakdown, GPU kullanımı (`nvidia-smi` snapshot), peak RAM.
- **Neden:** Sprint sonunda "X→Y" iddialarının kanıtı bu.

**Definition of Done:** Baseline sayılar kayıt altında; darboğaz sıralaması (B1–B8) gerçek ölçümle teyit/revize edilmiş.

---

## Epic 1 — Veri Katmanı I/O (B1, B8 · ~1 gün)

En düşük riskli, en görünür kazanç. Saf I/O; model davranışını etkilemez.

### 1.1 — Paralel çok-sembol çekme (B1)
- **Dosya:** [data/data_fetcher.py:66-131](../../data/data_fetcher.py#L66-L131)
- **Yaklaşım:** `for symbol in symbols` döngüsünü `ThreadPoolExecutor` (I/O-bound, GIL salınıyor) ile değiştir; `max_workers` yapılandırılabilir (varsayılan ~8, yfinance rate-limit'e saygılı). Alternatif: tek `yf.download(tickers=..., group_by='ticker', threads=True)` çağrısı — retry mantığı korunmalı.
- **Korunacak:** `_fetch_with_retry` exponential backoff, coverage check (%80), cache key mantığı, log formatı.
- **Kabul kriteri:** 30 sembol çekme baseline'a göre **≥%50 daha hızlı**; çekilen veri byte-eş (aynı satır sayısı, aynı değerler).
- **Risk:** yfinance rate-limit / geçici ban. Azaltma: `max_workers` cap + jitter; başarısız sembolde tek-tek fallback.

### 1.2 — Cache'e sınır + opsiyonel disk-önbellek (B8)
- **Dosya:** [data_fetcher.py:24](../../data/data_fetcher.py#L24)
- **Yaklaşım:** `_cache`'i basit LRU'ya çevir (ör. `functools.lru_cache` wrapper veya `OrderedDict` + `maxsize`); batch eğitimde sınırsız büyümeyi önle. Ek: Parquet disk cache (mtime invalidation) — CSV `read_csv` yerine `read_parquet` I/O'da belirgin hızlı.
- **Kabul kriteri:** 30 sembol × tekrarlı eğitimde peak RAM baseline'dan düşük; cache hit davranışı değişmemiş.

### 1.3 — Makro/fundamental çekmeyi paralelleştir (opsiyonel)
- **Dosya:** [macro_fetcher.py:118-197](../../data/macro_fetcher.py#L118-L197)
- yfinance sembol döngüsü ([macro_fetcher.py:124](../../data/macro_fetcher.py#L124)) aynı ThreadPool desenine geçebilir. EVDS çağrıları seri kalabilir (tek endpoint).

**Definition of Done:** Veri çekme ölçülebilir hızlanmış, çıktı byte-eş, RAM sınırlı.

---

## Epic 2 — Eğitim Mimarisi (B2, B3, B4, B5 · ~2-3 gün)

Wall-clock'un asıl kaynağı. Burada **davranış korunmalı** — data-leakage garantileri (3-way 60/20/20 split, OOF, purge/embargo) ve tahmin çıktısı değişmemeli.

### 2.1 — Çift eğitimi akıllı hale getir (B2) — **en yüksek ROI**
- **Dosya:** [ensemble.py:310-350](../../prediction/models/ensemble.py#L310-L350)
- **Sorun:** Base modeller önce %60 ile (test metriği için), sonra %80 ile (deployment için) **sıfırdan** eğitiliyor. Ağaç modelleri (XGB/LGBM/CatBoost) için ikinci tur ucuz ama BiLSTM/TFT için tam bir eğitim daha.
- **Yaklaşım seçenekleri (ölçüp karar ver):**
  - **(a) Warm-start:** İkinci tur (%80) modeli, birinci turun ağırlıklarından başlasın (DL için `state_dict` yükle + az epoch fine-tune; ağaçlar için `xgb.train(xgb_model=...)` / LightGBM `init_model`). Distribution-shift gerekçesi ([ensemble.py:311-316](../../prediction/models/ensemble.py#L311-L316) yorumu) korunur ama sıfırdan değil.
  - **(b) Koşullu ikinci tur:** Stationary varlıklarda (hisse, `target_type='log_return'`) ikinci tur atlanabilir mi? Yorumda gerekçe "non-stationary altın/döviz" — hisse için %80 yeniden eğitim gereksiz olabilir. Ölçümle doğrula.
- **Kabul kriteri:** Ensemble eğitim süresi **≥%30 azalır**; `ensemble_test_metrics` (MAPE, dir-acc) baseline ±%2 içinde; kaydedilen model formatı aynı.
- **Risk:** Warm-start kalite düşürebilir. Azaltma: A/B — baseline metrik dosyalarıyla karşılaştır, regresyon eşiği aşılırsa (a)→sıfırdan'a geri dön.

### 2.2 — Semboller arası paralellik (B3)
- **Dosya:** trainer orkestrasyonu ([trainer.py](../../prediction/trainer.py)) + batch training entry-point
- **Yaklaşım:** Sembol-başına eğitim bağımsız → `ProcessPoolExecutor` ile paralelleştir. **Kritik nüans:** GPU tek; DL modelleri GPU'da paralel çalışırsa VRAM (8GB) taşar. Bu yüzden:
  - Ağaç modelleri (XGB/LGBM/CatBoost, CPU) → çok-process paralel, sembol başına 1 process.
  - DL modelleri (BiLSTM/TFT, GPU) → GPU semaphore (aynı anda 1-2 sembolün DL adımı), sıra bazlı.
  - Basit ilk adım: "ağaç modelleri paralel, DL seri" ayrımı. İleri: CUDA MPS / stream.
- **Kabul kriteri:** 5 sembol batch eğitimi baseline'a göre **≥%40 hızlanır**; GPU OOM yok; her modelin metriği tek-sembol koşumuyla eş.
- **Risk:** Process çoğaltma overhead'i (veri pickle, model import). Küçük sembol setinde kazanç negatif olabilir → `--parallel` opt-in flag, eşik altında otomatik seri.

### 2.3 — Feature engineering fragmentasyonu (B4)
- **Dosya:** [feature_engineer.py:160-655](../../prediction/feature_engineer.py#L160-L655)
- **Yaklaşım:** Her `_add_*` metodunda kolonları tek tek `data[col]=` yerine bir dict'te biriktirip **tek `pd.concat`** ile ekle (fragmentasyonu ve `PerformanceWarning`'i bitirir). Rolling hesaplarını mümkün olduğunca vektörize et / tekrar edenleri paylaş (ör. `close.rolling(20)` birden fazla yerde hesaplanıyor → bir kez).
- **Kabul kriteri:** `build_features` tek sembolde **ölçülebilir hızlanır** (hedef ≥%20); çıktı DataFrame değer-eş (kolon isimleri + değerler bit-bit aynı), `PerformanceWarning` kaybolur.
- **Risk:** Yüksek — sayısal eş-değerlik şart. Azaltma: baseline feature matrisini pickle'la, refactor sonrası `assert_frame_equal`.

### 2.4 — Feature selection'ı tek sefere indir (B5)
- **Dosya:** [trainer.py:118-127](../../prediction/trainer.py#L118-L127), [feature_selector.py](../../prediction/feature_selector.py)
- **Sorun:** CV'nin her fold'unda feature selection tekrar edebiliyor; ayrıca final eğitimde bir daha. MI hesabı (`mutual_info_regression`) pahalı.
- **Yaklaşım:** Seçilen feature seti sembol+config bazında **cache**'lensin (disk, feature-eng çıktısının hash'ine bağlı invalidation). CV içinde her fold'da yeniden seçim yerine, ilk (expanding) fold'da seç → sonrakilerde yeniden kullan (leakage açısından güvenli çünkü sadece feature *seçimi*, model eğitimi hâlâ fold-içi).
- **Kabul kriteri:** Feature-selection toplam süresi baseline'a göre **belirgin azalır**; seçilen feature listesi determin**istik** (aynı seed → aynı liste).

**Definition of Done:** Regresyon testi (`tests/test_prediction_regression.py`, yeni) geçer — sabit sembol+seed için tahmin çıktısı ve test metrikleri baseline ile eşleşir; wall-clock ölçülü şekilde düşmüş.

---

## Epic 3 — DL / GPU İnce Ayar (B6, B7 · ~1-2 gün)

GPU'lu dev makinede BiLSTM/TFT eğitim döngüsünü doyurmak.

### 3.1 — Karışık hassasiyet (AMP) + DataLoader ayarı (B7)
- **Dosya:** [lstm_model.py:193-229](../../prediction/models/lstm_model.py#L193-L229), [tft_model.py:269+](../../prediction/models/tft_model.py#L269)
- **Yaklaşım:**
  - `torch.cuda.amp.autocast` + `GradScaler` — RTX 4060'ta ileri/geri geçişi hızlandırır, VRAM düşürür (daha büyük batch'e alan).
  - `DataLoader(num_workers>0, pin_memory=True)` — CPU↔GPU transfer örtüşür. (Windows'ta `num_workers` spawn maliyeti var → ölçüp ayarla, gerekirse `persistent_workers=True`.)
  - Tüm sekansı GPU'ya bir kez taşı (dataset küçük, tek sembol tarihsel veri ~birkaç bin satır) → per-batch `.to(device)` yerine.
  - CPU tarafı için `torch.set_num_threads` makul değere sabitle (process paralelliğiyle çakışmasın).
- **Kabul kriteri:** Tek BiLSTM eğitimi baseline'a göre **≥%25 hızlanır**; val MAPE ±%2; sayısal kararlılık korunur (AMP overflow yok).
- **Risk:** AMP bazı katmanlarda instabilite. Azaltma: sadece forward'da autocast, loss FP32; regresyon eşiği.

### 3.2 — HPO warm-start & bütçe (B6)
- **Dosya:** [hyperopt.py:76-159](../../prediction/hyperopt.py#L76-L159)
- **Yaklaşım:**
  - Optuna study'yi **kalıcı** yap (`storage=sqlite:///...`) → yarıda kesilen HPO devam etsin, tekrar sıfırdan başlamasın.
  - DL modelleri için trial içi epoch'u kıs (proxy/low-fidelity) + `MedianPruner` zaten var, `n_warmup_steps` ayarını gözden geçir.
  - Ağaç modellerinde `xgboost`/`lightgbm` native CV veya erken durdurma ile trial başına maliyet düşür.
  - `optimize_all_models` ([hyperopt.py:161](../../prediction/hyperopt.py#L161)) model tiplerini paralel çalıştırabilir (ağaç HPO'ları CPU, birbirinden bağımsız).
- **Kabul kriteri:** Aynı `n_trials` için HPO wall-clock **belirgin düşer**; kesinti sonrası resume çalışır; bulunan `best_params` kalitesi düşmez.

**Definition of Done:** DL eğitim ve HPO ölçülü hızlanmış; kalite regresyonu yok; GPU kullanımı baseline'dan yüksek (`nvidia-smi` ile teyit).

---

## Epic G — Güvenilirlik & Gözlemlenebilirlik (R1-R5 · ~1.5-2 gün)

**"Sorunsuz eğitim süreci"nin asıl taşıyıcısı.** Hız epic'leri eğitimi hızlandırır; bu epic onu *güvenilir* ve *teşhis edilebilir* kılar. Kısmen Epic 2'den önce gelir (aşağıya bak).

### G.1 — Sessiz hataları görünür kıl (R1, R2) — **Epic 2.2'den ÖNCE**
- **Dosya:** [ensemble.py:238-240](../../prediction/models/ensemble.py#L238-L240), [ensemble.py:340-341](../../prediction/models/ensemble.py#L340-L341), [trainer.py:182-183](../../prediction/trainer.py#L182-L183)
- **Yaklaşım:** `except ... continue` desenlerini **strict/lenient mod**'a bağla. Varsayılan `strict=True`: bir base model veya CV fold patlarsa eğitim **fail-fast** eder (veya en azından sonucu `status='degraded'` + `failed_models=[...]` ile işaretler). Lenient mod opt-in kalsın (örn. deneysel koşumlar için).
- **Kritik ayrım:** "1 model opsiyonel düştü" ≠ "ensemble başarılı". Ensemble sonucu, kaç modelle ve hangileriyle eğitildiğini **açıkça** taşımalı; eksik model varsa uyarı seviyesi log değil, sonuç nesnesinde alan.
- **Kabul kriteri:** Kasıtlı bir model hatası enjekte edildiğinde (test) eğitim ya durur ya da `status='degraded'` döner — asla sessizce "başarılı" demez.

### G.2 — Fallback verisini işaretle (R3)
- **Dosya:** [macro_fetcher.py:196-202](../../data/macro_fetcher.py#L196-L202)
- **Yaklaşım:** Sabit `50.0` fallback kullanıldığında bunu **veri kalite bayrağı** olarak yay (ör. dönen DataFrame'e `_is_fallback` meta / ayrı bir `data_quality` dict). Eğitim manifesti (G.3) bu bayrağı kaydetsin. Opsiyonel: `strict_data=True` modunda fallback → hata (production eğitimi yanlış veriyle çalışmasın).
- **Kabul kriteri:** Fallback devreye girdiğinde eğitim manifestinde `data_quality.policy_rate='fallback'` görünür; kullanıcı yanlış veriyle eğittiğini fark edebilir.

### G.3 — Eğitim manifesti / run kaydı (R4) — **gözlemlenebilirliğin kalbi**
- **Yeni:** Her eğitim koşumu için yapılandırılmış manifest (`results/training_runs/<run_id>.json`).
- **İçerik:** run_id, başlangıç/bitiş, sembol listesi, her sembol için `status` (ok/degraded/failed), eğitilen modeller, veri sürümü (son tarih + satır sayısı + `data_quality` bayrakları), feature config, wall-clock breakdown (Epic 0 timer'ından), git commit hash, GPU/host bilgisi.
- **Neden:** Gece batch'i sabah tek bakışta teşhis edilir: "3 sembol degraded, GARAN'da TFT patladı, makro fallback'teydi." Şu an bu bilgi sadece dağınık log satırlarında.
- **Kabul kriteri:** N-sembol batch sonrası tek JSON manifest üretilir; başarısız/degraded semboller ayırt edilebilir. (İsteğe bağlı: dashboard'da basit bir "son eğitim koşumları" görünümü — ama o Faz 7.)

### G.4 — Checkpoint & resume (R5)
- **Yaklaşım:** Batch eğitim orkestrasyonu, tamamlanan sembolleri manifest'e işaretlesin; `--resume` ile yarıda kalan koşum, biten sembolleri atlayıp kalanlardan devam etsin. (Optuna resume'u Epic 3.2 zaten ele alıyor — bu, sembol-seviyesi resume.)
- **Kabul kriteri:** 5 sembollük batch 3. sembolde kill edilir → `--resume` ile 4-5'ten devam eder, 1-3'ü yeniden eğitmez.

### G.5 — Determinizm & seed kontrolü
- **Yaklaşım:** Global seed politikası (numpy, torch, xgboost/lightgbm/catboost, optuna). Paralelleştirme (Epic 2.2) float toplama sırasını değiştirip non-determinizm getirebilir → belgele ve regresyon testinde tolerans eşiği tanımla.
- **Kabul kriteri:** Aynı seed + seri modda iki koşum bit-eş; paralel modda tanımlı tolerans içinde.

**Definition of Done:** Kasıtlı hata sessizce yutulmuyor; her koşum manifest üretiyor; fallback veri işaretli; batch resume çalışıyor. **Bu epic olmadan "sorunsuz eğitim" iddiası eksiktir.**

---

## Çapraz Kesen İşler

- **Regresyon test altyapısı** (Epic 0'da başlar, hepsinde kullanılır): `tests/test_prediction_regression.py` — sabit sembol+seed ile feature matrisi, tahmin çıktısı ve ensemble test metriklerini altın-dosyaya (golden file) karşı doğrular. Her epic'in DoD'u buna bağlı.
- **Determinizm:** Optimizasyonlar seed davranışını bozmamalı; paralellik non-determinizm getirirse (float toplama sırası) tolerans eşiği tanımla.
- **Config:** Yeni ayarlar (`max_workers`, `--parallel`, AMP on/off, HPO storage) [app/core/config.py](../../app/core/config.py) üzerinden okunabilir olsun; kod içinde hardcode değil.

---

## Öncelik & ROI Matrisi

| İş | Epic | Tahmini kazanç | Risk | Efor | Öncelik |
|----|------|----------------|------|------|---------|
| Profilleme baseline | 0 | (kanıt tabanı) | Düşük | S | **P0 — önkoşul** |
| **Sessiz hataları görünür kıl** | **G.1** | **Güvenilirlik (kritik)** | Düşük | S | **P0 — 2.2 öncesi** |
| Paralel veri çekme | 1.1 | Yüksek (I/O) | Düşük | S | **P0** |
| Çift eğitim → warm-start | 2.1 | **Çok yüksek** | Orta | M | **P0** |
| Eğitim manifesti | G.3 | Gözlemlenebilirlik (kritik) | Düşük | M | **P0** |
| Semboller arası paralellik | 2.2 | Çok yüksek | Orta-Yüksek | L | **P1** |
| DL AMP + DataLoader | 3.1 | Yüksek | Orta | M | **P1** |
| Feature-eng fragmentasyon | 2.3 | Orta | Yüksek (eşdeğerlik) | M | **P1** |
| Fallback veri işareti | G.2 | Güvenilirlik | Düşük | S | **P1** |
| Checkpoint & resume | G.4 | Dayanıklılık (uzun batch) | Orta | M | **P1** |
| Feature-selection cache | 2.4 | Orta | Orta | S | **P2** |
| HPO warm-start/resume | 3.2 | Yüksek (HPO kullanılırsa) | Orta | M | **P2** |
| Cache sınırı + Parquet | 1.2 | Düşük-Orta (RAM+I/O) | Düşük | S | **P2** |
| Determinizm & seed | G.5 | Tekrar üretilebilirlik | Düşük | S | **P2** |

**Önerilen koşum sırası:** Epic 0 → **G.1** → 1.1 → 2.1 → **G.3** → (2.2 ‖ 3.1) → G.2/G.4 → 2.3 → geri kalan P2'ler.

**Gerçekleşen (bugüne dek):** ✅ 0, 1.1, G.1, G.2, G.3, güvenlik-ağı (1. oturum) → ✅ 2.3, G.5, config, 1.2, G.4 (2. oturum, CPU) → ✅ R2, 2.4, 2.1-plumbing, 1.3, G.2-strict (3. oturum, CPU) → ⏭️ **GPU makinesi:** 2.1 A/B, 3.1, 2.2, 3.2 + golden `--update`.

> **Hız mı, güvenilirlik mi önce?** İkisi iç içe. G.1 (sessiz hata görünürlüğü) baseline'dan hemen sonra gelir çünkü hem paralelleştirmenin (2.2) ön koşulu hem de "sorunsuz"un temeli. Manifest (G.3) ilk hız kazanımlarıyla birlikte devreye girer ki her iyileştirmenin etkisi kayıt altına alınsın.

---

## Riskler & İlkeler

1. **"Önce ölç" ilkesi:** Hiçbir optimizasyon Epic 0 baseline'ı olmadan merge edilmez. Her PR "X s → Y s" kanıtı taşır.
2. **Davranış dondurma:** Data-leakage garantileri (purge/embargo/OOF/3-way split), tahmin çıktısı ve kaydedilen model formatı **değişmez**. Regresyon testi zorunlu geçiş şartı.
3. **GPU tekilliği:** VRAM 8GB — DL paralelliğinde OOM birincil risk. Semaphore/seri fallback ile korunur.
4. **Windows nüansları:** `ProcessPoolExecutor` ve `DataLoader(num_workers>0)` Windows'ta spawn maliyetli — küçük iş yükünde otomatik seri fallback eşiği koy.
5. **Geriye uyumluluk:** Mevcut eğitilmiş modeller ve `models/prediction/*_ensemble_meta.json` formatı bozulmaz (serving cache'leri bunlara bağlı).

---

## Definition of Done (Sprint)

**Hız:**
- [x] Epic 0 baseline dokümante, darboğaz sırası ölçümle teyit
- [~] `tests/test_prediction_regression.py` yeşil — GPU makinesinde golden'a karşı (bu iş bilgisayarında feature-eş ✓, ensemble metrikleri GPU golden'ından bekleniyor sapıyor; golden GPU'da)
- [ ] Uçtan uca N-sembol eğitim wall-clock'u baseline'a göre ölçülebilir düşük (hedef: birleşik ≥%40) — **GPU makinesinde ölçülecek**
- [x] Veri çekme ≥%50 hızlı, çıktı byte-eş (1.1)
- [ ] GPU kullanımı DL eğitiminde artmış (`nvidia-smi` teyit), val metrikleri regresyonsuz — **GPU makinesinde (Epic 3.1)**
- [x] Feature-eng fragmentasyon: `PerformanceWarning` 0, çıktı bit-eş (2.3)

**Dayanıklılık ("sorunsuz eğitim"):**
- [x] Kasıtlı model/fold hatası sessizce yutulmuyor — eğitim fail-fast eder veya `status='degraded'` döner (G.1 ensemble/R1 + R2 CV fold: `WalkForwardTrainer` `failed_folds`+`status`)
- [x] Her batch koşumu yapılandırılmış manifest üretiyor; degraded/failed semboller ayırt edilebilir (G.3 + batch orchestrator)
- [x] Fallback makro verisi kullanıldığında manifestte işaretli (G.2)
- [x] `--resume` yarıda kalan batch'i biten sembolleri atlayarak sürdürüyor (G.4)
- [x] Aynı seed + seri mod → tekrar üretilebilir çıktı (G.5 — merkezi seed + `seed_everything()` trainer başında)

**Genel:**
- [x] Yeni ayarlar config üzerinden, hardcode yok (`PREDICTION_SEED`, `DATA_FETCH_WORKERS`, `DATA_CACHE_MAXSIZE`, `ENSEMBLE_WARM_START`)
- [ ] `docs/development/roadmap.md` ve `docs/README.md` güncellenmiş — sprint kapanışında

---

## Sonuç: "Bu plan bizi sorunsuz eğitime geçirir mi?"

**Evet — ama iki koşulla.** (1) Hız epic'leri (0-3) eğitimi hızlandırır; **(2) Epic G olmadan "sorunsuz" iddiası eksiktir.** Hızlı ama sessizce yarım-eğitilmiş bir ensemble üreten sistem "sorunsuz" değildir — sadece hızlı yanlıştır. Bu plan ikisini birlikte ele aldığı için, tamamlandığında hem **hızlı** hem **güvenilir/gözlemlenebilir** bir eğitim sürecine geçilir. Kalan büyük dayanıklılık işleri (dağıtık eğitim, otomatik retrain tetikleyicileri, veri drift tespiti) bilinçli olarak sonraki fazlara bırakıldı.

---

## 🎯 GPU Makinesi — Test & Doğrulama Planı (merge sonrası)

> **Bu bölüm GPU'lu makineye (CUDA, ör. RTX 4060) geçen kişi içindir.** CPU-güvenli tüm kod işleri tamamlandı ve `main`'e **merge edildi** (PR #1). Burada, GPU gerektiren işler + doğrulamalar sırayla, çalıştırılabilir komutlar ve kabul kriterleriyle. Komutları Windows'ta `venv/Scripts/python.exe ...`, Linux'ta (CUDA'lı venv) `python ...` ile çalıştır.
>
> **Genel kural (davranış dondurma):** T3–T7 arasındaki **her kod değişikliğinden sonra** T1 golden'ı (`python tests/test_prediction_regression.py`) yeşil kalmalı.

### Adım 0 — Ortam & sağlık kontrolü (önkoşul, ~5 dk)

```bash
git checkout main && git pull                 # merge edilmiş güncel main
pip install -r requirements.txt               # lightgbm/catboost/torch(CUDA) dahil TAM kurulum
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Kabul:** `CUDA: True <GPU adı>`. `False` ise CUDA'lı torch kurulu değildir — bu plan çalışmaz.

**⚠️ Kritik:** `pip install -r requirements.txt` şart — lightgbm/catboost eksikse ensemble sessizce 3 modele düşer (R1). T1'in çıktısında `models_trained` 5 model listelemiyorsa önce bunu düzelt.

### T1 — Golden'ı bu donanımda yeniden üret — **her şeyden önce**

Golden dosyası başka donanıma ait. DL non-determinizmi donanıma bağlı olduğu için **bu makinede yeniden üretilmeli**, yoksa sonraki tüm regresyon kontrolleri yanlış "başarısız" verir.

```bash
python tests/test_prediction_regression.py --update      # golden'ı BU makinede üret
git add tests/golden/prediction_regression.json
git commit -m "test(phase6): rebaseline regression golden on GPU hardware"
python tests/test_prediction_regression.py               # --update olmadan: GEÇMELİ
```

**Kabul:** İkinci koşum `[REGRESYON] GECTI (OK)`; `models_trained` = 5 model (bilstm, catboost, lightgbm, tft, xgboost).

### T2 — Yeni baseline profili — "önce ölç"

Tüm "X→Y" iddialarının kanıt tabanı; bu donanıma özgü:

```bash
python scripts/benchmarking/profile_training.py --stage all --symbols 5 \
    --out results/benchmarks/phase6_baseline_gpu.md
```

**Kabul:** `phase6_baseline_gpu.md` üretildi (aşama süreleri + model breakdown + `nvidia-smi` snapshot + peak RAM). Bu dosyayı sakla — T3/T4/T5 kazançları buna karşı ölçülür.

### T3 — 2.1 Warm-start A/B — **en yüksek ROI, kalite riski var**

Plumbing hazır, default OFF (bit-eş doğrulandı). Burada gerçek DL eğitim süresiyle hız + kalite A/B.

```bash
# A) BASELINE (warm-start OFF)
python scripts/benchmarking/profile_training.py --stage train --symbols 5 \
    --out results/benchmarks/warmstart_OFF.md

# B) WARM-START ON
#   PowerShell:  $env:ENSEMBLE_WARM_START="true"
#   bash:        export ENSEMBLE_WARM_START=true
python scripts/benchmarking/profile_training.py --stage train --symbols 5 \
    --out results/benchmarks/warmstart_ON.md
#   (sonra kapat: Remove-Item Env:ENSEMBLE_WARM_START  /  unset ENSEMBLE_WARM_START)
```

**Kabul:** hız **≥%30 azalır** (ON vs OFF); `ensemble_test_metrics` (MAPE, dir-acc) baseline'ın **±%2** içinde; kaydedilen model formatı (`*_ensemble_meta.json`) değişmez.
**Karar:** Hız net **ve** kalite regresyonsuzsa → `ENSEMBLE_WARM_START=true` production default (.env/config). Aksi halde OFF kalır (kod güvenli default'ta). Sonucu bu dokümana "T3 sonucu" olarak yaz.

### T4 — 3.1 DL AMP + DataLoader (kod + ölçüm)

Kod henüz yazılmadı (GPU'ya özgü). Hedef: [lstm_model.py](../../prediction/models/lstm_model.py), [tft_model.py](../../prediction/models/tft_model.py).
- `torch.cuda.amp.autocast` + `GradScaler` (forward'da autocast, loss FP32).
- `DataLoader(num_workers>0, pin_memory=True)` — Windows'ta spawn maliyetini ölç, gerekiyorsa `persistent_workers=True`.
- Tüm sekansı GPU'ya bir kez taşı (per-batch `.to(device)` yerine).

**Kabul:** tek BiLSTM eğitimi **≥%25 hızlı**; val MAPE **±%2**; AMP overflow yok; T1 golden geçer.

### T5 — 2.2 Semboller arası paralellik (kod + ölçüm)

`train_batch` orchestrator hazır ([train_prediction_batch.py](../../scripts/training/train_prediction_batch.py)); paralellik üstüne eklenecek. **Kritik:** GPU tek + VRAM 8GB → DL paralel = OOM. Ağaç modelleri (CPU) çok-process paralel; DL (GPU) semaphore ile 1-2 sembol.

```bash
# Paralellik eklendikten sonra: küçük batch ile OOM + eşdeğerlik + resume kontrolü
python scripts/training/train_prediction_batch.py --symbols GARAN.IS AKBNK.IS THYAO.IS
# ortada Ctrl+C, sonra:
python scripts/training/train_prediction_batch.py --symbols GARAN.IS AKBNK.IS THYAO.IS --resume-latest
```

**Kabul:** 5 sembol batch **≥%40 hızlı**; **GPU OOM yok**; her modelin metriği tek-sembol koşumuyla eş; `--resume` biten sembolleri atlıyor.

### T6 — (Opsiyonel) 3.2 HPO sqlite resume

HPO nadiren kullanılıyor (P2). Optuna study'yi `storage=sqlite:///...` ile kalıcı yap → kesinti sonrası devam. **Kabul:** aynı `n_trials` için wall-clock belirgin düşer; resume çalışır; `best_params` kalitesi düşmez.

### T7 — Tam BIST-30 doğrulama koşumu — kapanış

```bash
python scripts/training/train_prediction_batch.py           # tüm BIST-30
# çıktı: results/training_runs/<run_id>.json — tek bakışta ok/degraded/failed
```

**Kabul:** manifest'te tüm semboller `status='ok'` (5 model); `degraded`/`failed` varsa `missing_models`/`error` ile teşhis; toplam wall-clock baseline'a göre **≥%40** düşük (birleşik hedef). Bu koşum "sorunsuz eğitim" iddiasının canlı kanıtıdır.

### 📋 GPU test özet tablosu

| # | Test | Komut / dosya | Kabul kriteri | Ön koşul |
|---|------|---------------|---------------|----------|
| T1 | Golden rebaseline | `test_prediction_regression.py --update` | 2. koşum GEÇER, 5 model | Adım 0 |
| T2 | Yeni baseline | `profile_training.py --stage all --symbols 5` | baseline md üretildi | T1 |
| T3 | 2.1 warm-start A/B | `ENSEMBLE_WARM_START` OFF vs ON | hız ≥%30↓, MAPE ±%2 | T2 |
| T4 | 3.1 DL AMP | (kod + ölçüm) | BiLSTM ≥%25↓, MAPE ±%2, golden geçer | T1 |
| T5 | 2.2 paralellik | `train_prediction_batch.py` (5 sembol) | ≥%40↓, OOM yok, metrik eş, resume | T1 |
| T6 | 3.2 HPO resume (ops) | Optuna sqlite storage | wall-clock↓, resume çalışır | — |
| T7 | Tam BIST-30 | `train_prediction_batch.py` | manifest hepsi ok, ≥%40↓ | T1–T5 |

**Kapsam dışı (sonraki fazlar):** dağıtık eğitim, otomatik retrain tetikleyicileri, veri drift tespiti.
