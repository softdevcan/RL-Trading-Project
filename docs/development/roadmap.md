# Geliştirme Yol Haritası

**Proje:** BIST-30 DRL Trading System
**Ana Referans (Faz 1-3):** Ansari et al. (2024) — *"A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning"*
**Tez Vizyonu:** Çoklu-ajan RL + İletişim + Tahmin entegrasyonu (detay için: [thesis/vision-and-scope.md](../thesis/vision-and-scope.md))

Bu belge projenin **geçmişini (tamamlanan fazlar)** ve **gelecek yol haritasını (tez milestone'ları)** özetler. Detaylı durum/mimari için:
- Tahmin sistemi: [prediction-system.md](prediction-system.md)
- Faz 3 uygulama detayları: [phase3-implementation.md](phase3-implementation.md)
- Tez vizyon ve kapsam: [../thesis/vision-and-scope.md](../thesis/vision-and-scope.md)

---

## Güncel Durum (2026-10-06)

Faz 1-3 + Faz 6 (backend perf/dayanıklılık) + Faz 7 (auth/multi-user) + Faz 8 (UI/UX) tamamlandı. Tek-ajanlı temel altyapı, ensemble tahmin sistemi, risk yönetimi, hızlandırılmış/gözlemlenebilir eğitim ve kullanıcı bazlı izolasyon çalışır durumda. Sonraki adım **tez için çoklu-ajan mimariye** geçiş (Milestone 0-4).

**2026-10-06 — literatür doğrulaması:** Ekim 2026 literatür taraması bağımsız olarak doğrulandı ([../thesis/literature-review-verification.md](../thesis/literature-review-verification.md)) ve tez vizyonu v0.2'ye güncellendi. Doğrulama sırasında kodda **akademik geçerliliği etkileyen açıklar** bulundu (veri sızıntısı + değerlendirme protokolü). Bunlar aşağıda [Bilinen Metodolojik Açıklar](#bilinen-metodolojik-açıklar-planlanmadı) başlığında kayıt altında; **henüz planlanmadı ve kod değişikliği yapılmadı.**

| Katman | Durum | Konum |
|---|---|---|
| Veri pipeline (OHLCV + makro + fundamental + altın/döviz + VIX/US10Y/DXY) | ✅ | `data/` |
| Teknik indikatörler (MACD, RSI, CCI, ADX, Turbulence) | ✅ | `data/technical_indicators.py` |
| RL environment (tek-ajan, PSR reward, ATR/Kelly) | ✅ | `env/` |
| Ensemble tahmin (XGB + LGBM + CatBoost + BiLSTM + TFT + Ridge/XGB meta) | ✅ | `prediction/` |
| ICEEMDAN gürültü filtresi + TATS trend düzeltici | ✅ | `prediction/iceemdan_processor.py`, `prediction/tats.py` |
| SHAP açıklanabilirlik | ✅ | `prediction/explainability.py` |
| FastAPI backend + Dash dashboard (`/dash/` mount) | ✅ | `app/`, `dashboard/` |
| Hiperparametre optimizasyonu (Optuna, RL + Prediction ayrı) | ✅ | `hyperparameter_optimization/`, `prediction/hyperopt.py` |

---

## Tamamlanan Fazlar

### ✅ Faz 1 — Proof of Concept

5 hisse (AKBNK, THYAO, TUPRS, BIMAS, ASELS) üzerinde A2C/PPO/TD3 ile temel RL altyapısı.

- Gymnasium tabanlı multi-stock trading environment (56 feature state space)
- Stable-Baselines3 entegrasyonu (A2C/PPO/TD3)
- 5 teknik indikatör (Ansari et al. seti)
- FastAPI backend + web UI (ilk sürüm)
- Tensorboard logging, metrik hesaplama (Sharpe, Return, Drawdown)

**Ana çıktı:** Çalışır tek-ajan RL baseline + dashboard.

### ✅ Faz 2 — Multifaceted Prediction System

Ansari et al. metodolojisinin tam uygulaması + tahmin katmanı.

- Fundamental veri entegrasyonu (ROE, ROA, P/E, P/B, D/E, profit margin, ...)
- Makro veri pipeline (TCMB EVDS faiz/enflasyon + yfinance döviz/BIST100)
- Altın/döviz pipeline (borsapy + yfinance)
- Feature engineering v2 (10 grup özellik; en az 1 gün gecikmeli). *2026-10-06 düzeltmesi:* "leakage yok" iddiası geri çekildi — 1 günlük gecikme fiyat türevli özellikler için yeterli, ama makro `bfill`, TÜFE/ÜFE açıklanma gecikmesi, statik temel oranlar ve tam seri ICEEMDAN sızıntı yaratıyor (bkz. [Bilinen Metodolojik Açıklar](#bilinen-metodolojik-açıklar-planlanmadı))
- Feature selector (mutual information + permutation importance, 3 aşamalı)
- Multi-model tahmin: XGBoost + LightGBM + CatBoost + BiLSTM (PyTorch/CUDA) + TFT
- Stacking ensemble (Ridge / XGBoost meta-learner)
- Optuna HPO (TPE sampler + median pruner, TimeSeriesSplit CV)
- Walk-forward eğitim (purge gap 5 gün, embargo 3 gün)
- RL entegrasyonu: observation space'e tahmin özellikleri eklendi (+4×N: return/direction/confidence/agreement)
- PSR reward function (Ansari Eq. 1) — `env/reward_functions.py`
- Dash dashboard (8 sayfa, `/dash/` altında mount)

**Ana çıktı:** Tahmin-destekli tek-ajan RL sistemi + tam dashboard.

### ✅ Faz 3 — Production Improvements

Bug fix'ler + tahmin kalitesi + risk yönetimi + açıklanabilirlik.

**3.1 Bug Fixes:**
- PSR reward `total_trades` sayım hatası giderildi
- Meta-learner data leakage 3-way chronological split ile çözüldü (60/20/20, OOF)
- Embargo `prev_test_end` takibi ile her fold'da doğru uygulanıyor
- TFT VSN 50-değişken sınırı kaldırıldı (feature selector zaten 80 ile sınırlar)
- Direction head: BiLSTM/TFT sigmoid çıktısı confidence hesabında kullanılıyor
- Permutation importance feature_selector'a entegre edildi

**3.2 Tahmin Kalitesi:**
- ICEEMDAN gürültü filtreleme (`prediction/iceemdan_processor.py`) — *2026-10-06:* seri bütününe uygulanıyor (sızıntı; opt-in grup)
- TATS trend-adjusted düzeltici (`prediction/tats.py`) — *2026-10-06:* yöntem Kazemdehbashi (2026, arXiv 2601.12706) tarafından yayımlanmış; **atıf zorunlu**, tezde/makalede uyarlanan yöntem olarak sunulur
- Global makro göstergeler: VIX, US10Y, DXY (`macro_fetcher.py` + `feature_engineer.py`)

**3.3 Risk Yönetimi:**
- ATR tabanlı dinamik pozisyon boyutlandırma (`use_atr_sizing=True`)
- Kelly Criterion pozisyon boyutlandırma (`use_kelly=True`, quarter-Kelly)

**3.4 Explainability & Monitoring:**
- SHAP explainability (`prediction/explainability.py`, `/prediction/explain/{symbol}` API)
- Sortino, Calmar, Deflated Sharpe Ratio, Turnover metrikleri

**Ana çıktı:** Faz 2 sisteminin "production-grade" kalite için düzeltilmiş ve zenginleştirilmiş sürümü.

### ✅ Faz 6 — Backend Performans & Eğitim Throughput

Eğitim hızı + "sorunsuz eğitim" (gözlemlenebilir, dayanıklı, tekrar üretilebilir). Detay: [phase-6-backend-performance.md](phase-6-backend-performance.md).

- **🔴 Kritik keşif:** Ensemble hiç 5 modelle eğitilmiyormuş — BiLSTM/TFT her koşumda `except...continue` ile sessizce düşüyor, sistem 3 ağaç modeliyle "başarılı" tamamlanıyordu. Düzeltilince 5 gizli bug açığa çıktı (hepsi DL sessiz düşmesiyle maskeliydi).
- **Ölçüm (Epic 0):** `profile_training.py` baseline. Eğitim = wall-clock'un %99.8'i; TFT tek başına ~%90 (değişken-seçim ağının özellik-başına Python döngüsü darboğaz).
- **Veri I/O (Epic 1):** paralel çok-sembol + makro çekme (ThreadPool), LRU cache sınırı.
- **Eğitim mimarisi (Epic 2):** feature-eng fragmentasyon giderme (bit-eş, PerfWarning 106→0), feature-selection disk cache, warm-start plumbing, **sembol paralelliği** (thread + VRAM semaphore).
- **DL/GPU ince ayar (Epic 3):** GPU-preload batch'leme (BiLSTM +%56), **TFT gruplanmış-VSN 4.7×** (eşdeğerlik kanıtlı), HPO sqlite resume. AMP ölçülüp reddedildi (küçük ağda daha yavaş). *DL perf knob'ları default OFF (opt-in) — davranış dondurma; sıfırdan retrain'de açılır.*
- **Güvenilirlik (Epic G):** sessiz model/fold düşmesi görünür (`status`/`degraded`/`failed_models`), fallback veri işareti + strict mod (cache yolu dahil), **eğitim manifesti** (`results/training_runs/<run_id>.json`), checkpoint/resume, merkezi seed politikası.
- **Kapanış koşumu (T7):** 5 sembol uçtan uca — varsayılan 533.9s → perf knob'ları açık **157.6s (−%70.5)**, 5/5 sembol `ok` (5 model), OOM yok, `--resume` 1.0s. Hedef ≥%40 idi.

**Ana çıktı:** Ölçülebilir daha hızlı + gözlemlenebilir/dayanıklı eğitim; her koşum tek-bakışta teşhis edilebilir manifest üretir. Regresyon golden'ı davranışı donduruyor.

### ✅ Faz 7 — Auth & Multi-User

Çerez tabanlı JWT oturum, bcrypt, roller (admin/user/viewer), admin-only kayıt, denetim kaydı, hibrit kullanıcı izolasyonu (piyasa verisi ortak; model/sonuç/karar/manifest kullanıcı bazlı). Detay: [phase7-auth.md](phase7-auth.md).

### ✅ Faz 8 — UI/UX: Tema, Profil, Üst Çubuk ve Yapıt Yönetimi
- **Token katmanı:** `dbc.themes.DARKLY` → `BOOTSTRAP`; tüm renkler tek kaynakta (`static/tokens.css`). `dashboard/theme.py` sabitleri artık `var(--token)` döndürüyor → ~550 inline stil çağrı yerine dokunmadan temaya duyarlı oldu. Plotly `var()` kabul etmediği için ayrı hex palet (`plot_palette()` + `apply_theme_template()`), tema çerezden çözülüyor.
- **3 durumlu tercih:** aydınlık / koyu / **sistem**. Tarayıcıya değil **hesaba** kayıtlı (`users.theme`); `PATCH /auth/preferences`, yeni `/dash/account` (Hesabım) sayfası ve kenar çubuğunda üç durumu dolaşan hızlı düğme. `<head>`'de senkron script ile FOUC yok.
- **Şema göçü:** alembic yok ve `create_all()` var olan tabloyu değiştirmiyor → `init_db()` içine idempotent additive `ALTER` eklendi; eski şemayla açılan test bunu doğruluyor.
- **Bileşenler:** `PageHeader`, `MetricCard v2` (renk yalnızca yön taşıyan değerde), `TABLE_STYLES`, `StateBlock`; kenar çubuğu üç gruba ayrıldı. (`FilterBar` da yazılmıştı ama hiçbir sayfaya bağlanmadı — Faz H'de silindi.)
- **Kontrast:** her metin tokeni kendi temasının **en kötü** zemininde WCAG AA ≥ 4.5:1 (en düşük 4.58). Mevcut koyu temada AA'yı geçmeyen 4 renk (BLUE 3.98, RED 3.89, PURPLE 3.70, MUTED 4.04) bu arada düzeldi. `tests/test_theme_contrast.py` paleti kalıcı olarak bekçiliyor.
- **Faz F — profil sayfası:** Hesabım sayfası panoda bulunamıyordu (kenar çubuğundaki tek giriş noktası düz metin görünümündeki bir addı). Avatar satırına çevrildi; sayfa gerçek bir profil sayfası oldu: ad soyad düzenleme, son giriş / hesap açılışı / çalışma alanı özeti, **kendi oturumlarını görme ve kapatma**. Yeni uçlar `/api/account/*` (hepsi `CurrentUser`, hedef her zaman oturumdaki kullanıcı). Bu arada kasıtlı oturum iptalinin 30 sn'lik grace penceresiyle atlatılabildiği bulundu ve kapatıldı. Ayrica kullanicinin kendi denetim kaydi (basarisiz giris denemeleri dahil) `GET /api/account/activity` ile sayfada gorunuyor. `tests/test_account_profile.py` 84 kontrol.
- **Faz G — üst çubuk:** kenar çubuğu "nereye gidebilirim"i anlatıyordu; üst çubuk "neredeyim ve buradan ne yapabilirim"i ekledi. Kırıntı (grup › sayfa, başlığı tekrarlamaz), belgelenen akışı izleyen bağlamsal eylem, sayfalar üzerinde role duyarlı komut paleti araması. Görünüm anahtarı kenar çubuğundan **taşındı** (çoğaltılmadı — iki kopya olsaydı biri tıklanınca diğerinin etiketi güncellenmezdi). Zil, bellekteki çalışma durumlarından (RL + tahmin eğitimi) beslenen bir **durum özeti**; kalıcı bildirim tablosu yok, veri tazeliği maliyeti yüzünden bilinçli olarak dışarıda. Yol üstünde `--rlt-` göçünden kalan bir kaçak: marka ikonu tanımsız `var(--primary)` kullanıyordu. `tests/test_topbar.py` 39 kontrol.
- **Görsel doğrulama (F+G):** 9 sayfa × 2 tema, headless Chrome/CDP ile ekran görüntüsü + hesaplanmış stil ölçümü. Yerleşim temiz; üç kusur çıktı ve düzeltildi: bildirim zili `dbc.DropdownMenu`'nün `btn-primary` varyantını alıp dolgulu mavi çıkıyordu, **devre dışı dolgulu düğmeler uygulama genelinde** Bootstrap'in ham `#0d6efd`'sine düşüyordu (`.btn-*:disabled` ezilmemiş), Dashboard'daki boş portföy grafiği `{"history": []}` durumunda mesajsız/eksenli kalıyordu.
- **Faz H — dar ekran + ölü bileşen:** varsayım "dar ekranda düzen bozuluyor" idi, **ölçüm çürüttü** — 1280/1024/820/640'ta hiçbir taşma veya yatay kaydırma yok. Gerçek sorun darlıktı (640px'de menü ekranın %34'ü); ≤820px'de kenar çubuğu 64px ikon rayına iniyor, kullanılabilir alan 640px'de 372 → 528px. Masaüstü değişmedi. İki tur boyunca hiçbir sayfaya bağlanmayan `create_filter_bar` silindi.
- **Faz I — yapıt silme:** eğitilmiş modeller ve optimizasyon çalışmaları panodan silinemiyordu; deneme koşumları ekranı kalıcı kirletiyordu. Model için uç vardı ama **arayüz yoktu**; hiperparametre için silme yeteneği **hiç yoktu** (`DELETE` aslında iptal ediyordu, çalışmayan kayda 404 dönüyordu). `DELETE /hyperopt/studies/{id}` gerçek silmeye çevrildi, iptal `POST .../cancel` oldu — ikisi durum bakımından ayrık olduğu için geçiş veri kaybı doğurmuyor. Faz 7'nin "kimse ortak dizindeki modeli silemez" kuralı iki katmana ayrıldı (`user` silemez, `admin` siler + denetim kaydı): eski kural kullanıcı sisteminden önce eğitilmiş deneme modellerini temizlemenin hiçbir yolunu bırakmıyordu. Yol üstünde iki gedik: `/hyperopt/start` **hiç RBAC taşımıyordu** (viewer optimizasyon başlatabiliyordu — düzeltildi) ve Optuna deposu çalışma alanına göre çözülmüyor (çalışmalar tüm kullanıcılar arasında ortak — belgelendi, **Faz 7 kapsamında ayrı iş**). `tests/test_delete_artifacts.py` 32 kontrol.

- **Canlı test turu (kapanış sonrası):** sistem uçtan uca çalıştırıldı ve beş kusur çıktı — hiçbiri test paketinde görünmüyordu. `/train/status` koşum sonunda 500 veriyordu (ilerleme oranı 1'i aşıyor, ETA de onunla ölüyordu); `dbc.Spinner(className=...)` ilerleme bloğunu hiç çizdirmiyordu; **hiçbir eğitim tamamlanamıyordu** (kronolojik split train'e 5, test'e 30 sembol veriyor → gözlem uzayı uyuşmazlığı); sayfadan çıkıp dönünce ilerleme kayboluyordu; tahmin eğitimi panoda hiç görünmüyordu. Üçü için bekçi eklendi (`test_dash_props`, `test_training_status`).

**Doğrulama:** 347 kontrol (9 paket), 9 sayfa × 2 tema görsel kontrol, container'da canlı teyit. Optuna deposu izolasyonu Faz 7'ye devredildi.

Detay: [phase-8-ui-theming.md](phase-8-ui-theming.md).

---

## Gelecek — Tez Yol Haritası

Tezin ana vizyonu: **tek-ajanlı sistemi çoklu-ajan + iletişim paradigmasına evirmek**, tahmin sistemini bu iletişime entegre etmek. Detaylı tasarım seçenekleri ve akademik konumlama için [thesis/vision-and-scope.md](../thesis/vision-and-scope.md) belgesine bakılmalıdır.

**Tez araştırma sorusu (kısaca):** BIST-30 portföy yönetiminde sektör-bazlı çoklu-ajan RL + tahmin-destekli iletişim, tek-ajanlı alternatiflere göre ne düzeyde üstünlük sağlar?

Milestone haritası:

### Milestone 0 — Baseline Dondurma (0-3 hafta)
Mevcut Faz 1-3 sistemini "kanonik baseline" olarak dondurmak, reprodüktibilite altyapısı kurmak.
- Veri snapshot (parquet), seed + hyperparam kilitleme
- Full BIST-30 (30 hisse) üzerinde baseline eğitimi
- `results/baseline/` versiyonlama
- Ensemble DT vs DTF ablation
- **Çıktı:** Tez Bölüm 4 taslağı + Makale 2 (Ensemble+RL) ham materyali

### Milestone 1 — Literatür + Tez Önerisi (3-5 hafta)
Akademik çerçeve formalize + ilk makale taslağı.
- ✅ Genişletilmiş literatür taraması (60+ makale) — 2026-10-06: 121 kaynak + bağımsız doğrulama; 4 uydurma vizyon kaynağı elendi, boşluk iddiaları daraltıldı ([vision-and-scope.md §2.5](../thesis/vision-and-scope.md))
- ✅ Seçilmiş kaynakça (2026-10-07): 169 kayıt değerlendirildi, 148 seçildi (85 SCIE/SSCI dergi + 41 üst düzey konferans); ileri atıf taraması, Clarivate MJL kontrolü, tez/kod bileşenleri için kapsam matrisi (açık yok) — [../thesis/literature-table.md](../thesis/literature-table.md); vision-and-scope v0.3
- ⏳ Dört 2026 tam metni (Hung & Huang, Yuan vd., CAROC, Xiong vd.) — G1, G2 ve Makale 2 ifadeleri bunlara bağlı
- ⏳ YÖK Ulusal Tez Merkezi elle taraması (G4 iddiası buna bağlı)
- ⚠️ Makale 2 yeniden konumlandırıldı (TATS atfı + sızıntısız ayrıştırma sorusu, vision-and-scope §6.2); gönderim, sızıntı düzeltmelerinin planlanmasına bağlı
- Danışman revizyonu
- Makale 2 ilk submission-ready taslak
- **Çıktı:** Tez Bölüm 1-3 taslağı

### Milestone 2 — MARL Framework (5-11 hafta)
SB3 → PettingZoo + RLlib geçişi + no-comm baseline + attention communication.
- Sektör-bazlı (~8 ajan) çevre tasarımı
- IPPO no-comm baseline
- Attention-based communication (TarMAC-benzeri)
- Tek-ajan vs no-comm vs attention karşılaştırması
- **Çıktı:** Tez Bölüm 5 taslağı + Makale 1 çekirdek deneyler

### Milestone 3 — Prediction-Augmented Communication (11-16 hafta)
Tezin özgün katkısı.
- GNN-based communication (PyTorch Geometric + GAT)
- Prediction confidence → message weight entegrasyonu
- Meta-learner regime signal broadcast
- 4-yol karşılaştırma: no-comm / attention / GNN / prediction-augmented
- Rejim-bazlı ablation (boğa/ayı/yatay)
- Attention ağırlıkları nitel yorumlama
- **Çıktı:** Tez Bölüm 6-8 taslağı + Makale 1 & 3 ham materyali

### Milestone 4 — Yazım, Submit, Savunma (16-20 hafta)
Paketleme.
- Tez tüm bölümler tamamlandı
- Makale 1 submit
- Makale 3 taslak
- Makale 4 (Türkçe, ulusal)
- LaTeX + figure paketi final
- Savunma slaytları + demo
- **Çıktı:** Tez teslim + makale submissionlar

---

<a id="bilinen-metodolojik-açıklar-planlanmadı"></a>

## Bilinen Metodolojik Açıklar

2026-10-06 literatür doğrulaması sırasında kodda doğrulandı. **Bu tablo bulgu kaydıdır.** Kapatılma planı Faz 9 + Faz 10'da ve **2026-11-06'da onaylandı**. Durumu güncel tutulan yer [Faz 9 §9 Uygulama durumu](phase-9-research-validity-plan.md#9-uygulama-durumu). 2026-11-06 itibarıyla:
- **D1 ve D6 kapandı:** RL tohumu kaydediliyor; DSR düzeltildi.
- **D2 kısmen kapandı:** DM-HLN, eşleştirilmiş test, Ledoit-Wolf, MCS ve FDR var; ECE bağlanmadı (F paketi).
- **D3 kısmen kapandı:** tek kaynak modülü TP.APIFON4 serisine bağlandı. Eski sabitler bağlanmadı.
- **S2 ve S3 kapandı (§9.3):** makroda `bfill` kalktı, TÜFE/ÜFE açıklanma tarihine hizalandı. Bu yalnızca 2026-11-06'dan sonra üretilen makro dosyası için geçerli. Ayrıca bulundu: politika faizi, TÜFE ve ÜFE hiçbir zaman EVDS'den gelmemişti (sabit 50 / 0).
- **Diğerleri açık.**
- **Bu kapıdan sonra üretilen RL sonuçları da** Y15/K18 nedeniyle ön sonuçtur (Faz 9 §9.2 #1). Y13 (test ölçeği) B9 ile kapandı ([Faz 9 §9.5](phase-9-research-validity-plan.md#95-b9--valtest-ortamının-fiyat-ölçeği-2026-11-06-dal-featphase9-b9-price-stats)); bedeli eski ortamda fiyat özelliklerinin doygunluğu. Yeni açık Y21: HPO göstergesiz ortamda ayar arıyor (R1). Kanıtlar (dosya:satır) ve literatür dayanakları: [../thesis/literature-review-verification.md §6](../thesis/literature-review-verification.md).

**Sızıntı (sonuç geçerliliği):**

| # | Açık | Varsayılan hatta etkin mi? |
|---|---|---|
| S1 | ICEEMDAN serinin tamamına uygulanıyor; hedef filtrelenmiş fiyattan türetiliyor | Hayır (`iceemdan` opt-in) — Makale 2 bu grubu kullanacak |
| S2 | Makro seride `bfill`: gelecekteki ilk değer geçmişe yayılıyor (yerel önbellekte 4 sütun 2026-04-23 → 2018) | **Evet** |
| S3 | TÜFE/ÜFE açıklanma gecikmesi modellenmiyor | Gizli — yerel önbellekte EVDS verisi fallback sabitlerinde (faiz 50,0; TÜFE/ÜFE 0) |
| S4 | Temel oranlar tek anlık değer olarak tüm geçmişe yayılıyor | Hayır (`fundamental` opt-in) |

**Değerlendirme protokolü (tez/makale iddiaları):**

| # | Açık |
|---|---|
| D1 | RL eğitim rotası SB3 modellerini tohumsuz kuruyor; tohum kaydedilmiyor (tahmin hattı `GLOBAL_SEED=42`) |
| D2 | DM testi HAC'siz, t-testi eşleştirilmemiş; ikisi de hiçbir akıştan çağrılmıyor. Ledoit-Wolf, MCS, FDR yok. ECE fonksiyonu var ama çağrılmıyor |
| D3 | Risksiz faiz üç yerde üç farklı değer (0 / 0,15 / 0,02) |
| D4 | Baseline seti eksik: 1/N (yeniden dengelemeli), min-varyans, HRP yok |
| D5 | Sembol evreni bugünkü BIST-30 (hayatta kalma yanlılığı) |
| D6 | *2026-10-07:* Deflated Sharpe formülü yanlış — makalenin kendi örneğine 0,9004 yerine 0,4031; tek denemede her zaman 1,0 (plan Y1) |
| D7 | *2026-10-07:* Ödüldeki "Differential Sharpe" Moody-Saffell türevi değil, EMA Sharpe düzeyi (plan Y2) |
| D8 | *2026-10-07:* Rejim metriği gerçekleşen getiriye koşullanıyor; güven tahmin edilen yönün olasılığı değil (plan Y3, Y4) |

**Planlama (2026-10-07):** Bu açıkların kapatılma planı ve MARL tasarım kararları → [phase-9-research-validity-plan.md](phase-9-research-validity-plan.md) ve [phase-10-marl-plan.md](phase-10-marl-plan.md) (**onaylandı 2026-11-06**; ilk kapı V + A + C uygulandı). Tam metinler: [../reference/papers/README.md](../reference/papers/README.md).

Hedef protokol: [../thesis/vision-and-scope.md §4.1](../thesis/vision-and-scope.md). Bu açıklardan etkilenen veri/yapılandırmayla üretilmiş sonuçlar tez/makalede ya yeniden üretilmeli ya da sınırlılık olarak beyan edilmeli. tez-final.pdf sonuçlarının hangi makro/temel veri durumuyla üretildiği ayrıca kontrol edilmeli (henüz kontrol edilmedi).

---

## Kapsam Dışı (Tez Bağlamında)

Tezin sınırlarını netleştirmek için **yapılmayacaklar listesi** (detay için vision-and-scope.md §4.2):
- Intraday / high-frequency trading
- Türev ürünler (opsiyon, future)
- Short selling
- Canlı para ile işlem
- Başka borsalar (BIST dışı)
- LLM-tabanlı haber/sentiment entegrasyonu
- Learned discrete communication protocols (DIAL/RIAL) — risk yüksek
- Derin hiyerarşik MARL (feudal RL) — karmaşıklık yüksek

---

## Belge Güncelleme Notu

Bu belge her milestone sonunda revize edilir. Milestone durumları burada güncellenir; tasarım seçenekleri ve kararlar `thesis/vision-and-scope.md` içinde belgelenir.

### Versiyon Geçmişi
- **2026-04-21:** Faz 1-3 durum dökümü + tez milestone haritası eklendi. Önceki (Kasım 2025) taslak "Faz 2 TODO" versiyonu arşivlendi (git history'den erişilebilir).
- **2026-11-06:** Faz 9 ve Faz 10 planları onaylandı; ilk kapı (V + A + C) uygulandı; D1 ve D6 kapandı, D2 ve D3 kısmen kapandı.
- **2026-10-07:** Faz 9 planı (o tarihte onay bekliyordu) ve seçilmiş kaynakça (literature-table.md) eklendi; D6–D8 bulguları kayda geçti.
- **2026-10-06:** Literatür doğrulaması işlendi: Milestone 1 literatür maddesi tamamlandı, Faz 2 "leakage yok" iddiası geri çekildi, TATS atfı eklendi, "Bilinen Metodolojik Açıklar" bölümü eklendi (planlanmadı; kod değişikliği yok).
