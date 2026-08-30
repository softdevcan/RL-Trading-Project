# CLAUDE.md - RL Trading Project

## Project Summary
Deep Reinforcement Learning-based algorithmic trading system for BIST-30 stocks.
Based on Ansari et al. (2024) paper. Phase 1 (POC), Phase 2 (Advanced Prediction System), Phase 3 (Production improvements), Phase 6 (Backend perf & training throughput), Phase 7 (Auth & multi-user), Phase 8 (UI/UX: tema + profil + üst çubuk + yapıt yönetimi) tamamlandı.

## Language
Respond in the same language the user writes in.

## Tech Stack
- **Backend**: FastAPI + Uvicorn, Python 3.x
- **ML/RL**: Stable-Baselines3 (A2C, PPO, TD3), Gymnasium, PyTorch
- **Prediction**: XGBoost, LightGBM, CatBoost, BiLSTM, TFT + Stacking Ensemble
- **HPO**: Optuna (TPESampler + MedianPruner, TimeSeriesSplit)
- **Data**: yfinance, pandas-ta, scikit-learn, pandas, numpy, evds (TCMB EVDS API)
- **Frontend**: Dash (Plotly) + dash-bootstrap-components, FastAPI üzerinde /dash/ mount edilmiş
- **Explainability**: SHAP (TreeExplainer, LinearExplainer, KernelExplainer)
- **Signal Processing**: EMD-signal (ICEEMDAN gürültü filtreleme)
- **Tests**: Standalone scripts in tests/ (run with `python`, not pytest)

## Project Structure
```
app/                  # FastAPI backend
  api/routes/         # API endpoints (trading.py, health.py, prediction.py, admin.py,
                      #                account.py — kullanicinin kendi hesabi)
  auth/               # Faz 7: kimlik dogrulama, yetkilendirme, calisma alanlari
    models.py         # User / SessionToken / AuditLog (SQLAlchemy)
    db.py             # SQLite engine + init_db
    security.py       # bcrypt + JWT + CSRF + parola politikasi
    service.py        # Kullanici CRUD, authenticate, oturum rotasyonu, audit
    deps.py           # CurrentUser / RequireWriter / RequireAdmin
    middleware.py     # AuthGateMiddleware (/dash + /api kapisi, sessiz yenileme)
    routes.py         # /login, /change-password, /auth/*
    workspace.py      # Kullanici bazli dizin cozumleyici (hibrit izolasyon)
    templates/        # login.html, change_password.html
  schemas/            # Pydantic models
  services/           # Business logic (model_analysis.py, daily_trading.py, prediction_service.py)
    training_eta.py   # Egitim suresi tahmini: on tahmin + canli ETA, gecmis kosumlardan ogrenir
  core/config.py      # Configuration
  main.py             # FastAPI app
data/                 # Data processing modules
  bist30_symbols.py   # Stock symbols list
  data_fetcher.py     # OHLCV - yfinance, retry + incremental + coverage check
  technical_indicators.py
  macro_fetcher.py    # Makro veri - TCMB EVDS (faiz, enflasyon) + yfinance (döviz, BIST100)
  fundamental_fetcher.py  # Fundamental - yfinance (ROE, ROA, P/E, P/B, ...)
  gold_fetcher.py     # Altın/döviz - borsapy veya yfinance
prediction/           # Gelişmiş tahmin sistemi (Faz 2+3)
  feature_engineer.py # 10 özellik grubu (getiri, vol, momentum, makro, fundamental, rejim) + ICEEMDAN + VIX/US10Y/DXY
  feature_selector.py # MI + permutation importance ile otomatik seçim (3 aşamalı)
  iceemdan_processor.py  # ICEEMDAN gürültü filtreleme (EMD-signal)
  tats.py             # TATS trend-adjusted düzeltici (XGBoost trend classifier)
  explainability.py   # SHAP explainability (Tree/Linear/Kernel)
  models/             # Multi-model mimarisi
    base.py           # BasePredictionModel ABC + _predict_direction_raw()
    xgboost_model.py
    lightgbm_model.py
    catboost_model.py
    lstm_model.py     # BiLSTM (PyTorch, CUDA) + direction head
    tft_model.py      # Temporal Fusion Transformer (PyTorch, CUDA) + direction head + Faz6 gruplanmış-VSN (opt-in)
    ensemble.py       # Stacking meta-learner (Ridge + XGBoost), 3-way split, OOF, TATS + warm-start plumbing
    torch_perf.py     # Faz 6 (3.1): DL eğitim ince ayarı — GPU-preload batch, AMP, thread pin, GPU semaphore
  hyperopt.py         # Optuna HPO + Faz 6 (3.2) sqlite resume (HPO_STORAGE)
  trainer.py          # Walk-forward + purge gap + embargo (prev_test_end takibi) + strict/failed_folds
  manifest.py         # Faz 6 (G.3): eğitim manifesti → results/training_runs/<run_id>.json (kullanıcı bazlı)
  evaluator.py        # Direction acc, Profit Factor, IC, Sortino, Calmar, DSR, turnover
  tracker.py          # Experiment tracking (JSON log)
  legacy_models.py    # Eski tek-model implementasyonu (referans)
env/                  # RL Environment (NOT venv!)
  trading_env.py      # Gymnasium custom environment + ATR sizing + Kelly criterion
  reward_functions.py # PSR reward (total_trades bug FIXED)
dashboard/            # Dash frontend (Plotly Dash, /dash/ altında mount)
  app.py              # Dash factory + PrefixMiddleware + FOUC engelleyici index_string
  pages/              # home, training, data, models, daily_trading, prediction,
                      # academic, hyperopt, users, account (Faz 8: Hesabım)
  components/         # sidebar, topbar, metric_card, page_header, table, state_block
  theme.py            # Faz 8: DOM icin var(--token) + Plotly icin hex palet
  assets/
    00-tokens.css     # static/tokens.css'i @import eder (alfabetik once yuklenir)
    custom.css        # Bilesen stilleri — icinde HEX YOK, hepsi token
    theme-toggle.js   # 3 durumlu anahtar + matchMedia + Plotly yeniden boyama
static/
  tokens.css          # Faz 8: TEMA TOKENLARININ TEK KAYNAGI (pano + giris sayfalari)
  favicon.ico
tests/                # Test scripts
scripts/              # Standalone scripts (training, debug, reports)
docs/                 # Documentation (development plan, guides)
models/               # Trained models (.zip) - gitignored (kullanici oncesi, salt-okunur)
results/              # Metrics (.json) - gitignored
logs/                 # TensorBoard logs - gitignored
workspaces/<user_id>/ # Faz 7: kullanici bazli models/results/predictions/live_trading/training_runs - gitignored
```

## Important Rules

### env/ vs venv/ distinction
- `env/` = Trading environment package (Gymnasium). PROJECT CODE.
- `venv/` = Python virtual environment. Gitignored, do not touch.

### Running the server
```bash
python run_server.py  # http://localhost:8000
```

### Running tests
```bash
python tests/test_env.py
python tests/test_ppo.py
python tests/test_all_algorithms.py
python tests/test_env_lookup_equivalence.py # RL env lookup cache bit-eslik (41 kontrol)
python tests/test_env_price_guards.py       # Gecersiz fiyat korumalari (26 kontrol)
python tests/test_training_eta.py           # Egitim suresi tahmini (50 kontrol)
python tests/test_training_status.py        # /train/status progress kirpma (5 kontrol)
python tests/test_auth.py                  # Faz 7: oturum akisi (28 kontrol)
python tests/test_workspace_isolation.py   # Faz 7: izolasyon + RBAC (18 kontrol)
python tests/test_prediction_regression.py # Faz 6: golden davranis dondurma (GPU'da rebaseline: --update)
python tests/test_tft_fast_vsn.py          # Faz 6: gruplanmis-VSN esdegerlik (12 kontrol)
python tests/test_train_batch_parallel.py  # Faz 6: batch paralellik + izolasyon (18 kontrol)
python tests/test_manifest_workspace.py    # Faz 6: manifest calisma alani cozumleme (13 kontrol)
python tests/test_hpo_resume.py            # Faz 6: HPO sqlite resume (12 kontrol)
python tests/test_macro_quality_flag.py    # Faz 6: makro kalite bayragi cache turu (14 kontrol)
python tests/test_theme_contrast.py        # Faz 8: token kontrasti, kacak hex,
                                           #        ucuncu parti cakismasi, kendi
                                           #        CSS siniflarimiz, devre disi
                                           #        varyantlar (92 kontrol)
python tests/test_theme_preference.py      # Faz 8: 3 durumlu tema, sema gocu, CSRF (31 kontrol)
python tests/test_topbar.py                # Faz 8/G: ust cubuk, kirinti, arama,
                                           #          bildirimler (39 kontrol)
python tests/test_delete_artifacts.py      # Faz 8/I: model + optimizasyon kaydi
                                           #          silme, RBAC (32 kontrol)
python tests/test_account_profile.py       # Faz 8/F: profil ucu, oturum yonetimi,
                                           #          etkinlik kaydi, Dash callback
                                           #          smoke (84 kontrol)
```

### Auth & kullanici bazli calisma (Faz 7)
- Giris: `/login` (cerez tabanli oturum — Dash WSGI header tasiyamaz)
- Kapi: `app/auth/middleware.py` — `/dash/*`, `/api/*`, `/docs` korumali;
  acik yollar: `/login`, `/auth/login`, `/auth/refresh`, `/health`, `/static/*`
- Roller: `viewer` (okuma) / `user` (kendi alaninda yazma) / `admin` (+ kullanici yonetimi)
- Kayit yok: hesaplari admin acar (`/dash/users` veya `python scripts/create_admin.py`)
- Dizin cozumleme: `app/auth/workspace.py` → `models_dir()`, `results_dir()`,
  `live_trading_dir()`, `training_runs_dir()`, `find_file(kind, name)`, `use_workspace(user_id)`
- Faz 6 manifest de kullanici bazli: `prediction/manifest.py` → `runs_dir()`/`find_manifest()`;
  `train_batch(user_id=)` arka plan gorevinde calisma alanini sarmalar (thread'e de tasinir)
- Detay: `docs/development/phase7-auth.md`, `docs/development/phase-6-backend-performance.md`

### Tema (Faz 8)
- **Renk degeri yalnizca `static/tokens.css`'te.** Sayfa/bilesen kodunda hex yasak;
  `tests/test_theme_contrast.py` kacaklari yakalar (`dashboard/pages/account.py` muaf —
  iki temanin onizlemesini ayni anda gostermek zorunda).
- Uc durum: `light` / `dark` / `system`. `system` = DOM'da damga YOK, karari
  `@media (prefers-color-scheme)` verir. Bu yuzden koyu blok tokens.css'te **iki kez**
  yazilir; ikisi ayrisirsa test kalir.
- **Tokenlar `--rlt-` onekli.** Dash DataTable kendi bundle'inda `--muted`,
  `--border`, `--accent` tanimliyor ve oneksiz adlari tablo icinde golgeliyor
  (tablo basligi 1.35:1 cikiyordu). Yeni token eklerken oneki koru.
- **Yeni bir `dcc.*` bileseni eklerken sinif ailesini custom.css'e bagla.**
  Dash surumleri DOM'u degistiriyor (`dcc.Dropdown` artik `button.dash-dropdown`,
  react-select degil); baglanmayan aile Dash'in kendi renginde kalir.
  `test_theme_contrast.py` bunu denetliyor.
- Tercih **hesaba** bagli (`users.theme`), cerez yalnizca okuma onbellegi.
  `rlt_theme` = tercih, `rlt_theme_r` = istemcinin cozdugu sonuc (Plotly icin sart).
- **DOM sabiti Plotly'ye verilmez**: `TEXT`, `BLUE` vb. artik `var(--token)` dizesi.
  Grafiklerde `plot_palette()` / `plot_rgba()` / `apply_theme_template()` kullan.
- Detay: `docs/development/phase-8-ui-theming.md`

### Ust cubuk ve bildirimler (Faz 8/G)
- Bilesen: `dashboard/components/topbar.py`. Kenar cubugu tam boy kalir, ust
  cubuk YALNIZCA icerik alanini kaplar (`theme.py::TOPBAR_STYLE`,
  `left: SIDEBAR_WIDTH`); `CONTENT_STYLE` ust boslugu `calc(54px + 24px)`.
- **Gorunum anahtari ust cubukta**, kenar cubugunda degil ve TEK KOPYA.
  `theme-toggle.js` `getElementById` kullaniyor; iki kopya olursa birine
  tiklaninca digerinin etiketi guncellenmez. Test DOM'da tam bir tane
  oldugunu denetliyor.
- **Kenar cubugu menusune madde eklerken `topbar.ROUTE_INDEX`'i unutma** —
  aksi halde ust cubuktaki kirinti o sayfada sessizce bosalir
  (`tests/test_topbar.py` yapisal bekci). `NEXT_STEP` haritasi da yalnizca
  var olan rotalara isaret etmeli.
- Arama role duyarli: `viewer`/`user` icin Yonetim grubu onerilmez.
- **Bildirimler OLAY GUNLUGU DEGIL, DURUM OZETI**: `GET /api/account/notifications`
  bellekteki calisma durumlarindan uretir (`trading._training_states`,
  `prediction_service._training_state`) — kalici tablo, "okundu" isareti yok.
  Is bitince satir kendiliginden kaybolur.
- Veri tazeligi bilincli olarak KAPSAM DISI: `/trading/data/status` paneli
  CSV'den okuyor, 60 sn'de bir yoklanamaz.
- "Bitti" satirlari 12 saatlik pencereyle sinirli (`NOTIFY_RECENT_SECONDS`);
  zaman damgasi olmayan `completed` HIC gosterilmez. Bunun icin
  `trading.py` kosum bitisini `finished_ts`'e yazar.
- **Bilinen ve KABUL EDILMIS davranis:** zil 60 sn'de bir yokladigi icin acik
  bir sekme sessiz yenilemeyi tetikler ve oturumu canli tutar. Ust sinir
  refresh token'in azami omru (`REFRESH_TOKEN_EXPIRE_DAYS`). Degistirmek
  gerekirse: cadansi dusur ya da zili yalnizca acilinca yoklat (rozet sayaci
  gider).

### Dar ekran (Faz 8/H)
- <=820px'de kenar cubugu 64px **ikon rayina** iner; masaustu degismez.
  Olculdu: 1280/1024/820/640'ta tasma veya yatay kaydirma YOK — sorun bozulma
  degil darlikti (640px'de menu ekranin %34'u).
- Menu etiketleri `nav-label` span'inde ki CSS yalnizca yaziyi gizleyebilsin;
  ipucu icin her madde `title` tasiyan bir Div'e sarili (**dbc.NavLink `title`
  kabul etmiyor** — verilince tum Dash agaci render edilemez, `/dash/` 500).
- **Medya sorgusu inline stili `!important` olmadan ezemez.** Konum/genislik
  `theme.py`'den inline geldigi icin ray kurallari `!important` kullanir;
  menu bosluklari bu yuzden inline'dan `#sidebar .nav-link` kuralina tasindi.
- Telefon boyu (<=480px) HEDEF DEGIL ve oyle iddia edilmiyor.

### Yapit silme (Faz 8/I)
- Model: `DELETE /api/trading/models/{name}` (RequireWriter). Panoda: Modeller
  sayfasinda cok secimli liste + "Secilenleri sil".
- **Ortak (kullanici oncesi) model iki katmanli**: `user` silemez (403, mesaj
  yoneticiyi isaret eder), `admin` siler + `model.delete_shared` denetim kaydi.
  Faz 7'de "kimse silemez" idi; o kural deneme modellerini panodan
  temizlemenin hicbir yolunu birakmiyordu.
- Optimizasyon: `DELETE /api/hyperopt/studies/{id}` **kaydi kalici siler**;
  iptal ayri uctadir (`POST /studies/{id}/cancel`). Calisan kosum silmede 409.
- **`OPTUNA_STORAGE` calisma alanina gore COZULMUYOR** — depo kokune sabit
  bagli, yani optimizasyon calismalari tum kullanicilar arasinda ORTAK.
  Bilinen gedik, ayri is olarak ele alinmali (bkz. Faz 8/I.3).

### Hesap ve profil (Faz 8/F)
- Sayfa: `/dash/account` ("Hesabim") — kenar cubugu altindaki **avatar satiri**
  buraya gider (`dashboard/components/sidebar.py::_account_link`). Her rol erisir.
- Uclar: `app/api/routes/account.py` → `GET /api/account/me`,
  `PATCH /api/account/profile`, `GET /api/account/sessions`,
  `POST /api/account/sessions/revoke-others`, `GET /api/account/activity`,
  `GET /api/account/notifications`. Hepsi `CurrentUser`.
- **Neden `/api/*`, `/auth/*` degil:** pano callback'leri `api_client` uzerinden
  cagiriyor; `/api/*` altinda CSRF + RBAC middleware'den bedava geliyor.
  `/auth/*` tarayicinin dogrudan cagirdigi yuzey (giris formu, tema anahtari) —
  orada CSRF ucun kendi isi.
- **Hedef her zaman oturumdaki kullanici**; govdeden kullanici kimligi ALINMAZ.
  `role`/`is_active`/`email` semada yok → kendi rolunu yukseltme yolu kapali.
- **Kasitli oturum iptali kaydi SILER**, `revoked_at` ile isaretlemez: iptal
  edilmis jti 30 sn'lik grace penceresinde (`REFRESH_REUSE_GRACE_SEC`) yeniden
  kullanilirsa `rotate_session` yeni oturum veriyor; isaretleme birakmak
  "diger oturumlari kapat"i atlatilabilir kilardi (bkz.
  `service.revoke_other_sessions` docstring'i).
- **Kendi etkinligi yalnizca `user_id` ile filtrelenir**, `target` ile DEGIL:
  yoneticinin bu hesap uzerindeki islemi o satirda YONETICIYI tasir; gostermek
  admin kimligini ve IP'sini yonetici olmayan bir yuzeye sizdirirdi
  (`service.list_audit_for_user`).
- `dbc.NavLink`'e `title` VERME — kabul etmedigi prop tum Dash agacini
  render edilemez yapar, `/dash/` 500 doner. Ipucunu sarmalayan Div'e koy.
- Detay: `docs/development/phase-8-ui-theming.md` → "Faz F — Profil sayfasi"

### Data pipeline
```
# RL pipeline
yfinance → data_fetcher.py → technical_indicators.py → trading_env.py → SB3 model

# Tahmin pipeline
yfinance       → data_fetcher.py       ─┐
TCMB EVDS      → macro_fetcher.py      ─┤→ feature_engineer.py → feature_selector.py
yfinance       → fundamental_fetcher.py─┤
borsapy/yf     → gold_fetcher.py       ─┘
                                         → trainer.py (walk-forward + purge/embargo)
                                         → models/ (XGB + LGBM + CatBoost + BiLSTM + TFT)
                                         → ensemble.py (stacking meta-learner)
                                         → evaluator.py → trading_env.py (RL obs space)
```

### data_fetcher.py özellikleri
- `fetch_stock_data()`: Retry (exponential backoff) + class-level cache + coverage check (%80)
- `fetch_incremental()`: Sadece eksik günleri çeker, mevcut CSV'ye append eder
- `get_source_status()`: Son tarih ve eksik gün sayısını raporlar
- `clean_data()`: ffill → bfill → negatif fiyat temizleme, sembol bazlı

### macro_fetcher.py özellikleri
- EVDS: policy_rate, cpi_inflation, ppi_inflation
- yfinance: usd_try, eur_try, bist100_index, **vix**, **us10y**, **dxy** (Faz 3.2'de eklendi)

### State space
- **Phase 1 (RL)**: 56 features — balance(1) + shares_owned(5) + OHLCV(25) + technicals(25)
- **Phase 2 (RL + Prediction)**: +4×N features — predicted_return, predicted_direction, prediction_confidence, ensemble_agreement (N sembol başına)
- **Phase 3 (RL)**: ATR tabanlı dinamik pozisyon boyutlandırma + Kelly Criterion (opt-in, `use_atr_sizing`, `use_kelly`)

### ensemble.py özellikleri (Faz 3)
- 3-way chronological split (60/20/20): base train / OOF meta-train / final test — data leakage yok
- Meta-learner: Ridge (default) veya XGBoost (`meta_learner_type='xgboost'`)
- TATS düzeltici: `use_tats=True` ile trend-adjusted output
- Direction head: BiLSTM/TFT'nin `_predict_direction_raw()` çıktısı confidence hesabında kullanılır

### evaluator.py metrikleri (Faz 3)
- Sortino Ratio, Calmar Ratio, Deflated Sharpe Ratio (Bailey & Lopez de Prado), Turnover

## Do NOT
- Read or modify files inside `venv/`
- Add `models/`, `results/`, `logs/`, `workspaces/`, `data/auth/` to git
- `AUTH_ENABLED=False` ile sunucuya cikma — pano ve tum API herkese acik kalir
- Yeni yazma ucu eklerken `RequireWriter`/`RequireAdmin` bagimliligini atlama
  (Faz 8/I'de yakalandi: `/api/hyperopt/start` hic RBAC tasimiyordu, viewer
  GPU'da optimizasyon baslatabiliyordu)
- Arka plan gorevine kullanici kimligini tasimayi unutma (`ws.use_workspace(user_id)`) —
  aksi halde dosyalar yanlis calisma alanina yazilir
- `models/`, `results/`, `data/live_trading` gibi yollari koda sabitleme;
  `app/auth/workspace.py` cozumleyicisini kullan
- Break existing state space structure when modifying `env/trading_env.py`
- Add hardcoded `macro_features=6` — global macro (VIX/US10Y/DXY) sadece prediction pipeline'a gider, RL state space'e eklenmez (trained model uyumluluğu)
- `use_atr_sizing` ve `use_kelly` varsayılan olarak False — mevcut eğitimli modeller bozulmaz
- Sayfa/bileşen koduna hex renk yazma — `static/tokens.css`'e token ekle, kontrastı ölç
- Yeni bir `dbc` bileşeni eklerken **hesaplanmış stile bak**: dbc kendi renk
  varyant sınıfını basıyor (`DropdownMenu` → `btn-primary`, `Badge` → `bg-secondary`)
  ve aynı özgüllükteki kendi kuralımızı kaskadda yenebiliyor. Test seçicinin
  *var olduğunu* doğruluyor, *kazandığını* değil
- Plotly'ye `TEXT`/`BLUE` gibi DOM sabitlerini verme (bunlar `var()` dizesi, grafik siyah çizer)
- `users` tablosuna sütun eklerken `app/auth/db.py::_ADDITIVE_COLUMNS`'a da ekle —
  alembic yok, `create_all()` var olan tabloyu değiştirmez

### Veri butunlugu (ONEMLI)
- **`raw_stock_data.csv` HAM veridir** — 8.155 satirda negatif fiyat var (yfinance'in
  2005 TL sadelestirmesi oncesi duzeltme artefaktlari, 2000-05-10..2004-08-30, 10 sembol).
  **Bu dosyayi TradingEnv'e vermeden once mutlaka `DataFetcher.clean_data()` cagir.**
  Cagirmazsan: negatif fiyat -> BUY'da `cost < 0` -> bakiye artar (yoktan para),
  SELL'de bakiye sinirsiz duser -> gozlemdeki `log()` NaN -> SB3 "Normal(loc: nan)" cokusu.
- `TradingEnv` artik `fiyat <= 0` veya NaN olan satirlarda **islem yapmaz** ve
  `balance_norm` tabanlidir — bozuk veri egitimi cokertemez, ama sessizce de kabul edilmez
  (kurulumda tek seferlik `GECERSIZ FIYAT` raporu basilir).
- Eksik `(sembol, tarih)` cifti yapisaldir (sembol borsaya sonradan girmis olabilir);
  uyari **sembol basina bir kez** verilir, toplam kapsam kurulumda ozetlenir.
- Panelde ayrica **38 tamamen hayalet gun** var (piyasa kapali; OHLC = onceki kapanis,
  volume = 0). Henuz filtrelenmiyor — bkz. asagidaki "bilinen acik".

### Egitim suresi tahmini (ETA)
- Servis: `app/services/training_eta.py` — `estimate()` (on tahmin), `live_eta()` (canli), `record_run()`
- Uclar: `GET /api/trading/train/estimate?algorithm=&phase=&total_timesteps=` ve
  `GET /api/trading/train/status` (eta_seconds/eta_text/finish_at/steps_per_sec/phase_name)
- Model: `toplam = hazirlik + total_timesteps x adim_maliyeti(algoritma, sembol) + degerlendirme`
- Ogrenme: her tamamlanan kosum `results/training_eta_history.json`'a yazilir (kullanici bazli);
  sonraki tahmin medyanini kullanir. Gecmis yoksa yerlesik katsayi (`confidence='default'`)
- **`n_symbols` `get_symbols(phase)`'ten ALINMAZ** — egitim rotasi o listeyi yalnizca veri
  cekerken kullanir, egitim yuklenen CSV'nin tamamiyla yapilir (faz 1 secilse de 30 sembol)
- Varsayilan katsayilari yeniden olcmek icin:
  `python scripts/benchmarking/measure_training_eta_defaults.py --steps 10000,25000`
  (kucuk N'de olcum ~2x oynar; egim buyuk N'de kararli)

## Development Plan
- Faz 1 (POC): Tamamlandı — 5 hisse, A2C/PPO/TD3, temel RL ortamı
- Faz 2 (Advanced Prediction): Tamamlandı — ensemble tahmin sistemi, HPO, RL entegrasyonu
- Faz 3 (Production improvements): Tamamlandı
  - 3.1: Bug fixes (reward total_trades, meta-learner data leakage, embargo, TFT VSN cap, direction head, permutation importance)
  - 3.2: Tahmin kalitesi (ICEEMDAN gürültü filtresi, TATS trend düzeltici, VIX/US10Y/DXY global makro)
  - 3.3: Risk yönetimi (ATR tabanlı pozisyon boyutlandırma, Kelly Criterion)
  - 3.4: Explainability & monitoring (SHAP, Sortino/Calmar/DSR/Turnover metrikleri, /explain API)
- Faz 6 (Backend perf & training throughput): Tamamlandı
  - Ölçüm: `profile_training.py` baseline — eğitim wall-clock'un %99.8'i, TFT ~%90 (VSN döngüsü darboğaz)
  - Veri I/O: paralel sembol + makro çekme, LRU cache; Eğitim: feature-eng bit-eş refactor, feature-sel cache
  - DL ince ayar: GPU-preload (BiLSTM +%56), TFT gruplanmış-VSN (4.7×), HPO sqlite resume; AMP ölçülüp reddedildi
  - Paralellik: sembol-bazlı thread + VRAM semaphore (`TRAIN_PARALLEL_SYMBOLS`, `DL_GPU_SLOTS`)
  - Güvenilirlik: sessiz model/fold düşmesi görünür, fallback işareti + strict mod (cache yolu dahil: `data/macro/macro_data_quality.json`), eğitim manifesti, checkpoint/resume, merkezi seed
  - Kapanış koşumu (T7, 5 sembol): 533.9s → **157.6s (−%70.5)** perf knob'ları açıkken; resume 1.0s; 5/5 sembol `ok` (5 model)
  - **DL perf knob'ları default OFF (opt-in)**: tam pipeline'da RNG sırasını kaydırıp golden'ı değiştirdikleri için (davranış dondurma). Sıfırdan retrain'de açılır, golden o donanımda yenilenir.
- Faz 8 (UI/UX): Tamamlandı — kapsam tur tur genişledi, tamamı
  `docs/development/phase-8-ui-theming.md`'de
  - A–E: aydınlık/koyu/sistem teması (hesaba kayıtlı, 3 durumlu), tek kaynak token
    katmanı (`static/tokens.css`), DARKLY→BOOTSTRAP, Plotly için ayrı hex palet,
    6 yeni/yenilenmiş bileşen, WCAG AA kontrast testi (mevcut koyu temadaki 4 AA
    hatası da düzeldi)
  - F: profil sayfası — kenar çubuğunda görünür giriş noktası, ad soyad düzenleme,
    son giriş/çalışma alanı özeti, kendi oturumlarını görme ve kapatma, kendi
    denetim kaydı (`/api/account/*`); kasıtlı oturum iptalinin grace penceresiyle
    atlatılabilmesi kapatıldı
  - G: üst çubuk — kırıntı, belgelenen akışı izleyen bağlamsal eylem, role duyarlı
    sayfa araması, taşınan görünüm anahtarı ve bellekteki çalışma durumlarından
    beslenen bildirim zili
  - Görsel doğrulama: 9 sayfa × 2 tema (headless Chrome/CDP). Üç kusur çıktı:
    zil `btn-primary` varyantını alıyordu, **devre dışı dolgulu düğmeler uygulama
    genelinde** Bootstrap'in ham paletine düşüyordu, boş portföy grafiği
    mesajsız/eksenli kalıyordu
  - H: ≤820px'de ikon rayı (ölçüm "düzen bozuluyor" varsayımını çürüttü — sorun
    darlıktı); iki turdur kullanılmayan `FilterBar` silindi
  - I: eğitilmiş model ve optimizasyon kaydı silme. Yol üstünde iki gedik:
    `/hyperopt/start` hiç RBAC taşımıyordu (viewer optimizasyon başlatabiliyordu),
    ortak model için Faz 7'nin "kimse silemez" kuralı hiçbir çıkış yolu
    bırakmıyordu → yönetici katmanı eklendi
- Faz 7 (Auth & multi-user): Tamamlandi — cerez tabanli JWT oturum, bcrypt, roller
  (admin/user/viewer), admin-only kayit, denetim kaydi, hibrit kullanici izolasyonu
  (piyasa verisi ortak; model/sonuc/karar/manifest kullanici bazli), kullanici basina egitim durumu
- Detaylar için: `docs/development/roadmap.md`, `docs/development/prediction-system.md`, `docs/development/phase3-implementation.md`, `docs/development/phase-6-backend-performance.md`. Dokümantasyon indeksi: `docs/README.md`.
