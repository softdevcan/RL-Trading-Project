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
    portfolio.py      # Kagit portfoy: pozisyon, ortalama maliyet, mark-to-market P&L
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
  integrity.py        # Faz 10/V: artefakt tarayici, hayalet gun, RL penceresi (prepare_rl_panel)
evaluation/           # Faz 9/A+C: tahmin ve RL'e ortak istatistik + tekrar uretilebilirlik
  sharpe.py           # A1 PSR / DSR (Bailey & LdP 2014; makale ornegi 0,9004)
  forecast_tests.py   # A2 Diebold-Mariano (HAC + HLN, t dagilimi)
  sharpe_tests.py     # A3 Ledoit-Wolf Sharpe farki (prewhitened QS HAC + blok bootstrap)
  model_sets.py       # A4 MCS (arch) · seed_stats.py A5 rliable · overfitting.py A6 PBO/CSCV
  multiple_testing.py # A7 BH-FDR · risk_free.py A8 · paired_tests.py A9
  trial_registry.py   # C2 deneme kaydi (DSR'nin N/V'si, PBO matrisi)
  reproducibility.py  # C1 tohum, C3 veri anlik goruntusu (SHA-256), C4 ortam kaydi
prediction/           # Gelişmiş tahmin sistemi (Faz 2+3)
  feature_engineer.py # 10 özellik grubu (getiri, vol, momentum, makro, fundamental, rejim) + ICEEMDAN + VIX/US10Y/DXY
  feature_selector.py # MI + permutation importance ile otomatik seçim (3 aşamalı)
  iceemdan_processor.py  # ICEEMDAN gürültü filtreleme (EMD-signal)
  tats.py             # TATS trend-adjusted düzeltici (XGBoost trend classifier) — Kazemdehbashi (2026) uyarlaması, atıf zorunlu
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
python tests/test_training_status.py        # /train/status progress, sembol uyarisi,
                                           #        RL + tahmin egitimi durumunun sayfaya
                                           #        donunce geri gelmesi (19 kontrol)
python tests/test_daily_trading_tz.py     # Gunluk karar: CSV (tz'siz) + yfinance (tz'li)
                                           #        birlesimi, sessiz sembol dusmesi (18 kontrol)
python tests/test_paper_portfolio.py      # Kagit portfoy: ortalama maliyet,
                                           #        gerceklesmis/gerceklesmemis kar,
                                           #        cift uygulama bekcisi (43 kontrol)
python tests/test_panel_freshness.py      # Gunluk karar: tazelik olcutu (seans),
                                           #        taslak satir, kapsam esigi,
                                           #        panel onbellegi (21 kontrol)
python tests/test_trade_universe.py       # Gunluk karar: modelin sembol evreni model
                                           #        ADINDAN degil MODELDEN cozulur (22 kontrol)
python tests/test_dash_props.py             # dbc/dcc/html kwarg uyumu (4 kontrol)
python tests/test_auth.py                  # Faz 7: oturum akisi (28 kontrol)
python tests/test_workspace_isolation.py   # Faz 7: izolasyon + RBAC (18 kontrol)
python tests/test_prediction_regression.py # Faz 6: golden davranis dondurma (GPU'da rebaseline: --update)
python tests/test_tft_fast_vsn.py          # Faz 6: gruplanmis-VSN esdegerlik (12 kontrol)
python tests/test_train_batch_parallel.py  # Faz 6: batch paralellik + izolasyon (18 kontrol)
python tests/test_manifest_workspace.py    # Faz 6: manifest calisma alani cozumleme (13 kontrol)
python tests/test_hpo_resume.py            # Faz 6: HPO sqlite resume (12 kontrol)
python tests/test_macro_quality_flag.py    # Faz 6: makro kalite bayragi cache turu (14 kontrol)
python tests/test_macro_pit.py             # Faz 9/B1-B3: EVDS ayristirma, aciklanma
                                           #          gecikmesi, bfill yok (kesme
                                           #          degismezligi), artimli + BIS gecikmesi,
                                           #          mutasyon oz-testi, risksiz faiz (33, ag yok)
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
python tests/test_eval_stats.py            # Faz 9/A: DSR makale ornegi, DM/LW boyut
                                           #          simulasyonu, MCS, rliable, PBO,
                                           #          BH, risksiz faiz (50 kontrol, ~45 sn)
python tests/test_data_integrity.py        # Faz 10/V: artefakt, hayalet gun, RL
                                           #          penceresi, gercek panel saglik (27)
python tests/test_env_missing_mode.py      # Faz 10/V: env eksik hucre modu (17)
python tests/test_rl_reproducibility.py    # Faz 9/C: tohum, deneme kaydi, veri
                                           #          anlik goruntusu, manifest (39)
python tests/test_price_stats_window.py    # Faz 9/B9: val/test olcegi = egitim olcegi;
                                           #          env + rota + HPO uctan uca, mutasyon,
                                           #          main ile bit-es varsayilan, AST bekcisi,
                                           #          sarmalayicili betikler (65)
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

### Pano / tema / UI kurallari → `dashboard/CLAUDE.md`
Tema tokenlari, ust cubuk, dar ekran, dbc/Dash prop tuzaklari orada (dashboard/ altinda calisinca otomatik yuklenir).
**Ozet:** hex renk yalnizca `static/tokens.css`'te; `dbc.NavLink`'e `title` verme; UI degisikliginden sonra
`python tests/test_theme_contrast.py` ve `python tests/test_dash_props.py` calistir.

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
- EVDS (Faz 9 B2, 2026-11-06'da EVDS kataloğundan doğrulandı):
  `policy_rate` = TP.BISPOLFAIZ.TUR (aylık), TÜFE = TP.TUKFIY2025.GENEL, ÜFE = TP.TUFE1YI.T1
  (ikisi de aylık, yıllık % değişim). **Eski kodlar (TP.YSSK.A01, TP.FG.G0) EVDS'de YOK**;
  TP.FG.J0 (2003=100) 2026-01'de bitiyor. Bu yüzden Temmuz 2026'ya kadarki her çekim faizde
  sabit 50'ye, TÜFE/ÜFE'de 0'a düşüyordu — 50 hiçbir zaman EVDS değeri değildi.
- **EVDS tuzakları:** sütun adı alt çizgili döner (`TP_FG_J0`); aylık `Tarih` 'YYYY-M';
  tek istekte en fazla 1000 satır, fazlası aralığın BAŞINDAN kesilir (`_evds_series` parçalar).
- **Nokta-zamanlı (B3):** ay m TÜFE/ÜFE'si ay m+1'in 6'sından, ay m politika faizi ay m+1'in
  1'inden itibaren geçerli (tasarım varsayımı). **`bfill` YOK (B1)**; baş boşluk atılır,
  bitiş tarihinden sonrası yazılmaz. `.bfill()` geri getirme — `tests/test_macro_pit.py`
  ilk satırın değerini DOĞRUDAN denetler (kesme değişmezliği bfill'i YAKALAMAZ).
- Olay zamanlı, **vintage değil**: BIS aylık faizi ~5 hafta gecikmeli; geçmişte ay m−1, canlıda
  m−2 görünür (açık iş, Faz 9 §9.4 #4).
- Artımlı çekim EVDS'i 120 gün geriye bakarak çeker; yedek değerli delta dosyaya YAZILMAZ.
  Kalite bayraklı mevcut dosya artımlı değil TAM çekimle yenilenir.
- **Risksiz faiz ayrı dosyada:** `data/macro/risk_free.csv` (`fetch_risk_free()`; TP.APIFON4
  TCMB ağırlıklı ortalama fonlama maliyeti, iş günü, 2011'den). Veri sayfası makro güncellemesi
  bunu da yeniler. Yedeği yok.
- yfinance: usd_try, eur_try, bist100_index, **vix**, **us10y**, **dxy** (Faz 3.2'de eklendi)
- **Makro 2018'den başlıyor** (varsayılan `start_date`); RL penceresi 2013-06. Faz 2 RL ortamı
  2013–2017'de makro gözlemi 0 görür — bilinen açık.

### State space
- **Phase 1 (RL)**: 56 features — balance(1) + shares_owned(5) + OHLCV(25) + technicals(25)
- **Phase 2 (RL + Prediction)**: +4×N features — predicted_return, predicted_direction, prediction_confidence, ensemble_agreement (N sembol başına)
- **Phase 3 (RL)**: ATR tabanlı dinamik pozisyon boyutlandırma + Kelly Criterion (opt-in, `use_atr_sizing`, `use_kelly`)

### ensemble.py özellikleri (Faz 3)
- 3-way chronological split (60/20/20): base train / OOF meta-train / final test.
  **"Leakage yok" DEGIL:** dagitim oncesi %80 yeniden uyumda meta-ogrenici orneklem ICI
  tahminlerle yeniden uyduruluyor (ensemble.py:345–384; Faz 10 §8 #7 → K19)
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
- `users` tablosuna sütun eklerken `app/auth/db.py::_ADDITIVE_COLUMNS`'a da ekle —
  alembic yok, `create_all()` var olan tabloyu değiştirmez

### Uzun suren isler ve sayfa gezinmesi (ONEMLI)
- `display_page` yalnizca `page-content`'i degistirir; sayfa her gezinmede
  **yeniden uretilir**. `dcc.Interval(disabled=True)` + "yalnizca Baslat
  dugmesi acar" kalibi bu yuzden bozuk: baska sayfaya gidip donen kullanici
  surmekte olan isin ilerlemesini bir daha goremez.
- Kural: uzun suren bir isin sayfasi **acilista durumu sormali** ve yoklamayi
  ona gore acmali (`training.py::layout` -> `_status_block`,
  `hyperopt.py::layout` -> calisan study'yi listeden bulur).
- Ayrica calisan isin kimligini yalnizca `dcc.Store`'da tutma — store da
  sifirlanir; kimlik backend'den geri bulunabilmeli. Tahmin egitiminde
  `/prediction/train/status` SEMBOL istiyor, sayfa donuste neyi soracagini
  bilmiyordu; `GET /api/prediction/train/active` listeyi backend'den verir.
- Yoklamayi is bitince KAPAT (`disabled=True`), yoksa sayfa acik kaldigi
  surece bos yere sorgu atar.

### Sembol evreni ve train/test bolmesi (ONEMLI)
- Gozlem uzayi sembol sayisina bagli: `1 + n + 5n + 5n`. 5 sembol -> 56,
  30 sembol -> 331 ozellik.
- `DataFetcher.split_data()` **KRONOLOJIK** boler ve sembol uyelugine bakmaz.
  Sembollerin gecmisleri esit degilse bolumlerin sembol sayisi FARKLI cikar:
  ```
  train 2018-01-01..2024-01-16 ->  5 sembol ->  56 ozellik
  test  2025-05-14..2026-08-28 -> 30 sembol -> 331 ozellik
  ```
  Model 56 ile egitilip 331 ile degerlendiriliyordu; SB3 `predict` asamasinda
  `Unexpected observation shape` ile patliyor ve HICBIR kosum tamamlanamiyordu.
- Egitim rotasi artik val/test'i **egitim bolumunun sembol evrenine hizaliyor**
  ve dusen sembolleri `training_state["warnings"]` ile panoya tasiyor. Sessiz
  hizalama YAPMA: kullanici 30 sembol sandigi modeli 5 sembolle egitmis olur.
- Kalici cozum veride: eksik sembollerin tam gecmisini indir (Veri sayfasi ->
  "Yeniden Indir (tam)").

### Modelin sembol evreni model ADINDAN cozulmez (ONEMLI)
- Egitim rotasi `get_symbols(phase)` listesini YALNIZCA veri cekerken kullanir;
  egitimi yuklenen panelin tamamiyla yapar. Yani `ppo_phase1_...` adli bir model
  pekala 30 sembolle egitilmistir (obs=331, PHASE1_SYMBOLS ise 5 -> obs=56).
  `/trading/daily-decision` evreni adindan tahmin edince durum vektoru 56
  cikiyordu ve SB3 `predict` asamasinda patliyordu:
  `Unexpected observation shape (56,) ... please use (331,)`.
- Tek gercek kaynak MODELIN KENDISI: `action_space.shape[0]` = sembol sayisi.
  `daily_trading.resolve_trade_universe()` adaylari bu sayiya gore eler:
  yan dosya -> egitim paneli (yeniden uretilir) -> ad tabanli sabitler.
- **Yeni modeller kendi evrenini tasir**: egitim `model.save()` yanina
  `<model>.meta.json` yazar (`write_model_meta`). Eski modeller icin:
  `python scripts/backfill_model_meta.py --write` (once `--write`siz deneme kosumu).
- **Sembol SIRASI onemli** — durum vektoru sembolleri env'deki sirayla diziyor
  (`TradingEnv.symbols` = panelin gorunme sirasi). `sorted()` uygulamak state'i
  sessizce bozar; cozumleyici sirayi oldugu gibi tasir.
- Model silinince yan dosya da silinir (`delete_model`); `list_models` yalnizca
  `.zip` suzdugu icin meta dosyasi model gibi listelenmez.
- Evren dogru sayida cikip durum yine de uymazsa (faz 2 modeli: hisse basina 17
  ozellik + 6 makro; `build_live_state` 10 uretir) rota SB3'a girmeden 400 ile
  nedeni soyler.

### Kagit portfoy ve kar/zarar (ONEMLI)
- **`summary.daily_return_pct` KAR/ZARAR DEGILDIR.** `portfolio_before` ve
  `portfolio_after` ayni gunun ayni fiyatlariyla hesaplanir; alim-satim
  nakit<->hisse takasi oldugu icin portfoy degeri degismez, geriye yalnizca
  komisyon kalir. Metrik tanim geregi ~0 veya negatif cikar (olculdu:
  100.000,00 -> 99.998,99 = "-0,0010%", komisyon 1,01 TL). Kar/zarar icin
  `portfolio.value_portfolio()` kullan — pozisyonu BASKA BIR GUNUN fiyatiyla
  degerler.
- Tek gercek kaynak `workspaces/<uid>/data/live_trading/portfolio.json`
  (`app/services/portfolio.py`). `daily_trading.py` kararin NASIL uretildigini,
  `portfolio.py` SONUCUNU bilir.
- **Nakit tanimlari `interpret_actions_with_risk` ile ayni olmali:**
  `BUY -> cost = adet*fiyat*(1+komisyon)` (komisyon DAHIL),
  `SELL -> revenue = adet*fiyat*(1-komisyon)` (DUSULMUS). `avg_cost` bu yuzden
  komisyon dahil tutulur; aksi halde gerceklesmis kar komisyon kadar sisik cikar.
- **Ayni tarih iki kez uygulanmaz** (`applied_dates`) — "Uygula"ya iki kez
  basmak pozisyonu iki katina cikarirdi.
- `/daily-decision` icin `balance`/`shares` artik OPSIYONEL: verilmezse
  portfoyden yuklenir. Panoda "Portfoyu elle gir" anahtari acikken gonderilir
  ve o karar UYGULANAMAZ (uydurma bir baslangictan ilerletirdi).
- Fiyati cekilemeyen sembol ortalama maliyetiyle degerlenir ve
  `missing_prices` ile raporlanir — 0 saymak portfoyu kucuk, sessiz kalmak
  kar/zarari sahte gosterirdi.
- `POST /portfolio/reset` gecmis dosyalarina DOKUNMAZ (trade_decisions.json,
  portfolio_history.csv); silmek geri alinamaz, ayri bir istek olmali.
- Test: `python tests/test_paper_portfolio.py` (43 kontrol)

### Panel tazeligi ve yfinance taslak satiri (ONEMLI)
- **Tazelik olcutu takvim gunu DEGIL, kapanmis olmasi beklenen son SEANS.**
  `cached_last < end_date.date()` sorusu 30 Agustos Pazar hedefinde CSV 28
  Agustos Cuma'ya kadar dolu olsa bile dogru cikiyordu; tek karar istegi 30 CSV
  okumasi + **30 yfinance indirmesi** yapiyordu, sonuc hic degismeden.
  `daily_trading.last_expected_session()` hafta sonunu geriye sarar. Resmi BIST
  tatilleri bilinmiyor; o gunlerde tek bosuna tur atilir, TTL onbellegi keser.
- **yfinance seans kapanmadan once OHLC'si NaN, volume'u DOLU taslak satir
  dondurur.** Bu satir diske yazilirsa iki zarar birden: panelde tamamen bos
  bir gun kalir **ve** `fetch_incremental`'in `min_last_date`'i o gune kayar,
  `fetch_from` ertesi gun olur, o seansin gercek verisi bir daha HIC cekilmez.
  Filtre `fetch_incremental`'da **yalnizca yeni veriye** uygulanir — `existing`
  ham panel ve 2005 oncesi negatif fiyat artefaktlarini tasiyor (asagiya bak);
  oraya `close > 0` uygulamak gecmisi sessizce degistirirdi.
- **`actual_date` sembol kapsamina bakar** (`MIN_SESSION_COVERAGE`, %90).
  Panelin bir kismi tazelenip kalani tazelenmeyince (28.08.2026: 30 sembolden
  3'u) en son tarihte yalnizca o 3 sembolun kapanisi oluyordu; kalan 27 sembol
  tek tek bir onceki gune dusuyor, durum vektoru **iki gunun karisimini**
  tasiyordu. Esigi gecemeyen gun atlanir.
- Panel `(semboller, hedef, lookback, CSV mtime)` anahtariyla 15 dk onbellege
  alinir. Veri sayfasi yeni veri indirince mtime degisir, onbellek TTL
  beklemeden duser. Testte `dt.clear_panel_cache()` cagir — yoksa ikinci
  cagriyi onbellek karsilar ve test sahte yesil verir.
- Test: `python tests/test_panel_freshness.py` (21 kontrol)

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
- **RL veri kapisi (Faz 10 / V, K17): `data/integrity.py::prepare_rl_panel`.** Egitim
  rotasi ve HPO paneli buradan gecirir:
  - pencere `RL_DATA_START = 2013-06-01` (`app/core/config.py`): panelde en gec baslayan
    sembol ODAS (2013-05-21); 2000-2005 artefaktlari pencere disinda kalir
  - pencere ICINDE fiyat artefakti varsa `DataIntegrityError` — sessizce temizlenmez
  - hayalet gun (aktif satirlarin >= %90'i duz: volume=0 ve O=H=L=C) atilir; tek sembolun
    duz satiri (bayat hucre) atilir, hucre EKSIK sayilir. "= onceki kapanis" sarti ARANMAZ:
    yfinance kapali gunde duzeltilmis fiyati farkli yazabiliyor (2013-10-15 boyle kaciyordu)
  - olcum (2026-11-06): tam panelde 251 hayalet gun, 235 artefakt satiri; pencerede
    0 artefakt, 72 hayalet gun + 39 bayat hucre atiliyor, eksik hucre %0,076
  - gosterge isinma `bfill`'i (ODAS/PGSUS CCI) ve ADX yalnizca RAPORLANIR (B7, K18 acik)
- **Tarayici surekli yuksek ama sicramayan seviyeyi YAKALAMAZ** (TUPRS 2000-2003 > 10.000 TL
  satirlari "level_jump" degil). Bu satirlar pencere disinda; pencereyi geri cekme.
- **`TradingEnv(missing_data_mode=...)`**: `'legacy'` (varsayilan, egitilmis modeller) eksik
  hucrede ONCEKI SEMBOLUN blogunu kopyalar ve fiyati 0 sayar — bilinen hata, bit-es dondurulmus.
  `'own_last'` sembolu kendi son gozlemiyle tasir, degerlemeyi kendi son kapanisiyla yapar,
  o hucrede islem yapmaz. Yeni egitim ve HPO `own_last` kullanir.
  Test: `python tests/test_data_integrity.py` (27), `python tests/test_env_missing_mode.py` (17)
- **Val/test/degerlendirme ortami EGITIM olcegini alir (Faz 9 B9, Faz 10 Y13):**
  `TradingEnv(price_stats=None)` fiyat z-skorunu ortamin KENDI df'inden kurar — yalnizca
  egitim ortami icin dogru. Held-out pencerede kurulan her env'e
  `price_stats=<egitim env>.get_price_stats()` (ya da modelin meta `price_stats`'i) VER;
  vermezsen t gozlemi t sonrasi fiyatlara baglanir. Eksik sembol / bozuk deger `ValueError`
  (sessiz yedek yok). Meta'sinda olcek olmayan eski model icin olcek UYDURMA (K13).
  - Protokol etiketi beyan degil olcum: `evaluation.trial_registry.rl_eval_protocol` (rota + HPO
    ortak). Sabit `'train_window'` YAZMA — olcek gecirilmezse kayit kendiliginden acik kalmali.
  - Bedeli: eski ortamda TL fiyatlari egitim olceginin cok ustunde -> test fiyat ozelliklerinin
    %74,7'si +-10'da kirpiliyor (Faz 9 §9.5). Rota her kosumda `observation_health` yazar; K18
    olcekten bagimsiz ozellik seti.
  - AST bekcisi held-out adli df (`test_df`, `val_df`, `data['test']`...) ile kurulan
    `make_env`/`TradingEnv` cagrisinda `price_stats` arar; bilerek eski olcekle olcum yapan
    satira `# B9-legacy` yaz. Test: `python tests/test_price_stats_window.py`
  - **HPO ham OHLCV okuyor** (`raw_stock_data.csv`, gosterge yok -> 5/10 ozellik sabit
    varsayilan; Faz 9 §9.5 Y21). Kayit `indicators_present=False` ile acik; duzeltme R1.
  - Testte CPU'ya zorlamak icin `CUDA_VISIBLE_DEVICES=-1`: Windows'ta bos dize degiskeni SILER.
- **ADX tum panelde 0** (sabit sutun) — Faz 10 K18, bu kapida duzeltilmedi;
  `test_data_integrity.py` bunu bilinen acik olarak sabitler.

### Akademik gecerlilik ve literatur (ONEMLI) → `docs/thesis/CLAUDE.md`
Tez/literatur kurallari orada (docs/thesis/ altinda calisinca otomatik yuklenir). Kodda da gecerli olan cekirdek:
- **"Leakage yok" DEME** — dogrulanmis sizinti kaynaklari var (`docs/development/roadmap.md` → "Bilinen Metodolojik Aciklar").
- **Faz 9 ve Faz 10 planlari ONAYLANDI (2026-11-06)** —
  `docs/development/phase-9-research-validity-plan.md`, `docs/development/phase-10-marl-plan.md`
  (§8 hakem-2 eklemeleriyle). Kararlar: K1 varsayilani cevir + golden'i Milestone 0' ile bir kez
  yenile; K17 RL penceresi 2013-06; K23 anahtar koddan kaldirildi (PR #3), yenilenmeyecek.
  Is planin paket SIRASIYLA yapilir; planda olmayan bir duzeltmeye girisme, plana ekletip yap.
  - **Ilk kapi (V + A + C) yapildi** — durum ve olcumler: Faz 9 planinda "Uygulama durumu".
  - **Degerlendirme icin `evaluation/` paketini kullan**, `prediction/evaluator.py`'deki eski
    DSR/DM degil (onlar yonlendirildi). DSR icin N ve V deneme kaydindan
    (`evaluation/trial_registry.py`, `results_dir()/trial_registry.jsonl`) gelir — N=1'de DSR yok.
  - **Deneme kaydi protokol tasir**: bilinen acik madde (test olcegi Y13, gosterge bfill Y15,
    ADX K18) varsa kayit `protocol_open` etiketi alir ve `load_trials` onu varsayilan DISLAR.
    Budanan/basarisiz denemeler de kaydedilir (N'ye sayilir). Kind/split/pencere karistirma.
  - **Bu kapidan SONRAKI RL sonuclari da on sonuctur** (B7 / K18 kapanana kadar). B9 kapandi
    (§9.5), ama eski ortamda fiyat ozellikleri doygun ve HPO gostergesiz (Y21, R1).
  - **Risksiz faiz** yalnizca `evaluation/risk_free.py`'den (varsayilan `risk_free.csv`
    `funding_rate`); dosya yoksa ya da seri yedek degerliyse acik hata — sabite dusurme.
    Eski 0,15 / 0,02 / 0 sabitleri BILEREK baglanmadi (odul = davranis dondurma; pano/rapor D'de).
  - **RL tohumu**: `TrainingRequest.seed` (yoksa uretilir); meta + manifest + deneme kaydina yazilir.
    Test: `python tests/test_eval_stats.py` (50), `python tests/test_rl_reproducibility.py` (39)
- Kaynak ekleme: `/kaynak-ekle` skill'i. Metodolojik elestiri: `hakem-2` alt ajani.
- Hook'lar (`.claude/settings.json` → `.claude/hooks/guard.py`): uretilen literatur dosyalarini
  elle duzenleme engellenir; tez metnine yeni "sizinti yok" iddiasi onay ister;
  `env/trading_env.py` degisince env bekci testleri otomatik kosar.

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
- Faz 8 (UI/UX): Tamamlandi — tema, profil, ust cubuk, dar ekran, yapit silme;
  ayrinti `docs/development/phase-8-ui-theming.md` ve `dashboard/CLAUDE.md`
- Faz 7 (Auth & multi-user): Tamamlandi — cerez tabanli JWT oturum, bcrypt, roller
  (admin/user/viewer), admin-only kayit, denetim kaydi, hibrit kullanici izolasyonu
  (piyasa verisi ortak; model/sonuc/karar/manifest kullanici bazli), kullanici basina egitim durumu
- Detaylar için: `docs/development/roadmap.md`, `docs/development/prediction-system.md`, `docs/development/phase3-implementation.md`, `docs/development/phase-6-backend-performance.md`. Dokümantasyon indeksi: `docs/README.md`.
