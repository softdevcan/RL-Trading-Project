# CLAUDE.md - RL Trading Project

## Project Summary
Deep Reinforcement Learning-based algorithmic trading system for BIST-30 stocks.
Based on Ansari et al. (2024) paper. Phase 1 (POC), Phase 2 (Advanced Prediction System), Phase 3 (Production improvements), Phase 6 (Backend perf & training throughput), Phase 7 (Auth & multi-user) tamamlandı.

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
  api/routes/         # API endpoints (trading.py, health.py, prediction.py, admin.py)
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
  app.py              # Dash factory + PrefixMiddleware
  pages/              # home, training, data, models, daily_trading, prediction, academic, hyperopt
  components/         # sidebar, metric_card
  theme.py            # Renk teması
static/               # Sadece favicon
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
python tests/test_auth.py                  # Faz 7: oturum akisi (28 kontrol)
python tests/test_workspace_isolation.py   # Faz 7: izolasyon + RBAC (18 kontrol)
python tests/test_prediction_regression.py # Faz 6: golden davranis dondurma (GPU'da rebaseline: --update)
python tests/test_tft_fast_vsn.py          # Faz 6: gruplanmis-VSN esdegerlik (12 kontrol)
python tests/test_train_batch_parallel.py  # Faz 6: batch paralellik + izolasyon (18 kontrol)
python tests/test_manifest_workspace.py    # Faz 6: manifest calisma alani cozumleme (13 kontrol)
python tests/test_hpo_resume.py            # Faz 6: HPO sqlite resume (12 kontrol)
python tests/test_macro_quality_flag.py    # Faz 6: makro kalite bayragi cache turu (14 kontrol)
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
- Arka plan gorevine kullanici kimligini tasimayi unutma (`ws.use_workspace(user_id)`) —
  aksi halde dosyalar yanlis calisma alanina yazilir
- `models/`, `results/`, `data/live_trading` gibi yollari koda sabitleme;
  `app/auth/workspace.py` cozumleyicisini kullan
- Break existing state space structure when modifying `env/trading_env.py`
- Add hardcoded `macro_features=6` — global macro (VIX/US10Y/DXY) sadece prediction pipeline'a gider, RL state space'e eklenmez (trained model uyumluluğu)
- `use_atr_sizing` ve `use_kelly` varsayılan olarak False — mevcut eğitimli modeller bozulmaz

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
- Faz 7 (Auth & multi-user): Tamamlandi — cerez tabanli JWT oturum, bcrypt, roller
  (admin/user/viewer), admin-only kayit, denetim kaydi, hibrit kullanici izolasyonu
  (piyasa verisi ortak; model/sonuc/karar/manifest kullanici bazli), kullanici basina egitim durumu
- Detaylar için: `docs/development/roadmap.md`, `docs/development/prediction-system.md`, `docs/development/phase3-implementation.md`, `docs/development/phase-6-backend-performance.md`. Dokümantasyon indeksi: `docs/README.md`.
