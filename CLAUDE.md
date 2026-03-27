# CLAUDE.md - RL Trading Project

## Project Summary
Deep Reinforcement Learning-based algorithmic trading system for BIST-30 stocks.
Based on Ansari et al. (2024) paper. Phase 1 (POC) completed, Phase 2 (Advanced Prediction System) completed, Phase 3 (Production improvements) completed.

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
  api/routes/         # API endpoints (trading.py, health.py, prediction.py)
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
    tft_model.py      # Temporal Fusion Transformer (PyTorch, CUDA) + direction head
    ensemble.py       # Stacking meta-learner (Ridge + XGBoost), 3-way split, OOF, TATS
  hyperopt.py         # Optuna HPO
  trainer.py          # Walk-forward + purge gap + embargo (prev_test_end takibi)
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
models/               # Trained models (.zip) - gitignored
results/              # Metrics (.json) - gitignored
logs/                 # TensorBoard logs - gitignored
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
```

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
- Add `models/`, `results/`, `logs/` to git
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
- Detaylar için: `docs/development.md`, `docs/advanced_prediction_system.md`
