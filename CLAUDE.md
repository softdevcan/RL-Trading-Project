# CLAUDE.md - RL Trading Project

## Project Summary
Deep Reinforcement Learning-based algorithmic trading system for BIST-30 stocks.
Based on Ansari et al. (2024) paper. Phase 1 (POC) completed, Phase 2 (Advanced Prediction System) completed.

## Language
Respond in the same language the user writes in.

## Tech Stack
- **Backend**: FastAPI + Uvicorn, Python 3.x
- **ML/RL**: Stable-Baselines3 (A2C, PPO, TD3), Gymnasium, PyTorch
- **Prediction**: XGBoost, LightGBM, CatBoost, BiLSTM, TFT + Stacking Ensemble
- **HPO**: Optuna (TPESampler + MedianPruner, TimeSeriesSplit)
- **Data**: yfinance, pandas-ta, scikit-learn, pandas, numpy, evds (TCMB EVDS API)
- **Frontend**: Vanilla JS (ES6+), Chart.js 4.4.0, CSS Grid
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
prediction/           # Gelişmiş tahmin sistemi (Faz 2)
  feature_engineer.py # 10 özellik grubu (getiri, vol, momentum, makro, fundamental, rejim)
  feature_selector.py # Mutual information + permutation importance ile otomatik seçim
  models/             # Multi-model mimarisi
    base.py           # BasePredictionModel ABC
    xgboost_model.py
    lightgbm_model.py
    catboost_model.py
    lstm_model.py     # BiLSTM (PyTorch, CUDA)
    tft_model.py      # Temporal Fusion Transformer (PyTorch, CUDA)
    ensemble.py       # Stacking meta-learner (Ridge + XGBoost)
  hyperopt.py         # Optuna HPO
  trainer.py          # Walk-forward + purge gap + embargo, multi-model paralel
  evaluator.py        # Direction acc, Profit Factor, IC, backtest, Diebold-Mariano
  tracker.py          # Experiment tracking (JSON log)
  legacy_models.py    # Eski tek-model implementasyonu (referans)
env/                  # RL Environment (NOT venv!)
  trading_env.py      # Gymnasium custom environment
static/               # Frontend (SPA)
  index.html
  css/styles.css
  js/dashboard.js
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

### State space
- **Phase 1 (RL)**: 56 features — balance(1) + shares_owned(5) + OHLCV(25) + technicals(25)
- **Phase 2 (RL + Prediction)**: +4×N features — predicted_return, predicted_direction, prediction_confidence, ensemble_agreement (N sembol başına)

## Do NOT
- Read or modify files inside `venv/`
- Add `models/`, `results/`, `logs/` to git
- Break existing state space structure when modifying `env/trading_env.py`
- Add JS frameworks to frontend (keep vanilla JS)

## Development Plan
- Faz 1 (POC): Tamamlandı — 5 hisse, A2C/PPO/TD3, temel RL ortamı
- Faz 2 (Advanced Prediction): Tamamlandı — ensemble tahmin sistemi, HPO, RL entegrasyonu
- Faz 3 (Production): Planlama aşamasında
- Detaylar için: `docs/development.md`, `docs/advanced_prediction_system.md`
