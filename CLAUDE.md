# CLAUDE.md - RL Trading Project

## Project Summary
Deep Reinforcement Learning-based algorithmic trading system for BIST-30 stocks.
Based on Ansari et al. (2024) paper. Phase 1 (POC) completed, Phase 2 in development.

## Language
Respond in the same language the user writes in.

## Tech Stack
- **Backend**: FastAPI + Uvicorn, Python 3.x
- **ML/RL**: Stable-Baselines3 (A2C, PPO, TD3), Gymnasium, PyTorch
- **Data**: yfinance, pandas-ta, scikit-learn, pandas, numpy
- **Frontend**: Vanilla JS (ES6+), Chart.js 4.4.0, CSS Grid
- **Tests**: Standalone scripts in tests/ (run with `python`, not pytest)

## Project Structure
```
app/                  # FastAPI backend
  api/routes/         # API endpoints (trading.py, health.py)
  schemas/            # Pydantic models
  services/           # Business logic (model_analysis.py, daily_trading.py)
  core/config.py      # Configuration
  main.py             # FastAPI app
data/                 # Data processing modules
  bist30_symbols.py   # Stock symbols list
  data_fetcher.py     # Data fetching via yfinance
  technical_indicators.py
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
yfinance → data_fetcher.py → technical_indicators.py → trading_env.py → SB3 model
```

### State space (Phase 1): 56 features
balance(1) + shares_owned(5) + OHLCV(25) + technicals(25)

## Do NOT
- Read or modify files inside `venv/`
- Add `models/`, `results/`, `logs/` to git
- Break existing state space structure when modifying `env/trading_env.py`
- Add JS frameworks to frontend (keep vanilla JS)

## Development Plan
See `docs/development.md` for detailed phase plan.
