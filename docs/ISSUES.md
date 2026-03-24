# Issue Tracker — Detailed Bug & Problem List

Analysis date: 2026-03-24. Each issue has file:line reference and severity.

Legend: `[FATAL]` = crashes/non-functional, `[BUG]` = wrong behavior, `[SEC]` = security,
`[DEPR]` = deprecated API, `[DESIGN]` = architecture/design problem

---

## 1. RL Environment & Training

### env/trading_env.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 1 | BUG | 71-76 | Unbounded observation space | `Box(low=-np.inf, high=np.inf)` causes gradient instability in SB3. Should use bounded space or normalized observations. |
| 2 | BUG | 305-308 | Hardcoded normalization | `(price - 50) / 50` assumes stocks ~50 TL. BIST-30 stocks range 1-500+ TL. Must use dynamic normalization (z-score or min-max). |
| 3 | BUG | 180-188 | Reward function diverges from Ansari | Uses `portfolio_change% - commission%`. Ansari uses risk-adjusted PSR (Probabilistic Sharpe Ratio). |
| 4 | BUG | 335-337 | Silent missing data | `KeyError` caught and filled with `0.0` instead of forward-fill + logging. Corrupts portfolio calculations. |
| 5 | BUG | 149 | Threshold disabled | `min_threshold = 0  # DISABLED!` — minimum trade threshold is off. |

### data/technical_indicators.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 6 | BUG | 58-59 | RSI uses SMA instead of Wilder's EMA | `gain.rolling(window=period).mean()` should be `.ewm(span=period, adjust=False).mean()`. Non-standard RSI. |
| 7 | BUG | 112-114 | ADX uses SMA instead of EMA | Same issue as RSI — Wilder's smoothing required for ATR and directional indicators. |
| 8 | BUG | 117 | Division by zero in ADX | `dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)` — no zero check on denominator. |
| 9 | DESIGN | 135-143 | Turbulence oversimplified | Just `rolling_std * 100`, not Mahalanobis distance as in Ansari/literature. |
| 10 | BUG | — | MACD missing signal line | Only returns MACD line, no 9-period EMA signal line or histogram. |
| 11 | DEPR | 166-167 | Deprecated pandas fillna | `.fillna(method='ffill')` → `.ffill()` (pandas 2.0+). |

### data/data_fetcher.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 12 | DEPR | 151, 154 | Deprecated pandas fillna | `.fillna(method='ffill')` → `.ffill()` (pandas 2.0+). |
| 13 | DESIGN | — | No retry logic | Network failures cause silent data loss. |
| 14 | DESIGN | — | No minimum data coverage validation | No check that enough data rows exist before proceeding. |

### scripts/generate_academic_report.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 15 | FATAL | 13 | Wrong import path | `from app.environments.trading_env import MultiStockTradingEnv` — module doesn't exist. Should be `from env.trading_env import TradingEnv`. |
| 16 | FATAL | 17 | Missing data file | Expects `data/preprocessed/train_val_test_data.pkl` — file doesn't exist. |
| 17 | BUG | 75 | Wrong attribute | `env.portfolio_value` — attribute doesn't exist on TradingEnv. |
| 18 | BUG | 82 | Wrong attribute name | `env.trade_history` — should be `env.trades_history`. |

### scripts/train_a2c_phase1.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 19 | BUG | 96, 121, 135 | n_steps parameter ignored | Function param `n_steps=5` is logged, but line 121 hardcodes `n_steps=256`. Misleading hyperparameter tracking. |
| 20 | BUG | 180, 182 | DummyVecEnv done array | `while not done:` with `obs, reward, done, info = env.step(action)` — DummyVecEnv returns `done` as ndarray, `not done` evaluates array truthiness incorrectly. |

---

## 2. Backend (FastAPI)

### app/core/config.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 21 | SEC | 21-22 | CORS wildcard + credentials | `CORS_ORIGINS: ["*"]` with `CORS_CREDENTIALS: True`. Browsers reject this combo, but it signals intent to allow everything. |

### app/api/routes/trading.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 22 | SEC | 323, 146 | Path traversal | Model names used directly in file paths without sanitization. `../../etc/passwd` style attacks possible. |
| 23 | BUG | 28-35 | Thread-unsafe global state | `training_state` dict accessed from multiple threads without locking. |
| 24 | BUG | 561 | Test eval uses deterministic=False | Should be `deterministic=True` for reproducible evaluation. |
| 25 | BUG | 563-575 | Auto-reset workaround | Patches symptom (DummyVecEnv auto-reset) instead of fixing root cause. |
| 26 | BUG | 914-921, 929-937 | Hardcoded wrong stocks | Uses EREGL, KCHOL, SAHOL — doesn't match Phase 1 stocks (AKBNK, TUPRS, BIMAS, etc.). |
| 27 | DESIGN | — | 22+ hardcoded paths | "models/", "results/", "data/", "logs/" scattered everywhere. Should be in config. |

### app/services/daily_trading.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 28 | FATAL | 110 | Wrong import | `from data.technical_indicators import TechnicalIndicatorCalculator` — class doesn't exist. Should be `TechnicalIndicators`. |
| 29 | BUG | 529-533 | Wrong price source | Portfolio value calculation iterates decisions instead of using `current_prices` dict. |
| 30 | BUG | 246-256 | Hardcoded normalization constants | Same hardcoded `(price - 50) / 50` as trading_env.py. Undocumented magic numbers. |
| 31 | DESIGN | — | File I/O race conditions | JSON read/write without file locking for concurrent access. |

### app/main.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 32 | DEPR | 63, 70 | Deprecated startup/shutdown | `@app.on_event("startup")` deprecated since FastAPI 0.93+. Use lifespan context manager. |

### app/schemas/trading.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 33 | DESIGN | 172 | action field unvalidated | `action: str` accepts any string. Should be `Literal["BUY", "SELL", "HOLD"]` or enum. |
| 34 | DESIGN | 136 | risk_mode unvalidated | `risk_mode: str` accepts any string. Should be enum. |

### app/services/model_analysis.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 35 | DESIGN | 16 | Global warning suppression | `warnings.filterwarnings('ignore')` suppresses ALL warnings globally. |
| 36 | BUG | 93 | profit_factor returns infinity | Returns `np.inf` when no losses instead of 0 or a capped value. |

### run_server.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 37 | DESIGN | — | reload=True hardcoded | Unsafe for production. Should be configurable via env var. |

---

## 3. Frontend

### static/js/dashboard.js (1091 lines)
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 38 | DESIGN | 6-13 | 13 global variables | No encapsulation. All state is global, unlike other JS files that use classes. |
| 39 | BUG | 197-200 | References non-existent DOM elements | `progressBar`, `stepInfo`, `progressContainer` — IDs don't exist in index.html. |
| 40 | BUG | 177, 438 | showError() signature mismatch | `showError('msg', 'success')` called with 2 args but function only accepts 1. |
| 41 | BUG | 99-134 | Chart double initialization | `algorithmComparisonChart` created at startup AND in `renderComparisonChart()`. |
| 42 | BUG | 988 | Wrong currency symbol | Displays `$` instead of `₺` in comparison table. |
| 43 | SEC | multiple | XSS via innerHTML | Model names inserted via innerHTML without sanitization. |
| 44 | DESIGN | — | Chart memory leaks | Charts created but never `.destroy()`ed before re-creation. |
| 45 | DESIGN | — | Polling at 30+ req/min | `setInterval` polling during training. Should use WebSocket (Phase 3). |

### static/js/daily-trading.js (505 lines)
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 46 | BUG | 156 | Hardcoded symbols mismatch | `['ASELS', 'THYAO', 'EREGL', 'KCHOL', 'SAHOL']` — 5 stocks, but HTML has 3, backend varies. None match Phase 1. |
| 47 | BUG | 242 | Button ID mismatch | References `apply-decision-btn` but HTML has `apply-decisions`. Button never gets enabled/disabled. |
| 48 | DESIGN | 491-498 | Blocking alert() for UX | `showError()` and `showSuccess()` use `alert()` — blocks entire UI thread. |
| 49 | DESIGN | multiple | 30+ console.log in production | Debug logging left in production code. |
| 50 | DESIGN | — | No input sanitization | Balance/shares values sent to API without validation. |

### static/js/academic-analysis.js (481 lines)
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 51 | SEC | 164 | XSS via innerHTML | `${modelName}` inserted into innerHTML without escaping. |
| 52 | BUG | 51 | Infinite polling | `setTimeout(() => this.checkReportStatus(), 5000)` with no max retry. Polls forever on silent failure. |
| 53 | BUG | 470-473 | Null pointer risk | `.style.display` access without null checks (other methods check, these don't). |
| 54 | DESIGN | 204 | Meaningless chart | Portfolio comparison only has 2 points: `['Start', 'End']`. Not useful. |

### static/index.html (782 lines)
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 55 | BUG | 479-497 | Wrong stock symbols | Hardcoded inputs for EREGL, KCHOL, SAHOL — should be Phase 1 stocks (AKBNK, TUPRS, BIMAS, etc.). |

### static/css/styles.css (1122 lines)
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 56 | DESIGN | — | Duplicate table styles | Multiple near-identical table styling rules. |
| 57 | DESIGN | — | Single breakpoint | Only 1 responsive breakpoint. Limited mobile support. |

---

## 4. Tests

| # | Severity | File(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 58 | FATAL | all tests/ | No test framework | All 6 test files are standalone scripts, not pytest. |
| 59 | FATAL | all tests/ | Zero assertions | All tests use `print()` for output — nothing is verified. |
| 60 | DESIGN | all tests/ | No mocking | API tests require running server. |
| 61 | DESIGN | requirements.txt | pytest missing | Not even in dependencies. |

---

## 5. Requirements.txt

| # | Severity | Issue | Detail |
|---|----------|-------|--------|
| 62 | DESIGN | Missing dev tools | pytest, black, flake8, mypy not included. |
| 63 | DESIGN | Possibly unnecessary deps | annotated-doc, frozendict, sympy — may not be used. |

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| FATAL | 5 |
| SEC | 3 |
| BUG | 22 |
| DEPR | 3 |
| DESIGN | 30 |
| **Total** | **63** |

## Recommended Fix Priority
1. **FATAL** — Non-functional code (generate_academic_report.py, daily_trading.py import, tests)
2. **SEC** — Security issues (CORS, path traversal, XSS)
3. **BUG (RL core)** — Indicators, observation space, reward function
4. **BUG (backend)** — Thread safety, wrong stocks, deterministic eval
5. **BUG (frontend)** — DOM mismatches, symbol consistency
6. **DEPR** — Deprecated API usage
7. **DESIGN** — Architecture improvements
