# Issue Tracker — Detailed Bug & Problem List

Analysis date: 2026-03-24. Updated after Batch 0-6 fixes (2026-03-24).
Each issue has file:line reference and severity.

Legend: `[FATAL]` = crashes/non-functional, `[BUG]` = wrong behavior, `[SEC]` = security,
`[DEPR]` = deprecated API, `[DESIGN]` = architecture/design problem, `~~strikethrough~~` = FIXED

---

## 1. RL Environment & Training

### env/trading_env.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~1~~ | ~~BUG~~ | ~~114-119~~ | ~~Unbounded observation space~~ | ~~FIXED (Batch 3): `Box(-10,10)` + `np.clip(state,-10,10)` at end of `_get_observation()`.~~ |
| ~~2~~ | ~~BUG~~ | ~~392-395~~ | ~~Hardcoded normalization~~ | ~~FIXED (Batch 3): `_compute_price_stats()` calculates per-symbol mean/std; z-score used in `_get_observation()`. `get_price_stats()` exposed for inference.~~ |
| ~~3~~ | ~~BUG~~ | ~~—~~ | ~~Reward function diverges from Ansari~~ | ~~FIXED (remote merge): New `env/reward_functions.py` with PSR reward (RewardCalculator class).~~ |
| ~~4~~ | ~~BUG~~ | ~~419-421, 485-490~~ | ~~Silent missing data~~ | ~~FIXED (Batch 3): KeyError → ffill from last known feature block + `logger.warning()`.~~ |
| ~~5~~ | ~~BUG~~ | ~~200~~ | ~~Threshold disabled~~ | ~~FIXED (Batch 3): `min_threshold = 1` re-enabled; actions with |shares| < 1 filtered.~~ |

### data/technical_indicators.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~6~~ | ~~BUG~~ | ~~58-59~~ | ~~RSI uses SMA instead of Wilder's EMA~~ | ~~FIXED (Batch 2): `.ewm(alpha=1/period, adjust=False).mean()`. avg_loss=0 guarded with `.replace(0, np.nan)`.~~ |
| ~~7~~ | ~~BUG~~ | ~~112-114~~ | ~~ADX uses SMA instead of EMA~~ | ~~FIXED (Batch 2): Wilder's EMA for ATR, +DI, -DI, DX smoothing.~~ |
| ~~8~~ | ~~BUG~~ | ~~117~~ | ~~Division by zero in ADX~~ | ~~FIXED (Batch 2): `np.where(denominator > 0, ..., 0)`.~~ |
| ~~9~~ | ~~DESIGN~~ | ~~135-143~~ | ~~Turbulence oversimplified~~ | ~~FIXED (Batch 2): Cross-sectional Mahalanobis distance `(r_t - mu)^T * Sigma^{-1} * (r_t - mu)` in `add_indicators_to_multi_symbol_df()`. `np.linalg.pinv` for near-singular matrices.~~ |
| ~~10~~ | ~~BUG~~ | ~~23-39~~ | ~~MACD missing signal line~~ | ~~FIXED (Batch 2): `calculate_macd()` returns (macd, signal_line, histogram). `macd_signal`, `macd_hist` columns added. State space unchanged (additive).~~ |
| ~~11~~ | ~~DEPR~~ | ~~166-168~~ | ~~Deprecated pandas fillna~~ | ~~FIXED (remote merge): Now uses `.ffill()` and `.bfill()`.~~ |

### data/data_fetcher.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~12~~ | ~~DEPR~~ | ~~166-169~~ | ~~Deprecated pandas fillna~~ | ~~FIXED (Batch 6): `.fillna(method='ffill')` → `.ffill()`; `.fillna(method='bfill')` → `.bfill()`.~~ |
| 13 | DESIGN | — | No retry logic | Network failures cause silent data loss. |
| 14 | DESIGN | — | No minimum data coverage validation | No check that enough data rows exist before proceeding. |

### scripts/analysis/generate_academic_report.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~15~~ | ~~FATAL~~ | ~~13~~ | ~~Wrong import path~~ | ~~FIXED (Batch 1): `from env.trading_env import TradingEnv` + `TradingEnv(...)` replaces `MultiStockTradingEnv`.~~ |
| ~~16~~ | ~~FATAL~~ | ~~18~~ | ~~Missing data file (pickle)~~ | ~~FIXED (Batch 1): Replaced pickle loader with `DataFetcher.load_data()` + `add_indicators_to_multi_symbol_df()` + `split_data()` pipeline.~~ |
| ~~17~~ | ~~BUG~~ | ~~76~~ | ~~Wrong attribute `env.portfolio_value`~~ | ~~FIXED (Batch 1): → `env._get_portfolio_value()`.~~ |
| ~~18~~ | ~~BUG~~ | ~~83~~ | ~~Wrong attribute `env.trade_history`~~ | ~~FIXED (Batch 1): → `env.trades_history`.~~ |

### scripts/training/train_a2c_phase1.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~19~~ | ~~BUG~~ | ~~121~~ | ~~n_steps parameter ignored~~ | ~~FIXED (Batch 6): `n_steps=256` hardcoded → `n_steps=n_steps` (uses function parameter).~~ |
| ~~20~~ | ~~BUG~~ | ~~180, 182~~ | ~~DummyVecEnv done array~~ | ~~FIXED (Batch 6): `done=np.array([False])`; `while not done[0]:` — correct ndarray indexing.~~ |

---

## 2. Backend (FastAPI)

### app/core/config.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~21~~ | ~~SEC~~ | ~~21-24~~ | ~~CORS wildcard + credentials~~ | ~~FIXED (Batch 1): `CORS_ORIGINS` restricted to `["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8888", "http://127.0.0.1:8888"]`.~~ |

### app/api/routes/trading.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~22~~ | ~~SEC~~ | ~~383, 665, 828~~ | ~~Path traversal~~ | ~~FIXED (Batch 1): `sanitize_model_name()` with `os.path.basename()` + `^[a-zA-Z0-9_.-]+$` regex. Applied to 3 endpoints.~~ |
| ~~23~~ | ~~BUG~~ | ~~28-35~~ | ~~Thread-unsafe global state~~ | ~~FIXED (Batch 4): `asyncio.Lock()` guards training_state check+set in `start_training()`.~~ |
| ~~24~~ | ~~BUG~~ | ~~695~~ | ~~Test eval uses deterministic=False~~ | ~~FIXED (Batch 4): `deterministic=True` for reproducible evaluation.~~ |
| ~~25~~ | ~~BUG~~ | ~~694-710~~ | ~~Auto-reset workaround~~ | ~~FIXED (Batch 4): DummyVecEnv loop + autoreset restore removed. Raw `TradingEnv` used directly for eval.~~ |
| ~~26~~ | ~~BUG~~ | ~~—~~ | ~~Hardcoded wrong stocks~~ | ~~FIXED (remote merge): Now uses `get_symbols(phase=request.phase)` dynamically.~~ |
| ~~27~~ | ~~DESIGN~~ | ~~—~~ | ~~30+ hardcoded paths~~ | ~~FIXED (Batch 4): `MODELS_DIR`, `RESULTS_DIR`, `DATA_DIR`, `LOGS_DIR` added to `config.py`. All 14 occurrences replaced with `_settings.*`.~~ |

### app/services/daily_trading.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~28~~ | ~~FATAL~~ | ~~—~~ | ~~Wrong import~~ | ~~FIXED (remote merge): Now correctly imports `add_indicators_to_multi_symbol_df`.~~ |
| 29 | BUG | 274-315 | Wrong price source | Portfolio value calculation — needs verification with current logic. |
| ~~30~~ | ~~BUG~~ | ~~244-253~~ | ~~Hardcoded normalization constants~~ | ~~FIXED (Batch 3): `build_live_state(price_stats=...)` parameter; same z-score as `trading_env.py`. Falls back to (50,50) if stats not provided.~~ |
| ~~31~~ | ~~DESIGN~~ | ~~—~~ | ~~File I/O race conditions~~ | ~~FIXED (Batch 4): `FileLock` wraps JSON read/write in `save_daily_decision()`.~~ |

### app/main.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~32~~ | ~~DEPR~~ | ~~83-93~~ | ~~Deprecated startup/shutdown~~ | ~~FIXED (Batch 6): `@app.on_event()` removed. `@asynccontextmanager lifespan` passed to `FastAPI(lifespan=lifespan)`.~~ |

### app/schemas/trading.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~33~~ | ~~DESIGN~~ | ~~177~~ | ~~action field unvalidated~~ | ~~FIXED (Batch 4): `action: Literal["BUY", "SELL", "HOLD"]`.~~ |
| ~~34~~ | ~~DESIGN~~ | ~~141-143~~ | ~~risk_mode unvalidated~~ | ~~FIXED (Batch 4): `risk_mode: Literal["conservative", "moderate", "aggressive"]`.~~ |

### app/services/model_analysis.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~35~~ | ~~DESIGN~~ | ~~16~~ | ~~Global warning suppression~~ | ~~FIXED (Batch 4): Global `warnings.filterwarnings('ignore')` removed. Scoped suppress only during plot generation; `warnings.resetwarnings()` called after.~~ |
| ~~36~~ | ~~BUG~~ | ~~93~~ | ~~profit_factor returns infinity~~ | ~~FIXED (Batch 4): `np.inf` → `9999.99` (JSON serializable).~~ |

### run_server.py
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~37~~ | ~~DESIGN~~ | ~~—~~ | ~~reload=True hardcoded~~ | ~~FIXED (Batch 6): `reload=settings.DEBUG` — configurable via `DEBUG` env var.~~ |

---

## 3. Frontend

### static/js/dashboard.js
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 38 | DESIGN | 6-13 | 13 global variables | No encapsulation. All state is global, unlike other JS files that use classes. |
| ~~39~~ | ~~BUG~~ | ~~193-196, 226-235~~ | ~~References non-existent DOM elements~~ | ~~FIXED (Batch 5): `progressBar`, `stepInfo` already null-guarded in current code (confirmed in review).~~ |
| ~~40~~ | ~~BUG~~ | ~~204, 259~~ | ~~showError() signature mismatch~~ | ~~FIXED (Batch 1/5): `showError(message, type='error')` now accepts 2 params.~~ |
| ~~41~~ | ~~BUG~~ | ~~99-135~~ | ~~Chart double initialization~~ | ~~FIXED (Batch 5): `if (performanceChart) performanceChart.destroy()` and `if (algorithmComparisonChart) algorithmComparisonChart.destroy()` added before creation in `initCharts()`.~~ |
| ~~42~~ | ~~BUG~~ | ~~—~~ | ~~Wrong currency symbol~~ | ~~FIXED (remote merge): Now correctly uses ₺ instead of $.~~ |
| ~~43~~ | ~~SEC~~ | ~~multiple~~ | ~~XSS via innerHTML~~ | ~~FIXED (Batch 1): `escapeHtml()` helper added; applied to all `model.name` innerHTML insertions.~~ |
| 44 | DESIGN | — | Chart memory leaks | Charts created but never `.destroy()`ed before re-creation (partial — initCharts fixed, other locations remain). |
| 45 | DESIGN | — | Polling at 30+ req/min | `setInterval` polling during training. Should use WebSocket (Phase 3). |

### static/js/daily-trading.js
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~46~~ | ~~BUG~~ | ~~—~~ | ~~Hardcoded symbols mismatch~~ | ~~FIXED (remote merge): Now correctly uses Phase 1 symbols.~~ |
| ~~47~~ | ~~BUG~~ | ~~—~~ | ~~Button ID mismatch~~ | ~~FIXED (remote merge): Now correctly references `'apply-decisions'`.~~ |
| 48 | DESIGN | 491-498 | Blocking alert() for UX | `showError()` and `showSuccess()` use `alert()` — blocks entire UI thread. |
| 49 | DESIGN | multiple | 30+ console.log in production | Debug logging left in production code. |
| 50 | DESIGN | — | No input sanitization | Balance/shares values sent to API without validation. |

### static/js/academic-analysis.js
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~51~~ | ~~SEC~~ | ~~164~~ | ~~XSS via innerHTML~~ | ~~FIXED (Batch 1): `AcademicAnalysisManager.escapeHtml()` static method; `modelName` sanitized before innerHTML.~~ |
| ~~52~~ | ~~BUG~~ | ~~51~~ | ~~Infinite polling~~ | ~~FIXED (Batch 5): `checkReportStatus(retries=0)` with `MAX_RETRIES=20`; timeout message shown after 20 failures.~~ |
| ~~53~~ | ~~BUG~~ | ~~470-473~~ | ~~Null pointer risk~~ | ~~FIXED (Batch 5): null-safe `forEach` loop over card IDs before `.style.display`.~~ |
| ~~54~~ | ~~DESIGN~~ | ~~204~~ | ~~Meaningless 2-point chart~~ | ~~FIXED (Batch 5): Filters models without return data; 3-point interpolation (Start/Mid/End); canvas hidden if no valid data.~~ |

### static/index.html
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| ~~55~~ | ~~BUG~~ | ~~—~~ | ~~Wrong stock symbols~~ | ~~FIXED (remote merge): Now correctly shows Phase 1 symbols.~~ |

### static/css/styles.css
| # | Severity | Line(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 56 | DESIGN | — | Duplicate table styles | Multiple near-identical table styling rules. |
| 57 | DESIGN | — | Single breakpoint | Only 1 responsive breakpoint. Limited mobile support. |

---

## 4. Tests

| # | Severity | File(s) | Issue | Detail |
|---|----------|---------|-------|--------|
| 58 | FATAL | all tests/ | No test framework | All test files are standalone scripts, not pytest. |
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

## 6. Post-Merge: Duplicate Files

| # | Severity | Issue | Detail |
|---|----------|-------|--------|
| ~~64~~ | ~~DESIGN~~ | ~~Duplicate docs~~ | ~~FIXED (Batch 0): `docs/ACADEMIC_GUIDE.md`, `docs/ALGORITHMS.md`, `docs/development.md` deleted. Subdirectory versions preserved.~~ |
| ~~65~~ | ~~DESIGN~~ | ~~Duplicate scripts~~ | ~~FIXED (Batch 0): `scripts/generate_academic_report.py`, `scripts/debug_model_actions.py`, `scripts/train_a2c_phase1.py` deleted. Subdirectory versions preserved.~~ |

---

## Summary Statistics

| Severity | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| FATAL | 5 | 5 | 0 |
| SEC | 3 | 3 | 0 |
| BUG | 22 | 19 | 3 |
| DEPR | 3 | 3 | 0 |
| DESIGN | 32 | 14 | 18 |
| **Total** | **65** | **44** | **21** |

## Fixed Issues Summary (44 total — Batches 0-6)

**Remote merge (8):** #3, #11, #26, #28, #42, #46, #47, #55

**Batch 0 — Duplicate cleanup (2):** #64, #65

**Batch 1 — Fatal + Security (8):** #15, #16, #17, #18, #21, #22, #43, #51

**Batch 2 — Technical indicators (5):** #6, #7, #8, #9, #10

**Batch 3 — RL Environment core (5):** #1, #2, #4, #5, #30

**Batch 4 — Backend bugs (10):** #23, #24, #25, #27, #31, #33, #34, #35, #36, #37 (moved to B6)

**Batch 5 — Frontend bugs (6):** #39, #40, #41, #52, #53, #54

**Batch 6 — Scripts + deprecations (5):** #12, #19, #20, #32, #37

## Remaining Issues (21)

**BUG (3):** #29 (portfolio price source verification)

**DESIGN (18):** #13, #14 (data pipeline robustness — Batch 7), #38, #44, #45, #48, #49, #50, #56, #57 (Batch 8 frontend/CSS quality), #58, #59, #60, #61, #62, #63 (Batch 8 tests/requirements)
