# Seminar Presentation — Page-by-Page Outline

> **Scope:** System architecture + key definitions (technical indicators, finance terms) + data sources.
> Not a full project walkthrough. Audience = general/academic, not RL specialists.
> Target: ~14 slides, ~15–20 min.

---

## Slide 1 — Title

- **Title:** Deep Reinforcement Learning-Based Algorithmic Trading System for BIST-30
- Subtitle: Multi-source data → ensemble prediction → RL decision agent
- Your name, advisor, date, institution.
- *Speaker note (15s):* one line — "I built a system that decides buy/sell/hold for Turkish stocks using ML + RL."

---

## Slide 2 — The Problem (1 sentence + 1 visual)

- **Question:** Given price history + macro + fundamentals, can an agent learn profitable buy/sell/hold decisions?
- Why hard: markets are noisy, non-stationary, BIST is an emerging market (less studied).
- One chart: a BIST-30 price line with "?" markers — frame the decision problem.
- *Keep it conceptual. No formulas yet.*

---

## Slide 3 — System Architecture (the core slide)

- **One diagram, full pipeline top→bottom. Same diagram as `seminar-overview.md` §4 — module names included so it doubles as a code map:**

```
                 ┌──────────────── DATA LAYER ──────────────────┐
yfinance  ───────│ OHLCV (data_fetcher.py)                       │
TCMB EVDS ───────│ Macro: rate, inflation (macro_fetcher.py)     │
yfinance  ───────│ FX, BIST100, VIX, US10Y, DXY                  │
yfinance  ───────│ Fundamental: ROE/ROA/PE/PB... (fund_fetcher)  │
borsapy/yf───────│ Gold / FX (gold_fetcher.py)                   │
                 └───────────────────┬──────────────────────────┘
                                     ▼
            feature_engineer.py  (10 feature groups, ICEEMDAN, ≥1-day lag)
                                     ▼
            feature_selector.py  (MI + permutation importance, 3-stage)
                                     ▼
   ┌─────────────────── PREDICTION LAYER ──────────────────┐
   │ XGBoost  LightGBM  CatBoost  BiLSTM  TFT              │
   │            │ (walk-forward + purge gap + embargo)     │
   │            ▼                                          │
   │     Stacking ensemble (Ridge / XGB meta, 3-way split) │
   │            ▼  TATS corrector                          │
   │     Output: predicted_return, direction, confidence,  │
   │            ensemble_agreement   + SHAP explanation     │
   └────────────────────┬──────────────────────────────────┘
                        ▼  (+4 features per symbol)
   ┌─────────────────── RL LAYER ──────────────────────────┐
   │ env/trading_env.py  (Gymnasium, multi-stock)          │
   │   state: balance + holdings + OHLCV + indicators + pred│
   │   reward: PSR (Ansari Eq. 1)                           │
   │   risk:  ATR sizing + Kelly criterion (opt-in)        │
   │   agent: Stable-Baselines3 — A2C / PPO / TD3          │
   └────────────────────┬──────────────────────────────────┘
                        ▼
   evaluator.py: Sharpe, Sortino, Calmar, Deflated Sharpe, PF, IC, Turnover, MaxDD
```

- Spend the most time here. Everything later is a zoom-in on one box.
- *Speaker note:* "Five data sources → feature engineering → 5-model ensemble → RL agent → evaluation."
- *If the audience is non-technical, hide the filenames (`*.py`) and keep just the layer labels.*

---

## Slide 4 — Data Sources (overview table)

| Category | Source | Examples |
|---|---|---|
| **Price (OHLCV)** | yfinance | Open, High, Low, Close, Volume |
| **Macro** | TCMB EVDS + yfinance | policy rate, CPI/PPI inflation, USD/TRY, EUR/TRY, BIST-100 |
| **Global macro** | yfinance | VIX, US 10Y yield, DXY, WTI oil |
| **Fundamental** | yfinance | ROE, ROA, P/E, P/B, Debt/Equity, Current Ratio, Profit Margin |

- *Note for audience:* TCMB EVDS = Turkish Central Bank's official economic data API.

---

## Slide 5 — Definition: OHLCV (the raw data)

- **OHLCV** = the 5 numbers describing one trading day for one stock:
  - **Open** — first traded price of the day
  - **High** — highest price reached
  - **Low** — lowest price reached
  - **Close** — last traded price (most important)
  - **Volume** — number of shares traded
- Foundation for everything else. All indicators are computed *from* these.

---

## Slide 6 — Definition: Technical Indicators (intro)

- **What they are:** math transforms of price/volume that summarize trend, momentum, and volatility into a single number.
- **Why:** raw price is hard to act on; indicators turn it into signals.
- This project uses 5 (from Ansari et al. 2024). Next slide defines each in plain language.

---

## Slide 7 — The 5 Technical Indicators (definitions)

| Indicator | Measures | Plain-language meaning |
|---|---|---|
| **MACD** | Trend / momentum | Difference of two moving averages — is the trend speeding up or slowing? (12/26/9) |
| **RSI** | Momentum | 0–100 scale — >70 "overbought", <30 "oversold" (14-day) |
| **CCI** | Deviation | How far price is from its average — spots extremes (20-day) |
| **ADX** | Trend strength | 0–100 — *how strong* a trend is (not direction) (14-day) |
| **Turbulence** | Market stress | Cross-sectional risk index — high = abnormal/crisis market |

- *Speaker note:* mention Turbulence is computed across all stocks (Mahalanobis distance), the rest per-stock.
- Optional: 1 small chart of RSI under a price line to make it concrete.

---

## Slide 8 — Definition: Macro & Fundamental Terms

- **Macro (the economy):**
  - *Policy rate* — central bank interest rate; drives borrowing cost.
  - *CPI / PPI inflation* — consumer / producer price increases.
  - *USD/TRY, EUR/TRY* — exchange rates (critical for Turkey).
  - *VIX* — US "fear index"; market volatility expectation.
  - *DXY / US10Y* — dollar strength / US bond yield (global risk appetite).
- **Fundamental (the company):**
  - *P/E* — price vs. earnings (valuation).
  - *P/B* — price vs. book value.
  - *ROE / ROA* — profitability vs. equity / assets.
  - *Debt/Equity* — leverage; *Current Ratio* — short-term liquidity.

---

## Slide 9 — Prediction Layer (zoom into box 2)

- **5 models trained in parallel, combined by a meta-learner (stacking ensemble):**
  - Tree models: **XGBoost, LightGBM, CatBoost** (strong on tabular data)
  - Deep models: **BiLSTM, TFT** (capture time-series patterns)
  - **Ensemble** learns how much to trust each model.
- **Output per stock:** predicted return + direction (up/down) + confidence.
- *Keep model internals shallow — this is an overview, not the ML deep-dive.*

---

## Slide 10 — RL Decision Layer (zoom into box 3)

- **Reinforcement Learning in one line:** an agent learns by trial-and-error to maximize reward.
- Map the terms to trading:
  - **State** — what the agent sees (prices, indicators, predictions, current holdings)
  - **Action** — BUY / SELL / HOLD
  - **Reward** — risk-adjusted profit (Probabilistic Sharpe Ratio)
- Algorithms tried: **A2C, PPO, TD3** (Stable-Baselines3). One sentence each is enough.

---

## Slide 11 — Definition: Finance Metrics (how we judge it)

| Metric | What it answers |
|---|---|
| **Sharpe Ratio** | Return per unit of risk |
| **PSR** (Probabilistic Sharpe) | Is the Sharpe statistically real, not luck? |
| **Sortino** | Like Sharpe but penalizes only *downside* risk |
| **Calmar** | Return vs. worst drawdown |
| **Max Drawdown** | Biggest peak-to-trough loss |
| **Profit Factor** | Total gains ÷ total losses |

- *Pick 3–4 if short on time; Sharpe + Max Drawdown + PSR are the must-haves.*

---

## Slide 12 — End-to-End Data Flow (recap diagram)

- **Simplified version of the slide-3 diagram — no filenames, just the flow:**

```
yfinance / EVDS  ──► data layer ──► feature engineering
                                      │
                                      ▼
                    5-model ensemble (prediction)
                                      │
                                      ▼
                    RL agent (state → BUY/SELL/HOLD)
                                      │
                                      ▼
                    evaluation: Sharpe / PSR / Drawdown
```

- Ties slides 3–11 together. Same colors as slide 3 so it reads as "the full picture now."
- *Trick: show the detailed slide-3 diagram here again but animated/highlighted — proves you've now explained every box.*

---

## Slide 13 — Current Status & Scope

- Phases 1–3 complete: working single-agent baseline + dashboard.
- Honest framing: this is the *foundation*; thesis extends it (multi-agent, per-sector).
- Keep brief — seminar is about understanding the system, not selling results.

---

## Slide 14 — Summary / Q&A

- 3 takeaways:
  1. **3-layer architecture:** data → prediction → decision.
  2. **Multi-source data:** price + technical + macro + fundamental.
  3. **RL agent** learns risk-adjusted trading decisions.
- "Questions?"

---

## Delivery Notes

- **Definitions slides (5–8, 11) are the heart of this talk** — the audience wants vocabulary, not RL theory.
- Use the *same* 3-layer diagram on slides 3 and 12 (anchor + recap).
- One chart max per slide. Prefer a labeled price chart over equations.
- If time is tight, drop slides 9–10 to one combined slide; never drop the definitions.
- Have backup detail slides (formulas for RSI/MACD/Sharpe) *after* slide 14 for Q&A only.
</content>
</invoke>
