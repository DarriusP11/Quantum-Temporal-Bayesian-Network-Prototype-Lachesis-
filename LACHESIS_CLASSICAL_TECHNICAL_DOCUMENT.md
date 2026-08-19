# Lachesis: Classical Financial Analytics Engine
## Technical Reference Document — Classical Subsystem Only

**Version:** 1.0 — August 2026
**Scope:** This document covers **only the classical (non-quantum) components** of the Lachesis platform — the Monte Carlo risk engine, regime forecasting, sentiment/macro stress, and insider-trading intelligence. Quantum components (QAOA, VQE, QAE, circuit simulation, gate diagnostics) are deliberately out of scope; see `LACHESIS_TECHNICAL_DOCUMENT.md` for the full platform including those.
**Purpose:** Context document for bringing an external LLM (ChatGPT) up to speed on the classical side's design and current status.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture — Classical Slice](#2-system-architecture--classical-slice)
3. [Classical Financial Risk Engine](#3-classical-financial-risk-engine)
4. [Correlation Matrix](#4-correlation-matrix)
5. [Data Sources & Fallbacks](#5-data-sources--fallbacks)
6. [QTBN Backend — Classical Markov Regime Chain](#6-qtbn-backend--classical-markov-regime-chain)
7. [Sentiment Analysis](#7-sentiment-analysis)
8. [Insider Trading Intelligence](#8-insider-trading-intelligence)
9. [Client-Side Financial Planning Tools](#9-client-side-financial-planning-tools-budgeting-retirement-credit-risk)
10. [Classical Tab Reference](#10-classical-tab-reference)
11. [Classical API Reference](#11-classical-api-reference)
12. [Mathematical Formulations](#12-mathematical-formulations)
13. [Current Status & Next Steps](#13-current-status--next-steps)
14. [Appendix: Classical Environment Setup](#14-appendix-classical-environment-setup)

---

## 1. Executive Summary

The classical subsystem is Lachesis's **production risk-analytics core**: a Monte Carlo-based financial risk engine, a classical Markov chain for regime forecasting, NLP-driven sentiment stress scoring, macro stress adjustment from Federal Reserve data, and an insider-trading intelligence pipeline sourced from SEC EDGAR. It runs entirely on standard numerical/statistical libraries (`numpy`, `pandas`, `scipy`) with no dependency on quantum simulation.

This is the layer that currently **drives real risk output** in the app. The quantum paths (QAOA portfolio optimization, VQE risk gating, QAE tail-probability estimation) are optional, experimental overlays that either wrap or substitute for pieces of this classical engine — none of them are required for the platform to function.

### What This Layer Does

- Computes **Value at Risk (VaR)** and **Conditional VaR (CVaR)** via Monte Carlo simulation over historical log-returns
- Produces **risk-adjusted return metrics**: Sharpe ratio, Sortino ratio, Max Drawdown, Annualized Volatility
- Classifies portfolios into a **volatility regime** (Low/Medium/High)
- Adjusts risk estimates for **news sentiment** (VADER NLP) and **macroeconomic conditions** (FRED unemployment + 10Y Treasury)
- Forecasts **multi-step regime evolution** via a classical Markov chain (calm → stressed → crisis)
- Surfaces **insider trading activity** per ticker from SEC EDGAR filings

---

## 2. System Architecture — Classical Slice

### 2.1 Technology Stack (classical-relevant only)

| Component | Version | Role |
|-----------|---------|------|
| FastAPI | 0.111.0+ | REST API framework |
| Uvicorn | 0.29.0+ | ASGI server (`0.0.0.0:8000`) |
| Pydantic | 2.0.0+ | Request/response validation |
| yfinance | 0.2.40+ | Live OHLCV price feeds |
| pandas | 2.0.0+ | Time series, log-return computation |
| numpy | 1.26.0+ | Monte Carlo simulation, numerical ops |
| scipy | 1.12.0+ | Statistical distributions (normal quantile/CDF) |
| fredapi | 0.5.0 | Federal Reserve macroeconomic data |
| NLTK/VADER | — | Sentiment scoring on news headlines |
| feedparser | — | Google News RSS parsing |

### 2.2 Data Flow (classical endpoints only)

```
┌──────────────────────────────────────────────────────────────┐
│                    BROWSER (React SPA)                       │
│  Financial / Sentiment / Insider / Q-TBN dashboards           │
│  └─→ lib/api.ts (typed fetch wrapper)                         │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP/JSON
┌────────────────────────▼─────────────────────────────────────┐
│                FastAPI Backend (api_server.py)               │
│                                                              │
│  /api/financial/*   → yfinance + Monte Carlo risk engine     │
│  /api/qtbn/forecast → classical Markov regime chain          │
│  /api/sentiment/*   → VADER / Perplexity NLP                 │
│  /api/insider/*     → SEC EDGAR                              │
│  /api/fred/*        → FRED macroeconomic data                │
│  /api/health        → capability/status flags                │
│                                                              │
│  Imported modules:                                           │
│  └── qtbn_core.py   (classical Markov regime engine)         │
└──────────────────────────────────────────────────────────────┘
          │              │             │
     yfinance         FRED API     SEC EDGAR
     (Yahoo)      (stlouisfed.org) (data.sec.gov)
```

### 2.3 Key Classical File Paths

```
/Applications/Quantum Temporal Bayesian Network Prototype/
├── api_server.py            # FastAPI entry point — financial + QTBN handlers
├── qtbn_core.py              # Classical Markov regime engine (77 lines)
├── sentiment_plugin.py       # Sentiment Streamlit UI (37 lines)
├── requirements.txt          # Full Python dependencies
└── requirements_api.txt      # Minimal API dependencies
```

---

## 3. Classical Financial Risk Engine

**Endpoint:** `POST /api/financial/analyze`
**Source:** `api_server.py` — functions `_monte_carlo_var_cvar()`, `_compute_risk_metrics()`

### Log-Return Computation

```
r_t = ln(P_t / P_{t-1})
```

Portfolio return (equal-weight):
```
r_portfolio(t) = (1/N) · Σ_i r_i(t)
```

### Monte Carlo VaR/CVaR

```python
# Horizon-adjusted parameters
mu_h    = mean(r) * horizon_days
sigma_h = std(r)  * sqrt(horizon_days)

# Simulate N paths
sim_returns = np.random.normal(mu_h, sigma_h, simulations)

# Value at Risk (1-day, 95% confidence)
VaR_95  = np.percentile(sim_returns, (1 - confidence) * 100)

# Conditional Value at Risk (Expected Shortfall)
CVaR_95 = sim_returns[sim_returns <= VaR_95].mean()
```

### Risk-Adjusted Return Metrics

```
Sharpe  = √252 · (μ_excess) / (σ_excess)
         where μ_excess = mean(r - r_f/252),  r_f = 4% annual

Sortino = √252 · (μ_excess) / (σ_downside)
         where σ_downside = √[ mean( min(r_excess, 0)² ) ]

MaxDD   = min_{t} [ (CumProd_t - CumMax_t) / CumMax_t ]

AnnVol  = std(r_21day_rolling) · √252
```

### Volatility Regime Classification

```python
if   ann_vol <= 0.225:  regime = "Low Volatility"    # Calm
elif ann_vol <= 0.375:  regime = "Medium Volatility"  # Stressed
else:                   regime = "High Volatility"    # Crisis
```

### Sentiment Stress

```python
# Sentiment score s ∈ [-1, +1]
multiplier = max(0.5, min(1.5, 1.0 - 0.5 * avg_sentiment_score))

# s = -1 (very bearish) → multiplier = 1.5 (escalate risk by 50%)
# s =  0 (neutral)      → multiplier = 1.0 (no change)
# s = +1 (very bullish) → multiplier = 0.5 (reduce estimated risk by 50%)

var_stressed  = var_mc  * multiplier
cvar_stressed = cvar_mc * multiplier
```

### Macro Stress (FRED API)

```python
# Fetch from FRED
unemployment_rate = fred.get_series('UNRATE').iloc[-1]   # %
treasury_10y      = fred.get_series('GS10').iloc[-1]      # %

# Construct stress factor
stress_factor = 1.0
if unemployment_rate > 6.0:   stress_factor += 0.15   # recession signal
if treasury_10y > 5.0:         stress_factor += 0.10   # high rate environment
if treasury_10y < 1.0:         stress_factor += 0.05   # deflation signal

var_macro_stressed = var_mc * stress_factor
```

---

## 4. Correlation Matrix

Pearson correlation on rolling log-returns, displayed as an interactive heatmap:

```python
corr_matrix = returns_df.corr(method='pearson')   # N×N matrix
```

---

## 5. Data Sources & Fallbacks

| Source | Type | Access | Fallback |
|--------|------|--------|---------|
| yfinance | Live OHLCV prices | `yfinance.download(tickers)` | Synthetic Gaussian random walk |
| FRED API | Macro indicators (CPI, unemployment, 10Y) | `fredapi.Fred(api_key)` | Hardcoded default values |
| Google News RSS | News headlines | `feedparser.parse(url)` | Empty sentiment (multiplier = 1.0) |
| Perplexity API | LLM-powered financial sentiment | REST API | Fallback to RSS/VADER |
| SEC EDGAR | Insider trading filings | `data.sec.gov/submissions` | Error message |

Every classical data path is designed to degrade gracefully — the app never hard-fails on a missing key, it falls back to either a synthetic/default value or a neutral (no-op) adjustment.

---

## 6. QTBN Backend — Classical Markov Regime Chain

**Endpoint:** `POST /api/qtbn/forecast`
**Source:** `qtbn_core.py`, `api_server.py`

> **Scope note:** The QTBN *backend* is a classical Markov chain — no quantum simulation involved. The frontend also ships a separate client-side TypeScript engine (`qtbn-engine.ts`) that uses quantum-inspired amplitude math purely as a UI/demo layer; that engine is out of scope for this document since it isn't part of the classical backend pipeline.

### Regime Transition Matrix (calibrated empirically)

```
T = [[0.85, 0.13, 0.02],   # calm → {calm, stressed, crisis}
     [0.25, 0.55, 0.20],   # stressed → {calm, stressed, crisis}
     [0.10, 0.25, 0.65]]   # crisis → {calm, stressed, crisis}
```

### Regime-Conditioned Volatility

```
σ(calm)    = 12%  annualized
σ(stressed) = 25% annualized
σ(crisis)  = 40% annualized
```

### 4-Bucket Outcome Probabilities (at `horizon_days`)

```python
sigma_h = vol[regime] * sqrt(horizon_days / 252)
mu_h    = drift_mu    * (horizon_days / 252)

# Normal distribution thresholds: [-2σ, -σ, σ]
P_gain        = 1 - Φ(σ_h - mu_h)
P_flat        = Φ(σ_h - mu_h) - Φ(-σ_h - mu_h)
P_loss        = Φ(-σ_h - mu_h) - Φ(-2·σ_h - mu_h)
P_severe_loss = Φ(-2·σ_h - mu_h)

# Risk-on tilt (biases toward gain/flat, away from loss)
tilt    = risk_on_prior - 0.5
P_gain        += 0.40 * tilt
P_flat        += 0.10 * tilt
P_loss        -= 0.10 * tilt
P_severe_loss -= 0.40 * tilt
```

### Temporal Evolution

```python
# Initialize prior regime distribution
p = regime_to_vector(prior_regime)   # e.g., "calm" → [0.9, 0.05, 0.05]

timeline = []
for step in range(n_steps):
    p = p @ T                         # Markov update
    drift_t    = sum(p[i] * mu_by_regime[i]      for i in range(3))
    risk_on_t  = sum(p[i] * risk_on_by_regime[i] for i in range(3))
    timeline.append({"calm": p[0], "stressed": p[1], "crisis": p[2],
                     "drift": drift_t, "risk_on": risk_on_t})
```

---

## 7. Sentiment Analysis

**Endpoint:** `POST /api/sentiment/analyze`
**Providers:** Google News RSS + NLTK VADER (default, free) or Perplexity API (requires key)

- **VADER scoring:** `-1` (very bearish) to `+1` (very bullish) per headline
- **Multiplier formula:** `max(0.5, min(1.5, 1.0 - 0.5 × avg_score))` — feeds directly into the Financial Risk Engine's sentiment stress adjustment (Section 3)
- **Outputs:** average sentiment score, VaR multiplier, headline table with per-headline scores

---

## 8. Insider Trading Intelligence

**Filings endpoint:** `POST /api/insider/load-filings` — ticker → CIK lookup → Forms 3, 4, 5
**Portfolio endpoint:** `POST /api/financial/insider` — annual return, vol, Sharpe, max drawdown per asset
**Data source:** `data.sec.gov/submissions/{CIK}.json` (24-hour cache)
**Outputs:** filing table (date, form, description, EDGAR link), asset stats table, Sharpe bar chart, equal-weight position bars

---

## 9. Client-Side Financial Planning Tools (Budgeting, Retirement, Credit Risk)

**Added by:** commit `d6f764d5`, "Add Classical section: Budgeting, Retirement, and Credit Risk tabs for college students" — this landed after the original `LACHESIS_TECHNICAL_DOCUMENT.md` was written, which is why it wasn't previously documented.

All three tools below are registered in the `CLASSICAL_TABS` array in `src/pages/Index.tsx`, each gated by `LockedTabOverlay requiredPlan="basic"` (see Section 13 for the current paywall-bypass status). **Unlike every other section in this document, all three are 100% client-side** — no `fetch`/API calls, no backend endpoint, no Python module. All computation and persistence happens in the browser.

### 9a. Budgeting

**Component:** `src/components/BudgetingDashboard.tsx` (393 lines)
**Persistence:** `localStorage` key `"lachesis_budget"` — no server-side storage, no multi-device sync
**Inputs:** income + itemized spending across 9 default categories (housing, food, transportation, entertainment, education, healthcare, clothing, subscriptions, savings) plus user-defined custom categories
**Visualization:** Recharts pie chart of spending breakdown

**Logic — 50/30/20 rule check:**
```
needs%   = (housing + food + transportation + healthcare) / income
savings% = savings / income
wants%   = totalSpend% − needs%

Compared against target: 50% needs / 30% wants / 20% savings
```
This is the only computed logic in the component; there is no forecasting or trend analysis.

### 9b. Retirement Planning

**Component:** `src/components/RetirementDashboard.tsx` (296 lines)

**Logic — `computeGrowth()` (discrete monthly compounding, simulated month-by-month, not a closed-form annuity formula):**
```
r = annualReturnPct / 100 / 12          // monthly rate

for each month over (retirementAge − currentAge) years:
    balance = balance * (1 + r) + monthlyContribution
```
Run twice to produce a "start early" comparison chart:
- **Start Now** — principal = `currentSavings`, starts at `currentAge`
- **Start at 30** — principal = 0, starts at `max(currentAge + 1, 30)`

**Derived stats:**
```
totalContributed = currentSavings + monthlyContribution * 12 * years
totalGrowth       = projectedBalance − totalContributed
growthMultiple    = projectedBalance / totalContributed
```

A static educational panel displays 2024 Roth IRA figures (annual limit $7,000, ≈$583/mo to max out, $161,000 single-filer income limit) — hardcoded, not computed.

**Scope gap to flag:** despite the "Planning" name, this only models the **accumulation phase**. There is no withdrawal/decumulation modeling, no tax-bracket treatment, and no sequence-of-returns risk (i.e. no Monte Carlo over historical return sequences) — it's a single deterministic growth projection.

### 9c. Classical Credit Risk Analysis

**Component:** `src/components/ClassicalCreditRiskDashboard.tsx` (394 lines)

> **Important distinction:** this is a *different* feature from the existing quantum "Credit Risk" tab (`CreditRiskDashboard.tsx`), which **does** hit a live backend (`POST /api/credit-risk/analyze`, backed by a Gaussian Conditional Independence Model + Iterative QAE) and is correctly out of scope for this classical-only document. The tab documented here — "Classical Credit Risk Analysis" — is a separate, purely client-side heuristic tool with no relation to that backend.

**Logic — `assessRisk()` heuristic point-scoring (not a statistical or ML model):**
```
dti      = monthlyDebt / monthlyIncome
newPayment = loanAmount / loanTermMonths           // principal-only, no interest/amortization
totalDti = (monthlyDebt + newPayment) / monthlyIncome

Point additions:
  FICO score:   <580 → +3,  <670 → +2,  <740 → +1,  else +0
  Total DTI:    >0.50 → +3,  >0.43 → +2,  >0.36 → +1,  else +0
  Employment:   unemployed → +3,  part-time → +1,  else +0

Risk level:  score ≥5 → "High",  score ≥2 → "Medium",  else "Low"
```
Also renders a static FICO tier reference table (300–850) and canned improvement tips based on threshold checks.

A "Send to Lachesis AI" action writes the assessment into React Context (`setClassicalCreditRiskSnapshot`, `src/contexts/AppContext.tsx`) so it can be referenced in the Lachesis AI chat tab — this is still client-side state, not a backend call.

---

## 10. Classical Tab Reference

| Tab | Purpose | Primary Endpoint(s) |
|-----|---------|---------------------|
| Financial Analytics | Portfolio risk analysis: VaR/CVaR, Sharpe/Sortino, correlation matrix, macro and sentiment stress | `POST /api/financial/analyze`, `POST /api/fred/macro` |
| Insider Trading | SEC EDGAR insider filing browser + per-asset portfolio analysis | `POST /api/insider/load-filings`, `POST /api/financial/insider` |
| Sentiment Analysis | News sentiment scoring and VaR stress multiplier generation | `POST /api/sentiment/analyze` |
| Q-TBN | Multi-step regime forecasting and probability distribution over portfolio outcomes | `POST /api/qtbn/forecast` |
| Present Scenarios | Curated market scenario library for structured risk analysis | `GET /api/foresight/scenarios` (references saved sweep results; classical display only) |
| Budgeting | 50/30/20 budget rule tracker with spending breakdown | None — client-side only (`localStorage`) |
| Retirement | Compound-growth retirement savings projection | None — client-side only |
| Credit Risk (classical) | Heuristic point-scoring credit risk assessment | None — client-side only |

*(Tabs 1 "Lachesis AI" and 5 "Prompt Studio" are LLM/NLP-driven but not part of the classical risk-analytics pipeline documented here; all remaining tabs are quantum-specific and excluded — including the separate, backend-driven quantum "Credit Risk" tab, see Section 9c.)*

---

## 11. Classical API Reference

### Financial Analytics (5 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/financial/analyze` | VaR/CVaR/Sharpe (classical Monte Carlo) |
| POST | `/api/financial/insider` | Per-asset stats + equal-weight VaR |
| POST | `/api/financial/extract-screenshot` | Vision → JSON portfolio (classical data extraction, LLM-backed) |
| POST | `/api/financial/lachesis-guide` | AI risk narrative (rule-based fallback) |
| POST | `/api/fred/macro` | FRED CPI + unemployment + 10Y |

### QTBN (1 endpoint)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/qtbn/forecast` | Markov regime evolution + outcome probabilities |

### Sentiment (1 endpoint)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/sentiment/analyze` | News sentiment scoring |

### Insider Trading (3 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/insider/load-filings` | SEC EDGAR Forms 3/4/5 |
| POST | `/api/insider/lookup-cik` | Ticker → CIK (legacy) |
| POST | `/api/insider/filings` | CIK → filings (legacy) |

### Health / Admin (2 endpoints)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | System status + capability flags |
| POST | `/api/admin/validate-key` | API key format check |

---

## 12. Mathematical Formulations

### 12.1 VaR and CVaR (Monte Carlo, classical)

```
Given portfolio returns R ~ N(μ_h, σ_h²):

VaR_α  = Φ⁻¹(1−α) · σ_h + μ_h        [classical quantile]
CVaR_α = μ_h − σ_h · φ(Φ⁻¹(α)) / α   [classical conditional expectation]
```

where `Φ` is the standard normal CDF and `φ` is the PDF.

### 12.2 Sharpe and Sortino Ratios

```
Sharpe  = √252 · (μ_excess) / σ_excess
Sortino = √252 · (μ_excess) / σ_downside

where:
  μ_excess    = mean(r_t − r_f/252)
  σ_excess    = std(r_t  − r_f/252)
  σ_downside  = √[ mean( (min(r_excess, 0))² ) ]
  r_f         = 0.04  (4% annual risk-free rate)
```

### 12.3 QTBN Regime Evolution

```
State vector:  p(t) ∈ Δ²  [probability simplex over 3 regimes]
Update:        p(t+1) = p(t) @ T

T = [[0.85, 0.13, 0.02],
     [0.25, 0.55, 0.20],
     [0.10, 0.25, 0.65]]

Expected drift at step t:
μ(t) = Σ_i p_i(t) · μ_regime[i]
```

---

## 13. Current Status & Next Steps

*(Descriptive status as of this writing — check specifics against your own notes before sending this to ChatGPT, since deployment state can drift.)*

### Deployed & Working

- Monte Carlo VaR/CVaR, Sharpe/Sortino/MaxDD, and correlation heatmap are live and computing off real `yfinance` data
- QTBN Markov regime forecasting is functioning end-to-end (transition matrix → multi-step timeline → outcome probabilities)
- Sentiment scoring via VADER/Google News RSS is live and feeding the VaR stress multiplier
- Insider trading intelligence (SEC EDGAR) is live for filing lookups and per-asset stats
- Backend deployed on Railway, frontend on Vercel, both pointing at the consolidated Supabase project (`mvprtzaatfbvdxwutrbo`)

### Known Gaps / Temporary States

- **FRED macro stress** falls back to hardcoded default values whenever `FRED_API_KEY` isn't set — not currently guaranteed to reflect live macro conditions
- **Sentiment** defaults to the free VADER/RSS path; the higher-quality Perplexity path requires an API key and isn't always active
- **Paywall gate is temporarily bypassed** for testing (all features unlocked) per a recent commit — this needs to be re-enabled before a real launch, and is a reminder that access control isn't currently enforced in production
- A `/health` endpoint and frontend API status badge were recently added specifically to surface backend connectivity issues (e.g. "Failed to fetch" errors) — indicates reliability/observability work is an active area, not yet considered fully solved
- **Budgeting, Retirement, and Classical Credit Risk (Section 9) are recently added and still UI-only:** no backend persistence (all state lives in `localStorage` or React Context — cleared cache or a new device means lost data), no cross-device sync, and Retirement in particular lacks any withdrawal/decumulation modeling despite the "Planning" name. They ship behind the same `requiredPlan="basic"` gate as the rest of the paid tier, currently bypassed for testing along with everything else noted above.

---

## 14. Appendix: Classical Environment Setup

### Local Development

```bash
cd "/Applications/Quantum Temporal Bayesian Network Prototype"
pip3 install -r requirements.txt
uvicorn api_server:app --reload --port 8000
```

### Required Environment Variables (classical-relevant)

```
SUPABASE_URL=https://mvprtzaatfbvdxwutrbo.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service_role_jwt>
FRED_API_KEY=...            (optional — fallback to hardcoded macro values)
```

Lachesis detects optional keys at startup and exposes capability flags via `GET /api/health`.

---

*Document generated August 2026 as a classical-subsystem-only companion to `LACHESIS_TECHNICAL_DOCUMENT.md`. Intended for use as external-LLM context, not as a substitute for the full technical reference.*
