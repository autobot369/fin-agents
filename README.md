# fin-agents

A collection of independent multi-agent systems (MAS) for financial research, ETF ranking, and automated SIP execution — built on LangGraph with Claude and Gemini.

---

## Systems

| System | LLM | Purpose |
|--------|-----|---------|
| [ticker-research-mas](#ticker-research-mas) | Claude Opus 4.6 | Bull vs Bear debate + arbitrated stock forecast |
| [etf-selection-mas](#etf-selection-mas) | Claude Sonnet 4.6 | Global ETF Top-20 ranking across 3 markets |
| [sip-execution-mas](#sip-execution-mas) | Gemini 2.0 Flash | Monthly SIP execution with risk auditing |

---

## ticker-research-mas

A 7-node LangGraph pipeline that researches any stock ticker, runs a structured Bull vs Bear debate, fact-checks all claims, and delivers an arbitrated investment forecast.

**Pipeline:**
```
researcher → bull → bear → bull_rebuttal → bear_counter → fact_checker → arbiter
```

**Data sources:** yfinance · DuckDuckGo · SEC EDGAR (10-Q/10-K)

**Usage:**
```bash
cd ticker-research-mas
pip install -r requirements.txt

python main.py AAPL
python main.py TSLA --horizon short --risk 0.9 --save
python main.py NVDA --horizon long --risk 0.1
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--horizon` | `mid` | `short` / `mid` / `long` |
| `--risk` | `0.5` | Risk aversity 0.0 (seeking) → 1.0 (averse) |
| `--save` | off | Save Markdown report to `outputs/` |
| `--verbose` | off | Print intermediate debate rounds |

**How it works:**
1. **Researcher** — Builds an immutable `ground_truth` dict (72 fields: price, RSI, MACD, Bollinger Bands, futures basis, earnings sentiment, SEC risk factors)
2. **Bull / Bear** — Two rounds of structured debate; all numerical claims must reference `ground_truth`
3. **Fact Checker** — Cross-validates every claim against `ground_truth`; flags unverified assertions
4. **Arbiter** — Scores Bull vs Bear using futures basis, volume trend, and a 10-point rubric; delivers `BUY / HOLD / SELL` with conviction level

**Env vars:** `ANTHROPIC_API_KEY`

---

## etf-selection-mas

A 5-node LangGraph pipeline that scores and ranks 31 ETFs across three global markets, producing a "Top-20 Boom List" with sentiment-driven ranking.

**Pipeline:**
```
MultiMarketResearcher → ExpenseGuard → SentimentScorer → GlobalRankingNode → PortfolioArbiter
```

**ETF Universe (31 total):**

| Market | Exchange | Count | Example Tickers |
|--------|----------|-------|----------------|
| INTL | NASDAQ | 11 | VXUS, SPDW, IEFA, EEM, ACWI |
| HKCN | NASDAQ | 10 | CQQQ, KWEB, MCHI, FXI, CHIQ |
| NSE | India (.NS) | 10 | NIFTYBEES, BANKBEES, GOLDBEES |

**Usage:**
```bash
cd etf-selection-mas
pip install -r requirements.txt

python main.py                     # Top-20, TER ≤ 0.70%
python main.py --ter 0.50          # Stricter TER ceiling
python main.py --top 10            # Top-10 instead of 20
```

**Scoring formula:**
```
expense_score  = max(0, 1 − TER / ter_threshold)
consensus_score = 0.60 × sentiment_score + 0.40 × expense_score
```

**Boom triggers** (9 total): `china_tech_rerating` (+0.35), `rbi_pivot` (+0.30), `fed_soft_landing` (+0.25), `india_inflation_target` (+0.25), `china_gdp_beat` (+0.25), `em_broad_rally` (+0.15), `china_tech_crackdown` (−0.20), `india_fiscal_slippage` (−0.15), `us_recession_risk` (−0.15)

**Outputs:**
- `outputs/TOP_20_BOOM_REPORT.md` — Ranked report with rationale per ETF
- `outputs/rankings.json` — Machine-readable rankings (consumed by sip-execution-mas)

**Env vars:** `ANTHROPIC_API_KEY`

---

## sip-execution-mas

A 6-node LangGraph execution loop that dynamically discovers ETF candidates via Gemini, scores sentiment, allocates a monthly SIP budget with core/satellite bucketing, audits against hard risk rules, and executes orders (paper or live).

**Pipeline:**
```
researcher → scorer → optimizer → auditor
                                      ↓
                          approved → broker → logger
                          retry    → optimizer (re-runs with tighter caps)
                          abort    → logger (exits after 2 retries)
```

**Usage:**
```bash
cd sip_execution_mas
pip install -r requirements.txt

python -m sip_execution_mas                              # Dry-run, $500 SIP
python -m sip_execution_mas --sip 750 --top 12 --core 6
python -m sip_execution_mas --ter 0.30                   # Strict TER filter
python -m sip_execution_mas --live                       # Live Alpaca trading
python -m sip_execution_mas --force                      # Bypass same-month rule
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sip` | `500` | Monthly SIP budget in USD |
| `--top` | `10` | Number of ETFs to select |
| `--core` | `5` | Number of core (low-risk) ETFs |
| `--ter` | `0.70` | Max TER % ceiling |
| `--max-position` | `0.25` | Max single-ETF allocation (fraction) |
| `--max-region` | `0.50` | Max single-region allocation (fraction) |
| `--live` | off | Enable live Alpaca order submission |
| `--force` | off | Bypass same-month duplicate check |

**Allocation logic:**
```
core_budget      = sip_amount × core_pct
satellite_budget = sip_amount × (1 − core_pct)
weight_i         = consensus_score_i / Σ scores_in_bucket
allocation_i     = weight_i × bucket_budget
```

**5 hard risk rules (auditor):**
1. No single ETF > `max_position_pct × SIP`
2. No single region > `max_region_pct × SIP`
3. No ETF allocation < $1.00
4. No duplicate entry for current month (bypass with `--force`)
5. At least 1 proposed order

**Execution modes:**

| Mode | Condition | Behavior |
|------|-----------|----------|
| Paper | default | Simulates fills at yfinance close price |
| Alpaca | `--live` + `ALPACA_API_KEY` set | Real MarketOrderRequest (US ETFs only) |
| Dhan | `.NS` / `.BO` tickers | Always paper (stub) |

**Outputs:**
- `outputs/execution_log.csv` — Per-order audit trail
- `simulator/outputs/portfolio_ledger.json` — Cumulative portfolio state

**Simulator sub-package:**

```bash
# Backtest historical SIP
python -m simulator.main --sip 500 --months 24 --top 10

# Monthly scheduler (runs 1st of each month at 09:00)
python -m simulator.scheduler
python -m simulator.scheduler --now      # Immediate one-off run

# Interactive portfolio dashboard
streamlit run sip_execution_mas/simulator/app.py
```

**Env vars:** `GEMINI_API_KEY` (required) · `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` (optional, live trading) · `DHAN_CLIENT_ID` + `DHAN_ACCESS_TOKEN` (optional, future)

---

## Setup

### Prerequisites
- Python 3.10+
- API keys (see below)

### Environment variables

Copy `.env.example` to `.env` and fill in the relevant keys:

```bash
cp .env.example .env
```

| Variable | Required By | Notes |
|----------|------------|-------|
| `ANTHROPIC_API_KEY` | ticker-research-mas, etf-selection-mas | |
| `GEMINI_API_KEY` | sip-execution-mas | Falls back to VADER sentiment if missing |
| `ALPACA_API_KEY` | sip-execution-mas | Optional — live trading only |
| `ALPACA_SECRET_KEY` | sip-execution-mas | Optional — live trading only |
| `ALPACA_PAPER` | sip-execution-mas | `true` (default) = paper account |
| `DHAN_CLIENT_ID` | sip-execution-mas | Optional — Indian ETF stub |
| `DHAN_ACCESS_TOKEN` | sip-execution-mas | Optional — Indian ETF stub |

### Install dependencies (per system)

Each system has its own `requirements.txt`. Install only what you need:

```bash
pip install -r ticker-research-mas/requirements.txt
pip install -r etf-selection-mas/requirements.txt
pip install -r sip_execution_mas/requirements.txt
```

---

## Data flow

```
etf-selection-mas
  └─ outputs/rankings.json
        └─→ sip_execution_mas (optional input for backtest)
                └─ simulator/outputs/portfolio_ledger.json
                        └─→ simulator/app.py (Streamlit dashboard)
```

`ticker-research-mas` is standalone — use it independently to research individual tickers before deciding ETF allocations.

---

## License

MIT
