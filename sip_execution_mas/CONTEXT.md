# sip_execution_mas — Context Document

> Single source of truth for agents, contributors, and AI assistants working in this module.

## Purpose

A **6-node LangGraph execution loop** that invests a fixed 5-ETF universe each month using Gemini for sentiment scoring and allocation weighting, audits the portfolio against hard risk rules, executes orders (paper or live), and persists every trade to a shared ledger. It is a **decision + execution system** — unlike `etf-selection-mas`, it can move money.

---

## Directory Structure

```
sip_execution_mas/
├── __init__.py              SSL patch (runs before any yfinance import)
├── main.py                  CLI entry point
├── requirements.txt
├── CONTEXT.md               ← this file
├── agents/
│   ├── regional_researcher.py   Node 1 — locked universe + yfinance enrichment
│   ├── signal_scorer.py         Node 2 — Gemini sentiment scoring (VADER fallback)
│   ├── portfolio_optimizer.py   Node 3 — pseudo-Sharpe scoring + 70/30 allocation
│   ├── risk_auditor.py          Node 4 — hard rules + Value-Averaging + Gemini explanation
│   ├── broker_connector.py      Node 5 — paper / IBKR (LSE) / Dhan (BSE) execution
│   └── execution_logger.py      Node 6 — ledger + CSV persistence
├── graph/
│   ├── state.py             SIPExecutionState TypedDict + sub-structs
│   └── workflow.py          LangGraph StateGraph assembly + run_sip_execution()
├── outputs/
│   └── execution_log.csv    Per-order audit trail
└── simulator/               Shared simulator sub-package
    ├── __init__.py           SSL patch
    ├── allocator.py          compute_allocation() — legacy backtest path
    ├── ledger.py             load/save/append ledger — used by Nodes 4, 6
    ├── app.py                Streamlit dashboard (reads portfolio_ledger.json)
    ├── scheduler.py          Monthly APScheduler loop → delegates to run_sip_execution()
    ├── main.py               Backtest CLI
    ├── backtest.py           Historical simulation helpers
    ├── ranker.py             VADER/no-LLM ranker (legacy fallback, locked universe)
    ├── report.py             Markdown / terminal report generator
    ├── state.py              AllocationPlan, ETFHolding, SimulationResult TypedDicts
    └── outputs/
        └── portfolio_ledger.json   Shared ledger (read by Streamlit)
```

---

## Architecture

### Graph topology (loop on audit rejection)

```
researcher → scorer → optimizer → auditor
                                      │
                          ┌───────────┼───────────┐
                       approved     retry       abort
                          ↓           ↓            ↓
                        broker     optimizer    logger
                          ↓                       ↓
                        logger ─────────────────→ END
```

**Audit retry cap:** `MAX_RETRIES = 2`. After two rejections the graph routes directly to Node 6 with `execution_status = "aborted"`.

---

## ETF Universe — Locked v4 (5 ETFs in 2 Permanent Buckets — Tax Optimised)

The production universe is fixed — Gemini never changes *which* ETFs are held. Gemini sentiment only determines *how much* capital each ETF receives within its bucket.

**Rationale for v4:** Eliminates the 30% US dividend withholding tax (all US-listed ETFs replaced by Ireland-domiciled UCITS equivalents or BSE-direct ETFs) and reduces commission drag to two brokers only (IBKR for LSE, Dhan for BSE).

### Core Bucket — 70% of SIP (always both, consensus-score-weighted)

| Ticker | Name | Market | TER | Currency | Theme |
|--------|------|--------|-----|----------|-------|
| VWRA.L | Vanguard FTSE All-World UCITS ETF | LSE | 0.22% | USD | Global All-World |
| NIFTYBEES.NS | Nippon India ETF Nifty 50 BeES | NSE | 0.04% | INR | India Large-Cap (NSE direct) |

### Satellite Bucket — 30% of SIP (always all 3, sentiment-weighted)

| Ticker | Name | Market | TER | Currency | Theme |
|--------|------|--------|-----|----------|-------|
| IUIT.L | iShares S&P 500 Tech Sector UCITS ETF | LSE | 0.15% | USD | Tech & AI |
| WSML.L | iShares MSCI World Small Cap UCITS ETF | LSE | 0.35% | USD | World Small-Cap |
| MOM100.NS | Motilal Oswal Nifty Midcap 100 ETF | NSE | 0.28% | INR | India Midcap (NSE direct) |

> **WSML.L replaced VVSM.L** (Vanguard MSCI USA Small Cap) — VVSM.L was delisted on Yahoo Finance. WSML.L is the iShares MSCI World Small Cap UCITS ETF, a higher-liquidity global alternative with reliable yfinance coverage.

> **Backtest mode** (`as_of_date` set): Node 1 still uses the same locked 5-ETF universe — no Gemini discovery in any mode. yfinance is date-bounded to the backtest month, DDGS headlines are filtered to that date, and `_fixed_bucket_alloc()` applies the configured core_pct split (default 70/30) as production.

---

## Nodes

### Node 1 — Regional Researcher (`agents/regional_researcher.py`)

**Both production and backtest** use the identical locked 5-ETF universe (`_LOCKED_UNIVERSE`). No Gemini calls in Node 1 under any circumstance. In backtest mode (`as_of_date` set), yfinance history is date-bounded and DDGS queries apply a `timelimit` filter — but universe selection never changes.

**yfinance enrichment:**
- 1-year price history: `current_price`, `momentum_3m` (63 bars), `momentum_1m` (21 bars), `ytd_return`
- **`trailing_volatility_3m`**: `pct_change().std() × √252` over last 63 bars (annualised decimal, e.g. 0.15 = 15%)
- `expense_ratio`, `aum_b` from yfinance `.info` (overrides locked seed if available)
- **`forward_pe`**: from `info["forwardPE"]` — valuation anchor for Gemini scoring
- **`beta`**: from `info["beta"]` — regime scaling input
- **`dividend_yield`**: from `info["dividendYield"]` or `info["trailingAnnualDividendYield"]` — quality signal
- **`trading_currency`**: captured from `info["currency"]` for LSE tickers — drives FX routing in Node 5

**LSE currency normalisation (two-pass):**

yfinance returns inconsistent currency labels for LSE tickers. Node 1 applies two passes:

| Pass | Condition | Action |
|------|-----------|--------|
| 1 — label check | `info["currency"] == "GBp"` | `price ÷ 100`, `trading_currency = "GBP"` |
| 2 — magnitude guard | label is `"USD"` or `"GBP"` but `price > 500` | `price ÷ 100`, `trading_currency = "GBP"` + warning |

Real USD/GBP prices for all ETFs in this universe are < $300; any price > 500 is unambiguously pence. Pass 2 catches the case where yfinance mis-labels a GBp-priced ticker as `"USD"`.

**LSE volume fallback:** `info["averageVolume"]` is often `None` for LSE tickers. Node 1 falls back to `hist["Volume"].tail(30).mean()` when the info field is missing.

**News fetch (DDGS) — 3 thematic categories, 2 queries each:**

| Category | What it captures |
|----------|-----------------|
| `TECH_SEMIS` | AI capex cycles, hyperscaler data-center spending, S&P 500 tech sector |
| `INDIA_EM` | FII flows, RBI policy, India midcap growth, NSE rally |
| `QUALITY_CORE` | FTSE All-World global markets, US small-cap, Fed outlook, developed-world equities |

Returns `all_macro_news` as `{"TECH_SEMIS": [...], "INDIA_EM": [...], "QUALITY_CORE": [...]}`.

**Ticker → Category mapping (`_TICKER_CATEGORY`):**

| Ticker | Category |
|--------|----------|
| IUIT.L | TECH_SEMIS |
| NIFTYBEES.NS, MOM100.NS | INDIA_EM |
| VWRA.L, WSML.L | QUALITY_CORE |

**Reads:** `ter_threshold`, `as_of_date`
**Writes:** `all_etf_data`, `all_macro_news`, `filtered_tickers`

---

### Node 2 — Signal Scorer (`agents/signal_scorer.py`)

**Gemini call (`gemini-2.0-flash`) — falls back to category-aware VADER if key missing or rate-limited.**

**System prompt — 10-year horizon macro strategist:** Gemini is instructed to score each ETF (0.00–1.00) as a quantitative macro strategist managing a 10-year horizon systematic portfolio. Five numbered rules applied in order:

1. **Structural tailwinds override short-term drawdowns** — a 3-month price dip during a semiconductor inventory cycle does NOT lower a score if the AI capex cycle is intact.
2. **Forward P/E is a valuation ceiling** — fwdPE > 40x tech → cap at 0.75; fwdPE > 30x broad → cap at 0.80; fwdPE < 15x with positive news → can reach 0.90+.
3. **Beta/risk-regime adjustment — satellite vs core distinction** — CORE ETFs: beta > 1.3 risk-off → −0.05–0.10; beta < 0.8 → +0.05. SATELLITE ETFs: do NOT penalize high beta during panics; only reduce if sector news is also structurally negative.
4. **Dividend yield signals quality** — rising yield on quality ETFs → +0.05.
5. **Sector news is paired directly with each ETF** — assess capital flows, not sentiment; distinguish policy from noise.

**Gemini returns JSON:**
```json
{
  "scores": {"TICKER": 0.72},
  "boom_triggers": ["AI_CAPEX_CYCLE", "SEMICONDUCTOR_UPCYCLE"],
  "macro_summary": "2-3 sentence 10-year structural outlook."
}
```

**Boom trigger examples:** `AI_CAPEX_CYCLE`, `INDIA_FII_INFLOWS`, `INDIA_GROWTH`, `SEMICONDUCTOR_UPCYCLE`, `FED_PIVOT`, `TECH_VALUATION_COMPRESSION`, `GLOBAL_RISK_OFF`, `MARKET_CRASH`

**Reads:** `filtered_tickers`, `all_etf_data`, `all_macro_news`, `as_of_date`
**Writes:** `sentiment_scores`, `boom_triggers`, `macro_summary`

**Gemini config:** `gemini-2.0-flash`, `temperature=0.1`, `response_mime_type="application/json"`

**Score bands:**
- `0.00 – 0.35` Structural headwinds / risk-off
- `0.35 – 0.65` Neutral / standard weight
- `0.65 – 1.00` Structural tailwinds / risk-on

**VADER fallback — category-aware:**
```
For each category in all_macro_news:
    compound_avg = mean of VADER compound scores for that category's headlines
    cat_score    = 0.10 + (compound_avg + 1) / 2 × 0.80   # normalised to [0.10, 0.90]

Each ticker is mapped to its category via _TICKER_CATEGORY.
```

---

### Node 3 — Portfolio Optimizer (`agents/portfolio_optimizer.py`)

**Gemini call for rationales only. Allocation math is pure Python.**

#### Scoring formula (pseudo-Sharpe rank)

```
expense_score_i  = max(0, 1 − TER_i / ter_threshold)   # unknown TER → 0.50
numerator_i      = 0.60 × sentiment_score_i + 0.40 × expense_score_i
vol_i            = trailing_volatility_3m_i   (floored at 0.05 = 5% annualised)
raw_i            = numerator_i / vol_i
consensus_score_i = raw_i / max_j(raw_j)               # batch normalised to [0, 1]
```

#### Production allocation path (locked universe)

When `filtered_tickers ⊆ _LOCKED_TICKERS`:

```
_CORE_TICKERS      = {"VWRA.L", "NIFTYBEES.NS"}
_SATELLITE_TICKERS = {"IUIT.L", "WSML.L", "MOM100.NS"}

core_budget      = sip_amount × 0.70   ($350 on a $500 SIP)
satellite_budget = sip_amount × 0.30   ($150 on a $500 SIP)

For each bucket, allocate proportionally to consensus_score:
  weight_i_in_bucket = consensus_score_i / Σ consensus_score_bucket
  monthly_usd_i      = weight_i_in_bucket × bucket_budget
```

All 2 core and all 3 satellite ETFs always receive a position. Gemini sentiment only rotates *weight* within each bucket — no ETF is ever dropped from the portfolio.

#### Retry enforcement (violation caps)

On Risk Auditor rejection (`audit_retry_count > 0`): `_apply_violation_caps()` enforces position and region caps.

**Rule 6 consolidation:** When `RULE_6_VIOLATION` tickers (dust positions < $30) are passed back via `dust_tickers`, `_apply_violation_caps()` drops those positions and redistributes their budget to the remaining positions proportional to `consensus_score`, before applying other caps.

**Reads:** `filtered_tickers`, `all_etf_data`, `sentiment_scores`, `ter_threshold`, SIP config, `risk_violations`, `audit_retry_count`, `dust_tickers`
**Writes:** `expense_scores`, `consensus_scores`, `allocation_plan`, `proposed_orders`, `optimizer_notes`

---

### Node 4 — Risk Auditor (`agents/risk_auditor.py`)

**Hard Python rules first. Then Value-Averaging check. Gemini explains violations in natural language.**

#### Six hard rules

| Rule | ID | Check | Violation prefix |
|------|----|-------|-----------------|
| 1 | MAX_POSITION | No ETF > `max_position_pct × sip_amount` | `RULE_1_VIOLATION:` |
| 2 | MAX_REGION | No region > `max_region_pct × sip_amount` | `RULE_2_VIOLATION:` |
| 3 | MIN_POSITION | No ETF allocated < $1.00 | `RULE_3_VIOLATION:` |
| 4 | NO_DUPLICATE | Ledger has no entry for current calendar month | `RULE_4_VIOLATION:` |
| 5 | MIN_ORDERS | At least 1 proposed order exists | `RULE_5_VIOLATION:` |
| 6 | MIN_LOT_SIZE | No trade ≤ $30.00 USD | `RULE_6_VIOLATION:` |

> Rule 4 is bypassed when `dry_run=True` **or** `force=True`.

**Rule 6 rationale:** IBKR charges ~$2.00 minimum per trade. Any position ≤ $30 incurs > 6% immediate fee drag. The violation message includes the computed drag percentage. On retry, `dust_tickers` (all Rule-6 violators) are passed to Node 3 which consolidates them into the larger positions.

**Risk defaults for v4 (5-ETF LSE-heavy universe):**
- `max_position_pct = 0.50` — core ETFs each naturally hold ~35% of SIP
- `max_region_pct  = 0.80` — LSE accounts for 3 of 5 ETFs

#### Routing

```python
MAX_RETRIES = 2

if no violations:                           → approved=True  → broker
if violations + retry_count < MAX_RETRIES:  → approved=False → optimizer (loop)
if violations + retry_count >= MAX_RETRIES: → approved=False → logger (abort)
```

#### Crash-Accumulator Value-Averaging (on clean approval only)

| Condition | Definition |
|-----------|------------|
| **Panic** | Any `_PANIC_TRIGGER_TOKENS` token in `boom_triggers` OR portfolio-average sentiment < `0.35` |
| **Tier 1 — Standard Dip** | Panic present AND `−15.0% < avg_momentum_3m ≤ 0.0%` |
| **Tier 2 — Generational Crash** | Panic present AND `avg_momentum_3m ≤ −15.0%` |

```
Tier 1: va_multiplier = 1.20  →  effective_sip = base_sip × 1.20  (+20%)
Tier 2: va_multiplier = 1.50  →  effective_sip = base_sip × 1.50  (+50%)
```

**Panic trigger tokens:** `GLOBAL_RISK_OFF`, `MARKET_CRASH`, `SYSTEMIC_RISK`, `RECESSION_FEAR`, `CREDIT_CRUNCH`, `BANKING_CRISIS`, `LIQUIDITY_CRISIS`, `BEAR_MARKET`, `GEOPOLITICAL_RISK`, `RATE_SHOCK`, `CHINA_TECH_CRACKDOWN`, `EMERGING_MARKETS_CRASH`, `FLASH_CRASH`, `VOLATILITY_SPIKE`

**Reads:** `proposed_orders`, `sip_amount`, `max_position_pct`, `max_region_pct`, `ledger_path`, `audit_retry_count`, `dry_run`, `force`, `sentiment_scores`, `all_etf_data`, `boom_triggers`
**Writes:** `risk_approved`, `risk_violations`, `risk_audit_notes`, `audit_retry_count`, `per_region_caps`, `dust_tickers`, `va_triggered`, `va_multiplier`, `proposed_orders` (scaled), `sip_amount` (effective)

---

### Node 5 — Broker Connector (`agents/broker_connector.py`)

**No LLM. Routes by ticker suffix and executes or simulates orders.**

#### Routing

| Suffix | Broker | Condition |
|--------|--------|-----------|
| `.L` | **IBKR** (`ib_insync`) | `dry_run=False` + IBKR env vars set |
| `.BO` | **DhanHQ** (`dhanhq`) | `dry_run=False` + Dhan env vars set |
| any | **Paper** | `dry_run=True` (default) or keys missing |

#### Currency routing within IBKR (LSE)

Node 1 captures the actual trading currency from yfinance for each `.L` ticker. Node 5 uses it to decide FX conversion:

| `order["currency"]` | Action | IBKR contract |
|---------------------|--------|---------------|
| `"USD"` | No conversion — budget is already in USD | `Stock(symbol, "LSE", "USD")` |
| `"GBP"` | `native_budget = monthly_usd / usd_gbp` | `Stock(symbol, "LSE", "GBP")` |

VWRA.L, IUIT.L, and WSML.L all trade in USD on LSE — no GBP conversion needed.

#### Unit calculation

```
USD tickers:  units = monthly_usd / price_usd
GBP tickers:  units = (monthly_usd / usd_gbp) / price_gbp   [whole shares only — LSE no fractional]
INR tickers:  units = (monthly_usd × usd_inr) / price_inr
```

#### FX rates fetched live

```python
_get_usd_inr()  → yf.Ticker("USDINR=X")  fallback: 84.0
_get_usd_gbp()  → yf.Ticker("GBPUSD=X")  fallback: 1.27
```

#### DhanHQ security ID mapping (BSE)

```python
_DHAN_SECURITY_IDS = {
    "NIFTYBEES": "532788",   # BSE scrip code — verify against scrip master CSV
    "MOM100":    "543590",   # BSE scrip code — verify against scrip master CSV
}
```

> Tickers not in the map fall back to paper mode. Populate from `https://images.dhan.co/api-data/api-scrip-master.csv` (column: `SEM_SMST_SECURITY_ID`, segment `BSE_EQ`).

#### IBKR env vars

| Var | Default | Notes |
|-----|---------|-------|
| `IBKR_HOST` | `127.0.0.1` | TWS / IB Gateway host |
| `IBKR_PORT` | `7497` | 7497 = paper, 7496 = live |
| `IBKR_CLIENT_ID` | `1` | Unique client ID per connection |

**Reads:** `proposed_orders`, `dry_run`
**Writes:** `execution_results`, `execution_status`, `usd_inr_rate`

---

### Node 6 — Execution Logger (`agents/execution_logger.py`)

**No LLM. Terminal node — always reached.**

Writes two persistence targets and prints a formatted terminal summary.

**1. `simulator/outputs/portfolio_ledger.json`** (shared with Streamlit dashboard)

**2. `sip_execution_mas/outputs/execution_log.csv`** (audit trail)

CSV columns: `run_id`, `timestamp`, `month`, `ticker`, `region`, `bucket`, `broker`, `status`, `requested_usd`, `filled_usd`, `units`, `price_native`, `currency`, `order_id`, `error`

**Reads:** Full state
**Writes:** `log_path`, `run_id_out`

---

## State Schema (`graph/state.py`)

```python
class SIPExecutionState(TypedDict):
    # Config
    run_id: str
    ter_threshold: float          # decimal, e.g. 0.007 = 0.70%
    sip_amount: float             # USD base amount (Node 4 may scale up via VA)
    top_n: int                    # ignored in locked-universe mode
    core_count: int               # ignored in locked-universe mode
    core_pct: float               # 0.70 (fixed)
    max_position_pct: float       # default 0.50 (v4: core ETFs each ~35%)
    max_region_pct: float         # default 0.80 (v4: LSE-heavy, 3 of 5 ETFs)
    dry_run: bool                 # True = paper mode
    force: bool                   # True = bypass Rule 4
    ledger_path: str

    # Node 1
    all_etf_data: Dict[str, ETFRecord]
    all_macro_news: Dict[str, List[str]]
    filtered_tickers: List[str]

    # Node 2
    sentiment_scores: Dict[str, float]
    macro_summary: str
    boom_triggers: List[str]

    # Node 3
    expense_scores: Dict[str, float]
    consensus_scores: Dict[str, float]
    allocation_plan: Optional[Dict[str, Any]]
    proposed_orders: List[ProposedOrder]
    optimizer_notes: str

    # Node 4
    risk_approved: bool
    risk_violations: List[str]
    risk_audit_notes: str
    audit_retry_count: int
    per_region_caps: Dict[str, float]   # dynamic VA-boosted caps passed to Node 3 on retry
    dust_tickers: List[str]             # Rule 6 violators passed to Node 3 for consolidation
    va_triggered: bool
    va_multiplier: float                # 1.0 / 1.20 / 1.50

    # Node 5
    execution_results: List[ExecutionResult]
    execution_status: str     # "success" | "partial" | "dry_run" | "failed" | "aborted"
    usd_inr_rate: float

    # Node 6
    log_path: str
    run_id_out: str
```

**Sub-structs:**

```python
class ETFRecord(TypedDict):
    ticker: str; name: str; region: str; market: str; category: str
    expense_ratio: Optional[float]; aum_b: Optional[float]
    ytd_return: Optional[float]; momentum_3m: Optional[float]; momentum_1m: Optional[float]
    trailing_volatility_3m: Optional[float]   # annualised daily vol, decimal
    forward_pe: Optional[float]; beta: Optional[float]; dividend_yield: Optional[float]
    current_price: Optional[float]
    currency: str          # "USD" | "INR" — trading currency (GBp normalised to GBP)
    trading_currency: Optional[str]   # raw from yfinance for LSE tickers
    data_source: str; fetch_error: Optional[str]
    recommended_broker: str; adv_usd: Optional[float]; liquidity_ok: bool
    est_entry_cost_pct: Optional[float]; is_proxy: bool; proxy_for: Optional[str]

class ProposedOrder(TypedDict):
    ticker: str; name: str; region: str; market: str; bucket: str
    monthly_usd: float; weight: float; consensus_score: float
    sentiment_score: float; expense_score: float
    sentiment_rationale: str; currency: str   # "USD" | "INR" | "GBP"

class ExecutionResult(TypedDict):
    ticker: str; status: str; requested_usd: float; filled_usd: float
    units: float; price_native: float; currency: str
    broker: str; order_id: Optional[str]; error: Optional[str]
```

---

## Outputs

| File | Path | Format | Consumer |
|------|------|--------|----------|
| Portfolio ledger | `sip_execution_mas/simulator/outputs/portfolio_ledger.json` | JSON | Streamlit (`simulator/app.py`) |
| Execution log    | `sip_execution_mas/outputs/execution_log.csv`              | CSV  | Audit / debugging |

---

## CLI

### `simulator/scheduler.py` — monthly APScheduler (primary entry point)

```bash
# Execute immediately (dry-run, $500 SIP)
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m sip_execution_mas.simulator.scheduler --now --sip 500

# Start persistent scheduler (runs at 09:00 every 4 weeks on Monday)
python -m sip_execution_mas.simulator.scheduler

# Force re-investment even if already done this month
python -m sip_execution_mas.simulator.scheduler --now --force

# Live execution (requires IBKR + Dhan env vars)
python -m sip_execution_mas.simulator.scheduler --now --live
```

### `sip_execution_mas` — direct MAS execution via main.py

```bash
python -m sip_execution_mas --sip 750
python -m sip_execution_mas --max-position 0.50 --max-region 0.80
python -m sip_execution_mas --live
python -m sip_execution_mas --force
```

**All flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sip` | float | 500.0 | Monthly SIP in USD |
| `--ter` | float | 0.70 | TER ceiling used in expense score calculation |
| `--max-position` | float | 0.50 | Max fraction per ETF |
| `--max-region` | float | 0.80 | Max fraction per region |
| `--live` | flag | off | Disable dry-run, use broker APIs |
| `--force` | flag | off | Bypass Rule 4 (allow re-invest same month) |
| `--ledger` | str | auto | Path to portfolio_ledger.json |
| `--run-id` | str | auto | Custom run identifier |

### `simulator/app.py` — Streamlit dashboard

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 streamlit run sip_execution_mas/simulator/app.py --server.port 8502
```

---

## Environment Variables

| Variable | Required | Used By | Notes |
|----------|----------|---------|-------|
| `GEMINI_API_KEY` | Recommended | Nodes 2, 3, 4 | Falls back to VADER (Node 2) / template (Nodes 3, 4) if missing or rate-limited |
| `IBKR_HOST` | Optional | Node 5 | Default `127.0.0.1` (TWS / IB Gateway) |
| `IBKR_PORT` | Optional | Node 5 | `7497` = paper TWS, `7496` = live TWS |
| `IBKR_CLIENT_ID` | Optional | Node 5 | Default `1` |
| `DHAN_CLIENT_ID` | Optional | Node 5 | Required for live BSE execution |
| `DHAN_ACCESS_TOKEN` | Optional | Node 5 | Required for live BSE execution |

Loaded from `fin-agents/.env` at startup via `python-dotenv`.

> **Run tip (Windows):** Always prefix with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` to avoid Unicode errors on Windows terminals.

---

## Cross-Module Imports

`simulator/` is a sub-package of `sip_execution_mas/`. Agents add `sip_execution_mas/` to `sys.path` so imports resolve as:

```python
from simulator.ledger import (                            # Nodes 4, 6
    load_ledger, save_ledger, build_and_append_entry,
    already_invested_this_month, aggregate_holdings, ledger_summary,
)
```

---

## Dependencies (`requirements.txt`)

```
yfinance>=0.2.40
pandas>=2.0.0
python-dotenv>=1.0.0
langgraph>=0.2.0
google-generativeai>=0.8.0
vaderSentiment>=3.3.2
duckduckgo-search>=6.0.0
certifi>=2024.0.0
ib_insync>=0.9.86        # IBKR LSE execution
dhanhq>=1.3.0            # DhanHQ BSE execution
```

---

## Relationship to Other Modules

```
fin-agents/
├── etf-selection-mas/     5-node Claude pipeline → rankings.json (optional legacy input)
├── ticker-research-mas/   7-node Claude debate pipeline → per-stock analysis
└── sip_execution_mas/     6-node Gemini execution loop (this module)
    └── simulator/         Shared sub-package: allocator, ledger, Streamlit app, scheduler
```

`sip_execution_mas` supersedes `etf-selection-mas` for ETF selection. The production universe is now locked (5 ETFs — v4, Tax Optimised), eliminating month-to-month portfolio cycling from Gemini discovery variability and US dividend withholding tax.

---

## Known Limitations

- LSE tickers (VWRA.L, IUIT.L, WSML.L) are paper via `ibkr_stub` until IBKR TWS / IB Gateway is connected. Set `IBKR_PORT=7497` for paper TWS, `IBKR_PORT=7496` for live.
- Dhan broker (BSE ETFs) is paper until `DHAN_CLIENT_ID` + `DHAN_ACCESS_TOKEN` are set. BSE scrip IDs in `_DHAN_SECURITY_IDS` must be verified against the DhanHQ scrip master CSV before going live.
- `audit_retry_count` increments on every rejection. After `MAX_RETRIES=2` the run aborts — by design, to prevent infinite loops.
- `portfolio_ledger.json` records the *planned* allocation, not the exact broker fill. For fill-exact records use `execution_log.csv`.
- Value-Averaging fires at most once per month (on clean approval). It increases *spending* but does not short or sell any position.
- `yfinance` TER data is often missing for LSE tickers; Node 1 falls back to the locked-universe `ter_pct` seed value for expense score calculation.
- `trailing_volatility_3m` may be `None` for tickers with < 20 trading days of history; Node 3 falls back to `_VOL_FLOOR = 0.05` (5% annualised).
- Gemini free tier: 20 requests/day. Node 2 retries up to 4× with backoff before falling back to VADER. Nodes 3 and 4 fall back to template output on quota exhaustion.
- DDGS news can be empty during rate-limit windows; Node 2 VADER fallback handles this gracefully.
