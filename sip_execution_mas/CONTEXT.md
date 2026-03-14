# sip_execution_mas — Context Document

> Single source of truth for agents, contributors, and AI assistants working in this module.

## Purpose

A **6-node LangGraph execution loop** that invests a fixed 12-ETF universe each month using Gemini for sentiment scoring and allocation weighting, audits the portfolio against hard risk rules, executes orders (paper or live), and persists every trade to a shared ledger. It is a **decision + execution system** — unlike `etf-selection-mas`, it can move money.

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
│   ├── broker_connector.py      Node 5 — paper / Alpaca / Dhan execution
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

## ETF Universe — Locked v3 (12 ETFs in 2 Permanent Buckets)

The production universe is fixed — Gemini never changes *which* ETFs are held. Gemini sentiment only determines *how much* capital each satellite ETF receives within its satellite budget.

### Core Bucket — 70% of SIP (always all 5, equal-weight by consensus score)

| Ticker | Name | TER | Theme |
|--------|------|-----|-------|
| USCA | BNY Mellon US Large Cap Core Equity ETF | 0.07% | US Large-Cap / Climate Action Anchor |
| SPDW | SPDR Portfolio Developed World ex-US ETF | 0.04% | Developed World ex-U.S. (Europe / Japan) |
| SPEM | SPDR Portfolio Emerging Markets ETF | 0.07% | Broad Emerging Markets |
| FLIN | Franklin FTSE India ETF | 0.19% | India Large-Cap (US-listed) |
| NIFTYBEES.NS | Nippon India ETF Nifty 50 BeES | 0.04% | India Large-Cap (NSE direct) |

### Satellite Bucket — 30% of SIP (always all 7, sentiment-weighted)

| Ticker | Name | TER | Theme |
|--------|------|-----|-------|
| SOXQ | Invesco PHLX Semiconductor ETF | 0.19% | Hardware Supercycle (AI Semis) |
| XLK | Technology Select Sector SPDR Fund | 0.10% | Broad Tech & Software Infrastructure |
| AVUV | Avantis U.S. Small Cap Value ETF | 0.25% | U.S. Small-Cap Value (VTI Replacement) |
| URNM | Sprott Uranium Miners ETF | 0.83% | Uranium / AI Energy Infrastructure |
| CIBR | First Trust NASDAQ Cybersecurity ETF | 0.60% | Cybersecurity |
| SMIN | iShares MSCI India Small-Cap ETF | 0.74% | India Small-Cap / Domestic Consumption |
| XBI | SPDR S&P Biotech ETF | 0.35% | Equal-Weight Biotech (Rate-Sensitive Innovation) |

> **Backtest mode** (`as_of_date` set): Node 1 still uses the same locked 12-ETF universe — no Gemini discovery in any mode. yfinance is date-bounded to the backtest month, DDGS headlines are filtered to that date, and `_fixed_bucket_alloc()` applies the configured core_pct split (default 70/30) as production.

---

## Nodes

### Node 1 — Regional Researcher (`agents/regional_researcher.py`)

**Both production and backtest** use the identical locked 12-ETF universe (`_LOCKED_UNIVERSE`). No Gemini calls in Node 1 under any circumstance. In backtest mode (`as_of_date` set), yfinance history is date-bounded and DDGS queries apply a `timelimit` filter — but universe selection never changes.

**yfinance enrichment:**
- 1-year price history: `current_price`, `momentum_3m` (63 bars), `momentum_1m` (21 bars), `ytd_return`
- **`trailing_volatility_3m`**: `pct_change().std() × √252` over last 63 bars (annualised decimal, e.g. 0.15 = 15%)
- `expense_ratio`, `aum_b` from yfinance `.info` (overrides locked seed if available)
- **`forward_pe`**: from `info["forwardPE"]` — valuation anchor for Gemini scoring
- **`beta`**: from `info["beta"]` — regime scaling input
- **`dividend_yield`**: from `info["dividendYield"]` or `info["trailingAnnualDividendYield"]` — quality signal

**News fetch (DDGS) — 5 thematic categories, 2 queries each:**

| Category | What it captures |
|----------|-----------------|
| `TECH_SEMIS` | AI capex cycles, hyperscaler data-center spending, semiconductor & cybersecurity demand |
| `INDIA_EM` | FII flows, RBI policy, emerging-market manufacturing rotation, India small-cap |
| `URANIUM_ENERGY` | Uranium spot price, nuclear energy demand from AI data centers, energy transition |
| `BIOTECH` | FDA approvals, GLP-1/RNA pipeline, biotech M&A, rate sensitivity |
| `QUALITY_CORE` | US large-cap breadth, developed-world international equities, Fed outlook |

Returns `all_macro_news` as `{"TECH_SEMIS": [...], "INDIA_EM": [...], "URANIUM_ENERGY": [...], "BIOTECH": [...], "QUALITY_CORE": [...]}`.

**Ticker → Category mapping (`_TICKER_CATEGORY`):**

| Ticker | Category |
|--------|----------|
| SOXQ, XLK, CIBR | TECH_SEMIS |
| FLIN, NIFTYBEES.NS, SPEM, SMIN | INDIA_EM |
| URNM | URANIUM_ENERGY |
| XBI | BIOTECH |
| USCA, SPDW, AVUV | QUALITY_CORE |

**Reads:** `ter_threshold`, `as_of_date`
**Writes:** `all_etf_data`, `all_macro_news`, `filtered_tickers`

---

### Node 2 — Signal Scorer (`agents/signal_scorer.py`)

**Gemini call (`gemini-2.0-flash`) — falls back to category-aware VADER if key missing.**

**System prompt — 10-year horizon macro strategist:** Gemini is instructed to score each ETF (0.00–1.00) as a quantitative macro strategist managing a 10-year horizon systematic portfolio. Five numbered rules applied in order:

1. **Structural tailwinds override short-term drawdowns** — a 3-month price dip during a semiconductor inventory cycle does NOT lower a score if the AI capex cycle is intact.
2. **Forward P/E is a valuation ceiling** — fwdPE > 40x tech → cap at 0.75; fwdPE > 30x broad → cap at 0.80; fwdPE < 15x with positive news → can reach 0.90+.
3. **Beta/risk-regime adjustment — satellite vs core distinction** — CORE ETFs (USCA, SPDW, SPEM, FLIN, NIFTYBEES.NS): beta > 1.3 risk-off → −0.05–0.10; beta < 0.8 → +0.05. SATELLITE ETFs (SOXQ, XLK, AVUV, URNM, CIBR, SMIN, XBI): do NOT penalize high beta during panics; if structural tailwind intact, maintain or +0.05; only reduce if sector news is also structurally negative.
4. **Dividend yield signals quality** — rising yield on QUAL/VTI/SPLG → +0.05.
5. **Sector news is paired directly with each ETF** — assess capital flows, not sentiment; distinguish policy from noise.

**User prompt — structured per-ETF blocks:**
```
[ETF: SOXQ] semiconductors | region=US
  [Fundamentals] fwdPE=28.5x  beta=1.30  TER=0.19%  3m=+8.0%  YTD=+12.3%  divYield=N/A
  [Sector News — TECH_SEMIS]
    • Hyperscaler data-center capex commitments reach $40B annualised...
    • TSMC reports chip-on-wafer-on-substrate yield improvement...
```

Each ETF block pairs its `[Fundamentals]` (valuation anchor) with `[Sector News — CATEGORY]` (capital-flow catalysts) inline, so Gemini weighs them together.

**Gemini returns JSON:**
```json
{
  "scores": {"TICKER": 0.72},
  "boom_triggers": ["AI_CAPEX_CYCLE", "SEMICONDUCTOR_UPCYCLE"],
  "macro_summary": "2-3 sentence 10-year structural outlook."
}
```

**Boom trigger examples:** `AI_CAPEX_CYCLE`, `CLEAN_ENERGY_POLICY_TAILWIND`, `INDIA_FII_INFLOWS`, `EM_SUPPLY_CHAIN_ROTATION`, `US_BUYBACK_SURGE`, `FED_PIVOT`, `SEMICONDUCTOR_UPCYCLE`, `ESG_REGULATORY_TIGHTENING`, `INDIA_RATE_CUT`, `TECH_VALUATION_COMPRESSION`

**Reads:** `filtered_tickers`, `all_etf_data`, `all_macro_news`, `as_of_date` (for backtest date-isolation header)
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
Tickers not in the map receive the overall average score.
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

The volatility divisor penalises high-vol ETFs (pseudo-Sharpe), and batch normalization keeps `EVICTION_THRESHOLD = 0.30` calibrated across months regardless of absolute volatility levels.

#### Production allocation path (locked universe)

When `filtered_tickers ⊆ _LOCKED_TICKERS`:

```
core_budget      = sip_amount × 0.70
satellite_budget = sip_amount × 0.30

For each bucket, allocate proportionally to consensus_score:
  weight_i_in_bucket = consensus_score_i / Σ consensus_score_bucket
  monthly_usd_i      = weight_i_in_bucket × bucket_budget
```

All 7 core and all 7 satellite ETFs always receive a position. Gemini sentiment only rotates *weight* within the satellite bucket — no ETF is ever dropped from the portfolio.

#### Backtest path

Same locked universe and `_fixed_bucket_alloc()` as production. yfinance is date-bounded; `score_etfs()` in `signal_scorer.py` receives the `reference_date` parameter and injects a `SIMULATION MODE` header into the Gemini system prompt so it scores only on provided data.

#### Retry enforcement

On Risk Auditor rejection (`audit_retry_count > 0`): `_apply_violation_caps()` enforces `max_position_pct` and `max_region_pct` hard caps on the allocation.

**Reads:** `filtered_tickers`, `all_etf_data`, `sentiment_scores`, `ter_threshold`, SIP config, `risk_violations`, `audit_retry_count`
**Writes:** `expense_scores`, `consensus_scores`, `allocation_plan`, `proposed_orders`, `optimizer_notes`

---

### Node 4 — Risk Auditor (`agents/risk_auditor.py`)

**Hard Python rules first. Then Value-Averaging check. Gemini explains violations in natural language.**

#### Five hard rules

| Rule | ID | Check | Violation prefix |
|------|----|-------|-----------------|
| 1 | MAX_POSITION | No ETF > `max_position_pct × sip_amount` | `RULE_1_VIOLATION:` |
| 2 | MAX_REGION | No region > `max_region_pct × sip_amount` | `RULE_2_VIOLATION:` |
| 3 | MIN_POSITION | No ETF allocated < $1.00 | `RULE_3_VIOLATION:` |
| 4 | NO_DUPLICATE | Ledger has no entry for current calendar month | `RULE_4_VIOLATION:` |
| 5 | MIN_ORDERS | At least 1 proposed order exists | `RULE_5_VIOLATION:` |

> Rule 4 is bypassed when `dry_run=True` **or** `force=True`.

#### Routing

```python
MAX_RETRIES = 2

if no violations:                           → approved=True  → broker
if violations + retry_count < MAX_RETRIES:  → approved=False → optimizer (loop)
if violations + retry_count >= MAX_RETRIES: → approved=False → logger (abort)
```

#### Crash-Accumulator Value-Averaging (on clean approval only)

When the portfolio is approved and a panic condition fires, the SIP is scaled up by a tier-dependent multiplier based on how deep the drawdown is:

| Condition | Definition |
|-----------|------------|
| **Panic** | Any `_PANIC_TRIGGER_TOKENS` token in `boom_triggers` OR portfolio-average sentiment < `0.35` |
| **Tier 1 — Standard Dip** | Panic present AND `−15.0% < avg_momentum_3m ≤ 0.0%` |
| **Tier 2 — Generational Crash** | Panic present AND `avg_momentum_3m ≤ −15.0%` |

```
Tier 1: va_multiplier = 1.20  →  effective_sip = base_sip × 1.20  (+20%)
Tier 2: va_multiplier = 1.50  →  effective_sip = base_sip × 1.50  (+50%)

monthly_usd_i = monthly_usd_i × va_multiplier   (all proposed orders scaled)
weight_i recomputed from scaled monthly_usd_i / effective_sip
```

Positive momentum with a panic signal (e.g., early sentiment deterioration before price falls) does not trigger VA — the discount must be present. State fields `va_triggered` and `va_multiplier` are always written (1.0 = no adjustment).

**Panic trigger tokens:** `GLOBAL_RISK_OFF`, `MARKET_CRASH`, `SYSTEMIC_RISK`, `RECESSION_FEAR`, `CREDIT_CRUNCH`, `BANKING_CRISIS`, `LIQUIDITY_CRISIS`, `BEAR_MARKET`, `GEOPOLITICAL_RISK`, `RATE_SHOCK`, `CHINA_TECH_CRACKDOWN`, `EMERGING_MARKETS_CRASH`, `FLASH_CRASH`, `VOLATILITY_SPIKE`

**VA constants:** `VA_MULTIPLIER_TIER1 = 1.20`, `VA_MULTIPLIER_TIER2 = 1.50`, `MOMENTUM_CRASH_THRESHOLD = −15.0%`, `PANIC_SENTIMENT_THRESHOLD = 0.35`

**Reads:** `proposed_orders`, `sip_amount`, `max_position_pct`, `max_region_pct`, `ledger_path`, `audit_retry_count`, `dry_run`, `force`, `sentiment_scores`, `all_etf_data`, `boom_triggers`
**Writes:** `risk_approved`, `risk_violations`, `risk_audit_notes`, `audit_retry_count`, `va_triggered`, `va_multiplier` (1.0 / 1.20 / 1.50), `proposed_orders` (scaled), `sip_amount` (effective)

---

### Node 5 — Broker Connector (`agents/broker_connector.py`)

**No LLM. Executes or simulates orders.**

| Mode | Condition | Behaviour |
|------|-----------|-----------|
| **Paper** | `dry_run=True` (default) | Simulates fills at yfinance close price. `status="dry_run"`. |
| **Alpaca** | `dry_run=False` + `ALPACA_API_KEY` set | Submits notional `MarketOrderRequest` via `alpaca-py`. US ETFs only. |
| **Dhan stub** | Any `.NS` / `.BO` ticker or `currency="INR"` | Always paper (Alpaca doesn't support NSE). `broker="dhan_stub"`. |

**Unit calculation:**
```
USD tickers:  units = monthly_usd / price_usd
INR tickers:  inr_amount = monthly_usd × usd_inr_rate
              units = inr_amount / price_inr
```

Note: `monthly_usd` already reflects Value-Averaging scaling if VA fired in Node 4.

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
    top_n: int                    # ignored in locked-universe mode (always 14)
    core_count: int               # ignored in locked-universe mode (always 7)
    core_pct: float               # ignored in locked-universe mode (always 0.70)
    max_position_pct: float       # hard cap per ETF (default 0.15)
    max_region_pct: float         # hard cap per region (default 0.50)
    dry_run: bool                 # True = paper mode
    force: bool                   # True = bypass Rule 4 (allow re-invest same month)
    ledger_path: str              # path to portfolio_ledger.json

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
    va_triggered: bool    # True when Value-Averaging 20% top-up was applied
    va_multiplier: float  # 1.0 = no adjustment; 1.20 = top-up applied

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
    trailing_volatility_3m: Optional[float]   # annualised daily vol, decimal (e.g. 0.15 = 15%)
    forward_pe: Optional[float]               # forward P/E ratio — valuation anchor for Node 2
    beta: Optional[float]                     # 1-year beta vs S&P 500 — regime scaling for Node 2
    dividend_yield: Optional[float]           # trailing annual dividend yield (decimal) — quality signal
    current_price: Optional[float]; currency: str   # "USD" | "INR" | "HKD"
    data_source: str; fetch_error: Optional[str]
    recommended_broker: str; adv_usd: Optional[float]; liquidity_ok: bool
    est_entry_cost_pct: Optional[float]; is_proxy: bool; proxy_for: Optional[str]

class ProposedOrder(TypedDict):
    ticker: str; name: str; region: str; market: str; bucket: str
    monthly_usd: float; weight: float; consensus_score: float
    sentiment_score: float; expense_score: float
    sentiment_rationale: str; currency: str

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

### `sip_execution_mas` — direct MAS execution

```bash
# Default: dry-run, $500 SIP, locked 12-ETF universe, 70/30 split
python -m sip_execution_mas

# Custom SIP amount
python -m sip_execution_mas --sip 750

# Custom risk limits
python -m sip_execution_mas --max-position 0.20 --max-region 0.60

# Stricter TER filter (used for consensus scoring; all 12 ETFs still included)
python -m sip_execution_mas --ter 0.30

# Live Alpaca paper trading (requires ALPACA_API_KEY)
python -m sip_execution_mas --live

# Force re-investment even if already done this month
python -m sip_execution_mas --force
```

**All flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sip` | float | 500.0 | Monthly SIP in USD |
| `--top` | int | 12 | Ignored in locked-universe mode |
| `--core` | int | 5 | Ignored in locked-universe mode |
| `--core-pct` | float | 0.70 | Ignored in locked-universe mode (always 70%) |
| `--ter` | float | 0.70 | TER ceiling used in expense score calculation |
| `--max-position` | float | 0.15 | Max fraction per ETF (hard risk rule) |
| `--max-region` | float | 0.50 | Max fraction per region (hard risk rule) |
| `--live` | flag | off | Disable dry-run, use broker API |
| `--force` | flag | off | Bypass Rule 4 (allow re-invest same month) |
| `--ledger` | str | auto | Path to portfolio_ledger.json |
| `--run-id` | str | auto | Custom run identifier |

> `--top`, `--core`, `--core-pct` have no effect — the locked universe always invests all 12 ETFs with a fixed 70/30 split in both production and backtest.

### `simulator/scheduler.py` — monthly APScheduler

```bash
python -m simulator.scheduler --now          # Execute immediately (dry-run)
python -m simulator.scheduler                # Start persistent scheduler (09:00 on 1st)
python -m simulator.scheduler --status       # Show ledger status
python -m simulator.scheduler --now --force  # Force re-investment this month
python -m simulator.scheduler --now --live   # Live execution
```

### `simulator/main.py` — backtest simulator

```bash
python -m simulator.main --sip 500 --months 24
python -m simulator.main --run-sip-mas       # Gemini rankings via sip_execution_mas
```

### `simulator/app.py` — Streamlit dashboard

```bash
streamlit run sip_execution_mas/simulator/app.py
```

---

## Environment Variables

| Variable | Required | Used By | Notes |
|----------|----------|---------|-------|
| `GEMINI_API_KEY` | Recommended | Nodes 2, 3, 4 | Falls back to VADER (Node 2) / template (Nodes 3, 4) if missing. Node 1 no longer needs Gemini in production. |
| `ALPACA_API_KEY` | Optional | Node 5 | Only needed for `--live` mode |
| `ALPACA_SECRET_KEY` | Optional | Node 5 | Paired with API key |
| `ALPACA_PAPER` | Optional | Node 5 | `"true"` (default) = paper account |
| `DHAN_CLIENT_ID` | Future | Node 5 | Not yet implemented |
| `DHAN_ACCESS_TOKEN` | Future | Node 5 | Not yet implemented |

Loaded from `fin-agents/.env` at startup via `python-dotenv`.

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
# alpaca-py>=0.21.0   # uncomment for live Alpaca execution
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

`sip_execution_mas` supersedes `etf-selection-mas` for ETF selection. The production universe is now locked (12 ETFs — v3), eliminating month-to-month portfolio cycling from Gemini discovery variability. Gemini is used only for sentiment scoring and rationale generation.

---

## Known Limitations

- Dhan broker (Indian ETFs) is always paper — real Dhan API integration is a stub.
- `audit_retry_count` increments on every rejection. After `MAX_RETRIES=2` the run aborts even if only one minor rule was violated — by design, to prevent infinite loops.
- `portfolio_ledger.json` records the *planned* allocation, not the exact broker fill. For fill-exact records use `execution_log.csv`.
- Value-Averaging fires at most once per month (on clean approval). It increases *spending* but does not short or sell any position.
- `yfinance` TER data is often missing; Node 1 falls back to the locked-universe `ter_pct` seed value, which is used for expense score calculation only (all 12 ETFs are always included regardless of TER).
- `trailing_volatility_3m` may be `None` for tickers with < 20 trading days of history; Node 3 falls back to `_VOL_FLOOR = 0.05` (5% annualised) in this case.
- DDGS news can be empty during rate-limit windows; Node 2 VADER fallback handles this gracefully.
- Gemini `response_mime_type="application/json"` occasionally wraps output in markdown; `_parse_json()` strips it.
