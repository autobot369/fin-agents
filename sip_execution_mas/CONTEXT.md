# sip_execution_mas — Context Document

> Single source of truth for agents, contributors, and AI assistants working in this module.

## Purpose

A **6-node LangGraph execution loop** that dynamically discovers ETF candidates via Gemini, scores sentiment, allocates a monthly SIP budget using the Sentiment-Weighted Dynamic SIP formula, audits the allocation against hard risk rules, executes orders (paper or live), and persists every trade to a shared ledger. It is a **decision + execution system** — unlike `etf-selection-mas`, it can move money.

---

## Directory Structure

```
sip_execution_mas/
├── __init__.py              SSL patch (runs before any yfinance import)
├── main.py                  CLI entry point
├── requirements.txt
├── CONTEXT.md               ← this file
├── agents/
│   ├── regional_researcher.py   Node 1 — Gemini universe discovery + yfinance
│   ├── signal_scorer.py         Node 2 — Gemini sentiment scoring
│   ├── portfolio_optimizer.py   Node 3 — allocation math + Gemini rationales
│   ├── risk_auditor.py          Node 4 — hard rules + Gemini explanation
│   ├── broker_connector.py      Node 5 — paper / Alpaca / Dhan execution
│   └── execution_logger.py      Node 6 — ledger + CSV persistence
├── graph/
│   ├── state.py             SIPExecutionState TypedDict + sub-structs
│   └── workflow.py          LangGraph StateGraph assembly + run_sip_execution()
├── outputs/
│   └── execution_log.csv    Per-order audit trail
└── simulator/               Shared simulator sub-package (moved here from root)
    ├── __init__.py           SSL patch
    ├── allocator.py          compute_allocation() — used by Node 3
    ├── ledger.py             load/save/append ledger — used by Nodes 4, 6
    ├── app.py                Streamlit dashboard (reads portfolio_ledger.json)
    ├── scheduler.py          Monthly APScheduler loop → delegates to run_sip_execution()
    ├── main.py               Backtest CLI (--run-sip-mas for Gemini rankings)
    ├── backtest.py           Historical simulation helpers
    ├── ranker.py             VADER/no-LLM ETF ranker (legacy fallback)
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

## ETF Universe (dynamic — defined by Gemini at runtime)

Node 1 asks Gemini to recommend ETFs per market on every run. Counts below are the Gemini-requested target sizes; the hardcoded fallback list is used if Gemini is unavailable.

| Market | Gemini target | Exchange | Gemini criteria |
|--------|--------------|----------|-----------------|
| INTL   | 15           | NASDAQ/NYSE | International/global, TER ≤ 0.30%, broad diversification |
| HKCN   | 10           | NASDAQ   | China/HK exposure, TER ≤ 0.75%, mix of broad + thematic |
| BSE    | 10           | NSE (.NS)| India index ETFs, TER ≤ 0.15%, high AUM + liquidity |

After Gemini returns tickers, yfinance validates live data. The TER threshold filter (`--ter` flag) is then applied to narrow the universe further before passing to Node 2.

---

## Nodes

### Node 1 — Regional Researcher (`agents/regional_researcher.py`)

**One Gemini call per market + yfinance + DDGS news.**

**Step 1 — Gemini universe discovery:**
- Sends one prompt per market (INTL / HKCN / BSE) to `gemini-2.0-flash`
- Prompt instructs: recommend ETFs ranked by low TER, AUM, current market landscape
- Receives JSON array: `[{ticker, name, ter_pct, aum_b_est, rationale}, ...]`
- Falls back silently to hardcoded seed list per market if Gemini fails

**Gemini prompt intent per market:**
- INTL: international/global US-listed ETFs, TER ≤ 0.30%, macro + policy context
- HKCN: China/HK US-listed ETFs, TER ≤ 0.75%, factor in policy + property + tech regulation
- BSE: NSE-listed (.NS) ETFs, TER ≤ 0.15%, RBI policy + FII flows + India growth

**Step 2 — yfinance enrichment:**
- Fetches 1-year price history: `current_price`, `momentum_3m` (63 bars), `momentum_1m` (21 bars), `ytd_return`
- Fetches `expense_ratio`, `aum_b` from yfinance `.info` — overrides Gemini estimate if available
- TER fallback priority: yfinance info → Gemini `ter_pct` estimate → None

**Step 3 — Targeted news fetch (DDGS):**
- Builds one query per market using the actual recommended ticker symbols
- e.g. `"VXUS SPDW IEFA VEA SCHF IXUS international global ETF macro outlook"`
- Returns `{market: [headline, ...]}` passed as `all_macro_news`

**Step 4 — TER filter:**
- `ter ≤ ter_threshold` → included in `filtered_tickers`
- `ter is None` → included (auditor can flag)

**Reads:** `ter_threshold`
**Writes:** `all_etf_data`, `all_macro_news`, `filtered_tickers`

**Key functions:**
```python
_ask_gemini_universe(market) -> List[Dict]          # one Gemini call per market
_build_universe(ter_threshold) -> Dict[str, List]   # calls Gemini for all 3 markets
_fetch_yfinance_batch(tickers) -> Dict[str, Dict]
_fetch_news_for_tickers(market_tickers) -> Dict[str, List[str]]
```

---

### Node 2 — Signal Scorer (`agents/signal_scorer.py`)

**Gemini call — falls back to VADER if key missing.**

- Sends all `filtered_tickers` metrics + region headlines to `gemini-2.0-flash`
- Receives per-ticker sentiment score (0.0–1.0), boom triggers, macro summary as JSON
- On failure: VADER regional sentiment propagated to each ticker in that region

**Reads:** `filtered_tickers`, `all_etf_data`, `all_macro_news`
**Writes:** `sentiment_scores`, `boom_triggers`, `macro_summary`

**Gemini config:** `gemini-2.0-flash`, `temperature=0.1`, `response_mime_type="application/json"`

**System prompt (summary):** Quantitative ETF signal scorer. Score each ETF 0.0–1.0. Identify boom triggers. Return JSON with keys `scores`, `boom_triggers`, `macro_summary`.

**Score bands:**
- `0.00 – 0.35` Bearish / risk-off
- `0.35 – 0.65` Neutral / mixed
- `0.65 – 1.00` Bullish / risk-on

**VADER fallback formula:**
```
compound_avg = mean of VADER compound scores for region headlines
region_score = 0.10 + (compound_avg + 1) / 2 × 0.80   # normalised to [0.10, 0.90]
```

---

### Node 3 — Portfolio Optimizer (`agents/portfolio_optimizer.py`)

**Gemini call for rationales only. Allocation math is pure Python via `simulator.allocator`.**

**Step 1 — Compute scores:**
```
expense_score_i  = max(0, 1 − TER_i / ter_threshold)   # unknown TER → 0.50
consensus_score_i = 0.60 × sentiment_score_i + 0.40 × expense_score_i
```

**Step 2 — Build ranked list** sorted by `consensus_score` DESC (tiebreak: `momentum_3m` DESC).

**Step 3 — Allocate** via `simulator.allocator.compute_allocation()`:
```
Core bucket    = top core_count ETFs, budget = sip_amount × core_pct
Satellite      = next (top_n − core_count) ETFs, budget = sip_amount × (1 − core_pct)
Weight_i       = consensus_score_i / Σ consensus_score_bucket
monthly_usd_i  = weight_i × bucket_budget
```

**Step 4 — Retry cap enforcement** (if `audit_retry_count > 0`):
```python
_apply_violation_caps(plan, violations, max_position_pct, max_region_pct, sip_amount)
```
- Caps each position at `sip_amount × max_position_pct`
- Caps each region total at `sip_amount × max_region_pct`

**Step 5 — Gemini rationales** (one sentence per selected ETF).

**Reads:** `filtered_tickers`, `all_etf_data`, `sentiment_scores`, `ter_threshold`, SIP config, `risk_violations`, `audit_retry_count`
**Writes:** `expense_scores`, `consensus_scores`, `allocation_plan`, `proposed_orders`, `optimizer_notes`

**Gemini config:** `gemini-2.0-flash`, `temperature=0.2`, JSON mode

---

### Node 4 — Risk Auditor (`agents/risk_auditor.py`)

**Hard Python rules first. Gemini explains violations in natural language.**

**Five hard rules:**

| Rule | ID | Check | Violation message prefix |
|------|----|-------|--------------------------|
| 1 | MAX_POSITION | No ETF > `max_position_pct × sip_amount` | `RULE_1_VIOLATION:` |
| 2 | MAX_REGION | No region > `max_region_pct × sip_amount` | `RULE_2_VIOLATION:` |
| 3 | MIN_POSITION | No ETF allocated < $1.00 | `RULE_3_VIOLATION:` |
| 4 | NO_DUPLICATE | Ledger has no entry for current calendar month | `RULE_4_VIOLATION:` |
| 5 | MIN_ORDERS | At least 1 proposed order exists | `RULE_5_VIOLATION:` |

> Rule 4 is bypassed when `dry_run=True` **or** `force=True`.

**`force` vs `dry_run`:**
- `dry_run=True` — paper mode (Node 5 simulates fills). Implicitly bypasses Rule 4.
- `force=True` — bypass Rule 4 only (allow re-investment same calendar month). Broker mode unaffected.

**Routing:**
```python
MAX_RETRIES = 2

if no violations:                           → approved=True  → broker
if violations + retry_count < MAX_RETRIES:  → approved=False → optimizer (loop)
if violations + retry_count >= MAX_RETRIES: → approved=False → logger (abort)
```

**Reads:** `proposed_orders`, `sip_amount`, `max_position_pct`, `max_region_pct`, `ledger_path`, `audit_retry_count`, `dry_run`, `force`
**Writes:** `risk_approved`, `risk_violations`, `risk_audit_notes`, `audit_retry_count`

**Gemini config:** `gemini-2.0-flash`, `temperature=0.1`, JSON mode

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

**Reads:** `proposed_orders`, `dry_run`
**Writes:** `execution_results`, `execution_status`, `usd_inr_rate`

**`execution_status` values:** `"success"` | `"partial"` | `"dry_run"` | `"failed"` | `"aborted"`

---

### Node 6 — Execution Logger (`agents/execution_logger.py`)

**No LLM. Terminal node — always reached.**

Writes two persistence targets and prints a formatted terminal summary.

**1. `simulator/outputs/portfolio_ledger.json`** (shared with Streamlit dashboard)
- Appends via `simulator.ledger.build_and_append_entry()`
- Skipped if `execution_status == "aborted"` or `allocation_plan` is None
- Price source: `execution_results` → fallback to `all_etf_data.current_price`

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
    sip_amount: float             # USD, e.g. 500.0
    top_n: int                    # total ETFs to invest in
    core_count: int               # size of core bucket
    core_pct: float               # fraction of SIP for core (e.g. 0.70)
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

    # Node 5
    execution_results: List[ExecutionResult]
    execution_status: str
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
    current_price: Optional[float]; currency: str   # "USD" | "INR"
    data_source: str; fetch_error: Optional[str]

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

The ledger is shared — `scheduler.py` and the MAS both write to the same file. The Streamlit app reads it transparently.

---

## CLI

### `sip_execution_mas` — direct MAS execution

```bash
# Default: dry-run, $500 SIP, top-10, 70/30 core/sat
python -m sip_execution_mas

# Custom SIP and sizing
python -m sip_execution_mas --sip 750 --top 12 --core 6

# Custom risk limits
python -m sip_execution_mas --max-position 0.20 --max-region 0.60

# Stricter TER filter (Gemini still discovers; TER filter applied after)
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
| `--top` | int | 10 | Total ETFs to invest in |
| `--core` | int | 5 | Core bucket size |
| `--core-pct` | float | 0.70 | Core fraction of SIP |
| `--ter` | float | 0.70 | TER ceiling in % (applied after Gemini discovery) |
| `--max-position` | float | 0.15 | Max fraction per ETF |
| `--max-region` | float | 0.50 | Max fraction per region |
| `--live` | flag | off | Disable dry-run, use broker API |
| `--force` | flag | off | Bypass Rule 4 (allow re-invest same month) |
| `--ledger` | str | auto | Path to portfolio_ledger.json |
| `--run-id` | str | auto | Custom run identifier |

### `simulator/scheduler.py` — monthly APScheduler

```bash
# Execute immediately (dry-run)
python -m simulator.scheduler --now

# Start persistent scheduler (fires 09:00 on 1st of each month)
python -m simulator.scheduler

# Show ledger status
python -m simulator.scheduler --status

# Force re-investment this month
python -m simulator.scheduler --now --force

# Live execution
python -m simulator.scheduler --now --live
```

### `simulator/main.py` — backtest simulator

```bash
# Backtest using stored rankings
python -m simulator.main

# Generate fresh Gemini rankings via sip_execution_mas then backtest
python -m simulator.main --run-sip-mas

# Generate fresh rankings via etf-selection-mas (requires ANTHROPIC_API_KEY)
python -m simulator.main --run-mas

# Custom period and SIP
python -m simulator.main --sip 1000 --months 24 --top 10
```

### `simulator/app.py` — Streamlit dashboard

```bash
streamlit run sip_execution_mas/simulator/app.py
```

---

## Environment Variables

| Variable | Required | Used By | Notes |
|----------|----------|---------|-------|
| `GEMINI_API_KEY` | Recommended | Nodes 1, 2, 3, 4 | Falls back to seed list (Node 1) / VADER (Node 2) if missing |
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
from simulator.allocator import compute_allocation        # Node 3
from simulator.ledger import (                            # Nodes 4, 6
    load_ledger, save_ledger, build_and_append_entry,
    already_invested_this_month, aggregate_holdings, ledger_summary,
)
```

`simulator/scheduler.py` also adds `fin-agents/` to sys.path to resolve `sip_execution_mas.graph.workflow`:

```python
from sip_execution_mas.graph.workflow import run_sip_execution
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

`sip_execution_mas` supersedes `etf-selection-mas` as an end-to-end execution system. Node 1 now uses Gemini to dynamically discover the ETF universe on every run (no stale rankings file). The `simulator/` package is embedded as a sub-package to keep all execution infrastructure co-located.

---

## Known Limitations

- Dhan broker (Indian ETFs) is always paper — real Dhan API integration is a stub.
- `audit_retry_count` increments on every rejection. After `MAX_RETRIES=2` the run aborts even if only one minor rule was violated — by design, to prevent infinite loops.
- `portfolio_ledger.json` records the *planned* allocation, not the exact broker fill. For fill-exact records use `execution_log.csv`.
- Node 1 makes 3 Gemini calls (one per market) before the graph runs. If Gemini is slow this adds latency; the per-market fallback ensures no full failure.
- Gemini `response_mime_type="application/json"` occasionally wraps output in markdown; `_parse_json()` strips it.
- DDGS news can be empty during rate-limit windows; Node 2 VADER fallback handles this gracefully.
- `yfinance` TER data is often missing; Node 1 falls back to Gemini's `ter_pct` estimate, then `None`.
