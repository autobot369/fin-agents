# etf-selection-mas — Context Document

> Single source of truth for agents, contributors, and AI assistants working in this module.

## Purpose

A **5-node LangGraph pipeline** that researches 31 ETFs across three global markets, scores them using Claude-powered sentiment analysis, and produces a ranked selection report. It is a **report generator** — no money moves, no orders are placed.

Output feeds `simulator/` (via `rankings.json`) and was the predecessor to `sip_execution_mas/`.

---

## Architecture

### Graph topology (linear, no loop)

```
MultiMarketResearcher → ExpenseGuard → SentimentScorer → GlobalRankingNode → PortfolioArbiter → END
```

All nodes share one `ETFSelectionState` TypedDict. Each node reads from state and writes back a subset of keys.

---

## ETF Universe

| Region | Count | Exchange | Tickers |
|--------|-------|----------|---------|
| INTL   | 11    | NASDAQ   | VXUS, SPDW, IEFA, SCHY, VEA, EEM, ACWI, SCHF, IXUS, VWO, IEMG |
| HKCN   | 10    | NASDAQ   | CQQQ, KWEB, MCHI, FXI, CHIQ, GXC, FLCH, KURE, CNYA, ASHR |
| BSE    | 10    | NSE (.NS)| NIFTYBEES.NS, BANKBEES.NS, GOLDBEES.NS, JUNIORBEES.NS, ITBEES.NS, INFRABEES.NS, SETFNIF50.NS, CONSUMBEES.NS, PHARMABEES.NS, DIVOPPBEES.NS |

---

## Nodes

### Node 1 — MultiMarketResearcher (`agents/regional_researcher.py`)

**No LLM cost here except one summary call.**

- Fetches yfinance metrics for all 31 ETFs: `expense_ratio`, `aum_b`, `ytd_return`, `momentum_3m`, `momentum_1m`, `current_price`, `volume_avg_30d`
- Fetches macro + ticker-level news via DDGS and `yf.Ticker.news` (free, no key)
- Calls **Claude** once to write a ≤500-word global macro brief across INTL / HKCN / BSE

**Reads:** `ter_threshold`
**Writes:** `all_etf_data`, `all_macro_news`, `researcher_notes`

**Claude config:** `claude-sonnet-4-6`, `max_tokens=2500`, adaptive thinking

**System prompt (summary):** Senior macro strategist, under 500 words, three sections (INTL / HKCN / BSE), cite figures, flag data gaps.

---

### Node 2 — ExpenseGuard (`agents/expense_guard.py`)

**Pure Python — no LLM.**

- TER = `None` → **KEEP** (benefit of doubt)
- TER ≤ `ter_threshold` → **PASS** → `all_filtered_tickers`
- TER > `ter_threshold` → **PRUNE** → `all_pruned_tickers`

**Reads:** `all_etf_data`, `ter_threshold`
**Writes:** `all_filtered_tickers`, `all_pruned_tickers`

---

### Node 3 — SentimentScorer (`agents/sentiment_scorer.py`)

**Most LLM calls happen here.**

- Groups articles into batches of ≤8
- Calls **Claude** per batch: each article gets `sentiment_score` in [-1.0, +1.0] (0.05 steps)
- Claude confirms which `BOOM_TRIGGERS` are explicitly evidenced (not inferred)
- Weight: yfinance articles = 1.3×, DDGS articles = 1.0×
- Min-max normalises per-region scores to [0.0, 1.0] → per-ticker scores

**Reads:** `all_macro_news`, `all_filtered_tickers`, `all_etf_data`
**Writes:** `all_sentiment_scores`, `boom_triggers_fired`, `raw_article_scores`, `sentiment_narrative`

**Claude config:** `claude-sonnet-4-6`, `max_tokens=4096`, adaptive thinking

**System prompt (summary):** Quantitative macro analyst. For each article: assign `sentiment_score`, confirm `BOOM_TRIGGERS_CONFIRMED`, write one-sentence `reason`. Return valid JSON only.

**Expected JSON per batch:**
```json
{
  "articles": [
    {
      "title": "...",
      "sentiment_score": 0.65,
      "boom_triggers_confirmed": ["china_tech_rerating"],
      "reason": "One sentence."
    }
  ],
  "regional_summary": "2–3 sentence macro narrative."
}
```

**Boom triggers catalogue:**

| Key | Direction | Boost | Applies To | Example Keywords |
|-----|-----------|-------|-----------|-----------------|
| `china_tech_rerating` | Bullish | +0.35 | HKCN | deepseek, kweb rally, regulatory thaw |
| `rbi_pivot` | Bullish | +0.30 | BSE | rbi rate cut, rbi accommodative |
| `fed_soft_landing` | Bullish | +0.25 | All | fed soft landing, fomc hold cut |
| `india_inflation_target` | Bullish | +0.25 | BSE | india cpi 2.75, inflation target met |
| `china_gdp_beat` | Bullish | +0.25 | HKCN, INTL | china gdp beat, china gdp 5% |
| `em_broad_rally` | Bullish | +0.15 | INTL, BSE | em inflows, em rotation |
| `china_tech_crackdown` | Bearish | -0.20 | HKCN | tech crackdown, delisting risk |
| `india_fiscal_slippage` | Bearish | -0.15 | BSE | fiscal deficit widens, bond yield spike |
| `us_recession_risk` | Bearish | -0.15 | INTL, HKCN | us recession 2026, us hard landing |

---

### Node 4 — GlobalRankingNode (`graph/workflow.py`)

**Pure Python — no LLM.**

```
expense_score_i  = max(0.0, 1.0 − TER_i / ter_threshold)   # unknown TER → 0.50
consensus_score_i = 0.60 × sentiment_score_i + 0.40 × expense_score_i
```

**Reads:** `all_sentiment_scores`, `all_filtered_tickers`, `all_etf_data`, `ter_threshold`
**Writes:** `all_expense_scores`, `all_consensus_scores`

Constants: `SENTIMENT_WEIGHT = 0.60`, `EXPENSE_WEIGHT = 0.40`

---

### Node 5 — PortfolioArbiter (`agents/portfolio_arbiter.py`)

**One LLM call for rationales. Report writing is pure Python.**

- Sorts all filtered ETFs by `consensus_score` DESC (tiebreak: `momentum_3m` DESC)
- Selects `top_n` (default 20)
- Calls **Claude** to generate a one-sentence boom rationale per ETF (≤20 words, cite macro trigger)
- Writes `TOP_20_BOOM_REPORT.md` (human-readable)
- Writes `rankings.json` (machine-readable, consumed by `simulator/`)

**Reads:** All state keys
**Writes:** `sentiment_rationales`, `final_ranking`, `output_markdown`

**Claude config:** `claude-sonnet-4-6`, `max_tokens=3000`, adaptive thinking

**System prompt (summary):** Concise ETF analyst. One sentence max 20 words per ETF, specific macro trigger, valid JSON only.

**Expected JSON:**
```json
{"rationales": {"VXUS": "Benefits from Fed soft landing...", "KWEB": "..."}}
```

---

## State Schema (`graph/state.py`)

```python
class ETFRecord(TypedDict):
    ticker: str
    name: str
    market: str           # "NASDAQ" | "NSE"
    region: str           # "INTL" | "HKCN" | "BSE"
    expense_ratio: Optional[float]   # decimal: 0.0007 = 0.07%
    aum_b: Optional[float]
    ytd_return: Optional[float]
    momentum_3m: Optional[float]
    momentum_1m: Optional[float]
    current_price: Optional[float]
    volume_avg_30d: Optional[int]
    data_source: str      # "yfinance" | "fallback"
    fetch_error: Optional[str]

class ETFSelectionState(TypedDict):
    ter_threshold: float
    top_n: int
    all_etf_data: Dict[str, ETFRecord]
    all_macro_news: Dict[str, List[Dict]]
    researcher_notes: str
    all_filtered_tickers: List[str]
    all_pruned_tickers: List[str]
    boom_triggers_fired: Annotated[List[str], operator.add]  # accumulates across batches
    raw_article_scores: Dict[str, List[Dict]]
    all_sentiment_scores: Dict[str, float]
    sentiment_narrative: str
    all_expense_scores: Dict[str, float]
    all_consensus_scores: Dict[str, float]
    sentiment_rationales: Dict[str, str]
    final_ranking: List[Dict[str, Any]]
    output_markdown: str
```

---

## Outputs

| File | Path | Format | Consumer |
|------|------|--------|----------|
| Markdown report | `outputs/TOP_20_BOOM_REPORT.md` | Markdown | Human |
| Machine rankings | `outputs/rankings.json` | JSON | `simulator/scheduler.py`, `sip_execution_mas/` |

**`rankings.json` schema:**
```json
{
  "generated_at": "2026-03-12",
  "ter_threshold": 0.007,
  "boom_triggers_fired": ["china_tech_rerating"],
  "rankings": [
    {
      "rank": 1,
      "ticker": "IEFA",
      "name": "...",
      "region": "INTL",
      "consensus_score": 0.81,
      "sentiment_score": 0.75,
      "expense_score": 0.90,
      "sentiment_rationale": "..."
    }
  ]
}
```

---

## CLI

```bash
# From fin-agents/etf-selection-mas/
python main.py                      # top-20, TER ≤ 0.70%
python main.py --ter 0.50           # stricter TER ceiling
python main.py --top 10             # top-10
python main.py --ter 0.65 --top 20
```

**Env vars:**
- `ANTHROPIC_API_KEY` — required

---

## Data Tools (`tools/`)

| File | Purpose | API Key? |
|------|---------|----------|
| `etf_data.py` | yfinance metrics + FALLBACK_TER dict | No |
| `news_search.py` | DDGS macro queries + yfinance.Ticker.news | No |
| `bse_data.py` | NSE ETF registry, `.NS` ticker metadata | No |
| `tavily_news.py` | Higher-quality news (optional upgrade) | Yes — Tavily |

**Macro news queries per region:**
- **INTL:** Fed soft landing, MSCI World momentum, USD weakness EM rally, international ETF outlook
- **HKCN:** China GDP, PBOC stimulus, China tech rerating DeepSeek, Hong Kong internet stocks
- **BSE:** RBI pivot, India inflation CPI target, India GDP Nifty 50, Fed vs RBI rate differential

---

## Dependencies (`requirements.txt`)

```
langgraph>=0.2.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
anthropic>=0.40.0
yfinance>=0.2.40
pandas>=2.0.0
duckduckgo-search>=6.0.0
tabulate>=0.9.0
python-dotenv>=1.0.0
```

---

## Relationship to Other Modules

```
etf-selection-mas  ──→  rankings.json  ──→  simulator/scheduler.py  ──→  portfolio_ledger.json
                                       ──→  sip_execution_mas/              ↓
                                                                      Streamlit dashboard
```

`etf-selection-mas` is the **research layer**. `sip_execution_mas` internalises and extends it with Gemini scoring, allocation logic, risk guardrails, and broker execution. Both can feed the same `simulator/` Streamlit dashboard.

---

## Known Limitations

- Runs once per invocation (no scheduler). Use `simulator/scheduler.py` for monthly automation.
- `boom_triggers_fired` uses `operator.add` annotation — it **accumulates** across all SentimentScorer batch calls (intended).
- yfinance `info` dict is unreliable on `expense_ratio`; `FALLBACK_TER` dict is the safety net.
- DDGS occasionally rate-limits; `0.4s` sleep between calls mitigates this.
- NSE ETF `volume_avg_30d` is unreliable from yfinance; treat as indicative only.
