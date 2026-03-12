# ticker-research-mas — Context Document

> Single source of truth for agents, contributors, and AI assistants working in this module.

## Purpose

A **7-node LangGraph debate pipeline** that researches any stock ticker, runs a structured two-round Bull vs Bear debate, fact-checks every numerical claim, and delivers a final arbitrated investment forecast with recommendation, conviction level, price targets, and a volatility warning. It is a **research and recommendation system** — no money moves.

---

## Architecture

### Graph topology (linear, no loop)

```
researcher → bull_initial → bear_response → bull_rebuttal → bear_counter → fact_checker → arbiter → END
```

Every node appends to `debate_history` via LangGraph's `Annotated[List, operator.add]` — entries accumulate without overwriting. **7 debate entries total** by the time the Arbiter runs.

---

## Nodes

### Node 1 — Researcher (`agents/researcher.py`)

**One Claude call. Produces the ground truth anchor and a 9-section research brief.**

- Calls all four data tools: `get_market_data()`, `fetch_recent_news()`, `fetch_risk_factors()`, `fetch_earnings_sentiment()`
- Builds `ground_truth` dict (72+ verified quantitative fields — **immutable** for all downstream nodes)
- Calls Claude once to produce a structured 9-section brief

**Reads:** `ticker`, `investment_horizon`, `risk_aversity`
**Writes:** `research_data`, `ground_truth`, `debate_round=0`, `debate_history=[]`

**Claude config:** `claude-opus-4-6`, `max_tokens=3500`, adaptive thinking

**9-section research brief:**
1. Price Action & Trend
2. Technical Signals
3. Fundamental Snapshot
4. Key Bullish Catalysts
5. Key Bearish Risks
6. Analyst Consensus
7. SEC Risk Factors
8. Management Tone
9. Futures Basis Signal

**`ground_truth` key groups:**

| Group | Keys |
|-------|------|
| Identity | `company_name`, `sector` |
| Futures Basis | `index_name`, `futures_symbol`, `futures_price`, `spot_price`, `futures_basis`, `basis_pct`, `basis_state` |
| Volume / Volatility | `futures_volume`, `futures_volume_avg30`, `futures_volume_ratio`, `volatility_level` |
| Price & Range | `current_price`, `high_52w`, `low_52w`, `price_change_6mo_pct` |
| Trend | `ma200` |
| Momentum | `rsi_14`, `macd_bullish`, `bb_position_pct` |
| Valuation | `pe_trailing`, `pe_forward`, `peg_ratio`, `price_to_book` |
| Fundamentals | `revenue_growth_yoy`, `earnings_growth_yoy`, `profit_margin`, `market_cap_b` |
| Analyst | `analyst_target_price`, `analyst_recommendation`, `short_percent_float` |
| Alt Data | `sec_filing_type`, `sec_risk_factors_excerpt`, `earnings_call_tone`, `earnings_call_guidance`, `earnings_call_sentiment_score` |

---

### Node 2 — Bull Initial (`agents/bull.py → bull_initial_node`)

**One Claude call. Opens the bull thesis.**

Persona: Aggressively optimistic analyst; every claim data-backed; high conviction.

**Reads:** `research_data`, `investment_horizon`, `risk_aversity`
**Writes:** `bull_case`, `debate_round=1`, `debate_history` (+1 entry)

**Claude config:** `claude-opus-4-6`, `max_tokens=2500`, adaptive thinking

**Horizon guidance injected via `_context_block()`:**

| Horizon | Weighting | Focus |
|---------|-----------|-------|
| `short` (1–3 mo) | Technicals 60% | RSI, MACD, futures basis, near-term catalysts |
| `mid` (6–12 mo) | Equal 50/50 | Balance technicals and fundamentals |
| `long` (1–3 yr) | Fundamentals 70% | FCF, moat, CAGR, margins; technicals are noise |

**Thesis sections:** Core Value Proposition · Technical Setup · Fundamental Drivers · Catalyst Pipeline · Risk-Adjusted Return Profile

---

### Node 3 — Bear Response (`agents/bear.py → bear_response_node`)

**One Claude call. Challenges every bull claim.**

Persona: Skeptical short-seller, forensic accounting background; exposes overvaluation and execution risk.

**Reads:** `research_data`, `bull_case`, `investment_horizon`, `risk_aversity`
**Writes:** `bear_case`, `debate_history` (+1 entry)

**Claude config:** `claude-opus-4-6`, `max_tokens=2500`, adaptive thinking

**Prompt tasks:** Quote each Bull claim → counter with data → add overlooked risks → propose bear-case price target.

---

### Node 4 — Bull Rebuttal (`agents/bull.py → bull_rebuttal_node`)

**One Claude call. Bull responds to Bear's critique.**

**Reads:** `research_data`, `bull_case`, `bear_case`, `investment_horizon`, `risk_aversity`
**Writes:** `bull_rebuttal`, `debate_round=2`, `debate_history` (+1 entry)

**Prompt tasks:** Address each bear objection by number → flip bearish evidence where possible → introduce additional data → revise price target if warranted.

---

### Node 5 — Bear Counter (`agents/bear.py → bear_counter_node`)

**One Claude call. Bear's closing argument before the Arbiter.**

**Reads:** `research_data`, `bear_case`, `bull_rebuttal`, `investment_horizon`, `risk_aversity`
**Writes:** `bear_counter`, `debate_history` (+1 entry)

**Prompt tasks:** Find Bull's weakest points → reinforce strongest bear arguments → highlight biggest tail risk → maintain or revise downside target.

---

### Node 6 — Fact Checker (`agents/fact_checker.py`)

**One Claude call. Cross-checks every numerical claim against `ground_truth`.**

**Tolerance rules:**
- ✅ **CONFIRMED** — matches `ground_truth` within 5%
- ❌ **CONTRADICTED** — direct conflict with `ground_truth`
- ⚠️ **LOGICAL MISUSE** — correct number, wrong conclusion (e.g. high RSI = overbought, not momentum)
- 🔍 **UNVERIFIABLE** — claim not in `ground_truth`

**Reads:** Full debate transcript, `ground_truth`
**Writes:** `fact_check_report`, `debate_history` (+1 entry)

**Claude config:** `claude-opus-4-6`, `max_tokens=2000`, adaptive thinking

**Report format:**
```
## Fact-Check Report — {TICKER}
### ✅ Confirmed Claims      (table: Agent | Claimed | Ground Truth | Note)
### ❌ Contradicted Claims   (table: Agent | Claimed | Ground Truth | Correction)
### ⚠️ Logical Misuse        (table: Agent | Claim | Why Logic Fails)
### 🔍 Unverifiable Claims   (table: Agent | Claim)
### Summary
  - Bull reliability score: X/10
  - Bear reliability score: X/10
  - Most egregious error: ...
  - Arbiter guidance: ...
```

---

### Node 7 — Arbiter (`agents/arbiter.py`)

**One Claude call. Scores the debate and delivers the final investment forecast.**

**Reads:** Full state (all debate rounds, fact-check report, ground_truth)
**Writes:** `final_forecast`

**Claude config:** `claude-opus-4-6`, `max_tokens=4500`, adaptive thinking

Persona: Objective Portfolio Manager, 20 years experience. Ruling anchored to three objective signals.

#### Signal A — Futures Basis (Price Discovery)

Sector → futures mapping:

| Sector | Futures | Index |
|--------|---------|-------|
| Technology | NQ=F | Nasdaq 100 |
| Communication Services | NQ=F | Nasdaq 100 |
| Consumer Cyclical | NQ=F | Nasdaq 100 |
| All others | ES=F | S&P 500 |

- **Contango** (`basis_pct > 0`): +1 pt to Bull Interpretation score
- **Backwardation** (`basis_pct < 0`): +1 pt to Bear Interpretation score
- **Neutral** (`|basis_pct| < 0.1%`): no adjustment

#### Signal B — Volume-Induced Volatility

| `futures_volume_ratio` | Level | Effect |
|------------------------|-------|--------|
| < 1.5× | NORMAL | No action |
| 1.5–2.0× | ELEVATED | Note in report; risk-averse users lose conviction one step |
| ≥ 2.0× | SPIKE | ⚠️ Volatility Warning mandatory |

Risk aversity adjustments:

| Range | Effect |
|-------|--------|
| 0.0–0.3 (risk-seeking) | Warning noted; no recommendation change |
| 0.3–0.7 (balanced) | SPIKE drops conviction one step (HIGH→MEDIUM, MEDIUM→LOW) |
| 0.7–1.0 (risk-averse) | ELEVATED drops conviction; SPIKE downgrades recommendation + conviction→LOW; Bear weight ×1.5 |

#### Signal C — Contextual Scorecard (10 pts per agent)

**Recognition (0–5):** Correctly cited price/technicals, valuation, SEC filings, earnings tone, futures basis.
Deduction: −1 per ❌ CONTRADICTED from fact-check.

**Interpretation (0–5):** Identified mispricing, non-obvious insight, correct technical read, correct futures framing, key catalyst or tail risk.
Bonus: +1 for correctly calling futures basis implication.
Deduction: −1 per ⚠️ LOGICAL MISUSE from fact-check.

#### Final report structure

```markdown
# {TICKER} — Arbiter's Final Ruling

## 1. Signal Analysis
### A. Futures Basis       (table: Field | Value)
### B. Volume-Induced Volatility  (table: Field | Value)

## 2. Contextual Scorecard
(table: Dimension | Bull pts | Bear pts — all 10 rows)
**TOTAL: Bull X/10 | Bear Y/10**

## 3. Final Synthesis
(2–3 paragraphs: who won, why, how signals tilted the ruling)

## 4. Recommendation
| Scenario | Price Target | Probability | Key Trigger |
**Recommendation:** STRONG BUY / BUY / HOLD / SELL / STRONG SELL
**Conviction:** HIGH / MEDIUM / LOW
**Time Horizon:** ...
⚠️ **Volatility Warning:** YES — reason / NO

## 5. Risks to Monitor
1. ...
```

---

## State Schema (`graph/state.py`)

```python
class MarketState(TypedDict):
    # User inputs
    ticker: str
    investment_horizon: str           # "short" | "mid" | "long"
    risk_aversity: float              # 0.0 (risk-seeking) → 1.0 (risk-averse)

    # Node 1
    research_data: Dict[str, Any]     # market_data, news, sec_filings, earnings_sentiment, summary
    ground_truth: Dict[str, Any]      # Verified quantitative anchor — never mutated

    # Debate
    bull_case: str                    # Round 1
    bear_case: str                    # Round 1
    bull_rebuttal: str                # Round 2
    bear_counter: str                 # Round 2
    debate_history: Annotated[List[Dict[str, str]], operator.add]
    debate_round: int

    # Outputs
    fact_check_report: str
    final_forecast: str
```

---

## Data Tools (`tools/`)

### `tools/market_data.py` — `get_market_data(ticker) -> Dict`

- 6-month stock history + 60-day futures history via yfinance
- RSI (14-period Wilder), MACD (12/26/9), MAs (20/50/200), Bollinger Bands (20-period, 2σ)
- Futures basis: `futures_price − spot_price`, `basis_pct`, `basis_state`
- Volume ratio: `futures_volume_today / mean(last 30 days)` → `volatility_level`
- Full fundamentals: P/E, PEG, P/B, margins, ROE, D/E, FCF, analyst target
- `_safe_round(val, n)` guards all fields — returns `None` rather than crashing

### `tools/news_search.py` — `fetch_recent_news(ticker, company_name) -> List[Dict]`

- DuckDuckGo, two queries per ticker (stock + macro/industry)
- No API key required. Deduplicates by title.
- Returns: `[{title, body (≤400 chars), date, source}]`

### `tools/earnings_sentiment.py` — `fetch_earnings_sentiment(ticker, company_name) -> Dict`

- DDG searches for earnings call content (3 queries)
- Claude JSON extraction of: `tone`, `key_phrases`, `unusual_focus_topics`, `guidance_direction`, `sentiment_score` (1–10), `one_line_summary`
- **Claude config:** `claude-opus-4-6`, `max_tokens=1200`, no thinking (JSON output)
- Neutral fallback dict if no content found

### `tools/sec_filings.py` — `fetch_risk_factors(ticker) -> Dict`

- Free SEC EDGAR API (`data.sec.gov`) — no key required
- Resolves CIK → fetches latest 10-Q (fallback 10-K) → regex-extracts Item 1A Risk Factors
- 0.5s sleep between calls (respects 10 req/s EDGAR limit)
- Returns: `{filing_type, filing_date, risk_factors (≤2500 chars), url}`

---

## Entry Points

### CLI — `main.py`

```bash
# Defaults: mid horizon, 0.5 risk aversity
python main.py AAPL

# Short-term, risk-averse, save file, verbose
python main.py TSLA --horizon short --risk 0.9 --save --verbose

# Long-term, risk-seeking
python main.py NVDA --horizon long --risk 0.1

# All flags
python main.py MSFT --horizon mid --risk 0.5 --save --verbose
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `ticker` | str | required | Stock symbol |
| `--horizon` | str | `"mid"` | `"short"` \| `"mid"` \| `"long"` |
| `--risk` | float | `0.5` | Risk aversity 0.0–1.0 |
| `--save` | flag | off | Write `{TICKER}_{timestamp}_forecast.md` |
| `--verbose` | flag | off | Print full debate transcript to terminal |

### Programmatic API — `api.py`

```python
from api import run_analysis

result = run_analysis(
    ticker="AAPL",
    investment_horizon="mid",   # "short" | "mid" | "long"
    risk_aversity=0.5,
    save=True,
    verbose=False,
)

result["recommendation"]      # "BUY"
result["conviction"]          # "HIGH" | "MEDIUM" | "LOW"
result["volatility_warning"]  # True / False
result["basis_state"]         # "Contango" | "Backwardation" | None
result["volatility_level"]    # "NORMAL" | "ELEVATED" | "SPIKE" | None
result["final_forecast"]      # Full Markdown report string
result["ground_truth"]        # Verified numbers dict
result["fact_check_report"]   # Per-claim verification string
result["debate_history"]      # List of {role, content} dicts
result["saved_to"]            # File path string or None
```

---

## Model Configuration

| Agent | Model | Max Tokens | Thinking |
|-------|-------|-----------|---------|
| Researcher | `claude-opus-4-6` | 3500 | adaptive |
| Bull (R1 + R2) | `claude-opus-4-6` | 2500 | adaptive |
| Bear (R1 + R2) | `claude-opus-4-6` | 2500 | adaptive |
| Fact Checker | `claude-opus-4-6` | 2000 | adaptive |
| Arbiter | `claude-opus-4-6` | 4500 | adaptive |
| Earnings Sentiment | `claude-opus-4-6` | 1200 | none (JSON) |

**Total Claude calls per run: 8** (Researcher + Bull×2 + Bear×2 + FactChecker + Arbiter + EarningsSentiment)

---

## Environment Variables

| Variable | Required | Notes |
|----------|----------|-------|
| `ANTHROPIC_API_KEY` | Yes | All Claude calls |

Loaded from `fin-agents/.env` at startup via `python-dotenv`.

---

## Dependencies (`requirements.txt`)

```
langgraph>=0.2.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
anthropic>=0.40.0
yfinance>=0.2.40
duckduckgo-search>=6.0.0
python-dotenv>=1.0.0
pandas>=2.0.0
tabulate>=0.9.0
```

---

## Critical Shared Function

**`_extract_text(response)`** — defined in `agents/researcher.py`, imported by all agents.

Strips adaptive thinking blocks from Claude responses and returns only the plain-text content:

```python
def _extract_text(response) -> str:
    if hasattr(response, "content") and isinstance(response.content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict)
            else getattr(block, "text", "")
            for block in response.content
            if (isinstance(block, dict) and block.get("type") == "text")
            or (hasattr(block, "type") and block.type == "text")
        ]
        return "\n".join(parts).strip()
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content.strip()
    return str(response)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `ground_truth` built once in Node 1, never mutated | Clean anchor; prevents Bull/Bear from disputing the verified data; fact-checker has a fixed reference |
| `debate_history` uses `operator.add` | LangGraph merges appends automatically; no node overwrites a prior round |
| `_context_block()` in Bull/Bear | DRY: single function injects `investment_horizon` + `risk_aversity` preamble into all 4 debate prompts consistently |
| Arbiter rules built at runtime | `horizon_rules` and `risk_instruction` read live state values; no stale module-level constants |
| Futures fetched on 60-day history | Need 30-day rolling average for `futures_volume_ratio`; 2-day period was insufficient |
| `_safe_round()` throughout market_data | Prevents crash on `round(None)` for missing yfinance fields |
| SEC EDGAR + DDG (no paid APIs) | `data.sec.gov` free, no key; DDG free, no key; fully self-contained |
| Fact-checker injected separately to Arbiter | Prevents circular self-reference in scoring; FC is external referee, not a debate participant |
| Linear graph, no conditionals | Fixed 2-round structure; dynamic routing adds complexity without benefit for a known debate format |

---

## Relationship to Other Modules

`ticker-research-mas` is **standalone** — it does not share state, ledgers, or outputs with `etf-selection-mas` or `sip_execution_mas`. It analyses individual stocks; the other two systems analyse ETF baskets.

```
ticker-research-mas  →  TICKER_forecast.md       (human, optional)
                     →  AnalysisResult dict       (programmatic via api.py)

etf-selection-mas    →  rankings.json
sip_execution_mas    →  portfolio_ledger.json + execution_log.csv
```

---

## Known Limitations

- `ma200` is `None` for stocks with fewer than 200 trading days of history (recent IPOs).
- yfinance `info` dict rate-limits intermittently; `_safe_round()` returns `None` cleanly rather than crashing.
- DDGS v8+ requires context manager (`with DDGS() as ddgs`); old `ddg()` call pattern removed.
- SEC EDGAR regex extraction fails on non-standard HTML; fallback returns `"Risk factors could not be parsed."`.
- Earnings sentiment is derived from DDG snippets (not full transcripts) — tone is approximate.
- `debate_history` holds 7 entries; fact-checker is entry 6 and is visible to Arbiter (intended).
- No output file is written unless `--save` / `save=True` is explicitly set.
- `pydantic v1` warning on Python 3.14 from `langchain_core` — cosmetic, no functional impact.

---

## Session Log

| Date | Change |
|------|--------|
| 2026-03-09 | Initial build: all agents, graph, tools, CLI |
| 2026-03-09 | Added guardrails: `fact_checker.py`, `sec_filings.py`, `earnings_sentiment.py`, `ground_truth` anchor |
| 2026-03-09 | Arbiter V2: Signal A/B/C scoring, futures basis bonus, volume thresholds, 10-pt rubric |
| 2026-03-11 | `api.py` programmatic interface; `investment_horizon` + `risk_aversity` in state; `_context_block()` helper; `--horizon` + `--risk` CLI flags; directory renamed to `ticker-research-mas` |
| 2026-03-12 | CONTEXT.md reformatted to match project-wide documentation standard |
