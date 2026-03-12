# Gemini Project Context: Ticker Research MAS

This document provides instructional context for Gemini CLI/IDE when working with the `ticker-research-mas` project. Keep it in sync with `CONTEXT.md`.

---

## Project Overview

A **Multi-Agent System (MAS)** that runs a structured bull-vs-bear debate on any stock ticker and produces a final investment forecast. Built with **Python + LangGraph + Claude Opus 4.6 (Adaptive Thinking)**.

The Arbiter scores the debate on three objective anchor signals — futures basis, volume-induced volatility, and contextual rigor — and its recommendation is calibrated by two user-supplied parameters: `investment_horizon` and `risk_aversity`.

---

## Architecture

Seven nodes wired linearly in a LangGraph `StateGraph`:

```
researcher → bull_initial → bear_response → bull_rebuttal → bear_counter → fact_checker → arbiter → END
```

| Node            | Role |
|-----------------|------|
| `researcher`    | Fetches yfinance market data (incl. index futures ES=F/NQ=F), DuckDuckGo news, SEC EDGAR risk factors, and earnings call sentiment. Emits a `ground_truth` dict and a 9-section research brief. |
| `bull_initial`  | Opens with an optimistic thesis grounded in data. Framed for the user's `investment_horizon`. |
| `bear_response` | Dismantles Bull's points one by one with specific data. Framed for the horizon. |
| `bull_rebuttal` | Round 2: Bull rebuts Bear's counter-arguments. |
| `bear_counter`  | Round 2: Bear's closing counter before the ruling. |
| `fact_checker`  | Scans all 4 debate turns for numerical claims and cross-checks each against `ground_truth`. Produces a table of ✅ ❌ ⚠️ 🔍 verdicts. |
| `arbiter`       | Issues the final ruling using a 3-signal framework (see below). |

---

## Arbiter V2 — Three-Signal Scoring Framework

### A. Futures Basis (Price Discovery)
Compares the front-month index future (e.g. NQ=F for Nasdaq, ES=F for S&P 500) against the spot index.
- **Contango** (futures > spot): +1 Interpretation bonus to Bull.
- **Backwardation** (spot > futures): +1 Interpretation bonus to Bear.

### B. Volume-Induced Volatility
Compares today's futures volume to its 30-day rolling average (`futures_volume_ratio`).
- Ratio ≥ 2.0× → ⚠️ **Volatility Warning** issued regardless of direction.
- Ratio 1.5–2.0× → ELEVATED; may reduce conviction for risk-averse users.

### C. Contextual Scoring (10-point rubric per agent)
- **Recognition (5 pts):** Did the agent accurately cite price/technical data, valuation, SEC filings, earnings call signal, and the futures basis?
- **Interpretation (5 pts):** Did the agent draw insightful, non-obvious conclusions from the data?
- Fact-Check deductions: −1 per ❌ CONTRADICTED claim, −1 per ⚠️ LOGICAL MISUSE.

### Horizon & Risk Adjustments
Injected live into the Arbiter's prompt each run:

| `investment_horizon` | Recognition weighting             | Price target |
|----------------------|-----------------------------------|--------------|
| `short`              | Technicals 60% / Fundamentals 40% | 1–3 months   |
| `mid`                | Equal (default)                   | 6–12 months  |
| `long`               | Fundamentals 70% / Technicals 30% | 1–3 years    |

| `risk_aversity` | Effect |
|-----------------|--------|
| 0.0–0.3         | Volatility Warning noted; doesn't change direction |
| 0.3–0.7         | SPIKE drops conviction one step |
| 0.7–1.0         | ELEVATED drops conviction; SPIKE downgrades recommendation + conviction → LOW |

---

## State Schema (`graph/state.py`)

```python
class MarketState(TypedDict):
    ticker: str
    investment_horizon: str   # "short" | "mid" | "long"
    risk_aversity: float      # 0.0 – 1.0
    research_data: Dict       # market_data, news, SEC, sentiment, summary
    ground_truth: Dict        # verified quantitative anchor
    bull_case: str
    bear_case: str
    bull_rebuttal: str
    bear_counter: str
    debate_history: Annotated[List[Dict], operator.add]
    fact_check_report: str
    final_forecast: str
    debate_round: int
```

---

## Environment Setup

1. Virtual environment: `.venv/` (Python 3.14)
2. Install dependencies:
   ```bash
   .venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
3. Create `.env` from the example:
   ```bash
   copy .env.example .env
   ```
4. Add your key to `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

---

## Running the Application

### CLI (`main.py`)

```bash
# Defaults: mid horizon, 0.5 risk aversity
python main.py AAPL

# Short-term, high risk aversity
python main.py TSLA --horizon short --risk 0.9 --save --verbose

# Long-term, risk-seeking
python main.py NVDA --horizon long --risk 0.1

# Flags:
#   --horizon  short | mid | long   (default: mid)
#   --risk     0.0 – 1.0            (default: 0.5)
#   --save     write TICKER_TIMESTAMP_forecast.md
#   --verbose  print full debate transcript
```

### Programmatic API (`api.py`)

```python
from api import run_analysis

result = run_analysis(
    ticker="AAPL",
    investment_horizon="mid",
    risk_aversity=0.5,
    save=False,
    verbose=False,
)

result["recommendation"]    # "BUY" | "HOLD" | "SELL" etc.
result["conviction"]        # "HIGH" | "MEDIUM" | "LOW"
result["volatility_warning"]# bool
result["basis_state"]       # "Contango" | "Backwardation" | None
result["volatility_level"]  # "NORMAL" | "ELEVATED" | "SPIKE" | None
result["final_forecast"]    # full Markdown report
result["ground_truth"]      # verified numbers dict
result["fact_check_report"] # per-claim verification string
result["debate_history"]    # list of {role, content} dicts
result["saved_to"]          # file path or None
```

---

## File Map

```
ticker-research-mas/
├── main.py                   ← CLI entry point
├── api.py                    ← Programmatic entry point
├── requirements.txt
├── .env.example
├── .env                      ← NEVER COMMIT
├── CONTEXT.md                ← Primary living context (for Claude)
├── GEMINI.md                 ← THIS FILE (mirror for Gemini)
├── agents/
│   ├── researcher.py         ← Data + ground_truth + 9-section brief
│   ├── bull.py               ← Horizon/risk-aware; _context_block() helper
│   ├── bear.py               ← Horizon/risk-aware; _context_block() helper
│   ├── fact_checker.py       ← Numerical guardrail node
│   └── arbiter.py            ← V2 signal-based scoring
├── graph/
│   ├── state.py              ← MarketState TypedDict
│   └── workflow.py           ← StateGraph assembly
└── tools/
    ├── market_data.py        ← yfinance: stock + futures (60d history)
    ├── news_search.py        ← DuckDuckGo news
    ├── sec_filings.py        ← SEC EDGAR risk factors
    └── earnings_sentiment.py ← Earnings call NLP
```

---

## Development Conventions

- **Model:** All agents use `claude-opus-4-6` with `thinking={"type": "adaptive"}`.
- **`ground_truth`:** Emitted by `researcher_node`; immutable for all downstream nodes.
- **`debate_history`:** `operator.add` annotation — each node appends, never overwrites.
- **`_extract_text()`:** In `researcher.py`, imported by all agents — strips thinking blocks.
- **`_context_block()`:** In `bull.py` / `bear.py` — generates horizon+risk preamble for debate prompts.
- **Futures mapping:** `Technology / CommServices / ConsumerCyclical → NQ=F`; all others → `ES=F`.
- **No `.gitignore`:** Do not commit `.env`, `.venv/`, `__pycache__/`, or `*_forecast.md`.

---

## Known Issues

- `yfinance` rate limits → `RemoteDataError`; retry once.
- `duckduckgo-search` v8 requires `DDGS()` context manager (old `ddg()` removed).
- `ma200` is `None` for tickers with < 200 trading days of history.
- SEC EDGAR regex may fail on non-standard HTML filings → graceful fallback string.
- `pydantic v1` warning on Python 3.14 from `langchain_core` — cosmetic only.
- Futures fields return `None` (not a crash) if yfinance can't reach the contract.
