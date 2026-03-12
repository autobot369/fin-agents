"""
Node 1 — Regional Alpha & Cost Orchestrator
=============================================
Identifies a monthly "Investment Universe" of 10–15 ETFs across three tiers
using Gemini for discovery and yfinance for validation.

Phase 1 — Universe Discovery (Three-Tier Scan)
  • India Tier   (BSE/NSE)      : Nifty 50, Next 50, PSU Bank ETFs. TER < 0.20%.
                                   2026 context: Reliance Industries + Infrastructure themes.
  • US Tier      (NASDAQ/NYSE)  : Core (VOO/QQQ) at TER < 0.15% + Thematic (SOXX/AI).
  • HK Proxy Tier (HKEX → US)  : Checks for US-listed equivalents before recommending
                                   a local HKEX ticker. If TER diff < 0.10%, use US proxy
                                   to save the 0.10% HK Stamp Duty.

Phase 2 — Data Extraction & Cost Attribution
  Each ticker gets: expense_ratio (yfinance / Gemini estimate),
  adv_usd (liquidity), est_entry_cost_pct, recommended_broker, is_proxy.

  Entry cost formula:
    cost = (brokerage_min / trade_size) + stamp_duty + (expense_ratio / 12)

Phase 3 — Guardrails
  • Liquidity: ADV < $1M USD → excluded with warning.
  • No Redundancy: US market provides sufficient HK/China exposure in most cases.
  • FX flag: HKD and INR trades noted in category field.

Falls back per-tier to hardcoded seed lists if Gemini is unavailable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

from sip_execution_mas.graph.state import ETFRecord, SIPExecutionState


# ── Broker cost constants ─────────────────────────────────────────────────────

_BROKERAGE_MIN: Dict[str, float] = {
    "Dhan":   0.00,   # Zero brokerage for BSE/NSE ETFs
    "Alpaca": 0.00,   # Commission-free US ETFs
    "Tiger":  0.50,   # Tiger Brokers HK: ~HKD 3 ≈ $0.50 USD
    "IBKR":   2.00,   # IBKR minimum for HK trades
}

_HK_STAMP_DUTY = 0.001    # 0.10% on every HK buy
_TRADE_SIZE    = 500.0    # Reference trade size in USD
_ADV_MIN_USD   = 1_000_000  # $1M USD minimum average daily volume

# FX conversion factors (approximate)
_HKD_USD = 0.128          # 1 HKD ≈ $0.128 USD


# ── Gemini client ─────────────────────────────────────────────────────────────

def _get_gemini_model(temperature: float = 0.2):
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=temperature,
        ),
    )


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


# ── Phase 1 Gemini Prompts ────────────────────────────────────────────────────

_INDIA_PROMPT = """
You are a Regional Alpha & Cost Orchestrator for a monthly SIP investment system.

Identify exactly 10 NSE-listed ETFs for the India tier.

Selection criteria:
1. Expense Ratio MUST be < 0.20% (0.0020 decimal). Reject any ETF above this.
2. High AUM (prefer > ₹500 crore) and liquid daily traded volume.
3. NSE-listed only — ticker symbols MUST end in .NS (e.g. NIFTYBEES.NS).

Coverage mix required:
- 3–4 Nifty 50 index ETFs (lowest TER options, e.g. NIFTYBEES.NS, SETFNIF50.NS, HDFCNIFTY.NS)
- 2–3 Nifty Next 50 / Midcap ETFs (e.g. JUNIORBEES.NS, MIDCAPETF.NS)
- 2–3 Sectoral / Thematic: PSU Bank, Infrastructure, or high-conviction themes

2026 context — prioritise:
- ETFs with high Reliance Industries weighting (US refinery deal + energy transition tailwind)
- Infrastructure-focused ETFs (India capex supercycle, PLI schemes)
- PSU Bank ETFs if current RBI rate environment is supportive

Return a JSON array of exactly 10 objects:
{
  "ticker": "NIFTYBEES.NS",
  "name": "Nippon India ETF Nifty 50 BeES",
  "ter_pct": 0.04,
  "aum_b_est": 30.5,
  "adv_usd_est": 5.2,
  "category": "Nifty 50 | core",
  "theme": "India Large-Cap",
  "rationale": "Lowest-cost Nifty 50 tracker with high Reliance weighting"
}
""".strip()

_US_PROMPT = """
You are a Regional Alpha & Cost Orchestrator for a monthly SIP investment system.

Identify exactly 12 US-listed ETFs (NASDAQ or NYSE) for the US tier.

Coverage mix required:
- 4–5 Core broad-market ETFs: expense ratio MUST be < 0.15%. Include options like VOO, QQQ, VTI, SCHB, IVV.
- 4–5 Thematic high-conviction ETFs: AI/ML, Semiconductors, Clean Energy, Biotech.
  No strict TER limit for thematic but flag if TER > 0.60%.
  Examples: SOXX, SMH, BOTZ, IRBO, SOXQ, CLOU.
- 2–3 International developed-market ETFs at TER < 0.15%: VEA, SPDW, SCHF.

2026 context — prioritise:
- AI infrastructure buildout (data centres, hyperscalers): SOXX, BOTZ, IRBO
- Semiconductor cycle recovery: SMH, SOXQ
- US rate cut environment: growth tilt justified

Selection criteria:
- Core: TER < 0.15% strictly enforced
- Thematic: ADV > $10M USD strongly preferred
- All: ADV > $1M USD minimum

Return a JSON array of exactly 12 objects:
{
  "ticker": "VOO",
  "name": "Vanguard S&P 500 ETF",
  "ter_pct": 0.03,
  "aum_b_est": 470.0,
  "adv_usd_est": 450.0,
  "category": "core",
  "theme": "US Large-Cap",
  "rationale": "Lowest-cost S&P 500 tracker, bedrock holding"
}
""".strip()

_HK_PROXY_PROMPT = """
You are a Regional Alpha & Cost Orchestrator specialising in China/HK exposure optimisation.

Identify 8 China/HK investment themes. For each theme provide BOTH the HKEX-listed ETF
AND the best US-listed proxy available on NASDAQ/NYSE.

Decision rule (implement this logic):
- If |hk_ter_pct - us_proxy_ter_pct| < 0.10 → recommend "us_proxy"
  (saves the 0.10% HK Stamp Duty per trade + lower IBKR minimums)
- Only recommend "hk_local" if there is SPECIFIC regional alpha NOT available via US proxy
  (e.g. HK Biotech, HK REIT, A-share sectors with no US equivalent)

Theme examples: Broad China, China Internet, China Tech, China A-Shares (CSI 300),
China Consumer, China Healthcare, HK Blue-Chip, China EV/Green Energy.

Return a JSON array of exactly 8 objects:
{
  "theme": "Broad China",
  "hk_ticker": "2800.HK",
  "hk_name": "Tracker Fund of Hong Kong",
  "hk_ter_pct": 0.09,
  "hk_adv_usd_est": 50.0,
  "us_proxy_ticker": "MCHI",
  "us_proxy_name": "iShares MSCI China ETF",
  "us_proxy_ter_pct": 0.19,
  "us_proxy_adv_usd_est": 80.0,
  "recommended": "hk_local",
  "has_regional_alpha": true,
  "recommendation_reason": "HK local has 0.09% TER vs 0.19% proxy — saving justifies stamp duty"
}
""".strip()


# ── Fallback seed universe ────────────────────────────────────────────────────

_FALLBACK: Dict[str, List[Dict[str, Any]]] = {
    "BSE": [
        {"ticker": "NIFTYBEES.NS",  "name": "Nippon Nifty 50 BeES",      "ter_pct": 0.04, "category": "core",      "theme": "Nifty 50"},
        {"ticker": "SETFNIF50.NS",  "name": "SBI Nifty 50 ETF",          "ter_pct": 0.04, "category": "core",      "theme": "Nifty 50"},
        {"ticker": "HDFCNIFTY.NS",  "name": "HDFC Nifty 50 ETF",         "ter_pct": 0.05, "category": "core",      "theme": "Nifty 50"},
        {"ticker": "JUNIORBEES.NS", "name": "Nippon Nifty Next 50 BeES", "ter_pct": 0.06, "category": "midcap",    "theme": "Nifty Next 50"},
        {"ticker": "BANKBEES.NS",   "name": "Nippon Bank BeES",           "ter_pct": 0.18, "category": "sectoral",  "theme": "PSU Bank"},
        {"ticker": "CPSEETF.NS",    "name": "Nippon CPSE ETF",            "ter_pct": 0.10, "category": "sectoral",  "theme": "PSU/Infra"},
        {"ticker": "ITBEES.NS",     "name": "Nippon IT BeES",             "ter_pct": 0.10, "category": "sectoral",  "theme": "India IT"},
        {"ticker": "GOLDBEES.NS",   "name": "Nippon Gold BeES",           "ter_pct": 0.05, "category": "commodity", "theme": "Gold"},
        {"ticker": "LIQUIDBEES.NS", "name": "Nippon Liquid BeES",         "ter_pct": 0.01, "category": "money_mkt", "theme": "Liquid"},
        {"ticker": "MAFANG.NS",     "name": "Mirae FANG+ ETF",            "ter_pct": 0.83, "category": "thematic",  "theme": "Global Tech"},
    ],
    "US": [
        {"ticker": "VOO",   "name": "Vanguard S&P 500 ETF",           "ter_pct": 0.03, "category": "core",      "theme": "US Large-Cap"},
        {"ticker": "QQQ",   "name": "Invesco QQQ Trust",               "ter_pct": 0.20, "category": "core",      "theme": "US Tech"},
        {"ticker": "VTI",   "name": "Vanguard Total Stock Market",     "ter_pct": 0.03, "category": "core",      "theme": "US Total Market"},
        {"ticker": "SCHB",  "name": "Schwab US Broad Market ETF",      "ter_pct": 0.03, "category": "core",      "theme": "US Broad"},
        {"ticker": "VEA",   "name": "Vanguard FTSE Dev Markets",       "ter_pct": 0.05, "category": "core",      "theme": "Developed Markets"},
        {"ticker": "SPDW",  "name": "SPDR Portfolio Dev World ex-US",  "ter_pct": 0.04, "category": "core",      "theme": "Developed Markets"},
        {"ticker": "SOXX",  "name": "iShares Semiconductor ETF",       "ter_pct": 0.35, "category": "thematic",  "theme": "Semiconductors"},
        {"ticker": "SMH",   "name": "VanEck Semiconductor ETF",        "ter_pct": 0.35, "category": "thematic",  "theme": "Semiconductors"},
        {"ticker": "BOTZ",  "name": "Global X Robotics & AI ETF",      "ter_pct": 0.68, "category": "thematic",  "theme": "AI/Robotics"},
        {"ticker": "IRBO",  "name": "iShares Robotics & AI ETF",       "ter_pct": 0.47, "category": "thematic",  "theme": "AI/Robotics"},
        {"ticker": "SCHF",  "name": "Schwab International Equity ETF", "ter_pct": 0.06, "category": "core",      "theme": "International"},
        {"ticker": "IEFA",  "name": "iShares Core MSCI EAFE ETF",      "ter_pct": 0.07, "category": "core",      "theme": "Developed Markets"},
    ],
    "HKCN": [
        {"ticker": "MCHI",  "name": "iShares MSCI China",           "ter_pct": 0.19, "category": "broad_china", "theme": "Broad China",    "is_proxy": True,  "proxy_for": "2800.HK"},
        {"ticker": "FLCH",  "name": "Franklin FTSE China ETF",      "ter_pct": 0.08, "category": "broad_china", "theme": "Broad China",    "is_proxy": True,  "proxy_for": "3115.HK"},
        {"ticker": "KWEB",  "name": "KraneShares China Internet",   "ter_pct": 0.76, "category": "thematic",    "theme": "China Internet", "is_proxy": True,  "proxy_for": "3174.HK"},
        {"ticker": "CQQQ",  "name": "Invesco China Technology ETF", "ter_pct": 0.65, "category": "thematic",    "theme": "China Tech",     "is_proxy": True,  "proxy_for": "3086.HK"},
        {"ticker": "ASHR",  "name": "Xtrackers CSI 300",            "ter_pct": 0.65, "category": "a_shares",    "theme": "CSI 300",        "is_proxy": True,  "proxy_for": "3188.HK"},
        {"ticker": "CNYA",  "name": "iShares MSCI China A ETF",     "ter_pct": 0.55, "category": "a_shares",    "theme": "China A-Shares", "is_proxy": True,  "proxy_for": "3188.HK"},
        {"ticker": "KURE",  "name": "KraneShares China Healthcare", "ter_pct": 0.75, "category": "thematic",    "theme": "China Health",   "is_proxy": True,  "proxy_for": "2820.HK"},
        {"ticker": "CHIQ",  "name": "Global X China Consumer Disc", "ter_pct": 0.65, "category": "thematic",    "theme": "China Consumer", "is_proxy": True,  "proxy_for": "3032.HK"},
    ],
}

# HKEX → US proxy mapping (for cost-comparison logic)
_HKEX_TO_US_PROXY: Dict[str, str] = {
    "2800.HK": "MCHI",   # Tracker Fund → iShares MSCI China
    "3033.HK": "KWEB",   # CSOP MSCI China Internet → KraneShares
    "3086.HK": "CQQQ",   # CSOP MSCI China Tech → Invesco China Tech
    "3115.HK": "FLCH",   # CSOP FTSE China A50 → Franklin FTSE China
    "3188.HK": "ASHR",   # ChinaAMC CSI 300 → Xtrackers CSI 300
    "2820.HK": "KURE",   # CSOP China Healthcare → KraneShares China HC
    "3032.HK": "CHIQ",   # CSOP China Consumer → Global X China Consumer
    "3174.HK": "KWEB",   # Hang Seng Internet → KraneShares Internet
    "9834.HK": "SMH",    # CSOP NASDAQ-100 → VanEck Semiconductors
}


# ── Phase 2: Cost attribution ─────────────────────────────────────────────────

def _recommend_broker(ticker: str, is_hkex_local: bool = False) -> str:
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "Dhan"
    if is_hkex_local or ticker.endswith(".HK"):
        return "Tiger"
    return "Alpaca"


def _calc_entry_cost(
    expense_ratio: float,
    broker: str,
    trade_size: float = _TRADE_SIZE,
    is_hk_local: bool = False,
) -> float:
    """
    Total drag for a single trade, as a decimal fraction.
    Formula: (brokerage_min / trade_size) + stamp_duty + (expense_ratio / 12)
    """
    brokerage_min = _BROKERAGE_MIN.get(broker, 0.0)
    stamp_duty    = _HK_STAMP_DUTY if is_hk_local else 0.0
    return (brokerage_min / trade_size) + stamp_duty + (expense_ratio / 12)


def _adv_usd(
    ticker: str,
    avg_volume: Optional[int],
    price: Optional[float],
    usd_inr: float = 84.0,
) -> Optional[float]:
    """Convert average daily volume (shares × price) to USD equivalent."""
    if avg_volume is None or price is None or price <= 0:
        return None
    adv_native = avg_volume * price
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return adv_native / usd_inr
    if ticker.endswith(".HK"):
        return adv_native * _HKD_USD
    return adv_native


# ── Phase 1: Gemini discovery ─────────────────────────────────────────────────

def _ask_gemini(prompt: str) -> List[Dict[str, Any]]:
    model  = _get_gemini_model()
    resp   = model.generate_content(prompt)
    data   = _parse_json(resp.text)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Gemini returned empty or non-list")
    return data


def _build_india_tier(ter_threshold: float) -> List[Dict[str, Any]]:
    """Ask Gemini for India BSE ETFs. Falls back to seed list."""
    try:
        recs = _ask_gemini(_INDIA_PROMPT)
        cleaned = []
        for r in recs:
            ticker = str(r.get("ticker") or "").strip()
            ter    = float(r.get("ter_pct") or 0.0)
            if not ticker or ter >= 0.20:   # enforce India TER hard cap
                continue
            cleaned.append({
                "ticker":      ticker,
                "name":        str(r.get("name") or ticker),
                "ter_pct":     ter,
                "aum_b_est":   float(r.get("aum_b_est") or 0.0),
                "adv_usd_est": float(r.get("adv_usd_est") or 0.0),
                "category":    str(r.get("category") or "India ETF"),
                "theme":       str(r.get("theme") or ""),
                "rationale":   str(r.get("rationale") or ""),
                "source":      "gemini",
            })
        print(f"  [Node 1] Gemini → India tier: {len(cleaned)} ETFs (TER < 0.20%)")
        return cleaned or _FALLBACK["BSE"]
    except Exception as exc:
        print(f"  [Node 1] Gemini India tier failed ({exc}) — using seed list")
        return _FALLBACK["BSE"]


def _build_us_tier() -> List[Dict[str, Any]]:
    """Ask Gemini for US core + thematic ETFs. Falls back to seed list."""
    try:
        recs = _ask_gemini(_US_PROMPT)
        cleaned = []
        for r in recs:
            ticker   = str(r.get("ticker") or "").upper().strip()
            ter      = float(r.get("ter_pct") or 0.0)
            category = str(r.get("category") or "thematic").lower()
            if not ticker:
                continue
            # Enforce TER < 0.15% for core holdings only
            if category == "core" and ter >= 0.15:
                continue
            cleaned.append({
                "ticker":      ticker,
                "name":        str(r.get("name") or ticker),
                "ter_pct":     ter,
                "aum_b_est":   float(r.get("aum_b_est") or 0.0),
                "adv_usd_est": float(r.get("adv_usd_est") or 0.0),
                "category":    category,
                "theme":       str(r.get("theme") or ""),
                "rationale":   str(r.get("rationale") or ""),
                "source":      "gemini",
            })
        print(f"  [Node 1] Gemini → US tier: {len(cleaned)} ETFs")
        return cleaned or _FALLBACK["US"]
    except Exception as exc:
        print(f"  [Node 1] Gemini US tier failed ({exc}) — using seed list")
        return _FALLBACK["US"]


def _build_hk_proxy_tier() -> List[Dict[str, Any]]:
    """
    Ask Gemini for HK/China themes. Apply proxy decision logic:
    if |hk_ter - us_proxy_ter| < 0.10 → use US proxy (saves stamp duty).
    """
    try:
        recs = _ask_gemini(_HK_PROXY_PROMPT)
        selected = []
        for r in recs:
            hk_ter = float(r.get("hk_ter_pct") or 0.75)
            us_ter = float(r.get("us_proxy_ter_pct") or 0.75)
            ter_diff = abs(hk_ter - us_ter)
            # Gemini recommendation + enforce proxy rule
            use_proxy = (
                r.get("recommended") == "us_proxy"
                or (ter_diff < 0.10 and not r.get("has_regional_alpha", False))
            )

            if use_proxy:
                ticker = str(r.get("us_proxy_ticker") or "").upper().strip()
                if not ticker:
                    continue
                selected.append({
                    "ticker":      ticker,
                    "name":        str(r.get("us_proxy_name") or ticker),
                    "ter_pct":     us_ter,
                    "aum_b_est":   float(r.get("us_proxy_adv_usd_est") or 0.0),
                    "adv_usd_est": float(r.get("us_proxy_adv_usd_est") or 0.0),
                    "category":    str(r.get("theme") or "China/HK"),
                    "theme":       str(r.get("theme") or ""),
                    "rationale":   (
                        f"US proxy for {r.get('hk_ticker','?')} — "
                        f"saves 0.10% HK stamp duty (TER diff {ter_diff:.2f}%)"
                    ),
                    "is_proxy":    True,
                    "proxy_for":   str(r.get("hk_ticker") or ""),
                    "source":      "gemini",
                })
            else:
                # Use HKEX local — specific regional alpha justifies stamp duty
                ticker = str(r.get("hk_ticker") or "").strip()
                if not ticker:
                    continue
                selected.append({
                    "ticker":      ticker,
                    "name":        str(r.get("hk_name") or ticker),
                    "ter_pct":     hk_ter,
                    "aum_b_est":   float(r.get("hk_adv_usd_est") or 0.0),
                    "adv_usd_est": float(r.get("hk_adv_usd_est") or 0.0),
                    "category":    str(r.get("theme") or "China/HK"),
                    "theme":       str(r.get("theme") or ""),
                    "rationale":   str(r.get("recommendation_reason") or "Regional alpha"),
                    "is_proxy":    False,
                    "proxy_for":   None,
                    "source":      "gemini",
                })
        print(f"  [Node 1] Gemini → HK/China tier: {len(selected)} ETFs "
              f"({sum(1 for x in selected if x.get('is_proxy'))} US proxies, "
              f"{sum(1 for x in selected if not x.get('is_proxy'))} HK local)")
        return selected or _FALLBACK["HKCN"]
    except Exception as exc:
        print(f"  [Node 1] Gemini HK tier failed ({exc}) — using seed list")
        return _FALLBACK["HKCN"]


# ── yfinance enrichment ───────────────────────────────────────────────────────

def _fetch_yfinance_batch(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        try:
            tk   = yf.Ticker(ticker)
            hist = tk.history(period="1y", auto_adjust=True)
            info: Dict[str, Any] = {}
            try:
                info = tk.info or {}
            except Exception:
                pass

            if hist.empty:
                results[ticker] = {"error": "no_history"}
                continue

            close = hist["Close"]
            price = float(close.iloc[-1])

            mom3m = mom1m = ytd = None
            if len(close) >= 63:
                mom3m = round((price / float(close.iloc[-63]) - 1) * 100, 2)
            if len(close) >= 21:
                mom1m = round((price / float(close.iloc[-21]) - 1) * 100, 2)
            if len(close) >= 5:
                ytd = round((price / float(close.iloc[0]) - 1) * 100, 2)

            ter = None
            for key in ("annualReportExpenseRatio", "expenseRatio", "totalExpenseRatio"):
                v = info.get(key)
                if v and isinstance(v, (int, float)) and v > 0:
                    ter = float(v)
                    break

            aum   = info.get("totalAssets")
            aum_b = round(float(aum) / 1e9, 2) if aum else None

            avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")

            results[ticker] = {
                "price":   round(price, 4),
                "ytd":     ytd,
                "mom3m":   mom3m,
                "mom1m":   mom1m,
                "ter":     ter,
                "aum_b":   aum_b,
                "avg_vol": int(avg_vol) if avg_vol else None,
                "source":  "yfinance",
            }
        except Exception as exc:
            results[ticker] = {"error": str(exc)}
        time.sleep(0.05)
    return results


# ── News fetch ────────────────────────────────────────────────────────────────

def _fetch_news(market_tickers: Dict[str, List[str]], max_per_query: int = 5) -> Dict[str, List[str]]:
    """
    Fetch headlines targeted at the specific recommended tickers per market.
    """
    news: Dict[str, List[str]] = {}
    context = {
        "BSE":  "India NSE ETF Nifty Reliance Infrastructure RBI 2026",
        "US":   "US ETF semiconductor AI market outlook NASDAQ 2026",
        "HKCN": "China Hong Kong ETF market stimulus tech 2026",
    }
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            for market, tickers in market_tickers.items():
                sample = " ".join(tickers[:5])
                q = f"{sample} {context.get(market, 'ETF market outlook')}"
                try:
                    results  = list(ddgs.news(q, max_results=max_per_query))
                    news[market] = [r.get("title", "") for r in results if r.get("title")]
                except Exception:
                    news[market] = []
    except Exception:
        for m in market_tickers:
            news[m] = []
    return news


# ── Node function ─────────────────────────────────────────────────────────────

def regional_researcher_node(state: SIPExecutionState) -> dict:
    """
    Node 1 — Regional Alpha & Cost Orchestrator

    Phase 1: Gemini discovers ETF universe across 3 tiers.
    Phase 2: yfinance validates metrics; cost attribution computed.
    Phase 3: Liquidity filter (ADV < $1M USD excluded); proxy flag; FX notation.
    """
    ter_threshold = state["ter_threshold"]

    print(f"\n[Node 1] Regional Alpha & Cost Orchestrator — building universe …")

    # ── Phase 1: Discover universe via Gemini ─────────────────────────────────
    india_recs = _build_india_tier(ter_threshold)
    us_recs    = _build_us_tier()
    hk_recs    = _build_hk_proxy_tier()

    market_recs: Dict[str, List[Dict[str, Any]]] = {
        "BSE":  india_recs,
        "US":   us_recs,
        "HKCN": hk_recs,
    }
    all_tickers = [r["ticker"] for recs in market_recs.values() for r in recs]
    print(f"[Node 1] Universe: {len(all_tickers)} candidates — fetching yfinance data …")

    # ── Phase 2: yfinance enrichment ──────────────────────────────────────────
    raw        = _fetch_yfinance_batch(all_tickers)
    macro_news = _fetch_news({m: [r["ticker"] for r in recs] for m, recs in market_recs.items()})

    etf_data:  Dict[str, ETFRecord] = {}
    filtered:  List[str]            = []
    excluded:  List[str]            = []

    for market, recs in market_recs.items():
        for rec in recs:
            ticker     = rec["ticker"]
            r          = raw.get(ticker, {})
            error      = r.get("error")
            is_hk_local = ticker.endswith(".HK")
            is_proxy    = rec.get("is_proxy", not is_hk_local and market == "HKCN")

            # TER: yfinance first, then Gemini estimate
            ter = r.get("ter") if not error else None
            if ter is None and rec.get("ter_pct"):
                ter = rec["ter_pct"] / 100.0

            price  = r.get("price") if not error else None
            aum_b  = (r.get("aum_b") if not error else None) or (rec.get("aum_b_est") or None)

            # ── Phase 3: Liquidity check ──────────────────────────────────
            avg_vol  = r.get("avg_vol") if not error else None
            adv      = _adv_usd(ticker, avg_vol, price)
            liq_ok   = (adv is None) or (adv >= _ADV_MIN_USD)   # unknown → include

            # ── Cost attribution ──────────────────────────────────────────
            broker    = _recommend_broker(ticker, is_hkex_local=is_hk_local)
            entry_cost = _calc_entry_cost(
                expense_ratio = ter if ter is not None else 0.005,
                broker        = broker,
                is_hk_local   = is_hk_local,
            ) if ter is not None else None

            # ── FX flag in category ───────────────────────────────────────
            currency = "INR" if market == "BSE" else ("HKD" if is_hk_local else "USD")
            category = rec.get("category", rec.get("theme", "ETF"))
            if is_hk_local:
                category = f"{category} [HKD — FX conversion required]"
            elif currency == "INR":
                category = f"{category} [INR — FX via Dhan]"

            record: ETFRecord = {
                "ticker":              ticker,
                "name":                rec["name"],
                "region":              market,
                "market":              "NSE" if market == "BSE" else ("HKEX" if is_hk_local else "NASDAQ"),
                "category":            category,
                "expense_ratio":       ter,
                "aum_b":               aum_b,
                "ytd_return":          r.get("ytd") if not error else None,
                "momentum_3m":         r.get("mom3m") if not error else None,
                "momentum_1m":         r.get("mom1m") if not error else None,
                "current_price":       price,
                "currency":            currency,
                "data_source":         "yfinance" if not error else "gemini_estimate",
                "fetch_error":         error,
                # Cost & routing
                "recommended_broker":  broker,
                "adv_usd":             adv,
                "liquidity_ok":        liq_ok,
                "est_entry_cost_pct":  entry_cost,
                "is_proxy":            is_proxy,
                "proxy_for":           rec.get("proxy_for"),
            }
            etf_data[ticker] = record

            # Phase 3: reject illiquid, apply TER filter
            if not liq_ok:
                excluded.append(ticker)
                print(f"  [Node 1] EXCLUDED {ticker} — ADV ${adv:,.0f} below $1M minimum")
                continue

            if ter is not None and ter > ter_threshold:
                excluded.append(ticker)
                continue

            filtered.append(ticker)

    print(f"[Node 1] Result: {len(filtered)} ETFs pass | {len(excluded)} excluded")
    for market in ("BSE", "US", "HKCN"):
        n = sum(1 for t in filtered if etf_data[t]["region"] == market)
        proxies = sum(1 for t in filtered if etf_data[t]["region"] == market and etf_data[t]["is_proxy"])
        print(f"  {market}: {n} ETFs  ({proxies} US proxies for HK)" if market == "HKCN"
              else f"  {market}: {n} ETFs")

    return {
        "all_etf_data":     etf_data,
        "all_macro_news":   macro_news,
        "filtered_tickers": filtered,
    }
