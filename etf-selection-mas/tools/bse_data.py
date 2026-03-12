"""
BSE / NSE ETF Data Registry
============================
Uses the `.NS` suffix (National Stock Exchange of India) for yfinance tickers.
NSE is preferred over `.BO` (BSE) because yfinance returns richer metadata —
including price history and, occasionally, fund info — for NSE-listed symbols.

Schema per entry:
  ticker    : yfinance ticker string (e.g. "NIFTYBEES.NS")
  nse_symbol: Raw NSE trading symbol (without suffix)
  name      : Full fund name
  isin      : ISIN code
  category  : ETF category used in the output Markdown table
  trade_on  : Brokerage platforms for Indian retail investors
  amc       : Asset Management Company

PRODUCTION UPGRADE PATH
-----------------------
Replace the static registry with a live AMFI data pull:

    import requests, io
    resp = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=10)
    # Parse the pipe-delimited file, filter Scheme Type == "ETF"
    # Map scheme names to NSE symbols via NSE's ETF master file:
    # https://archives.nseindia.com/content/indices/ETF_Weightages.csv

Current status: Static registry verified against NSE ETF master list, Q1 2026.
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# NSE ETF Registry
# ---------------------------------------------------------------------------
_NSE_ETF_REGISTRY: List[Dict[str, str]] = [
    # ── Broad Market ──────────────────────────────────────────────────────
    {
        "ticker":     "NIFTYBEES.NS",
        "nse_symbol": "NIFTYBEES",
        "name":       "Nippon India ETF Nifty 50 BeES",
        "isin":       "INF204KB13I2",
        "category":   "India Broad Market (Nifty 50)",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    {
        "ticker":     "SETFNIF50.NS",
        "nse_symbol": "SETFNIF50",
        "name":       "SBI ETF Nifty 50",
        "isin":       "INF200KA1UF7",
        "category":   "India Broad Market (Nifty 50)",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "SBI MF",
    },
    {
        "ticker":     "JUNIORBEES.NS",
        "nse_symbol": "JUNIORBEES",
        "name":       "Nippon India ETF Nifty Next 50",
        "isin":       "INF204KB16I5",
        "category":   "India Mid-Large Cap (Nifty Next 50)",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    # ── Sectoral ─────────────────────────────────────────────────────────
    {
        "ticker":     "BANKBEES.NS",
        "nse_symbol": "BANKBEES",
        "name":       "Nippon India ETF Bank BeES",
        "isin":       "INF204KB14I0",
        "category":   "India Sectoral — Banking & Finance",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    {
        "ticker":     "ITBEES.NS",
        "nse_symbol": "ITBEES",
        "name":       "Nippon India ETF Nifty IT",
        "isin":       "INF204KB17I3",
        "category":   "India Sectoral — Information Technology",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    {
        "ticker":     "INFRABEES.NS",
        "nse_symbol": "INFRABEES",
        "name":       "Nippon India ETF Infra BeES",
        "isin":       "INF204KB18I1",
        "category":   "India Sectoral — Infrastructure",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    {
        "ticker":     "PHARMABEES.NS",
        "nse_symbol": "PHARMABEES",
        "name":       "Nippon India ETF Pharma BeES",
        "isin":       "INF204KB10I8",
        "category":   "India Sectoral — Pharmaceuticals",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    {
        "ticker":     "CONSUMBEES.NS",
        "nse_symbol": "CONSUMBEES",
        "name":       "Nippon India ETF Consumption",
        "isin":       "INF204KB19I9",
        "category":   "India Sectoral — Consumer Discretionary",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    # ── Commodity ────────────────────────────────────────────────────────
    {
        "ticker":     "GOLDBEES.NS",
        "nse_symbol": "GOLDBEES",
        "name":       "Nippon India ETF Gold BeES",
        "isin":       "INF204KB15I7",
        "category":   "India Commodity — Gold",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
    # ── Smart Beta / Factor ───────────────────────────────────────────────
    {
        "ticker":     "DIVOPPBEES.NS",
        "nse_symbol": "DIVOPPBEES",
        "name":       "Nippon India ETF Dividend Opportunities",
        "isin":       "INF204KB11I6",
        "category":   "India Smart Beta — Dividend Yield",
        "trade_on":   "Zerodha / Groww / Upstox",
        "amc":        "Nippon India MF",
    },
]

# ---------------------------------------------------------------------------
# NASDAQ ETF metadata — category + trade_on for the output report
# (INTL and HKCN tickers listed here for lookup in PortfolioArbiter)
# ---------------------------------------------------------------------------
_NASDAQ_ETF_METADATA: Dict[str, Dict[str, str]] = {
    # ── INTL ───────────────────────────────────────────────────────────────
    "VXUS":  {"category": "International ex-US (Total Market)", "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "SPDW":  {"category": "International Developed ex-US",      "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "IEFA":  {"category": "MSCI EAFE — Developed Markets",      "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "SCHY":  {"category": "International Dividend Equity",       "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "VEA":   {"category": "FTSE Developed Markets ex-US",        "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "EEM":   {"category": "MSCI Emerging Markets",               "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "ACWI":  {"category": "MSCI All Country World (ACWI)",       "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "SCHF":  {"category": "International Large-Cap Developed",   "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "IXUS":  {"category": "MSCI Total International Stock",      "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "VWO":   {"category": "FTSE Emerging Markets",               "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "IEMG":  {"category": "MSCI Core Emerging Markets",          "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    # ── HKCN ──────────────────────────────────────────────────────────────
    "CQQQ":  {"category": "China Internet / Technology",         "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "KWEB":  {"category": "China Internet / e-Commerce",         "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "MCHI":  {"category": "MSCI China Broad Market",             "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "FXI":   {"category": "China Large-Cap (H-Shares)",          "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "CHIQ":  {"category": "China Consumer Discretionary",        "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "GXC":   {"category": "S&P China Broad Market",              "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "FLCH":  {"category": "FTSE China — Low Cost",               "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "KURE":  {"category": "China Healthcare",                    "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "CNYA":  {"category": "MSCI China A-Shares (Onshore)",       "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
    "ASHR":  {"category": "CSI 300 China A-Shares",              "trade_on": "Interactive Brokers / Robinhood / Fidelity"},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_nse_tickers(categories: Optional[List[str]] = None) -> List[str]:
    """
    Return NSE ETF tickers for yfinance.

    Args:
        categories: Optional filter. E.g. ["India Broad Market (Nifty 50)"].
                    If None, returns all 10 tickers.

    Returns:
        List of tickers with .NS suffix.
    """
    if categories:
        return [e["ticker"] for e in _NSE_ETF_REGISTRY if e["category"] in categories]
    return [e["ticker"] for e in _NSE_ETF_REGISTRY]


def get_nse_registry() -> List[Dict[str, str]]:
    """Return the full NSE ETF registry (all metadata)."""
    return list(_NSE_ETF_REGISTRY)


def get_etf_metadata(ticker: str) -> Dict[str, str]:
    """
    Lookup display metadata for any ticker (NSE or NASDAQ).
    Returns dict with `category` and `trade_on` keys.
    Falls back to generic values if ticker not found.
    """
    # NSE lookup
    for e in _NSE_ETF_REGISTRY:
        if e["ticker"] == ticker or e["nse_symbol"] == ticker:
            return {"category": e["category"], "trade_on": e["trade_on"]}

    # NASDAQ lookup
    if ticker in _NASDAQ_ETF_METADATA:
        return _NASDAQ_ETF_METADATA[ticker]

    # Generic fallback
    region = "India NSE" if ticker.endswith(".NS") else "NASDAQ"
    brokers = "Zerodha / Groww / Upstox" if ticker.endswith(".NS") else "Interactive Brokers / Robinhood"
    return {"category": region, "trade_on": brokers}
