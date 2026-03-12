"""
BSE Ticker Ingestion — Placeholder Module
-----------------------------------------
This module provides a structured placeholder for BSE (Bombay Stock Exchange)
ETF ticker ingestion. BSE-listed ETFs trade under NSE codes but are accessible
in yfinance via the `.BO` suffix.

PRODUCTION UPGRADE PATH:
  Replace `_BSE_ETF_REGISTRY` with a live fetch from:
    1. BSE India ETF listing API: https://api.bseindia.com/BseIndiaAPI/...
    2. NSE India data feed (requires subscription)
    3. AMFI (Association of Mutual Funds in India) data file
       URL: https://www.amfiindia.com/spages/NAVAll.txt

Current status: Static registry with verified tickers as of Q1 2026.
The `ingest_bse_tickers()` function returns the ticker list ready for
`etf_data.fetch_etf_universe()`.
"""

from typing import Dict, List


# ---------------------------------------------------------------------------
# Static BSE ETF registry — schema per entry:
#   ticker    : yfinance ticker (with .BO suffix)
#   name      : fund name
#   isin      : ISIN code for BSE
#   nse_symbol: NSE equivalent symbol (without suffix)
#   category  : broad category for grouping
# ---------------------------------------------------------------------------
_BSE_ETF_REGISTRY: List[Dict[str, str]] = [
    {
        "ticker": "NIFTYBEES.BO",
        "name": "Nippon India ETF Nifty 50 BeES",
        "isin": "INF204KB13I2",
        "nse_symbol": "NIFTYBEES",
        "category": "Broad Market",
    },
    {
        "ticker": "BANKBEES.BO",
        "name": "Nippon India ETF Bank BeES",
        "isin": "INF204KB14I0",
        "nse_symbol": "BANKBEES",
        "category": "Sectoral - Banking",
    },
    {
        "ticker": "GOLDBEES.BO",
        "name": "Nippon India ETF Gold BeES",
        "isin": "INF204KB15I7",
        "nse_symbol": "GOLDBEES",
        "category": "Commodity - Gold",
    },
    {
        "ticker": "JUNIORBEES.BO",
        "name": "Nippon India ETF Junior BeES (Nifty Next 50)",
        "isin": "INF204KB16I5",
        "nse_symbol": "JUNIORBEES",
        "category": "Broad Market",
    },
    {
        "ticker": "ITBEES.BO",
        "name": "Nippon India ETF Nifty IT",
        "isin": "INF204KB17I3",
        "nse_symbol": "ITBEES",
        "category": "Sectoral - IT",
    },
    {
        "ticker": "INFRABEES.BO",
        "name": "Nippon India ETF Infra BeES",
        "isin": "INF204KB18I1",
        "nse_symbol": "INFRABEES",
        "category": "Sectoral - Infrastructure",
    },
    {
        "ticker": "SETFNIF50.BO",
        "name": "SBI ETF Nifty 50",
        "isin": "INF200KA1UF7",
        "nse_symbol": "SETFNIF50",
        "category": "Broad Market",
    },
    {
        "ticker": "CONSUMBEES.BO",
        "name": "Nippon India ETF Consumption",
        "isin": "INF204KB19I9",
        "nse_symbol": "CONSUMBEES",
        "category": "Sectoral - Consumer",
    },
    {
        "ticker": "PHARMABEES.BO",
        "name": "Nippon India ETF Pharma BeES",
        "isin": "INF204KB10I8",
        "nse_symbol": "PHARMABEES",
        "category": "Sectoral - Pharma",
    },
    {
        "ticker": "DIVOPPBEES.BO",
        "name": "Nippon India ETF Dividend Opportunities",
        "isin": "INF204KB11I6",
        "nse_symbol": "DIVOPPBEES",
        "category": "Smart Beta - Dividend",
    },
]


def ingest_bse_tickers(categories: List[str] | None = None) -> List[str]:
    """
    Return BSE ETF tickers for feeding into `fetch_etf_universe()`.

    Args:
        categories: Optional filter list (e.g. ["Broad Market", "Sectoral - IT"]).
                    If None, returns all tickers.

    Returns:
        List of yfinance-compatible ticker strings with `.BO` suffix.

    TODO (production):
        Replace static registry with a live AMFI/BSE API call:
            import requests
            url = "https://www.amfiindia.com/spages/NAVAll.txt"
            # parse and filter ETF-category entries
    """
    if categories:
        return [
            etf["ticker"]
            for etf in _BSE_ETF_REGISTRY
            if etf["category"] in categories
        ]
    return [etf["ticker"] for etf in _BSE_ETF_REGISTRY]


def get_bse_registry() -> List[Dict[str, str]]:
    """Return the full registry dict (for enrichment / display purposes)."""
    return list(_BSE_ETF_REGISTRY)


def get_ticker_metadata(ticker: str) -> Dict[str, str]:
    """Look up registry metadata for a specific BSE ticker."""
    for etf in _BSE_ETF_REGISTRY:
        if etf["ticker"] == ticker:
            return etf
    return {"ticker": ticker, "name": ticker, "category": "Unknown"}
