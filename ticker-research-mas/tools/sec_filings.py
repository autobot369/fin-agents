"""
SEC EDGAR Filing Parser
-----------------------
Fetches the most recent 10-Q (fallback: 10-K) for a ticker and extracts
the "Risk Factors" section from the filing text.

Uses only the free, public SEC EDGAR APIs — no API key required.
Rate-limit: SEC requests a max of 10 req/s; we sleep 0.5s between calls.
"""

import re
import time
from typing import Any, Dict, Optional

import requests

_HEADERS = {"User-Agent": "MarketResearchMAS contact@example.com"}
_TIMEOUT = 15


def _get_cik(ticker: str) -> Optional[str]:
    """Resolve ticker symbol → zero-padded 10-digit CIK."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        for entry in r.json().values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def _get_filing_url(cik: str, form_type: str) -> Optional[str]:
    """Return URL of the most recent primary document for form_type."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        time.sleep(0.5)
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        dates = recent.get("filingDate", [])

        for i, form in enumerate(forms):
            if form == form_type:
                acc = accessions[i].replace("-", "")
                cik_int = int(cik)
                doc = docs[i]
                filing_date = dates[i] if i < len(dates) else "unknown"
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"
                )
                return filing_url, filing_date
    except Exception:
        pass
    return None, None


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_risk_factors(html: str, max_chars: int) -> str:
    """
    Attempt to extract Item 1A Risk Factors section from filing HTML.
    Tries multiple regex patterns for robustness across filing formats.
    """
    patterns = [
        # Standard Item 1A with boundary at Item 1B or Item 2
        r"(?is)item\s+1a[\.\s\-:]+risk\s+factors\s*(.*?)(?=item\s+1b|item\s+2\b)",
        # Looser version
        r"(?is)risk\s+factors\s*(.*?)(?=unresolved\s+staff\s+comments|properties|legal\s+proceedings)",
        # Fallback: just find a big block with "risk" in it
        r"(?is)((?:risk[^<]{20,}){5,})",
    ]

    text = _strip_html(html)

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            extracted = match.group(1).strip()
            if len(extracted) > 200:
                return extracted[:max_chars]

    return "Risk factors section could not be parsed from this filing format."


def fetch_risk_factors(ticker: str, max_chars: int = 2500) -> Dict[str, Any]:
    """
    Main entry point.
    Returns dict with keys: filing_type, filing_date, risk_factors, url, error (if any).
    """
    cik = _get_cik(ticker)
    if not cik:
        return {
            "filing_type": None,
            "risk_factors": f"CIK not found for ticker '{ticker}' in SEC EDGAR.",
            "url": None,
        }

    for form_type in ("10-Q", "10-K"):
        url, filing_date = _get_filing_url(cik, form_type)
        if url:
            try:
                time.sleep(0.5)
                r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
                r.raise_for_status()
                risk_text = _extract_risk_factors(r.text, max_chars)
                return {
                    "filing_type": form_type,
                    "filing_date": filing_date,
                    "risk_factors": risk_text,
                    "url": url,
                }
            except Exception as e:
                return {
                    "filing_type": form_type,
                    "filing_date": filing_date,
                    "risk_factors": f"Failed to fetch filing: {e}",
                    "url": url,
                }

    return {
        "filing_type": None,
        "risk_factors": f"No 10-Q or 10-K found for '{ticker}' in SEC EDGAR.",
        "url": None,
    }
