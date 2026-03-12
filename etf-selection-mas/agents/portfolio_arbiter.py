"""
Portfolio Arbiter Agent — Global Top-20 Edition
================================================
Final node in the ETF Selection pipeline.

Responsibilities:
  1. Sort ALL filtered ETFs (INTL + HKCN + BSE combined) by consensus_score.
  2. Select the global top-20.
  3. Call Claude once to generate a one-sentence "Sentiment Rationale" per ETF.
  4. Render the output Markdown table:

     | Rank | Ticker & Market | Category | TER% | Trade On | Sentiment Rationale |

  5. Write the report to /outputs/TOP_20_BOOM_REPORT.md

Tiebreaker: equal consensus → higher momentum_3m wins.
"""

import json
import os
from datetime import date
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.sentiment_scorer import BOOM_TRIGGERS
from graph.state import ETFSelectionState
from tools.bse_data import get_etf_metadata

_llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=3000)

_SYSTEM_RATIONALE = """\
You are a concise ETF analyst. For each ETF in the provided list, write exactly
ONE sentence (max 20 words) explaining why it is expected to BOOM in March 2026.
The sentence must be specific: cite the relevant macro trigger or theme.
Return ONLY valid JSON — no markdown, no extra text:
{"rationales": {"TICKER": "sentence.", ...}}
"""

OUTPUT_FILENAME = "TOP_20_BOOM_REPORT.md"


def _fmt_pct(val, d: int = 2) -> str:
    return f"{val * 100:.{d}f}%" if val is not None else "N/A"


def _fmt_ter(val) -> str:
    return f"{val * 100:.3f}%" if val is not None else "N/A ⚠"


def _generate_rationales(
    top_entries: List[Dict[str, Any]],
    boom_triggers_fired: List[str],
    researcher_notes: str,
) -> Dict[str, str]:
    """
    One LLM call: generate a one-sentence boom rationale per ETF in the top-20.
    Returns dict {ticker → sentence}.
    """
    etf_list = "\n".join(
        f"  {e['ticker']:<20} region={e['region']:<5} "
        f"sentiment={e.get('sentiment_score', 0):.3f}  "
        f"TER={_fmt_ter(e.get('expense_ratio'))}  "
        f"category={e.get('category', '?')}"
        for e in top_entries
    )

    trigger_ctx = (
        "Active boom triggers: " +
        ", ".join(f"{k} ({BOOM_TRIGGERS[k]['boost']:+.2f})" for k in boom_triggers_fired)
        if boom_triggers_fired else "No confirmed boom triggers — rank by momentum/TER."
    )

    macro_ctx = researcher_notes[:800] if researcher_notes else ""

    prompt = (
        f"March 2026 macro context:\n{macro_ctx}\n\n"
        f"{trigger_ctx}\n\n"
        f"ETFs requiring rationales (one sentence each, max 20 words):\n{etf_list}\n\n"
        f"Return only the JSON object with 'rationales' key."
    )

    try:
        resp = _llm.invoke([SystemMessage(content=_SYSTEM_RATIONALE), HumanMessage(content=prompt)])
        raw = _extract_text(resp)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return parsed.get("rationales", {})
    except Exception as exc:
        print(f"  [arbiter] Rationale LLM error: {exc}")
        top_trigger = boom_triggers_fired[0] if boom_triggers_fired else "macro momentum"
        return {
            e["ticker"]: f"Strong {top_trigger.replace('_', ' ')} thesis drives expected outperformance."
            for e in top_entries
        }


def _build_report(
    ranked: List[Dict[str, Any]],
    boom_triggers_fired: List[str],
    pruned_tickers: List[str],
    researcher_notes: str,
    sentiment_narrative: str,
    ter_threshold: float,
) -> str:
    today = date.today().strftime("%B %d, %Y")
    lines = [
        f"# Global ETF Top-20 Boom List — {today}\n\n",
        f"> **TER Ceiling:** {ter_threshold * 100:.2f}%  |  "
        f"**Consensus:** 60% News Sentiment + 40% Expense Ratio  |  "
        f"**Universe:** NASDAQ (INTL + China/HK) + NSE (India)\n\n",
        "---\n\n",
        "## Top 20 Global ETFs — Boom Ranking\n\n",
        "| Rank | Ticker & Market | Category | TER% | Trade On | Sentiment Rationale |\n",
        "|:----:|:----------------|:---------|:----:|:---------|:--------------------|\n",
    ]

    for e in ranked:
        ticker       = e["ticker"]
        market_label = e.get("market_label", "NASDAQ")
        category     = e.get("category", "—")
        ter          = _fmt_ter(e.get("expense_ratio"))
        trade_on     = e.get("trade_on", "—")
        rationale    = e.get("sentiment_rationale", "—")

        lines.append(
            f"| **{e['rank']}** "
            f"| `{ticker}` — {market_label} "
            f"| {category} "
            f"| {ter} "
            f"| {trade_on} "
            f"| {rationale} |\n"
        )

    # ── Collapsed score details ────────────────────────────────────────────
    lines += [
        "\n\n<details>\n<summary>Score Details (click to expand)</summary>\n\n",
        "| Rank | Ticker | Region | YTD% | Mom 3M | Sentiment | Expense | Consensus |\n",
        "|:----:|:-------|:------:|:----:|:------:|:---------:|:-------:|:---------:|\n",
    ]
    for e in ranked:
        lines.append(
            f"| {e['rank']:>2} "
            f"| {e['ticker']:<12} "
            f"| {e['region']:<5} "
            f"| {_fmt_pct(e.get('ytd_return')):>7} "
            f"| {_fmt_pct(e.get('momentum_3m')):>7} "
            f"| {e.get('sentiment_score', 0):.4f} "
            f"| {e.get('expense_score', 0):.4f} "
            f"| **{e.get('consensus_score', 0):.4f}** |\n"
        )
    lines.append("\n</details>\n\n")

    # ── Boom triggers ──────────────────────────────────────────────────────
    lines.append("---\n\n## Active March 2026 Boom Triggers\n\n")
    if boom_triggers_fired:
        for key in boom_triggers_fired:
            spec = BOOM_TRIGGERS.get(key, {})
            sign = "BULLISH" if spec.get("boost", 0) > 0 else "BEARISH"
            lines.append(
                f"### `{key}` — {sign} ({spec.get('boost', 0):+.2f})\n"
                f"**Applies to:** `{spec.get('applies_to', [])}`\n\n"
                f"> {spec.get('hypothesis', '')}\n\n"
            )
    else:
        lines.append("_No high-confidence boom triggers detected._\n\n")

    # ── Pruned ────────────────────────────────────────────────────────────
    if pruned_tickers:
        lines.append(f"---\n\n## Pruned ETFs (TER > {ter_threshold*100:.2f}%)\n\n")
        lines.append("| Ticker | Reason |\n|--------|--------|\n")
        for t in pruned_tickers:
            lines.append(f"| {t} | TER exceeded {ter_threshold*100:.2f}% ceiling |\n")

    # ── Macro brief ───────────────────────────────────────────────────────
    lines.append(f"\n---\n\n## Global Macro Brief (March 2026)\n\n{researcher_notes}\n\n")

    # ── Sentiment narrative ───────────────────────────────────────────────
    lines.append(f"---\n\n{sentiment_narrative}\n")

    return "".join(lines)


def portfolio_arbiter_node(state: ETFSelectionState) -> dict:
    """
    LangGraph node: Portfolio Arbiter — Global Top-20.

    Reads  : state["all_consensus_scores"], state["all_sentiment_scores"],
             state["all_expense_scores"], state["all_etf_data"],
             state["all_filtered_tickers"], state["top_n"],
             state["boom_triggers_fired"], state["all_pruned_tickers"],
             state["ter_threshold"], state["researcher_notes"],
             state["sentiment_narrative"]
    Writes : state["sentiment_rationales"], state["final_ranking"],
             state["output_markdown"]
    """
    top_n            = state.get("top_n", 20)
    consensus_scores = state.get("all_consensus_scores", {})
    sentiment_scores = state.get("all_sentiment_scores", {})
    expense_scores   = state.get("all_expense_scores", {})
    etf_data         = state.get("all_etf_data", {})
    filtered         = state.get("all_filtered_tickers", [])
    pruned           = state.get("all_pruned_tickers", [])
    boom_fired       = state.get("boom_triggers_fired", [])
    ter_threshold    = state.get("ter_threshold", 0.007)

    print(f"\n{'='*60}")
    print(f"[PORTFOLIO_ARBITER]  Building Global Top-{top_n} Boom List ...")
    print(f"  Eligible pool: {len(filtered)} ETFs across INTL + HKCN + BSE")
    print(f"{'='*60}")

    # ── Sort: primary = consensus desc, tiebreak = momentum_3m desc ───────
    def _sort_key(t: str):
        return (
            round(consensus_scores.get(t, 0.0), 4),
            round(etf_data.get(t, {}).get("momentum_3m") or 0.0, 4),
        )

    top_tickers = sorted(filtered, key=_sort_key, reverse=True)[:top_n]

    # ── Build pre-rationale entry list ────────────────────────────────────
    entries: List[Dict[str, Any]] = []
    for rank, ticker in enumerate(top_tickers, 1):
        rec  = etf_data.get(ticker, {})
        meta = get_etf_metadata(ticker)
        entries.append({
            "rank":            rank,
            "ticker":          ticker,
            "name":            rec.get("name", ticker),
            "market_label":    rec.get("market", "NASDAQ"),
            "region":          rec.get("region", "?"),
            "category":        meta.get("category", "—"),
            "trade_on":        meta.get("trade_on", "—"),
            "expense_ratio":   rec.get("expense_ratio"),
            "aum_b":           rec.get("aum_b"),
            "ytd_return":      rec.get("ytd_return"),
            "momentum_3m":     rec.get("momentum_3m"),
            "momentum_1m":     rec.get("momentum_1m"),
            "current_price":   rec.get("current_price"),
            "sentiment_score": sentiment_scores.get(ticker),
            "expense_score":   expense_scores.get(ticker),
            "consensus_score": consensus_scores.get(ticker),
        })

    # ── LLM: one call for all rationales ─────────────────────────────────
    print(f"\n[PORTFOLIO_ARBITER]  Generating {top_n} sentiment rationales ...")
    rationales = _generate_rationales(
        entries, boom_fired, state.get("researcher_notes", "")
    )
    for e in entries:
        e["sentiment_rationale"] = rationales.get(
            e["ticker"], "Macro tailwinds support near-term outperformance."
        )

    # ── Terminal summary ──────────────────────────────────────────────────
    print(f"\n  {'Rank':<5} {'Ticker':<22} {'Market':<7} {'TER%':>7} {'Consensus':>10}")
    print(f"  {'-'*60}")
    for e in entries:
        print(
            f"  #{e['rank']:<4} {e['ticker']:<22} {e['market_label']:<7} "
            f"{_fmt_ter(e['expense_ratio']):>7}  {e.get('consensus_score',0):.4f}"
        )

    # ── Write report ──────────────────────────────────────────────────────
    report_md = _build_report(
        ranked=entries,
        boom_triggers_fired=boom_fired,
        pruned_tickers=pruned,
        researcher_notes=state.get("researcher_notes", ""),
        sentiment_narrative=state.get("sentiment_narrative", ""),
        ter_threshold=ter_threshold,
    )

    out_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUTPUT_FILENAME)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[PORTFOLIO_ARBITER]  Report -> {out_path}")

    # ── Machine-readable rankings JSON (consumed by simulator) ────────────
    rankings_payload = {
        "generated_at": date.today().isoformat(),
        "ter_threshold": ter_threshold,
        "boom_triggers_fired": boom_fired,
        "rankings": entries,
    }
    json_path = os.path.join(out_dir, "rankings.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rankings_payload, f, indent=2, default=str)
    print(f"[PORTFOLIO_ARBITER]  Rankings JSON -> {json_path}")

    return {
        "sentiment_rationales": rationales,
        "final_ranking":        entries,
        "output_markdown":      report_md,
    }


def _extract_text(response) -> str:
    if hasattr(response, "content") and isinstance(response.content, list):
        parts = [
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in response.content
            if (isinstance(b, dict) and b.get("type") == "text")
            or (hasattr(b, "type") and b.type == "text")
        ]
        return "\n".join(parts).strip()
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content.strip()
    return str(response)
