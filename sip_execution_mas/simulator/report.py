"""
SIP Simulator Report Generator
================================
Produces a terminal summary and a full Markdown report for the
Sentiment-Weighted Dynamic SIP simulation.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from simulator.state import AllocationPlan, SimulationResult


# ── Formatting helpers ────────────────────────────────────────────────────────

def _usd(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"${v:,.2f}"


def _pct(v: Optional[float], plus: bool = True) -> str:
    if v is None:
        return "N/A"
    sign = "+" if plus and v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _ter(v: Optional[float]) -> str:
    return f"{v * 100:.3f}%" if v is not None else "N/A"


def _score(v: float) -> str:
    return f"{v:.4f}"


# ── Terminal report ───────────────────────────────────────────────────────────

def print_terminal_report(
    plan: AllocationPlan,
    result: SimulationResult,
    months: int,
    boom_triggers: List[str],
) -> None:
    SEP  = "=" * 72
    THIN = "-" * 72

    print(f"\n{SEP}")
    print(f"  SIP SIMULATOR  —  Sentiment-Weighted Dynamic SIP  (Buy-Only)")
    print(f"  Monthly SIP  : {_usd(plan['sip_amount'])}  |  Backtest: {months} months")
    print(f"  Core/Satellite: {plan['core_pct']*100:.0f}% / {(1-plan['core_pct'])*100:.0f}%"
          f"  |  USD/INR: {result['usd_inr_rate']:.2f}  |  {date.today()}")
    print(f"{SEP}")

    # ── Core bucket
    print(f"\n  ★  CORE BUCKET  —  {_usd(plan['core_budget'])} / month  "
          f"({plan['core_pct']*100:.0f}% of SIP, top {len(plan['core'])} ETFs)")
    _print_alloc_table(plan["core"], plan["sip_amount"])

    # ── Satellite bucket
    if plan["satellite"]:
        print(f"\n  ◈  SATELLITE BUCKET  —  {_usd(plan['satellite_budget'])} / month  "
              f"({(1-plan['core_pct'])*100:.0f}% of SIP, ETFs {len(plan['core'])+1}–"
              f"{len(plan['core'])+len(plan['satellite'])})")
        _print_alloc_table(plan["satellite"], plan["sip_amount"])

    # ── Portfolio summary
    print(f"\n{THIN}")
    print(f"  BACKTEST RESULTS  ({result['months_simulated']} SIP instalments, no selling)")
    print(f"{THIN}")
    pnl_sign = "▲" if result["total_pnl_usd"] >= 0 else "▼"
    print(f"  {'Total Invested':<32}  {_usd(result['total_invested_usd'])}")
    print(f"  {'Portfolio Value (today)':<32}  {_usd(result['current_value_usd'])}")
    print(f"  {'Unrealised P&L':<32}  {pnl_sign} {_usd(abs(result['total_pnl_usd']))}  "
          f"({_pct(result['total_return_pct'])})")
    print(f"  {'CAGR (annualised)':<32}  {_pct(result['cagr'])}")

    if result["skipped_tickers"]:
        print(f"\n  ⚠  No price data: {', '.join(result['skipped_tickers'])}")

    # ── Per-ETF performance
    if result["holdings"]:
        print(f"\n{THIN}")
        print(f"  PER-ETF PERFORMANCE  (sorted by return)")
        print(f"{THIN}")
        print(f"  {'Ticker':<22} {'Bkt':<4} {'Region':<6} {'Invested':>10} "
              f"{'Value':>10} {'P&L':>10} {'Return':>8}")
        print(f"  {THIN}")
        for h in sorted(result["holdings"].values(), key=lambda x: x["return_pct"], reverse=True):
            pnl_str = ("+" if h["pnl_usd"] >= 0 else "") + _usd(abs(h["pnl_usd"]))
            print(
                f"  {h['ticker']:<22} {h['bucket'][:3].upper():<4} {h['region']:<6} "
                f"{_usd(h['invested_usd']):>10} "
                f"{_usd(h['value_usd']):>10} "
                f"{pnl_str:>10} "
                f"{_pct(h['return_pct']):>8}"
            )

    if boom_triggers:
        print(f"\n  Active boom triggers: {', '.join(boom_triggers)}")

    print(f"\n{SEP}\n")


def _print_alloc_table(allocs: list, sip_amount: float) -> None:
    THIN = "-" * 72
    print(f"  {'#':<3} {'Ticker':<22} {'Region':<6} {'Score':>7} "
          f"{'Monthly':>9} {'SIP%':>6} {'TER':>8}")
    print(f"  {THIN}")
    for i, a in enumerate(allocs, 1):
        print(
            f"  {i:<3} {a['ticker']:<22} {a['region']:<6} "
            f"{_score(a['consensus_score']):>7} "
            f"{_usd(a['monthly_usd']):>9} "
            f"{a['monthly_usd']/sip_amount*100:>5.1f}% "
            f"{_ter(a['expense_ratio']):>8}"
        )


# ── Markdown report ───────────────────────────────────────────────────────────

def generate_markdown_report(
    plan: AllocationPlan,
    result: SimulationResult,
    months: int,
    boom_triggers: List[str],
    rankings_generated_at: str,
) -> str:
    today = date.today().strftime("%B %d, %Y")
    lines: List[str] = [
        "# SIP Simulator — Sentiment-Weighted Dynamic SIP\n\n",
        f"**Generated:** {today}  \n",
        f"**Monthly SIP:** {_usd(plan['sip_amount'])}  \n",
        f"**Strategy:** Core ({plan['core_pct']*100:.0f}%) + "
        f"Satellite ({(1-plan['core_pct'])*100:.0f}%)  \n",
        f"**Simulation:** {months}-month historical backtest, buy-only (no selling)  \n",
        f"**Rankings Source:** etf-selection-mas (generated: {rankings_generated_at})  \n",
        f"**USD/INR Rate:** {result['usd_inr_rate']:.2f}  \n",
        f"**Allocation Formula:** Weight_i = ConsensusScore_i / Σ ConsensusScore_bucket\n\n",
        "---\n\n",
    ]

    # ── Summary metrics
    pnl_sign = "+" if result["total_pnl_usd"] >= 0 else ""
    lines += [
        "## Portfolio Summary\n\n",
        "| Metric | Value |\n",
        "|--------|-------|\n",
        f"| Total Invested | {_usd(result['total_invested_usd'])} |\n",
        f"| Portfolio Value (today) | **{_usd(result['current_value_usd'])}** |\n",
        f"| Unrealised P&L | {pnl_sign}{_usd(result['total_pnl_usd'])} |\n",
        f"| Total Return | **{_pct(result['total_return_pct'])}** |\n",
        f"| CAGR (annualised) | {_pct(result['cagr'])} |\n",
        f"| SIP Instalments | {result['months_simulated']} |\n",
        f"| ETFs in Portfolio | {len(result['holdings'])} |\n",
        f"| USD/INR Rate Used | {result['usd_inr_rate']:.2f} |\n",
        "\n",
    ]

    if boom_triggers:
        lines.append(
            f"> **Active Boom Triggers:** "
            f"{', '.join(f'`{t}`' for t in boom_triggers)}\n\n"
        )

    lines.append("---\n\n")

    # ── Core bucket table
    lines += [
        f"## ★ Core Bucket — {_usd(plan['core_budget'])}/month "
        f"({plan['core_pct']*100:.0f}% of SIP)\n\n",
        "> Top-scored ETFs providing stable, all-weather exposure.\n\n",
    ]
    lines += _md_alloc_table(plan["core"], plan["sip_amount"])
    lines.append("\n")

    # ── Satellite bucket table
    if plan["satellite"]:
        lines += [
            f"## ◈ Satellite Bucket — {_usd(plan['satellite_budget'])}/month "
            f"({(1-plan['core_pct'])*100:.0f}% of SIP)\n\n",
            "> Alpha-tilt positions doubling down on active boom signals.\n\n",
        ]
        lines += _md_alloc_table(plan["satellite"], plan["sip_amount"])
        lines.append("\n")

    lines.append("---\n\n")

    # ── Per-ETF performance table
    if result["holdings"]:
        lines += [
            "## Per-ETF Performance\n\n",
            "| Ticker | Bucket | Region | Units | Invested | Value | P&L | Return |\n",
            "|:-------|:------:|:------:|------:|:--------:|:-----:|:---:|:------:|\n",
        ]
        for h in sorted(result["holdings"].values(), key=lambda x: x["return_pct"], reverse=True):
            pnl_str = ("+" if h["pnl_usd"] >= 0 else "") + _usd(abs(h["pnl_usd"]))
            lines.append(
                f"| `{h['ticker']}` "
                f"| {h['bucket'].capitalize()} "
                f"| {h['region']} "
                f"| {h['total_units']:.4f} "
                f"| {_usd(h['invested_usd'])} "
                f"| {_usd(h['value_usd'])} "
                f"| {pnl_str} "
                f"| **{_pct(h['return_pct'])}** |\n"
            )
        lines.append("\n")

    lines.append("---\n\n")

    # ── Monthly SIP timeline
    if result["monthly_snapshots"]:
        lines += [
            "## Monthly SIP Timeline\n\n",
            "| Month | ETFs Bought | Invested (Month) | Cumulative Invested |\n",
            "|:------|:-----------:|:----------------:|:-------------------:|\n",
        ]
        cumulative = 0.0
        for snap in result["monthly_snapshots"]:
            cumulative += snap["usd_invested"]
            lines.append(
                f"| {snap['date'][:7]} "
                f"| {len(snap['units_bought'])} "
                f"| {_usd(snap['usd_invested'])} "
                f"| {_usd(cumulative)} |\n"
            )
        lines.append("\n")

    lines.append("---\n\n")

    # ── Regional breakdown
    region_totals: Dict[str, float] = {}
    for a in plan["all"]:
        region_totals[a["region"]] = region_totals.get(a["region"], 0) + a["monthly_usd"]

    lines += [
        "## Allocation by Region\n\n",
        "| Region | Monthly | % of SIP | Currency |\n",
        "|:------:|:-------:|:--------:|:--------:|\n",
    ]
    for region, total in sorted(region_totals.items()):
        currency = "INR (via USD conversion)" if region == "BSE" else "USD"
        lines.append(
            f"| {region} | {_usd(total)} | {total/plan['sip_amount']*100:.1f}% | {currency} |\n"
        )
    lines.append("\n")

    if result["skipped_tickers"]:
        lines += [
            "---\n\n",
            "## ⚠ Tickers Skipped (No Price Data)\n\n",
            ", ".join(f"`{t}`" for t in result["skipped_tickers"]) + "\n\n",
        ]

    lines += [
        "---\n\n",
        "## Notes\n\n",
        f"- **No selling** — units are accumulated each month without rebalancing.\n",
        f"- **BSE/NSE ETFs** priced in INR; converted at 1 USD = {result['usd_inr_rate']:.2f} INR.\n",
        f"- **Fractional units** shown for illustration; actual brokers may require whole lots.\n",
        f"- To refresh rankings, re-run `etf-selection-mas/main.py`.\n",
        f"- Past performance does not guarantee future results.\n\n",
        "> _Generated by etf-selection-mas SIP Simulator_\n",
    ]

    return "".join(lines)


def _md_alloc_table(allocs: list, sip_amount: float) -> List[str]:
    rows = [
        "| # | Ticker | Region | Category | Consensus | Monthly SIP | SIP % | TER | Rationale |\n",
        "|:-:|:-------|:------:|:---------|:---------:|:-----------:|:-----:|:---:|:----------|\n",
    ]
    for i, a in enumerate(allocs, 1):
        rat = a.get("sentiment_rationale", "—")
        if len(rat) > 80:
            rat = rat[:77] + "…"
        rows.append(
            f"| {i} "
            f"| `{a['ticker']}` "
            f"| {a['region']} "
            f"| {a['category']} "
            f"| {_score(a['consensus_score'])} "
            f"| **{_usd(a['monthly_usd'])}** "
            f"| {a['monthly_usd']/sip_amount*100:.1f}% "
            f"| {_ter(a['expense_ratio'])} "
            f"| {rat} |\n"
        )
    return rows


def save_report(content: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
