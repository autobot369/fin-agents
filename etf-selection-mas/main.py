"""
Global ETF Top-20 Boom List — Entry Point
==========================================

Usage:
  python main.py                          # Standard run: all markets, 20 ETFs, TER ≤ 0.70%
  python main.py --ter 0.50               # Stricter TER ceiling (0.50%)
  python main.py --top 10                 # Produce a top-10 instead
  python main.py --ter 0.65 --top 20      # Custom ceiling + size

No --market flag required — the pipeline always runs INTL + HKCN + BSE
simultaneously and consolidates into a single global ranking.

Environment variable required:
  ANTHROPIC_API_KEY  — for LLM agents (researcher, sentiment scorer, arbiter)

News is fetched via DuckDuckGo (DDGS) + yfinance.Ticker.news — 100% free.

Output: /outputs/TOP_20_BOOM_REPORT.md
"""

import argparse
import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()


def _validate_env() -> list[str]:
    return [k for k in ["ANTHROPIC_API_KEY"] if not os.getenv(k)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global ETF Top-20 Boom List — NASDAQ + NSE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Universe (always runs all three segments):
  INTL   VXUS SPDW IEFA SCHY VEA EEM ACWI SCHF IXUS VWO IEMG   (NASDAQ)
  HKCN   CQQQ KWEB MCHI FXI CHIQ GXC FLCH KURE CNYA ASHR       (NASDAQ)
  BSE    NIFTYBEES BANKBEES GOLDBEES JUNIORBEES ITBEES ...       (NSE .NS)

Boom Triggers (March 2026):
  china_tech_rerating  +0.35   rbi_pivot        +0.30
  fed_soft_landing     +0.25   india_inflation  +0.25
  china_gdp_beat       +0.25   em_broad_rally   +0.15

Consensus:  0.60 x news_sentiment + 0.40 x expense_ratio_score
Output:     outputs/TOP_20_BOOM_REPORT.md
        """,
    )
    parser.add_argument(
        "--ter",
        type=float,
        default=0.70,
        metavar="PCT",
        help="Max TER ceiling in percent (default: 0.70). Use --ter 0.50 for 0.50%%.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of ETFs in the final list (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    missing = _validate_env()
    if missing:
        print(f"\n[ERROR] Missing environment variables: {missing}")
        print("  Add ANTHROPIC_API_KEY to a .env file or export it.")
        sys.exit(1)

    ter_threshold = args.ter / 100.0

    print(f"""
+--------------------------------------------------------------+
|  Global ETF Top-20 Boom List  --  {date.today().strftime('%B %Y')}            |
+--------------------------------------------------------------+
|  Markets    : INTL + HKCN (NASDAQ)  +  BSE/NSE (India)      |
|  TER Ceiling: {args.ter:.2f}% ({ter_threshold:.4f} decimal)                      |
|  Final List : Top {args.top:<42} |
|  Consensus  : 60% News Sentiment + 40% Expense Ratio         |
|  News Source: DuckDuckGo (DDGS) + yfinance.Ticker.news       |
|  Cost       : $0  (100% free data pipeline)                  |
+--------------------------------------------------------------+
""")

    from graph.workflow import build_etf_graph

    graph = build_etf_graph()

    initial_state = {
        "ter_threshold":        ter_threshold,
        "top_n":                args.top,
        "all_etf_data":         {},
        "all_macro_news":       {},
        "researcher_notes":     "",
        "all_filtered_tickers": [],
        "all_pruned_tickers":   [],
        "boom_triggers_fired":  [],
        "raw_article_scores":   {},
        "all_sentiment_scores": {},
        "sentiment_narrative":  "",
        "all_expense_scores":   {},
        "all_consensus_scores": {},
        "sentiment_rationales": {},
        "final_ranking":        [],
        "output_markdown":      "",
    }

    final_state = graph.invoke(initial_state)

    # ── Terminal summary ──────────────────────────────────────────────────
    ranking = final_state.get("final_ranking", [])

    print(f"\n{'='*72}")
    print(f"  GLOBAL TOP-{args.top} ETF BOOM LIST  --  {date.today().strftime('%B %d, %Y')}")
    print(f"{'='*72}")
    print(f"  {'#':<4} {'Ticker':<22} {'Market':<7} {'TER%':>7}  {'Consensus':>10}  Rationale")
    print(f"  {'-'*68}")
    for e in ranking:
        ter_d = f"{e['expense_ratio']*100:.3f}%" if e.get("expense_ratio") else "N/A"
        rationale_short = (e.get("sentiment_rationale") or "")[:45]
        print(
            f"  #{e['rank']:<3} {e['ticker']:<22} {e.get('market_label',''):<7} "
            f"{ter_d:>7}  {e.get('consensus_score', 0):>10.4f}  {rationale_short}"
        )

    boom_triggers = final_state.get("boom_triggers_fired", [])
    if boom_triggers:
        print(f"\n  Active Boom Triggers: {', '.join(boom_triggers)}")

    out_path = os.path.join(
        os.path.dirname(__file__), "outputs", "TOP_20_BOOM_REPORT.md"
    )
    print(f"\n  Full report  ->  {out_path}\n")


if __name__ == "__main__":
    main()
