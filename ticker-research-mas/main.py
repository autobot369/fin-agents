"""
Market Research Multi-Agent System
====================================
Usage:
    python main.py AAPL
    python main.py NVDA --horizon short --risk 0.8 --save --verbose

Requires ANTHROPIC_API_KEY in environment or .env file.
"""

import argparse
import datetime
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market Research MAS — LangGraph + Claude Opus 4.6"
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="mid",
        choices=["short", "mid", "long"],
        help="Investment horizon: short (1-3mo), mid (6-12mo), long (1-3yr). Default: mid",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=0.5,
        metavar="0.0-1.0",
        help="Risk aversity: 0.0 = risk-seeking, 1.0 = risk-averse. Default: 0.5",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the final forecast as a Markdown file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full debate transcript to stdout",
    )
    return parser.parse_args()


def print_section(title: str, content: str, width: int = 70) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    print(content)


def save_forecast(ticker: str, forecast: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(f"{ticker.upper()}_{ts}_forecast.md")
    filename.write_text(forecast, encoding="utf-8")
    return filename


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper().strip()

    if not 0.0 <= args.risk <= 1.0:
        print("[ERROR]  --risk must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'█' * 60}")
    print(f"  Market Research MAS  ·  {ticker}")
    print(f"  Horizon: {args.horizon.upper()}  |  Risk aversity: {args.risk:.2f}")
    print(f"  LangGraph + Claude Opus 4.6 (Adaptive Thinking)")
    print(f"{'█' * 60}")

    # Lazy import after env is loaded
    from graph.workflow import build_graph

    graph = build_graph()

    initial_state = {
        "ticker": ticker,
        "investment_horizon": args.horizon,
        "risk_aversity": args.risk,
        "research_data": {},
        "ground_truth": {},
        "bull_case": "",
        "bear_case": "",
        "bull_rebuttal": "",
        "bear_counter": "",
        "debate_history": [],
        "fact_check_report": "",
        "final_forecast": "",
        "debate_round": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"\n[ERROR]  Graph execution failed: {e}", file=sys.stderr)
        raise

    if args.verbose:
        for entry in final_state.get("debate_history", []):
            role = entry["role"].upper().replace("_", " ")
            print_section(role, entry["content"])

    print_section("FINAL CONSENSUS FORECAST", final_state["final_forecast"])

    if args.save:
        path = save_forecast(ticker, final_state["final_forecast"])
        print(f"\n[INFO]  Forecast saved to: {path.resolve()}")

    print(f"\n{'█' * 60}\n")


if __name__ == "__main__":
    main()
