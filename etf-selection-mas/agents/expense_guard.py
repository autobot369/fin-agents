"""
Expense Guard Agent
--------------------
Prunes any ETF from the global universe whose TER exceeds the threshold.

Rules (unchanged from v1, state keys updated to `all_*`):
  • TER unknown (None) → KEEP with warning (benefit of the doubt)
  • TER > ter_threshold → PRUNE
  • TER ≤ ter_threshold → PASS

Reads  : state["all_etf_data"], state["ter_threshold"]
Writes : state["all_filtered_tickers"], state["all_pruned_tickers"]
"""

from typing import List

from graph.state import ETFSelectionState


def expense_guard_node(state: ETFSelectionState) -> dict:
    etf_data: dict = state.get("all_etf_data", {})
    threshold: float = state.get("ter_threshold", 0.007)

    print(f"\n{'='*60}")
    print(f"[EXPENSE_GUARD]  TER threshold = {threshold*100:.2f}%")
    print(f"  Universe: {len(etf_data)} ETFs across INTL + HKCN + BSE")
    print(f"{'='*60}")

    filtered: List[str] = []
    pruned:   List[str] = []

    for ticker, rec in etf_data.items():
        ter = rec.get("expense_ratio")
        region = rec.get("region", "?")

        if ter is None:
            print(f"  KEEP  [{region}] {ticker:<20}  TER=Unknown  ⚠ unverified")
            filtered.append(ticker)
        elif ter > threshold:
            print(f"  PRUNE [{region}] {ticker:<20}  TER={ter*100:.3f}%  > {threshold*100:.2f}%")
            pruned.append(ticker)
        else:
            print(f"  PASS  [{region}] {ticker:<20}  TER={ter*100:.3f}%")
            filtered.append(ticker)

    print(f"\n  Passed: {len(filtered)} | Pruned: {len(pruned)}")
    by_region = {}
    for t in filtered:
        r = etf_data.get(t, {}).get("region", "?")
        by_region[r] = by_region.get(r, 0) + 1
    for r, cnt in sorted(by_region.items()):
        print(f"    {r}: {cnt} ETFs passed")

    return {
        "all_filtered_tickers": filtered,
        "all_pruned_tickers":   pruned,
    }
